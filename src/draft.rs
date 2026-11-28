//! DSpark's noise block is refined by three MTP layers, then a rank-256
//! Markov head conditions each proposed token on the one proposed before it.
use crate::{
    draft_attention::DraftAttention,
    model::residual,
    moe,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};

#[path = "confidence.rs"]
pub(crate) mod confidence;

pub struct DraftState {
    attention: DraftAttention,
    trace: Option<Vec<f32>>,
}
impl DraftState {
    pub fn new() -> Self {
        Self {
            attention: DraftAttention::default(),
            trace: None,
        }
    }
    pub fn ready(&self, pos: usize) -> bool {
        self.attention.ready(pos)
    }
    pub fn snapshot(&self, rt: &mut Runtime, pos: usize) -> Result<Vec<u8>> {
        self.attention.snapshot(rt, pos)
    }
    pub fn restore(rt: &mut Runtime, pos: usize, bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            attention: DraftAttention::restore(rt, pos, bytes)?,
            trace: None,
        })
    }
    pub(crate) fn rotation_snapshot(&self, rt: &mut Runtime, pos: usize) -> Result<Vec<u8>> {
        self.attention
            .encode_rotation(rt, pos, self.trace.as_deref())
    }
    pub(crate) fn restore_rotation(rt: &mut Runtime, pos: usize, bytes: &[u8]) -> Result<Self> {
        let (attention, trace) = DraftAttention::decode_rotation(rt, pos, bytes)?;
        Ok(Self { attention, trace })
    }
    pub(crate) fn release_device(self) {
        self.attention.release_device();
    }
    pub fn observe(
        &mut self,
        rt: &mut Runtime,
        features: &Dev,
        pos: usize,
        committed_rows: usize,
    ) -> Result<()> {
        self.attention.observe(rt, features, pos, committed_rows)
    }
    pub fn report_confidence(&mut self, kept: usize, accepted: usize) {
        if let Some(scores) = self.trace.take() {
            eprintln!("draft-confidence kept={kept} accepted={accepted} scores={scores:?}");
        }
    }
    pub fn propose(
        &mut self,
        rt: &mut Runtime,
        pos: usize,
        seed: u32,
        width: usize,
        concurrent_sessions: usize,
    ) -> Result<Vec<u32>> {
        propose_many(
            rt,
            &mut [DraftInput {
                state: self,
                pos,
                seed,
                width,
            }],
            concurrent_sessions,
            None,
        )?
        .pop()
        .context("missing draft result")
    }
}

pub(crate) struct DraftInput<'a> {
    pub state: &'a mut DraftState,
    pub pos: usize,
    pub seed: u32,
    pub width: usize,
}

// Sharing the feed-forward and head weights saves a traversal per additional
// request. Noise seeds, attention rings, Markov chains, and confidence decisions
// belong to the originating sequence. Proposing never commits any KV.
pub(crate) fn propose_many(
    rt: &mut Runtime,
    requests: &mut [DraftInput<'_>],
    concurrent_sessions: usize,
    adaptive_cost: Option<confidence::CostCurve>,
) -> Result<Vec<Vec<u32>>> {
    ensure!(!requests.is_empty(), "empty draft batch");
    ensure!(
        concurrent_sessions >= requests.len(),
        "draft batch exceeds active sessions"
    );
    let mut noise = Vec::new();
    for r in requests.iter() {
        ensure!(
            (1..=16).contains(&r.width),
            "DSpark proposal length out of range"
        );
        noise.push(r.seed);
        noise.resize(noise.len() + r.width - 1, 128799);
    }
    let rows = noise.len();
    let mut h = residual::embed(rt, &noise)?;
    let mut mix = residual::initial_mix(rt, rows)?;
    for stage in 0..3 {
        let stem = format!("mtp.{stage}");
        let x = residual::prepare(rt, &stem, "attn", &h, &mut mix, rows)?;
        let y = if let [r] = requests {
            r.state.attention.forward(rt, stage, &x, r.pos, r.width)?
        } else {
            let mut all = rt.buffer(x.len())?;
            let mut offset = 0;
            for r in requests.iter_mut() {
                let part = rt.slice(&x, offset, r.width * crate::model::DIM)?;
                let y = r
                    .state
                    .attention
                    .forward(rt, stage, &part, r.pos, r.width)?;
                rt.copy_into(&mut all, offset, &y)?;
                offset += y.len();
            }
            all
        };
        residual::expand(rt, &mut h, &y, &mix, rows)?;
        drop((x, y));
        let x = residual::prepare(rt, &stem, "ffn", &h, &mut mix, rows)?;
        let x = moe::draft_forward(rt, stage, &x, rows)?;
        residual::expand(rt, &mut h, &x, &mix, rows)?;
    }
    let x = residual::readout(rt, &h, &mut mix, "mtp.2.norm.weight", 0, rows)?;
    let logits = rt.linear_dev("head.weight", &x, rows, 1)?;
    let gate = confidence::Gate::configured(concurrent_sessions, requests.len())?;
    let batch_budget = gate
        .as_ref()
        .filter(|_| requests.len() > 1)
        .and_then(confidence::Gate::batch_budget_per_request)
        .map(|n| n.saturating_mul(requests.len()));
    if let [r] = requests {
        let ids = rt.markov(&logits, r.seed, r.width)?;
        return Ok(vec![finish_proposal(rt, r, gate.as_ref(), &h, &mix, &ids)?]);
    }
    let mut proposals = Vec::with_capacity(requests.len());
    let mut offset = 0;
    for r in requests.iter_mut() {
        let logits = rt.slice(&logits, offset * 129280, r.width * 129280)?;
        let ids = rt.markov(&logits, r.seed, r.width)?;
        let result = if gate.is_some() {
            let hc = h.len() / rows;
            let mc = mix.len() / rows;
            let hidden = rt.slice(&h, offset * hc, r.width * hc)?;
            let mixture = rt.slice(&mix, offset * mc, r.width * mc)?;
            finish_proposal(rt, r, gate.as_ref(), &hidden, &mixture, &ids)?
        } else {
            rt.token_ids(&ids)?
        };
        proposals.push(result);
        offset += r.width;
    }
    if let Some(budget) = batch_budget {
        let curves = requests
            .iter_mut()
            .map(|r| {
                r.state
                    .trace
                    .take()
                    .context("missing DSpark confidence curve for batch allocation")
            })
            .collect::<Result<Vec<_>>>()?;
        let kept = confidence::allocate_batch_budget(&curves, budget)?;
        for (proposal, keep) in proposals.iter_mut().zip(kept) {
            proposal.truncate(keep);
        }
    } else if gate.as_ref().is_some_and(confidence::Gate::adaptive_budget) {
        let curves = requests
            .iter_mut()
            .map(|r| {
                r.state
                    .trace
                    .take()
                    .context("missing DSpark confidence curve for adaptive allocation")
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(cost) = adaptive_cost {
            let kept = confidence::allocate_cost_aware(&curves, &cost)?;
            trace_budget(
                "adaptive",
                kept.iter().sum(),
                curves.iter().map(Vec::len).sum(),
                Some(&cost),
            );
            for (proposal, keep) in proposals.iter_mut().zip(kept) {
                proposal.truncate(keep);
            }
        } else {
            let threshold = gate
                .as_ref()
                .expect("adaptive gate exists")
                .fallback_threshold();
            for (proposal, curve) in proposals.iter_mut().zip(curves) {
                let keep = curve
                    .iter()
                    .position(|&p| p < threshold)
                    .unwrap_or(curve.len());
                proposal.truncate(keep);
            }
            trace_budget(
                "confidence-fallback",
                proposals.iter().map(Vec::len).sum(),
                0,
                None,
            );
        }
    }
    Ok(proposals)
}

fn trace_budget(
    kind: &str,
    selected: usize,
    available: usize,
    cost: Option<&confidence::CostCurve>,
) {
    if std::env::var("DS4_ADAPTIVE_DRAFT_BUDGET_TRACE").as_deref() != Ok("1") {
        return;
    }
    static STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let step = STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if step % 256 == 0 {
        if let Some(cost) = cost {
            eprintln!("adaptive-draft-budget mode={kind} selected={selected}/{available} base_rows={} sampling_requests={} cost_shapes={} draft_us={:.1} verify_points={:?}",
                      cost.base_rows, cost.sampling_requests, cost.verify_points.len(), cost.draft_ns / 1000.,
                      cost.verify_points.iter().map(|(rows, ns)| (*rows, (*ns / 1000.).round())).collect::<Vec<_>>());
        } else {
            eprintln!("adaptive-draft-budget mode={kind} selected={selected}");
        }
    }
}

fn finish_proposal(
    rt: &mut Runtime,
    r: &mut DraftInput<'_>,
    gate: Option<&confidence::Gate>,
    h: &Dev,
    mix: &Dev,
    ids: &Dev,
) -> Result<Vec<u32>> {
    if let Some(gate) = gate {
        let (proposals, trace) = gate.evaluate(rt, h, mix, r.seed, ids)?;
        r.state.trace = trace;
        Ok(proposals)
    } else {
        rt.token_ids(ids)
    }
}
