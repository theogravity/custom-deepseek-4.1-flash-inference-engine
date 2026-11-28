use crate::{
    model::DIM,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};
use std::{
    collections::{HashMap, VecDeque},
    ffi::c_void,
};

unsafe extern "C" {
    fn ds4_confidence(
        device: i32,
        h: *const f32,
        mix: *const f32,
        w: *const f32,
        embed: *const c_void,
        ids: *const f32,
        seed: i32,
        vocab: i32,
        out: *mut f32,
        rows: i32,
    ) -> i32;
}

pub(super) struct Gate {
    threshold: f32,
    fallback_threshold: f32,
    trace: bool,
    batch_budget_per_request: Option<usize>,
    adaptive_budget: bool,
}
impl Gate {
    pub(super) fn configured(
        concurrent_sessions: usize,
        draft_batch_size: usize,
    ) -> Result<Option<Self>> {
        let mut threshold = match std::env::var("DS4_CONFIDENCE_THRESHOLD") {
            Ok(v) => v
                .parse::<f32>()
                .context("invalid DS4_CONFIDENCE_THRESHOLD")?,
            Err(std::env::VarError::NotPresent) => 0.,
            Err(e) => return Err(e.into()),
        };
        ensure!(
            (0. ..=1.).contains(&threshold),
            "DS4_CONFIDENCE_THRESHOLD must be in [0,1]"
        );
        let batch_delta = match std::env::var("DS4_CONFIDENCE_BATCH_DELTA") {
            Ok(v) => v
                .parse::<f32>()
                .context("invalid DS4_CONFIDENCE_BATCH_DELTA")?,
            Err(std::env::VarError::NotPresent) => 0.1,
            Err(e) => return Err(e.into()),
        };
        ensure!(
            (0. ..=1.).contains(&batch_delta),
            "DS4_CONFIDENCE_BATCH_DELTA must be in [0,1]"
        );
        let trace = std::env::var_os("DS4_CONFIDENCE_TRACE").is_some_and(|v| v == "1");
        let batch_budget_per_request = match std::env::var("DS4_BATCH_DRAFT_BUDGET_PER_REQUEST") {
            Ok(v) if v.is_empty() => None,
            Ok(v) => {
                let value = v
                    .parse::<usize>()
                    .context("invalid DS4_BATCH_DRAFT_BUDGET_PER_REQUEST")?;
                ensure!(
                    value > 0,
                    "DS4_BATCH_DRAFT_BUDGET_PER_REQUEST must be positive"
                );
                Some(value)
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(e.into()),
        };
        let adaptive_budget = switch("DS4_ADAPTIVE_DRAFT_BUDGET", false)? && draft_batch_size > 1;
        let fixed_budget_active = batch_budget_active(batch_budget_per_request, draft_batch_size);
        ensure!(
            !(adaptive_budget && fixed_budget_active),
            "DS4_ADAPTIVE_DRAFT_BUDGET and DS4_BATCH_DRAFT_BUDGET_PER_REQUEST cannot both be enabled"
        );
        let fallback_threshold = threshold_for_batch(threshold, batch_delta, concurrent_sessions);
        if fixed_budget_active || adaptive_budget {
            // The batch allocator needs each full confidence curve to price
            // candidates across requests; per-request thresholding happens after.
            threshold = 0.;
        }
        let threshold = threshold_for_batch(threshold, batch_delta, concurrent_sessions);
        Ok(
            (threshold > 0. || trace || fixed_budget_active || adaptive_budget).then_some(Self {
                threshold,
                fallback_threshold,
                trace: trace || fixed_budget_active || adaptive_budget,
                batch_budget_per_request: if fixed_budget_active {
                    batch_budget_per_request
                } else {
                    None
                },
                adaptive_budget,
            }),
        )
    }
    pub(super) fn evaluate(
        &self,
        rt: &mut Runtime,
        h: &Dev,
        mix: &Dev,
        seed: u32,
        ids: &Dev,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
        let width = ids.len();
        ensure!(
            (1..=16).contains(&width) && h.len() == width * 4 * DIM && mix.len() == width * 24,
            "invalid DSpark confidence dimensions"
        );
        let embedding = rt.raw_bf16("mtp.2.markov_head.embed.weight", 129280, 256)?;
        let w = rt.vector_ptr("mtp.2.confidence_head.proj.weight", DIM + 256)?;
        let mut output = rt.buffer(width)?;
        let rc = unsafe {
            ds4_confidence(
                rt.device,
                h.as_ptr(),
                mix.as_ptr(),
                w,
                embedding,
                ids.as_ptr(),
                seed.try_into()?,
                129280,
                output.as_mut_ptr(),
                width.try_into()?,
            )
        };
        ensure!(rc == 0, "CUDA confidence head failed ({rc})");
        let scores = rt.to_host(&output)?;
        let mut proposed = rt.token_ids(ids)?;
        let (keep, confidence) = select(&scores, self.threshold)?;
        if !self.adaptive_budget {
            proposed.truncate(keep);
        }
        Ok((proposed, self.trace.then_some(confidence)))
    }

    pub(super) fn batch_budget_per_request(&self) -> Option<usize> {
        self.batch_budget_per_request
    }

    pub(super) fn adaptive_budget(&self) -> bool {
        self.adaptive_budget
    }

    pub(super) fn fallback_threshold(&self) -> f32 {
        self.fallback_threshold
    }
}

fn switch(name: &str, default: bool) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(_) => anyhow::bail!("{name} must be 0 or 1"),
        Err(e) => Err(e.into()),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CostCurve {
    pub draft_ns: f64,
    pub verify_points: Vec<(usize, f64)>,
    pub base_rows: usize,
    pub sampling_requests: usize,
}

impl CostCurve {
    pub fn verify_ns(&self, rows: usize) -> f64 {
        let points = &self.verify_points;
        let upper = points.partition_point(|(measured_rows, _)| *measured_rows < rows);
        let (left, right) = if upper == 0 {
            (points[0], points[1])
        } else if upper == points.len() {
            (points[points.len() - 2], points[points.len() - 1])
        } else {
            (points[upper - 1], points[upper])
        };
        let span = (right.0 - left.0).max(1) as f64;
        let slope = ((right.1 - left.1) / span).max(0.);
        (left.1 + slope * (rows as f64 - left.0 as f64)).max(1.)
    }
}

#[derive(Default)]
pub(crate) struct CostProfile {
    draft_ns: HashMap<(usize, usize, usize), Ema>,
    verify_ns: HashMap<(usize, usize, usize), Ema>,
}

#[derive(Default)]
struct Ema {
    mean: f64,
    count: u32,
    recent: VecDeque<f64>,
}

impl Ema {
    fn observe(&mut self, value: f64) {
        self.recent.push_back(value);
        if self.recent.len() > 31 {
            self.recent.pop_front();
        }
        let mut sorted: Vec<_> = self.recent.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        self.mean = if sorted.len() % 2 == 0 {
            let upper = sorted.len() / 2;
            (sorted[upper - 1] + sorted[upper]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };
        self.count = self.count.saturating_add(1);
    }
}

impl CostProfile {
    pub fn observe_draft(
        &mut self,
        context_bucket: usize,
        requests: usize,
        draft_tokens: usize,
        elapsed_ns: f64,
    ) {
        self.draft_ns
            .entry((context_bucket, requests, draft_tokens))
            .or_default()
            .observe(elapsed_ns);
    }

    pub fn observe_verify(
        &mut self,
        context_bucket: usize,
        requests: usize,
        rows: usize,
        elapsed_ns: f64,
    ) {
        self.verify_ns
            .entry((context_bucket, requests, rows))
            .or_default()
            .observe(elapsed_ns);
    }

    pub fn curve(
        &self,
        context_bucket: usize,
        draft_requests: usize,
        target_requests: usize,
        draft_tokens: usize,
        base_rows: usize,
        sampling_requests: usize,
    ) -> Option<CostCurve> {
        let draft = self
            .draft_ns
            .get(&(context_bucket, draft_requests, draft_tokens))?;
        if draft.count < 8 {
            return None;
        }
        let mut points: Vec<_> = self
            .verify_ns
            .iter()
            .filter_map(|(&(ctx, batch, rows), value)| {
                (ctx == context_bucket && batch == target_requests && value.count >= 4)
                    .then_some((rows, value.mean))
            })
            .collect();
        points.sort_by_key(|point| point.0);
        if points.len() < 2 {
            return None;
        }
        // GPU step costs should be nondecreasing with scheduled query rows.
        // Remove measurement noise without discarding local kernel-shape steps.
        for index in 1..points.len() {
            points[index].1 = points[index].1.max(points[index - 1].1);
        }
        Some(CostCurve {
            draft_ns: draft.mean,
            verify_points: points,
            base_rows,
            sampling_requests,
        })
    }
}

fn threshold_for_batch(threshold: f32, delta: f32, concurrent_sessions: usize) -> f32 {
    if threshold > 0. && concurrent_sessions > 1 {
        (threshold + delta).min(1.)
    } else {
        threshold
    }
}

fn batch_budget_active(per_request: Option<usize>, draft_batch_size: usize) -> bool {
    per_request.is_some() && draft_batch_size > 1
}

fn select(logits: &[f32], threshold: f32) -> Result<(usize, Vec<f32>)> {
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite DSpark confidence logit"
    );
    let scores: Vec<_> = logits.iter().map(|v| 1. / (1. + (-v).exp())).collect();
    let keep = scores
        .iter()
        .position(|&v| v < threshold)
        .unwrap_or(scores.len());
    Ok((keep, scores))
}

pub(super) fn allocate_batch_budget(conditional: &[Vec<f32>], budget: usize) -> Result<Vec<usize>> {
    let mut candidates = Vec::new();
    for (request, probabilities) in conditional.iter().enumerate() {
        let mut survival = 1.0f32;
        for (depth, &probability) in probabilities.iter().enumerate() {
            ensure!(
                probability.is_finite() && (0.0..=1.0).contains(&probability),
                "invalid draft confidence probability"
            );
            survival *= probability;
            candidates.push((survival, request, depth));
        }
    }
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut kept = vec![0; conditional.len()];
    for &(_, request, depth) in candidates.iter().take(budget.min(candidates.len())) {
        kept[request] = kept[request].max(depth + 1);
    }
    Ok(kept)
}

pub(super) fn allocate_cost_aware(
    conditional: &[Vec<f32>],
    curve: &CostCurve,
) -> Result<Vec<usize>> {
    ensure!(!conditional.is_empty(), "empty confidence cohort");
    let mut candidates = Vec::new();
    for (request, probabilities) in conditional.iter().enumerate() {
        let mut survival = 1.0f64;
        for (depth, &probability) in probabilities.iter().enumerate() {
            ensure!(
                probability.is_finite() && (0.0..=1.0).contains(&probability),
                "invalid draft confidence probability"
            );
            survival *= f64::from(probability);
            candidates.push((survival, request, depth));
        }
    }
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut cumulative = 0.;
    let mut best_budget = 0;
    let mut best_score = f64::NEG_INFINITY;
    for budget in 0..=candidates.len() {
        if budget > 0 {
            cumulative += candidates[budget - 1].0;
        }
        let elapsed = curve.draft_ns + curve.verify_ns(curve.base_rows + budget);
        let score = (curve.sampling_requests as f64 + cumulative) / elapsed;
        if score > best_score {
            best_score = score;
            best_budget = budget;
        }
    }
    let mut kept = vec![0; conditional.len()];
    for &(_, request, depth) in candidates.iter().take(best_budget) {
        kept[request] = kept[request].max(depth + 1);
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_gate_keeps_contiguous_verified_candidates() -> Result<()> {
        let logits = [0., 2., -4., 3.];
        assert_eq!(select(&logits, 0.)?.0, 4);
        assert_eq!(select(&logits, 0.5)?.0, 2);
        assert_eq!(select(&logits, 0.9)?.0, 0);
        assert!(select(&[f32::NAN], 0.).is_err());
        Ok(())
    }

    #[test]
    fn confidence_threshold_rises_only_for_batched_decode() {
        assert_eq!(threshold_for_batch(0.4, 0.1, 1), 0.4);
        assert_eq!(threshold_for_batch(0.4, 0.1, 6), 0.5);
        assert_eq!(threshold_for_batch(0.4, 0.0, 6), 0.4);
        assert_eq!(threshold_for_batch(0.8, 0.5, 6), 1.0);
        assert_eq!(threshold_for_batch(0.0, 0.1, 6), 0.0);
    }

    #[test]
    fn shared_budget_requires_a_real_grouped_draft_batch() {
        assert!(!batch_budget_active(Some(4), 1));
        assert!(batch_budget_active(Some(4), 2));
        assert!(!batch_budget_active(None, 6));
    }

    #[test]
    fn batch_budget_allocates_by_cumulative_survival_and_keeps_prefixes() -> Result<()> {
        let kept = allocate_batch_budget(&[vec![0.9, 0.9, 0.2], vec![0.7, 0.7, 0.7]], 3)?;
        assert_eq!(kept, [2, 1]);
        assert!(allocate_batch_budget(&[vec![f32::NAN]], 1).is_err());
        Ok(())
    }

    #[test]
    fn cost_aware_budget_prices_draft_yield_against_measured_step_cost() -> Result<()> {
        let curve = CostCurve {
            draft_ns: 10.,
            verify_points: vec![(2, 104.), (5, 106.), (10, 116.)],
            base_rows: 2,
            sampling_requests: 2,
        };
        let kept = allocate_cost_aware(&[vec![0.9, 0.9], vec![0.2, 0.2]], &curve)?;
        assert_eq!(kept, [2, 1]);
        assert_eq!(kept.iter().sum::<usize>(), 3);
        assert!(allocate_cost_aware(&[vec![f32::NAN]], &curve).is_err());
        Ok(())
    }

    #[test]
    fn cost_profile_waits_for_repeated_draft_and_multiple_verify_shapes() {
        let mut profile = CostProfile::default();
        for _ in 0..8 {
            profile.observe_draft(3, 3, 21, 50_000.);
        }
        for _ in 0..4 {
            profile.observe_verify(3, 4, 10, 100_000.);
            profile.observe_verify(3, 4, 20, 120_000.);
        }
        let curve = profile.curve(3, 3, 4, 21, 4, 4).unwrap();
        assert_eq!(curve.draft_ns, 50_000.);
        assert!((curve.verify_ns(15) - 110_000.).abs() < 0.1);
        assert!(profile.curve(3, 3, 3, 21, 4, 4).is_none());
        assert!(profile.curve(4, 3, 4, 21, 4, 4).is_none());
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to two inference GPUs"]
    fn confidence_uses_raw_collapse_and_previous_markov_embedding() -> Result<()> {
        use std::path::PathBuf;
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(&PathBuf::from(std::env::var("DS4_TEST_PACK")?), [0, 1])?;
        rt.layer(39)?;
        let width = 3;
        let values: Vec<_> = (0..width * 4 * DIM)
            .map(|i| crate::model::bf((i * 29 % 151) as f32 / 31. - 2.))
            .collect();
        let mut mix = vec![0.; width * 24];
        for (t, p) in mix.as_chunks_mut::<24>().0.iter_mut().enumerate() {
            p[..4].copy_from_slice(&[0.3125, 0.427, -0.14, 0.15 * t as f32]);
        }
        let q = rt.weights.vector("mtp.2.confidence_head.proj.weight")?;
        let embeddings = rt.weights.tensor("mtp.2.markov_head.embed.weight")?;
        let mut expected = Vec::new();
        for (t, id) in [21usize, 1234, 981].into_iter().enumerate() {
            let mut sum = 0f64;
            for d in 0..DIM {
                let v = crate::model::bf(
                    (0..4)
                        .map(|c| values[(t * 4 + c) * DIM + d] * mix[t * 24 + c])
                        .sum(),
                );
                sum += f64::from(v) * f64::from(q[d]);
            }
            let bytes = &embeddings.data[id * 512..(id + 1) * 512];
            for (e, &v) in bytes.as_chunks::<2>().0.iter().zip(&q[DIM..]) {
                sum += f64::from(half::bf16::from_le_bytes(*e).to_f32()) * f64::from(v);
            }
            expected.push(1. / (1. + (-sum).exp()));
        }
        let h = rt.to_device(&values)?;
        let mix = rt.to_device(&mix)?;
        let ids = rt.to_device(&[1234., 981., 17.])?;
        let (tokens, actual) = Gate {
            threshold: 0.,
            fallback_threshold: 0.,
            trace: true,
            batch_budget_per_request: None,
            adaptive_budget: false,
        }
        .evaluate(&mut rt, &h, &mix, 21, &ids)?;
        assert_eq!(tokens, [1234, 981, 17]);
        for (a, b) in actual.unwrap().into_iter().zip(expected) {
            assert!((f64::from(a) - b).abs() < 2e-5, "{a} vs {b}");
        }
        Ok(())
    }
}
