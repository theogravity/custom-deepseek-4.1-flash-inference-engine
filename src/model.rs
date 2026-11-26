use crate::{
    attention::{self, AttentionState},
    draft::DraftState,
    engram::Engram,
    moe,
    prefix::PrefixCache,
    runtime::{Dev, Runtime},
};
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

#[path = "prompt_lookup.rs"]
mod prompt_lookup;
#[path = "residual.rs"]
pub(crate) mod residual;

pub const DIM: usize = 5120;
#[cfg(test)]
#[inline(always)]
pub fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

struct Target {
    logits: Option<Dev>,
    features: Option<Dev>,
}

pub(crate) fn spec_k() -> Result<usize> {
    let mode = std::env::var("DS4_DRAFTER").unwrap_or_else(|_| "none".into());
    let default = match mode.as_str() {
        "none" => return Ok(0),
        "released" => 5,
        "code" => 7,
        _ => bail!("DS4_DRAFTER must be none, released or code"),
    };
    let width = match std::env::var("DS4_SPEC_K") {
        Ok(v) => v.parse::<usize>().context("invalid DS4_SPEC_K")?,
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => return Err(e.into()),
    };
    ensure!((1..=16).contains(&width), "DS4_SPEC_K must be in 1..=16");
    Ok(width)
}

// Geometric boundaries bound the number of snapshots on long prefills while
// retaining branches well before the final complete chunk.
fn checkpoints(prompt: usize, chunk: usize) -> Vec<usize> {
    let last = (prompt - 1) / chunk * chunk;
    let mut points = Vec::new();
    let mut boundary = chunk;
    while boundary < last {
        if boundary >= 256 {
            points.push(boundary);
        }
        boundary *= 2;
    }
    if last > 0 {
        points.push(last);
    }
    points
}

fn cached<T>(result: Result<T>, operation: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            eprintln!("prefix {operation} skipped: {error:#}");
            None
        }
    }
}

// The target draws one independent sample for each candidate position on GPU.
// Rejected suffix samples are discarded with the corresponding target state.
struct Scores(Vec<u32>);
impl Scores {
    fn read(rt: &mut Runtime, target: &Target, rows: usize, temperature: f32) -> Result<Self> {
        let logits = target.logits.as_ref().context("missing target scores")?;
        Ok(Self(rt.sample(logits, rows, temperature)?))
    }
    fn sample(&self, row: usize) -> Result<u32> {
        self.0.get(row).copied().context("missing target sample")
    }
}
// Consume independent target draws only through the first rejection. The next
// anchor is always sampled by the target, including when every proposal agrees.
fn verify_proposals(
    proposals: &[u32],
    mut draw: impl FnMut(usize) -> Result<u32>,
) -> Result<(usize, u32)> {
    for (i, &proposed) in proposals.iter().enumerate() {
        let token = draw(i)?;
        if token != proposed || token == 1 {
            return Ok((i, token));
        }
    }
    Ok((proposals.len(), draw(proposals.len())?))
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum ProposalSource {
    Target,
    Learned,
    Lookup,
}

#[derive(Default)]
struct DraftStats {
    learned: Counts,
    lookup: Counts,
}

#[derive(Default)]
struct Counts {
    rounds: usize,
    proposed: usize,
    accepted: usize,
}

impl DraftStats {
    fn record(&mut self, source: ProposalSource, proposed: usize, accepted: usize) {
        let counts = match source {
            ProposalSource::Target => return,
            ProposalSource::Learned => &mut self.learned,
            ProposalSource::Lookup => &mut self.lookup,
        };
        counts.rounds += 1;
        counts.proposed += proposed;
        counts.accepted += accepted;
    }
}

impl Drop for DraftStats {
    fn drop(&mut self) {
        for (name, count) in [("learned", &self.learned), ("lookup", &self.lookup)] {
            if count.rounds != 0 {
                eprintln!(
                    "speculative source={name} rounds={} proposed={} accepted={}",
                    count.rounds, count.proposed, count.accepted
                );
            }
        }
    }
}

pub struct Model {
    prefix: PrefixCache,
    spec_k: usize,
    lookup: prompt_lookup::Config,
    rt: Runtime,
    engram: Engram,
    max_ctx: usize,
    chunk: usize,
}
impl Model {
    pub fn open(
        pack: &Path,
        tok: &Tokenizer,
        devices: [i32; 2],
        max_ctx: usize,
        chunk: usize,
    ) -> Result<Self> {
        ensure!(
            chunk > 0 && chunk <= 512 && (1..=524288).contains(&max_ctx),
            "invalid chunk or context limit"
        );
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(pack.join("config.json"))?)?;
        ensure!(
            cfg["model_type"] == "deepseek_v41"
                && cfg["text_config"]["hidden_size"] == DIM
                && cfg["text_config"]["num_hidden_layers"] == 40
                && cfg["eos_token_id"] == 1,
            "unexpected model configuration"
        );
        Ok(Self {
            prefix: PrefixCache::new(pack, chunk, max_ctx)?,
            spec_k: spec_k()?,
            lookup: prompt_lookup::configured()?,
            rt: Runtime::open(pack, devices)?,
            engram: Engram::new(tok)?,
            max_ctx,
            chunk,
        })
    }
    pub fn max_context(&self) -> usize {
        self.max_ctx
    }
    fn forward(
        &mut self,
        cache: &mut AttentionState,
        ids: &[u32],
        pos: usize,
        b: usize,
        logit_rows: usize,
        capture_features: bool,
    ) -> Result<Target> {
        ensure!(
            logit_rows <= b,
            "requested more logit rows than target inputs"
        );
        let mut features = None;
        self.rt.layer(0)?;
        let mut h = residual::embed(&mut self.rt, &ids[pos..pos + b])?;
        let mut mix = residual::initial_mix(&mut self.rt, b)?;
        let mut engram_rows: Option<crate::engram::EngramRows> = None;
        for layer in 0..40 {
            let stem = format!("layers.{layer}");
            if layer == 20 {
                let host_h = self.rt.to_host(&h)?;
                let host_mix = self.rt.to_host(&mix)?;
                self.rt.layer(layer)?;
                h = self.rt.to_device(&host_h)?;
                mix = self.rt.to_device(&host_mix)?;
            }
            if let Some(rows) = engram_rows.take() {
                rows.apply(&mut self.rt, &mut h)?;
            }
            if capture_features && layer >= 37 {
                if layer == 37 {
                    features = Some(self.rt.buffer(b * 3 * DIM)?);
                }
                residual::capture(&self.rt, &h, features.as_mut().unwrap(), layer - 37, b)?;
            }
            for branch in ["attn", "ffn"] {
                let input = residual::prepare(&mut self.rt, &stem, branch, &h, &mut mix, b)?;
                let output = if branch == "attn" {
                    attention::forward_dev(&mut self.rt, cache, layer, pos, &input, b)?
                } else {
                    moe::forward_dev(&mut self.rt, layer, &input, b)?
                };
                residual::expand(&self.rt, &mut h, &output, &mix, b)?;
                if branch == "attn" && (layer == 0 || layer == 13) {
                    // Gather the next Engram's mmap rows while this layer's
                    // attention output and residual expansion run on the GPU.
                    engram_rows =
                        Some(
                            self.engram
                                .gather(&self.rt.weights, layer + 1, ids, pos, b)?,
                        );
                }
            }
        }
        let logits = if logit_rows > 0 {
            let input = residual::readout(
                &mut self.rt,
                &h,
                &mut mix,
                "norm.weight",
                b - logit_rows,
                logit_rows,
            )?;
            Some(self.rt.linear_dev("head.weight", &input, logit_rows, 1)?)
        } else {
            None
        };
        Ok(Target { logits, features })
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        temperature: f32,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.generate_cancellable(input, limit, temperature, || false, emit)
    }
    pub fn generate_cancellable(
        &mut self,
        input: &[u32],
        limit: usize,
        temperature: f32,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        ensure!(
            !input.is_empty() && input.len() <= self.max_ctx && input.iter().all(|t| *t < 129280),
            "invalid prompt length or token ID"
        );
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "invalid temperature"
        );
        let limit = limit.min(self.max_ctx - input.len());
        if limit == 0 {
            return Ok(0);
        }
        let mut ids = input.to_vec();
        // Cache a completed prefill chunk. Reusing the same chunk boundaries preserves
        // the sparse attention and projection rounding seen by a fresh request.
        let points = if self.prefix.enabled() {
            checkpoints(input.len(), self.chunk)
        } else {
            vec![]
        };
        let snapshot_pos = points.last().copied().unwrap_or(0);
        let span = input.len().min(128);
        let (mut pos, mut cache) =
            cached(self.prefix.restore(input, snapshot_pos, span), "restore")
                .flatten()
                .unwrap_or_default();
        let mut next_point = points.partition_point(|&at| at <= pos);
        let mut draft = (self.spec_k > 0 && limit > 2).then(DraftState::new);
        if pos > 0 && draft.is_some() {
            let restored = self.prefix.read_draft(input, pos, span).and_then(|bytes| {
                bytes
                    .map(|bytes| DraftState::restore(&mut self.rt, pos, &bytes))
                    .transpose()
            });
            if let Some(state) = cached(restored, "DSpark restore").flatten() {
                eprintln!("prefix DSpark tokens={pos}");
                draft = Some(state);
            }
        }
        let mut draft_stats = DraftStats::default();
        let lookup = prompt_lookup::PromptLookup::new(input, self.lookup);
        cache.begin_prefill(input.len());
        let reused = pos;
        let started = std::time::Instant::now();
        let mut first = None;
        while pos < input.len() {
            ensure!(!cancelled(), "request cancelled");
            if points.get(next_point) == Some(&pos) {
                if cached(self.prefix.store(input, pos, span, &cache), "store").is_some() {
                    if let Some(state) = draft.as_ref().filter(|d| d.ready(pos)) {
                        let saved = state
                            .snapshot(&mut self.rt, pos)
                            .and_then(|data| self.prefix.write_draft(input, pos, span, &data));
                        cached(saved, "DSpark store");
                    }
                }
                next_point += 1;
            }
            let n = self.chunk.min(input.len() - pos);
            let last = pos + n == input.len();
            // Seed only chunks needed by the next saved window or the actual decode.
            let horizon = points.get(next_point).copied().unwrap_or(input.len());
            let observe = draft.is_some() && pos + n > horizon.saturating_sub(128);
            let target = self.forward(&mut cache, &ids, pos, n, usize::from(last), observe)?;
            if let Some(draft) = draft.as_mut().filter(|_| observe) {
                draft.observe(
                    &mut self.rt,
                    target
                        .features
                        .as_ref()
                        .context("missing DSpark prefill features")?,
                    pos,
                    n,
                )?;
            }
            if last {
                first = Some(Scores::read(&mut self.rt, &target, 1, temperature)?);
            }
            pos += n;
        }
        eprintln!(
            "prefill tokens={} reused={} seconds={:.3}",
            input.len(),
            reused,
            started.elapsed().as_secs_f64()
        );
        let mut token = first.context("missing prefill logits")?.sample(0)?;
        let mut emitted = 0;
        loop {
            ensure!(!cancelled(), "request cancelled");
            emitted += 1;
            if token == 1 || !emit(token) || emitted == limit {
                return Ok(emitted);
            }
            // A restored target snapshot can start after the drafter's window.
            // Seed the complete visible history before generating its first proposal.
            ids.push(token);
            let remaining = limit - emitted - 1;
            let width = self.spec_k.min(remaining);
            let learned = draft.as_mut().filter(|d| width > 0 && d.ready(pos));
            let mut proposed = lookup.propose(&ids, remaining, learned.is_some());
            let mut source = if proposed.is_empty() {
                ProposalSource::Target
            } else {
                ProposalSource::Lookup
            };
            if source == ProposalSource::Target {
                if let Some(draft) = learned {
                    source = ProposalSource::Learned;
                    proposed = draft.propose(&mut self.rt, pos, token, width)?;
                    ensure!(
                        proposed.len() <= width && proposed.iter().all(|&t| t < 129280),
                        "invalid DSpark proposal"
                    );
                }
            }
            ids.extend_from_slice(&proposed);
            let batch = 1 + proposed.len();
            if batch > 1 {
                cache.begin_verify(pos, batch)?;
            }
            self.rt.verification = batch > 1;
            let target = self.forward(&mut cache, &ids, pos, batch, batch, draft.is_some());
            self.rt.verification = false;
            let target = target?;
            let scores = Scores::read(&mut self.rt, &target, batch, temperature)?;
            let (accepted, next) = verify_proposals(&proposed, |row| scores.sample(row))?;
            let committed = accepted + 1; // the previously emitted input anchor is always committed
            if batch > 1 {
                cache.finish_verify(committed)?;
            }
            draft_stats.record(source, proposed.len(), accepted);
            if let Some(draft) = draft.as_mut() {
                if source == ProposalSource::Learned {
                    draft.report_confidence(proposed.len(), accepted);
                }
                draft.observe(
                    &mut self.rt,
                    target
                        .features
                        .as_ref()
                        .context("missing DSpark verification features")?,
                    pos,
                    committed,
                )?;
            }
            pos += committed;
            ids.truncate(pos);
            for &verified in &proposed[..accepted] {
                ensure!(!cancelled(), "request cancelled");
                emitted += 1;
                if !emit(verified) {
                    return Ok(emitted);
                }
            }
            token = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_boundaries_and_draft_observations_cover_each_live_window() {
        assert_eq!(checkpoints(1025, 64), [256, 512, 1024]);
        assert_eq!(checkpoints(1050, 48), [384, 768, 1008]);
        assert_eq!(checkpoints(33, 32), [32]);
        assert!(checkpoints(32, 32).is_empty());
        // For each saved point and the final decode, every live row lies in
        // an observed chunk even for widths that do not divide 128.
        for chunk in [1, 3, 48, 64, 255, 512] {
            let prompt: usize = 1803;
            let mut observed = vec![false; prompt];
            let mut horizons = checkpoints(prompt, chunk);
            horizons.push(prompt);
            for pos in (0..prompt).step_by(chunk) {
                let end = (pos + chunk).min(prompt);
                let next = horizons.iter().copied().find(|&h| h > pos).unwrap();
                if end > next.saturating_sub(128) {
                    observed[pos..end].fill(true);
                }
            }
            for h in horizons {
                assert!(observed[h.saturating_sub(128)..h].iter().all(|&v| v));
            }
        }
    }

    #[test]
    fn learned_attempt_with_zero_confident_rows_counts_as_a_draft_round() {
        let mut stats = DraftStats::default();
        stats.record(ProposalSource::Learned, 0, 0);
        stats.record(ProposalSource::Learned, 5, 3);
        stats.record(ProposalSource::Lookup, 8, 4);
        stats.record(ProposalSource::Target, 0, 0);
        assert_eq!(
            (
                stats.learned.rounds,
                stats.learned.proposed,
                stats.learned.accepted
            ),
            (2, 5, 3)
        );
        assert_eq!(
            (
                stats.lookup.rounds,
                stats.lookup.proposed,
                stats.lookup.accepted
            ),
            (1, 8, 4)
        );
    }

    #[test]
    fn speculative_acceptance_draws_once_per_position_until_mismatch_or_eos() -> Result<()> {
        let proposed = [3, 5, 7];
        for (draws, expected) in [
            (vec![2], (0, 2)),
            (vec![3, 2], (1, 2)),
            (vec![3, 5, 8], (2, 8)),
            (vec![3, 5, 7, 9], (3, 9)),
            (vec![3, 1], (1, 1)),
        ] {
            let mut seen = Vec::new();
            assert_eq!(
                verify_proposals(&proposed, |row| {
                    seen.push(row);
                    Ok(draws[row])
                })?,
                expected
            );
            assert_eq!(seen, (0..draws.len()).collect::<Vec<_>>());
        }
        assert_eq!(verify_proposals(&[3, 1], |row| Ok([3, 1][row]))?, (1, 1));
        assert_eq!(verify_proposals(&[], |_| Ok(0))?, (0, 0));
        Ok(())
    }
}
