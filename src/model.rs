use crate::{
    attention::AttentionState,
    draft::DraftState,
    engram::Engram,
    prefix::PrefixCache,
    runtime::{Dev, Runtime},
};
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

#[path = "batch.rs"]
pub(crate) mod batch;
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

struct ForwardItem<'a> {
    cache: &'a mut AttentionState,
    ids: &'a [u32],
    pos: usize,
    rows: usize,
    logit_rows: usize,
    capture_features: bool,
}

/// One request's model state. After the last prefill chunk, ids[pos] is the
/// sampled anchor already delivered to the client and awaiting target KV.
pub(crate) struct Session {
    ids: Vec<u32>,
    prompt_len: usize,
    pos: usize,
    limit: usize,
    temperature: f32,
    cache: AttentionState,
    draft: Option<DraftState>,
    lookup: prompt_lookup::PromptLookup,
    stats: DraftStats,
    points: Vec<usize>,
    next_point: usize,
    reused: usize,
    started: std::time::Instant,
    emitted: usize,
    finished: bool,
}

#[derive(Default)]
pub(crate) struct Step {
    pub tokens: Vec<u32>,
    pub eos: bool,
    pub finished: bool,
}

impl Session {
    /// Number of committed target input positions currently in this request's KV.
    pub(crate) fn processed_tokens(&self) -> usize {
        self.pos
    }

    pub(crate) fn is_prefill(&self) -> bool {
        !self.finished && self.pos < self.prompt_len
    }

    fn emit(&mut self, mut tokens: Vec<u32>, next: u32) -> Step {
        self.emitted += tokens.len() + 1;
        let eos = next == 1;
        self.finished = eos || self.emitted == self.limit;
        if !eos {
            tokens.push(next);
            if !self.finished {
                self.ids.push(next);
            }
        }
        Step {
            tokens,
            eos,
            finished: self.finished,
        }
    }
}

struct DecodePlan {
    source: ProposalSource,
    rows: usize,
}

struct DecodeWork<'a> {
    index: usize,
    session: &'a mut Session,
    plan: DecodePlan,
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
        self.forward_batch(&mut [ForwardItem {
            cache,
            ids,
            pos,
            rows: b,
            logit_rows,
            capture_features,
        }])?
        .pop()
        .context("missing single-sequence target")?
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
        let mut session = self.start_request(input.to_vec(), limit, temperature)?;
        let mut delivered = 0;
        while !session.finished {
            ensure!(!cancelled(), "request cancelled");
            let step = self.step_batch(&mut [&mut session]).pop().unwrap()?;
            for token in step.tokens {
                ensure!(!cancelled(), "request cancelled");
                delivered += 1;
                if !emit(token) {
                    return Ok(delivered);
                }
            }
            if step.eos {
                ensure!(!cancelled(), "request cancelled");
                delivered += 1;
            }
        }
        Ok(delivered)
    }

    pub(crate) fn start_request(
        &mut self,
        ids: Vec<u32>,
        limit: usize,
        temperature: f32,
    ) -> Result<Session> {
        let prompt_len = ids.len();
        ensure!(
            prompt_len > 0 && prompt_len <= self.max_ctx && ids.iter().all(|t| *t < 129280),
            "invalid prompt length or token ID"
        );
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "invalid temperature"
        );
        let limit = limit.min(self.max_ctx - prompt_len);
        let points = if limit > 0 && self.prefix.enabled() {
            checkpoints(prompt_len, self.chunk)
        } else {
            vec![]
        };
        let span = prompt_len.min(128);
        let (pos, mut cache) = points
            .last()
            .and_then(|&last| cached(self.prefix.restore(&ids, last, span), "restore").flatten())
            .unwrap_or_default();
        let mut draft = (self.spec_k > 0 && limit > 2).then(DraftState::new);
        if pos > 0 && draft.is_some() {
            let restored = self.prefix.read_draft(&ids, pos, span).and_then(|bytes| {
                bytes
                    .map(|b| DraftState::restore(&mut self.rt, pos, &b))
                    .transpose()
            });
            if let Some(state) = cached(restored, "DSpark restore").flatten() {
                eprintln!("prefix DSpark tokens={pos}");
                draft = Some(state);
            }
        }
        let next_point = points.partition_point(|&at| at <= pos);
        let lookup = prompt_lookup::PromptLookup::new(&ids, self.lookup);
        cache.begin_prefill_with_limit(prompt_len, prompt_len + limit);
        Ok(Session {
            ids,
            prompt_len,
            pos,
            limit,
            temperature,
            cache,
            draft,
            lookup,
            stats: DraftStats::default(),
            points,
            next_point,
            reused: pos,
            started: std::time::Instant::now(),
            emitted: 0,
            finished: limit == 0,
        })
    }

    /// Advance each admitted request by one prefill chunk or one verification
    /// round. Each result corresponds to its input slot. Drop failed sessions.
    pub(crate) fn step_batch(&mut self, sessions: &mut [&mut Session]) -> Vec<Result<Step>> {
        let mut results: Vec<_> = (0..sessions.len()).map(|_| None).collect();
        let mut work = Vec::new();
        let mut rows = 0;
        for (index, session) in sessions.iter_mut().enumerate() {
            if session.finished {
                results[index] = Some(Ok(Step {
                    finished: true,
                    ..Step::default()
                }));
            } else if session.is_prefill() {
                // Retain the same chunk boundaries and prefill precision as the
                // snapshot's original request. Admit just one chunk this tick.
                results[index] = Some(self.prefill_step(session));
            } else {
                match self.plan_decode(session) {
                    Ok(plan) => {
                        if rows + plan.rows > 512 {
                            self.decode_group(&mut work, &mut results);
                            work.clear();
                            rows = 0;
                        }
                        rows += plan.rows;
                        work.push(DecodeWork {
                            index,
                            session,
                            plan,
                        });
                    }
                    Err(e) => results[index] = Some(Err(e)),
                }
            }
        }
        if !work.is_empty() {
            self.decode_group(&mut work, &mut results);
        }
        results.into_iter().map(Option::unwrap).collect()
    }

    fn prefill_step(&mut self, session: &mut Session) -> Result<Step> {
        let pos = session.pos;
        let prompt = &session.ids[..session.prompt_len];
        let span = prompt.len().min(128);
        if session.points.get(session.next_point) == Some(&pos) {
            if cached(
                self.prefix.store(prompt, pos, span, &session.cache),
                "store",
            )
            .is_some()
            {
                if let Some(draft) = session.draft.as_ref().filter(|d| d.ready(pos)) {
                    let saved = draft
                        .snapshot(&mut self.rt, pos)
                        .and_then(|data| self.prefix.write_draft(prompt, pos, span, &data));
                    cached(saved, "DSpark store");
                }
            }
            session.next_point += 1;
        }
        let n = self.chunk.min(prompt.len() - pos);
        let last = pos + n == prompt.len();
        let horizon = session
            .points
            .get(session.next_point)
            .copied()
            .unwrap_or(prompt.len());
        let observe = session.draft.is_some() && pos + n > horizon.saturating_sub(128);
        let target = self.forward(
            &mut session.cache,
            prompt,
            pos,
            n,
            usize::from(last),
            observe,
        )?;
        if let Some(draft) = session.draft.as_mut().filter(|_| observe) {
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
        session.pos += n;
        if !last {
            return Ok(Step::default());
        }
        let token = Scores::read(&mut self.rt, &target, 1, session.temperature)?.sample(0)?;
        eprintln!(
            "prefill tokens={} reused={} seconds={:.3}",
            session.prompt_len,
            session.reused,
            session.started.elapsed().as_secs_f64()
        );
        Ok(session.emit(Vec::new(), token))
    }

    fn plan_decode(&mut self, session: &mut Session) -> Result<DecodePlan> {
        let remaining = session.limit - session.emitted - 1;
        let width = self.spec_k.min(remaining);
        let learned = session
            .draft
            .as_mut()
            .filter(|d| width > 0 && d.ready(session.pos));
        let mut proposed = session
            .lookup
            .propose(&session.ids, remaining, learned.is_some());
        let source = if !proposed.is_empty() {
            ProposalSource::Lookup
        } else if let Some(draft) = learned {
            self.rt.layer(39)?;
            proposed = draft.propose(&mut self.rt, session.pos, session.ids[session.pos], width)?;
            ensure!(
                proposed.len() <= width && proposed.iter().all(|&t| t < 129280),
                "invalid DSpark proposal"
            );
            ProposalSource::Learned
        } else {
            ProposalSource::Target
        };
        let rows = proposed.len() + 1;
        if rows > 1 {
            session.cache.begin_verify(session.pos, rows)?;
        }
        session.ids.extend_from_slice(&proposed);
        Ok(DecodePlan { source, rows })
    }

    fn decode_group(&mut self, work: &mut [DecodeWork<'_>], results: &mut [Option<Result<Step>>]) {
        let mut items: Vec<_> = work
            .iter_mut()
            .map(|w| ForwardItem {
                cache: &mut w.session.cache,
                ids: &w.session.ids,
                pos: w.session.pos,
                rows: w.plan.rows,
                logit_rows: w.plan.rows,
                capture_features: w.session.draft.is_some(),
            })
            .collect();
        self.rt.verification = items.len() > 1 || items[0].rows > 1;
        let targets = self.forward_batch(&mut items);
        self.rt.verification = false;
        match targets {
            Ok(targets) if targets.len() == work.len() => {
                for (one, target) in work.iter_mut().zip(targets) {
                    results[one.index] =
                        Some(target.and_then(|t| self.finish_decode(one.session, &one.plan, t)));
                }
            }
            other => {
                let message = other.err().map_or_else(
                    || "wrong number of target outputs".to_string(),
                    |e| format!("{e:#}"),
                );
                for one in work {
                    results[one.index] =
                        Some(Err(anyhow::anyhow!("target forward failed: {message}")));
                }
            }
        }
    }

    fn finish_decode(
        &mut self,
        session: &mut Session,
        plan: &DecodePlan,
        target: Target,
    ) -> Result<Step> {
        let pos = session.pos;
        let proposed = &session.ids[pos + 1..];
        let scores = Scores::read(&mut self.rt, &target, plan.rows, session.temperature)?;
        let (accepted, next) = verify_proposals(proposed, |row| scores.sample(row))?;
        let committed = accepted + 1;
        if plan.rows > 1 {
            session.cache.finish_verify(committed)?;
        }
        session.stats.record(plan.source, proposed.len(), accepted);
        if let Some(draft) = session.draft.as_mut() {
            if plan.source == ProposalSource::Learned {
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
        let tokens = proposed[..accepted].to_vec();
        session.pos += committed;
        session.ids.truncate(session.pos);
        Ok(session.emit(tokens, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_sessions_emit_verified_tokens_and_count_eos_once() {
        fn sequence(limit: usize) -> Session {
            let ids = vec![5, 6, 7];
            Session {
                prompt_len: ids.len(),
                pos: ids.len(),
                limit,
                temperature: 0.,
                cache: AttentionState::default(),
                draft: None,
                lookup: prompt_lookup::PromptLookup::new(&ids, Default::default()),
                stats: DraftStats::default(),
                points: vec![],
                next_point: 0,
                reused: 0,
                started: std::time::Instant::now(),
                emitted: 0,
                finished: false,
                ids,
            }
        }
        let mut length = sequence(3);
        let mut stop = sequence(12);
        let first = length.emit(vec![], 20);
        assert_eq!(first.tokens, [20]);
        assert!(!first.finished && !first.eos);
        assert_eq!(length.ids[length.pos], 20);
        let last = length.emit(vec![30], 40);
        assert_eq!(last.tokens, [30, 40]);
        assert!(last.finished && !last.eos);
        assert_eq!(length.emitted, 3);
        assert_eq!(stop.emitted, 0);
        let first = stop.emit(vec![], 22);
        assert_eq!(first.tokens, [22]);
        let last = stop.emit(vec![90, 91], 1);
        assert_eq!(last.tokens, [90, 91]);
        assert!(last.finished && last.eos);
        assert_eq!(stop.emitted, 4);
    }

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
