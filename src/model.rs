use crate::{
    attention::AttentionState,
    draft::{self, DraftInput, DraftState},
    engram::Engram,
    prefix::PrefixCache,
    restore_timing::{timed, Host},
    runtime::{Dev, Runtime},
    swap::{SwapHandle, SwapStore},
};
use anyhow::{bail, ensure, Context, Result};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::{path::Path, sync::Arc};
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
    rng: Option<StdRng>,
    suspended: Option<SwapHandle>,
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

    pub(crate) fn is_suspended(&self) -> bool {
        self.suspended.is_some()
    }

    fn scores(&mut self, rt: &mut Runtime, target: &Target, rows: usize) -> Result<Scores> {
        let logits = target.logits.as_ref().context("missing target scores")?;
        let seed = self.rng.as_mut().map_or(0, RngCore::next_u64);
        Ok(Scores(rt.sample(logits, rows, self.temperature, seed)?))
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

struct Proposal {
    source: ProposalSource,
    tokens: Vec<u32>,
    learned_width: usize,
}

fn rotation_payload(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let (body, footer) = bytes.split_at(bytes.len().saturating_sub(16));
    ensure!(
        footer.len() == 16 && &footer[8..] == b"DS4LIVE1",
        "invalid live session payload"
    );
    let draft_len: usize = u64::from_le_bytes(footer[..8].try_into()?).try_into()?;
    let target_len = body
        .len()
        .checked_sub(draft_len)
        .context("invalid live draft size")?;
    ensure!(target_len >= 16, "missing live target cache");
    Ok(body.split_at(target_len))
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
    batch_drafter: bool,
    rt: Runtime,
    engram: Arc<Engram>,
    overlap_prefill_engram: bool,
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
            batch_drafter: match std::env::var("DS4_BATCH_DRAFT").as_deref() {
                Ok("0") => false,
                Ok("1") | Err(std::env::VarError::NotPresent) => true,
                _ => bail!("DS4_BATCH_DRAFT must be 0 or 1"),
            },
            rt: Runtime::open(pack, devices)?,
            engram: Arc::new(Engram::new(tok)?),
            overlap_prefill_engram: match std::env::var("DS4_OVERLAP_PREFILL_ENGRAM").as_deref() {
                Ok("1") => true,
                Ok("0") | Err(std::env::VarError::NotPresent) => false,
                _ => bail!("DS4_OVERLAP_PREFILL_ENGRAM must be 0 or 1"),
            },
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
        let (pos, mut cache, mut timing) = points
            .last()
            .and_then(|&last| cached(self.prefix.restore(&ids, last, span), "restore").flatten())
            .unwrap_or_default();
        let mut draft = (self.spec_k > 0 && limit > 2).then(DraftState::new);
        if pos > 0 && draft.is_some() {
            let (sidecar, read_time) =
                timed(timing.is_some(), || self.prefix.read_draft(&ids, pos, span));
            if let Some(profile) = &mut timing {
                profile.draft_read(read_time);
            }
            if let Some((tier, bytes)) = cached(sidecar, "DSpark restore").flatten() {
                let (restored, elapsed) = timed(timing.is_some(), || {
                    DraftState::restore(&mut self.rt, pos, &bytes)
                });
                if let Some(profile) = &mut timing {
                    profile.draft_restore(tier, bytes.len(), elapsed, restored.is_ok());
                }
                if let Some(state) = cached(restored, "DSpark restore") {
                    eprintln!("prefix DSpark tokens={pos}");
                    draft = Some(state);
                }
            }
        }
        let next_point = points.partition_point(|&at| at <= pos);
        let lookup = prompt_lookup::PromptLookup::new(&ids, self.lookup);
        cache.begin_prefill_with_limit(prompt_len, prompt_len + limit);
        if let Some(profile) = timing {
            cache.profile_restore(profile.probe(pos));
        }
        Ok(Session {
            ids,
            prompt_len,
            pos,
            limit,
            temperature,
            rng: (temperature > 0.).then(StdRng::from_entropy),
            suspended: None,
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

    /// Persist committed state before releasing either GPU cache. A failed
    /// backing allocation leaves this request runnable in its original slot.
    pub(crate) fn suspend(&mut self, session: &mut Session, backing: &SwapStore) -> Result<()> {
        ensure!(!session.finished, "cannot suspend a finished request");
        if session.is_suspended() {
            return Ok(());
        }
        let draft = session
            .draft
            .as_ref()
            .map(|d| d.rotation_snapshot(&mut self.rt, session.pos))
            .transpose()?;
        let draft = draft.as_deref().unwrap_or_default();
        let tail = draft.len() + 16;
        let bytes = AttentionState::snapshot_bytes(session.pos)?
            .checked_add(tail)
            .context("live KV image size overflow")?;
        let handle = backing.put_with(bytes, || {
            let mut payload = session.cache.encode_snapshot_with_tail(session.pos, tail)?;
            payload.extend_from_slice(draft);
            payload.extend_from_slice(&(draft.len() as u64).to_le_bytes());
            payload.extend_from_slice(b"DS4LIVE1");
            Ok(payload)
        })?;
        session.suspended = Some(handle);
        session.cache = AttentionState::default();
        if let Some(draft) = session.draft.take() {
            draft.release_device();
        }
        Ok(())
    }

    /// Rebuild both caches at the recorded position without generating or
    /// sampling. Keep the handle until both caches are ready to commit.
    pub(crate) fn resume(&mut self, session: &mut Session) -> Result<()> {
        let Some(handle) = session.suspended.as_ref() else {
            return Ok(());
        };
        let mut timing = Host::start("live");
        let bytes = handle.read()?;
        if let Some(profile) = &mut timing {
            profile.backing_read(handle.tier());
        }
        let (decoded, elapsed) = timed(timing.is_some(), || -> Result<_> {
            let (target, draft) = rotation_payload(&bytes)?;
            let cache = AttentionState::restore_rotation(
                target,
                session.pos,
                session.prompt_len,
                session.prompt_len + session.limit,
            )?;
            Ok((cache, target.len(), draft))
        });
        let (mut cache, target_bytes, draft) = decoded?;
        if let Some(profile) = &mut timing {
            profile.decoded(target_bytes, elapsed);
        }
        let draft = if draft.is_empty() {
            None
        } else {
            let (restored, elapsed) = timed(timing.is_some(), || {
                DraftState::restore_rotation(&mut self.rt, session.pos, draft)
            });
            let state = restored?;
            if let Some(profile) = &mut timing {
                profile.draft_restore(handle.tier(), draft.len(), elapsed, true);
            }
            Some(state)
        };
        if let Some(profile) = timing {
            cache.profile_restore(profile.probe(session.pos));
        }
        session.cache = cache;
        session.draft = draft;
        session.suspended = None;
        Ok(())
    }

    /// Advance each admitted request by one prefill chunk or one verification
    /// round. Each result corresponds to its input slot. Drop failed sessions.
    pub(crate) fn step_batch(&mut self, sessions: &mut [&mut Session]) -> Vec<Result<Step>> {
        let mut results: Vec<_> = (0..sessions.len()).map(|_| None).collect();
        let mut decoding = Vec::new();
        for (index, session) in sessions.iter_mut().enumerate() {
            if session.is_suspended() {
                results[index] = Some(Err(anyhow::anyhow!(
                    "suspended request must resume before execution"
                )));
            } else if session.finished {
                results[index] = Some(Ok(Step {
                    finished: true,
                    ..Step::default()
                }));
            } else if session.is_prefill() {
                results[index] = Some(self.prefill_step(session));
            } else {
                let remaining = session.limit - session.emitted - 1;
                let width = self.spec_k.min(remaining);
                let learned =
                    width > 0 && session.draft.as_ref().is_some_and(|d| d.ready(session.pos));
                let tokens = session.lookup.propose(&session.ids, remaining, learned);
                let source = if !tokens.is_empty() {
                    ProposalSource::Lookup
                } else if learned {
                    ProposalSource::Learned
                } else {
                    ProposalSource::Target
                };
                decoding.push((
                    index,
                    &mut **session,
                    Proposal {
                        source,
                        tokens,
                        learned_width: width,
                    },
                ));
            }
        }
        let mut inputs = Vec::new();
        let mut slots = Vec::new();
        for (slot, (_, s, plan)) in decoding.iter_mut().enumerate() {
            if plan.source == ProposalSource::Learned {
                slots.push(slot);
                inputs.push(DraftInput {
                    state: s.draft.as_mut().unwrap(),
                    pos: s.pos,
                    seed: s.ids[s.pos],
                    width: plan.learned_width,
                });
            }
        }
        let drafts = self.learned_proposals(&mut inputs);
        for (slot, proposed) in slots.into_iter().zip(drafts) {
            match proposed {
                Ok(tokens) => decoding[slot].2.tokens = tokens,
                Err(e) => results[decoding[slot].0] = Some(Err(e)),
            }
        }
        let mut work = Vec::new();
        let mut rows = 0;
        for (index, session, proposal) in decoding {
            if results[index].is_some() {
                continue;
            }
            match Self::plan_decode(session, proposal) {
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
        self.rt.target_prefill = true;
        let target = self.forward(
            &mut session.cache,
            prompt,
            pos,
            n,
            usize::from(last),
            observe,
        );
        self.rt.target_prefill = false;
        let target = target?;
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
        let token = session.scores(&mut self.rt, &target, 1)?.sample(0)?;
        eprintln!(
            "prefill tokens={} reused={} seconds={:.3}",
            session.prompt_len,
            session.reused,
            session.started.elapsed().as_secs_f64()
        );
        Ok(session.emit(Vec::new(), token))
    }

    fn learned_proposals(&mut self, inputs: &mut [DraftInput<'_>]) -> Vec<Result<Vec<u32>>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if let Err(e) = self.rt.layer(39) {
            return inputs
                .iter()
                .map(|_| Err(anyhow::anyhow!("{e:#}")))
                .collect();
        }
        if self.batch_drafter && inputs.len() > 1 {
            match draft::propose_many(&mut self.rt, inputs) {
                Ok(proposals) => return proposals.into_iter().map(Ok).collect(),
                // No target or draft KV has been modified. Retry on the smaller
                // per-request workspace, isolating a malformed individual ring.
                Err(e) => eprintln!("grouped DSpark proposal failed: {e:#}; retrying separately"),
            }
        }
        inputs
            .iter_mut()
            .map(|r| r.state.propose(&mut self.rt, r.pos, r.seed, r.width))
            .collect()
    }

    fn plan_decode(session: &mut Session, proposal: Proposal) -> Result<DecodePlan> {
        if proposal.source == ProposalSource::Learned {
            ensure!(
                proposal.tokens.len() <= proposal.learned_width
                    && proposal.tokens.iter().all(|&t| t < 129280),
                "invalid DSpark proposal"
            );
        }
        let rows = proposal.tokens.len() + 1;
        if rows > 1 {
            session.cache.begin_verify(session.pos, rows)?;
        }
        session.ids.extend_from_slice(&proposal.tokens);
        Ok(DecodePlan {
            source: proposal.source,
            rows,
        })
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
        let scores = session.scores(&mut self.rt, &target, plan.rows)?;
        let proposed = &session.ids[pos + 1..];
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
    fn live_payload_checks_trailer_and_lengths() {
        let target = [42; 16];
        let mut bytes = target.to_vec();
        bytes.extend_from_slice(b"draft");
        bytes.extend_from_slice(&5u64.to_le_bytes());
        bytes.extend_from_slice(b"DS4LIVE1");
        assert_eq!(
            rotation_payload(&bytes).unwrap(),
            (&target[..], &b"draft"[..])
        );
        for short in [
            &bytes[..0],
            &bytes[..8],
            &bytes[1..],
            &bytes[..bytes.len() - 1],
        ] {
            assert!(rotation_payload(short).is_err());
        }
        let len = bytes.len();
        bytes[len - 16..len - 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(rotation_payload(&bytes).is_err());
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to both inference GPUs"]
    fn cuda_live_session_rotates_before_prefill_mid_prompt_and_during_sampled_decode() -> Result<()>
    {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask both inference GPUs"
        );
        let pack = std::path::PathBuf::from(std::env::var("DS4_TEST_PACK")?);
        let tok = Tokenizer::from_file(pack.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut plain = tok.clone();
        plain.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        let req = serde_json::json!({
            "messages":[{"role":"user","content": format!(
                "Transcript {}. Print the numbers from one through one hundred, one number per line. Start immediately with one.", rand::random::<u64>()
            )}],"reasoning_effort":"none"});
        let (chat, _, _) = crate::chat::render(&req)?;
        let ids = chat.encode(&tok, &plain)?;
        let mut model = Model::open(&pack, &tok, [0, 1], 256, 8)?;
        model.spec_k = 4; // The base pack also ships its three learned MTP stages.
        let mut reference = model.start_request(ids.clone(), 16, 0.5)?;
        let mut rotated = model.start_request(ids, 16, 0.5)?;
        reference.rng = Some(StdRng::seed_from_u64(194));
        rotated.rng = reference.rng.clone();
        let dir = tempfile::tempdir()?;
        let ram = SwapStore::new(32 << 20, 0, None)?;
        let disk = SwapStore::new(0, 32 << 20, Some(dir.path().to_owned()))?;
        let tiny = SwapStore::new(1, 0, None)?;
        let (mut before, mut mid, mut decoding) = (false, false, false);
        let mut ticks = 0;
        while !reference.finished {
            let pos = rotated.pos;
            before |= pos == 0;
            mid |= pos > 0 && rotated.is_prefill();
            decoding |= !rotated.is_prefill();
            let target = rotated.cache.encode_snapshot(pos)?;
            let draft = rotated
                .draft
                .as_ref()
                .unwrap()
                .rotation_snapshot(&mut model.rt, pos)?;
            if ticks < 2 {
                assert!(model.suspend(&mut rotated, &tiny).is_err());
                assert!(!rotated.is_suspended() && rotated.draft.is_some());
            }
            let store = if ticks % 2 == 0 { &ram } else { &disk };
            model.suspend(&mut rotated, store)?;
            assert!(rotated.is_suspended() && rotated.draft.is_none());
            assert_eq!(
                rotated.cache.encode_snapshot(0)?,
                AttentionState::default().encode_snapshot(0)?
            );
            if ticks == 0 {
                // A malformed drafter can fail after decoding a valid target.
                // Preserve the failed handle and keep all state unpublished.
                let saved = rotated.suspended.take().unwrap();
                let mut broken = saved.read()?.to_vec();
                broken[target.len()] ^= 1;
                rotated.suspended = Some(store.put(broken)?);
                assert!(model.resume(&mut rotated).is_err());
                assert!(rotated.is_suspended() && rotated.draft.is_none());
                assert_eq!(
                    rotated.cache.encode_snapshot(0)?,
                    AttentionState::default().encode_snapshot(0)?
                );
                rotated.suspended = Some(saved);
            }
            let mut both = model
                .step_batch(&mut [&mut rotated, &mut reference])
                .into_iter();
            assert!(both.next().unwrap().is_err());
            let expected = both.next().unwrap()?;
            assert_eq!(rotated.processed_tokens(), pos);
            model.resume(&mut rotated)?;
            assert_eq!(rotated.cache.encode_snapshot(pos)?, target);
            assert_eq!(
                rotated
                    .draft
                    .as_ref()
                    .unwrap()
                    .rotation_snapshot(&mut model.rt, pos)?,
                draft
            );
            assert_eq!(store.stats().ram_bytes + store.stats().disk_bytes, 0);
            let got = model.step_batch(&mut [&mut rotated]).pop().unwrap()?;
            assert_eq!(
                (got.tokens, got.eos, got.finished),
                (expected.tokens, expected.eos, expected.finished)
            );
            assert_eq!(
                (rotated.pos, rotated.emitted, &rotated.ids),
                (reference.pos, reference.emitted, &reference.ids)
            );
            ticks += 1;
        }
        assert!(before && mid && decoding);
        assert!(ram.stats().ram_reads > 0 && disk.stats().disk_reads > 0);
        Ok(())
    }

    #[test]
    fn independent_sessions_emit_verified_tokens_and_count_eos_once() {
        fn sequence(limit: usize) -> Session {
            let ids = vec![5, 6, 7];
            Session {
                prompt_len: ids.len(),
                pos: ids.len(),
                limit,
                temperature: 0.,
                rng: None,
                suspended: None,
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
