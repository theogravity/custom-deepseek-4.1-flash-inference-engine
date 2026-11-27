use crate::{
    attention::AttentionState,
    draft::{self, DraftInput, DraftState},
    engram::Engram,
    prefix::{PrefixCache, PREFIX_BLOCK_TOKENS},
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

pub(crate) fn bounded_replay() -> Result<bool> {
    match std::env::var("DS4_PREFILL_BOUNDED_REPLAY").as_deref() {
        Ok("1") => Ok(true),
        Ok("0") | Err(std::env::VarError::NotPresent) => Ok(false),
        _ => bail!("DS4_PREFILL_BOUNDED_REPLAY must be 0 or 1"),
    }
}

// Preserve the encoder batch at every boundary. A following short chunk can
// still read this one when fewer than 129 prompt rows remain.
fn skip_replay_chunk(pos: usize, rows: usize, horizon: usize) -> bool {
    rows > 128 && pos + rows + 128 < horizon
}

// Geometric boundaries bound the number of snapshots on long prefills while
// retaining branches well before the final complete chunk.
fn checkpoints(prompt: usize, chunk: usize) -> Vec<usize> {
    let last_compute = (prompt - 1) / chunk * chunk;
    let last_prefix = (prompt - 1) / PREFIX_BLOCK_TOKENS * PREFIX_BLOCK_TOKENS;
    let last = last_compute.min(last_prefix);
    let mut points = Vec::new();
    let mut boundary = chunk;
    while boundary < last {
        if boundary >= 256 && boundary.is_multiple_of(PREFIX_BLOCK_TOKENS) {
            points.push(boundary);
        }
        boundary *= 2;
    }
    if last_compute > 0
        && last_compute.is_multiple_of(PREFIX_BLOCK_TOKENS)
        && points.last() != Some(&last_compute)
    {
        points.push(last_compute);
    }
    if last_prefix > 0 && points.last() != Some(&last_prefix) {
        points.push(last_prefix);
    }
    points.sort_unstable();
    points.dedup();
    points
}

// At most eight RAM checkpoints for an idle replay, even at the full window.
fn idle_checkpoints(start: usize, last: usize) -> Vec<usize> {
    if start >= last {
        return Vec::new();
    }
    let stride = (last - start)
        .div_ceil(8)
        .max(512)
        .div_ceil(PREFIX_BLOCK_TOKENS)
        * PREFIX_BLOCK_TOKENS;
    ((start + stride..last).step_by(stride))
        .chain(std::iter::once(last))
        .collect()
}

fn prefill_rows(pos: usize, prompt: usize, chunk: usize, next_checkpoint: Option<usize>) -> usize {
    let rows = chunk.min(prompt - pos);
    next_checkpoint
        .filter(|&point| point > pos)
        .map_or(rows, |point| rows.min(point - pos))
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

// Only input IDs cross from a completed decode into this worker. Every KV
// value it publishes is recomputed by the ordinary prompt-prefill kernels.
enum IdleToolPrefill {
    Pending(Vec<u32>, std::time::Instant),
    Running(Box<Session>),
}

pub struct Model {
    idle_tool_prefill: Option<IdleToolPrefill>,
    idle_tool_limit: usize,
    prefix: PrefixCache,
    spec_k: usize,
    lookup: prompt_lookup::Config,
    batch_drafter: bool,
    rt: Runtime,
    engram: Arc<Engram>,
    overlap_prefill_engram: bool,
    prune_prefill_decoder: bool,
    bounded_replay: bool,
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
        let open_started = std::time::Instant::now();
        let tp2 = crate::moe::tp2_prefill_enabled()?;
        let preload =
            crate::moe::encoder_preload_layers(std::env::var("DS4_TP2_PRELOAD_LAYERS"), tp2)?;
        let max_chunk = if tp2 && crate::runtime::tp_dense() {
            16384
        } else {
            512
        };
        ensure!(
            (1..=max_chunk).contains(&chunk) && (1..=524288).contains(&max_ctx),
            "chunk must be 1..={max_chunk} and context must be 1..=524288"
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
        let idle_tool_limit = match std::env::var("DS4_IDLE_TOOL_PREFILL_MAX_TOKENS") {
            Ok(value) => value
                .parse::<usize>()
                .context("invalid DS4_IDLE_TOOL_PREFILL_MAX_TOKENS")?,
            Err(std::env::VarError::NotPresent) => 0,
            Err(e) => return Err(e.into()),
        };
        let model = Self {
            idle_tool_prefill: None,
            idle_tool_limit: idle_tool_limit.min(max_ctx),
            prefix: PrefixCache::new(pack, chunk, max_ctx)?,
            spec_k: spec_k()?,
            lookup: prompt_lookup::configured(tok)?,
            batch_drafter: match std::env::var("DS4_BATCH_DRAFT").as_deref() {
                Ok("0") => false,
                Ok("1") | Err(std::env::VarError::NotPresent) => true,
                _ => bail!("DS4_BATCH_DRAFT must be 0 or 1"),
            },
            rt: Runtime::open(pack, devices)?,
            engram: Arc::new(Engram::new(tok)?),
            bounded_replay: bounded_replay()?,
            prune_prefill_decoder: match std::env::var("DS4_PREFILL_DECODER_PRUNE").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("0") => false,
                Ok("1") => true,
                _ => anyhow::bail!("DS4_PREFILL_DECODER_PRUNE must be 0 or 1"),
            },
            overlap_prefill_engram: match std::env::var("DS4_OVERLAP_PREFILL_ENGRAM").as_deref() {
                Ok("1") => true,
                Ok("0") | Err(std::env::VarError::NotPresent) => false,
                _ => bail!("DS4_OVERLAP_PREFILL_ENGRAM must be 0 or 1"),
            },
            max_ctx,
            chunk,
        };
        #[cfg(feature = "recipe-prefill")]
        let mut model = model;
        #[cfg(feature = "recipe-prefill")]
        if preload > 0 {
            let started = std::time::Instant::now();
            let slots = crate::moe::preload_encoder(&mut model.rt, devices, preload)?;
            let bytes = slots.map(|n| n * crate::moe::RANK_SLOT_BYTES);
            eprintln!("tp2_preload requested_layers={preload} devices={devices:?} slots={slots:?} bytes={bytes:?} preload_seconds={:.3} model_open_seconds={:.3}",
                      started.elapsed().as_secs_f64(), open_started.elapsed().as_secs_f64());
        }
        #[cfg(not(feature = "recipe-prefill"))]
        let _ = (preload, open_started);
        Ok(model)
    }
    pub fn max_context(&self) -> usize {
        self.max_ctx
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
        let (pos, mut cache, mut timing) = (!points.is_empty())
            .then(|| cached(self.prefix.restore_at(&ids, &points, span), "restore").flatten())
            .flatten()
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
        let lookup = prompt_lookup::PromptLookup::new(&ids, self.lookup.clone());
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

    pub(crate) fn finished_tool(&mut self, mut session: Session) {
        if session.pos > self.idle_tool_limit
            || !self.prefix.can_warm(session.pos)
            || !session.finished
            || session.emitted >= session.limit
            || session.pos.saturating_sub(session.prompt_len) < 512
        {
            return;
        }
        session.ids.truncate(session.pos);
        self.idle_tool_prefill = Some(IdleToolPrefill::Pending(
            session.ids,
            std::time::Instant::now(),
        ));
    }

    pub(crate) fn idle_tool_delay(&self) -> Option<std::time::Duration> {
        let IdleToolPrefill::Pending(_, since) = self.idle_tool_prefill.as_ref()? else {
            return None;
        };
        // Most local reads and edits submit the next request immediately.
        std::time::Duration::from_millis(200).checked_sub(since.elapsed())
    }

    pub(crate) fn interrupt_tool_prefill(&mut self) {
        // Foreground admission must never perform a speculative snapshot write.
        self.idle_tool_prefill = None;
    }

    pub(crate) fn idle_tool_step(&mut self) -> Result<usize> {
        let Some(work) = self.idle_tool_prefill.take() else {
            return Ok(0);
        };
        let mut warm = match work {
            IdleToolPrefill::Pending(ids, _) => {
                // Reserve three output positions to enable the normal drafter.
                // No token is sampled: stop at the last full input checkpoint.
                let mut w = self.start_request(ids, 3, 0.)?;
                if !w.is_prefill() {
                    return Ok(0);
                }
                let last = (w.prompt_len - 1) / PREFIX_BLOCK_TOKENS * PREFIX_BLOCK_TOKENS;
                w.points = idle_checkpoints(w.pos, last);
                w.next_point = 0;
                Box::new(w)
            }
            IdleToolPrefill::Running(warm) => warm,
        };
        let before = warm.pos;
        let last = (warm.prompt_len - 1) / PREFIX_BLOCK_TOKENS * PREFIX_BLOCK_TOKENS;
        if before >= last {
            if before > warm.reused {
                self.store_prefill(&warm, true);
            }
            return Ok(0);
        }
        self.prefill_step(&mut warm, true)?;
        let rows = warm.pos - before;
        self.idle_tool_prefill = Some(IdleToolPrefill::Running(warm));
        Ok(rows)
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
        let single_client = sessions
            .iter()
            .filter(|s| !s.is_suspended() && !s.finished && !s.is_prefill())
            .take(2)
            .count()
            == 1;
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
                results[index] = Some(self.prefill_step(session, false));
            } else {
                let remaining = session.limit - session.emitted - 1;
                let width = self.spec_k.min(remaining);
                let learned =
                    width > 0 && session.draft.as_ref().is_some_and(|d| d.ready(session.pos));
                let tokens =
                    session
                        .lookup
                        .propose(&session.ids, remaining, learned, single_client);
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

    fn store_prefill(&mut self, session: &Session, ram_only: bool) {
        let pos = session.pos;
        let prompt = &session.ids[..session.prompt_len];
        let span = prompt.len().min(128);
        if cached(
            self.prefix
                .store(prompt, pos, span, &session.cache, ram_only),
            "store",
        )
        .is_some()
        {
            if let Some(draft) = session.draft.as_ref().filter(|d| d.ready(pos)) {
                let saved = draft
                    .snapshot(&mut self.rt, pos)
                    .and_then(|data| self.prefix.write_draft(prompt, pos, span, &data, ram_only));
                cached(saved, "DSpark store");
            }
        }
    }

    fn prefill_step(&mut self, session: &mut Session, interruptible: bool) -> Result<Step> {
        let pos = session.pos;
        let prompt = &session.ids[..session.prompt_len];
        if session.points.get(session.next_point) == Some(&pos) {
            self.store_prefill(session, interruptible);
            session.next_point += 1;
        }
        let horizon = session
            .points
            .get(session.next_point)
            .copied()
            .unwrap_or(prompt.len());
        let n = prefill_rows(
            pos,
            prompt.len(),
            self.chunk,
            session.points.get(session.next_point).copied(),
        );
        let last = pos + n == prompt.len();
        let replay = self.bounded_replay && self.chunk > 128;
        let skip_replay = replay && skip_replay_chunk(pos, n, horizon);
        let observe = session.draft.is_some()
            && !skip_replay
            && (interruptible || pos + n > horizon.saturating_sub(128));
        self.rt.target_prefill = true;
        let skip_decoder = skip_replay
            || (!replay
                && self.prune_prefill_decoder
                && !interruptible
                && !observe
                && !last
                && !session.cache.decoder_restore_pending()
                && crate::attention::can_defer_decoder(pos + n, horizon, self.chunk));
        let target = self.forward_prefill(
            ForwardItem {
                cache: &mut session.cache,
                ids: prompt,
                pos,
                rows: n,
                logit_rows: usize::from(last),
                capture_features: observe,
            },
            skip_decoder,
        );
        self.rt.target_prefill = false;
        session.cache.replay_fresh_window = false;
        let target = target?;
        if let Some(draft) = session.draft.as_mut().filter(|_| observe) {
            let features = target
                .features
                .as_ref()
                .context("missing DSpark prefill features")?;
            let retained = if replay { n.min(128) } else { n };
            draft.observe(&mut self.rt, features, pos + n - retained, retained)?;
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
    fn replay_preserves_encoder_chunks_and_short_final_dependencies() {
        for tail in 1..=128 {
            assert!(!skip_replay_chunk(0, 8192, 8192 + tail));
        }
        assert!(skip_replay_chunk(0, 8192, 8192 + 129));
        assert!(!skip_replay_chunk(8192, 8, 8200));
        assert!(!skip_replay_chunk(0, 128, 10000));
    }

    #[test]
    fn decoder_history_rounds_every_local_dependency_to_a_clean_chunk() {
        for chunk in [1, 48, 64, 127, 129, 512] {
            let end = 4096;
            let history =
                19 * 127_usize.next_multiple_of(chunk) + 128_usize.next_multiple_of(chunk);
            assert!(!crate::attention::can_defer_decoder(
                end,
                end + history - 1,
                chunk
            ));
            assert!(crate::attention::can_defer_decoder(
                end,
                end + history,
                chunk
            ));
            assert!(crate::attention::can_defer_decoder(
                end,
                end + history + 1,
                chunk
            ));
        }
    }

    #[test]
    #[ignore = "requires both inference GPUs, bounded replay and a private RAM prefix tier"]
    fn replay_prefix_matches_independent_physical_chunks() -> Result<()> {
        let pack = std::path::PathBuf::from(std::env::var("DS4_TEST_PACK")?);
        let tok = Tokenizer::from_file(pack.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<u32> = (0..2311).map(|i| 1000 + (i * 37 % 7800) as u32).collect();
        let mut model = Model::open(&pack, &tok, [0, 1], 4096, 512)?;
        assert!(model.bounded_replay && model.prefix.enabled());
        let mut idle = model.start_request(ids[..1701].to_vec(), 4, 0.)?;
        let mut cold = model.start_request(ids.clone(), 4, 0.)?;
        let mut short = model.start_request(ids[..1701].to_vec(), 4, 0.)?;
        assert_eq!((idle.pos, cold.pos, short.pos), (0, 0, 0));
        // 1536 is a final, non-geometric reusable checkpoint for the short
        // request. Process it independently through the idle entry point.
        idle.points = vec![1536];
        while idle.pos < 1536 {
            model.prefill_step(&mut idle, true)?;
        }
        while short.pos < 1536 {
            model.prefill_step(&mut short, false)?;
        }
        assert_eq!(
            idle.cache.encode_snapshot(1536)?,
            short.cache.encode_snapshot(1536)?
        );
        assert_eq!(
            idle.draft.as_ref().unwrap().snapshot(&mut model.rt, 1536)?,
            short
                .draft
                .as_ref()
                .unwrap()
                .snapshot(&mut model.rt, 1536)?
        );
        model.store_prefill(&idle, true);
        let mut branch = model.start_request(ids, 4, 0.)?;
        assert_eq!(branch.pos, 1536);
        let mut samples = Vec::new();
        for session in [&mut cold, &mut branch] {
            let mut out = Vec::new();
            while !session.finished {
                out.extend(model.step_batch(&mut [session]).pop().unwrap()?.tokens);
            }
            samples.push((out, session.cache.encode_snapshot(session.pos)?));
        }
        assert_eq!(samples[0].0, samples[1].0);
        ensure!(
            samples[0].1 == samples[1].1,
            "restored continuation diverged"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK, DS4_DRAFTER=code, independent prefix tiers and both inference GPUs"]
    fn deferred_decoder_rebuilds_every_target_and_draft_checkpoint() -> Result<()> {
        let pack = std::path::PathBuf::from(std::env::var("DS4_TEST_PACK")?);
        let tok = Tokenizer::from_file(pack.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let text: String = (0..1400)
            .map(|i| {
                format!(
                    "def checksum_{i}(x):\n    return (x * {} + {}) % 65519\n",
                    2 * i + 1,
                    i * 31
                )
            })
            .collect();
        let prompt = tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = &prompt.get_ids()[..9607];
        let extended = [ids, &prompt.get_ids()[9607..9615]].concat();
        for chunk in [64, 127] {
            let mut model = Model::open(&pack, &tok, [0, 1], 16384, chunk)?;
            assert!(model.prefix.enabled() && model.spec_k > 0);
            let mut control = model.start_request(ids.to_vec(), 12, 0.)?;
            let mut trial = model.start_request(ids.to_vec(), 12, 0.)?;
            let mut uncached = model.start_request(extended.clone(), 12, 0.)?;
            assert_eq!((control.pos, trial.pos, uncached.pos), (0, 0, 0));
            let ram = SwapStore::new(512 << 20, 0, None)?;
            let mut control_prefix = PrefixCache::new(&pack, chunk, 16384)?;
            let mut rotated = false;
            let mut compared = 0;
            while !control.finished {
                model.prune_prefill_decoder = false;
                std::mem::swap(&mut model.prefix, &mut control_prefix);
                let a = model.step_batch(&mut [&mut control]).pop().unwrap()?;
                std::mem::swap(&mut model.prefix, &mut control_prefix);
                model.prune_prefill_decoder = true;
                let b = model.step_batch(&mut [&mut trial]).pop().unwrap()?;
                assert_eq!((a.tokens, a.eos, a.finished), (b.tokens, b.eos, b.finished));
                assert_eq!(control.pos, trial.pos);
                let pos = trial.pos;
                if trial.is_prefill() && trial.cache.has_deferred_decoder_window() && !rotated {
                    model.suspend(&mut trial, &ram)?;
                    model.resume(&mut trial)?;
                    rotated = true;
                }
                if trial.points.contains(&pos) {
                    let expected = control.cache.encode_snapshot(pos)?;
                    let actual = trial.cache.encode_snapshot(pos)?;
                    ensure!(
                        actual == expected,
                        "target checkpoint mismatch at {pos}, chunk {chunk}"
                    );
                    let expected = control
                        .draft
                        .as_ref()
                        .unwrap()
                        .snapshot(&mut model.rt, pos)?;
                    let actual = trial.draft.as_ref().unwrap().snapshot(&mut model.rt, pos)?;
                    ensure!(
                        actual == expected,
                        "draft checkpoint mismatch at {pos}, chunk {chunk}"
                    );
                    compared += 1;
                }
            }
            assert!(rotated && compared == trial.points.len());
            let mut cached = model.start_request(extended.clone(), 12, 0.)?;
            assert_eq!(cached.pos, (ids.len() - 1) / chunk * chunk);
            let mut outputs = Vec::new();
            for (session, prune) in [(&mut uncached, false), (&mut cached, true)] {
                model.prune_prefill_decoder = prune;
                let mut out = Vec::new();
                while !session.finished {
                    out.extend(model.step_batch(&mut [session]).pop().unwrap()?.tokens);
                }
                outputs.push(out);
            }
            assert_eq!(outputs[0], outputs[1]);
        }
        Ok(())
    }

    #[test]
    fn idle_checkpoints_are_aligned_bounded_and_include_the_last_reusable_token_block() {
        for (start, last) in [(0, 524224), (2752, 6848), (0, 512), (64, 128), (128, 128)] {
            let points = idle_checkpoints(start, last);
            assert!(points.len() <= 8);
            assert!(points
                .iter()
                .all(|&p| p > start && p <= last && p % PREFIX_BLOCK_TOKENS == 0));
            assert!(points.windows(2).all(|p| p[0] < p[1]));
            assert_eq!(points.last().copied(), (start < last).then_some(last));
        }
    }

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
        let ids = chat.encode(&tok, &plain, None)?;
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
        assert_eq!(checkpoints(1050, 48), [384, 768, 1024]);
        assert!(checkpoints(33, 32).is_empty());
        assert!(checkpoints(32, 32).is_empty());
        assert_eq!(checkpoints(25_625, 8192), [8192, 16384, 24576, 25600]);
        assert_eq!(checkpoints(505_082, 8192).last(), Some(&505_024));
        assert_eq!(prefill_rows(24576, 25625, 8192, Some(25600)), 1024);
        assert_eq!(prefill_rows(25600, 25625, 8192, Some(25600)), 25);
        assert_eq!(prefill_rows(0, 25625, 8192, Some(8192)), 8192);
        // For each saved point and the final decode, every live row lies in
        // an observed chunk even for widths that do not divide 128.
        for chunk in [1, 3, 48, 64, 127, 255, 512, 8192] {
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
