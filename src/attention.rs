//! DeepSeek V4.1 Flash latent attention. The checkpoint shares compressed KV and
//! index keys at layers 2/8/14/20, while every backbone layer owns its window.
#[path = "attention_snapshot.rs"]
mod snapshot;

use crate::restore_timing::Probe;
use crate::runtime::{Dev, Runtime};
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use half::bf16;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::OnceLock;
use std::time::Instant;

const HIDDEN: usize = 5120;
const HEADS: usize = 64;
const DIM: usize = 512;
const ROPE: usize = 64;
const WINDOW: usize = 128;
const INDEX_HEADS: usize = 32;
const INDEX_DIM: usize = 128;
const TOP: usize = 512;
const CANDIDATE_LAYER: usize = 20;
const BLOCK: usize = 8;
const TOP_BLOCKS: usize = 2048;
const MAX_POS: usize = 1 << 20;
// SM120's fp8_ds_mla record: 448 E4M3 values, 64 BF16 rotary
// dimensions, seven exponent bytes and one byte of padding.
const KV_BYTES: usize = 584;
const NOPE: usize = DIM - ROPE;
const INDEX_BYTES: usize = INDEX_DIM + 4;

unsafe extern "C" {
    fn ds4_attn_zero(dev: i32, dst: *mut f32, len: usize) -> i32;
    #[cfg(feature = "flashinfer-prefill")]
    fn ds4_attn_flash_bytes(pos: i32, batch: i32, compressed_rows: i32, width: i32) -> usize;
    #[cfg(feature = "flashinfer-prefill")]
    fn ds4_attn_flash(
        dev: i32,
        scratch: *mut c_void,
        q: *const f32,
        prior: *const u16,
        current: *const u16,
        pos: i32,
        batch: i32,
        cache: *const c_void,
        compressed_rows: i32,
        ids: *const i32,
        width: i32,
        sink: *const f32,
        out: *mut f32,
    ) -> i32;
    fn ds4_attn_sparse_bytes(fast: i32, span: i32, width: i32, batch: i32) -> usize;
    fn ds4_attn_capacity(buf: *mut c_void) -> usize;
    fn ds4_attn_growth(buf: *mut c_void, bytes: usize) -> usize;
    fn ds4_attn_prepare_tail(dev: i32, cache: *mut c_void, old: usize, bytes: usize) -> i32;
    fn ds4_attn_pool_pairs(
        dev: i32,
        kv: *const f32,
        gates: *const f32,
        carry: *const f32,
        pairs: i32,
        out: *mut f32,
    ) -> i32;
    fn ds4_attn_normalize_raw(dev: i32, x: *mut f32, g: *const f32, n: i32) -> i32;
    fn ds4_attn_pack_compressed(
        dev: i32,
        values: *const f32,
        keys: *const f32,
        vdst: *mut u8,
        kdst: *mut u8,
        n: i32,
    ) -> i32;
    fn ds4_attn_scan_bytes(
        width: i32,
        rows: i32,
        score_mode: i32,
        out_blocks: i32,
        bytes: *mut usize,
    ) -> i32;
    #[cfg(test)]
    fn ds4_attn_index_bytes(width: i32, rows: i32, bytes: *mut usize) -> i32;
    fn ds4_attn_index_ids(
        dev: i32,
        out: *mut i32,
        width: i32,
        pos: i32,
        ratio: i32,
        rows: i32,
    ) -> i32;
    fn ds4_attn_index_select(
        dev: i32,
        work: *mut u8,
        cache: *const c_void,
        q: *const f32,
        weights: *const f32,
        blocks: *const i32,
        width: i32,
        pos: i32,
        ratio: i32,
        rows: i32,
        picks: *mut i32,
        out_blocks: *mut i32,
        score_mode: i32,
    ) -> i32;
    fn ds4_attn_buffer_new() -> *mut c_void;
    fn ds4_attn_buffer_free(device: i32, buf: *mut c_void);
    fn ds4_attn_upload(
        device: i32,
        buf: *mut c_void,
        src: *const u8,
        old: usize,
        len: usize,
    ) -> i32;
    fn ds4_attn_data(buf: *mut c_void) -> *mut c_void;
    fn ds4_attn_read(device: i32, src: *const c_void, dst: *mut c_void, bytes: usize) -> i32;
    fn ds4_attn_window(
        device: i32,
        buf: *mut c_void,
        start: i32,
        rows: i32,
        data: *mut c_void,
        kind: i32,
    ) -> i32;
    fn ds4_attn_normalize(
        device: i32,
        x: *mut f32,
        gamma: *const f32,
        width: i32,
        batch: i32,
    ) -> i32;
    fn ds4_attn_transform(
        device: i32,
        x: *mut f32,
        window: *mut u16,
        freqs: *const f32,
        heads: i32,
        cols: i32,
        group: i32,
        inverse: i32,
        batch: i32,
    ) -> i32;
    fn ds4_attn_sparse(
        device: i32,
        scratch: *mut c_void,
        q: *const f32,
        prior: *const u16,
        current: *const u16,
        pos: i32,
        tile: i32,
        span: i32,
        decode: i32,
        kv: *const c_void,
        selected: *const i32,
        width: i32,
        sink: *const f32,
        batch: i32,
        out: *mut f32,
    ) -> i32;
    fn ds4_attn_sparse_fast(
        device: i32,
        scratch: *mut c_void,
        q: *const f32,
        prior: *const u16,
        current: *const u16,
        pos: i32,
        tile: i32,
        span: i32,
        decode: i32,
        kv: *const c_void,
        selected: *const i32,
        width: i32,
        sink: *const f32,
        batch: i32,
        out: *mut f32,
    ) -> i32;
    fn ds4_attn_fp8_query_bytes(rows: i32) -> usize;
    fn ds4_attn_scores(
        device: i32,
        scratch: *mut c_void,
        keys: *const c_void,
        q: *const f32,
        weights: *const f32,
        ids: *const i32,
        width: i32,
        batch: i32,
        out: *mut f32,
        score_mode: i32,
    ) -> i32;
}

struct GpuBuffer {
    device: i32,
    ptr: NonNull<c_void>,
}

// Buffers are owned by a session, accessed exclusively, and explicitly select
// their CUDA device in every native entry point, including destruction.
unsafe impl Send for GpuBuffer {}
impl GpuBuffer {
    fn new(device: i32) -> Result<Self> {
        Ok(Self {
            device,
            ptr: NonNull::new(unsafe { ds4_attn_buffer_new() })
                .context("creating CUDA attention buffer")?,
        })
    }
    fn data<T>(&self) -> *mut T {
        unsafe { ds4_attn_data(self.ptr.as_ptr()).cast() }
    }

    fn reserve_growth(&self, cache: &mut crate::moe::MoeCache, bytes: usize) -> Result<()> {
        let growth = unsafe { ds4_attn_growth(self.ptr.as_ptr(), bytes) };
        if growth != 0 {
            cache.reserve(self.device, growth)?;
        }
        Ok(())
    }
    fn writable(&self, cache: &mut crate::moe::MoeCache, bytes: usize) -> Result<()> {
        let growth = unsafe { ds4_attn_growth(self.ptr.as_ptr(), bytes) };
        if growth != 0 {
            cache.reserve(self.device, growth)?;
            state_allocation(
                unsafe { ds4_attn_prepare_tail(self.device, self.ptr.as_ptr(), 0, growth) },
                "allocating session index workspace",
            )?;
        }
        Ok(())
    }
    fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        if !data.is_empty() {
            state_allocation(
                unsafe {
                    ds4_attn_upload(
                        self.device,
                        self.ptr.as_ptr(),
                        data.as_ptr().cast(),
                        0,
                        std::mem::size_of_val(data),
                    )
                },
                "uploading attention buffer",
            )?;
        }
        Ok(())
    }
}
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe { ds4_attn_buffer_free(self.device, self.ptr.as_ptr()) }
    }
}

#[derive(Debug)]
struct StateError(String);
impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for StateError {}
fn state_error(message: impl Into<String>) -> anyhow::Error {
    StateError(message.into()).into()
}
pub(crate) fn request_local(error: &anyhow::Error) -> bool {
    error.is::<StateError>()
}
macro_rules! require_state {
    ($condition:expr, $($message:tt)+) => {
        if !$condition {return Err(state_error(format!($($message)+)));}
    };
}
// Physical KV, window and index-pick allocations belong to one session. The
// runtime's activation pool, model projections and execution scratch are shared.
fn state_allocation(code: i32, op: &str) -> Result<()> {
    if code == 2 {
        return Err(state_error(format!("CUDA {op} failed with error {code}")));
    }
    cuda_ok(code, op)
}
fn cuda_ok(code: i32, op: &str) -> Result<()> {
    ensure!(code == 0, "CUDA {op} failed with error {code}");
    Ok(())
}

// Each tag describes only bytes present in the saved image. A CPU compressor
// can append fresh host rows before the same allocation and upload take place.
struct RestoredPart {
    probe: Probe,
    bytes: usize,
}
impl RestoredPart {
    fn attach(probe: &Probe, bytes: usize, count: &mut usize, total: &mut usize) -> Option<Self> {
        if bytes == 0 {
            return None;
        }
        *count += 1;
        *total += bytes;
        Some(Self {
            probe: probe.clone(),
            bytes,
        })
    }

    fn installed(self, device: i32, transferred: usize, start: Instant) {
        self.probe
            .installed(device, self.bytes, transferred, start.elapsed());
    }
}

/// GPU uploads can read a range of the immutable prefix image directly.
enum PackedRows {
    Owned(Vec<u8>),
    Prefix(Bytes),
}

impl Default for PackedRows {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl std::ops::Deref for PackedRows {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(rows) => rows,
            Self::Prefix(rows) => rows,
        }
    }
}

impl PackedRows {
    fn for_append(&mut self) -> &mut Vec<u8> {
        if let Self::Prefix(rows) = self {
            *self = Self::Owned(rows.to_vec());
        }
        match self {
            Self::Owned(rows) => rows,
            Self::Prefix(_) => unreachable!(),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            Self::Owned(rows) => rows.truncate(len),
            Self::Prefix(_) if len == 0 => *self = Self::default(),
            Self::Prefix(rows) => rows.truncate(len),
        }
    }
}

#[derive(Default)]
struct Rows {
    restored: Option<RestoredPart>,
    initial_bytes: usize,
    max_bytes: usize,
    writer: Option<i32>,
    packed: PackedRows,
    devices: HashMap<i32, (GpuBuffer, usize)>,
}
impl Rows {
    // Reserve the known prefill once, including at most one verification block
    // beyond it. Long generation still grows on demand up to the request limit.
    fn plan(&mut self, prompt: usize, limit: usize, ratio: usize, stride: usize) {
        self.initial_bytes = prompt.saturating_add(WINDOW).min(limit) / ratio * stride;
        self.max_bytes = limit / ratio * stride;
    }
    fn next_capacity(&self, current: usize, requested: usize) -> usize {
        if current >= requested {
            return current;
        }
        let geometric = current + current / 2;
        let cap = if self.max_bytes > 0 {
            geometric.min(self.max_bytes)
        } else {
            geometric
        };
        requested.max(self.initial_bytes).max(cap)
    }
    fn len(&self) -> usize {
        self.writer
            .map_or_else(|| self.packed.len(), |d| self.devices[&d].1)
    }

    fn append_snapshot(&self, out: &mut Vec<u8>) -> Result<()> {
        if let Some(dev) = self.writer {
            let (buffer, bytes) = &self.devices[&dev];
            let old = out.len();
            out.try_reserve(*bytes)?;
            out.resize(old + bytes, 0);
            if *bytes > 0 {
                cuda_ok(
                    unsafe {
                        ds4_attn_read(dev, buffer.data(), out[old..].as_mut_ptr().cast(), *bytes)
                    },
                    "reading compressed snapshot rows",
                )?;
            }
        } else {
            out.extend_from_slice(&self.packed);
        }
        Ok(())
    }

    fn prepare_tail(&mut self, rt: &mut Runtime, bytes: usize) -> Result<*mut u8> {
        let device = rt.device;
        let old = self.len();
        let handle = self.with_tail(device, Some(&mut rt.moe_cache), bytes)?;
        Ok(unsafe { ds4_attn_data(handle).cast::<u8>().add(old) })
    }

    fn finish_append(&mut self, dev: i32, bytes: usize) {
        self.devices.get_mut(&dev).unwrap().1 += bytes;
        self.writer = Some(dev);
        self.packed = PackedRows::default();
    }
    fn truncate(&mut self, bytes: usize) {
        self.packed.truncate(bytes);
        for (_, uploaded) in self.devices.values_mut() {
            *uploaded = (*uploaded).min(bytes);
        }
    }

    fn on(&mut self, dev: i32, cache: Option<&mut crate::moe::MoeCache>) -> Result<*mut c_void> {
        self.with_tail(dev, cache, 0)
    }

    fn with_tail(
        &mut self,
        dev: i32,
        cache: Option<&mut crate::moe::MoeCache>,
        tail: usize,
    ) -> Result<*mut c_void> {
        use std::collections::hash_map::Entry;
        if let Some(writer) = self.writer {
            require_state!(writer == dev, "compressed source device changed");
        }
        let started = self.restored.as_ref().map(|_| Instant::now());
        let requested = self.len() + tail;
        let current = self
            .devices
            .get(&dev)
            .map_or(0, |(b, _)| unsafe { ds4_attn_capacity(b.ptr.as_ptr()) });
        let capacity = self.next_capacity(current, requested);
        let (buf, old) = match self.devices.entry(dev) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert((GpuBuffer::new(dev)?, 0)),
        };
        // A restored host prefix and its first appended rows share one fresh
        // allocation. Reserve the complete new buffer while the old one lives.
        if capacity > current {
            if let Some(cache) = cache {
                cache.reserve(dev, capacity)?;
            }
            state_allocation(
                unsafe { ds4_attn_prepare_tail(dev, buf.ptr.as_ptr(), *old, capacity - *old) },
                "growing compressed GPU rows",
            )?;
        }
        if self.writer.is_none() && *old < self.packed.len() {
            let extra = &self.packed[*old..];
            cuda_ok(
                unsafe {
                    ds4_attn_upload(dev, buf.ptr.as_ptr(), extra.as_ptr(), *old, extra.len())
                },
                "uploading compressed attention rows",
            )?;
            *old = self.packed.len();
            if let Some(start) = started {
                self.restored
                    .take()
                    .unwrap()
                    .installed(dev, extra.len(), start);
            }
        }
        Ok(buf.ptr.as_ptr())
    }
}

#[derive(Default)]
struct Source {
    values: Rows,
    keys: Rows,
    pending_kv: Vec<f32>,
    pending_gate: Vec<f32>,
}

/// Layer 20 produces the last shared compressed KV source. Subsequent layers
/// can affect later rows only through their own 128-token sliding windows.
/// Keep a full 128-row DSpark feature history after the last dependency expires.
pub(crate) const FIRST_LOCAL_DECODER: usize = 21;
pub(crate) fn can_defer_decoder(end: usize, next_snapshot_or_output: usize, chunk: usize) -> bool {
    // One altered router assignment can change the split-K plan for other
    // tokens in that chunk. Require entirely clean chunks at every layer.
    let history = (40 - FIRST_LOCAL_DECODER) * (WINDOW - 1).next_multiple_of(chunk)
        + WINDOW.next_multiple_of(chunk);
    next_snapshot_or_output.saturating_sub(end) >= history
}

#[derive(Default)]
struct Window {
    restored: Option<RestoredPart>,
    rows: Vec<u16>, // Chronological history in a host snapshot.
    ring: Option<GpuBuffer>,
    next_pos: usize,
}

// Copies select a CUDA memcpy direction. The native helper splits wraparound.
const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;
const DEVICE_TO_DEVICE: i32 = 3;
impl Window {
    fn on(&mut self, dev: i32, cache: Option<&mut crate::moe::MoeCache>) -> Result<&GpuBuffer> {
        if self.ring.is_none() {
            let started = self.restored.as_ref().map(|_| Instant::now());
            let ring = GpuBuffer::new(dev)?;
            if let Some(cache) = cache {
                ring.reserve_growth(cache, 2 * WINDOW * DIM * size_of::<u16>())?;
            }
            let rows = self.next_pos.min(WINDOW - 1);
            require_state!(
                self.rows.len() == rows * DIM,
                "invalid restored attention window"
            );
            state_allocation(
                unsafe {
                    ds4_attn_window(
                        dev,
                        ring.ptr.as_ptr(),
                        (self.next_pos - rows) as i32,
                        rows as i32,
                        self.rows.as_mut_ptr().cast(),
                        HOST_TO_DEVICE,
                    )
                },
                "restoring attention window",
            )?;
            self.ring = Some(ring);
            if let Some(start) = started {
                self.restored.take().unwrap().installed(
                    dev,
                    self.rows.len() * size_of::<u16>(),
                    start,
                );
            }
            self.rows = Vec::new();
        }
        let ring = self.ring.as_ref().unwrap();
        require_state!(ring.device == dev, "attention window device changed");
        Ok(ring)
    }

    fn host_snapshot(&self) -> Result<Self> {
        let rows = if let Some(ring) = &self.ring {
            let n = self.next_pos.min(WINDOW - 1);
            let mut rows = vec![0u16; n * DIM];
            cuda_ok(
                unsafe {
                    ds4_attn_window(
                        ring.device,
                        ring.ptr.as_ptr(),
                        (self.next_pos - n) as i32,
                        n as i32,
                        rows.as_mut_ptr().cast(),
                        DEVICE_TO_HOST,
                    )
                },
                "saving attention window",
            )?;
            rows
        } else {
            self.rows.clone()
        };
        Ok(Self {
            restored: None,
            rows,
            ring: None,
            next_pos: self.next_pos,
        })
    }
}

// A verification block may commit any prefix. Each even-position row can
// become the unpaired half for the next decode at one of the ratio-two sources.
#[derive(Default)]
struct PendingPairs {
    kv: Vec<f32>,
    gates: Vec<f32>,
}

impl PendingPairs {
    fn capture(pos: usize, src: &Source, kv: &[f32], gates: &[f32]) -> Self {
        let mut rows = Self {
            kv: src.pending_kv.clone(),
            gates: src.pending_gate.clone(),
        };
        for (k, g) in kv
            .as_chunks::<DIM>()
            .0
            .iter()
            .zip(gates.as_chunks::<DIM>().0)
            .skip(pos % 2)
            .step_by(2)
        {
            rows.kv.extend_from_slice(k);
            rows.gates.extend_from_slice(g);
        }
        rows
    }
}

struct Verification {
    pos: usize,
    batch: usize,
    pending: HashMap<usize, PendingPairs>,
}

#[derive(Default)]
pub struct AttentionState {
    pub(crate) replay_fresh_window: bool,
    replay_window: Option<Dev>,
    prompt_len: usize,
    capacity_limit: usize,
    verify: Option<Verification>,
    windows: HashMap<usize, Window>,
    sources: HashMap<usize, Source>,
    selected: Vec<Vec<i32>>,
    picks: HashMap<i32, GpuBuffer>,
    device_blocks: Option<GpuBuffer>,
    width: usize,
    freqs: HashMap<(i32, bool), (usize, usize, GpuBuffer)>,
    blocks: Vec<Vec<usize>>,
}

impl AttentionState {
    /// Start a speculative target forward. The device ring retains the prior
    /// window and every tentative row until the caller commits an accepted prefix.
    pub fn begin_verify(&mut self, pos: usize, batch: usize) -> Result<()> {
        require_state!(
            self.verify.is_none(),
            "attention verification already active"
        );
        require_state!(
            (1..=WINDOW + 1).contains(&batch)
                && pos.checked_add(batch).is_some_and(|n| n <= MAX_POS),
            "invalid attention verification block"
        );
        require_state!(
            self.windows.values().all(|w| w.next_pos == pos),
            "discontinuous verification start"
        );
        self.verify = Some(Verification {
            pos,
            batch,
            pending: HashMap::new(),
        });
        Ok(())
    }

    /// Commit the first `accepted` inputs from the target forward, including
    /// its input anchor when present. Subsequent proposals overwrite the suffix.
    pub fn finish_verify(&mut self, accepted: usize) -> Result<()> {
        let mark = self
            .verify
            .as_ref()
            .ok_or_else(|| state_error("no active attention verification"))?;
        require_state!(
            accepted <= mark.batch,
            "accepted prefix exceeds verified block"
        );
        let verified = mark.pos + mark.batch;
        require_state!(
            self.windows.len() == 40 && self.windows.values().all(|w| w.next_pos == verified),
            "incomplete verification forward"
        );
        if accepted < mark.batch {
            let next = mark.pos + accepted;
            for (&owner, src) in &mut self.sources {
                let (_, ratio) =
                    source(owner).ok_or_else(|| state_error("unknown compressed source"))?;
                if ratio == 2 && next % 2 == 1 {
                    let pairs = mark
                        .pending
                        .get(&owner)
                        .ok_or_else(|| state_error("missing verified compressor rows"))?;
                    let start = ((next - 1) / 2 - mark.pos / 2) * DIM;
                    src.pending_kv.clear();
                    src.pending_gate.clear();
                    src.pending_kv
                        .extend_from_slice(&pairs.kv[start..start + DIM]);
                    src.pending_gate
                        .extend_from_slice(&pairs.gates[start..start + DIM]);
                } else {
                    src.pending_kv.clear();
                    src.pending_gate.clear();
                }
                let rows = next / ratio;
                src.keys.truncate(rows * INDEX_BYTES);
                src.values.truncate(rows * KV_BYTES);
            }
            for w in self.windows.values_mut() {
                w.next_pos = next;
            }
            self.selected.clear();
            self.blocks.clear();
            self.width = 0;
        }
        self.verify = None;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn has_deferred_decoder_window(&self) -> bool {
        self.windows
            .get(&39)
            .is_some_and(|w| w.next_pos > 0 && w.ring.is_none())
    }

    pub(crate) fn decoder_restore_pending(&self) -> bool {
        self.windows
            .iter()
            .any(|(&layer, w)| layer >= FIRST_LOCAL_DECODER && w.restored.is_some())
    }

    pub(crate) fn advance_deferred_windows(&mut self, pos: usize, count: usize) -> Result<()> {
        require_state!(
            self.verify.is_none(),
            "deferred attention during verification"
        );
        for layer in FIRST_LOCAL_DECODER..40 {
            let w = self.windows.entry(layer).or_default();
            require_state!(w.next_pos == pos, "deferred window position changed");
            w.ring = None;
            w.restored = None;
            w.next_pos = pos + count;
            // A live request may rotate at this point. Its placeholder rows are
            // serializable, and expire before its next reusable prefix or output.
            w.rows.resize(w.next_pos.min(WINDOW - 1) * DIM, 0);
            w.rows.fill(0);
        }
        Ok(())
    }

    /// Set the original prompt length once, including when restoring a prefix.
    /// All prefill chunks then share the reference's window-column layout.
    #[cfg(test)]
    fn begin_prefill(&mut self, prompt_len: usize) {
        self.begin_prefill_with_limit(prompt_len, MAX_POS);
    }
    pub fn begin_prefill_with_limit(&mut self, prompt_len: usize, available_positions: usize) {
        self.prompt_len = prompt_len;
        self.capacity_limit = available_positions.clamp(prompt_len, MAX_POS);
    }

    fn window_layout(&self, pos: usize, batch: usize) -> (usize, bool) {
        if pos < self.prompt_len {
            (self.prompt_len.min(WINDOW), false)
        } else if pos == 0 {
            // A standalone first forward also supports single-chunk prefill.
            (batch.min(WINDOW), false)
        } else {
            (WINDOW, true)
        }
    }

    /// Branch at a completed backbone forward. Each resumed request uploads its
    /// own device rows lazily. Top-k selections and candidate blocks belong to
    /// the past chunk and are rebuilt by their source layers on the next call.
    #[cfg(test)]
    pub fn host_snapshot(&self) -> Result<Self> {
        ensure!(self.verify.is_none(), "cannot snapshot active verification");
        fn host_rows(rows: &Rows) -> Result<Rows> {
            let mut copy = Rows::default();
            rows.append_snapshot(copy.packed.for_append())?;
            Ok(copy)
        }
        let sources = self
            .sources
            .iter()
            .map(|(&layer, source)| {
                Ok((
                    layer,
                    Source {
                        values: host_rows(&source.values)?,
                        keys: host_rows(&source.keys)?,
                        pending_kv: source.pending_kv.clone(),
                        pending_gate: source.pending_gate.clone(),
                    },
                ))
            })
            .collect::<Result<_>>()?;
        let windows = self
            .windows
            .iter()
            .map(|(&layer, w)| Ok((layer, w.host_snapshot()?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            sources,
            windows,
            ..Self::default()
        })
    }
}

fn b16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn normalize(x: &mut [f32], gamma: &[f32]) {
    for row in x.chunks_exact_mut(gamma.len()) {
        let sum: f32 = row.iter().map(|v| v * v).sum();
        let inv = (sum / row.len() as f32 + 1e-20).sqrt().recip();
        for (v, &g) in row.iter_mut().zip(gamma) {
            *v = b16((*v * inv) * g);
        }
    }
}

pub(crate) type Freq = [[f32; 2]; ROPE / 2];
pub(crate) fn rotary_freqs(pos: usize, compressed: bool) -> Freq {
    static BASE: OnceLock<[f32; ROPE / 2]> = OnceLock::new();
    static YARN: OnceLock<[f32; ROPE / 2]> = OnceLock::new();
    let cache = if compressed { &YARN } else { &BASE };
    let rates = cache.get_or_init(|| {
        let base: f64 = if compressed { 160000.0 } else { 10000.0 };
        let corrected =
            |r: f64| ROPE as f64 * (65536.0 / (r * std::f64::consts::TAU)).ln() / (2. * base.ln());
        let low = corrected(32.).floor().max(0.) as f32;
        let high = corrected(1.).ceil().min((ROPE - 1) as f64) as f32;
        std::array::from_fn(|i| {
            let raw = (base as f32).powf((2 * i) as f32 / ROPE as f32).recip();
            if !compressed {
                return raw;
            }
            let smooth = 1. - ((i as f32 - low) / (high - low).max(1e-3)).clamp(0., 1.);
            raw / 16. * (1. - smooth) + raw * smooth
        })
    });
    std::array::from_fn(|i| {
        let (s, c) = (pos as f32 * rates[i]).sin_cos();
        [s, c]
    })
}

fn rotate(row: &mut [f32], freq: &Freq, inverse: bool) {
    let tail = row.len() - ROPE;
    for (pair, &[s, c]) in row[tail..].as_chunks_mut::<2>().0.iter_mut().zip(freq) {
        let (a, b) = (pair[0], pair[1]);
        let s = if inverse { -s } else { s };
        pair[0] = b16(a * c - b * s);
        pair[1] = b16(a * s + b * c);
    }
}

fn pow2(exp: i32) -> f32 {
    f32::from_bits(((exp + 127) as u32) << 23)
}

fn ceil_scale(x: f32) -> f32 {
    // Callers clamp x to a positive normal before scale rounding.
    f32::from_bits(x.to_bits().saturating_add(0x7f_ffff) & 0x7f80_0000)
}

fn e4m3_code(x: f32) -> u8 {
    let a = x.abs().min(448.);
    let mag = if a < 1. / 64. {
        (a * 512.).round_ties_even() as u8
    } else {
        let exp = ((a.to_bits() >> 23) & 255) as i32 - 127;
        let mant = (a * pow2(3 - exp)).round_ties_even() as i32;
        ((exp + 6) * 8 + mant) as u8
    };
    mag | ((x.to_bits() >> 24) as u8 & 128)
}

#[cfg(test)]
fn e4m3(code: u8) -> f32 {
    static VALUES: OnceLock<[f32; 256]> = OnceLock::new();
    VALUES.get_or_init(|| {
        std::array::from_fn(|i| {
            let exp = (i as i32 >> 3) & 15;
            let m = (i & 7) as f32;
            let a = if exp == 0 {
                m / 512.
            } else {
                (1. + m / 8.) * pow2(exp - 7)
            };
            if i & 128 != 0 {
                -a
            } else {
                a
            }
        })
    })[code as usize]
}

fn fp8_scale(group: &[f32]) -> f32 {
    let peak = group.iter().fold(1e-4f32, |m, x| m.max(x.abs()));
    ceil_scale(peak * (1. / 448.))
}

#[cfg(test)]
fn fp8_qdq(row: &mut [f32], block: usize) {
    for group in row.chunks_exact_mut(block) {
        let scale = fp8_scale(group);
        for x in group {
            *x = e4m3(e4m3_code(*x / scale)) * scale;
        }
    }
}

fn pack_values(rows: &[f32], dst: &mut Vec<u8>) {
    dst.reserve(rows.len() / DIM * KV_BYTES);
    for row in rows.as_chunks::<DIM>().0 {
        let mut scales = [0u8; 8];
        for (j, g) in row[..NOPE].as_chunks::<64>().0.iter().enumerate() {
            let scale = fp8_scale(g);
            scales[j] = (scale.to_bits() >> 23) as u8;
            dst.extend(g.iter().map(|&x| e4m3_code(x / scale)));
        }
        for &v in &row[NOPE..] {
            dst.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        dst.extend_from_slice(&scales);
    }
}

fn pack_index_keys(rows: &[f32], dst: &mut Vec<u8>) {
    dst.reserve(rows.len() / INDEX_DIM * INDEX_BYTES);
    for row in rows.as_chunks::<INDEX_DIM>().0 {
        let scale = fp8_scale(row);
        dst.extend(row.iter().map(|&x| e4m3_code(x / scale)));
        dst.extend_from_slice(&scale.to_le_bytes());
    }
}

fn weights(rt: &Runtime, prefix: &str, name: &str, width: usize) -> Result<Vec<f32>> {
    let k = format!("{prefix}.{name}");
    let v = rt.weights.vector(&k)?;
    ensure!(v.len() == width, "{k} must contain {width} values");
    Ok(v)
}

fn linear(
    rt: &mut Runtime,
    prefix: &str,
    name: &str,
    x: &[f32],
    n: usize,
    out: usize,
) -> Result<Vec<f32>> {
    let k = format!("{prefix}.{name}.weight");
    let v = rt.linear(&k, x, n)?;
    ensure!(v.len() == n * out, "{k} output shape mismatch");
    Ok(v)
}

pub(crate) fn project(
    rt: &mut Runtime,
    prefix: &str,
    name: &str,
    x: &Dev,
    batch: usize,
    out: usize,
    groups: usize,
) -> Result<Dev> {
    let k = format!("{prefix}.{name}.weight");
    let result = rt.linear_dev(&k, x, batch, groups)?;
    ensure!(result.len() == batch * out, "{k} output shape mismatch");
    Ok(result)
}

// A frame of angles is shared by every layer on a device with the same RoPE base.
fn device_freqs(
    rt: &mut Runtime,
    state: &mut AttentionState,
    pos: usize,
    n: usize,
    compressed: bool,
) -> Result<*const f32> {
    let key = (rt.device, compressed);
    use std::collections::hash_map::Entry;
    let (p, len, freqs) = match state.freqs.entry(key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => e.insert((0, 0, GpuBuffer::new(rt.device)?)),
    };
    if (*p, *len) != (pos, n) {
        let host: Vec<_> = (pos..pos + n)
            .map(|i| rotary_freqs(i, compressed))
            .collect();
        freqs.reserve_growth(&mut rt.moe_cache, std::mem::size_of_val(host.as_slice()))?;
        freqs.write(&host)?;
        (*p, *len) = (pos, n);
    }
    Ok(freqs.data())
}

fn pool_pairs(
    kv: &[f32],
    gates: &[f32],
    pending_kv: &mut Vec<f32>,
    pending_gate: &mut Vec<f32>,
) -> Vec<f32> {
    let mut pooled = Vec::with_capacity((kv.len() + pending_kv.len()) / (2 * DIM) * DIM);
    for (k, g) in kv
        .as_chunks::<DIM>()
        .0
        .iter()
        .zip(gates.as_chunks::<DIM>().0)
    {
        if pending_kv.is_empty() {
            pending_kv.extend_from_slice(k);
            pending_gate.extend_from_slice(g);
        } else {
            for j in 0..DIM {
                let m = pending_gate[j].max(g[j]);
                let a = (pending_gate[j] - m).exp();
                let b = (g[j] - m).exp();
                pooled.push((pending_kv[j] * a + k[j] * b) / (a + b));
            }
            pending_kv.clear();
            pending_gate.clear();
        }
    }
    pooled
}

pub(crate) fn prefill_gpu_pool_pairs() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("DS4_PREFILL_GPU_POOL_PAIRS").is_some_and(|v| v == "1"))
}

fn pair_span(src: &Source, pos: usize, batch: usize) -> Result<(usize, bool)> {
    let carry = pos % 2;
    require_state!(
        batch > 0 && src.pending_kv.len() == carry * DIM && src.pending_gate.len() == carry * DIM,
        "inconsistent pending compressor row at {pos}"
    );
    Ok(((carry + batch) / 2, (carry + batch) % 2 != 0))
}

fn device_pool_pairs(
    rt: &mut Runtime,
    src: &Source,
    kv: &Dev,
    gates: &Dev,
    pos: usize,
    batch: usize,
) -> Result<(Option<Dev>, PendingPairs)> {
    let (pairs, has_tail) = pair_span(src, pos, batch)?;
    let pooled = if pairs != 0 {
        let carry = if src.pending_kv.is_empty() {
            None
        } else {
            Some(rt.to_device(&[src.pending_kv.as_slice(), src.pending_gate.as_slice()].concat())?)
        };
        let mut pooled = rt.buffer(pairs * DIM)?;
        cuda_ok(
            unsafe {
                ds4_attn_pool_pairs(
                    rt.device,
                    kv.as_ptr(),
                    gates.as_ptr(),
                    carry.as_ref().map_or(ptr::null(), Dev::as_ptr),
                    pairs as i32,
                    pooled.as_mut_ptr(),
                )
            },
            "pooling resident compressor pairs",
        )?;
        Some(pooled)
    } else {
        None
    };
    let pending = if has_tail {
        let last = (batch - 1) * DIM;
        PendingPairs {
            kv: rt.to_host_range(kv, last, DIM)?,
            gates: rt.to_host_range(gates, last, DIM)?,
        }
    } else {
        PendingPairs::default()
    };
    Ok((pooled, pending))
}

#[allow(clippy::too_many_arguments)]
fn append_source(
    rt: &mut Runtime,
    src: &mut Source,
    prefix: &str,
    x: &Dev,
    pos: usize,
    batch: usize,
    ratio: usize,
    owner: usize,
    verify: Option<&mut Verification>,
    resident: bool,
) -> Result<()> {
    require_state!(
        src.keys.len() == pos / ratio * INDEX_BYTES && src.values.len() == pos / ratio * KV_BYTES,
        "discontinuous compressed cache at {pos}"
    );
    let kv = project(rt, prefix, "compressor.wkv", x, batch, DIM, 1)?;
    if ratio == 1 && resident {
        return append_resident(rt, src, prefix, kv, pos, batch, ratio);
    }
    let mut latent = if ratio == 1 {
        rt.to_host(&kv)?
    } else {
        let gates = project(rt, prefix, "compressor.wgate", x, batch, DIM, 1)?;
        if ratio == 2
            && resident
            && rt.target_prefill
            && !rt.verification
            && verify.is_none()
            && rt.gpu_pair_pool
        {
            let (pooled, pending) = device_pool_pairs(rt, src, &kv, &gates, pos, batch)?;
            drop((gates, kv));
            if let Some(latent) = pooled {
                let n = latent.len() / DIM;
                append_resident(rt, src, prefix, latent, pos, n, ratio)?;
            }
            // A snapshot or verification can reuse the original projected
            // orphan. Publish it only after the complete pair append succeeds.
            src.pending_kv = pending.kv;
            src.pending_gate = pending.gates;
            return Ok(());
        }
        let kv = rt.to_host(&kv)?;
        let gates = rt.to_host(&gates)?;
        if let Some(mark) = verify {
            mark.pending
                .insert(owner, PendingPairs::capture(pos, src, &kv, &gates));
        }
        pool_pairs(&kv, &gates, &mut src.pending_kv, &mut src.pending_gate)
    };
    if latent.is_empty() {
        return Ok(());
    }
    if resident {
        let n = latent.len() / DIM;
        let latent = rt.to_device(&latent)?;
        return append_resident(rt, src, prefix, latent, pos, n, ratio);
    }
    normalize(
        &mut latent,
        &weights(rt, prefix, "compressor.norm.weight", DIM)?,
    );
    let n = latent.len() / DIM;
    let mut keys = linear(rt, prefix, "indexer.wk", &latent, n, INDEX_DIM)?;
    keys.iter_mut().for_each(|v| *v = b16(*v));
    normalize(
        &mut keys,
        &weights(rt, prefix, "indexer.k_norm.weight", INDEX_DIM)?,
    );
    for (i, (key, value)) in keys
        .as_chunks_mut::<INDEX_DIM>()
        .0
        .iter_mut()
        .zip(latent.as_chunks_mut::<DIM>().0)
        .enumerate()
    {
        let freq = rotary_freqs((pos / ratio + i) * ratio, true);
        rotate(key, &freq, false);
        rotate(value, &freq, false);
    }
    pack_index_keys(&keys, src.keys.packed.for_append());
    pack_values(&latent, src.values.packed.for_append());
    Ok(())
}

fn gpu_compress() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS4_GPU_COMPRESS").is_some_and(|v| v == "1"))
}

fn append_resident(
    rt: &mut Runtime,
    src: &mut Source,
    prefix: &str,
    mut latent: Dev,
    pos: usize,
    n: usize,
    ratio: usize,
) -> Result<()> {
    let g = rt.vector_ptr(&format!("{prefix}.compressor.norm.weight"), DIM)?;
    cuda_ok(
        unsafe { ds4_attn_normalize_raw(rt.device, latent.as_mut_ptr(), g, n as i32) },
        "normalizing compressed values",
    )?;
    let mut keys = project(rt, prefix, "indexer.wk", &latent, n, INDEX_DIM, 1)?;
    let g = rt.vector_ptr(&format!("{prefix}.indexer.k_norm.weight"), INDEX_DIM)?;
    cuda_ok(
        unsafe { ds4_attn_normalize(rt.device, keys.as_mut_ptr(), g, INDEX_DIM as i32, n as i32) },
        "normalizing compressed index keys",
    )?;
    let host: Vec<_> = (0..n)
        .map(|i| rotary_freqs((pos / ratio + i) * ratio, true))
        .collect();
    let freqs = rt.to_device(host.as_flattened().as_flattened())?;
    for (x, width) in [(&mut keys, INDEX_DIM), (&mut latent, DIM)] {
        cuda_ok(
            unsafe {
                ds4_attn_transform(
                    rt.device,
                    x.as_mut_ptr(),
                    ptr::null_mut(),
                    freqs.as_ptr(),
                    1,
                    width as i32,
                    0,
                    0,
                    n as i32,
                )
            },
            "rotating compressed source rows",
        )?;
    }
    let values = src.values.prepare_tail(rt, n * KV_BYTES)?;
    let target = src.keys.prepare_tail(rt, n * INDEX_BYTES)?;
    cuda_ok(
        unsafe {
            ds4_attn_pack_compressed(
                rt.device,
                latent.as_ptr(),
                keys.as_ptr(),
                values,
                target,
                n as i32,
            )
        },
        "packing compressed source rows",
    )?;
    src.values.finish_append(rt.device, n * KV_BYTES);
    src.keys.finish_append(rt.device, n * INDEX_BYTES);
    Ok(())
}

// Rank only when the visible prefix exceeds the budget. Keep position order for
// attention's accumulation, with earlier positions first on equal scores.
fn best(scores: &[f32], limit: usize) -> Vec<usize> {
    let mut ids: Vec<_> = (0..scores.len()).collect();
    if ids.len() > limit {
        ids.select_nth_unstable_by(limit, |&a, &b| {
            scores[b].total_cmp(&scores[a]).then(a.cmp(&b))
        });
        ids.truncate(limit);
        ids.sort_unstable();
    }
    ids
}

fn candidate_blocks(scores: &[f32]) -> Vec<usize> {
    if scores.len() <= TOP_BLOCKS * BLOCK {
        return Vec::new();
    }
    let mut peaks: Vec<_> = scores
        .chunks(BLOCK)
        .map(|g| g.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    *peaks.last_mut().unwrap() = f32::INFINITY;
    best(&peaks, TOP_BLOCKS)
}

fn source(layer: usize) -> Option<(usize, usize)> {
    match layer {
        2..=7 => Some((2, 2)),
        8..=13 => Some((8, 2)),
        14..=19 => Some((14, 2)),
        20..=39 => Some((20, 1)),
        _ => None,
    }
}

/// A runtime submits all attention operations to each card's default stream.
/// Scratch can be reused immediately by the next chunk or another session.
#[derive(Default)]
pub(crate) struct Workspace(HashMap<i32, GpuBuffer>);
fn scratch(rt: &mut Runtime, bytes: usize) -> Result<*mut c_void> {
    use std::collections::hash_map::Entry;
    let buf = match rt.attention_scratch.0.entry(rt.device) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => e.insert(GpuBuffer::new(rt.device)?),
    };
    buf.reserve_growth(&mut rt.moe_cache, bytes)?;
    Ok(buf.ptr.as_ptr())
}

#[allow(clippy::too_many_arguments)]
fn select(
    rt: &mut Runtime,
    state: &mut AttentionState,
    owner: usize,
    ratio: usize,
    prefix: &str,
    layer: usize,
    x: &Dev,
    qr: &Dev,
    pos: usize,
    n: usize,
    freqs: *const f32,
) -> Result<()> {
    let max_len = (pos + n) / ratio;
    state.selected.clear();
    let gpu = gpu_index();
    if gpu {
        state.width = max_len.min(TOP);
        if state.width != 0 {
            use std::collections::hash_map::Entry;
            let ids = match state.picks.entry(rt.device) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(GpuBuffer::new(rt.device)?),
            };
            ids.writable(&mut rt.moe_cache, n * state.width * size_of::<i32>())?;
        }
    }
    if layer == CANDIDATE_LAYER {
        if gpu && max_len > TOP_BLOCKS * BLOCK {
            if state.device_blocks.is_none() {
                state.device_blocks = Some(GpuBuffer::new(rt.device)?);
            }
            state
                .device_blocks
                .as_ref()
                .unwrap()
                .writable(&mut rt.moe_cache, n * TOP_BLOCKS * size_of::<i32>())?;
        } else {
            state.device_blocks = None;
        }
        state.blocks.clear();
    }
    if max_len <= TOP {
        if gpu {
            if state.width != 0 {
                let ids = &state.picks[&rt.device];
                cuda_ok(
                    unsafe {
                        ds4_attn_index_ids(
                            rt.device,
                            ids.data(),
                            state.width as i32,
                            pos as i32,
                            ratio as i32,
                            n as i32,
                        )
                    },
                    "building causal index rows",
                )?;
            }
            return Ok(());
        }
        state.selected = (0..n)
            .map(|t| (0..((pos + t + 1) / ratio) as i32).collect())
            .collect();
        if layer == CANDIDATE_LAYER {
            state.blocks.resize_with(n, Vec::new);
        }
        return Ok(());
    }
    let mut q = project(
        rt,
        prefix,
        "indexer.wq_b",
        qr,
        n,
        INDEX_HEADS * INDEX_DIM,
        1,
    )?;
    let w = project(rt, prefix, "indexer.weights_proj", x, n, INDEX_HEADS, 1)?;
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                q.as_mut_ptr(),
                ptr::null_mut(),
                freqs,
                INDEX_HEADS as i32,
                INDEX_DIM as i32,
                INDEX_DIM as i32,
                0,
                n as i32,
            )
        },
        "rotating and quantizing index queries",
    )?;
    let keys = state
        .sources
        .get_mut(&owner)
        .ok_or_else(|| state_error("missing index key source"))?
        .keys
        .on(rt.device, Some(&mut rt.moe_cache))?;
    if gpu {
        let picks = state
            .picks
            .get(&rt.device)
            .ok_or_else(|| state_error("missing GPU index output"))?;
        let blocks = state
            .device_blocks
            .as_mut()
            .map_or(ptr::null_mut(), |v| v.data::<i32>());
        return device_select(
            rt, layer, pos, n, ratio, max_len, keys, &q, &w, blocks, picks,
        );
    }
    // Tile host scores to keep long-context prefill scratch bounded. After layer
    // 20, only the 2048 winning eight-token blocks need scoring.
    let filtered_width = TOP_BLOCKS * BLOCK;
    require_state!(
        layer <= CANDIDATE_LAYER || state.blocks.len() == n,
        "missing causal index candidates"
    );
    let mut t = 0;
    while t < n {
        let filtering = layer > CANDIDATE_LAYER && !state.blocks[t].is_empty();
        let bound = if filtering { filtered_width } else { max_len };
        let mut rows = (2_097_152 / bound).max(1).min(n - t);
        if layer > CANDIDATE_LAYER {
            rows = (0..rows)
                .take_while(|&j| (!state.blocks[t + j].is_empty()) == filtering)
                .count();
        }
        let width = if filtering {
            filtered_width
        } else {
            (pos + t + rows) / ratio
        };
        let mut ids = Vec::new();
        if filtering {
            ids.reserve(rows * width);
            for (j, blocks) in state.blocks[t..t + rows].iter().enumerate() {
                let visible = (pos + t + j + 1) / ratio;
                for &block in blocks {
                    for id in block * BLOCK..(block + 1) * BLOCK {
                        ids.push(if id < visible { id as i32 } else { -1 });
                    }
                }
            }
        }
        let mode = index_score_mode(rt.target_prefill, width, rows, filtering);
        let mut bytes =
            (rows * width * size_of::<f32>()).next_multiple_of(256) * (1 + usize::from(filtering));
        if mode >= 3 {
            bytes += unsafe { ds4_attn_fp8_query_bytes(rows as i32) };
        }
        let work = scratch(rt, bytes)?;
        let mut scores = vec![0.; rows * width];
        cuda_ok(
            unsafe {
                ds4_attn_scores(
                    rt.device,
                    work,
                    keys,
                    q.as_ptr().add(t * INDEX_HEADS * INDEX_DIM),
                    w.as_ptr().add(t * INDEX_HEADS),
                    if filtering { ids.as_ptr() } else { ptr::null() },
                    width as i32,
                    rows as i32,
                    scores.as_mut_ptr(),
                    mode,
                )
            },
            "scoring compressed index keys",
        )?;
        for (j, row) in scores.chunks_exact(width).enumerate() {
            let visible = (pos + t + j + 1) / ratio;
            let row = if filtering { row } else { &row[..visible] };
            if layer == CANDIDATE_LAYER {
                state.blocks.push(candidate_blocks(row));
            }
            let selected = best(row, TOP)
                .into_iter()
                .map(|p| {
                    if filtering {
                        ids[j * width + p]
                    } else {
                        p as i32
                    }
                })
                .collect();
            state.selected.push(selected);
        }
        t += rows;
    }
    Ok(())
}

pub(crate) fn prefill_index_mma() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS4_PREFILL_INDEX_MMA").is_some_and(|v| v == "1"))
}

pub(crate) fn prefill_index_fp8() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS4_PREFILL_INDEX_FP8").is_some_and(|v| v == "1"))
}

pub(crate) fn prefill_batched_mqa() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS4_PREFILL_BATCHED_MQA").is_some_and(|v| v == "1"))
}

fn batched_mqa_tile(width: usize, rows: usize, filtered: bool) -> bool {
    // The single-query kernel wins small scans. Four-query tiles amortize query
    // setup and share keys on long, dense scans with at least 1M query/key pairs.
    !filtered && rows >= 4 && width >= 16_384 && width.saturating_mul(rows) >= 1_048_576
}

fn index_score_mode(target_prefill: bool, width: usize, rows: usize, filtered: bool) -> i32 {
    static PACKED: OnceLock<bool> = OnceLock::new();
    if target_prefill && prefill_batched_mqa() {
        if batched_mqa_tile(width, rows, filtered) {
            4
        } else {
            3
        }
    } else if target_prefill && prefill_index_fp8() {
        3
    } else if target_prefill && prefill_index_mma() {
        2
    } else {
        i32::from(
            *PACKED.get_or_init(|| std::env::var_os("DS4_PACKED_INDEX").is_none_or(|v| v != "0")),
        )
    }
}

fn gpu_index() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS4_GPU_INDEX").is_some_and(|v| v == "1"))
}

#[allow(clippy::too_many_arguments)]
fn device_select(
    rt: &mut Runtime,
    layer: usize,
    pos: usize,
    n: usize,
    ratio: usize,
    max_len: usize,
    keys: *const c_void,
    q: &Dev,
    w: &Dev,
    blocks: *mut i32,
    picks: &GpuBuffer,
) -> Result<()> {
    let block_span = TOP_BLOCKS * BLOCK;
    // Layers after 20 use its candidate blocks only when the individual row
    // exceeds the block budget. A tile may straddle that boundary in prefill.
    let start_filtered = ((block_span + 1) * ratio - 1).saturating_sub(pos).min(n);
    if n == 0 {
        return Ok(());
    }
    let mut tiles = Vec::new();
    let mut workspace_bytes = 0;
    let mut t = 0;
    while t < n {
        let filtered = layer > CANDIDATE_LAYER && t >= start_filtered;
        let end = if layer > CANDIDATE_LAYER && !filtered {
            start_filtered
        } else {
            n
        };
        let bound = if filtered { block_span } else { max_len };
        let rows = (2_097_152 / bound).max(1).min(end - t);
        let width = if filtered {
            block_span
        } else {
            (pos + t + rows) / ratio
        };
        let score_mode = index_score_mode(rt.target_prefill,width,rows,filtered);
        let mut bytes = 0;
        cuda_ok(
            unsafe {
                ds4_attn_scan_bytes(
                    width as i32,
                    rows as i32,
                    score_mode,
                    (layer == CANDIDATE_LAYER && !blocks.is_null()).into(),
                    &mut bytes,
                )
            },
            "sizing index sort scratch",
        )?;
        workspace_bytes = workspace_bytes.max(bytes);
        require_state!(
            !filtered || !blocks.is_null(),
            "missing GPU candidate blocks"
        );
        tiles.push((t, rows, width, filtered));
        t += rows;
    }
    // Every tile is submitted to this device's default stream. The workspace
    // belongs to this call until the last launch and is then recyclable by
    // the next operation on that stream.
    let mut work = rt.buffer(workspace_bytes.div_ceil(4))?;
    for (t, rows, width, filtered) in tiles {
        cuda_ok(
            unsafe {
                let b = if blocks.is_null() || !(filtered || layer == CANDIDATE_LAYER) {
                    blocks
                } else {
                    blocks.add(t * TOP_BLOCKS)
                };
                ds4_attn_index_select(
                    rt.device,
                    work.as_mut_ptr().cast(),
                    keys,
                    q.as_ptr().add(t * INDEX_HEADS * INDEX_DIM),
                    w.as_ptr().add(t * INDEX_HEADS),
                    if filtered { b } else { ptr::null() },
                    width as i32,
                    (pos + t) as i32,
                    ratio as i32,
                    rows as i32,
                    picks.data::<i32>().add(t * TOP),
                    if layer == CANDIDATE_LAYER {
                        b
                    } else {
                        ptr::null_mut()
                    },
                    score_mode,
                )
            },
            "ranking index keys on GPU",
        )?;
    }
    Ok(())
}

pub(crate) fn flash_prefill() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DS4_FLASHINFER_PREFILL").is_some_and(|v| v == "1"))
}

pub(crate) fn fast_attention() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| std::env::var_os("DS4_FAST_ATTN").is_some_and(|v| v == "1"))
}

/// Each span owns its causal state. The projections operate on concatenated
/// rows, keeping every sequence's rotary positions and KV selections local.
pub(crate) struct Span<'a> {
    pub state: &'a mut AttentionState,
    pub pos: usize,
    pub rows: usize,
    pub error: &'a mut Option<anyhow::Error>,
}
/// The last KV source owns both compressed KV and a full sliding window.
pub(crate) fn write_last_source(
    rt: &mut Runtime,
    state: &mut AttentionState,
    pos: usize,
    batch: usize,
    x: &Dev,
    retain: usize,
) -> Result<()> {
    let owner = FIRST_LOCAL_DECODER - 1;
    let prefix = format!("layers.{owner}.attn");
    require_state!(
        state.verify.is_none() && state.windows.get(&owner).map_or(0, |w| w.next_pos) == pos,
        "discontinuous replay KV write"
    );
    let src = state.sources.entry(owner).or_default();
    if state.capacity_limit > 0 {
        src.values
            .plan(state.prompt_len, state.capacity_limit, 1, KV_BYTES);
        src.keys
            .plan(state.prompt_len, state.capacity_limit, 1, INDEX_BYTES);
    }
    append_source(
        rt,
        src,
        &prefix,
        x,
        pos,
        batch,
        1,
        owner,
        None,
        gpu_compress(),
    )?;
    let mut kv = project(rt, &prefix, "wkv", x, batch, DIM, 1)?;
    let gamma = rt.vector_ptr(&format!("{prefix}.kv_norm.weight"), DIM)?;
    cuda_ok(
        unsafe { ds4_attn_normalize(rt.device, kv.as_mut_ptr(), gamma, DIM as i32, batch as i32) },
        "normalizing replay window",
    )?;
    let kept = batch.min(2 * WINDOW);
    let start = pos + batch - kept;
    let freqs = device_freqs(rt, state, start, kept, true)?;
    let mut packed = rt.buffer(kept * DIM / 2)?;
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                kv.as_mut_ptr().add((batch - kept) * DIM),
                packed.as_mut_ptr().cast(),
                freqs,
                1,
                DIM as i32,
                64,
                0,
                kept as i32,
            )
        },
        "packing replay window",
    )?;
    let window = state.windows.entry(owner).or_default();
    let ring = window.on(rt.device, Some(&mut rt.moe_cache))?;
    cuda_ok(
        unsafe {
            ds4_attn_window(
                rt.device,
                ring.ptr.as_ptr(),
                start as i32,
                kept as i32,
                packed.as_mut_ptr().cast(),
                DEVICE_TO_DEVICE,
            )
        },
        "writing replay window",
    )?;
    let skipped = batch - retain;
    window.next_pos = pos + skipped;
    // A live rotation may have installed a timing marker for a discarded
    // window. Materialize it once so its recorded transfer can finish.
    for w in state
        .windows
        .iter_mut()
        .filter(|(l, w)| **l >= FIRST_LOCAL_DECODER && w.restored.is_some())
        .map(|(_, w)| w)
    {
        w.on(rt.device, Some(&mut rt.moe_cache))?;
    }
    state.advance_deferred_windows(pos, skipped)?;
    if retain != 0 {
        state.replay_window =
            Some(rt.slice(&packed, (kept - retain) * DIM / 2, retain * DIM / 2)?);
        state.replay_fresh_window = true;
    }
    Ok(())
}

pub fn forward_dev(
    rt: &mut Runtime,
    state: &mut AttentionState,
    layer: usize,
    pos: usize,
    x: &Dev,
    batch: usize,
) -> Result<Dev> {
    let mut error = None;
    let out = forward_group(
        rt,
        layer,
        x,
        &mut [Span {
            state,
            pos,
            rows: batch,
            error: &mut error,
        }],
    )?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(out)
}
pub(crate) fn forward_group(
    rt: &mut Runtime,
    layer: usize,
    x: &Dev,
    spans: &mut [Span<'_>],
) -> Result<Dev> {
    let batch: usize = spans.iter().map(|s| s.rows).sum();
    ensure!(
        layer < 40 && batch > 0 && x.len() == batch * HIDDEN,
        "invalid grouped attention shape"
    );
    let prefix = format!("layers.{layer}.attn");
    let mut qr = project(rt, &prefix, "wq_a", x, batch, 1280, 1)?;
    let gamma = rt.vector_ptr(&format!("{prefix}.q_norm.weight"), 1280)?;
    cuda_ok(
        unsafe { ds4_attn_normalize(rt.device, qr.as_mut_ptr(), gamma, 1280, batch as i32) },
        "normalizing attention query projection",
    )?;
    let mut q = project(rt, &prefix, "wq_b", &qr, batch, HEADS * DIM, 1)?;
    let mut kv = if spans.len() == 1 && spans[0].state.replay_window.is_some() {
        None
    } else {
        let mut kv = project(rt, &prefix, "wkv", x, batch, DIM, 1)?;
        let gamma = rt.vector_ptr(&format!("{prefix}.kv_norm.weight"), DIM)?;
        cuda_ok(
            unsafe {
                ds4_attn_normalize(rt.device, kv.as_mut_ptr(), gamma, DIM as i32, batch as i32)
            },
            "normalizing window keys",
        )?;
        Some(kv)
    };
    let mut o = rt.buffer(q.len())?;
    let mut start = 0;
    for s in spans {
        let count = s.rows * HEADS * DIM;
        let out_ptr = unsafe { o.as_mut_ptr().add(start * HEADS * DIM) };
        if s.error.is_none() {
            let result = apply_span(
                rt,
                s.state,
                layer,
                &prefix,
                s.pos,
                s.rows,
                x,
                &qr,
                start,
                unsafe { q.as_mut_ptr().add(start * HEADS * DIM) },
                kv.as_mut().map_or(ptr::null_mut(), |v| unsafe {
                    v.as_mut_ptr().add(start * DIM)
                }),
                out_ptr,
            );
            if let Err(err) = result {
                if !request_local(&err) {
                    return Err(err);
                }
                *s.error = Some(err);
                *s.state = AttentionState::default();
            }
        }
        if s.error.is_some() {
            cuda_ok(
                unsafe { ds4_attn_zero(rt.device, out_ptr, count) },
                "clearing failed attention rows",
            )?;
        }
        start += s.rows;
    }
    drop((q, kv, qr));
    let down = project(rt, &prefix, "wo_a", &o, batch, 8 * 1024, 8)?;
    drop(o);
    project(rt, &prefix, "wo_b", &down, batch, HIDDEN, 1)
}
#[allow(clippy::too_many_arguments)]
fn apply_span(
    rt: &mut Runtime,
    state: &mut AttentionState,
    layer: usize,
    prefix: &str,
    pos: usize,
    batch: usize,
    x: &Dev,
    qr: &Dev,
    start: usize,
    q_ptr: *mut f32,
    kv_ptr: *mut f32,
    out_ptr: *mut f32,
) -> Result<()> {
    require_state!(
        batch > 0 && pos.checked_add(batch).is_some_and(|v| v <= MAX_POS),
        "invalid attention position {pos} and span {batch}"
    );
    if let Some(mark) = &state.verify {
        require_state!(
            mark.pos == pos && mark.batch == batch,
            "attention verification block changed"
        );
    }
    let prior = state.windows.get(&layer);
    require_state!(
        prior.map_or(0, |w| w.next_pos) == pos,
        "non-contiguous attention window at layer {layer}, position {pos}"
    );
    let group = source(layer);
    let freqs = device_freqs(rt, state, pos, batch, group.is_some())?;
    // Only the four compressors and the learned indexers read input rows
    // after the shared projections. Slicing those smaller operands avoids
    // copying the 32K query/output features per token between sessions.
    let owner = group.is_some_and(|(owner, _)| owner == layer);
    let index = group.is_some_and(|(s, ratio)| {
        (layer == s || [24, 28, 32, 36].contains(&layer)) && (pos + batch) / ratio > TOP
    });
    let x_slice;
    let local_x = if (owner || index) && x.len() != batch * HIDDEN {
        x_slice = rt.slice(x, start * HIDDEN, batch * HIDDEN)?;
        &x_slice
    } else {
        x
    };
    let qr_slice;
    let local_qr = if index && qr.len() != batch * 1280 {
        qr_slice = rt.slice(qr, start * 1280, batch * 1280)?;
        &qr_slice
    } else {
        qr
    };
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                q_ptr,
                ptr::null_mut(),
                freqs,
                HEADS as i32,
                DIM as i32,
                0,
                0,
                batch as i32,
            )
        },
        "rotating attention queries",
    )?;
    // Storage is packed BF16, with two values in each pooled four-byte element.
    let prepared = state.replay_window.take();
    let source_written = prepared.is_some();
    let mut current = if let Some(window) = prepared {
        require_state!(
            layer + 1 == FIRST_LOCAL_DECODER && window.len() == batch * DIM / 2,
            "invalid replay window"
        );
        window
    } else {
        let mut current = rt.buffer(batch * DIM / 2)?;
        cuda_ok(
            unsafe {
                ds4_attn_transform(
                    rt.device,
                    kv_ptr,
                    current.as_mut_ptr().cast(),
                    freqs,
                    1,
                    DIM as i32,
                    64,
                    0,
                    batch as i32,
                )
            },
            "packing attention window",
        )?;

        current
    };
    let mut width = 0;
    let mut ids: *const i32 = ptr::null();
    let mut values = ptr::null_mut();
    if let Some((owner, ratio)) = group {
        if layer == owner && !source_written {
            let src = state.sources.entry(owner).or_default();
            if state.capacity_limit > 0 {
                src.values
                    .plan(state.prompt_len, state.capacity_limit, ratio, KV_BYTES);
                src.keys
                    .plan(state.prompt_len, state.capacity_limit, ratio, INDEX_BYTES);
            }
            append_source(
                rt,
                src,
                prefix,
                local_x,
                pos,
                batch,
                ratio,
                owner,
                state.verify.as_mut(),
                gpu_compress(),
            )?;
        }
        if layer == owner || [24, 28, 32, 36].contains(&layer) {
            select(
                rt, state, owner, ratio, prefix, layer, local_x, local_qr, pos, batch, freqs,
            )?;
            if !gpu_index() {
                require_state!(
                    state.selected.len() == batch,
                    "missing shared top-k index rows"
                );
                state.width = state.selected.iter().map(Vec::len).max().unwrap_or(0);
                if state.width > 0 {
                    let mut flat = Vec::with_capacity(batch * state.width);
                    for row in &state.selected {
                        flat.extend_from_slice(row);
                        flat.resize(flat.len() + state.width - row.len(), -1);
                    }
                    use std::collections::hash_map::Entry;
                    let picks = match state.picks.entry(rt.device) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => e.insert(GpuBuffer::new(rt.device)?),
                    };
                    picks.reserve_growth(&mut rt.moe_cache, flat.len() * size_of::<i32>())?;
                    picks.write(&flat)?;
                }
            }
        }
        width = state.width;
        if width != 0 {
            ids = state
                .picks
                .get(&rt.device)
                .ok_or_else(|| state_error("missing selected attention indices"))?
                .data();
            values = state
                .sources
                .get_mut(&owner)
                .ok_or_else(|| state_error("missing compressed attention source"))?
                .values
                .on(rt.device, Some(&mut rt.moe_cache))?;
        }
    }

    let sink = rt.vector_ptr(&format!("{prefix}.attn_sink"), HEADS)?;
    // A retained replay tail starts its own local decoder window. Absolute
    // positions still drive rotary encoding, compressed selection and ring writes.
    let window_pos = if state.replay_fresh_window && layer >= FIRST_LOCAL_DECODER {
        0
    } else {
        pos
    };
    let (span, decode) = state.window_layout(pos, batch);
    let window = state.windows.entry(layer).or_default();
    let ring = window.on(rt.device, Some(&mut rt.moe_cache))?;

    // Bound temporary expanded compressed values to 16 MiB per call.
    let fast = fast_attention();
    #[cfg(feature = "flashinfer-prefill")]
    let used_flash = rt.target_prefill && flash_prefill();
    #[cfg(not(feature = "flashinfer-prefill"))]
    let used_flash = false;
    #[cfg(feature = "flashinfer-prefill")]
    if used_flash {
        // A replay may have materialized more source rows than this causal
        // span sees. Only pack the visible prefix, once for all query tiles.
        let compressed = group
            .filter(|_| width != 0)
            .map_or(0, |(_, ratio)| (pos + batch) / ratio);
        if let Some((owner, _)) = group.filter(|_| width != 0) {
            require_state!(
                state.sources[&owner].values.len() >= compressed * KV_BYTES,
                "short compressed MLA source"
            );
        }
        let needed = unsafe {
            ds4_attn_flash_bytes(
                window_pos as i32,
                batch as i32,
                compressed as i32,
                width as i32,
            )
        };
        let work = scratch(rt, needed)?;
        cuda_ok(
            unsafe {
                ds4_attn_flash(
                    rt.device,
                    work,
                    q_ptr,
                    ring.data(),
                    current.as_ptr().cast(),
                    window_pos as i32,
                    batch as i32,
                    values,
                    compressed as i32,
                    ids,
                    width as i32,
                    sink,
                    out_ptr,
                )
            },
            "FlashInfer sparse MLA prompt attention",
        )?;
    }
    let mut t = if used_flash { batch } else { 0 };
    while t < batch {
        // Short contexts fit the original single CTA per head. Choose by
        // absolute position so a verification block uses the same kernel as
        // decoding each of its tokens separately.
        let use_tensor = fast && pos + t >= 64;
        let sparse = if use_tensor {
            ds4_attn_sparse_fast
        } else {
            ds4_attn_sparse
        };
        let bytes_per_token = if use_tensor {
            let cols = (span + width).next_multiple_of(64);
            cols * (DIM * 2 + HEADS * 6) + HEADS * (cols / 64 + 1) * 4
        } else {
            width.max(1) * DIM * 2
        };
        let mut n = (16 * 1024 * 1024 / bytes_per_token).max(1).min(batch - t);
        if fast && !use_tensor {
            n = n.min(64 - (pos + t));
        }
        let needed = unsafe {
            ds4_attn_sparse_bytes(use_tensor.into(), span as i32, width as i32, n as i32)
        };
        let work = scratch(rt, needed)?;
        cuda_ok(
            unsafe {
                sparse(
                    rt.device,
                    work,
                    q_ptr.add(t * HEADS * DIM),
                    ring.data(),
                    current.as_ptr().cast(),
                    window_pos as i32,
                    t as i32,
                    span as i32,
                    i32::from(decode),
                    values,
                    ids.wrapping_add(t * width),
                    width as i32,
                    sink,
                    n as i32,
                    out_ptr.add(t * HEADS * DIM),
                )
            },
            "sparse attention",
        )?;
        t += n;
    }

    let kept = batch.min(2 * WINDOW);
    cuda_ok(
        unsafe {
            ds4_attn_window(
                rt.device,
                ring.ptr.as_ptr(),
                (pos + batch - kept) as i32,
                kept as i32,
                current
                    .as_mut_ptr()
                    .cast::<u16>()
                    .add((batch - kept) * DIM)
                    .cast(),
                DEVICE_TO_DEVICE,
            )
        },
        "updating attention window",
    )?;
    drop(current);
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                out_ptr,
                ptr::null_mut(),
                freqs,
                HEADS as i32,
                DIM as i32,
                32,
                1,
                batch as i32,
            )
        },
        "rotating and quantizing attention output",
    )?;
    window.next_pos = pos + batch;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "flashinfer-prefill")]
    include!("attention_flash_tests.rs");
    use super::*;

    #[test]
    fn only_state_continuity_and_session_memory_errors_are_request_local() {
        let mut state = AttentionState::default();
        state.windows.insert(
            0,
            Window {
                next_pos: 4,
                ..Window::default()
            },
        );
        let e = state
            .begin_verify(3, 1)
            .unwrap_err()
            .context("prepare target verification");
        assert!(request_local(&e));
        assert!(request_local(
            &state_allocation(2, "allocating one request's KV").unwrap_err()
        ));
        let picks = state_allocation(2, "uploading index picks")
            .unwrap_err()
            .context("preparing source rows");
        assert!(request_local(&picks));
        assert!(!request_local(
            &state_allocation(700, "restoring KV").unwrap_err()
        ));
        assert!(!request_local(
            &cuda_ok(2, "allocating shared workspace").unwrap_err()
        ));
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs"]
    fn cuda_grouped_attention_matches_live_spans_and_retires_only_failed_state() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask to both inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        rt.verification = true;
        fn values(rows: usize, seed: usize) -> Vec<f32> {
            (0..rows * HIDDEN)
                .map(|i| b16((((i * 13 + seed) % 257) as f32 - 129.) / 97.))
                .collect()
        }
        for layer in [0, 2, 20] {
            rt.layer(layer)?;
            let (mut left, mut right) = (AttentionState::default(), AttentionState::default());
            for (cache, rows, seed) in [(&mut left, 3, 17), (&mut right, 7, 57)] {
                cache.begin_prefill_with_limit(rows, 640);
                let input = rt.to_device(&values(rows, seed))?;
                forward_dev(&mut rt, cache, layer, 0, &input, rows)?;
            }
            let mut bad = left.host_snapshot()?;
            let runs = [(3, 4, 55), (7, 2, 99)];
            let mut expected = Vec::new();
            for (s, (pos, rows, seed)) in [left.host_snapshot()?, right.host_snapshot()?]
                .into_iter()
                .zip(runs)
            {
                let mut s = s;
                s.begin_prefill_with_limit(pos, 640);
                let input = rt.to_device(&values(rows, seed))?;
                let y = forward_dev(&mut rt, &mut s, layer, pos, &input, rows)?;
                expected.push(rt.to_host(&y)?);
            }
            let mut all = values(4, 55);
            all.extend(values(3, 71));
            all.extend(values(2, 99));
            let input = rt.to_device(&all)?;
            let (mut l, mut b, mut r) = (None, None, None);
            let combined = forward_group(
                &mut rt,
                layer,
                &input,
                &mut [
                    Span {
                        state: &mut left,
                        pos: 3,
                        rows: 4,
                        error: &mut l,
                    },
                    Span {
                        state: &mut bad,
                        pos: 9,
                        rows: 3,
                        error: &mut b,
                    },
                    Span {
                        state: &mut right,
                        pos: 7,
                        rows: 2,
                        error: &mut r,
                    },
                ],
            )?;
            assert!(l.is_none() && r.is_none());
            assert!(request_local(&b.unwrap()));
            assert!(bad.windows.is_empty() && bad.sources.is_empty());
            let out = rt.to_host(&combined)?;
            assert!(out[4 * HIDDEN..7 * HIDDEN].iter().all(|&x| x == 0.));
            for (actual, reference) in [
                (&out[..4 * HIDDEN], &expected[0]),
                (&out[7 * HIDDEN..], &expected[1]),
            ] {
                if crate::runtime::packed_dense() {
                    assert_eq!(actual, reference);
                } else {
                    let sq = actual
                        .iter()
                        .zip(reference)
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>();
                    let norm = reference.iter().map(|x| x * x).sum::<f32>();
                    assert!((sq / norm.max(1e-20)).sqrt() < 0.01);
                }
            }
            assert_eq!(left.windows[&layer].next_pos, 7);
            assert_eq!(right.windows[&layer].next_pos, 9);
        }
        Ok(())
    }

    #[test]
    fn six_full_sessions_reserve_only_their_live_compressed_bytes() {
        let limit = 524_288;
        let mut six = [0; 2];
        for (device, ratio, sources) in [(0, 2, 3), (1, 1, 1)] {
            let mut cache = 0;
            for stride in [INDEX_BYTES, KV_BYTES] {
                let mut rows = Rows::default();
                rows.plan(520_362, limit, ratio, stride);
                let initial = rows.next_capacity(0, stride);
                assert_eq!(initial, (520_362 + WINDOW) / ratio * stride);
                let full = rows.next_capacity(initial, limit / ratio * stride);
                assert_eq!(full, limit / ratio * stride);
                cache += sources * full;
            }
            six[device] = 6 * (cache + 20 * 2 * WINDOW * DIM * size_of::<u16>());
        }
        assert_eq!(six, [3252 << 20, 2178 << 20]);
        // A short request asking for a large output allocates incrementally.
        let mut short = Rows::default();
        short.plan(48, limit, 1, KV_BYTES);
        assert_eq!(short.next_capacity(0, KV_BYTES), (48 + WINDOW) * KV_BYTES);
        assert!(short.next_capacity(0, KV_BYTES) < limit * KV_BYTES);
    }

    #[test]
    fn fp_quant_rounding_and_packed_scales() {
        for mag in 0..127 {
            for sign in [0, 128] {
                let code = mag | sign;
                assert_eq!(e4m3_code(e4m3(code)), code);
            }
        }
        // Golden row uses the pinned SM120 record layout, independently packed
        // with PyTorch E4M3 round-to-nearest-even and identity RoPE.
        let row: Vec<_> = (0..DIM)
            .map(|i| b16((i * 8191 % 3323) as f32 / 1024. - 1.))
            .collect();
        let mut packed = Vec::new();
        pack_values(&row, &mut packed);
        assert_eq!(packed.len(), 584);
        let hash = packed.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, 0xe41b5c3a87eb5c6c);
        let mut keys = Vec::new();
        pack_index_keys(&row[..INDEX_DIM], &mut keys);
        assert_eq!(keys.len(), 132);
        let scale = f32::from_le_bytes(keys[128..].try_into().unwrap());
        assert_eq!(scale, 1. / 128.);
        let mut expected = row[..128].to_vec();
        fp8_qdq(&mut expected, 128);
        for (x, &c) in expected.iter().zip(&keys[..128]) {
            assert_eq!(*x, e4m3(c) * scale);
        }
    }

    #[test]
    fn pair_plan_honors_the_saved_orphan() -> Result<()> {
        let mut src = Source::default();
        assert_eq!(pair_span(&src, 0, 1)?, (0, true));
        assert_eq!(pair_span(&src, 12, 4)?, (2, false));
        assert!(pair_span(&src, 13, 4).is_err());
        src.pending_kv.resize(DIM, 1.);
        assert!(pair_span(&src, 13, 4).is_err());
        src.pending_gate.resize(DIM, 0.);
        assert_eq!(pair_span(&src, 13, 4)?, (2, true));
        assert_eq!(pair_span(&src, 13, 1)?, (1, false));
        assert!(pair_span(&src, 14, 2).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_pool_pairs_matches_cpu_weighted_values_and_packed_bytes() -> Result<()> {
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut kv: Vec<_> = (0..37 * DIM)
            .map(|j| ((j * 1117 % 4991) as f32 - 2451.) / 117.)
            .collect();
        let mut gates: Vec<_> = (0..kv.len())
            .map(|j| match j % 6 {
                0 => 0.,
                1 => -170.,
                2 => 80.,
                _ => ((j * 313 % 619) as f32 - 311.) / 43.,
            })
            .collect();
        // Exact cancellation across the first odd chunk boundary. An FMA
        // combining 3*exp(-1) with the other product would leave a residual.
        gates[0] = 0.;
        gates[DIM] = -1.;
        kv[DIM] = 3.;
        kv[0] = -(3. * (-1f32).exp());
        assert_ne!(kv[DIM].mul_add((-1f32).exp(), kv[0]), 0.);
        let kd = upload(dev, &kv)?;
        let gd = upload(dev, &gates)?;
        let mut src = Source::default();
        let mut actual_all = Vec::new();
        let mut expected_all = Vec::new();
        let mut pos = 0;
        let mut max_error = 0f32;
        for n in [1, 4, 3, 9, 2, 5, 13] {
            let (pairs, tail) = pair_span(&src, pos, n)?;
            let carry = if src.pending_kv.is_empty() {
                None
            } else {
                Some(upload(
                    dev,
                    &[src.pending_kv.as_slice(), src.pending_gate.as_slice()].concat(),
                )?)
            };
            let expected = pool_pairs(
                &kv[pos * DIM..(pos + n) * DIM],
                &gates[pos * DIM..(pos + n) * DIM],
                &mut src.pending_kv,
                &mut src.pending_gate,
            );
            assert_eq!(expected.len(), pairs * DIM);
            if pairs != 0 {
                let out = upload(dev, &vec![f32::NAN; pairs * DIM])?;
                cuda_ok(
                    unsafe {
                        ds4_attn_pool_pairs(
                            dev,
                            kd.data::<f32>().add(pos * DIM),
                            gd.data::<f32>().add(pos * DIM),
                            carry.as_ref().map_or(ptr::null(), |b| b.data()),
                            pairs as i32,
                            out.data(),
                        )
                    },
                    "GPU pooling fixture",
                )?;
                let mut got = vec![0f32; pairs * DIM];
                read(&out, &mut got)?;
                for (&actual, &reference) in got.iter().zip(&expected) {
                    let error = (actual - reference).abs();
                    assert!(error <= 1e-5, "pooled value {actual} vs {reference}");
                    max_error = max_error.max(error);
                }
                actual_all.extend(got);
                expected_all.extend(expected);
            }
            pos += n;
            if tail {
                assert_eq!(src.pending_kv, kv[(pos - 1) * DIM..pos * DIM]);
                assert_eq!(src.pending_gate, gates[(pos - 1) * DIM..pos * DIM]);
            } else {
                assert!(src.pending_kv.is_empty() && src.pending_gate.is_empty());
            }
        }
        assert_eq!(actual_all[0].to_bits(), 0f32.to_bits());
        let (mut bytes, mut expected) = (vec![], vec![]);
        pack_values(&actual_all, &mut bytes);
        pack_values(&expected_all, &mut expected);
        assert_eq!(bytes, expected);
        eprintln!(
            "{} CPU/GPU pooled features, max absolute error {max_error:e}, {} packed bytes",
            actual_all.len(),
            bytes.len()
        );
        Ok(())
    }

    #[test]
    fn pooling_across_unaligned_chunks_is_identical_to_prefill() {
        let x: Vec<_> = (0..DIM * 9)
            .map(|i| ((i * 711 % 881) as f32) / 97. - 4.)
            .collect();
        let g: Vec<_> = (0..DIM * 9)
            .map(|i| ((i * 303 % 933) as f32) / 77. - 5.)
            .collect();
        let (mut k, mut gate) = (Vec::new(), Vec::new());
        let expected = pool_pairs(&x, &g, &mut k, &mut gate);
        for chunks in [vec![1; 9], vec![3, 4, 2], vec![2, 1, 5, 1]] {
            let (mut ck, mut cg) = (vec![], vec![]);
            let mut actual = vec![];
            let mut offset = 0;
            for n in chunks {
                let end = offset + n * DIM;
                actual.extend(pool_pairs(
                    &x[offset..end],
                    &g[offset..end],
                    &mut ck,
                    &mut cg,
                ));
                offset = end;
            }
            assert_eq!(actual, expected);
            assert_eq!(ck, k);
            assert_eq!(cg, gate);
        }
    }

    #[test]
    fn prefill_column_layout_is_independent_of_chunk_size() {
        let mut state = AttentionState::default();
        for full in [65, 128, 200] {
            state.begin_prefill(full);
            let layout = state.window_layout(0, full);
            assert_eq!(layout, (full.min(WINDOW), false));
            for pos in (0..full).step_by(32) {
                assert_eq!(state.window_layout(pos, 32.min(full - pos)), layout);
            }
            assert_eq!(state.window_layout(full, 1), (WINDOW, true));
        }
    }

    #[test]
    fn host_snapshot_branches_from_an_unfinished_pair() {
        let mut original = AttentionState::default();
        let source = original.sources.entry(2).or_default();
        source.keys.packed = PackedRows::Owned(vec![1; INDEX_BYTES]);
        source.values.packed = PackedRows::Owned(vec![2; KV_BYTES]);
        source.pending_kv = vec![1.0; DIM];
        source.pending_gate = vec![0.0; DIM];
        original.windows.insert(
            2,
            Window {
                restored: None,
                rows: vec![17; DIM * 3],
                ring: None,
                next_pos: 3,
            },
        );
        original.selected = vec![vec![0]];
        original.blocks = vec![vec![0]];
        let mut branch = original.host_snapshot().unwrap();
        assert!(branch.blocks.is_empty() && branch.selected.is_empty());
        assert_eq!(branch.windows[&2].next_pos, 3);
        assert_eq!(branch.windows[&2].rows, original.windows[&2].rows);
        let copied = branch.sources.get_mut(&2).unwrap();
        assert!(copied.keys.devices.is_empty() && copied.values.devices.is_empty());
        assert_eq!(&*copied.keys.packed, &*original.sources[&2].keys.packed);
        assert_eq!(&*copied.values.packed, &*original.sources[&2].values.packed);
        let output = pool_pairs(
            &[3.; DIM],
            &[0.; DIM],
            &mut copied.pending_kv,
            &mut copied.pending_gate,
        );
        assert_eq!(output, [2.; DIM]);
        assert_eq!(original.sources[&2].pending_kv, [1.; DIM]);
    }

    #[test]
    fn two_stage_index_keeps_newest_partial_block() {
        let mut scores = vec![0.; TOP_BLOCKS * BLOCK + 3];
        scores[..BLOCK].fill(-3.);
        scores[TOP_BLOCKS * BLOCK..].fill(-10.);
        let keep = candidate_blocks(&scores);
        assert_eq!(keep.len(), TOP_BLOCKS);
        assert_eq!(keep[0], 1);
        assert_eq!(keep.last(), Some(&TOP_BLOCKS));
        assert_eq!(best(&[1., 3., 2., 3.], 2), [1, 3]);
    }

    #[test]
    fn verification_restores_compressed_bytes_and_each_possible_pending_pair() -> Result<()> {
        for pos in [3, 4] {
            let n = 7;
            let raw: Vec<_> = (0..(pos + n) * DIM).map(|i| (i % 3741) as f32).collect();
            let gate: Vec<_> = raw.iter().map(|v| -v).collect();
            for accepted in 0..=n {
                let mut state = AttentionState::default();
                let mut src = Source::default();
                let base = pos * DIM;
                pool_pairs(
                    &raw[..base],
                    &gate[..base],
                    &mut src.pending_kv,
                    &mut src.pending_gate,
                );
                state.begin_verify(pos, n)?;
                let pending = PendingPairs::capture(pos, &src, &raw[base..], &gate[base..]);
                state.verify.as_mut().unwrap().pending.insert(2, pending);
                pool_pairs(
                    &raw[base..],
                    &gate[base..],
                    &mut src.pending_kv,
                    &mut src.pending_gate,
                );
                src.keys
                    .packed
                    .for_append()
                    .resize((pos + n) / 2 * INDEX_BYTES, 1);
                src.values
                    .packed
                    .for_append()
                    .resize((pos + n) / 2 * KV_BYTES, 2);
                state.sources.insert(2, src);
                for layer in 0..40 {
                    state.windows.insert(
                        layer,
                        Window {
                            next_pos: pos + n,
                            ..Default::default()
                        },
                    );
                }
                state.finish_verify(accepted)?;
                let next = pos + accepted;
                assert!(state.windows.values().all(|w| w.next_pos == next));
                let src = &state.sources[&2];
                assert_eq!(src.values.packed.len(), next / 2 * KV_BYTES);
                assert_eq!(src.keys.packed.len(), next / 2 * INDEX_BYTES);
                let mut kv = Vec::new();
                let mut gates = Vec::new();
                pool_pairs(&raw[..next * DIM], &gate[..next * DIM], &mut kv, &mut gates);
                assert_eq!(src.pending_kv, kv);
                assert_eq!(src.pending_gate, gates);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible CUDA sm_120 inference GPU ordinal"]
    fn cuda_window_wrap_snapshot_and_speculative_rewind() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let pos = 291;
        let batch = WINDOW + 1;
        let mut state = AttentionState::default();
        state.begin_verify(pos, batch)?;
        let rows: Vec<_> = (0..(pos + batch) * DIM)
            .map(|i| (i * 613 % 65521) as u16)
            .collect();
        let mut window = Window {
            restored: None,
            rows: rows[(pos - WINDOW + 1) * DIM..pos * DIM].to_vec(),
            ring: None,
            next_pos: pos,
        };
        let new = upload(device, &rows[pos * DIM..])?;
        let ring = window.on(device, None)?;
        cuda_ok(
            unsafe {
                ds4_attn_window(
                    device,
                    ring.ptr.as_ptr(),
                    pos as i32,
                    batch as i32,
                    new.data(),
                    DEVICE_TO_DEVICE,
                )
            },
            "writing tentative window",
        )?;
        for keep in 0..=batch {
            let end = pos + keep;
            window.next_pos = end;
            let snapshot = window.host_snapshot()?;
            assert_eq!(snapshot.rows, rows[(end - WINDOW + 1) * DIM..end * DIM]);
        }
        Ok(())
    }

    fn upload<T: Copy>(device: i32, host: &[T]) -> Result<GpuBuffer> {
        let buf = GpuBuffer::new(device)?;
        buf.write(host)?;
        Ok(buf)
    }

    fn read<T: Copy>(buf: &GpuBuffer, host: &mut [T]) -> Result<()> {
        cuda_ok(
            unsafe {
                ds4_attn_read(
                    buf.device,
                    buf.data(),
                    host.as_mut_ptr().cast(),
                    std::mem::size_of_val(host),
                )
            },
            "reading test output",
        )
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible CUDA sm_120 inference GPU ordinal"]
    fn cuda_normalization_rope_and_quantization_match_reference() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let n = 3;
        let sample = |i: usize| ((i * 7057 % 19009) as f32 - 9000.) / 307.;
        for width in [512, 1280, 5120] {
            let x: Vec<_> = (0..n * width).map(sample).collect();
            let gamma: Vec<_> = (0..width).map(|i| b16(sample(i + 302) / 19.)).collect();
            let values = upload(device, &x)?;
            let g = upload(device, &gamma)?;
            let mut expected: Vec<_> = x.iter().copied().map(b16).collect();
            normalize(&mut expected, &gamma);
            cuda_ok(
                unsafe {
                    ds4_attn_normalize(device, values.data(), g.data(), width as i32, n as i32)
                },
                "normalization regression",
            )?;
            let mut actual = vec![0.; x.len()];
            read(&values, &mut actual)?;
            assert_eq!(actual, expected);
        }
        let angles: Vec<_> = (524284..524284 + n)
            .map(|pos| rotary_freqs(pos, true))
            .collect();
        let freqs = upload(device, &angles)?;
        for (heads, cols, group, inverse, window) in [
            (HEADS, DIM, 0, false, false),
            (HEADS, DIM, 32, true, false),
            (1, DIM, 64, false, true),
            (INDEX_HEADS, INDEX_DIM, INDEX_DIM, false, false),
        ] {
            let x: Vec<_> = (0..n * heads * cols).map(sample).collect();
            let mut expected: Vec<_> = x.iter().copied().map(b16).collect();
            for (i, row) in expected.chunks_exact_mut(cols).enumerate() {
                rotate(row, &angles[i / heads], inverse);
                if group > 0 {
                    fp8_qdq(&mut row[..if window { NOPE } else { cols }], group);
                }
            }
            let values = upload(device, &x)?;
            let packed = upload(device, &vec![0u16; x.len()])?;
            cuda_ok(
                unsafe {
                    ds4_attn_transform(
                        device,
                        values.data(),
                        if window {
                            packed.data()
                        } else {
                            ptr::null_mut()
                        },
                        freqs.data(),
                        heads as i32,
                        cols as i32,
                        group as i32,
                        i32::from(inverse),
                        n as i32,
                    )
                },
                "rotation regression",
            )?;
            if window {
                let mut actual = vec![0u16; x.len()];
                read(&packed, &mut actual)?;
                assert_eq!(
                    actual,
                    expected
                        .iter()
                        .map(|x| bf16::from_f32(*x).to_bits())
                        .collect::<Vec<_>>()
                );
            } else {
                let mut actual = vec![0.; x.len()];
                read(&values, &mut actual)?;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_index_packed_scores_match_scalar_bits() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let rows = 8;
        let keys = 65_551;
        let mut packed = vec![0u8; keys * INDEX_BYTES];
        for (i, key) in packed
            .as_chunks_mut::<INDEX_BYTES>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (j, v) in key[..INDEX_DIM].iter_mut().enumerate() {
                let code = (i * 41 + j * 137 + 3 * i * j) % 253;
                *v = (code % 127) as u8 | if code >= 127 { 0x80 } else { 0 };
            }
            let scale = 2f32.powi((i % 19) as i32 - 9);
            key[INDEX_DIM..].copy_from_slice(&scale.to_le_bytes());
        }
        let packed = upload(device, &packed)?;
        let q: Vec<_> = (0..rows * INDEX_HEADS * INDEX_DIM)
            .map(|i| ((i * 503 % 1093) as f32 - 546.) / 77.)
            .collect();
        let w: Vec<_> = (0..rows * INDEX_HEADS)
            .map(|i| ((i * 571 % 1597) as f32 - 798.) / 109.)
            .collect();
        let q = upload(device, &q)?;
        let w = upload(device, &w)?;
        let scratch = GpuBuffer::new(device)?;
        for width in [1, 7, 8, 9, 31, 32, 33, 607, 65_537] {
            let ids: Vec<i32> = (0..rows * width)
                .map(|i| {
                    let (r, c) = (i / width, i % width);
                    if r == 0 || c % 5 == r % 5 || c >= width.saturating_sub(r) {
                        -1
                    } else {
                        ((c / 2 * 17 + r * 47) % keys) as i32
                    }
                })
                .collect();
            for candidates in [ptr::null(), ids.as_ptr()] {
                let score = |mode| -> Result<Vec<u32>> {
                    let mut out = vec![0f32; rows * width];
                    cuda_ok(
                        unsafe {
                            ds4_attn_scores(
                                device,
                                scratch.ptr.as_ptr(),
                                packed.ptr.as_ptr(),
                                q.data(),
                                w.data(),
                                candidates,
                                width as i32,
                                rows as i32,
                                out.as_mut_ptr(),
                                mode,
                            )
                        },
                        "checking packed index score bits",
                    )?;
                    Ok(out.into_iter().map(f32::to_bits).collect())
                };
                assert_eq!(score(0)?, score(1)?, "index scores width={width}");
                // These FP32 queries do not round-trip through one E4M3
                // group scale per head. Every fallback bit must match BF16 MMA.
                assert_eq!(score(2)?, score(3)?, "FP8 fallback width={width}");
                assert_eq!(score(2)?, score(4)?, "batched MQA fallback width={width}");
            }
        }
        Ok(())
    }

    #[test]
    fn batched_mqa_selects_large_contiguous_scans() {
        assert!(batched_mqa_tile(524288, 4, false));
        assert!(batched_mqa_tile(16384, 64, false));
        assert!(!batched_mqa_tile(16384, 64, true));
        assert!(!batched_mqa_tile(65536, 8, false));
        assert!(!batched_mqa_tile(600, 8192, false));
        assert!(!batched_mqa_tile(524288, 3, false));
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_index_mma_matches_quantized_scores() -> Result<()> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut rng = StdRng::seed_from_u64(0x8e39);
        let rows = 9;
        let count = 16407;
        let mut packed = Vec::new();
        for i in 0..count {
            let gain = 2f32.powi((i % 5) - 2);
            let row: [f32; INDEX_DIM] = std::array::from_fn(|_| rng.gen_range(-gain..gain));
            pack_index_keys(&row, &mut packed);
        }
        // Masked lanes and partially filled warps must tolerate stale NaN
        // data in the cache row used to pad the MMA.
        packed[..INDEX_DIM].fill(0x7f);
        let keys = upload(device, &packed)?;
        let mut q: Vec<f32> = (0..rows * INDEX_HEADS * INDEX_DIM)
            .map(|_| rng.gen_range(-2.0..2.0))
            .collect();
        fp8_qdq(&mut q, INDEX_DIM);
        let q = upload(device, &q)?;
        let w: Vec<f32> = (0..rows * INDEX_HEADS)
            .map(|_| rng.gen_range(-3.0..3.0))
            .collect();
        let w = upload(device, &w)?;
        for width in [1usize, 15, 16, 17, 255, 256, 257, 613, 16_391] {
            let ids: Vec<i32> = (0..rows * width)
                .map(|i| {
                    let (r, c) = (i / width, i % width);
                    if r == 1 || c % 7 == r % 7 || c >= width.saturating_sub(r) {
                        -1
                    } else {
                        ((c / 2 * 157 + r * 13) % count as usize) as i32
                    }
                })
                .collect();
            for filtered in [false, true] {
                let score = |mode| -> Result<Vec<f32>> {
                    let scratch = upload(device, &vec![0xffu8; rows * width * 8 + 512])?;
                    let mut out = vec![f32::NAN; rows * width];
                    cuda_ok(
                        unsafe {
                            ds4_attn_scores(
                                device,
                                scratch.ptr.as_ptr(),
                                keys.ptr.as_ptr(),
                                q.data(),
                                w.data(),
                                if filtered { ids.as_ptr() } else { ptr::null() },
                                width as i32,
                                rows as i32,
                                out.as_mut_ptr(),
                                mode,
                            )
                        },
                        "checking prompt index scores",
                    )?;
                    Ok(out)
                };
                let reference = score(0)?;
                let fp8 = score(3)?;
                let batched = score(4)?;
                for (i, (single, grouped)) in fp8.iter().zip(&batched).enumerate() {
                    assert_eq!(
                        single.to_bits(),
                        grouped.to_bits(),
                        "batched MQA width={width} slot={i}"
                    );
                }
                for mode in [2, 3] {
                    let actual = score(mode)?;
                    for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
                        if filtered && ids[i] < 0 {
                            assert_eq!(a, f32::NEG_INFINITY);
                        } else {
                            assert!(
                            (a - b).abs() <= 1e-5 + 3e-6 * b.abs(),
                            "width={width} filtered={filtered} slot={i} expected={b} actual={a}"
                        );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_index_fp8_preserves_query_exponent_and_mixed_fallback_rows() -> Result<()> {
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let rows = 2;
        let mut keys = [0u8; INDEX_BYTES];
        keys[0] = e4m3_code(448.);
        keys[INDEX_DIM..].copy_from_slice(&2f32.powi(118).to_le_bytes());
        let keys = upload(dev, &keys)?;
        let mut q = vec![0f32; rows * INDEX_HEADS * INDEX_DIM];
        q[0] = 1. / 64.; // E4M3 encoded as 256 with scale 2^-14.
        q[INDEX_HEADS * INDEX_DIM] = 0.014111; // Must use the BF16 fallback.
        let q = upload(dev, &q)?;
        let mut w = vec![0f32; rows * INDEX_HEADS];
        w[0] = 64.;
        w[INDEX_HEADS] = 64.;
        let w = upload(dev, &w)?;
        let scratch = GpuBuffer::new(dev)?;
        let score = |mode| -> Result<[f32; 2]> {
            let mut out = [f32::NAN; 2];
            cuda_ok(
                unsafe {
                    ds4_attn_scores(
                        dev,
                        scratch.ptr.as_ptr(),
                        keys.ptr.as_ptr(),
                        q.data(),
                        w.data(),
                        ptr::null(),
                        1,
                        rows as i32,
                        out.as_mut_ptr(),
                        mode,
                    )
                },
                "checking mixed FP8 and BF16 index rows",
            )?;
            Ok(out)
        };
        let result = score(3)?;
        assert_eq!(result, score(4)?);
        assert!(result.iter().all(|s| s.is_finite()));
        assert_eq!(result[0], 7. * 2f32.powi(118));
        assert_eq!(result, score(2)?);
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_index_signed_heads_resolve_close_cutoff_and_exact_ties() -> Result<()> {
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let width = TOP + 8;
        let rows = 2;
        let pos = width - rows;

        // 511 positions score above the cutoff. Row 0 must choose the earlier
        // of two tied keys. Row 1 exposes a new key that beats both by 2^-15.
        let mut keys = vec![0u8; width * INDEX_BYTES];
        let mut expected_scores = Vec::with_capacity(width);
        for (i, key) in keys.as_chunks_mut::<INDEX_BYTES>().0.iter_mut().enumerate() {
            let step = if i < TOP - 1 {
                -2.
            } else if i == width - 1 {
                -1.
            } else if i == TOP + 1 || i == TOP + 3 {
                0.
            } else {
                1.
            };
            key[..INDEX_DIM].fill(e4m3_code(1.));
            key[1] = e4m3_code(step / 512.);
            key[INDEX_DIM..].copy_from_slice(&1f32.to_le_bytes());
            expected_scores.push(-4. / 64. - step / 32768.);
        }
        let mut queries = vec![0f32; rows * INDEX_HEADS * INDEX_DIM];
        let mut weights = vec![0f32; rows * INDEX_HEADS];
        for (i, q) in queries
            .as_chunks_mut::<INDEX_DIM>()
            .0
            .iter_mut()
            .enumerate()
        {
            let head = i % INDEX_HEADS;
            q[0] = [0.5, 1., 2., -1.][head % 4];
            q[1] = f32::from(head == 1);
            // Changing only head 2 contributes a constant -32/64 shift.
            weights[i] = if head == 2 {
                -14.
            } else {
                [1., -1., 2., -1.][head % 4]
            };
            // Opposite features cancel. ReLU suppresses every fourth head
            // before the remaining positive and negative weights are applied.
            for pair in q[2..].chunks_exact_mut(2) {
                let v = ((head % 8) + 1) as f32 / 16.;
                pair.copy_from_slice(&[v, -v]);
            }
        }
        let keys = upload(dev, &keys)?;
        let q = upload(dev, &queries)?;
        let w = upload(dev, &weights)?;
        let mut bytes = 0;
        cuda_ok(
            unsafe { ds4_attn_index_bytes(width as i32, rows as i32, &mut bytes) },
            "sizing close-cutoff scratch",
        )?;
        let scratch = upload(dev, &vec![0xffu8; bytes])?;
        for mode in [1, 2, 3, 4] {
            let picks = upload(dev, &vec![-9i32; rows * TOP])?;
            let mut actual = vec![f32::NAN; width * rows];
            cuda_ok(
                unsafe {
                    ds4_attn_scores(
                        dev,
                        scratch.ptr.as_ptr(),
                        keys.ptr.as_ptr(),
                        q.data(),
                        w.data(),
                        ptr::null(),
                        width as i32,
                        rows as i32,
                        actual.as_mut_ptr(),
                        mode,
                    )
                },
                "checking close-cutoff scores",
            )?;
            // This bound is less than half the 2^-15 cutoff gap. Ranking
            // must agree exactly, including the lower-index tie breaker.
            for (i, row) in actual.chunks_exact(width).enumerate() {
                let boundary_key = if i == 0 { TOP + 1 } else { width - 1 };
                assert!(
                    row[boundary_key] < 0.,
                    "cutoff must be negative, mode={mode} row={i}"
                );
                for (&a, &b) in row.iter().zip(&expected_scores) {
                    assert!((a - b).abs() <= 1e-5, "mode={mode}: {a} vs {b}");
                }
            }
            cuda_ok(
                unsafe {
                    ds4_attn_index_select(
                        dev,
                        scratch.data(),
                        keys.ptr.as_ptr(),
                        q.data(),
                        w.data(),
                        ptr::null(),
                        width as i32,
                        pos as i32,
                        1,
                        rows as i32,
                        picks.data(),
                        ptr::null_mut(),
                        mode,
                    )
                },
                "ranking close-cutoff keys",
            )?;
            let mut actual = vec![-1i32; rows * TOP];
            read(&picks, &mut actual)?;
            for (i, row) in actual.chunks_exact(TOP).enumerate() {
                let expected = best(&expected_scores[..pos + i + 1], TOP);
                assert_eq!(expected[TOP - 1], if i == 0 { TOP + 1 } else { width - 1 });
                assert_eq!(
                    row,
                    expected.into_iter().map(|id| id as i32).collect::<Vec<_>>(),
                    "close-cutoff winners mode={mode}, row={i}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_index_matches_cpu_ranking_across_causal_thresholds() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let short = upload(device, &[-9i32; 6])?;
        cuda_ok(
            unsafe { ds4_attn_index_ids(device, short.data(), 2, 1, 2, 3) },
            "short index rows",
        )?;
        let mut host = [0i32; 6];
        read(&short, &mut host)?;
        assert_eq!(host, [0, -1, 0, -1, 0, 1]);
        check_index_ranking(device, 1)?;
        check_index_ranking(device, 2)?;
        check_index_ranking(device, 3)?;
        check_index_ranking(device, 4)
    }

    fn check_index_ranking(device: i32, score_mode: i32) -> Result<()> {
        for (pos, ratio, n, zero) in [
            (1020, 2, 129, false),
            (1530, 1, 7, false),
            (4096, 1, 7, false),
            (24575, 1, 8, true),
            (32760, 1, 8, false),
            (TOP_BLOCKS * BLOCK - 2, 1, 9, false),
            (TOP_BLOCKS * BLOCK + 109, 1, 3, true),
        ] {
            let width = (pos + n) / ratio;
            let scores: Vec<f32> = (0..width)
                .map(|k| {
                    if zero {
                        0.
                    } else {
                        ((k.wrapping_mul(1717) ^ (k / 7)) % 997) as f32
                    }
                })
                .collect();
            let mut packed = vec![0u8; width * INDEX_BYTES];
            for (row, &score) in packed
                .as_chunks_mut::<INDEX_BYTES>()
                .0
                .iter_mut()
                .zip(&scores)
            {
                row[0] = 0x38; // E4M3 1.0
                row[INDEX_DIM..].copy_from_slice(&score.to_le_bytes());
            }
            let packed = upload(device, &packed)?;
            let mut q = vec![0f32; n * INDEX_HEADS * INDEX_DIM];
            let mut w = vec![0f32; n * INDEX_HEADS];
            for j in 0..n {
                q[j * INDEX_HEADS * INDEX_DIM] = 1.;
                w[j * INDEX_HEADS] = 64.;
            }
            let q = upload(device, &q)?;
            let w = upload(device, &w)?;
            let blocks = upload(device, &vec![-1i32; n * TOP_BLOCKS])?;
            let picks = upload(device, &vec![-1i32; n * TOP])?;
            let mut bytes = 0;
            cuda_ok(
                unsafe { ds4_attn_index_bytes(width as i32, n as i32, &mut bytes) },
                "index fixture size",
            )?;
            let work = upload(device, &vec![0u8; bytes])?;
            let has_blocks = width > TOP_BLOCKS * BLOCK;
            cuda_ok(
                unsafe {
                    ds4_attn_index_select(
                        device,
                        work.data(),
                        packed.ptr.as_ptr(),
                        q.data(),
                        w.data(),
                        ptr::null(),
                        width as i32,
                        pos as i32,
                        ratio as i32,
                        n as i32,
                        picks.data(),
                        if has_blocks {
                            blocks.data()
                        } else {
                            ptr::null_mut()
                        },
                        score_mode,
                    )
                },
                "index fixture",
            )?;
            let mut got = vec![0i32; n * TOP];
            read(&picks, &mut got)?;
            let mut chosen = vec![0i32; n * TOP_BLOCKS];
            let mut batched = vec![0i32; n * TOP];
            if has_blocks {
                read(&blocks, &mut chosen)?;
                let filtered_width = (TOP_BLOCKS * BLOCK) as i32;
                let mut bytes = 0;
                cuda_ok(
                    unsafe { ds4_attn_index_bytes(filtered_width, n as i32, &mut bytes) },
                    "batched filtered index fixture size",
                )?;
                let scratch = upload(device, &vec![0u8; bytes])?;
                cuda_ok(
                    unsafe {
                        ds4_attn_index_select(
                            device,
                            scratch.data(),
                            packed.ptr.as_ptr(),
                            q.data(),
                            w.data(),
                            blocks.data(),
                            filtered_width,
                            pos as i32,
                            ratio as i32,
                            n as i32,
                            picks.data(),
                            ptr::null_mut(),
                            score_mode,
                        )
                    },
                    "batched filtered index fixture",
                )?;
                read(&picks, &mut batched)?;
            }
            for j in 0..n {
                let visible = (pos + j + 1) / ratio;
                let mut want: Vec<i32> = best(&scores[..visible], TOP)
                    .into_iter()
                    .map(|x| x as i32)
                    .collect();
                want.resize(TOP, -1);
                assert_eq!(got[j * TOP..(j + 1) * TOP], want, "visible {visible}");
                if visible <= TOP_BLOCKS * BLOCK {
                    continue;
                }
                let spans = candidate_blocks(&scores[..visible]);
                assert_eq!(
                    chosen[j * TOP_BLOCKS..(j + 1) * TOP_BLOCKS],
                    spans.iter().map(|&k| k as i32).collect::<Vec<_>>()
                );
                let ids: Vec<usize> = spans
                    .iter()
                    .flat_map(|&b| b * BLOCK..((b + 1) * BLOCK).min(visible))
                    .collect();
                let mut weight = 0;
                cuda_ok(
                    unsafe { ds4_attn_index_bytes((TOP_BLOCKS * BLOCK) as i32, 1, &mut weight) },
                    "filtered index fixture size",
                )?;
                let scratch = upload(device, &vec![0u8; weight])?;
                cuda_ok(
                    unsafe {
                        ds4_attn_index_select(
                            device,
                            scratch.data(),
                            packed.ptr.as_ptr(),
                            q.data::<f32>().add(j * INDEX_HEADS * INDEX_DIM),
                            w.data::<f32>().add(j * INDEX_HEADS),
                            blocks.data::<i32>().add(j * TOP_BLOCKS),
                            (TOP_BLOCKS * BLOCK) as i32,
                            (pos + j) as i32,
                            ratio as i32,
                            1,
                            picks.data(),
                            ptr::null_mut(),
                            score_mode,
                        )
                    },
                    "filtered index fixture",
                )?;
                let mut actual = vec![0i32; TOP];
                read(&picks, &mut actual)?;
                let filtered: Vec<f32> = ids.iter().map(|&k| scores[k]).collect();
                let want: Vec<i32> = best(&filtered, TOP)
                    .into_iter()
                    .map(|k| ids[k] as i32)
                    .collect();
                assert_eq!(actual, want, "filtered visible {visible}");
                assert_eq!(
                    batched[j * TOP..(j + 1) * TOP],
                    want,
                    "batched visible {visible}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn cuda_resident_compressor_bytes_match_restored_and_rejected_prefixes() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let n = 9;
        let mut values: Vec<f32> = (0..n * DIM)
            .map(|i| ((i as f32 + 0.1) * 0.179).sin() * 13.317)
            .collect();
        let mut keys: Vec<f32> = (0..n * INDEX_DIM)
            .map(|i| b16(((i as f32 + 0.9) * 0.207).cos() * 7.059))
            .collect();
        let gamma: Vec<f32> = (0..DIM)
            .map(|j| 0.01 + ((j + 3) % 93) as f32 / 88.)
            .collect();
        let angles: Vec<_> = (0..n).map(|t| rotary_freqs(991 + 2 * t, true)).collect();
        let v = upload(device, &values)?;
        let k = upload(device, &keys)?;
        let g = upload(device, &gamma)?;
        let f = upload(device, &angles)?;
        cuda_ok(
            unsafe { ds4_attn_normalize_raw(device, v.data(), g.data(), n as i32) },
            "raw compressor normalization fixture",
        )?;
        cuda_ok(
            unsafe { ds4_attn_normalize(device, k.data(), g.data(), INDEX_DIM as i32, n as i32) },
            "index normalization fixture",
        )?;
        normalize(&mut values, &gamma);
        normalize(&mut keys, &gamma[..INDEX_DIM]);
        for (d, src, width) in [(&v, &mut values, DIM), (&k, &mut keys, INDEX_DIM)] {
            cuda_ok(
                unsafe {
                    ds4_attn_transform(
                        device,
                        d.data(),
                        ptr::null_mut(),
                        f.data(),
                        1,
                        width as i32,
                        0,
                        0,
                        n as i32,
                    )
                },
                "compressed rotary fixture",
            )?;
            for (row, angle) in src.chunks_exact_mut(width).zip(&angles) {
                rotate(row, angle, false);
            }
        }
        let mut ref_v = Vec::new();
        let mut ref_k = Vec::new();
        pack_values(&values, &mut ref_v);
        pack_index_keys(&keys, &mut ref_k);
        let mut vs = Rows {
            packed: PackedRows::Owned(ref_v[..KV_BYTES].to_vec()),
            ..Rows::default()
        };
        let mut ks = Rows {
            packed: PackedRows::Owned(ref_k[..INDEX_BYTES].to_vec()),
            ..Rows::default()
        };
        // Begin with a host snapshot. Append GPU rows, roll back three, and
        // append new bytes into the previously rejected slots.
        for (start, end) in [(1, n), (n - 3, n)] {
            let size = end - start;
            let mut dest = [ptr::null_mut(); 2];
            for ((r, width), dst) in [(&mut vs, KV_BYTES), (&mut ks, INDEX_BYTES)]
                .into_iter()
                .zip(&mut dest)
            {
                r.truncate(start * width);
                let h = r.with_tail(device, None, size * width)?;
                *dst = unsafe { ds4_attn_data(h).cast::<u8>().add(start * width) };
            }
            cuda_ok(
                unsafe {
                    ds4_attn_pack_compressed(
                        device,
                        v.data::<f32>().add(start * DIM),
                        k.data::<f32>().add(start * INDEX_DIM),
                        dest[0],
                        dest[1],
                        size as i32,
                    )
                },
                "packing compressed fixture",
            )?;
            vs.finish_append(device, size * KV_BYTES);
            ks.finish_append(device, size * INDEX_BYTES);
            for (actual, expected) in [(&vs, &ref_v), (&ks, &ref_k)] {
                let mut bytes = Vec::new();
                actual.append_snapshot(&mut bytes)?;
                assert_eq!(&bytes, expected);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs"]
    fn cuda_checkpoint_compressor_matches_across_split_pairs() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask to both inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        for (owner, ratio) in [(2, 2), (20, 1)] {
            rt.layer(owner)?;
            let (mut cpu, mut gpu) = (Source::default(), Source::default());
            let prefix = format!("layers.{owner}.attn");
            for (pos, n) in [(0, 1), (1, 4), (5, 2), (7, 8)] {
                let x: Vec<_> = (0..n * HIDDEN)
                    .map(|i| b16(((i + pos * HIDDEN) as f32 * 0.1369).cos()))
                    .collect();
                let x = rt.to_device(&x)?;
                let mark = || Verification {
                    pos,
                    batch: n,
                    pending: HashMap::new(),
                };
                let (mut cp, mut gp) = (mark(), mark());
                append_source(
                    &mut rt,
                    &mut cpu,
                    &prefix,
                    &x,
                    pos,
                    n,
                    ratio,
                    owner,
                    Some(&mut cp),
                    false,
                )?;
                append_source(
                    &mut rt,
                    &mut gpu,
                    &prefix,
                    &x,
                    pos,
                    n,
                    ratio,
                    owner,
                    Some(&mut gp),
                    true,
                )?;
                for (got, want) in [(&gpu.keys, &cpu.keys), (&gpu.values, &cpu.values)] {
                    let mut bytes = Vec::new();
                    got.append_snapshot(&mut bytes)?;
                    assert_eq!(
                        bytes, &*want.packed,
                        "checkpoint compressed rows owner={owner}, pos={pos}"
                    );
                }
                assert_eq!(cpu.pending_kv, gpu.pending_kv);
                assert_eq!(cpu.pending_gate, gpu.pending_gate);
                if ratio == 2 {
                    assert_eq!(cp.pending[&owner].kv, gp.pending[&owner].kv);
                    assert_eq!(cp.pending[&owner].gates, gp.pending[&owner].gates);
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs"]
    fn cuda_checkpoint_prefill_pool_pairs_preserves_cache_and_orphans() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask the two inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        let owner = 2;
        rt.layer(owner)?;
        rt.target_prefill = true;
        rt.gpu_pair_pool = true;
        let (mut cpu, mut gpu) = (Source::default(), Source::default());
        let mut pos = 0;
        for n in [4, 1, 3, 9, 2, 1] {
            let x: Vec<_> = (pos * HIDDEN..(pos + n) * HIDDEN)
                .map(|i| b16((i as f32 * 0.1369).cos()))
                .collect();
            let x = rt.to_device(&x)?;
            // Verification must retain the original host pooling/capture path.
            let mut mark = Verification {
                pos,
                batch: n,
                pending: HashMap::new(),
            };
            append_source(
                &mut rt,
                &mut cpu,
                "layers.2.attn",
                &x,
                pos,
                n,
                2,
                owner,
                Some(&mut mark),
                true,
            )?;
            assert!(mark.pending.contains_key(&owner));
            append_source(
                &mut rt,
                &mut gpu,
                "layers.2.attn",
                &x,
                pos,
                n,
                2,
                owner,
                None,
                true,
            )?;
            for (a, b) in [(&gpu.keys, &cpu.keys), (&gpu.values, &cpu.values)] {
                let (mut got, mut expected) = (Vec::new(), Vec::new());
                a.append_snapshot(&mut got)?;
                b.append_snapshot(&mut expected)?;
                assert_eq!(got, expected, "packed checkpoint bytes at {pos}+{n}");
            }
            assert_eq!(gpu.pending_kv, cpu.pending_kv);
            assert_eq!(gpu.pending_gate, cpu.pending_gate);
            pos += n;
        }
        Ok(())
    }

    #[test]
    #[ignore = "compares full 512k index selection timing on DS4_TEST_DEVICE"]
    fn cuda_index_512k_timing_and_parity() -> Result<()> {
        unsafe extern "C" {
            fn cudaDeviceSynchronize() -> i32;
        }
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        for (width, n) in [
            (600, 129),
            (8192, 4),
            (16391, 64),
            (32768, 4),
            (65536, 8),
            (65536, 32),
            (131072, 16),
            (262144, 8),
            (505000, 4),
            (524288, 4),
            (524288, 1),
        ] {
            let mut keys = vec![0u8; width * INDEX_BYTES];
            for (k, row) in keys.as_chunks_mut::<INDEX_BYTES>().0.iter_mut().enumerate() {
                row[0] = 0x38;
                let scale = ((k.wrapping_mul(6563) ^ (k / 19)) % 8191) as f32;
                row[INDEX_DIM..].copy_from_slice(&scale.to_le_bytes());
            }
            let keys = upload(dev, &keys)?;
            let q = upload(dev, &vec![1f32; n * INDEX_HEADS * INDEX_DIM])?;
            let w = upload(dev, &vec![2f32; n * INDEX_HEADS])?;
            let picks = upload(dev, &vec![-1i32; n * TOP])?;
            let blocks = upload(dev, &vec![-1i32; n * TOP_BLOCKS])?;
            let mut bytes = 0;
            cuda_ok(
                unsafe { ds4_attn_index_bytes(width as i32, n as i32, &mut bytes) },
                "index benchmark layout",
            )?;
            let workspace = upload(dev, &vec![0u8; bytes])?;
            let scratch = GpuBuffer::new(dev)?;
            let has_blocks = width > TOP_BLOCKS * BLOCK;
            let pos = width - n;
            let run = |score_mode| {
                cuda_ok(
                    unsafe {
                        ds4_attn_index_select(
                            dev,
                            workspace.data(),
                            keys.ptr.as_ptr(),
                            q.data(),
                            w.data(),
                            ptr::null(),
                            width as i32,
                            pos as i32,
                            1,
                            n as i32,
                            picks.data(),
                            if has_blocks {
                                blocks.data()
                            } else {
                                ptr::null_mut()
                            },
                            score_mode,
                        )
                    },
                    "index benchmark selection",
                )
            };
            let mut scores = vec![0f32; width * n];
            let mut reference = || -> Result<(Vec<i32>, Vec<i32>)> {
                cuda_ok(
                    unsafe {
                        ds4_attn_scores(
                            dev,
                            scratch.ptr.as_ptr(),
                            keys.ptr.as_ptr(),
                            q.data(),
                            w.data(),
                            ptr::null(),
                            width as i32,
                            n as i32,
                            scores.as_mut_ptr(),
                            0,
                        )
                    },
                    "index benchmark baseline",
                )?;
                let mut selected = Vec::with_capacity(n * TOP);
                let mut candidates = Vec::with_capacity(n * TOP_BLOCKS);
                for (j, row) in scores.chunks_exact(width).enumerate() {
                    let visible = &row[..pos + j + 1];
                    selected.extend(best(visible, TOP).into_iter().map(|id| id as i32));
                    selected.resize((j + 1) * TOP, -1);
                    if has_blocks {
                        candidates.extend(
                            if visible.len() <= TOP_BLOCKS * BLOCK {
                                // The native tile emits blocks for every row, even if the
                                // caller ignores these below the filtering threshold.
                                (0..visible.len().div_ceil(BLOCK)).collect()
                            } else {
                                candidate_blocks(visible)
                            }
                            .into_iter()
                            .map(|id| id as i32),
                        );
                        candidates.resize((j + 1) * TOP_BLOCKS, -1);
                    }
                }
                Ok((selected, candidates))
            };
            let (expected, expected_blocks) = reference()?;
            if width == 16391 {
                for out_blocks in [false, true] {
                    let mut needed = 0;
                    cuda_ok(
                        unsafe {
                            ds4_attn_scan_bytes(
                                width as i32,
                                n as i32,
                                3,
                                out_blocks.into(),
                                &mut needed,
                            )
                        },
                        "sizing prompt selection",
                    )?;
                    assert!(needed <= bytes);
                    let guarded = upload(dev, &vec![0xa5u8; needed + 512])?;
                    picks.write(&vec![-9i32; n * TOP])?;
                    cuda_ok(
                        unsafe {
                            ds4_attn_index_select(
                                dev,
                                guarded.data::<u8>().add(256),
                                keys.ptr.as_ptr(),
                                q.data(),
                                w.data(),
                                ptr::null(),
                                width as i32,
                                pos as i32,
                                1,
                                n as i32,
                                picks.data(),
                                if out_blocks {
                                    blocks.data()
                                } else {
                                    ptr::null_mut()
                                },
                                3,
                            )
                        },
                        "using the serving allocation size",
                    )?;
                    let mut got = vec![0; n * TOP];
                    read(&picks, &mut got)?;
                    assert_eq!(got, expected);
                    for offset in [0, 256 + needed] {
                        let mut guard = [0u8; 256];
                        cuda_ok(
                            unsafe {
                                ds4_attn_read(
                                    dev,
                                    guarded.data::<u8>().add(offset).cast(),
                                    guard.as_mut_ptr().cast(),
                                    guard.len(),
                                )
                            },
                            "reading attention scratch guard",
                        )?;
                        assert_eq!(guard, [0xa5; 256]);
                    }
                }
            }
            for mode in [1, 2, 3, 4] {
                // Poison both outputs so an omitted write by the second
                // scorer cannot inherit a passing result from the first.
                picks.write(&vec![-9i32; n * TOP])?;
                blocks.write(&vec![-9i32; n * TOP_BLOCKS])?;
                run(mode)?;
                let mut got = vec![0i32; n * TOP];
                read(&picks, &mut got)?;
                assert_eq!(
                    got, expected,
                    "index token winners width={width} rows={n} mode={mode}"
                );
                if has_blocks {
                    let mut got = vec![0i32; n * TOP_BLOCKS];
                    read(&blocks, &mut got)?;
                    assert_eq!(
                        got, expected_blocks,
                        "index block winners width={width} rows={n} mode={mode}"
                    );
                }
            }
            let time_gpu = |run: &dyn Fn(i32) -> Result<()>, mode| -> Result<f64> {
                run(mode)?;
                cuda_ok(unsafe { cudaDeviceSynchronize() }, "warming GPU index")?;
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    run(mode)?;
                }
                cuda_ok(unsafe { cudaDeviceSynchronize() }, "timing GPU index")?;
                Ok(start.elapsed().as_secs_f64() * 1e6 / 5.)
            };
            let scalar_us = time_gpu(&run, 0)?;
            let packed_us = time_gpu(&run, 1)?;
            let mma_us = time_gpu(&run, 2)?;
            let fp8_us = time_gpu(&run, 3)?;
            let mqa_us = time_gpu(&run, 4)?;
            if pos >= TOP_BLOCKS * BLOCK {
                let filtered_width = (TOP_BLOCKS * BLOCK) as i32;
                let mut bytes = 0;
                cuda_ok(
                    unsafe { ds4_attn_index_bytes(filtered_width, n as i32, &mut bytes) },
                    "sizing filtered index benchmark",
                )?;
                let filtered_workspace = upload(dev, &vec![0u8; bytes])?;
                let filtered = |mode| {
                    cuda_ok(
                        unsafe {
                            ds4_attn_index_select(
                                dev,
                                filtered_workspace.data(),
                                keys.ptr.as_ptr(),
                                q.data(),
                                w.data(),
                                blocks.data(),
                                filtered_width,
                                pos as i32,
                                1,
                                n as i32,
                                picks.data(),
                                ptr::null_mut(),
                                mode,
                            )
                        },
                        "benchmarking filtered index selection",
                    )
                };
                // Every full top-512 winner is inside one of the winning
                // 2048 blocks, even with the newest block forced in.
                for mode in [2, 3] {
                    picks.write(&vec![-9i32; n * TOP])?;
                    filtered(mode)?;
                    let mut got = vec![0i32; n * TOP];
                    read(&picks, &mut got)?;
                    assert_eq!(
                        got, expected,
                        "filtered winners at width={width} mode={mode}"
                    );
                }
                let bf16_us = time_gpu(&filtered, 2)?;
                let fp8_us = time_gpu(&filtered, 3)?;
                eprintln!("filtered index rows={n} source={width} candidates={filtered_width} mma={bf16_us:.1}us fp8={fp8_us:.1}us");
            }
            let start = std::time::Instant::now();
            for _ in 0..3 {
                std::hint::black_box(reference()?);
            }
            let host_us = start.elapsed().as_secs_f64() * 1e6 / 3.;
            eprintln!(
                "index rows={n} positions={width} workspace={} bytes scalar={scalar_us:.1}us packed={packed_us:.1}us mma={mma_us:.1}us fp8={fp8_us:.1}us batched-mqa={mqa_us:.1}us CPU-rank={host_us:.1}us",
                bytes
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible CUDA sm_120 inference GPU ordinal"]
    fn cuda_tensor_core_sparse_matches_scalar_and_rowwise_verification() -> Result<()> {
        unsafe extern "C" {
            fn cudaDeviceSynchronize() -> i32;
        }
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let sample = |i: usize| {
            let mut z = (i as u32).wrapping_mul(0x9e3779b9);
            z ^= z >> 15;
            z = z.wrapping_mul(0x735a2d97);
            b16(((z >> 16) as f32 - 32768.) / 24491.)
        };
        for (pos, n, width, span, decode) in [
            (0, 5, 0, 5, 0),
            (11, 3, 5, 128, 1),
            (260, 8, 128, 128, 0),
            (902, 8, 512, 128, 1),
            (902, 1, 512, 128, 1),
        ] {
            let prior: Vec<_> = (0..2 * WINDOW * DIM)
                .map(|i| bf16::from_f32(sample(i + 707)).to_bits())
                .collect();
            let current: Vec<_> = (0..n * DIM)
                .map(|i| bf16::from_f32(sample(i + 307)).to_bits())
                .collect();
            let queries: Vec<_> = (0..n * HEADS * DIM).map(|i| sample(i + 993)).collect();
            let sinks: Vec<_> = (0..HEADS).map(|i| sample(i + 107) * 3.).collect();
            let ids: Vec<_> = (0..n * width)
                .map(|i| {
                    let j = i % width;
                    if j % 19 == i / width {
                        -1
                    } else {
                        ((37 * j + i / width) % (width + 1)) as i32
                    }
                })
                .collect();
            let mut source = Rows::default();
            pack_values(
                &(0..(width + 1) * DIM)
                    .map(|i| sample(i + 223))
                    .collect::<Vec<_>>(),
                source.packed.for_append(),
            );
            let cache = source.on(device, None)?;
            let prior = upload(device, &prior)?;
            let current = upload(device, &current)?;
            let q = upload(device, &queries)?;
            let sinks = upload(device, &sinks)?;
            let ids = upload(device, &ids)?;
            let work = GpuBuffer::new(device)?;
            let out = upload(device, &vec![0f32; queries.len()])?;
            let run = |fast: bool, t: usize, rows: usize| -> Result<()> {
                let sparse = if fast {
                    ds4_attn_sparse_fast
                } else {
                    ds4_attn_sparse
                };
                cuda_ok(
                    unsafe {
                        sparse(
                            device,
                            work.ptr.as_ptr(),
                            q.data::<f32>().add(t * HEADS * DIM),
                            prior.data(),
                            current.data(),
                            pos,
                            t as i32,
                            span,
                            decode,
                            cache,
                            ids.data::<i32>().wrapping_add(t * width),
                            width as i32,
                            sinks.data(),
                            rows as i32,
                            out.data::<f32>().add(t * HEADS * DIM),
                        )
                    },
                    "sparse parity fixture",
                )
            };
            let mut expected = vec![0f32; queries.len()];
            let mut actual = expected.clone();
            run(false, 0, n)?;
            read(&out, &mut expected)?;
            run(true, 0, n)?;
            read(&out, &mut actual)?;
            let mut sq_err = 0f64;
            let mut sq_ref = 0f64;
            let mut max_err = 0f32;
            for (x, y) in actual.iter().zip(&expected) {
                assert!(x.is_finite());
                sq_err += f64::from(x - y).powi(2);
                sq_ref += f64::from(*y).powi(2);
                max_err = max_err.max((x - y).abs());
            }
            let relative = (sq_err / sq_ref).sqrt();
            assert!(
                relative < 0.006 && max_err < 0.015,
                "width {width}: relative {relative}, max {max_err}"
            );
            for t in 0..n {
                run(true, t, 1)?;
            }
            let mut split = vec![0f32; queries.len()];
            read(&out, &mut split)?;
            assert_eq!(
                split, actual,
                "verification rows changed with batch size at {pos}"
            );
            if std::env::var_os("DS4_BENCH_ATTN").is_some() {
                for fast in [false, true] {
                    let timed = std::time::Instant::now();
                    for _ in 0..60 {
                        run(fast, 0, n)?;
                    }
                    cuda_ok(
                        unsafe { cudaDeviceSynchronize() },
                        "timing attention fixture",
                    )?;
                    eprintln!(
                        "batch={n} width={width} fast={fast} us={:.2} relative_error={relative:.6} max_error={max_err:.6}",
                        timed.elapsed().as_secs_f64() * 1e6 / 60.
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible CUDA sm_120 inference GPU ordinal"]
    fn cuda_sparse_places_compressed_values_after_masked_window_columns() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let work = GpuBuffer::new(device)?;
        // Encode two raw window values and a third, exactly representable FP8
        // latent. Vary the head queries to exercise changes in the block max.
        let mut window = vec![0u16; 2 * DIM];
        for (pos, (k, v)) in [(1.5, -1.75), (1., 2.5)].into_iter().enumerate() {
            window[pos * DIM] = bf16::from_f32(k).to_bits();
            window[pos * DIM + 1] = bf16::from_f32(v).to_bits();
        }
        let mut key = vec![0.; DIM];
        key[0] = 2.;
        key[1] = -0.25;
        key[2] = 3.;
        let mut rows = Rows::default();
        pack_values(&key, rows.packed.for_append());
        let latent = rows.on(device, None)?;
        let mut query = vec![0.; HEADS * DIM];
        let sink = [-2.0f32; HEADS];
        for h in 0..HEADS {
            query[h * DIM] = b16((h + 1) as f32 * 0.75);
        }
        let mut actual = vec![0.; query.len()];
        let q_dev = upload(device, &query)?;
        let win_dev = upload(device, &window)?;
        let ids_dev = upload(device, &[0i32])?;
        let sink_dev = upload(device, &sink)?;
        let out = upload(device, &actual)?;
        let mut distinguished = 0;
        for decode in [0, 1] {
            cuda_ok(
                unsafe {
                    ds4_attn_sparse(
                        device,
                        work.ptr.as_ptr(),
                        q_dev.data(),
                        ptr::null(),
                        win_dev.data(),
                        0,
                        1,
                        128,
                        decode,
                        latent,
                        ids_dev.data(),
                        1,
                        sink_dev.data(),
                        1,
                        out.data(),
                    )
                },
                "padded window regression",
            )?;
            read(&out, &mut actual)?;
            for h in 0..HEADS {
                let q = query[h * DIM] / (DIM as f32).sqrt();
                let scores = [1.5 * q, q, 2. * q];
                let m = scores[2];
                let exp = scores.map(|s| (s - m).exp());
                let norm = exp.iter().sum::<f32>() + (-2. - m).exp();
                let f = (scores[0] - m).exp();
                let expected =
                    b16((-1.75 * f + 2.5 * b16((scores[1] - scores[0]).exp()) * f - 0.25) / norm);
                let compact = b16((-1.75 * b16(exp[0]) + 2.5 * b16(exp[1]) - 0.25) / norm);
                assert!(
                    (actual[h * DIM + 1] - expected).abs() <= 0.00001,
                    "head {h}: {} vs {expected}",
                    actual[h * DIM + 1]
                );
                if expected != compact {
                    distinguished += 1;
                }
            }
        }
        assert!(distinguished > 0);
        Ok(())
    }
}
