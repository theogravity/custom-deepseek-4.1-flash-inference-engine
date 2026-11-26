//! DeepSeek V4.1 Flash latent attention. The checkpoint shares compressed KV and
//! index keys at layers 2/8/14/20, while every backbone layer owns its window.
#[path = "attention_snapshot.rs"]
mod snapshot;

use crate::runtime::{Dev, Runtime};
use anyhow::{ensure, Context, Result};
use half::bf16;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::OnceLock;

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
    #[cfg(test)]
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

    fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        if !data.is_empty() {
            cuda_ok(
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

fn cuda_ok(code: i32, op: &str) -> Result<()> {
    ensure!(code == 0, "CUDA {op} failed with error {code}");
    Ok(())
}

#[derive(Default)]
struct Rows {
    packed: Vec<u8>,
    devices: HashMap<i32, (GpuBuffer, usize)>,
}
impl Rows {
    fn truncate(&mut self, bytes: usize) {
        self.packed.truncate(bytes);
        for (_, uploaded) in self.devices.values_mut() {
            *uploaded = (*uploaded).min(bytes);
        }
    }

    fn on(&mut self, dev: i32) -> Result<*mut c_void> {
        use std::collections::hash_map::Entry;
        let (buf, old) = match self.devices.entry(dev) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert((GpuBuffer::new(dev)?, 0)),
        };
        let extra = &self.packed[*old..];
        if !extra.is_empty() {
            cuda_ok(
                unsafe {
                    ds4_attn_upload(dev, buf.ptr.as_ptr(), extra.as_ptr(), *old, extra.len())
                },
                "uploading compressed attention rows",
            )?;
            *old = self.packed.len();
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

#[derive(Default)]
struct Window {
    rows: Vec<u16>, // Chronological history in a host snapshot.
    ring: Option<GpuBuffer>,
    next_pos: usize,
}

// Copies select a CUDA memcpy direction. The native helper splits wraparound.
const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;
const DEVICE_TO_DEVICE: i32 = 3;
impl Window {
    fn on(&mut self, dev: i32) -> Result<&GpuBuffer> {
        if self.ring.is_none() {
            let ring = GpuBuffer::new(dev)?;
            let rows = self.next_pos.min(WINDOW - 1);
            ensure!(
                self.rows.len() == rows * DIM,
                "invalid restored attention window"
            );
            cuda_ok(
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
            self.rows = Vec::new();
        }
        let ring = self.ring.as_ref().unwrap();
        ensure!(ring.device == dev, "attention window device changed");
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
            .chunks_exact(DIM)
            .zip(gates.chunks_exact(DIM))
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
    prompt_len: usize,
    verify: Option<Verification>,
    windows: HashMap<usize, Window>,
    sources: HashMap<usize, Source>,
    scratch: HashMap<i32, GpuBuffer>,
    selected: Vec<Vec<i32>>,
    picks: HashMap<i32, GpuBuffer>,
    width: usize,
    freqs: HashMap<(i32, bool), (usize, usize, Dev)>,
    blocks: Vec<Vec<usize>>,
}

impl AttentionState {
    /// Start a speculative target forward. The device ring retains the prior
    /// window and every tentative row until the caller commits an accepted prefix.
    pub fn begin_verify(&mut self, pos: usize, batch: usize) -> Result<()> {
        ensure!(
            self.verify.is_none(),
            "attention verification already active"
        );
        ensure!(
            (1..=WINDOW).contains(&batch) && pos.checked_add(batch).is_some_and(|n| n <= MAX_POS),
            "invalid attention verification block"
        );
        ensure!(
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
            .context("no active attention verification")?;
        ensure!(
            accepted <= mark.batch,
            "accepted prefix exceeds verified block"
        );
        let verified = mark.pos + mark.batch;
        ensure!(
            self.windows.len() == 40 && self.windows.values().all(|w| w.next_pos == verified),
            "incomplete verification forward"
        );
        if accepted < mark.batch {
            let next = mark.pos + accepted;
            for (&owner, src) in &mut self.sources {
                let (_, ratio) = source(owner).context("unknown compressed source")?;
                if ratio == 2 && next % 2 == 1 {
                    let pairs = mark
                        .pending
                        .get(&owner)
                        .context("missing verified compressor rows")?;
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

    /// Set the original prompt length once, including when restoring a prefix.
    /// All prefill chunks then share the reference's window-column layout.
    pub fn begin_prefill(&mut self, prompt_len: usize) {
        self.prompt_len = prompt_len;
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
    pub fn host_snapshot(&self) -> Result<Self> {
        ensure!(self.verify.is_none(), "cannot snapshot active verification");
        fn host_rows(rows: &Rows) -> Rows {
            Rows {
                packed: rows.packed.clone(),
                devices: HashMap::new(),
            }
        }
        let sources = self
            .sources
            .iter()
            .map(|(&layer, source)| {
                (
                    layer,
                    Source {
                        values: host_rows(&source.values),
                        keys: host_rows(&source.keys),
                        pending_kv: source.pending_kv.clone(),
                        pending_gate: source.pending_gate.clone(),
                    },
                )
            })
            .collect();
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
    if let Some((p, len, freqs)) = state.freqs.get(&key) {
        if (*p, *len) == (pos, n) {
            return Ok(freqs.as_ptr());
        }
    }
    let host: Vec<_> = (pos..pos + n)
        .map(|p| rotary_freqs(p, compressed))
        .collect();
    let freqs = rt.to_device(host.as_flattened().as_flattened())?;
    let ptr = freqs.as_ptr();
    state.freqs.insert(key, (pos, n, freqs));
    Ok(ptr)
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
) -> Result<()> {
    ensure!(
        src.keys.packed.len() == pos / ratio * INDEX_BYTES
            && src.values.packed.len() == pos / ratio * KV_BYTES,
        "discontinuous compressed cache at {pos}"
    );
    let kv = project(rt, prefix, "compressor.wkv", x, batch, DIM, 1)?;
    let kv = rt.to_host(&kv)?;
    let mut latent = if ratio == 1 {
        kv
    } else {
        let gates = project(rt, prefix, "compressor.wgate", x, batch, DIM, 1)?;
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
    pack_index_keys(&keys, &mut src.keys.packed);
    pack_values(&latent, &mut src.values.packed);
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

fn scratch(state: &mut AttentionState, dev: i32) -> Result<*mut c_void> {
    use std::collections::hash_map::Entry;
    let buf = match state.scratch.entry(dev) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => e.insert(GpuBuffer::new(dev)?),
    };
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
    if layer == CANDIDATE_LAYER {
        state.blocks.clear();
    }
    if max_len <= TOP {
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
        .context("missing index key source")?
        .keys
        .on(rt.device)?;
    let work = scratch(state, rt.device)?;
    // Tile host scores to keep long-context prefill scratch bounded. After layer
    // 20, only the 2048 winning eight-token blocks need scoring.
    let filtered_width = TOP_BLOCKS * BLOCK;
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

/// Process one contiguous prompt chunk or one decode token in token-major order.
/// A new request owns a fresh state. Layers are invoked in model order.
pub fn forward_dev(
    rt: &mut Runtime,
    state: &mut AttentionState,
    layer: usize,
    pos: usize,
    x: &Dev,
    batch: usize,
) -> Result<Dev> {
    ensure!(
        layer < 40 && batch > 0 && pos.checked_add(batch).is_some_and(|end| end <= MAX_POS),
        "invalid backbone attention layer or position"
    );
    ensure!(
        x.len() == batch * HIDDEN,
        "attention input has incorrect shape"
    );
    if let Some(mark) = &state.verify {
        ensure!(
            mark.pos == pos && mark.batch == batch,
            "attention verification block changed"
        );
    }
    let prior = state.windows.get(&layer);
    ensure!(
        prior.map_or(0, |w| w.next_pos) == pos,
        "non-contiguous attention window at layer {layer}, position {pos}"
    );
    let prefix = format!("layers.{layer}.attn");
    let group = source(layer);
    let freqs = device_freqs(rt, state, pos, batch, group.is_some())?;
    let mut qr = project(rt, &prefix, "wq_a", x, batch, 1280, 1)?;
    let gamma = rt.vector_ptr(&format!("{prefix}.q_norm.weight"), 1280)?;
    cuda_ok(
        unsafe { ds4_attn_normalize(rt.device, qr.as_mut_ptr(), gamma, 1280, batch as i32) },
        "normalizing attention query projection",
    )?;
    let mut q = project(rt, &prefix, "wq_b", &qr, batch, HEADS * DIM, 1)?;
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                q.as_mut_ptr(),
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
    let mut kv = project(rt, &prefix, "wkv", x, batch, DIM, 1)?;
    let gamma = rt.vector_ptr(&format!("{prefix}.kv_norm.weight"), DIM)?;
    cuda_ok(
        unsafe { ds4_attn_normalize(rt.device, kv.as_mut_ptr(), gamma, DIM as i32, batch as i32) },
        "normalizing window keys",
    )?;
    // Storage is packed BF16, with two values in each pooled four-byte element.
    let mut current = rt.buffer(batch * DIM / 2)?;
    cuda_ok(
        unsafe {
            ds4_attn_transform(
                rt.device,
                kv.as_mut_ptr(),
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
    drop(kv);

    let mut width = 0;
    let mut ids: *const i32 = ptr::null();
    let mut values = ptr::null_mut();
    if let Some((owner, ratio)) = group {
        if layer == owner {
            append_source(
                rt,
                state.sources.entry(owner).or_default(),
                &prefix,
                x,
                pos,
                batch,
                ratio,
                owner,
                state.verify.as_mut(),
            )?;
        }
        if layer == owner || [24, 28, 32, 36].contains(&layer) {
            select(
                rt, state, owner, ratio, &prefix, layer, x, &qr, pos, batch, freqs,
            )?;
            ensure!(
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
                picks.write(&flat)?;
            }
        }
        width = state.width;
        if width != 0 {
            ids = state
                .picks
                .get(&rt.device)
                .context("missing selected attention indices")?
                .data();
            values = state
                .sources
                .get_mut(&owner)
                .context("missing compressed attention source")?
                .values
                .on(rt.device)?;
        }
    }
    drop(qr);
    let sink = rt.vector_ptr(&format!("{prefix}.attn_sink"), HEADS)?;
    let work = scratch(state, rt.device)?;
    let (span, decode) = state.window_layout(pos, batch);
    let window = state.windows.entry(layer).or_default();
    let ring = window.on(rt.device)?;
    let mut o = rt.buffer(q.len())?;
    // Bound temporary expanded compressed values to 16 MiB per call.
    let step = (16 * 1024 * 1024 / (width.max(1) * DIM * 2)).max(1);
    for t in (0..batch).step_by(step) {
        let n = step.min(batch - t);
        cuda_ok(
            unsafe {
                ds4_attn_sparse(
                    rt.device,
                    work,
                    q.as_ptr().add(t * HEADS * DIM),
                    ring.data(),
                    current.as_ptr().cast(),
                    pos as i32,
                    t as i32,
                    span as i32,
                    i32::from(decode),
                    values,
                    ids.wrapping_add(t * width),
                    width as i32,
                    sink,
                    n as i32,
                    o.as_mut_ptr().add(t * HEADS * DIM),
                )
            },
            "sparse attention",
        )?;
    }
    drop(q);
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
                o.as_mut_ptr(),
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
    let down = project(rt, &prefix, "wo_a", &o, batch, 8 * 1024, 8)?;
    drop(o);
    let out = project(rt, &prefix, "wo_b", &down, batch, HIDDEN, 1)?;
    window.next_pos = pos + batch;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        source.keys.packed = vec![1; INDEX_BYTES];
        source.values.packed = vec![2; KV_BYTES];
        source.pending_kv = vec![1.0; DIM];
        source.pending_gate = vec![0.0; DIM];
        original.windows.insert(
            2,
            Window {
                rows: vec![17; DIM * 3],
                ring: None,
                next_pos: 3,
            },
        );
        original.selected = vec![vec![0]];
        original.blocks = vec![vec![0]];
        let mut branch = original.host_snapshot().unwrap();
        assert!(
            branch.scratch.is_empty() && branch.blocks.is_empty() && branch.selected.is_empty()
        );
        assert_eq!(branch.windows[&2].next_pos, 3);
        assert_eq!(branch.windows[&2].rows, original.windows[&2].rows);
        let copied = branch.sources.get_mut(&2).unwrap();
        assert!(copied.keys.devices.is_empty() && copied.values.devices.is_empty());
        assert_eq!(copied.keys.packed, original.sources[&2].keys.packed);
        assert_eq!(copied.values.packed, original.sources[&2].values.packed);
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
                src.keys.packed.resize((pos + n) / 2 * INDEX_BYTES, 1);
                src.values.packed.resize((pos + n) / 2 * KV_BYTES, 2);
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
        let batch = WINDOW;
        let rows: Vec<_> = (0..(pos + batch) * DIM)
            .map(|i| (i * 613 % 65521) as u16)
            .collect();
        let mut window = Window {
            rows: rows[(pos - WINDOW + 1) * DIM..pos * DIM].to_vec(),
            ring: None,
            next_pos: pos,
        };
        let new = upload(device, &rows[pos * DIM..])?;
        let ring = window.on(device)?;
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
        pack_values(&key, &mut rows.packed);
        let latent = rows.on(device)?;
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
