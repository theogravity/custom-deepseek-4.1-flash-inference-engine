//! DeepSeek V4.1 Flash text MoE. Weights stay mmap-backed on the host until an
//! expert is actually selected. Each GPU independently selects reusable slots.
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::{
    collections::{BTreeSet, HashMap},
    ffi::{c_char, c_void, CStr},
    ptr,
};

use crate::{
    runtime::{Dev, Runtime},
    weights::Weights,
};

const HIDDEN: usize = 5120;
const INTER: usize = 2304;
const EXPERTS: usize = 384;
const TOP_K: usize = 6;
const DRAFT_EXPERTS: usize = 128;
const DRAFT_TOP_K: usize = 3;
const SLOT_BYTES: usize = 3 * (HIDDEN * INTER * 3 / 8 + 2 * (HIDDEN + INTER));
#[cfg(feature = "recipe-prefill")]
mod tp2;
#[cfg(feature = "recipe-prefill")]
const RANK_SLOT_BYTES: usize = 3 * (HIDDEN * (INTER / 2) * 3 / 8 + 2 * (HIDDEN + INTER / 2));
const MAX_GROUP_ROWS: usize = 256;
const MAX_GROUP_EXPERTS: usize = 64;

#[repr(C)]
struct Job {
    expert: usize,
    begin: i32,
    count: i32,
}

#[repr(C)]
struct RoutedRow {
    expert: usize,
    token: i32,
    output: i32,
    weight: f32,
}

unsafe extern "C" {
    fn ds_moe_error(error: i32) -> *const c_char;
    fn ds_moe_free_bytes(device: i32, bytes: *mut usize) -> i32;
    fn cudaPeekAtLastError() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaMalloc(slot: *mut *mut c_void, bytes: usize) -> i32;
    fn ds_moe_upload_parts(slot: *mut c_void, parts: *const *const c_void, coalesce: i32) -> i32;
    fn ds_moe_router_start(
        host: *mut *mut f32,
        capacity: *mut usize,
        ready: *mut *mut c_void,
        logits: *const f32,
        count: usize,
    ) -> i32;
    fn ds_moe_router_wait(ready: *mut c_void) -> i32;
    fn ds_moe_router_free(device: i32, host: *mut f32, ready: *mut c_void);

    fn ds_moe_free(device: i32, slot: *mut c_void);
    fn cudaSetDevice(device: i32) -> i32;
    fn ds_moe_peer_begin(link: *mut *mut c_void, primary: i32, peer: i32) -> i32;
    fn ds_moe_peer_end(link: *mut c_void) -> i32;
    fn ds_moe_peer_free(link: *mut c_void);
    fn ds_moe_peer_capacity(link: *mut c_void) -> usize;
    fn ds_moe_peer_stage(
        link: *mut c_void,
        x: *const f32,
        nx: usize,
        rows: usize,
        peer_x: *mut *mut f32,
        peer_output: *mut *mut f32,
    ) -> i32;
    fn ds_moe_peer_collect(
        link: *mut c_void,
        peer_output: *const f32,
        compact: *mut f32,
        destinations: *const i32,
        rows: i32,
        weighted: *mut f32,
    ) -> i32;
    fn ds_moe_shared_swiglu(gate: *mut f32, up: *const f32, count: usize) -> i32;
    fn ds_moe_finish(routed: *const f32, shared_out: *mut f32, count: usize, top_k: i32) -> i32;
    fn ds_moe_run_dev(
        scratch: *mut *mut c_void,
        scratch_bytes: *mut usize,
        jobs: *const Job,
        n_jobs: i32,
        routes: *const RoutedRow,
        rows: i32,
        x: *const f32,
        weighted: *mut f32,
        mcg: i32,
        splits: i32,
        inline_metadata: i32,
        fused: i32,
    ) -> i32;
    #[cfg(test)]
    fn ds_moe_run(
        scratch: *mut *mut c_void,
        scratch_bytes: *mut usize,
        jobs: *const Job,
        n_jobs: i32,
        experts: *const usize,
        x: *const f32,
        y: *mut f32,
        rows: i32,
    ) -> i32;
}

fn cuda(code: i32) -> Result<()> {
    if code != 0 {
        // cudaErrorMemoryAllocation is 2 in the CUDA runtime ABI. An exhausted
        // request may retire its state and retry on this same worker thread.
        // Leave unrelated asynchronous errors available to their own caller.
        const CUDA_ERROR_MEMORY_ALLOCATION: i32 = 2;
        if code == CUDA_ERROR_MEMORY_ALLOCATION && unsafe { cudaPeekAtLastError() } == code {
            unsafe { cudaGetLastError() };
        }
        let message = unsafe { CStr::from_ptr(ds_moe_error(code)) }.to_string_lossy();
        bail!("CUDA MoE error {code}: {message}");
    }
    Ok(())
}

#[derive(Default)]
pub struct MoeCache {
    pub(crate) reclaimable: HashMap<i32, std::sync::Weak<std::sync::Mutex<crate::runtime::Pool>>>,
    devices: HashMap<i32, DeviceCache>,
    metrics: Metrics,
}

#[derive(Default)]
struct Metrics {
    routes: usize,
    unique: usize,
    uploads: usize,
    capacity_queries: usize,
}

impl MoeCache {
    fn device(&mut self, device: i32) -> Result<&mut DeviceCache> {
        Ok(match self.devices.entry(device) {
            std::collections::hash_map::Entry::Occupied(v) => v.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => v.insert(DeviceCache::new(device)?),
        })
    }
    /// Make room before an uncached dense projection uploads its resident
    /// weight and conversion buffers. Freeing a slot waits for pending uses.
    pub fn reserve(&mut self, device: i32, bytes: usize) -> Result<()> {
        let pool = self.reclaimable.get(&device).and_then(|p| p.upgrade());
        let cache = self.devices.get_mut(&device);
        if pool.is_none() && cache.as_ref().is_none_or(|c| c.slots.is_empty()) {
            return Ok(());
        }
        let mut free = 0;
        cuda(unsafe { ds_moe_free_bytes(device, &mut free) })?;
        let missing = bytes.saturating_add(64 << 20).saturating_sub(free);
        let mut released = 0;
        if missing > 0 {
            if let Some(pool) = pool {
                released = pool
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .reclaim(missing);
            }
        }
        let Some(cache) = cache else {
            return Ok(());
        };
        while released < missing && !cache.slots.is_empty() {
            let (slot, size) = cache.evict_slot(u64::MAX)?;
            released += size;
            unsafe { ds_moe_free(device, slot as *mut c_void) };
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Layer {
    Backbone(usize),
    Draft(usize),
}

impl Layer {
    fn full(self, expert: usize) -> Key {
        Key(self, expert, SlotFormat::Full)
    }
    fn base(self) -> String {
        match self {
            Self::Backbone(n) => format!("layers.{n}.ffn"),
            Self::Draft(n) => format!("mtp.{n}.ffn"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SlotFormat {
    Full,
    #[cfg(feature = "recipe-prefill")]
    Rank(u8),
}
impl SlotFormat {
    fn bytes(self) -> usize {
        match self {
            Self::Full => SLOT_BYTES,
            #[cfg(feature = "recipe-prefill")]
            Self::Rank(_) => RANK_SLOT_BYTES,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Key(Layer, usize, SlotFormat);
const EXPERT_KEYS: usize = 40 * EXPERTS + 3 * DRAFT_EXPERTS;

fn frequency_index(Key(layer, expert, _): Key) -> usize {
    match layer {
        Layer::Backbone(n) => n * EXPERTS + expert,
        Layer::Draft(n) => 40 * EXPERTS + n * DRAFT_EXPERTS + expert,
    }
}
pub(crate) fn recipe_prefill_enabled() -> Result<bool> {
    let enabled = switch("DS4_RECIPE_PREFILL", false)?;
    ensure!(
        !enabled || cfg!(feature = "recipe-prefill"),
        "DS4_RECIPE_PREFILL requires the recipe-prefill build feature"
    );
    Ok(enabled)
}

pub(crate) fn tp2_prefill_enabled() -> Result<bool> {
    let enabled = switch("DS4_TP2_PREFILL", false)?;
    ensure!(
        !enabled || cfg!(feature = "recipe-prefill"),
        "DS4_TP2_PREFILL requires the recipe-prefill build feature"
    );
    Ok(enabled)
}
fn switch(name: &str, default: bool) -> Result<bool> {
    match std::env::var(name).as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok("0") => Ok(false),
        Ok("1") => Ok(true),
        _ => bail!("{name} must be 0 or 1"),
    }
}

#[derive(Default)]
struct RouterRead {
    host: *mut f32,
    capacity: usize,
    ready: *mut c_void,
}

impl RouterRead {
    fn start(&mut self, logits: &Dev) -> Result<()> {
        cuda(unsafe {
            ds_moe_router_start(
                &mut self.host,
                &mut self.capacity,
                &mut self.ready,
                logits.as_ptr(),
                logits.len(),
            )
        })
    }

    fn scores(&self, len: usize) -> Result<&[f32]> {
        ensure!(
            len <= self.capacity && !self.host.is_null(),
            "uninitialized router read"
        );
        cuda(unsafe { ds_moe_router_wait(self.ready) })?;
        Ok(unsafe { std::slice::from_raw_parts(self.host, len) })
    }
}

// Keeps the peer's default stream ordered with the primary activation pool.
// Finish the join on every error path, before the borrowed GPU tensors drop.
struct PeerWork(*mut c_void);
impl PeerWork {
    fn begin(cache: &mut DeviceCache, peer: i32) -> Result<Self> {
        cuda(unsafe { ds_moe_peer_begin(&mut cache.peer_link, cache.device, peer) })
            .context("enabling CUDA peer access for expert parallelism")?;
        Ok(Self(cache.peer_link))
    }

    fn finish(mut self) -> Result<()> {
        let link = std::mem::replace(&mut self.0, ptr::null_mut());
        cuda(unsafe { ds_moe_peer_end(link) })
    }
}
impl Drop for PeerWork {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ds_moe_peer_end(self.0) };
        }
    }
}

struct DeviceCache {
    device: i32,
    limit: usize,
    slots: HashMap<Key, (usize, u64)>,
    resident_bytes: usize,
    lru: BTreeSet<(u64, Key)>,
    frequencies: Option<Box<[u64]>>,
    clock: u64,
    scratch: usize,
    scratch_bytes: usize,
    router: RouterRead,
    uploads: usize,
    overlap_router: bool,
    parallel_prefill_router: bool,
    cached_capacity: bool,
    packed_upload: bool,
    inline_metadata: bool,
    recipe_prefill: bool,
    #[cfg(feature = "recipe-prefill")]
    tp2_prefill: bool,
    #[cfg(feature = "recipe-prefill")]
    tp2_packed_input: bool,
    #[cfg(feature = "recipe-prefill")]
    tp2_peer_dma: bool,
    #[cfg(feature = "recipe-prefill")]
    stage_rank_weights: bool,
    #[cfg(feature = "recipe-prefill")]
    upload_workspace: *mut c_void,
    #[cfg(feature = "recipe-prefill")]
    rank_workspace: tp2::Workspace,
    fused: bool,
    packed_decode: bool,
    parallel: bool,
    parallel_draft: bool,
    staged: bool,
    peer_link: *mut c_void,
    metrics: bool,
}

impl DeviceCache {
    fn new(device: i32) -> Result<Self> {
        let gib: f64 = std::env::var("DS4_MOE_CACHE_GIB")
            .unwrap_or_else(|_| "78".to_owned())
            .parse()
            .context("DS4_MOE_CACHE_GIB must be a number")?;
        ensure!(gib.is_finite() && gib >= 0.0, "invalid DS4_MOE_CACHE_GIB");
        let limit = (((gib * 1073741824.0) as usize) / SLOT_BYTES).max(1);
        let recipe_prefill = recipe_prefill_enabled()?;
        Ok(Self {
            recipe_prefill,
            #[cfg(feature = "recipe-prefill")]
            tp2_prefill: tp2_prefill_enabled()?,
            #[cfg(feature = "recipe-prefill")]
            tp2_packed_input: switch("DS4_TP2_PACKED_INPUT", false)?,
            #[cfg(feature = "recipe-prefill")]
            stage_rank_weights: switch("DS4_TP2_STAGE_WEIGHTS", false)?,
            #[cfg(feature = "recipe-prefill")]
            upload_workspace: ptr::null_mut(),
            #[cfg(feature = "recipe-prefill")]
            tp2_peer_dma: switch("DS4_TP2_PEER_DMA", false)?,
            #[cfg(feature = "recipe-prefill")]
            rank_workspace: tp2::Workspace::default(),
            device,
            limit,
            slots: HashMap::new(),
            resident_bytes: 0,
            lru: BTreeSet::new(),
            frequencies: match std::env::var("DS4_EXPERT_EVICTION").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("lru") => None,
                Ok("lfu") => Some(vec![0; EXPERT_KEYS].into_boxed_slice()),
                _ => bail!("DS4_EXPERT_EVICTION must be lru or lfu"),
            },
            clock: 0,
            scratch: 0,
            scratch_bytes: 0,
            router: RouterRead::default(),
            uploads: 0,
            parallel_prefill_router: switch("DS4_PARALLEL_PREFILL_ROUTER", false)?,
            overlap_router: switch("DS4_MOE_OVERLAP_ROUTER", true)?,
            cached_capacity: switch("DS4_MOE_CACHED_CAPACITY", true)?,
            packed_upload: switch("DS4_MOE_PACKED_UPLOAD", true)?,
            inline_metadata: switch("DS4_MOE_INLINE_METADATA", true)?,
            fused: switch("DS4_MOE_FUSED", false)?,
            packed_decode: switch("DS4_MOE_PACKED_DECODE", false)?,
            parallel: switch("DS4_EXPERT_PARALLEL", false)?,
            parallel_draft: switch("DS4_EXPERT_PARALLEL_DRAFT", false)?,
            staged: switch("DS4_EXPERT_STAGED", true)?,
            peer_link: ptr::null_mut(),
            metrics: switch("DS4_MOE_STATS", false)?,
        })
    }

    fn evict_slot(&mut self, group_start: u64) -> Result<(usize, usize)> {
        let key = if let Some(frequencies) = &self.frequencies {
            // Already acquired slots are still referenced by this physical CUDA group.
            self.slots
                .iter()
                .filter(|(_, (_, last))| *last <= group_start)
                .min_by_key(|(key, (_, last))| (frequencies[frequency_index(**key)], *last))
                .map(|(key, _)| *key)
                .context("no evictable expert slot")?
        } else {
            let (age, _) = self.lru.first().context("empty expert cache")?;
            ensure!(*age <= group_start, "no evictable expert slot");
            self.lru.pop_first().unwrap().1
        };
        let bytes = key.2.bytes();
        self.resident_bytes -= bytes;
        Ok((
            self.slots.remove(&key).context("missing LRU slot")?.0,
            bytes,
        ))
    }

    fn available_slots(
        resident_bytes: usize,
        free: usize,
        limit: usize,
        slot_bytes: usize,
    ) -> usize {
        let available = (resident_bytes + free).saturating_sub(512 << 20);
        (available.min(limit * SLOT_BYTES) / slot_bytes).max(1)
    }

    fn capacity(&mut self, slot_bytes: usize) -> Result<usize> {
        let mut free = 0;
        cuda(unsafe { ds_moe_free_bytes(self.device, &mut free) })?;
        // Leave room for scratch, dense projections and the CUDA context. If
        // another allocation consumes the reserve, later groups shrink again.
        let capacity = Self::available_slots(self.resident_bytes, free, self.limit, slot_bytes);
        while self.resident_bytes > capacity * slot_bytes {
            let (slot, _) = self.evict_slot(u64::MAX)?;
            unsafe { ds_moe_free(self.device, slot as *mut c_void) };
        }
        Ok(capacity)
    }

    fn can_reuse_capacity(&self, layer: Layer, routes: &[Route]) -> bool {
        // Even in the most expensive split (8) and with one job per row, this
        // group needs no new device allocation. External VRAM consumers cannot
        // change that. A miss or larger scratch request uses a fresh query.
        let max_rows = routes.len().min(MAX_GROUP_ROWS);
        let row_bytes = (8 * HIDDEN + 2 * INTER) * size_of::<f32>()
            + HIDDEN * size_of::<half::f16>()
            + size_of::<Job>()
            + size_of::<RoutedRow>();
        self.cached_capacity
            && self.scratch_bytes >= max_rows * row_bytes
            && routes
                .chunk_by(|a, b| a.expert == b.expert)
                .all(|group| self.slots.contains_key(&layer.full(group[0].expert)))
    }

    fn touch(&mut self, key: Key, stamp: u64) -> Option<usize> {
        let (slot, age) = self.slots.get_mut(&key)?;
        if let Some(frequencies) = &mut self.frequencies {
            let frequency = &mut frequencies[frequency_index(key)];
            *frequency = frequency.saturating_add(1);
        } else {
            self.lru.remove(&(*age, key));
            self.lru.insert((stamp, key));
        }
        *age = stamp;
        Some(*slot)
    }

    fn acquire(
        &mut self,
        weights: &Weights,
        key: Key,
        capacity: usize,
        group_start: u64,
    ) -> Result<usize> {
        self.clock += 1;
        let stamp = self.clock;
        if let Some(slot) = self.touch(key, stamp) {
            return Ok(slot);
        }
        // Check all sizes before passing mmap pointers over FFI.
        let parts = checkpoint_parts(weights, key)?;
        let slot_bytes = key.2.bytes();
        let mut reuse = None;
        while self.resident_bytes + slot_bytes > capacity * slot_bytes {
            let (old, bytes) = self.evict_slot(group_start)?;
            if bytes == slot_bytes && reuse.is_none() {
                reuse = Some(old);
            } else {
                unsafe { ds_moe_free(self.device, old as *mut c_void) };
            }
        }
        let slot = if let Some(slot) = reuse {
            slot as *mut c_void
        } else {
            let mut slot = ptr::null_mut();
            cuda(unsafe { cudaMalloc(&mut slot, slot_bytes) })?;
            slot
        };
        let status = match key.2 {
            #[cfg(feature = "recipe-prefill")]
            SlotFormat::Rank(rank) => unsafe {
                if self.stage_rank_weights {
                    tp2::ds_recipe_staged_rank(
                        &mut self.upload_workspace,
                        slot,
                        slot_bytes,
                        parts.as_ptr(),
                        rank.into(),
                    )
                } else {
                    tp2::ds_recipe_upload_rank(slot, slot_bytes, parts.as_ptr(), rank.into())
                }
            },
            SlotFormat::Full => unsafe {
                ds_moe_upload_parts(slot, parts.as_ptr(), self.packed_upload.into())
            },
        };
        if let Err(e) = cuda(status) {
            unsafe { ds_moe_free(self.device, slot) };
            return Err(e);
        }
        let slot = slot as usize;
        self.resident_bytes += slot_bytes;
        self.uploads += 1;
        self.slots.insert(key, (slot, stamp));
        if let Some(frequencies) = &mut self.frequencies {
            let frequency = &mut frequencies[frequency_index(key)];
            *frequency = frequency.saturating_add(1);
        } else {
            self.lru.insert((stamp, key));
        }
        Ok(slot)
    }
}

impl Drop for DeviceCache {
    fn drop(&mut self) {
        #[cfg(feature = "recipe-prefill")]
        unsafe {
            tp2::ds_recipe_free_uploads(self.device, self.upload_workspace);
        }

        #[cfg(feature = "recipe-prefill")]
        self.rank_workspace.free(self.device);
        unsafe { ds_moe_peer_free(self.peer_link) };
        unsafe { ds_moe_router_free(self.device, self.router.host, self.router.ready) };
        for &(slot, _) in self.slots.values() {
            unsafe { ds_moe_free(self.device, slot as *mut c_void) };
        }
        unsafe { ds_moe_free(self.device, self.scratch as *mut c_void) };
    }
}

fn checkpoint_parts(weights: &Weights, Key(layer, expert, _): Key) -> Result<[*const c_void; 9]> {
    let mut parts = [ptr::null(); 9];
    let base = layer.base();
    let (codebook, multiplier) = match layer {
        Layer::Backbone(_) => ("mul1", 0x83dc_d12du32),
        Layer::Draft(_) => ("mcg", 0xcbac_1fedu32),
    };
    for (p, (name, k, n)) in [
        ("w1", HIDDEN, INTER),
        ("w3", HIDDEN, INTER),
        ("w2", INTER, HIDDEN),
    ]
    .into_iter()
    .enumerate()
    {
        let key = format!("{base}.experts.{expert}.{name}");
        let mul = weights.tensor(&format!("{key}.{codebook}"))?;
        ensure!(
            mul.dtype == "I32" && mul.data == multiplier.to_le_bytes(),
            "{key}: expected {codebook} multiplier"
        );
        let expected_tiles = [k / 16, n / 16, 48];
        for (suffix, dtype, shape, slot, bytes) in [
            (
                "trellis",
                "I16",
                expected_tiles.as_slice(),
                p,
                k * n * 3 / 8,
            ),
            ("suh", "F16", [k].as_slice(), 3 + p * 2, 2 * k),
            ("svh", "F16", [n].as_slice(), 4 + p * 2, 2 * n),
        ] {
            let t = weights.tensor(&format!("{key}.{suffix}"))?;
            ensure!(
                t.shape == shape && t.dtype == dtype && t.data.len() == bytes,
                "{key}.{suffix}: unexpected layout"
            );
            parts[slot] = t.data.as_ptr().cast();
        }
    }
    Ok(parts)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Route {
    expert: usize,
    token: usize,
    rank: usize,
    weight: f32,
}

fn routes<const E: usize, const K: usize>(
    logits: &[f32],
    bias: &[f32],
    batch: usize,
    parallel: bool,
) -> Result<Vec<Route>> {
    ensure!(
        logits.len() == batch * E && bias.len() == E,
        "router shape mismatch"
    );
    let mut output = vec![Route::default(); batch * K];
    let assign = |(token, (scores, dest)): (usize, (&[f32; E], &mut [Route; K]))| {
        let mut top = [(f32::NEG_INFINITY, 0.0, 0); K];
        for (expert, &logit) in scores.iter().enumerate() {
            let score = (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt();
            let corrected = score + bias[expert];
            if corrected > top[K - 1].0 {
                let mut i = K - 1;
                while i != 0 && corrected > top[i - 1].0 {
                    top[i] = top[i - 1];
                    i -= 1;
                }
                top[i] = (corrected, score, expert);
            }
        }
        let norm = top.iter().map(|t| t.1).sum::<f32>() + 1e-20;
        for (route, (_, score, expert)) in dest.iter_mut().zip(top) {
            *route = Route {
                expert,
                token,
                rank: 0,
                weight: 1.5 * score / norm,
            };
        }
    };
    let scores = logits.as_chunks::<E>().0;
    let dest = output.as_chunks_mut::<K>().0;
    if parallel && batch >= 32 {
        scores
            .par_iter()
            .zip(dest.par_iter_mut())
            .enumerate()
            .with_min_len(8)
            .for_each(assign);
    } else {
        scores
            .iter()
            .zip(dest.iter_mut())
            .enumerate()
            .for_each(assign);
    }
    output.sort_unstable_by_key(|r| (r.expert, r.token));
    // Match the original expert-major sequential accumulation, including when
    // a token's routes span multiple groups or its expert slot is evicted.
    let mut counts = vec![0; batch];
    for route in &mut output {
        route.rank = counts[route.token];
        counts[route.token] += 1;
    }
    Ok(output)
}

fn shared(rt: &mut Runtime, base: &str, x: &Dev, batch: usize) -> Result<Dev> {
    let mut gate = rt.linear_dev(&format!("{base}.shared_experts.w1.weight"), x, batch, 1)?;
    let up = rt.linear_dev(&format!("{base}.shared_experts.w3.weight"), x, batch, 1)?;
    ensure!(
        gate.len() == batch * INTER && gate.len() == up.len(),
        "shared expert shape mismatch"
    );
    cuda(unsafe { ds_moe_shared_swiglu(gate.as_mut_ptr(), up.as_ptr(), gate.len()) })?;
    rt.linear_dev(&format!("{base}.shared_experts.w2.weight"), &gate, batch, 1)
}

#[derive(Clone, Copy)]
struct PeerBuffers {
    x: *mut f32,
    output: *mut f32,
}

struct Dispatch<'a> {
    weights: &'a Weights,
    layer: Layer,
    x: &'a Dev,
    weighted: &'a mut Dev,
    top_k: usize,
    recipe_eligible: bool,
}

impl Dispatch<'_> {
    fn run_group(
        &mut self,
        device: &mut DeviceCache,
        routes: &[Route],
        capacity: usize,
        splits: i32,
        staged: Option<PeerBuffers>,
    ) -> Result<()> {
        let mut jobs = Vec::new();
        let mut rows = Vec::with_capacity(routes.len());
        let recipe = self.recipe_eligible && device.recipe_prefill;
        let job_size = if recipe { 128 } else { 16 };
        let group_start = device.clock;
        let mut begin = 0;
        while begin < routes.len() {
            let expert = routes[begin].expert;
            let count = routes[begin..].partition_point(|r| r.expert == expert);
            let slot =
                device.acquire(self.weights, self.layer.full(expert), capacity, group_start)?;
            for start in (begin..begin + count).step_by(job_size) {
                jobs.push(Job {
                    expert: slot,
                    begin: start as i32,
                    count: (begin + count - start).min(job_size) as i32,
                });
            }
            for route in &routes[begin..begin + count] {
                rows.push(RoutedRow {
                    expert: slot,
                    token: route.token.try_into()?,
                    output: if staged.is_some() {
                        rows.len()
                    } else {
                        route.token * self.top_k + route.rank
                    }
                    .try_into()?,
                    weight: half::f16::from_f32(route.weight).to_f32(),
                });
            }
            begin += count;
        }
        let mut scratch = device.scratch as *mut c_void;
        let status = unsafe {
            ds_moe_run_dev(
                &mut scratch,
                &mut device.scratch_bytes,
                jobs.as_ptr(),
                jobs.len() as i32,
                rows.as_ptr(),
                rows.len() as i32,
                staged.map_or(self.x.as_ptr(), |p| p.x.cast_const()),
                staged.map_or_else(|| self.weighted.as_mut_ptr(), |p| p.output),
                i32::from(matches!(self.layer, Layer::Draft(_))),
                splits,
                device.inline_metadata.into(),
                if recipe {
                    3
                } else if device.fused {
                    1 + i32::from(device.packed_decode)
                } else {
                    0
                },
            )
        };
        device.scratch = scratch as usize;
        cuda(status)
    }
}

// Each group respects the slot capacity and contains at most 16 rows per MMA
// job. Compute its split-K once before distributing jobs across the cards.
fn group_shape(routes: &[Route], max_experts: usize) -> (usize, usize) {
    routes[..routes.len().min(MAX_GROUP_ROWS)]
        .chunk_by(|a, b| a.expert == b.expert)
        .take(max_experts)
        .fold((0, 0), |(rows, jobs), group| {
            (rows + group.len(), jobs + group.len().div_ceil(16))
        })
}

// Prefix snapshots must reproduce the same arithmetic at any physical slot limit.
fn logical_plan(routes: &[Route], verification: bool) -> (usize, i32) {
    let (rows, jobs) = group_shape(routes, MAX_GROUP_EXPERTS);
    let splits = if verification || jobs < 32 { 8 } else { 2 };
    (rows, splits)
}

struct Placement {
    device: i32,
    capacity: usize,
    queried: usize,
    uploads_before: usize,
    staged: Option<PeerBuffers>,
    completed_rows: usize,
}

impl Placement {
    fn prepare(
        cache: &mut DeviceCache,
        layer: Layer,
        owned: &[Route],
        select: bool,
    ) -> Result<Self> {
        if select {
            cuda(unsafe { cudaSetDevice(cache.device) })?;
        }
        let query = !owned.is_empty() && !cache.can_reuse_capacity(layer, owned);
        Ok(Self {
            device: cache.device,
            capacity: if query {
                cache.capacity(SLOT_BYTES)?
            } else {
                cache.slots.len().max(1)
            },
            queried: usize::from(query),
            uploads_before: cache.uploads,
            staged: None,
            completed_rows: 0,
        })
    }

    fn run(
        &mut self,
        cache: &mut MoeCache,
        dispatch: &mut Dispatch<'_>,
        mut rows: &[Route],
        splits: i32,
        select: bool,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if select {
            cuda(unsafe { cudaSetDevice(self.device) })?;
        }
        let device = cache.device(self.device)?;
        while !rows.is_empty() {
            let (n, _) = group_shape(rows, self.capacity);
            let staged = self.staged.map(|p| PeerBuffers {
                x: p.x,
                output: unsafe { p.output.add(self.completed_rows * HIDDEN) },
            });
            dispatch.run_group(device, &rows[..n], self.capacity, splits, staged)?;
            self.completed_rows += n;
            rows = &rows[n..];
        }
        Ok(())
    }
}

pub fn forward_dev(rt: &mut Runtime, layer: usize, x: &Dev, batch: usize) -> Result<Dev> {
    forward::<EXPERTS, TOP_K>(rt, Layer::Backbone(layer), x, batch)
}

pub fn draft_forward(rt: &mut Runtime, stage: usize, x: &Dev, batch: usize) -> Result<Dev> {
    ensure!(stage < 3, "invalid DSpark stage");
    forward::<DRAFT_EXPERTS, DRAFT_TOP_K>(rt, Layer::Draft(stage), x, batch)
}

fn forward<const E: usize, const K: usize>(
    rt: &mut Runtime,
    layer: Layer,
    x: &Dev,
    batch: usize,
) -> Result<Dev> {
    ensure!(
        batch > 0 && batch.checked_mul(HIDDEN) == Some(x.len()),
        "invalid FFN input shape"
    );
    let base = layer.base();
    // Reserve the output before recording readiness: a pooled allocation may
    // still have ordered uses from the previous layer on the primary stream.
    let mut weighted = rt.buffer(x.len().checked_mul(K).context("routed output overflow")?)?;
    let peer_id = rt.peer_device();
    let cache = rt.moe_cache.device(rt.device)?;
    #[cfg(feature = "recipe-prefill")]
    let tp2 = tp2::eligible(
        cache.tp2_prefill,
        rt.target_prefill,
        rt.verification,
        layer,
        batch,
    );
    #[cfg(not(feature = "recipe-prefill"))]
    let tp2 = false;
    let parallel =
        !tp2 && cache.parallel && (matches!(layer, Layer::Backbone(_)) || cache.parallel_draft);
    // Wide staged prompt batches triggered a copy-engine fault on the dual
    // RTX PRO 6000. Peer kernels can read and write the primary buffers
    // directly. Keep compact transfers for decode and smaller prompt batches.
    let wide_prefill = rt.target_prefill && batch > 128;
    let stage_copy = parallel && cache.staged && !wide_prefill;
    let peer_work = if parallel {
        Some(PeerWork::begin(cache, peer_id)?)
    } else {
        None
    };
    let logits = rt.linear_dev(&format!("{base}.gate.weight"), x, batch, 1)?;
    let bias = rt.weights.vector(&format!("{base}.gate.bias"))?;
    // Only router logits visit the CPU, which computes the checkpoint's top-k
    // and chooses mmap-backed expert weights for the GPU slot cache.
    let cache = rt.moe_cache.device(rt.device)?;
    let parallel_router =
        cache.parallel_prefill_router && rt.target_prefill && matches!(layer, Layer::Backbone(_));
    let (routes, mut shared_out) = if cache.overlap_router {
        cache.router.start(&logits)?;
        let shared_out = shared(rt, &base, x, batch)?;
        let read = &rt.moe_cache.device(rt.device)?.router;
        (
            routes::<E, K>(read.scores(logits.len())?, &bias, batch, parallel_router)?,
            shared_out,
        )
    } else {
        let routes = routes::<E, K>(&rt.to_host(&logits)?, &bias, batch, parallel_router)?;
        (routes, shared(rt, &base, x, batch)?)
    };
    ensure!(
        shared_out.len() == x.len(),
        "shared expert output shape mismatch"
    );
    #[cfg(feature = "recipe-prefill")]
    if tp2 {
        let Layer::Backbone(index) = layer else {
            unreachable!()
        };
        tp2::dispatch(rt, index, x, &mut weighted, &mut shared_out, &routes)?;
        return Ok(shared_out);
    }
    let peer_rows;
    let primary_rows;
    let (local, remote): (&[Route], &[Route]) = if parallel {
        (primary_rows, peer_rows) = routes
            .iter()
            .copied()
            .partition::<Vec<Route>, _>(|r| r.expert % 2 == 0);
        (&primary_rows, &peer_rows)
    } else {
        (&routes, &[])
    };
    let mut compact = if stage_copy && !remote.is_empty() {
        Some(rt.buffer(remote.len() * HIDDEN)?)
    } else {
        None
    };
    let device = rt.moe_cache.device(rt.device)?;
    let mut primary = Placement::prepare(device, layer, local, false)?;
    let report = device.metrics;
    let mut peer = if let Some(work) = &peer_work {
        let buffers = if compact.is_some() {
            let required = x.len() + remote.len() * HIDDEN;
            let allocated = unsafe { ds_moe_peer_capacity(work.0) };
            if required > allocated {
                cuda(unsafe { cudaSetDevice(peer_id) })?;
                rt.moe_cache
                    .reserve(peer_id, (required - allocated) * size_of::<f32>())?;
            }
            let mut buffers = PeerBuffers {
                x: ptr::null_mut(),
                output: ptr::null_mut(),
            };
            cuda(unsafe {
                ds_moe_peer_stage(
                    work.0,
                    x.as_ptr(),
                    x.len(),
                    remote.len(),
                    &mut buffers.x,
                    &mut buffers.output,
                )
            })?;
            Some(buffers)
        } else {
            None
        };
        // Staging already occupies its physical bytes before sizing expert slots.
        let mut placement = Placement::prepare(rt.moe_cache.device(peer_id)?, layer, remote, true)?;
        placement.staged = buffers;
        Some(placement)
    } else {
        None
    };
    let mut dispatch = Dispatch {
        weights: &rt.weights,
        layer,
        x,
        weighted: &mut weighted,
        top_k: K,
        recipe_eligible: rt.target_prefill && batch >= 128 && matches!(layer, Layer::Backbone(_)),
    };
    let (mut begin, mut ni, mut pi) = (0, 0, 0);
    while begin < routes.len() {
        let (count, splits) = logical_plan(
            &routes[begin..],
            rt.verification && matches!(layer, Layer::Backbone(_)),
        );
        let nr = if let Some(peer) = &mut peer {
            let n = routes[begin..begin + count]
                .iter()
                .filter(|r| r.expert % 2 != 0)
                .count();
            peer.run(
                &mut rt.moe_cache,
                &mut dispatch,
                &remote[pi..pi + n],
                splits,
                true,
            )?;
            pi += n;
            n
        } else {
            0
        };
        let n = count - nr;
        primary.run(
            &mut rt.moe_cache,
            &mut dispatch,
            &local[ni..ni + n],
            splits,
            parallel,
        )?;
        ni += n;
        begin += count;
    }
    if let Some(work) = peer_work {
        let link = work.0;
        work.finish()?;
        if let (Some(compact), Some(buffers)) =
            (compact.as_mut(), peer.as_ref().and_then(|p| p.staged))
        {
            let destinations: Vec<i32> = remote
                .iter()
                .map(|r| (r.token * K + r.rank) as i32)
                .collect();
            cuda(unsafe {
                ds_moe_peer_collect(
                    link,
                    buffers.output,
                    compact.as_mut_ptr(),
                    destinations.as_ptr(),
                    remote.len().try_into()?,
                    weighted.as_mut_ptr(),
                )
            })?;
        }
    }
    cuda(unsafe {
        ds_moe_finish(
            weighted.as_ptr(),
            shared_out.as_mut_ptr(),
            shared_out.len(),
            K as i32,
        )
    })?;
    if report {
        let mut uploads = rt.moe_cache.device(primary.device)?.uploads - primary.uploads_before;
        let mut queries = primary.queried;
        if let Some(peer) = peer {
            uploads += rt.moe_cache.device(peer.device)?.uploads - peer.uploads_before;
            queries += peer.queried;
        }
        let m = &mut rt.moe_cache.metrics;
        if matches!(layer, Layer::Backbone(0) | Layer::Draft(0)) {
            *m = Metrics::default();
        }
        m.routes += routes.len();
        m.unique += routes.chunk_by(|a, b| a.expert == b.expert).count();
        m.uploads += uploads;
        m.capacity_queries += queries;
        if matches!(layer, Layer::Backbone(39) | Layer::Draft(2)) {
            let kind = match layer {
                Layer::Backbone(_) if rt.verification => "verify",
                Layer::Backbone(_) => "target",
                Layer::Draft(_) => "draft",
            };
            eprintln!("moe pass={kind} rows={batch} routes={} unique_layer_experts={} expert_uploads={} expert_h2d_bytes={} capacity_queries={}",
                      m.routes, m.unique, m.uploads, m.uploads * SLOT_BYTES, m.capacity_queries);
        }
    }
    Ok(shared_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Layer::Backbone;
    #[test]
    fn cache_capacity_reclaims_headroom_and_honors_the_user_limit() {
        assert_eq!(
            DeviceCache::available_slots(5 * SLOT_BYTES, 512 << 20, 10, SLOT_BYTES),
            5
        );
        assert_eq!(
            DeviceCache::available_slots(5 * SLOT_BYTES, (512 << 20) - SLOT_BYTES, 10, SLOT_BYTES),
            4
        );
        assert_eq!(
            DeviceCache::available_slots(5 * SLOT_BYTES, 8192 << 20, 10, SLOT_BYTES),
            10
        );
        assert_eq!(DeviceCache::available_slots(0, 0, 10, SLOT_BYTES), 1);
    }

    #[test]
    fn logical_job_count_survives_smaller_physical_cache_groups() {
        let routes: Vec<_> = (0..40)
            .map(|expert| Route {
                expert,
                token: expert,
                rank: 0,
                weight: 1.,
            })
            .collect();
        assert_eq!(logical_plan(&routes, false), (40, 2));
        assert_eq!(logical_plan(&routes, true), (40, 8));
        let owned: Vec<_> = routes
            .iter()
            .copied()
            .filter(|r| r.expert % 2 == 1)
            .collect();
        for limit in [1, 8, MAX_GROUP_EXPERTS] {
            let (n, _) = group_shape(&owned, limit);
            assert_eq!(n, limit.min(owned.len()));
            // An isolated physical subset would choose eight splits. Dispatch
            // uses the full group’s two splits at every VRAM/cache limit.
            assert_eq!(logical_plan(&owned[..n], false), (n, 8));
        }
    }

    #[test]
    fn cached_capacity_needs_both_resident_experts_and_sufficient_scratch() {
        let mut cache = DeviceCache::new(0).unwrap();
        let layer = Backbone(0);
        let routes = [Route {
            expert: 0,
            token: 0,
            rank: 0,
            weight: 1.,
        }];
        cache.cached_capacity = true;
        cache.scratch_bytes = usize::MAX;
        assert!(!cache.can_reuse_capacity(layer, &routes));
        cache.slots.insert(layer.full(0), (0, 1));
        assert!(cache.can_reuse_capacity(layer, &routes));
        cache.scratch_bytes = 0;
        assert!(!cache.can_reuse_capacity(layer, &routes));
    }

    #[test]
    fn cached_hits_protect_their_slot_during_eviction() {
        let mut cache = DeviceCache::new(0).unwrap();
        cache.frequencies = None;
        cache.resident_bytes = 3 * SLOT_BYTES;
        // Null entries need no CUDA allocation. Exercise the same ordering
        // that protects experts collected for a pending group kernel.
        for i in 0..3 {
            cache.slots.insert(Backbone(2).full(i), (0, i as u64));
            cache.lru.insert((i as u64, Backbone(2).full(i)));
        }
        assert_eq!(cache.touch(Backbone(2).full(0), 3), Some(0));
        cache.evict_slot(u64::MAX).unwrap();
        assert!(!cache.slots.contains_key(&Backbone(2).full(1)));
        assert!(cache.slots.contains_key(&Backbone(2).full(0)));
        assert!(cache.slots.contains_key(&Backbone(2).full(2)));
    }

    #[test]
    fn lfu_preserves_group_slots_and_breaks_frequency_ties_by_recency() {
        let mut cache = DeviceCache::new(0).unwrap();
        let keys = [
            Backbone(39).full(383),
            Layer::Draft(0).full(0),
            Layer::Draft(2).full(127),
        ];
        let mut frequency = vec![0; EXPERT_KEYS].into_boxed_slice();
        for (i, key) in keys.iter().copied().enumerate() {
            cache.slots.insert(key, (0, i as u64 + 1));
            frequency[frequency_index(key)] = [8, 8, 1][i];
        }
        cache.frequencies = Some(frequency);
        cache.resident_bytes = 3 * SLOT_BYTES;
        assert_eq!(cache.touch(keys[2], 4), Some(0));
        cache.evict_slot(3).unwrap();
        assert!(!cache.slots.contains_key(&keys[0]));
        assert!(cache.slots.contains_key(&keys[2]));
        cache.evict_slot(u64::MAX).unwrap();
        assert!(!cache.slots.contains_key(&keys[2]));
        assert_eq!(
            cache.frequencies.as_ref().unwrap()[frequency_index(keys[2])],
            2
        );
        assert!(cache.evict_slot(0).is_err());
        assert_eq!(cache.slots.len(), 1);
    }

    #[test]
    #[ignore = "needs DS4_TEST_CUDA_DEVICE and the EXL3 checkpoint"]
    fn real_checkpoint_eviction_and_reload() -> Result<()> {
        unsafe extern "C" {
            fn ds_set_device(device: i32) -> i32;
        }
        let device = std::env::var("DS4_TEST_CUDA_DEVICE")?.parse()?;
        ensure!(
            unsafe { ds_set_device(device) } == 0,
            "cannot select test device"
        );
        let root = std::env::var("DS4_TEST_PACK")
            .unwrap_or_else(|_| "/mnt/models/dsv41-flash-exl3".into());
        let weights = Weights::open(std::path::Path::new(&root))?;
        let x = (0..HIDDEN)
            .map(|i| (i as f32 * 0.3).sin())
            .collect::<Vec<_>>();
        let run = |cache: &mut DeviceCache, expert| -> Result<Vec<f32>> {
            let slot = cache.acquire(&weights, Backbone(0).full(expert), 2, u64::MAX)?;
            let mut y = vec![0.; HIDDEN];
            let job = Job {
                expert: slot,
                begin: 0,
                count: 1,
            };
            let mut scratch = cache.scratch as *mut c_void;
            let status = unsafe {
                ds_moe_run(
                    &mut scratch,
                    &mut cache.scratch_bytes,
                    &job,
                    1,
                    &slot,
                    x.as_ptr(),
                    y.as_mut_ptr(),
                    1,
                )
            };
            cache.scratch = scratch as usize;
            cuda(status)?;
            Ok(y)
        };
        for lfu in [false, true] {
            let mut cache = DeviceCache::new(device)?;
            cache.frequencies = lfu.then(|| vec![0; EXPERT_KEYS].into_boxed_slice());
            let first = run(&mut cache, 0)?;
            let second = run(&mut cache, 1)?;
            let first_slot = cache.slots[&Backbone(0).full(0)].0;
            let second_slot = cache.slots[&Backbone(0).full(1)].0;
            assert_eq!(first, run(&mut cache, 0)?);
            let _ = run(&mut cache, 2)?;
            assert!(!cache.slots.contains_key(&Backbone(0).full(1)));
            assert_eq!(cache.slots[&Backbone(0).full(2)].0, second_slot);
            assert_eq!(second, run(&mut cache, 1)?);
            let expected = if lfu { second_slot } else { first_slot };
            assert_eq!(cache.slots[&Backbone(0).full(1)].0, expected);
            assert!(cache
                .slots
                .contains_key(&Backbone(0).full(if lfu { 0 } else { 2 })));
            assert_eq!(cache.slots.len(), 2);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs in CUDA_VISIBLE_DEVICES"]
    fn peer_experts_match_single_device_in_both_directions() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        let batch = 3;
        let input: Vec<_> = (0..batch * HIDDEN)
            .map(|i| (i as f32 * 0.057).cos())
            .collect();
        for layer in [Layer::Backbone(0), Layer::Draft(2)] {
            rt.layer(if matches!(layer, Layer::Backbone(_)) {
                0
            } else {
                39
            })?;
            let xs = [rt.to_device(&input[..HIDDEN])?, rt.to_device(&input)?];
            rt.verification = matches!(layer, Layer::Backbone(_));
            let mut baseline = vec![];
            for (parallel, staged) in [(false, false), (true, false), (true, true)] {
                rt.moe_cache = MoeCache::default();
                let cache = rt.moe_cache.device(rt.device)?;
                cache.parallel = parallel;
                cache.parallel_draft = true;
                cache.staged = staged;
                cache.fused = true;
                cache.limit = 2; // force slot reuse while remote kernels hold the old data
                let peer = rt.peer_device();
                rt.moe_cache.device(peer)?.limit = 2;
                for (index, x) in [&xs[0], &xs[1], &xs[1]].into_iter().enumerate() {
                    let batch = x.len() / HIDDEN;
                    let y = match layer {
                        Layer::Backbone(n) => forward_dev(&mut rt, n, x, batch)?,
                        Layer::Draft(n) => draft_forward(&mut rt, n, x, batch)?,
                    };
                    let actual: Vec<_> = rt.to_host(&y)?.into_iter().map(f32::to_bits).collect();
                    if parallel {
                        assert_eq!(actual, baseline[index]);
                    } else {
                        baseline.push(actual);
                    }
                }
                if parallel {
                    for (device, cache) in &rt.moe_cache.devices {
                        assert!(!cache.slots.is_empty());
                        assert!(cache
                            .slots
                            .keys()
                            .all(|&Key(_, e, _)| (e % 2 == 0) == (*device == rt.device)));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs in CUDA_VISIBLE_DEVICES"]
    fn peer_staging_boundary_preserves_expert_outputs() -> Result<()> {
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        let input: Vec<_> = (0..129 * HIDDEN)
            .map(|i| (i as f32 * 0.057).cos())
            .collect();
        for layer in [0, 20] {
            rt.layer(layer)?;
            for (batch, prefill, expect_staged) in
                [(128, true, true), (129, true, false), (129, false, true)]
            {
                rt.target_prefill = prefill;
                let x = rt.to_device(&input[..batch * HIDDEN])?;
                let mut expected = Vec::new();
                for parallel in [false, true] {
                    rt.moe_cache = MoeCache::default();
                    let primary = rt.device;
                    let peer = rt.peer_device();
                    {
                        let cache = rt.moe_cache.device(primary)?;
                        cache.parallel = parallel;
                        cache.staged = true;
                        cache.fused = true;
                    }
                    let y = forward_dev(&mut rt, layer, &x, batch)?;
                    let actual: Vec<_> = rt.to_host(&y)?.into_iter().map(f32::to_bits).collect();
                    if parallel {
                        assert!(!rt.moe_cache.device(peer)?.slots.is_empty());
                        let link = rt.moe_cache.device(primary)?.peer_link;
                        assert!(!link.is_null());
                        assert_eq!(unsafe { ds_moe_peer_capacity(link) } > 0, expect_staged);
                        assert_eq!(actual, expected);
                    } else {
                        expected = actual;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn draft_routes_keep_expert_major_sums_for_each_token() {
        let mut logits = vec![0.; 2 * DRAFT_EXPERTS];
        for (token, picked) in [[24, 78, 40], [78, 11, 91]].iter().enumerate() {
            for (rank, &expert) in picked.iter().enumerate() {
                logits[token * DRAFT_EXPERTS + expert] = 10. - rank as f32;
            }
        }
        let routes =
            routes::<DRAFT_EXPERTS, DRAFT_TOP_K>(&logits, &[0.; DRAFT_EXPERTS], 2, false).unwrap();
        let rows: Vec<_> = routes.iter().map(|r| (r.expert, r.token, r.rank)).collect();
        assert_eq!(
            rows,
            [
                (11, 1, 0),
                (24, 0, 0),
                (40, 0, 1),
                (78, 0, 2),
                (78, 1, 1),
                (91, 1, 2)
            ]
        );
    }

    #[test]
    fn parallel_prefill_router_preserves_every_assignment() {
        let rows = 131;
        let logits: Vec<f32> = (0..rows * EXPERTS)
            .map(|i| ((i * 29 + i / EXPERTS) % 37) as f32 / 5. - 3.)
            .collect();
        let bias: Vec<f32> = (0..EXPERTS).map(|i| (i % 7) as f32 * 0.01).collect();
        let serial = routes::<EXPERTS, TOP_K>(&logits, &bias, rows, false).unwrap();
        assert_eq!(
            serial,
            routes::<EXPERTS, TOP_K>(&logits, &bias, rows, true).unwrap()
        );
    }

    #[test]
    fn correction_bias_selects_experts_but_does_not_weight_them() {
        let mut bias = vec![0.0; EXPERTS];
        bias[300] = 2.0;
        let mut logits = vec![0.0; EXPERTS];
        logits[42] = 2.0;
        let selected = routes::<EXPERTS, TOP_K>(&logits, &bias, 1, false).unwrap();
        assert_eq!(
            selected.iter().map(|r| r.expert).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 42, 300]
        );
        assert!((selected.iter().map(|r| r.weight).sum::<f32>() - 1.5).abs() < 1e-6);
        assert_eq!(selected[0].weight, selected[5].weight);
        assert!(selected[4].weight > selected[5].weight);
    }
}
