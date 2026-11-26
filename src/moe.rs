//! DeepSeek V4.1 Flash text MoE. Weights stay mmap-backed on the host until an
//! expert is actually selected. Slots are reused in LRU order on each GPU.
use anyhow::{bail, ensure, Context, Result};
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
    fn ds_moe_free_bytes(bytes: *mut usize) -> i32;
    fn ds_moe_alloc(slot: *mut *mut c_void) -> i32;
    fn ds_moe_upload(slot: *mut c_void, parts: *const *const c_void) -> i32;
    fn ds_moe_free(device: i32, slot: *mut c_void);
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
        verify: i32,
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
        let message = unsafe { CStr::from_ptr(ds_moe_error(code)) }.to_string_lossy();
        bail!("CUDA MoE error {code}: {message}");
    }
    Ok(())
}

#[derive(Default)]
pub struct MoeCache {
    devices: HashMap<i32, DeviceCache>,
}

impl MoeCache {
    /// Make room before an uncached dense projection uploads its resident
    /// weight and conversion buffers. Freeing a slot waits for pending uses.
    pub fn reserve(&mut self, device: i32, bytes: usize) -> Result<()> {
        let Some(cache) = self.devices.get_mut(&device) else {
            return Ok(());
        };
        let mut free = 0;
        cuda(unsafe { ds_moe_free_bytes(&mut free) })?;
        let missing = bytes.saturating_add(64 << 20).saturating_sub(free);
        let to_remove = missing.div_ceil(SLOT_BYTES).min(cache.slots.len());
        for _ in 0..to_remove {
            let slot = cache.evict_oldest()?;
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
    fn base(self) -> String {
        match self {
            Self::Backbone(n) => format!("layers.{n}.ffn"),
            Self::Draft(n) => format!("mtp.{n}.ffn"),
        }
    }
}

type Key = (Layer, usize);
struct DeviceCache {
    device: i32,
    limit: usize,
    slots: HashMap<Key, (usize, u64)>,
    lru: BTreeSet<(u64, Key)>,
    clock: u64,
    scratch: usize,
    scratch_bytes: usize,
}

impl DeviceCache {
    fn new(device: i32) -> Result<Self> {
        let gib: f64 = std::env::var("DS4_MOE_CACHE_GIB")
            .unwrap_or_else(|_| "78".to_owned())
            .parse()
            .context("DS4_MOE_CACHE_GIB must be a number")?;
        ensure!(gib.is_finite() && gib >= 0.0, "invalid DS4_MOE_CACHE_GIB");
        let limit = (((gib * 1073741824.0) as usize) / SLOT_BYTES).max(1);
        Ok(Self {
            device,
            limit,
            slots: HashMap::new(),
            lru: BTreeSet::new(),
            clock: 0,
            scratch: 0,
            scratch_bytes: 0,
        })
    }

    fn evict_oldest(&mut self) -> Result<usize> {
        let (_, key) = self.lru.pop_first().context("empty expert cache")?;
        Ok(self.slots.remove(&key).context("missing LRU slot")?.0)
    }

    fn available_slots(resident: usize, free: usize, limit: usize) -> usize {
        let available = (resident * SLOT_BYTES + free).saturating_sub(512 << 20);
        (available / SLOT_BYTES).clamp(1, limit)
    }

    fn capacity(&mut self) -> Result<usize> {
        let mut free = 0;
        cuda(unsafe { ds_moe_free_bytes(&mut free) })?;
        // Leave room for scratch, dense projections and the CUDA context. If
        // another allocation consumes the reserve, later groups shrink again.
        let capacity = Self::available_slots(self.slots.len(), free, self.limit);
        while self.slots.len() > capacity {
            let slot = self.evict_oldest()?;
            unsafe { ds_moe_free(self.device, slot as *mut c_void) };
        }
        Ok(capacity)
    }

    fn touch(&mut self, key: Key, stamp: u64) -> Option<usize> {
        let (slot, age) = self.slots.get_mut(&key)?;
        self.lru.remove(&(*age, key));
        *age = stamp;
        self.lru.insert((stamp, key));
        Some(*slot)
    }

    fn acquire(&mut self, weights: &Weights, key: Key, capacity: usize) -> Result<usize> {
        self.clock += 1;
        let stamp = self.clock;
        if let Some(slot) = self.touch(key, stamp) {
            return Ok(slot);
        }
        // Check all sizes before passing mmap pointers over FFI.
        let parts = checkpoint_parts(weights, key)?;
        let slot = if self.slots.len() >= capacity {
            self.evict_oldest()?
        } else {
            let mut slot = ptr::null_mut();
            cuda(unsafe { ds_moe_alloc(&mut slot) })?;
            slot as usize
        };
        if let Err(e) = cuda(unsafe { ds_moe_upload(slot as *mut c_void, parts.as_ptr()) }) {
            unsafe { ds_moe_free(self.device, slot as *mut c_void) };
            return Err(e);
        }
        self.slots.insert(key, (slot, stamp));
        self.lru.insert((stamp, key));
        Ok(slot)
    }
}

impl Drop for DeviceCache {
    fn drop(&mut self) {
        for &(slot, _) in self.slots.values() {
            unsafe { ds_moe_free(self.device, slot as *mut c_void) };
        }
        unsafe { ds_moe_free(self.device, self.scratch as *mut c_void) };
    }
}

fn checkpoint_parts(weights: &Weights, (layer, expert): Key) -> Result<[*const c_void; 9]> {
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

#[derive(Clone, Copy)]
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
) -> Result<Vec<Route>> {
    ensure!(
        logits.len() == batch * E && bias.len() == E,
        "router shape mismatch"
    );
    let mut output = Vec::with_capacity(batch * K);
    for (token, scores) in logits.as_chunks::<E>().0.iter().enumerate() {
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
        for (_, score, expert) in top {
            output.push(Route {
                expert,
                token,
                rank: 0,
                weight: 1.5 * score / norm,
            });
        }
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

fn run_group<const K: usize>(
    rt: &mut Runtime,
    layer: Layer,
    routes: &[Route],
    x: &Dev,
    weighted: &mut Dev,
    capacity: usize,
) -> Result<()> {
    let device = rt
        .moe_cache
        .devices
        .get_mut(&rt.device)
        .context("missing expert cache")?;
    let mut jobs = Vec::new();
    let mut rows = Vec::with_capacity(routes.len());
    let mut begin = 0;
    while begin < routes.len() {
        let expert = routes[begin].expert;
        let count = routes[begin..].partition_point(|r| r.expert == expert);
        let slot = device.acquire(&rt.weights, (layer, expert), capacity)?;
        for start in (begin..begin + count).step_by(16) {
            jobs.push(Job {
                expert: slot,
                begin: start as i32,
                count: (begin + count - start).min(16) as i32,
            });
        }
        for route in &routes[begin..begin + count] {
            rows.push(RoutedRow {
                expert: slot,
                token: route.token.try_into()?,
                output: (route.token * K + route.rank).try_into()?,
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
            x.as_ptr(),
            weighted.as_mut_ptr(),
            i32::from(matches!(layer, Layer::Draft(_))),
            i32::from(rt.verification && matches!(layer, Layer::Backbone(_))),
        )
    };
    device.scratch = scratch as usize;
    cuda(status)
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
    let logits = rt.linear_dev(&format!("{base}.gate.weight"), x, batch, 1)?;
    let bias = rt.weights.vector(&format!("{base}.gate.bias"))?;
    // Only router logits visit the CPU, which computes the checkpoint's top-k
    // and chooses mmap-backed expert weights for the GPU slot cache.
    let routes = routes::<E, K>(&rt.to_host(&logits)?, &bias, batch)?;
    let mut shared_out = shared(rt, &base, x, batch)?;
    ensure!(
        shared_out.len() == x.len(),
        "shared expert output shape mismatch"
    );
    let mut weighted = rt.buffer(x.len().checked_mul(K).context("routed output overflow")?)?;
    let device = match rt.moe_cache.devices.entry(rt.device) {
        std::collections::hash_map::Entry::Occupied(v) => v.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => v.insert(DeviceCache::new(rt.device)?),
    };
    let capacity = device.capacity()?;
    let max_experts = capacity.min(MAX_GROUP_EXPERTS);
    let mut begin = 0;
    while begin < routes.len() {
        let mut end = begin;
        let mut distinct = 0;
        while end < routes.len() && end - begin < MAX_GROUP_ROWS {
            if end == begin || routes[end].expert != routes[end - 1].expert {
                if distinct == max_experts {
                    break;
                }
                distinct += 1;
            }
            end += 1;
        }
        run_group::<K>(rt, layer, &routes[begin..end], x, &mut weighted, capacity)?;
        begin = end;
    }
    cuda(unsafe {
        ds_moe_finish(
            weighted.as_ptr(),
            shared_out.as_mut_ptr(),
            shared_out.len(),
            K as i32,
        )
    })?;
    Ok(shared_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Layer::Backbone;
    #[test]
    fn cache_capacity_reclaims_headroom_and_honors_the_user_limit() {
        assert_eq!(DeviceCache::available_slots(5, 512 << 20, 10), 5);
        assert_eq!(
            DeviceCache::available_slots(5, (512 << 20) - SLOT_BYTES, 10),
            4
        );
        assert_eq!(DeviceCache::available_slots(5, 8192 << 20, 10), 10);
        assert_eq!(DeviceCache::available_slots(0, 0, 10), 1);
    }

    #[test]
    fn cached_hits_protect_their_slot_during_eviction() {
        let mut cache = DeviceCache::new(0).unwrap();
        // Null entries need no CUDA allocation. Exercise the same ordering
        // that protects experts collected for a pending group kernel.
        for i in 0..3 {
            cache.slots.insert((Backbone(2), i), (0, i as u64));
            cache.lru.insert((i as u64, (Backbone(2), i)));
        }
        assert_eq!(cache.touch((Backbone(2), 0), 3), Some(0));
        cache.evict_oldest().unwrap();
        assert!(!cache.slots.contains_key(&(Backbone(2), 1)));
        assert!(cache.slots.contains_key(&(Backbone(2), 0)));
        assert!(cache.slots.contains_key(&(Backbone(2), 2)));
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
        let mut cache = DeviceCache::new(device)?;
        let x = (0..HIDDEN)
            .map(|i| (i as f32 * 0.3).sin())
            .collect::<Vec<_>>();
        let run = |cache: &mut DeviceCache, expert| -> Result<Vec<f32>> {
            let slot = cache.acquire(&weights, (Backbone(0), expert), 2)?;
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
        let first = run(&mut cache, 0)?;
        let second = run(&mut cache, 1)?;
        let first_slot = cache.slots[&(Backbone(0), 0)].0;
        let second_slot = cache.slots[&(Backbone(0), 1)].0;
        assert_eq!(first, run(&mut cache, 0)?);
        let _ = run(&mut cache, 2)?;
        assert!(!cache.slots.contains_key(&(Backbone(0), 1)));
        assert_eq!(cache.slots[&(Backbone(0), 2)].0, second_slot);
        assert_eq!(second, run(&mut cache, 1)?);
        assert_eq!(cache.slots[&(Backbone(0), 1)].0, first_slot);
        assert_eq!(cache.slots.len(), 2);
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
            routes::<DRAFT_EXPERTS, DRAFT_TOP_K>(&logits, &[0.; DRAFT_EXPERTS], 2).unwrap();
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
    fn correction_bias_selects_experts_but_does_not_weight_them() {
        let mut bias = vec![0.0; EXPERTS];
        bias[300] = 2.0;
        let mut logits = vec![0.0; EXPERTS];
        logits[42] = 2.0;
        let selected = routes::<EXPERTS, TOP_K>(&logits, &bias, 1).unwrap();
        assert_eq!(
            selected.iter().map(|r| r.expert).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 42, 300]
        );
        assert!((selected.iter().map(|r| r.weight).sum::<f32>() - 1.5).abs() < 1e-6);
        assert_eq!(selected[0].weight, selected[5].weight);
        assert!(selected[4].weight > selected[5].weight);
    }
}
