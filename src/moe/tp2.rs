//! Intra-layer S8B prefill. Both devices run every routed expert using their
//! own 1,152 channels. Only token activations and reduced token outputs cross
//! between GPUs. Each GPU launches kernels on its legacy default stream.
use super::*;

const GROUP_ROWS: usize = 2048;
const COPY_FLOATS: usize = (2 << 20) / size_of::<f32>();

unsafe extern "C" {
    pub(super) fn ds_recipe_free_uploads(device: i32, work: *mut c_void);
    pub(super) fn ds_recipe_staged_rank(
        work: *mut *mut c_void,
        slot: *mut c_void,
        bytes: usize,
        parts: *const *const c_void,
        rank: i32,
    ) -> i32;
    pub(super) fn ds_recipe_upload_rank(
        slot: *mut c_void,
        bytes: usize,
        parts: *const *const c_void,
        rank: i32,
    ) -> i32;
    fn ds_recipe_rank_group(
        scratch: *mut *mut c_void,
        capacity: *mut usize,
        jobs: *const Job,
        nj: i32,
        routes: *const RoutedRow,
        rows: i32,
        x: *const f32,
        weighted: *mut f32,
    ) -> i32;
    fn ds_recipe_sum_rank(weighted: *const f32, tokens: *mut f32, count: usize) -> i32;
    fn ds_recipe_finish_ranks(
        local: *const f32,
        peer: *const f32,
        shared: *mut f32,
        start: usize,
        count: usize,
    ) -> i32;
    fn ds_recipe_enable_peer_dma(owner: i32, peer: i32, available: *mut i32) -> i32;
    fn ds_recipe_collect_dma(
        routed: *mut f32,
        shared: *mut f32,
        owner: i32,
        reduced: *const f32,
        peer: i32,
        count: usize,
    ) -> i32;
    fn ds_recipe_broadcast(
        src: *const f32,
        owner: i32,
        dst: *mut f32,
        peer: i32,
        bounce: *mut f32,
        send: *mut c_void,
        receive: *mut c_void,
        count: usize,
        packed: i32,
        direct: i32,
    ) -> i32;
    fn ds_recipe_transfer(
        dst: *mut f32,
        dst_device: i32,
        src: *const f32,
        src_device: i32,
        bounce: *mut f32,
        count: usize,
    ) -> i32;
    fn cudaHostAlloc(host: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
    fn cudaFreeHost(host: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

fn preload_fits(resident: usize, capacity: usize, free: usize) -> bool {
    let budget = capacity * RANK_SLOT_BYTES;
    budget.saturating_sub(resident) >= RANK_SLOT_BYTES && free >= EXPERT_HEADROOM + RANK_SLOT_BYTES
}

fn preload_rank(
    cache: &mut DeviceCache,
    weights: &Weights,
    layers: usize,
    rank: u8,
) -> Result<usize> {
    cuda(unsafe { cudaSetDevice(cache.device) })?;
    let result = (|| {
        let capacity = cache.limit_bytes / RANK_SLOT_BYTES;
        let mut loaded = 0;
        for layer in 0..layers {
            for expert in 0..EXPERTS {
                let key = Key(Layer::Backbone(layer), expert, SlotFormat::Rank(rank));
                if cache.slots.contains_key(&key) {
                    continue;
                }
                let mut free = 0;
                cuda(unsafe { ds_moe_free_bytes(cache.device, &mut free) })?;
                if !preload_fits(cache.resident_bytes, capacity, free) {
                    return Ok(loaded);
                }
                let slot = match cache.upload_slot(weights, key, capacity, 0) {
                    Ok(slot) => slot,
                    Err(e)
                        if e.downcast_ref::<MoeCudaError>()
                            .is_some_and(|e| e.0 == CUDA_ERROR_MEMORY_ALLOCATION) =>
                    {
                        return Ok(loaded)
                    }
                    Err(e) => return Err(e),
                };
                // Age zero keeps unsolicited weights behind routed entries of the
                // same frequency. A format sibling can already have a routed count.
                let frequency = cache
                    .frequencies
                    .as_ref()
                    .map_or(0, |f| f[frequency_index(key)]);
                cache.eviction_order.insert((frequency, 0, key));
                cache.slots.insert(key, (slot, 0));
                loaded += 1;
            }
        }
        Ok(loaded)
    })();
    // The staged uploader may still own either pinned slab when the last
    // copy successfully enqueues, or when a later checkpoint fails to load.
    let drained = cuda(unsafe { cudaStreamSynchronize(ptr::null_mut()) });
    result.and_then(|loaded| drained.map(|_| loaded))
}

pub(crate) fn preload_encoder(
    rt: &mut Runtime,
    devices: [i32; 2],
    layers: usize,
) -> Result<[usize; 2]> {
    let active = rt.device;
    let result = (|| {
        for device in devices {
            rt.moe_cache.device(device)?;
        }
        let [Some(first), Some(second)] = rt
            .moe_cache
            .devices
            .get_disjoint_mut([&devices[0], &devices[1]])
        else {
            bail!("missing encoder rank cache");
        };
        let weights = &*rt.weights;
        let mut loaded = [0; 2];
        let [left, right] = &mut loaded;
        run_ranks(
            std::thread::Builder::new(),
            || {
                *right = preload_rank(second, weights, layers, 1)?;
                Ok(())
            },
            || {
                *left = preload_rank(first, weights, layers, 0)?;
                Ok(())
            },
        )?;
        Ok(loaded)
    })();
    let restored = cuda(unsafe { cudaSetDevice(active) });
    result.and_then(|loaded| restored.map(|_| loaded))
}

pub(super) fn eligible(
    enabled: bool,
    prefill: bool,
    verify: bool,
    layer: Layer,
    batch: usize,
) -> bool {
    enabled && prefill && !verify && batch >= 128 && matches!(layer, Layer::Backbone(_))
}

// Exclusive mutable access to one rank's cache moves to a scoped launch
// thread. The thread selects that CUDA device before accessing any allocation.
// CUDA resources are owned by the device/context, not the creating OS thread.
unsafe impl Send for DeviceCache {}

#[derive(Default)]
pub(super) struct Workspace {
    // On this layer's peer: replicated input followed by all routed outputs.
    // The reduced output overwrites the replica once the expert kernels finish.
    io: usize,
    io_bytes: usize,
    // On the layer owner: one bounded, portable host bounce and one local
    // receive tile. Their lifetimes extend across calls and error returns.
    bounce: usize,
    receive: usize,
    // Runtime has one fixed partner per device. Probe both directions once.
    peer_available: Option<bool>,
}

impl Workspace {
    pub(super) fn free(&mut self, device: i32) {
        unsafe {
            ds_moe_free(device, self.io as *mut c_void);
            ds_moe_free(device, self.receive as *mut c_void);
            if self.bounce != 0 {
                cudaFreeHost(self.bounce as *mut c_void);
            }
        }
    }

    fn grow_io(&mut self, device: i32, bytes: usize) -> Result<()> {
        if self.io_bytes < bytes {
            cuda(unsafe { cudaSetDevice(device) })?;
            unsafe { ds_moe_free(device, self.io as *mut c_void) };
            self.io = 0;
            self.io_bytes = 0;
            let mut io = ptr::null_mut();
            cuda(unsafe { cudaMalloc(&mut io, bytes) })?;
            self.io = io as usize;
            self.io_bytes = bytes;
        }
        Ok(())
    }

    fn direct_dma(&mut self, enabled: bool, owner: i32, peer: i32) -> Result<bool> {
        if !enabled {
            return Ok(false);
        }
        if let Some(available) = self.peer_available {
            return Ok(available);
        }
        let mut available = 0;
        cuda(unsafe { ds_recipe_enable_peer_dma(owner, peer, &mut available) })?;
        self.peer_available = Some(available != 0);
        Ok(available != 0)
    }

    fn prepare_transfer(&mut self, device: i32) -> Result<()> {
        cuda(unsafe { cudaSetDevice(device) })?;
        if self.bounce == 0 {
            let mut bounce = ptr::null_mut();
            // cudaHostAllocPortable: both contexts DMA to/from this host page.
            cuda(unsafe { cudaHostAlloc(&mut bounce, COPY_FLOATS * size_of::<f32>(), 1) })?;
            self.bounce = bounce as usize;
        }
        if self.receive == 0 {
            let mut receive = ptr::null_mut();
            cuda(unsafe { cudaMalloc(&mut receive, COPY_FLOATS * size_of::<f32>()) })?;
            self.receive = receive as usize;
        }
        Ok(())
    }
}

fn group_len(routes: &[Route], capacity: usize) -> usize {
    // Preserve whole experts whenever one fits in a group, including hot
    // experts that span several 128-row S8B jobs.
    let mut rows = 0;
    for group in routes.chunk_by(|a, b| a.expert == b.expert).take(capacity) {
        if rows + group.len() > GROUP_ROWS {
            return if rows == 0 { GROUP_ROWS } else { rows };
        }
        rows += group.len();
    }
    rows
}

fn run_rank(
    cache: &mut DeviceCache,
    weights: &Weights,
    layout: (usize, u8),
    mut routes: &[Route],
    input: usize,
    output: usize,
) -> Result<()> {
    cuda(unsafe { cudaSetDevice(cache.device) })?;
    let capacity = cache.capacity(RANK_SLOT_BYTES)?;
    let mut jobs = Vec::new();
    let mut descriptors = Vec::with_capacity(GROUP_ROWS.min(routes.len()));
    while !routes.is_empty() {
        let (group, rest) = routes.split_at(group_len(routes, capacity));
        jobs.clear();
        descriptors.clear();
        let group_start = cache.clock;
        for expert in group.chunk_by(|a, b| a.expert == b.expert) {
            let slot = cache.acquire(
                weights,
                Key(
                    Layer::Backbone(layout.0),
                    expert[0].expert,
                    SlotFormat::Rank(layout.1),
                ),
                capacity,
                group_start,
            )?;
            for job in expert.chunks(128) {
                jobs.push(Job {
                    expert: slot,
                    begin: descriptors.len() as i32,
                    count: job.len() as i32,
                });
                for route in job {
                    descriptors.push(RoutedRow {
                        expert: slot,
                        token: route.token.try_into()?,
                        output: (route.token * TOP_K + route.rank).try_into()?,
                        weight: route.weight,
                    });
                }
            }
        }
        let mut scratch = cache.scratch as *mut c_void;
        let status = unsafe {
            ds_recipe_rank_group(
                &mut scratch,
                &mut cache.scratch_bytes,
                jobs.as_ptr(),
                jobs.len() as i32,
                descriptors.as_ptr(),
                group.len() as i32,
                input as *const f32,
                output as *mut f32,
            )
        };
        cache.scratch = scratch as usize;
        cuda(status)?;
        routes = rest;
    }
    Ok(())
}

fn run_ranks(
    thread: std::thread::Builder,
    peer: impl FnOnce() -> Result<()> + Send,
    local: impl FnOnce() -> Result<()>,
) -> Result<()> {
    std::thread::scope(|scope| {
        let worker = thread
            .spawn_scoped(scope, peer)
            .context("starting TP2 peer worker")?;
        let local = local();
        let peer = worker
            .join()
            .map_err(|_| anyhow::anyhow!("peer expert worker panicked"))?;
        local.and(peer)
    })
}

pub(super) fn dispatch(
    rt: &mut Runtime,
    layer: usize,
    x: &Dev,
    weighted: &mut Dev,
    shared: &mut Dev,
    routes: &[Route],
) -> Result<()> {
    let owner = rt.device;
    let peer = rt.peer_device();
    let len = x.len();
    let io_bytes = (len + weighted.len())
        .checked_mul(size_of::<f32>())
        .context("rank I/O overflow")?;
    let old_bytes = rt.moe_cache.device(peer)?.rank_workspace.io_bytes;
    let result = (|| {
        if old_bytes < io_bytes {
            rt.moe_cache.reserve(peer, io_bytes - old_bytes)?;
        }
        rt.moe_cache
            .device(peer)?
            .rank_workspace
            .grow_io(peer, io_bytes)?;
        let cache = rt.moe_cache.device(owner)?;
        let direct = cache
            .rank_workspace
            .direct_dma(cache.tp2_peer_dma, owner, peer)?;
        if !direct && cache.rank_workspace.receive == 0 {
            rt.moe_cache
                .reserve(owner, COPY_FLOATS * size_of::<f32>())?;
        }
        let owner_workspace = &mut rt.moe_cache.device(owner)?.rank_workspace;
        if !direct {
            owner_workspace.prepare_transfer(owner)?;
        }
        let bounce = owner_workspace.bounce as *mut f32;
        let receive = owner_workspace.receive as *mut f32;
        let [Some(local_cache), Some(peer_cache)] =
            rt.moe_cache.devices.get_disjoint_mut([&owner, &peer])
        else {
            bail!("missing TP2 device cache");
        };
        let replica = peer_cache.rank_workspace.io as *mut f32;
        let partials = unsafe { replica.add(len) };
        cuda(unsafe {
            ds_recipe_broadcast(
                x.as_ptr(),
                owner,
                replica,
                peer,
                bounce,
                if direct {
                    weighted.as_mut_ptr().cast()
                } else {
                    receive.cast()
                },
                partials.cast(),
                len,
                local_cache.tp2_packed_input.into(),
                direct.into(),
            )
        })?;

        let weights = &*rt.weights;
        let peer_input = replica as usize;
        let peer_output = partials as usize;
        let before = [local_cache.uploads, peer_cache.uploads];
        run_ranks(
            std::thread::Builder::new(),
            move || {
                run_rank(
                    peer_cache,
                    weights,
                    (layer, 1),
                    routes,
                    peer_input,
                    peer_output,
                )?;
                cuda(unsafe {
                    ds_recipe_sum_rank(peer_output as *const f32, peer_input as *mut f32, len)
                })
            },
            || {
                run_rank(
                    local_cache,
                    weights,
                    (layer, 0),
                    routes,
                    x.as_ptr() as usize,
                    weighted.as_mut_ptr() as usize,
                )
            },
        )?;
        if direct {
            cuda(unsafe {
                ds_recipe_collect_dma(
                    weighted.as_mut_ptr(),
                    shared.as_mut_ptr(),
                    owner,
                    replica,
                    peer,
                    len,
                )
            })?;
        } else {
            for start in (0..len).step_by(COPY_FLOATS) {
                let count = COPY_FLOATS.min(len - start);
                cuda(unsafe {
                    ds_recipe_transfer(receive, owner, replica.add(start), peer, bounce, count)
                })?;
                cuda(unsafe {
                    ds_recipe_finish_ranks(
                        weighted.as_ptr(),
                        receive,
                        shared.as_mut_ptr(),
                        start,
                        count,
                    )
                })?;
            }
        }
        let owner_cache = rt.moe_cache.device(owner)?;
        if owner_cache.metrics {
            let uploads =
                owner_cache.uploads - before[0] + rt.moe_cache.device(peer)?.uploads - before[1];
            let metrics = &mut rt.moe_cache.metrics;
            if layer == 0 {
                *metrics = Metrics::default();
            }
            metrics.routes += routes.len();
            metrics.unique += routes.chunk_by(|a, b| a.expert == b.expert).count();
            metrics.uploads += uploads;
            metrics.capacity_queries += 2;
            if layer == 39 {
                eprintln!("moe pass=target-tp2 peer_dma={direct} rows={} routes={} unique_layer_experts={} rank_uploads={} expert_h2d_bytes={} capacity_queries={}",
                          len/HIDDEN, metrics.routes, metrics.unique, metrics.uploads, metrics.uploads*RANK_SLOT_BYTES, metrics.capacity_queries);
            }
        }
        Ok(())
    })();
    // Every layer caller expects its owner to remain current after FFN work.
    let restored = cuda(unsafe { cudaSetDevice(owner) });
    result.and(restored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_preload_preserves_resident_bytes_and_headroom() {
        let required = EXPERT_HEADROOM + RANK_SLOT_BYTES;
        assert!(preload_fits(0, 1, required));
        assert!(!preload_fits(0, 1, required - 1));
        assert!(!preload_fits(RANK_SLOT_BYTES, 1, required));
        assert!(!preload_fits(SLOT_BYTES, 2, required));
        assert!(!preload_fits(SLOT_BYTES, 1, required));
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs in CUDA_VISIBLE_DEVICES"]
    fn startup_preload_reuses_exact_shards_without_routed_accesses() -> Result<()> {
        unsafe extern "C" {
            fn cudaGetDevice(device: *mut i32) -> i32;
            fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
        }
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        for device in [0, 1] {
            let cache = rt.moe_cache.device(device)?;
            cache.limit_bytes = if device == 0 {
                2 * SLOT_BYTES
            } else {
                RANK_SLOT_BYTES - 1
            };
            cache.frequencies = Some(vec![0; EXPERT_KEYS].into_boxed_slice());
            cache.stage_rank_weights = true;
        }
        rt.layer(20)?;
        let mut active = -1;
        // The peer can still route on demand with the one-slot minimum, but
        // its configured budget cannot hold any unsolicited rank weights.
        assert_eq!(preload_encoder(&mut rt, [0, 1], 20)?, [3, 0]);
        assert_eq!(rt.moe_cache.device(1)?.capacity(RANK_SLOT_BYTES)?, 1);
        rt.moe_cache.device(1)?.limit_bytes = RANK_SLOT_BYTES;
        assert_eq!(preload_encoder(&mut rt, [0, 1], 20)?, [0, 1]);
        let loaded = [3, 1];
        cuda(unsafe { cudaGetDevice(&mut active) })?;
        assert_eq!(active, 1);
        for (rank, count) in loaded.into_iter().enumerate() {
            let device = rank as i32;
            cuda(unsafe { cudaSetDevice(device) })?;
            let cache = rt.moe_cache.device(device)?;
            assert_eq!(cache.clock, 0);
            assert!(cache.frequencies.as_ref().unwrap().iter().all(|&n| n == 0));
            assert_eq!(cache.slots.len(), count);
            assert_eq!(cache.resident_bytes, count * RANK_SLOT_BYTES);
            let key = |expert| Key(Layer::Backbone(0), expert, SlotFormat::Rank(rank as u8));
            let capacity = cache.capacity(RANK_SLOT_BYTES)?;
            assert_eq!(capacity, count);

            // Compare every uploaded byte to the direct, unstaged routed path.
            let mut reference = DeviceCache::new(device)?;
            reference.stage_rank_weights = false;
            let mut actual = vec![0u8; RANK_SLOT_BYTES];
            let mut expected = vec![0u8; RANK_SLOT_BYTES];
            for expert in 0..count {
                let (slot, age) = cache.slots[&key(expert)];
                assert_eq!(age, 0);
                let direct = reference.acquire(&rt.weights, key(expert), 1, reference.clock)?;
                for (dst, src) in [(&mut actual, slot), (&mut expected, direct)] {
                    cuda(unsafe {
                        cudaMemcpy(dst.as_mut_ptr().cast(), src as *const c_void, dst.len(), 2)
                    })?;
                }
                ensure!(
                    actual == expected,
                    "wrong preloaded bytes on rank {rank} expert {expert}"
                );
            }
            let first = cache.slots[&key(0)].0;
            assert_eq!(
                cache.acquire(&rt.weights, key(0), capacity, cache.clock)?,
                first
            );
            assert_eq!(cache.uploads, count);
            assert_eq!(
                cache.frequencies.as_ref().unwrap()[frequency_index(key(0))],
                1
            );
            cache.acquire(&rt.weights, key(count), capacity, cache.clock)?;
            let victim = if count > 1 { 1 } else { 0 };
            assert!(!cache.slots.contains_key(&key(victim)));
            assert_eq!(cache.uploads, count + 1);
            assert_eq!(cache.slots.len(), count);
        }
        rt.layer(20)?;
        assert_eq!(preload_encoder(&mut rt, [0, 1], 20)?, [0; 2]);
        cuda(unsafe { cudaGetDevice(&mut active) })?;
        assert_eq!(active, 1);
        for (device, previous) in loaded.into_iter().enumerate() {
            let cache = rt.moe_cache.device(device as i32)?;
            assert_eq!(cache.clock, 2);
            assert_eq!(cache.uploads, previous + 1);
        }
        Ok(())
    }

    #[test]
    fn tp2_only_runs_for_wide_target_prefill() {
        let target = Layer::Backbone(20);
        assert!(eligible(true, true, false, target, 128));
        for (enabled, prefill, verify, layer, batch) in [
            (true, true, false, target, 127),
            (true, true, true, target, 128),
            (true, false, false, target, 8192),
            (false, true, false, target, 8192),
            (true, true, false, Layer::Draft(1), 8192),
        ] {
            assert!(!eligible(enabled, prefill, verify, layer, batch));
        }
    }

    #[test]
    fn tp2_grouping_respects_slot_capacity_and_job_boundaries() {
        let routes: Vec<_> = [127, 129, 1900, 2049]
            .into_iter()
            .enumerate()
            .flat_map(|(expert, count)| {
                (0..count).map(move |token| Route {
                    expert,
                    token,
                    ..Route::default()
                })
            })
            .collect();
        assert_eq!(group_len(&routes, 1), 127);
        assert_eq!(group_len(&routes, 384), 256);
        assert_eq!(group_len(&routes[256..], 384), 1900);
        assert_eq!(group_len(&routes[2156..], 384), GROUP_ROWS);
        unsafe extern "C" {
            fn ds_recipe_rank_slot_bytes() -> usize;
        }
        assert_eq!(RANK_SLOT_BYTES, unsafe { ds_recipe_rank_slot_bytes() });
        assert_eq!(
            DeviceCache::available_slots(
                2 * SLOT_BYTES,
                512 << 20,
                2 * SLOT_BYTES,
                RANK_SLOT_BYTES
            ),
            3
        );
    }

    #[test]
    fn peer_thread_exhaustion_fails_only_this_dispatch() {
        // This stack exceeds the workstation's userspace address range. The
        // actual OS spawn fails before either rank's closure is entered.
        let unavailable = std::thread::Builder::new().stack_size(usize::MAX / 4);
        let error = run_ranks(
            unavailable,
            || panic!("unexpected peer launch"),
            || panic!("unexpected local launch"),
        )
        .unwrap_err();
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        run_ranks(std::thread::Builder::new(), || Ok(()), || Ok(())).unwrap();
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an inference GPU"]
    fn failed_rank_allocation_leaves_next_cuda_launch_clean() -> Result<()> {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
            fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
            fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        }
        let device = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut cache = DeviceCache::new(device)?;
        let workspace = &mut cache.rank_workspace;
        workspace.grow_io(device, 8)?;
        let (mut free, mut total) = (0, 0);
        cuda(unsafe { cudaMemGetInfo(&mut free, &mut total) })?;
        // Exceeds the physical device's entire memory; this cannot consume
        // memory belonging to another GPU client.
        let error = workspace.grow_io(device, total + 1).unwrap_err();
        ensure!(
            error.to_string().contains("CUDA MoE error 2:"),
            "expected CUDA OOM, received {error}"
        );
        assert_eq!((workspace.io, workspace.io_bytes), (0, 0));
        assert_eq!(unsafe { cudaPeekAtLastError() }, 0);
        workspace.grow_io(device, 8)?;
        let values = workspace.io as *mut f32;
        cuda(unsafe { cudaMemset(values.cast(), 0, 8) })?;
        cuda(unsafe { ds_moe_shared_swiglu(values, values.add(1), 1) })?;
        cuda(unsafe { cudaStreamSynchronize(ptr::null_mut()) })
    }

    // Independent single-device reference. Gather the expert's selected input
    // rows on the CPU, run each shard alone, then accumulate in route order.
    // No production dispatch descriptors, bridge, or GPU reduction are reused.
    fn reference(
        rt: &mut Runtime,
        layer: usize,
        input: &[f32],
        x: &Dev,
        batch: usize,
    ) -> Result<Vec<f32>> {
        let base = Layer::Backbone(layer).base();
        let logits = rt.linear_dev(&format!("{base}.gate.weight"), x, batch, 1)?;
        let logits = rt.to_host(&logits)?;
        let bias = rt.weights.vector(&format!("{base}.gate.bias"))?;
        let routes = routes::<EXPERTS, TOP_K>(&logits, &bias, batch, false)?;
        let mut sums = [vec![0f32; input.len()], vec![0.; input.len()]];
        let mut cache = DeviceCache::new(rt.device)?;
        cache.stage_rank_weights = false;
        for (rank, sum) in sums.iter_mut().enumerate() {
            for expert in routes.chunk_by(|a, b| a.expert == b.expert) {
                let key = Key(
                    Layer::Backbone(layer),
                    expert[0].expert,
                    SlotFormat::Rank(rank as u8),
                );
                let slot = cache.acquire(&rt.weights, key, 1, u64::MAX)?;
                let gathered: Vec<_> = expert
                    .iter()
                    .flat_map(|r| {
                        input[r.token * HIDDEN..(r.token + 1) * HIDDEN]
                            .iter()
                            .copied()
                    })
                    .collect();
                let gx = rt.to_device(&gathered)?;
                let mut gy = rt.buffer(gathered.len())?;
                let mut scratch = cache.scratch as *mut c_void;
                let rows: Vec<_> = expert
                    .iter()
                    .enumerate()
                    .map(|(i, r)| RoutedRow {
                        expert: slot,
                        token: i as i32,
                        output: i as i32,
                        weight: r.weight,
                    })
                    .collect();
                let jobs: Vec<_> = (0..rows.len())
                    .step_by(128)
                    .map(|i| Job {
                        expert: slot,
                        begin: i as i32,
                        count: (rows.len() - i).min(128) as i32,
                    })
                    .collect();
                let status = unsafe {
                    ds_recipe_rank_group(
                        &mut scratch,
                        &mut cache.scratch_bytes,
                        jobs.as_ptr(),
                        jobs.len() as i32,
                        rows.as_ptr(),
                        rows.len() as i32,
                        gx.as_ptr(),
                        gy.as_mut_ptr(),
                    )
                };
                cache.scratch = scratch as usize;
                cuda(status)?;
                for (r, y) in expert.iter().zip(rt.to_host(&gy)?.chunks_exact(HIDDEN)) {
                    for (v, value) in sum[r.token * HIDDEN..(r.token + 1) * HIDDEN]
                        .iter_mut()
                        .zip(y)
                    {
                        *v += value;
                    }
                }
            }
        }
        let shared_y = shared(rt, &base, x, batch)?;
        let mut result = rt.to_host(&shared_y)?;
        for ((y, a), b) in result.iter_mut().zip(&sums[0]).zip(&sums[1]) {
            *y = half::bf16::from_f32((a + b) + *y).to_f32();
        }
        Ok(result)
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs in CUDA_VISIBLE_DEVICES"]
    fn tp2_prefill_matches_rank_reference() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        // The second batch needs multiple 2 MiB tiles even for FP16 input.
        // Repeated directions share the same ordinary cache with full decode slots.
        for (layer, batch) in [(0, 128), (20, 257)] {
            rt.layer(layer)?;
            rt.target_prefill = true;
            let input: Vec<_> = (0..batch * HIDDEN)
                .map(|i| ((i % HIDDEN) as f32 * 0.057 + ((i / HIDDEN) % 3) as f32 * 0.12).cos())
                .collect();
            let x = rt.to_device(&input)?;
            let expected = reference(&mut rt, layer, &input, &x, batch)?;
            for d in [0, 1] {
                let c = rt.moe_cache.device(d)?;
                c.tp2_prefill = true;
                c.stage_rank_weights = true;
                c.limit_bytes = 2 * SLOT_BYTES; // three rank slots; more than six selected experts.
            }
            assert!(input.iter().any(|&v| half::f16::from_f32(v).to_f32() != v));
            for (direct, packed) in [(false, false), (false, true), (true, false), (true, true)] {
                let c = rt.moe_cache.device(rt.device)?;
                c.tp2_packed_input = packed;
                c.tp2_peer_dma = direct;
                let out = forward_dev(&mut rt, layer, &x, batch)?;
                let actual = rt.to_host(&out)?;
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                ensure!(
                    mismatch.is_none(),
                    "TP2 layer {layer} direct={direct} packed={packed} first unequal result at {mismatch:?}"
                );
            }
            let available = rt
                .moe_cache
                .device(rt.device)?
                .rank_workspace
                .peer_available;
            ensure!(available.is_some(), "requested peer DMA was not probed");
            if std::env::var_os("DS4_TEST_REQUIRE_PEER_DMA").is_some_and(|v| v == "1") {
                ensure!(
                    available == Some(true),
                    "test requires a bidirectional peer-DMA pair"
                );
            }
            for (d, rank) in [(rt.device, 0), (rt.peer_device(), 1)] {
                let c = rt.moe_cache.device(d)?;
                ensure!(
                    c.slots
                        .keys()
                        .any(|&Key(l, _, f)| l == Layer::Backbone(layer)
                            && f == SlotFormat::Rank(rank)),
                    "missing rank {rank} slots"
                );
                ensure!(c.resident_bytes <= c.limit_bytes, "slot budget exceeded");
            }
            // A verification call with the switch still on must use full experts.
            rt.target_prefill = false;
            rt.verification = true;
            let decode_x = rt.to_device(&input[..HIDDEN])?;
            let decode = forward_dev(&mut rt, layer, &decode_x, 1)?;
            let enabled = rt.to_host(&decode)?;
            rt.moe_cache.device(rt.device)?.tp2_prefill = false;
            let decode = forward_dev(&mut rt, layer, &decode_x, 1)?;
            assert_eq!(enabled, rt.to_host(&decode)?);
            rt.verification = false;
        }
        Ok(())
    }
}
