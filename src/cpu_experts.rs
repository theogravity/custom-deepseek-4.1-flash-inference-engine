//! Sparse CPU verification using the model recipe's native EXL3 host kernel.
//! One owner per inference worker. Rayon workers only see the active CEPF job.
use super::{
    checkpoint_parts, frequency_index, switch, Dev, Layer, Route, Runtime, EXPERT_KEYS, HIDDEN,
    INTER, TOP_K,
};
use anyhow::{ensure, Context, Result};
use half::f16;
use libloading::Library;
use rayon::ThreadPool;
use std::{ffi::c_void, path::Path, ptr};

use std::sync::atomic::{AtomicU64, Ordering};
// CEPF initializes process-global lookup tables during its first create. Serialize
// constructors even when distinct runtimes open the same pack concurrently.
static CREATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static JOB_COUNT: AtomicU64 = AtomicU64::new(0);
static ROW_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) fn counters() -> (u64, u64) {
    (
        JOB_COUNT.load(Ordering::Relaxed),
        ROW_COUNT.load(Ordering::Relaxed),
    )
}

const ROWS: usize = 8;
const JOBS: usize = 4;
const PER_EXPERT: usize = 2;

#[repr(C)]
struct Linear {
    trellis: *const c_void,
    suh: *const c_void,
    svh: *const c_void,
    input: i32,
    output: i32,
    layout: i32,
    pad: i32,
}
#[repr(C)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}
#[repr(C)]
struct Job {
    experts: *const *const Expert,
    offsets: *const i32,
    input: *const f16,
    weights: *const f32,
    output: *mut f16,
    abort: *const i32,
    stall: *const i32,
    limit: f32,
    n_experts: i32,
    rows: i32,
    stall_tid: i32,
    pad: i32,
}
type Create = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32) -> *mut c_void;
type Begin = unsafe extern "C" fn(*mut c_void, *const Job) -> i32;
type Work = unsafe extern "C" fn(*mut c_void, i32);
type Destroy = unsafe extern "C" fn(*mut c_void);

pub(super) struct CpuExperts {
    pool: ThreadPool,
    ctx: *mut c_void,
    begin: Begin,
    work: Work,
    destroy: Destroy,
    _library: Library,
    deferred: Box<[bool]>,
}
impl Drop for CpuExperts {
    fn drop(&mut self) {
        // All worker tasks join their scope before the caller can drop the context.
        unsafe { (self.destroy)(self.ctx) }
    }
}
impl CpuExperts {
    pub(super) fn open(root: &Path) -> Result<Option<Self>> {
        if !switch("DS4_CPU_EXPERTS", false)? {
            return Ok(None);
        }
        let dir = root.join("recipe/cpuexp/build");
        // The recipe builds this library with -march=native. Validate its provenance
        // before even running the library's own ISA-dependent tile self-test.
        let info = std::fs::read_to_string(dir.join("BUILD_INFO"))?;
        let model = info
            .lines()
            .find_map(|l| l.strip_prefix("cpu_model="))
            .context("missing CPU build identity")?;
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")?;
        let host = cpuinfo
            .lines()
            .find_map(|l| {
                l.split_once(':')
                    .filter(|(key, _)| key.trim() == "model name")
            })
            .context("unknown CPU model")?
            .1
            .trim();
        ensure!(
            model.trim() == host,
            "host-expert library was built for another CPU"
        );
        let flags = cpuinfo
            .lines()
            .find_map(|l| l.split_once(':').filter(|(key, _)| key.trim() == "flags"))
            .context("unknown CPU capabilities")?
            .1;
        ensure!(
            [
                "avx512f",
                "avx512bw",
                "avx512_vnni",
                "avx512vbmi",
                "avx512_vbmi2",
                "f16c"
            ]
            .iter()
            .all(|f| flags.split_whitespace().any(|v| v == *f)),
            "missing host expert CPU instructions"
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(10)
            .thread_name(|n| format!("cpu-expert-{n}"))
            .build()?;
        unsafe {
            let library = Library::new(dir.join("libcpuexp_rt.so"))?;
            let version = *library.get::<unsafe extern "C" fn() -> i32>(b"cert_version")?;
            ensure!(version() == 2, "unsupported host expert ABI");
            let create = *library.get::<Create>(b"cepf_create")?;
            let begin = *library.get::<Begin>(b"cepf_begin")?;
            let work = *library.get::<Work>(b"cepf_work")?;
            let destroy = *library.get::<Destroy>(b"cepf_destroy")?;
            let _initializing = CREATE_LOCK
                .lock()
                .map_err(|_| anyhow::anyhow!("host expert initialization lock poisoned"))?;
            let ctx = create(
                HIDDEN as i32,
                INTER as i32,
                ROWS as i32,
                JOBS as i32,
                PER_EXPERT as i32,
                10,
            );
            ensure!(!ctx.is_null(), "host expert context initialization failed");
            Ok(Some(Self {
                pool,
                ctx,
                begin,
                work,
                destroy,
                _library: library,
                deferred: vec![false; EXPERT_KEYS].into_boxed_slice(),
            }))
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch(
        &mut self,
        rt: &mut Runtime,
        layer: Layer,
        routes: &[Route],
        x: &Dev,
        weighted: &mut Dev,
        parallel: bool,
        gpu: impl FnOnce(&mut Runtime, &[Route], &mut Dev) -> Result<()>,
    ) -> Result<()> {
        // Keep GPU capacity growing normally. At capacity, bypass a small
        // nonresident expert once, then admit it on its next eligible miss.
        let mut cold = Vec::new();
        for group in routes.chunk_by(|a, b| a.expert == b.expert) {
            let expert = group[0].expert;
            let device = if parallel && expert % 2 == 1 {
                rt.peer_device()
            } else {
                rt.device
            };
            let Some(cache) = rt.moe_cache.devices.get(&device) else {
                continue;
            };
            if cold.len() == JOBS
                || group.len() > PER_EXPERT
                || cache.slots.len() < cache.last_capacity
                || cache.slots.contains_key(&(layer, expert))
            {
                continue;
            }
            let deferred = &mut self.deferred[frequency_index((layer, expert))];
            *deferred = !*deferred;
            if *deferred {
                cold.push(group);
            }
        }
        if cold.is_empty() {
            return gpu(rt, routes, weighted);
        }
        let mut experts = Vec::with_capacity(cold.len());
        let mut offsets = vec![0i32];
        let mut input = Vec::new();
        let mut weights = Vec::new();
        let mut dest = Vec::new();
        for group in &cold {
            let parts = checkpoint_parts(&rt.weights, (layer, group[0].expert))?;
            let linear = |p: usize, k: usize, n: usize| Linear {
                trellis: parts[p],
                suh: parts[3 + 2 * p],
                svh: parts[4 + 2 * p],
                input: k as i32,
                output: n as i32,
                layout: 0,
                pad: 0,
            };
            experts.push(Expert {
                gate: linear(0, HIDDEN, INTER),
                up: linear(1, HIDDEN, INTER),
                down: linear(2, INTER, HIDDEN),
            });
            for row in *group {
                input.extend(
                    rt.to_host_range(x, row.token * HIDDEN, HIDDEN)?
                        .into_iter()
                        .map(f16::from_f32),
                );
                weights.push(f16::from_f32(row.weight).to_f32());
                dest.push(row.token * TOP_K + row.rank);
            }
            offsets.push(dest.len() as i32);
        }
        let hot: Vec<_> = routes
            .iter()
            .copied()
            .filter(|r| !cold.iter().any(|g| g[0].expert == r.expert))
            .collect();
        let pointers: Vec<_> = experts.iter().map(ptr::from_ref).collect();
        let mut output = vec![f16::ZERO; dest.len() * HIDDEN];
        let job = Job {
            experts: pointers.as_ptr(),
            offsets: offsets.as_ptr(),
            input: input.as_ptr(),
            weights: weights.as_ptr(),
            output: output.as_mut_ptr(),
            abort: ptr::null(),
            stall: ptr::null(),
            limit: 10.,
            n_experts: experts.len() as i32,
            rows: dest.len() as i32,
            stall_tid: -1,
            pad: 0,
        };
        ensure!(
            unsafe { (self.begin)(self.ctx, &job) } == 0,
            "invalid host expert job"
        );
        let addr = self.ctx as usize;
        let work = self.work;
        // CEPF retains the above arrays until all unique worker IDs have returned.
        // The inference thread continues submitting resident work on its CUDA streams.
        self.pool.in_place_scope(|scope| {
            scope.spawn_broadcast(move |_, thread| unsafe {
                work(addr as *mut c_void, thread.index() as i32)
            });
            gpu(rt, &hot, weighted)
        })?;
        let mut wide = vec![0.; HIDDEN];
        for (&slot, row) in dest.iter().zip(output.as_chunks::<HIDDEN>().0) {
            for (w, h) in wide.iter_mut().zip(row) {
                *w = h.to_f32();
            }
            rt.write_host_range(weighted, slot * HIDDEN, &wide)?;
        }
        JOB_COUNT.fetch_add(experts.len() as u64, Ordering::Relaxed);
        ROW_COUNT.fetch_add(dest.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::run_routed_gpu;
    #[test]
    #[ignore = "requires the host EXL3 pack, DS4_CPU_EXPERTS=1 and two idle inference GPUs"]
    fn fixed_routes_match_reference_within_cpu_precision() -> Result<()> {
        let pack = std::env::var("DS4_TEST_PACK")?;
        let mut rt = Runtime::open(Path::new(&pack), [0, 1])?;
        rt.verification = true;
        let mut cpu = rt.moe_cache.cpu.take().context("enable DS4_CPU_EXPERTS")?;
        // Routes with two weights, a repeated expert, separate rank slots, and an
        // untouched token. The remaining slots must retain their sentinel.
        let routes = [
            Route {
                expert: 6,
                token: 1,
                rank: 0,
                weight: 0.8,
            },
            Route {
                expert: 17,
                token: 0,
                rank: 0,
                weight: 0.31,
            },
            Route {
                expert: 17,
                token: 1,
                rank: 2,
                weight: 0.77,
            },
            Route {
                expert: 237,
                token: 0,
                rank: 3,
                weight: 0.53,
            },
        ];
        let input: Vec<_> = (0..3 * HIDDEN)
            .map(|i| (i as f32 * 0.073).cos() + (i as f32 * 0.013).sin() * 0.25)
            .collect();
        for index in [0, 23, 39] {
            rt.layer(index)?;
            let layer = Layer::Backbone(index);
            let x = rt.to_device(&input)?;
            rt.moe_cache
                .device(rt.device)?
                .acquire(&rt.weights, (layer, 6), 4, u64::MAX)?;

            let initial = vec![3.25; 3 * TOP_K * HIDDEN];
            let mut actual = rt.to_device(&initial)?;
            let mut expected = rt.to_device(&initial)?;
            // Force capacity pressure in this fixture only; actual serving learns
            // the capacity of each GPU from real memory availability.
            rt.moe_cache.device(rt.device)?.last_capacity = 0;
            cpu.dispatch(
                &mut rt,
                layer,
                &routes,
                &x,
                &mut actual,
                false,
                |rt, hot, dst| {
                    ensure!(
                        hot.len() == 1 && hot[0].expert == 6,
                        "only the resident expert should execute on GPU"
                    );
                    run_routed_gpu::<TOP_K>(rt, layer, hot, &routes, &x, dst, false, false, None)
                },
            )?;
            rt.moe_cache.device(rt.device)?.last_capacity = 0;
            cpu.dispatch(
                &mut rt,
                layer,
                &routes,
                &x,
                &mut expected,
                false,
                |rt, hot, dst| {
                    ensure!(
                        hot.len() == routes.len(),
                        "expected repeat misses to promote"
                    );
                    run_routed_gpu::<TOP_K>(rt, layer, hot, &routes, &x, dst, false, false, None)
                },
            )?;
            let av = rt.to_host(&actual)?;
            let ev = rt.to_host(&expected)?;
            for (slot, (a, e)) in av
                .as_chunks::<HIDDEN>()
                .0
                .iter()
                .zip(ev.as_chunks::<HIDDEN>().0)
                .enumerate()
            {
                if slot == TOP_K {
                    ensure!(a == e, "resident route changed");
                } else if routes.iter().any(|r| r.token * TOP_K + r.rank == slot) {
                    let square = |v: f32| f64::from(v).powi(2);
                    let norm = e.iter().map(|x| square(*x)).sum::<f64>();
                    let err =
                        (a.iter().zip(e).map(|(v, w)| square(v - w)).sum::<f64>() / norm).sqrt();
                    eprintln!("host_expert_relative_l2 slot={slot} error={err:.7}");
                    ensure!(norm > 0. && err < 0.01, "expert error {err}");
                } else {
                    ensure!(
                        a == e && a.iter().all(|v| *v == 3.25),
                        "overwrote unused route slot"
                    )
                }
            }
        }
        Ok(())
    }
}
