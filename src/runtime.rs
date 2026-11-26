#[path = "parallel_dense.rs"]
mod parallel_dense;
use crate::{moe::MoeCache, weights::Weights};
use anyhow::{ensure, Context, Result};
use std::{
    collections::HashMap,
    ffi::{c_char, c_int, c_void, CStr},
    path::Path,
    ptr,
    sync::{Arc, Mutex},
};
extern "C" {
    fn ds4_sample(
        device: i32,
        logits: *const f32,
        scratch: *mut f32,
        ids: *mut f32,
        rows: i32,
        cols: i32,
        temperature: f32,
        seed: u64,
    ) -> i32;
    fn ds4_argmax(
        device: i32,
        logits: *const f32,
        scratch: *mut f32,
        ids: *mut f32,
        rows: i32,
        cols: i32,
    ) -> i32;
    fn ds4_markov(
        device: i32,
        emb: *const c_void,
        head: *const c_void,
        base: *const f32,
        scratch: *mut f32,
        ids: *mut f32,
        seed: i32,
        rows: i32,
        vocab: i32,
    ) -> i32;
    fn ds_upload(
        dev: c_int,
        out: *mut *mut c_void,
        src: *const c_void,
        scales: *const c_void,
        rows: c_int,
        cols: c_int,
        mode: c_int,
    ) -> c_int;
    fn ds_alloc_f32(dev: c_int, out: *mut *mut f32, len: usize) -> c_int;
    fn ds_copy_f32(dev: c_int, dst: *mut f32, src: *const f32, len: usize, to_host: c_int)
        -> c_int;
    fn ds_dense_growth(
        ctx: *mut c_void,
        rows: i32,
        cols: i32,
        batch: i32,
        groups: i32,
        mode: i32,
        full_rows: i32,
    ) -> usize;
    fn ds_project(
        ctx: *mut c_void,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        groups: c_int,
        mode: c_int,
        round_output: c_int,
        prefill: c_int,
        mx_tiles: c_int,
        full_rows: c_int,
    ) -> c_int;
    #[cfg(test)]
    fn ds_gemm(
        ctx: *mut c_void,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        groups: c_int,
        mode: c_int,
    ) -> c_int;
    fn ds_dense_create(device: c_int, out: *mut *mut c_void) -> c_int;
    fn ds_dense_destroy(ctx: *mut c_void);
    fn ds_free(device: c_int, ptr: *mut c_void);
    fn ds_dense_error() -> *const c_char;
    fn ds_set_device(device: c_int) -> c_int;
}
fn checked(code: c_int) -> Result<()> {
    ensure!(
        code == 0,
        "{}",
        unsafe { CStr::from_ptr(ds_dense_error()) }.to_string_lossy()
    );
    Ok(())
}
pub fn packed_dense() -> bool {
    std::env::var_os("DS4_PACKED_DENSE").is_some_and(|v| v == "1")
}
pub fn tp_dense() -> bool {
    std::env::var_os("DS4_TP_DENSE").is_some_and(|v| v == "1")
}
struct Allocation {
    device: i32,
    ptr: *mut f32,
    capacity: usize,
}
// Device pointers may be moved between threads. All CUDA work uses the device's legacy default stream.
unsafe impl Send for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe { ds_free(self.device, self.ptr.cast()) }
    }
}
#[derive(Default)]
struct Pool {
    idle: Vec<Allocation>,
    elements: usize,
}
pub struct Dev {
    allocation: Option<Allocation>,
    pool: Arc<Mutex<Pool>>,
    len: usize,
}
impl Dev {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn as_ptr(&self) -> *const f32 {
        self.allocation.as_ref().unwrap().ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.allocation.as_ref().unwrap().ptr
    }
    fn device(&self) -> i32 {
        self.allocation.as_ref().unwrap().device
    }
    /// Release session storage at eviction, bypassing the transient buffer pool.
    pub(crate) fn release(mut self) {
        drop(self.allocation.take());
    }
}
impl Drop for Dev {
    fn drop(&mut self) {
        let Some(a) = self.allocation.take() else {
            return;
        };
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        // Limit idle workspace to 256 MiB per card even as prompt chunk sizes vary.
        if a.capacity <= (64 << 20) - pool.elements {
            pool.elements += a.capacity;
            pool.idle.push(a);
        }
    }
}
struct Matrix {
    peer: Option<parallel_dense::PeerWeight>,
    device: i32,
    ptr: *mut c_void,
    rows: usize,
    cols: usize,
    mode: c_int,
    round_output: bool,
}
impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { ds_free(self.device, self.ptr) }
    }
}
pub struct Runtime {
    pub weights: Arc<Weights>,
    pub device: i32,
    pub moe_cache: MoeCache,
    pub(crate) attention_scratch: crate::attention::Workspace,
    pub verification: bool,
    pub(crate) target_prefill: bool,
    prefill_tiles: i32,
    packed: bool,
    parallel: Option<parallel_dense::ParallelDense>,
    devices: [i32; 2],
    pools: HashMap<i32, Arc<Mutex<Pool>>>,
    vectors: HashMap<String, Dev>,
    matrices: HashMap<String, Matrix>,
    contexts: HashMap<i32, *mut c_void>,
}
// All CUDA handles and mutable caches belong to the single locked inference worker.
// Native entry points bind the current OS thread to the requested CUDA device.
unsafe impl Send for Runtime {}
impl Runtime {
    pub fn open(path: &Path, devices: [i32; 2]) -> Result<Self> {
        ensure!(
            devices[0] != devices[1],
            "two distinct CUDA devices required"
        );
        let mut weights = Weights::open(path)?;
        match std::env::var("DS4_DRAFTER").as_deref() {
            Ok("code") => weights.overlay_mtp(&path.join("drafter-code-k7"))?,
            Ok("released" | "none") | Err(_) => (),
            Ok(name) => anyhow::bail!("unknown DS4_DRAFTER {name}"),
        }
        let prefill_tiles = match std::env::var("DS4_PREFILL_MX_TILES").as_deref() {
            Ok("4") => 4,
            Ok("2") => 2,
            Ok("1") | Err(std::env::VarError::NotPresent) => 1,
            _ => anyhow::bail!("DS4_PREFILL_MX_TILES must be 1, 2 or 4"),
        };
        let mut rt = Self {
            weights: Arc::new(weights),
            device: devices[0],
            devices,
            moe_cache: MoeCache::default(),
            attention_scratch: Default::default(),
            verification: false,
            target_prefill: false,
            prefill_tiles,
            packed: packed_dense(),
            parallel: None,
            pools: devices.into_iter().map(|d| (d, Arc::default())).collect(),
            vectors: HashMap::new(),
            matrices: HashMap::new(),
            contexts: HashMap::new(),
        };
        for dev in devices {
            let mut ctx = ptr::null_mut();
            checked(unsafe { ds_dense_create(dev, &mut ctx) })?;
            rt.contexts.insert(dev, ctx);
        }
        if tp_dense() {
            rt.parallel = Some(parallel_dense::ParallelDense::new(devices)?);
        }
        Ok(rt)
    }
    pub fn peer_device(&self) -> i32 {
        self.devices[usize::from(self.device == self.devices[0])]
    }
    pub fn layer(&mut self, layer: usize) -> Result<()> {
        self.device = self.devices[usize::from(layer >= 20)];
        checked(unsafe { ds_set_device(self.device) })
    }
    pub fn buffer(&mut self, len: usize) -> Result<Dev> {
        ensure!(
            len > 0 && len <= isize::MAX as usize / 4,
            "invalid device buffer size"
        );
        let pool = self.pools[&self.device].clone();
        let cached = {
            let mut p = pool.lock().unwrap_or_else(|e| e.into_inner());
            let index = p
                .idle
                .iter()
                .enumerate()
                .filter(|(_, a)| a.capacity >= len)
                .min_by_key(|(_, a)| a.capacity)
                .map(|(i, _)| i);
            index.map(|i| {
                let a = p.idle.swap_remove(i);
                p.elements -= a.capacity;
                a
            })
        };
        let allocation = if let Some(a) = cached {
            a
        } else {
            self.moe_cache.reserve(self.device, len * 4)?;
            let mut raw = ptr::null_mut();
            checked(unsafe { ds_alloc_f32(self.device, &mut raw, len) })?;
            Allocation {
                device: self.device,
                ptr: raw,
                capacity: len,
            }
        };
        Ok(Dev {
            allocation: Some(allocation),
            len,
            pool,
        })
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, input: &[f32]) -> Result<Dev> {
        let mut out = self.buffer(input.len())?;
        checked(unsafe {
            ds_copy_f32(
                self.device,
                out.as_mut_ptr(),
                input.as_ptr(),
                input.len(),
                0,
            )
        })?;
        Ok(out)
    }
    pub fn to_host(&self, input: &Dev) -> Result<Vec<f32>> {
        self.to_host_range(input, 0, input.len())
    }
    pub(crate) fn to_host_range(&self, input: &Dev, start: usize, len: usize) -> Result<Vec<f32>> {
        ensure!(
            start.checked_add(len).is_some_and(|end| end <= input.len()),
            "invalid device read range"
        );
        let mut out = vec![0.; len];
        if len == 0 {
            return Ok(out);
        }
        let copied = checked(unsafe {
            ds_copy_f32(
                input.device(),
                out.as_mut_ptr(),
                input.as_ptr().add(start),
                len,
                1,
            )
        });
        if self.device != input.device() {
            checked(unsafe { ds_set_device(self.device) })?;
        }
        copied?;
        Ok(out)
    }
    pub fn slice(&mut self, x: &Dev, start: usize, len: usize) -> Result<Dev> {
        ensure!(
            x.device() == self.device && start.checked_add(len).is_some_and(|end| end <= x.len()),
            "invalid device slice"
        );
        let mut out = self.buffer(len)?;
        checked(unsafe {
            ds_copy_f32(self.device, out.as_mut_ptr(), x.as_ptr().add(start), len, 2)
        })?;
        Ok(out)
    }
    pub(crate) fn copy_into(&self, dst: &mut Dev, offset: usize, src: &Dev) -> Result<()> {
        ensure!(
            dst.device() == self.device
                && src.device() == self.device
                && offset
                    .checked_add(src.len())
                    .is_some_and(|end| end <= dst.len()),
            "invalid batched device copy"
        );
        checked(unsafe {
            ds_copy_f32(
                self.device,
                dst.as_mut_ptr().add(offset),
                src.as_ptr(),
                src.len(),
                2,
            )
        })
    }

    pub(crate) fn raw_bf16(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<*const c_void> {
        if !self.matrices.contains_key(name) {
            let w = self.weights.tensor(name)?;
            ensure!(
                w.dtype == "BF16" && w.shape == [rows, cols],
                "invalid BF16 table {name}"
            );
            self.moe_cache.reserve(self.device, w.data.len())?;
            let mut raw = ptr::null_mut();
            checked(unsafe {
                ds_upload(
                    self.device,
                    &mut raw,
                    w.data.as_ptr().cast(),
                    ptr::null(),
                    rows.try_into()?,
                    cols.try_into()?,
                    1,
                )
            })?;
            self.matrices.insert(
                name.into(),
                Matrix {
                    peer: None,
                    device: self.device,
                    ptr: raw,
                    rows,
                    cols,
                    mode: 1,
                    round_output: false,
                },
            );
        }
        let m = &self.matrices[name];
        ensure!(
            m.mode == 1 && m.device == self.device && m.rows == rows && m.cols == cols,
            "wrong Markov table placement"
        );
        Ok(m.ptr.cast_const())
    }
    pub(crate) fn token_ids(&self, ids: &Dev) -> Result<Vec<u32>> {
        self.to_host(ids)?
            .into_iter()
            .map(|x| {
                ensure!(
                    (0.0..129280.0).contains(&x),
                    "invalid or nonfinite logits during token selection"
                );
                Ok(x as u32)
            })
            .collect()
    }
    pub fn sample(
        &mut self,
        logits: &Dev,
        rows: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<u32>> {
        const VOCAB: usize = 129280;
        ensure!(
            rows > 0
                && rows <= 512
                && logits.device() == self.device
                && logits.len() == rows * VOCAB,
            "invalid greedy logits shape"
        );
        let mut tmp = self.buffer(rows * VOCAB.div_ceil(256) * 2)?;
        let mut ids = self.buffer(rows)?;
        let e = unsafe {
            if temperature == 0. {
                ds4_argmax(
                    self.device,
                    logits.as_ptr(),
                    tmp.as_mut_ptr(),
                    ids.as_mut_ptr(),
                    rows as i32,
                    VOCAB as i32,
                )
            } else {
                ds4_sample(
                    self.device,
                    logits.as_ptr(),
                    tmp.as_mut_ptr(),
                    ids.as_mut_ptr(),
                    rows as i32,
                    VOCAB as i32,
                    temperature,
                    seed,
                )
            }
        };
        ensure!(e == 0, "CUDA greedy selection failed ({e})");
        self.token_ids(&ids)
    }
    pub fn markov(&mut self, logits: &Dev, seed: u32, rows: usize) -> Result<Dev> {
        const VOCAB: usize = 129280;
        ensure!(
            seed < VOCAB as u32
                && (1..=16).contains(&rows)
                && logits.len() == rows * VOCAB
                && logits.device() == self.device,
            "invalid Markov input"
        );
        let emb = self.raw_bf16("mtp.2.markov_head.embed.weight", VOCAB, 256)?;
        let head = self.raw_bf16("mtp.2.markov_head.head.weight", VOCAB, 256)?;
        let mut tmp = self.buffer(VOCAB.div_ceil(64) * 2)?;
        let mut ids = self.buffer(rows)?;
        let e = unsafe {
            ds4_markov(
                self.device,
                emb,
                head,
                logits.as_ptr(),
                tmp.as_mut_ptr(),
                ids.as_mut_ptr(),
                seed as i32,
                rows as i32,
                VOCAB as i32,
            )
        };
        ensure!(e == 0, "CUDA Markov selection failed ({e})");
        Ok(ids)
    }
    pub fn vector_ptr(&mut self, name: &str, len: usize) -> Result<*const f32> {
        if !self.vectors.contains_key(name) {
            let host = self.weights.vector(name)?;
            ensure!(host.len() == len, "invalid vector shape at {name}");
            let data = self.to_device(&host)?;
            self.vectors.insert(name.to_owned(), data);
        }
        let v = &self.vectors[name];
        ensure!(
            v.len() == len && v.device() == self.device,
            "wrong vector placement or shape at {name}"
        );
        Ok(v.as_ptr())
    }
    pub fn linear(&mut self, name: &str, x: &[f32], batch: usize) -> Result<Vec<f32>> {
        self.group_linear(name, x, batch, 1)
    }
    pub fn group_linear(
        &mut self,
        name: &str,
        x: &[f32],
        batch: usize,
        groups: usize,
    ) -> Result<Vec<f32>> {
        let input = self.to_device(x)?;
        let output = self.linear_dev(name, &input, batch, groups)?;
        self.to_host(&output)
    }
    pub fn linear_dev(&mut self, name: &str, x: &Dev, batch: usize, groups: usize) -> Result<Dev> {
        ensure!(groups > 0, "invalid group count");
        if !self.matrices.contains_key(name) {
            let w = self.weights.tensor(name)?;
            ensure!(w.shape.len() == 2, "expected matrix {name}");
            let (rows, cols) = (w.shape[0], w.shape[1]);
            let mut scales = ptr::null();
            let mode = match w.dtype {
                "F32" => {
                    if self.packed {
                        5
                    } else {
                        0
                    }
                }
                "BF16" if name == "head.weight" => {
                    ensure!(cols % 32 == 0, "unaligned vocabulary head");
                    if self.packed
                        && !std::env::var_os("DS4_MMA_HEAD").is_some_and(|v| v == "0")
                        && rows.is_multiple_of(64)
                    {
                        7
                    } else {
                        3
                    }
                }
                "BF16" => {
                    if self.packed {
                        6
                    } else {
                        1
                    }
                }
                "F8_E4M3" => {
                    let s = self.weights.tensor(&name.replace(".weight", ".scale"))?;
                    ensure!(
                        s.dtype == "F8_E8M0" && s.shape == [rows.div_ceil(32), cols.div_ceil(32)],
                        "invalid MXFP8 scales {name}"
                    );
                    scales = s.data.as_ptr().cast();
                    if self.packed
                        && rows.is_multiple_of(64)
                        && cols.is_multiple_of(32)
                        && (rows / groups).is_multiple_of(64)
                    {
                        4
                    } else {
                        2
                    }
                }
                other => anyhow::bail!("unsupported matrix {name} {other}"),
            };
            let split =
                self.parallel.is_some() && parallel_dense::eligible(rows, cols, groups, mode);
            let shard_rows = if split { rows / 2 } else { rows };
            // The packed MXFP8 tensor, scales and expanded BF16 destination coexist during upload.
            let elements = shard_rows
                .checked_mul(cols)
                .context("dense shape overflow")?;
            let mut transient = elements
                .checked_mul([4, 2, 3, 3, 2, 4, 2, 4][mode as usize])
                .context("dense size overflow")?;
            if mode == 2 || mode == 4 {
                transient = transient
                    .checked_add(shard_rows.div_ceil(32) * cols.div_ceil(32))
                    .context("scale size overflow")?;
            }
            if mode == 3 || mode == 7 {
                transient = transient
                    .checked_add(elements / 32)
                    .context("head scale size overflow")?;
            }
            let peer = if split {
                let peer_device = self.peer_device();
                Some(parallel_dense::PeerWeight::upload(
                    &mut self.moe_cache,
                    self.device,
                    peer_device,
                    w.data,
                    scales,
                    shard_rows,
                    cols,
                    mode,
                    transient,
                )?)
            } else {
                None
            };
            self.moe_cache.reserve(self.device, transient)?;
            let mut buf = ptr::null_mut();
            checked(unsafe {
                ds_upload(
                    self.device,
                    &mut buf,
                    w.data.as_ptr().cast(),
                    scales.cast(),
                    shard_rows.try_into()?,
                    cols.try_into()?,
                    mode,
                )
            })
            .with_context(|| format!("upload {name}"))?;
            self.matrices.insert(
                name.to_string(),
                Matrix {
                    peer,
                    device: self.device,
                    ptr: buf,
                    rows,
                    cols,
                    mode,
                    round_output: mode != 0
                        && mode != 5
                        && mode != 3
                        && !name.ends_with(".ffn.gate.weight")
                        && !(name.contains(".attn.compressor.")
                            && (name.ends_with(".wkv.weight") || name.ends_with(".wgate.weight"))),
                },
            );
        }
        let w = &self.matrices[name];
        ensure!(
            groups > 0
                && w.rows.is_multiple_of(groups)
                && (w.mode != 4 || (w.rows / groups).is_multiple_of(64))
                && x.device() == self.device
                && w.device == self.device
                && batch > 0
                && x.len() == batch * groups * w.cols,
            "input or placement mismatch at {name}"
        );
        ensure!(
            !matches!(w.mode, 3 | 7) || groups == 1,
            "vocabulary head expects one group"
        );
        let n = batch.checked_mul(w.rows).context("dense output overflow")?;
        let mut out = self.buffer(n)?;
        let w = &self.matrices[name];
        let mx_tiles = if self.target_prefill {
            self.prefill_tiles
        } else {
            1
        };
        if w.peer.is_some() {
            ensure!(groups == 1, "grouped call on tensor-parallel weight {name}");
            let peer_device = self.peer_device();
            self.parallel
                .as_mut()
                .context("missing parallel projection")?
                .project(
                    &mut self.moe_cache,
                    w,
                    self.contexts[&self.device],
                    self.contexts[&peer_device],
                    x.as_ptr(),
                    out.as_mut_ptr(),
                    batch,
                    !self.verification,
                    mx_tiles,
                )
                .with_context(|| format!("parallel project {name}"))?;
            return Ok(out);
        }
        let extra = unsafe {
            ds_dense_growth(
                self.contexts[&self.device],
                w.rows.try_into()?,
                w.cols.try_into()?,
                batch.try_into()?,
                groups.try_into()?,
                w.mode,
                w.rows.try_into()?,
            )
        };
        if extra > 0 {
            self.moe_cache.reserve(self.device, extra)?;
        }
        checked(unsafe {
            ds_project(
                self.contexts[&self.device],
                w.ptr,
                x.as_ptr(),
                out.as_mut_ptr(),
                w.rows.try_into()?,
                w.cols.try_into()?,
                batch.try_into()?,
                groups.try_into()?,
                w.mode,
                w.round_output.into(),
                (!self.verification).into(),
                mx_tiles,
                w.rows.try_into()?,
            )
        })
        .with_context(|| format!("project {name}"))?;
        Ok(out)
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        for ctx in self.contexts.values() {
            unsafe { ds_dense_destroy(*ctx) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an inference GPU"]
    fn failed_request_allocation_leaves_next_cuda_launch_clean() -> Result<()> {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
            fn cudaPeekAtLastError() -> i32;
        }
        let dev = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        checked(unsafe { ds_set_device(dev) })?;
        let (mut free, mut total) = (0, 0);
        assert_eq!(unsafe { cudaMemGetInfo(&mut free, &mut total) }, 0);
        let mut p = ptr::null_mut();
        assert_ne!(unsafe { ds_alloc_f32(dev, &mut p, total / 4 + 1) }, 0);
        assert!(p.is_null());
        assert_eq!(unsafe { cudaPeekAtLastError() }, 0);
        checked(unsafe { ds_alloc_f32(dev, &mut p, 1) })?;
        unsafe { ds_free(dev, p.cast()) };
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn scratch_growth_covers_grouped_inputs_and_large_batch_head() -> Result<()> {
        let dev = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut ctx = ptr::null_mut();
        checked(unsafe { ds_dense_create(dev, &mut ctx) })?;
        let growth = |rows, cols, n, groups, mode| unsafe {
            ds_dense_growth(ctx, rows, cols, n, groups, mode, rows)
        };
        let head = growth(129280, 5120, 512, 1, 7);
        let grouped = growth(1024, 512, 32, 8, 2);
        let stable = growth(24, 20480, 17, 1, 5);
        let fp32 = growth(512, 512, 32, 1, 0);
        unsafe { ds_dense_destroy(ctx) };
        assert_eq!(head, 512 * (5120 * 2 + 129280 * 4));
        assert_eq!(grouped, 32 * 8 * 512 * 2);
        assert_eq!(stable, 17 * 20 * 24 * 4);
        assert_eq!(fp32, 0);
        Ok(())
    }
    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn packed_head_uses_independent_bf16_to_e4m3_group_scales() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let mut ctx = ptr::null_mut();
        checked(unsafe { ds_dense_create(device, &mut ctx) })?;
        struct Handle(*mut c_void);
        impl Drop for Handle {
            fn drop(&mut self) {
                unsafe { ds_dense_destroy(self.0) }
            }
        }
        let ctx = Handle(ctx);
        let weights: Vec<_> = (0..3)
            .flat_map(|r| {
                (0..64).map(move |i| {
                    let n = if i < 32 { 1.0625 } else { 1024.0 }; // The first group ties to 1.0 in E4M3.
                    half::bf16::from_f32([n, -n, 0.][r])
                })
            })
            .collect();
        let mut data = ptr::null_mut();
        checked(unsafe {
            ds_upload(
                device,
                &mut data,
                weights.as_ptr().cast(),
                ptr::null(),
                3,
                64,
                3,
            )
        })?;
        let w = Matrix {
            peer: None,
            device,
            ptr: data,
            rows: 3,
            cols: 64,
            mode: 3,
            round_output: false,
        };
        let x: Vec<_> = (0..64)
            .map(|i| if i < 32 { 1. } else { 1. / 64. })
            .collect();
        let mut y = [0.; 3];
        checked(unsafe { ds_gemm(ctx.0, w.ptr, x.as_ptr(), y.as_mut_ptr(), 3, 64, 1, 1, 3) })?;
        assert_eq!(y, [544., -544., 0.]);
        let mut batch = vec![];
        for n in 1..=5 {
            batch.extend(x.iter().map(|v| v * n as f32));
        }
        let mut multi = [0.; 15];
        checked(unsafe {
            ds_gemm(
                ctx.0,
                w.ptr,
                batch.as_ptr(),
                multi.as_mut_ptr(),
                3,
                64,
                5,
                1,
                3,
            )
        })?;
        for (n, v) in multi.as_chunks::<3>().0.iter().enumerate() {
            let positive = half::bf16::from_f32(544. * (n + 1) as f32).to_f32();
            assert_eq!(*v, [positive, -positive, 0.]);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn markov_chain_and_greedy_reductions() -> Result<()> {
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let cols = 517usize;
        let upload =
            |bytes: *const c_void, rows: usize, cols: usize, mode: i32| -> Result<Matrix> {
                let mut p = ptr::null_mut();
                checked(unsafe {
                    ds_upload(
                        dev,
                        &mut p,
                        bytes,
                        ptr::null(),
                        rows.try_into()?,
                        cols.try_into()?,
                        mode,
                    )
                })?;
                Ok(Matrix {
                    peer: None,
                    device: dev,
                    ptr: p,
                    rows,
                    cols,
                    mode,
                    round_output: false,
                })
            };
        let as_f32 = |v: &[f32]| upload(v.as_ptr().cast(), 1, v.len(), 0);
        let mut x = vec![-1f32; cols * 3];
        x[15] = 1.;
        x[512] = 1.;
        x[cols + 300] = 2.;
        x[cols * 2 + 2] = f32::NAN;
        let mut scores = as_f32(&x)?;
        let tmp = as_f32(&vec![0.; 3 * 32])?;
        let ids = as_f32(&[0.; 3])?;
        let mut host = [0.; 3];
        let read = |dst: &mut [f32]| {
            checked(unsafe { ds_copy_f32(dev, dst.as_mut_ptr(), ids.ptr.cast(), dst.len(), 1) })
        };
        assert_eq!(
            unsafe {
                ds4_argmax(
                    dev,
                    scores.ptr.cast(),
                    tmp.ptr.cast(),
                    ids.ptr.cast(),
                    3,
                    cols as i32,
                )
            },
            0
        );
        read(&mut host)?;
        assert_eq!(host, [15., 300., -1.]);
        x.fill(0.);
        scores = as_f32(&x)?;
        let mut emb = vec![half::bf16::ZERO; cols * 256];
        let mut head = emb.clone();
        for (i, (from, to)) in [(4, 33), (33, 512), (512, 9)].into_iter().enumerate() {
            emb[from * 256 + i] = half::bf16::ONE;
            head[to * 256 + i] = half::bf16::from_f32(2.);
        }
        let e = upload(emb.as_ptr().cast(), cols, 256, 1)?;
        let h = upload(head.as_ptr().cast(), cols, 256, 1)?;
        assert_eq!(
            unsafe {
                ds4_markov(
                    dev,
                    e.ptr,
                    h.ptr,
                    scores.ptr.cast(),
                    tmp.ptr.cast(),
                    ids.ptr.cast(),
                    4,
                    3,
                    cols as i32,
                )
            },
            0
        );
        read(&mut host)?;
        assert_eq!(host, [33., 512., 9.]);
        x[1] = f32::INFINITY;
        scores = as_f32(&x)?;
        assert_eq!(
            unsafe {
                ds4_markov(
                    dev,
                    e.ptr,
                    h.ptr,
                    scores.ptr.cast(),
                    tmp.ptr.cast(),
                    ids.ptr.cast(),
                    4,
                    3,
                    cols as i32,
                )
            },
            0
        );
        read(&mut host)?;
        assert_eq!(host, [-1.; 3]);
        let n = 4096;
        let mut logits = [0., 3f32.ln()].repeat(n);
        logits[0] = f32::NAN;
        let scores = as_f32(&logits)?;
        let scratch = as_f32(&vec![0.; n * 2])?;
        let samples = as_f32(&vec![0.; n])?;
        assert_eq!(
            unsafe {
                ds4_sample(
                    dev,
                    scores.ptr.cast(),
                    scratch.ptr.cast(),
                    samples.ptr.cast(),
                    n as i32,
                    2,
                    1.,
                    123456,
                )
            },
            0
        );
        let mut draws = vec![0.; n];
        checked(unsafe { ds_copy_f32(dev, draws.as_mut_ptr(), samples.ptr.cast(), n, 1) })?;
        assert_eq!(draws[0], -1.);
        let frequency = draws.iter().filter(|&&x| x == 1.).count() as f64 / (n - 1) as f64;
        assert!(
            (0.72..0.78).contains(&frequency),
            "unexpected categorical probability {frequency}"
        );
        assert_eq!(
            unsafe {
                ds4_sample(
                    dev,
                    scores.ptr.cast(),
                    scratch.ptr.cast(),
                    samples.ptr.cast(),
                    n as i32,
                    2,
                    f32::MAX,
                    123456,
                )
            },
            0
        );
        checked(unsafe { ds_copy_f32(dev, draws.as_mut_ptr(), samples.ptr.cast(), n, 1) })?;
        assert!(draws[1..].iter().all(|v| *v == 0. || *v == 1.));
        let hot = draws[1..].iter().filter(|&&v| v == 1.).count();
        assert!((1800..2300).contains(&hot));
        Ok(())
    }
    #[test]
    #[ignore = "requires DS4_TEST_DEVICE on an RTX PRO 6000"]
    fn packed_dense_matches_mxfp8_and_is_batch_invariant() -> Result<()> {
        let dev = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let (rows, cols, batch, groups) = (128usize, 128usize, 9usize, 2usize);
        let w: Vec<u8> = (0..rows * cols)
            .map(|i| (((i * 37 + i / 13) % 125) as u8) | (((i % 3 == 0) as u8) << 7))
            .collect();
        let sc: Vec<u8> = (0..rows / 32 * cols / 32)
            .map(|j| 119 + (j % 7) as u8)
            .collect();
        let x: Vec<f32> = (0..cols * groups * batch)
            .map(|i| half::bf16::from_f32(((i * 17 % 257) as f32 - 127.) / 97.).to_f32())
            .collect();
        let mut ctx = ptr::null_mut();
        checked(unsafe { ds_dense_create(dev, &mut ctx) })?;
        let run = |mode: i32| -> Result<(Vec<f32>, Vec<f32>)> {
            let mut raw = ptr::null_mut();
            checked(unsafe {
                ds_upload(
                    dev,
                    &mut raw,
                    w.as_ptr().cast(),
                    sc.as_ptr().cast(),
                    rows as i32,
                    cols as i32,
                    mode,
                )
            })?;
            let w = Matrix {
                peer: None,
                device: dev,
                ptr: raw,
                rows,
                cols,
                mode,
                round_output: false,
            };
            let mut all = vec![0.; rows * batch];
            let mut single = all.clone();
            checked(unsafe {
                ds_gemm(
                    ctx,
                    w.ptr,
                    x.as_ptr(),
                    all.as_mut_ptr(),
                    rows as i32,
                    cols as i32,
                    batch as i32,
                    groups as i32,
                    mode,
                )
            })?;
            for (input, out) in x
                .chunks_exact(cols * groups)
                .zip(single.chunks_exact_mut(rows))
            {
                checked(unsafe {
                    ds_gemm(
                        ctx,
                        w.ptr,
                        input.as_ptr(),
                        out.as_mut_ptr(),
                        rows as i32,
                        cols as i32,
                        1,
                        groups as i32,
                        mode,
                    )
                })?;
            }
            Ok((all, single))
        };
        // Per-row head scales include independent signs and exponents for every
        // vocabulary row and each group of 32 channels. Exercise a partial 8-row tile.
        let head_weights: Vec<_> = (0..rows * cols)
            .map(|i| {
                half::bf16::from_f32(
                    (((i * 11) % 91) as f32 - 45.)
                        * 2f32.powi(((i / cols + i % cols / 32) % 13) as i32 - 6),
                )
            })
            .collect();
        let head_run = |mode: i32| -> Result<Vec<f32>> {
            let mut raw = ptr::null_mut();
            checked(unsafe {
                ds_upload(
                    dev,
                    &mut raw,
                    head_weights.as_ptr().cast(),
                    ptr::null(),
                    rows as i32,
                    cols as i32,
                    mode,
                )
            })?;
            let w = Matrix {
                peer: None,
                device: dev,
                ptr: raw,
                rows,
                cols,
                mode,
                round_output: true,
            };
            let mut out = vec![0.; rows * batch];
            checked(unsafe {
                ds_gemm(
                    ctx,
                    w.ptr,
                    x.as_ptr(),
                    out.as_mut_ptr(),
                    rows as i32,
                    cols as i32,
                    batch as i32,
                    1,
                    mode,
                )
            })?;
            Ok(out)
        };
        for (a, b) in head_run(3)?.into_iter().zip(head_run(7)?) {
            assert!(
                (a - b).abs() <= (0.01 * a.abs()).max(0.01),
                "wrong per-row head scale {a}, {b}"
            );
        }
        let (base, _) = run(2)?;
        let (packed, single) = run(4)?;
        unsafe { ds_dense_destroy(ctx) };
        assert_eq!(
            packed, single,
            "projection depended on other tokens in the block"
        );
        for (a, b) in base.into_iter().zip(packed) {
            assert!(
                (a - b).abs() < 0.001 * (1. + a.abs()),
                "wrong packed tile: reference {a}, got {b}"
            );
        }
        Ok(())
    }
}
