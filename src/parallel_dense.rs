//! Output-row tensor parallelism. Each direction owns independent, growing
//! staging buffers; DEFAULT streams own the input and joined output lifetimes.
use super::{checked, ds_alloc_f32, ds_dense_growth, ds_free, ds_set_device, ds_upload, Matrix};
use crate::moe::MoeCache;
use anyhow::{bail, ensure, Context, Result};
use std::{
    ffi::{c_char, c_void, CStr},
    ptr,
};

unsafe extern "C" {
    fn ds_tp_new(owner: i32, peer: i32, out: *mut *mut c_void) -> i32;
    fn ds_tp_destroy(link: *mut c_void);
    fn ds_tp_project(
        link: *mut c_void,
        owner_ctx: *mut c_void,
        peer_ctx: *mut c_void,
        home_weight: *const c_void,
        peer_weight: *const c_void,
        input: *const f32,
        out: *mut f32,
        home_parts: *mut f32,
        peer_input: *mut f32,
        peer_parts: *mut f32,
        rows: i32,
        cols: i32,
        batch: i32,
        mode: i32,
        round: i32,
        prefill: i32,
        mx_tiles: i32,
    ) -> i32;
    fn cudaGetErrorString(code: i32) -> *const c_char;
}

fn cuda(code: i32) -> Result<()> {
    if code < 0 {
        return checked(code);
    }
    if code != 0 {
        bail!(
            "parallel dense CUDA: {}",
            unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy()
        );
    }
    Ok(())
}

pub(super) fn eligible(rows: usize, cols: usize, groups: usize, mode: i32) -> bool {
    groups == 1
        && matches!(mode, 2 | 3 | 4 | 7)
        && rows.is_multiple_of(128)
        && cols.is_multiple_of(32)
        && rows.saturating_mul(cols) >= (32 << 20)
}

pub(super) struct PeerWeight {
    device: i32,
    ptr: *mut c_void,
}

impl Drop for PeerWeight {
    fn drop(&mut self) {
        unsafe { ds_free(self.device, self.ptr) };
    }
}
impl PeerWeight {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upload(
        cache: &mut MoeCache,
        owner: i32,
        peer: i32,
        weights: &[u8],
        scales: *const u8,
        half: usize,
        cols: usize,
        mode: i32,
        transient: usize,
    ) -> Result<Self> {
        checked(unsafe { ds_set_device(peer) })?;
        let result = (|| {
            cache.reserve(peer, transient)?;
            let mut ptr = ptr::null_mut();
            let elements = half
                .checked_mul(cols)
                .context("tensor parallel size overflow")?;
            let offset = elements
                .checked_mul(if matches!(mode, 3 | 7) { 2 } else { 1 })
                .context("parallel weight offset overflow")?;
            ensure!(
                weights.len()
                    == offset
                        .checked_mul(2)
                        .context("parallel weight size overflow")?,
                "invalid TP shard weight size"
            );
            let scales = if scales.is_null() {
                ptr::null()
            } else {
                unsafe { scales.add((half / 32) * (cols / 32)) }
            };
            checked(unsafe {
                ds_upload(
                    peer,
                    &mut ptr,
                    weights[offset..].as_ptr().cast(),
                    scales.cast(),
                    half.try_into()?,
                    cols.try_into()?,
                    mode,
                )
            })?;
            Ok(Self { device: peer, ptr })
        })();
        checked(unsafe { ds_set_device(owner) })?;
        result
    }
}

struct Staging {
    device: i32,
    ptr: *mut f32,
    capacity: usize,
}
impl Staging {
    fn new(device: i32) -> Self {
        Self {
            device,
            ptr: ptr::null_mut(),
            capacity: 0,
        }
    }
    fn grow(&mut self, len: usize, cache: &mut MoeCache) -> Result<()> {
        if self.capacity >= len {
            return Ok(());
        }
        checked(unsafe { ds_set_device(self.device) })?;
        cache.reserve(
            self.device,
            len.checked_mul(4).context("TP staging size overflow")?,
        )?;
        let mut ptr = ptr::null_mut();
        checked(unsafe { ds_alloc_f32(self.device, &mut ptr, len) })?;
        let old = std::mem::replace(
            self,
            Self {
                device: self.device,
                ptr,
                capacity: len,
            },
        );
        drop(old);
        Ok(())
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ds_free(self.device, self.ptr.cast()) };
        }
    }
}

struct Link {
    raw: *mut c_void,
    home_parts: Staging,
    peer_input: Staging,
    peer_parts: Staging,
}
impl Link {
    fn new(home: i32, peer: i32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        cuda(unsafe { ds_tp_new(home, peer, &mut raw) })?;
        Ok(Self {
            raw,
            home_parts: Staging::new(home),
            peer_input: Staging::new(peer),
            peer_parts: Staging::new(peer),
        })
    }
}
impl Drop for Link {
    fn drop(&mut self) {
        unsafe { ds_tp_destroy(self.raw) };
    }
}

pub(super) struct ParallelDense([Link; 2]);
impl ParallelDense {
    pub(super) fn new(devices: [i32; 2]) -> Result<Self> {
        let [a, b] = devices;
        let links = [Link::new(a, b)?, Link::new(b, a)?];
        checked(unsafe { ds_set_device(a) })?;
        Ok(Self(links))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        &mut self,
        cache: &mut MoeCache,
        w: &Matrix,
        home_ctx: *mut c_void,
        peer_ctx: *mut c_void,
        x: *const f32,
        y: *mut f32,
        batch: usize,
        prefill: bool,
        mx_tiles: i32,
    ) -> Result<()> {
        let peer = w.peer.as_ref().context("missing peer weight")?;
        let link = self
            .0
            .iter_mut()
            .find(|l| l.home_parts.device == w.device)
            .context("invalid tensor parallel owner")?;
        let result = (|| {
            let rows: i32 = w.rows.try_into()?;
            let cols: i32 = w.cols.try_into()?;
            let n_input = batch.checked_mul(w.cols).context("TP input overflow")?;
            let n_output = batch.checked_mul(w.rows).context("TP output overflow")?;
            ensure!(n_output <= i32::MAX as usize, "TP result too large");
            let batch: i32 = batch.try_into()?;
            link.home_parts.grow(n_output, cache)?;
            link.peer_input.grow(n_input, cache)?;
            link.peer_parts.grow(n_output / 2, cache)?;
            for (ctx, dev) in [(home_ctx, w.device), (peer_ctx, peer.device)] {
                let extra = unsafe { ds_dense_growth(ctx, rows / 2, cols, batch, 1, w.mode, rows) };
                if extra > 0 {
                    checked(unsafe { ds_set_device(dev) })?;
                    cache.reserve(dev, extra)?;
                }
            }
            cuda(unsafe {
                ds_tp_project(
                    link.raw,
                    home_ctx,
                    peer_ctx,
                    w.ptr,
                    peer.ptr,
                    x,
                    y,
                    link.home_parts.ptr,
                    link.peer_input.ptr,
                    link.peer_parts.ptr,
                    rows,
                    cols,
                    batch,
                    w.mode,
                    w.round_output.into(),
                    prefill.into(),
                    mx_tiles,
                )
            })
        })();
        // Cache reservations and partial allocation failures may leave the peer current.
        checked(unsafe { ds_set_device(w.device) })?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ds_project, Runtime};
    use std::path::Path;

    #[allow(clippy::too_many_arguments)]
    fn upload_matrix(
        cache: &mut MoeCache,
        owner: i32,
        peer: i32,
        mode: i32,
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        cols: usize,
        split: bool,
    ) -> Result<Matrix> {
        let local = if split { rows / 2 } else { rows };
        let data = weights.as_ptr().cast();
        let scale_ptr = if scales.is_empty() {
            ptr::null()
        } else {
            scales.as_ptr()
        };
        let other = if split {
            Some(PeerWeight::upload(
                cache, owner, peer, weights, scale_ptr, local, cols, mode, 0,
            )?)
        } else {
            None
        };
        let mut raw = ptr::null_mut();
        checked(unsafe {
            ds_upload(
                owner,
                &mut raw,
                data,
                scale_ptr.cast(),
                local as i32,
                cols as i32,
                mode,
            )
        })?;
        Ok(Matrix {
            peer: other,
            device: owner,
            ptr: raw,
            rows,
            cols,
            mode,
            round_output: mode != 3 && mode != 7,
        })
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to two peer-accessible inference GPUs"]
    fn peer_dense_preserves_shard_order_and_split_k_in_both_directions() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let pack = std::env::var("DS4_TEST_PACK")?;
        let mut rt = Runtime::open(Path::new(&pack), [0, 1])?;
        let mut parallel = ParallelDense::new([0, 1])?;
        let (rows, cols) = (2048usize, 1024usize);
        let mut state = 17u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 23) as i32 - 256) as f32 / 32.
        };
        for mode in [2, 4, 3, 7] {
            let (data, scales) = if mode == 2 || mode == 4 {
                // E4M3 normal numbers with both signs and independently varying block scales.
                let weights = (0..rows * cols)
                    .map(|_| {
                        let v = next();
                        (((v.abs() * 9.) as u8) + 16) | if v < 0. { 0x80 } else { 0 }
                    })
                    .collect::<Vec<_>>();
                let scales = (0..(rows / 32) * (cols / 32))
                    .map(|i| (125 + (i * 17) % 5) as u8)
                    .collect();
                (weights, scales)
            } else {
                let words = (0..rows * cols)
                    .flat_map(|_| half::bf16::from_f32(next()).to_le_bytes())
                    .collect();
                (words, vec![])
            };
            let mut matrices = Vec::new();
            for (owner, peer) in [(0, 1), (1, 0)] {
                matrices.push((
                    upload_matrix(
                        &mut rt.moe_cache,
                        owner,
                        peer,
                        mode,
                        &data,
                        &scales,
                        rows,
                        cols,
                        false,
                    )?,
                    upload_matrix(
                        &mut rt.moe_cache,
                        owner,
                        peer,
                        mode,
                        &data,
                        &scales,
                        rows,
                        cols,
                        true,
                    )?,
                ));
            }
            let mut pending = Vec::new();
            for (batch, prefill, tiles) in [
                (1usize, false, 1),
                (8, false, 1),
                (33, true, 2),
                (64, true, 4),
                (129, true, 4),
                (129, false, 1),
                (3, false, 1),
            ] {
                for (owner, (full, shards)) in matrices.iter().enumerate() {
                    rt.layer(if owner == 0 { 0 } else { 20 })?;
                    let input: Vec<_> = (0..batch * cols).map(|_| next()).collect();
                    let x = rt.to_device(&input)?;
                    let mut expected = rt.buffer(batch * rows)?;
                    let mut out = rt.buffer(batch * rows)?;
                    let ctx = rt.contexts[&(owner as i32)];
                    let other = rt.contexts[&(1 - owner as i32)];
                    checked(unsafe {
                        ds_project(
                            ctx,
                            full.ptr,
                            x.as_ptr(),
                            expected.as_mut_ptr(),
                            rows as i32,
                            cols as i32,
                            batch as i32,
                            1,
                            mode,
                            full.round_output.into(),
                            prefill.into(),
                            1,
                            rows as i32,
                        )
                    })?;
                    parallel.project(
                        &mut rt.moe_cache,
                        shards,
                        ctx,
                        other,
                        x.as_ptr(),
                        out.as_mut_ptr(),
                        batch,
                        prefill,
                        tiles,
                    )?;
                    pending.push((expected, out));
                }
            }
            for (expected, out) in pending {
                let a = rt.to_host(&expected)?;
                let b = rt.to_host(&out)?;
                // GEMM may choose a different reduction tree when output width changes.
                // Packed kernels retain the original split count and identical outputs.
                if mode == 2 {
                    for (a, b) in a.into_iter().zip(b) {
                        assert!(
                            (a - b).abs() <= a.abs().max(b.abs()) / 128. + 1e-4,
                            "sharded GEMM {a} vs {b}"
                        );
                    }
                } else {
                    assert_eq!(a, b, "packed mode {mode}");
                }
            }
        }
        Ok(())
    }
}
