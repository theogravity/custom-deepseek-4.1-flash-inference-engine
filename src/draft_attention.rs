//! DSpark stages attend to a rolling window of committed target projections
//! plus every fresh KV in the candidate block, including later noise rows.
use crate::{
    attention::{project, rotary_freqs},
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, ptr};

const HIDDEN: usize = 5120;
const DIM: usize = 512;
const HEADS: usize = 64;
const WINDOW: usize = 128;
const STAGES: usize = 3;

unsafe extern "C" {
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
    fn ds_draft_store(kv: *const f32, ring: *mut c_void, pos: i32, rows: i32) -> i32;
    fn ds_draft_attend(
        q: *const f32,
        ring: *const c_void,
        fresh: *const f32,
        start: i32,
        valid: i32,
        width: i32,
        sink: *const f32,
        out: *mut f32,
    ) -> i32;

}

fn checked(code: i32, operation: &str) -> Result<()> {
    ensure!(code == 0, "CUDA {operation} failed with error {code}");
    Ok(())
}

fn angles(rt: &mut Runtime, pos: usize, rows: usize) -> Result<Dev> {
    let freqs: Vec<_> = (pos..pos + rows).map(|p| rotary_freqs(p, false)).collect();
    rt.to_device(freqs.as_flattened().as_flattened())
}

fn normalize(
    rt: &mut Runtime,
    values: &mut Dev,
    name: &str,
    cols: usize,
    rows: usize,
) -> Result<()> {
    let gamma = rt.vector_ptr(name, cols)?;
    checked(
        unsafe {
            ds4_attn_normalize(
                rt.device,
                values.as_mut_ptr(),
                gamma,
                cols.try_into()?,
                rows.try_into()?,
            )
        },
        "normalizing DSpark projection",
    )
}

fn transform(
    rt: &Runtime,
    values: &mut Dev,
    freqs: &Dev,
    rows: usize,
    heads: usize,
    group: i32,
    inverse: bool,
) -> Result<()> {
    checked(
        unsafe {
            ds4_attn_transform(
                rt.device,
                values.as_mut_ptr(),
                ptr::null_mut(),
                freqs.as_ptr(),
                heads.try_into()?,
                DIM as i32,
                group,
                inverse.into(),
                rows.try_into()?,
            )
        },
        "rotating DSpark activations",
    )
}

fn key_values(rt: &mut Runtime, stem: &str, x: &Dev, rows: usize, freqs: &Dev) -> Result<Dev> {
    let mut kv = project(rt, stem, "wkv", x, rows, DIM, 1)?;
    normalize(rt, &mut kv, &format!("{stem}.kv_norm.weight"), DIM, rows)?;
    // DSpark rounds the entire post-RoPE key through 32-channel FP8 groups.
    transform(rt, &mut kv, freqs, rows, 1, 32, false)?;
    Ok(kv)
}

#[derive(Default)]
pub struct DraftAttention {
    rings: [Option<Dev>; STAGES],
    start: usize,
    end: usize,
    proposal_freqs: Option<(usize, usize, Dev)>,
}

impl DraftAttention {
    /// Input can contain all verification rows. Only the committed prefix and
    /// at most its most recent window are projected into persistent KV.
    pub fn observe(
        &mut self,
        rt: &mut Runtime,
        features: &Dev,
        pos: usize,
        committed: usize,
    ) -> Result<()> {
        let feature_cols = STAGES * HIDDEN;
        ensure!(
            features.len().is_multiple_of(feature_cols)
                && committed <= features.len() / feature_cols,
            "invalid DSpark observation shape"
        );
        let end = pos
            .checked_add(committed)
            .context("DSpark position overflow")?;
        ensure!(end <= i32::MAX as usize, "DSpark position overflow");
        let start = if self.end == pos { self.start } else { pos };
        let n = committed.min(WINDOW);
        if n == 0 {
            self.start = start;
            self.end = end;
            return Ok(());
        }
        let subset;
        let observed = if n * feature_cols == features.len() {
            features
        } else {
            subset = rt.slice(features, (committed - n) * feature_cols, n * feature_cols)?;
            &subset
        };
        let mut main = project(rt, "mtp.0", "main_proj", observed, n, HIDDEN, 1)?;
        normalize(rt, &mut main, "mtp.0.main_norm.weight", HIDDEN, n)?;
        let write_pos = end - n;
        let freqs = angles(rt, write_pos, n)?;
        let mut keys = Vec::with_capacity(STAGES);
        for (stage, ring) in self.rings.iter_mut().enumerate() {
            if ring.is_none() {
                *ring = Some(rt.buffer(WINDOW * DIM / 2)?); // two BF16 values per f32 slot
            }
            keys.push(key_values(
                rt,
                &format!("mtp.{stage}.attn"),
                &main,
                n,
                &freqs,
            )?);
        }
        // All projections are complete. Invalidate the old window until all
        // stages have enqueued the same segment successfully.
        self.start = end;
        self.end = end;
        for (ring, key) in self.rings.iter_mut().zip(keys) {
            let ring = ring.as_mut().context("missing DSpark ring")?;
            checked(
                unsafe {
                    ds_draft_store(
                        key.as_ptr(),
                        ring.as_mut_ptr().cast(),
                        write_pos as i32,
                        n as i32,
                    )
                },
                "appending committed DSpark KV",
            )?;
        }
        self.start = start.max(end.saturating_sub(WINDOW));
        Ok(())
    }

    /// pos is the absolute position of the first noise-block token, directly
    /// after the most recently observed target token.
    pub fn forward(
        &mut self,
        rt: &mut Runtime,
        stage: usize,
        x: &Dev,
        pos: usize,
        width: usize,
    ) -> Result<Dev> {
        ensure!(
            stage < STAGES && width > 0 && width <= WINDOW && x.len() == width * HIDDEN,
            "invalid DSpark attention input"
        );
        ensure!(
            pos == self.end
                && pos
                    .checked_add(width)
                    .is_some_and(|p| p <= i32::MAX as usize),
            "DSpark proposal position does not follow observed target rows"
        );
        if self
            .proposal_freqs
            .as_ref()
            .is_none_or(|(p, n, _)| (*p, *n) != (pos, width))
        {
            self.proposal_freqs = Some((pos, width, angles(rt, pos, width)?));
        }
        let freqs = &self.proposal_freqs.as_ref().unwrap().2;
        let stem = format!("mtp.{stage}.attn");
        let mut qr = project(rt, &stem, "wq_a", x, width, 1280, 1)?;
        normalize(rt, &mut qr, &format!("{stem}.q_norm.weight"), 1280, width)?;
        let mut q = project(rt, &stem, "wq_b", &qr, width, HEADS * DIM, 1)?;
        transform(rt, &mut q, freqs, width, HEADS, 0, false)?;
        let kv = key_values(rt, &stem, x, width, freqs)?;
        let sink = rt.vector_ptr(&format!("{stem}.attn_sink"), HEADS)?;
        let ring: *const c_void = self.rings[stage]
            .as_ref()
            .map_or(ptr::null(), |r| r.as_ptr().cast());
        let valid = self.end - self.start;
        ensure!(
            valid == 0 || !ring.is_null(),
            "missing DSpark attention cache"
        );
        let mut out = rt.buffer(q.len())?;
        checked(
            unsafe {
                ds_draft_attend(
                    q.as_ptr(),
                    ring,
                    kv.as_ptr(),
                    (self.start % WINDOW) as i32,
                    valid as i32,
                    width as i32,
                    sink,
                    out.as_mut_ptr(),
                )
            },
            "attending to DSpark target window and proposal block",
        )?;
        transform(rt, &mut out, freqs, width, HEADS, 32, true)?;
        let down = project(rt, &stem, "wo_a", &out, width, 8192, 8)?;
        project(rt, &stem, "wo_b", &down, width, HIDDEN, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercise the native kernels without opening the checkpoint or a cuBLAS
    // context. These buffers use the same single default stream as the engine.
    unsafe extern "C" {
        fn ds_alloc_f32(device: i32, out: *mut *mut f32, len: usize) -> i32;
        fn ds_copy_f32(
            device: i32,
            dst: *mut f32,
            src: *const f32,
            len: usize,
            to_host: i32,
        ) -> i32;
        fn ds_free(device: i32, ptr: *mut c_void);
    }

    struct TestBuffer {
        device: i32,
        ptr: *mut f32,
    }
    impl TestBuffer {
        fn upload(device: i32, data: &[f32]) -> Result<Self> {
            let mut ptr = ptr::null_mut();
            checked(
                unsafe { ds_alloc_f32(device, &mut ptr, data.len()) },
                "allocating fixture",
            )?;
            let buffer = Self { device, ptr };
            checked(
                unsafe { ds_copy_f32(device, ptr, data.as_ptr(), data.len(), 0) },
                "uploading fixture",
            )?;
            Ok(buffer)
        }
    }
    impl Drop for TestBuffer {
        fn drop(&mut self) {
            unsafe { ds_free(self.device, self.ptr.cast()) }
        }
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible CUDA sm_120 inference GPU ordinal"]
    fn cuda_draft_attention_reads_wrapped_target_and_all_proposal_rows() -> Result<()> {
        let d: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let bf = |x: f32| half::bf16::from_f32(x).to_f32();
        let width = 3;
        let fresh = [[-1., 6.], [2., -3.], [8., 12.]];
        let mut kv = vec![0.; width * DIM];
        for (row, pair) in fresh.iter().enumerate() {
            kv[row * DIM..row * DIM + 2].copy_from_slice(pair);
        }
        let mut queries = vec![0.; width * HEADS * DIM];
        for t in 0..width {
            for h in 0..HEADS {
                queries[(t * HEADS + h) * DIM] = bf((t as f32 - 1.0) * 2.0 + h as f32 * 0.25);
            }
        }
        let sinks: Vec<_> = (0..HEADS).map(|h| (h % 7) as f32 - 3.).collect();
        let q = TestBuffer::upload(d, &queries)?;
        let sink = TestBuffer::upload(d, &sinks)?;
        let new = TestBuffer::upload(d, &kv)?;
        let out = TestBuffer::upload(d, &vec![0.; queries.len()])?;
        let ring = TestBuffer::upload(d, &vec![f32::NAN; WINDOW * DIM / 2])?;
        for (start, valid) in [(126, 5), (71, WINDOW), (0, 0)] {
            let mut physical = [None; WINDOW];
            if valid > 0 {
                let mut main = vec![0.; valid * DIM];
                for i in 0..valid {
                    let pair = [(i % 13) as f32 - 6., (i % 9) as f32 - 5.];
                    main[i * DIM..i * DIM + 2].copy_from_slice(&pair);
                    physical[(start + i) % WINDOW] = Some(pair);
                }
                let src = TestBuffer::upload(d, &main)?;
                checked(
                    unsafe { ds_draft_store(src.ptr, ring.ptr.cast(), start as i32, valid as i32) },
                    "writing fixture target rows",
                )?;
            }
            checked(
                unsafe {
                    ds_draft_attend(
                        q.ptr,
                        ring.ptr.cast(),
                        new.ptr,
                        start as i32,
                        valid as i32,
                        width as i32,
                        sink.ptr,
                        out.ptr,
                    )
                },
                "evaluating fixture attention",
            )?;
            let mut actual = vec![0.; queries.len()];
            checked(
                unsafe { ds_copy_f32(d, actual.as_mut_ptr(), out.ptr, actual.len(), 1) },
                "reading fixture output",
            )?;
            let keys: Vec<_> = physical.into_iter().flatten().chain(fresh).collect();
            for t in 0..width {
                for h in 0..HEADS {
                    let offset = (t * HEADS + h) * DIM;
                    let query = queries[offset];
                    let mut max = -1e30;
                    let (mut sum, mut value) = (0.0f32, 0.0f32);
                    for tile in keys.chunks(64) {
                        let scores: Vec<_> = tile
                            .iter()
                            .map(|r| query * r[0] * (DIM as f32).sqrt().recip())
                            .collect();
                        let next = scores.iter().copied().fold(max, f32::max);
                        let rescale = (max - next).exp();
                        max = next;
                        sum *= rescale;
                        value *= rescale;
                        for (r, score) in tile.iter().zip(scores) {
                            let p = (score - max).exp();
                            value = bf(p).mul_add(r[1], value);
                            sum += p;
                        }
                    }
                    let expected = bf(value / (sum + (sinks[h] - max).exp()));
                    assert!(
                        (actual[offset + 1] - expected).abs() < 1e-5,
                        "start {start}, rows {valid}, query {t}, head {h}: {} vs {expected}",
                        actual[offset + 1]
                    );
                }
            }
        }
        Ok(())
    }
}
