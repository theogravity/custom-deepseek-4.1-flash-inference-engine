//! DSpark stages attend to a rolling window of committed target projections
//! plus every fresh KV in the candidate block, including later noise rows.
use crate::{
    attention::{project, rotary_freqs, Freq},
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, ptr};

const HIDDEN: usize = 5120;
const DIM: usize = 512;
const HEADS: usize = 64;
const WINDOW: usize = 128;
const STAGES: usize = 3;

#[path = "draft_rotation.rs"]
mod rotation;

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
}

pub(crate) fn validate_snapshot(pos: usize, bytes: &[u8]) -> Result<usize> {
    let words = pos.min(WINDOW) * DIM / 2;
    ensure!(
        pos > 0 && pos <= i32::MAX as usize && bytes.len() == 16 + STAGES * words * 4,
        "invalid DSpark snapshot dimensions"
    );
    ensure!(
        &bytes[..8] == b"DS4DRF01" && bytes[8..16] == (pos as u64).to_le_bytes(),
        "incompatible DSpark snapshot position or layout"
    );
    Ok(words)
}

impl DraftAttention {
    pub fn ready(&self, pos: usize) -> bool {
        pos > 0
            && self.end == pos
            && self.start == pos.saturating_sub(WINDOW)
            && self.rings.iter().all(Option::is_some)
    }

    // The ring is physical-order BF16 packed two per f32 storage word.
    // Before the first wrap only slots 0..pos have been written.
    pub fn snapshot(&self, rt: &mut Runtime, pos: usize) -> Result<Vec<u8>> {
        ensure!(self.ready(pos), "incomplete DSpark prefix window");
        rt.layer(39)?;
        let words = pos.min(WINDOW) * DIM / 2;
        let mut bytes = Vec::with_capacity(16 + STAGES * words * 4);
        bytes.extend_from_slice(b"DS4DRF01");
        bytes.extend_from_slice(&(pos as u64).to_le_bytes());
        for stage in &self.rings {
            let ring = stage.as_ref().context("missing DSpark prefix ring")?;
            let subset;
            let live = if words == ring.len() {
                ring
            } else {
                subset = rt.slice(ring, 0, words)?;
                &subset
            };
            for word in rt.to_host(live)? {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    pub fn restore(rt: &mut Runtime, pos: usize, bytes: &[u8]) -> Result<Self> {
        let words = validate_snapshot(pos, bytes)?;
        rt.layer(39)?;
        let mut state = Self {
            start: pos.saturating_sub(WINDOW),
            end: pos,
            ..Self::default()
        };
        for (slot, packed) in state
            .rings
            .iter_mut()
            .zip(bytes[16..].chunks_exact(4 * words))
        {
            let mut ring = vec![0.; WINDOW * DIM / 2];
            for (dst, src) in ring.iter_mut().zip(packed.as_chunks::<4>().0) {
                *dst = f32::from_le_bytes(*src);
            }
            *slot = Some(rt.to_device(&ring)?);
        }
        Ok(state)
    }

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
}

// Each proposal can see its own committed ring and all of its fresh rows.
// Only row-independent projections and RoPE evaluation span requests.
pub(crate) struct ProposalSpan<'a> {
    pub attention: &'a DraftAttention,
    pub pos: usize,
    pub width: usize,
}

pub(crate) fn proposal_angles(rt: &mut Runtime, spans: &[ProposalSpan<'_>]) -> Result<Dev> {
    let mut freqs = Vec::new();
    for span in spans {
        ensure!(
            (1..=WINDOW).contains(&span.width)
                && span.pos == span.attention.end
                && span
                    .pos
                    .checked_add(span.width)
                    .is_some_and(|end| end <= i32::MAX as usize),
            "invalid DSpark proposal width or position"
        );
        freqs.extend((span.pos..span.pos + span.width).map(|p| rotary_freqs(p, false)));
    }
    rt.to_device(freqs.as_flattened().as_flattened())
}

pub(crate) fn proposal_attention(
    rt: &mut Runtime,
    stage: usize,
    x: &Dev,
    spans: &[ProposalSpan<'_>],
    freqs: &Dev,
) -> Result<Dev> {
    let rows: usize = spans.iter().map(|s| s.width).sum();
    ensure!(
        stage < STAGES && rows > 0 && rows * HIDDEN == x.len(),
        "invalid DSpark attention input"
    );
    const ANGLES_PER_ROW: usize = size_of::<Freq>() / size_of::<f32>();
    ensure!(
        freqs.len() == rows * ANGLES_PER_ROW,
        "invalid DSpark angle buffer"
    );
    if spans.len() > 1 && !rt.uses_packed_weights() {
        // cuBLAS may select a different accumulation path when the matrix
        // dimensions change. Keep the reference span shapes for unpacked weights.
        let mut out = rt.buffer(rows * HIDDEN)?;
        let mut offset = 0;
        for span in spans {
            let one = rt.slice(x, offset * HIDDEN, span.width * HIDDEN)?;
            let angles = rt.slice(freqs, offset * ANGLES_PER_ROW, span.width * ANGLES_PER_ROW)?;
            let y = proposal_attention(rt, stage, &one, std::slice::from_ref(span), &angles)?;
            rt.copy_into(&mut out, offset * HIDDEN, &y)?;
            offset += span.width;
        }
        return Ok(out);
    }
    let stem = format!("mtp.{stage}.attn");
    let mut qr = project(rt, &stem, "wq_a", x, rows, 1280, 1)?;
    normalize(rt, &mut qr, &format!("{stem}.q_norm.weight"), 1280, rows)?;
    let mut q = project(rt, &stem, "wq_b", &qr, rows, HEADS * DIM, 1)?;
    transform(rt, &mut q, freqs, rows, HEADS, 0, false)?;
    let kv = key_values(rt, &stem, x, rows, freqs)?;
    let sink = rt.vector_ptr(&format!("{stem}.attn_sink"), HEADS)?;
    let mut out = rt.buffer(q.len())?;
    let mut offset = 0;
    for span in spans {
        let state = span.attention;
        let ring: *const c_void = state.rings[stage]
            .as_ref()
            .map_or(ptr::null(), |r| r.as_ptr().cast());
        let valid = state.end - state.start;
        ensure!(
            valid <= WINDOW && (valid == 0 || !ring.is_null()),
            "invalid DSpark attention window"
        );
        checked(
            unsafe {
                ds_draft_attend(
                    q.as_ptr().add(offset * HEADS * DIM),
                    ring,
                    kv.as_ptr().add(offset * DIM),
                    (state.start % WINDOW) as i32,
                    valid as i32,
                    span.width as i32,
                    sink,
                    out.as_mut_ptr().add(offset * HEADS * DIM),
                )
            },
            "attending to DSpark target window and proposal block",
        )?;
        offset += span.width;
    }
    transform(rt, &mut out, freqs, rows, HEADS, 32, true)?;
    let down = project(rt, &stem, "wo_a", &out, rows, 8192, 8)?;
    project(rt, &stem, "wo_b", &down, rows, HIDDEN, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_snapshot_checks_position_and_window_layout() {
        for pos in [1usize, 127, 128, 129, 524287] {
            let words = pos.min(WINDOW) * DIM / 2;
            let mut bytes = vec![0; 16 + words * STAGES * 4];
            bytes[..8].copy_from_slice(b"DS4DRF01");
            bytes[8..16].copy_from_slice(&(pos as u64).to_le_bytes());
            assert_eq!(validate_snapshot(pos, &bytes).unwrap(), words);
            assert!(validate_snapshot(pos + 1, &bytes).is_err());
            assert!(validate_snapshot(pos, &bytes[1..]).is_err());
            bytes[6] ^= 1;
            assert!(validate_snapshot(pos, &bytes).is_err());
        }
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to both inference GPUs"]
    fn draft_prefix_round_trips_packed_words_across_window_wrap() -> Result<()> {
        use std::path::PathBuf;
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(&PathBuf::from(std::env::var("DS4_TEST_PACK")?), [0, 1])?;
        for pos in [7usize, 128, 241] {
            let mut bytes = vec![0; 16 + pos.min(WINDOW) * DIM * 2 * STAGES];
            bytes[..8].copy_from_slice(b"DS4DRF01");
            bytes[8..16].copy_from_slice(&(pos as u64).to_le_bytes());
            for (i, byte) in bytes[16..].iter_mut().enumerate() {
                *byte = (i.wrapping_mul(111) + i / 129) as u8;
            }
            rt.layer(0)?;
            let restored = DraftAttention::restore(&mut rt, pos, &bytes)?;
            assert!(restored.ready(pos));
            assert!(!restored.ready(pos + 1));
            assert_eq!(restored.snapshot(&mut rt, pos)?, bytes);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs"]
    fn joint_proposal_attention_preserves_each_sessions_projected_rows() -> Result<()> {
        let mut rt = Runtime::open_with_packing(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
            true,
        )?;
        rt.layer(39)?;
        let layout: [(usize, usize); 4] = [(1, 1), (127, 7), (132, 3), (404151, 6)];
        let mut states = Vec::new();
        for (index, (pos, _)) in layout.iter().copied().enumerate() {
            let mut bytes = Vec::from(*b"DS4DRF01");
            bytes.extend_from_slice(&(pos as u64).to_le_bytes());
            for j in 0..STAGES * pos.min(WINDOW) * DIM {
                let value = ((j * 23 + index * 13) % 211) as f32 / 83. - 1.;
                bytes.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            }
            states.push(DraftAttention::restore(&mut rt, pos, &bytes)?);
        }
        let spans: Vec<_> = states
            .iter()
            .zip(layout)
            .map(|(attention, (pos, width))| ProposalSpan {
                attention,
                pos,
                width,
            })
            .collect();
        let rows: usize = spans.iter().map(|s| s.width).sum();
        let x = rt.to_device(
            &(0..rows * HIDDEN)
                .map(|j| ((j * 71) % 113) as f32 / 37. - 1.)
                .collect::<Vec<_>>(),
        )?;
        let angles = proposal_angles(&mut rt, &spans)?;
        let short = rt.buffer(1)?;
        assert!(proposal_attention(&mut rt, 0, &x, &spans, &short).is_err());
        for stage in 0..STAGES {
            let y = proposal_attention(&mut rt, stage, &x, &spans, &angles)?;
            let combined = rt.to_host(&y)?;
            let mut offset = 0;
            for span in &spans {
                let one = std::slice::from_ref(span);
                let a = proposal_angles(&mut rt, one)?;
                let n = span.width * HIDDEN;
                let local = rt.slice(&x, offset, n)?;
                let y = proposal_attention(&mut rt, stage, &local, one, &a)?;
                let serial = rt.to_host(&y)?;
                assert!(
                    combined[offset..offset + n]
                        .iter()
                        .zip(serial)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "joint DSpark stage {stage} changed an output at offset {offset}"
                );
                offset += n;
            }
        }
        Ok(())
    }

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
                for (h, &sink) in sinks.iter().enumerate() {
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
                    let expected = bf(value / (sum + (sink - max).exp()));
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
