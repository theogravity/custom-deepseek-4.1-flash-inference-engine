use super::DIM;
use crate::runtime::{Dev, Runtime};
use anyhow::{bail, ensure, Result};
use std::ffi::{c_char, c_int, CStr};

const CONTROLS: usize = 24;

pub(crate) fn fast_rms() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DS4_FAST_RMS").is_some_and(|v| v == "1"))
}

extern "C" {
    fn ds_hc_prepare(
        device: c_int,
        h: *const f32,
        mix: *mut f32,
        projection: *const f32,
        scale: *const f32,
        base: *const f32,
        norm: *const f32,
        out: *mut f32,
        batch: c_int,
        fast: c_int,
    ) -> c_int;
    fn ds_hc_expand(
        device: c_int,
        h: *mut f32,
        x: *const f32,
        mix: *const f32,
        batch: c_int,
    ) -> c_int;
    fn ds_hc_final(
        device: c_int,
        h: *const f32,
        mix: *mut f32,
        norm: *const f32,
        out: *mut f32,
        start: c_int,
        rows: c_int,
        fast: c_int,
    ) -> c_int;
    fn ds_hc_capture(
        device: c_int,
        h: *const f32,
        features: *mut f32,
        slot: c_int,
        batch: c_int,
    ) -> c_int;
    fn ds_hc_broadcast(device: c_int, embeddings: *const f32, h: *mut f32, batch: c_int) -> c_int;
    fn cudaGetErrorString(code: c_int) -> *const c_char;
}

fn checked(code: c_int, op: &str) -> Result<()> {
    if code != 0 {
        let error = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy();
        bail!("{op}: {error}");
    }
    Ok(())
}

pub(crate) fn embed(rt: &mut Runtime, ids: &[u32]) -> Result<Dev> {
    let e = rt.weights.tensor("embed.weight")?;
    ensure!(
        e.dtype == "BF16" && e.shape == [129280, DIM],
        "invalid token embedding"
    );
    let mut values = Vec::with_capacity(ids.len() * DIM);
    for &id in ids {
        ensure!((id as usize) < e.shape[0], "invalid embedding token {id}");
        let start = id as usize * 2 * DIM;
        values.extend(
            e.data[start..start + 2 * DIM]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| half::bf16::from_le_bytes(*v).to_f32()),
        );
    }
    let embedding = rt.to_device(&values)?;
    let mut h = rt.buffer(ids.len() * 4 * DIM)?;
    checked(
        unsafe {
            ds_hc_broadcast(
                rt.device,
                embedding.as_ptr(),
                h.as_mut_ptr(),
                ids.len().try_into()?,
            )
        },
        "broadcast token embeddings",
    )?;
    Ok(h)
}

pub(super) fn capture(
    rt: &Runtime,
    h: &Dev,
    features: &mut Dev,
    slot: usize,
    batch: usize,
) -> Result<()> {
    ensure!(
        slot < 3 && h.len() == batch * 4 * DIM && features.len() == batch * 3 * DIM,
        "invalid DSpark capture"
    );
    checked(
        unsafe {
            ds_hc_capture(
                rt.device,
                h.as_ptr(),
                features.as_mut_ptr(),
                slot.try_into()?,
                batch.try_into()?,
            )
        },
        "capture DSpark target activations",
    )
}

pub(crate) fn initial_mix(rt: &mut Runtime, batch: usize) -> Result<Dev> {
    let mut weights = vec![0.; batch * CONTROLS];
    for row in weights.as_chunks_mut::<CONTROLS>().0 {
        row[0] = 1.;
    }
    rt.to_device(&weights)
}

pub(crate) fn prepare(
    rt: &mut Runtime,
    stem: &str,
    branch: &str,
    h: &Dev,
    mix: &mut Dev,
    batch: usize,
) -> Result<Dev> {
    let projection = rt.linear_dev(&format!("{stem}.hc_{branch}_fn"), h, batch, 1)?;
    ensure!(
        projection.len() == batch * CONTROLS
            && mix.len() == projection.len()
            && h.len() == batch * 4 * DIM,
        "invalid hyper-connection dimensions at {stem}.{branch}"
    );
    let scale = rt.vector_ptr(&format!("{stem}.hc_{branch}_scale"), 3)?;
    let base = rt.vector_ptr(&format!("{stem}.hc_{branch}_base"), CONTROLS)?;
    let norm = rt.vector_ptr(&format!("{stem}.{branch}_norm.weight"), DIM)?;
    let mut input = rt.buffer(batch * DIM)?;
    checked(
        unsafe {
            ds_hc_prepare(
                rt.device,
                h.as_ptr(),
                mix.as_mut_ptr(),
                projection.as_ptr(),
                scale,
                base,
                norm,
                input.as_mut_ptr(),
                batch.try_into()?,
                fast_rms().into(),
            )
        },
        "prepare residual branch",
    )?;
    Ok(input)
}

pub(crate) fn expand(rt: &Runtime, h: &mut Dev, x: &Dev, mix: &Dev, batch: usize) -> Result<()> {
    ensure!(
        h.len() == batch * 4 * DIM && x.len() == batch * DIM && mix.len() == batch * CONTROLS,
        "invalid residual branch output dimensions"
    );
    checked(
        unsafe {
            ds_hc_expand(
                rt.device,
                h.as_mut_ptr(),
                x.as_ptr(),
                mix.as_ptr(),
                batch.try_into()?,
            )
        },
        "expand residual branch",
    )
}

// Normalize any contiguous range of final hidden rows for the vocabulary head.
pub(crate) fn readout(
    rt: &mut Runtime,
    h: &Dev,
    mix: &mut Dev,
    norm_name: &str,
    start: usize,
    rows: usize,
) -> Result<Dev> {
    ensure!(
        rows > 0
            && start.saturating_add(rows) <= h.len() / (4 * DIM)
            && start.saturating_add(rows) <= mix.len() / CONTROLS,
        "final residual row out of bounds"
    );
    let norm = rt.vector_ptr(norm_name, DIM)?;
    let mut input = rt.buffer(rows * DIM)?;
    checked(
        unsafe {
            ds_hc_final(
                rt.device,
                h.as_ptr(),
                mix.as_mut_ptr(),
                norm,
                input.as_mut_ptr(),
                start.try_into()?,
                rows.try_into()?,
                fast_rms().into(),
            )
        },
        "collapse final residual",
    )?;
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::bf;
    use rayon::prelude::*;
    use std::path::PathBuf;

    // Scalar implementation retained as an independent reference for both rounding
    // points, four-stream orientation and the Sinkhorn iteration count.
    struct Mix {
        pre: [f32; 4],
        post: [f32; 4],
        comb: [[f32; 4]; 4],
    }
    fn mix(logits: &[f32], inv: f32, base: &[f32], scale: &[f32]) -> Mix {
        let val = |i: usize, s: usize| logits[i] * inv * scale[s] + base[i];
        let pre = std::array::from_fn(|j| 1. / (1. + (-val(j, 0)).exp()) + 1e-6);
        let post = std::array::from_fn(|j| 2. / (1. + (-val(4 + j, 1)).exp()));
        let mut comb: [[f32; 4]; 4] =
            std::array::from_fn(|j| std::array::from_fn(|k| val(8 + 4 * j + k, 2)));
        for row in &mut comb {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            *row = row.map(|v| (v - max).exp());
            let sum = row.iter().sum::<f32>();
            *row = row.map(|v| v / sum + 1e-6);
        }
        for iter in 0..20 {
            if iter > 0 {
                for row in &mut comb {
                    let sum = row.iter().sum::<f32>() + 1e-6;
                    *row = row.map(|v| v / sum);
                }
            }
            for j in 0..4 {
                let sum = comb.iter().map(|r| r[j]).sum::<f32>() + 1e-6;
                for row in &mut comb {
                    row[j] /= sum;
                }
            }
        }
        Mix { pre, post, comb }
    }
    fn collapse(h: &[f32], pre: &[[f32; 4]], norm: &[f32]) -> Vec<f32> {
        let mut y = vec![0.; pre.len() * DIM];
        y.par_chunks_exact_mut(DIM)
            .zip(h.par_chunks_exact(4 * DIM))
            .zip(pre.par_iter())
            .for_each(|((row, old), p)| {
                for i in 0..DIM {
                    row[i] = bf((0..4).map(|c| p[c] * old[c * DIM + i]).sum());
                }
                let inv = 1. / (row.iter().map(|v| v * v).sum::<f32>() / DIM as f32 + 1e-20).sqrt();
                for (v, w) in row.iter_mut().zip(norm) {
                    *v = bf(*v * inv * w);
                }
            });
        y
    }
    fn scalar_expand(h: &mut [f32], x: &[f32], mixes: &[Mix]) {
        h.par_chunks_exact_mut(4 * DIM)
            .zip(x.par_chunks_exact(DIM))
            .zip(mixes.par_iter())
            .for_each(|((old, row), m)| {
                for i in 0..DIM {
                    let r = [old[i], old[DIM + i], old[2 * DIM + i], old[3 * DIM + i]];
                    for c in 0..4 {
                        old[c * DIM + i] =
                            bf(row[i] * m.post[c]
                                + (0..4).map(|j| r[j] * m.comb[j][c]).sum::<f32>());
                    }
                }
            });
    }

    fn close_bf16(actual: &[f32], expected: &[f32]) {
        // A reduction-order difference may move a value across a BF16 rounding boundary.
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= a.abs().max(b.abs()) / 128., "{a} vs {b}");
        }
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to both inference GPUs"]
    fn resident_branches_match_scalar_mixing_and_final_readout() -> Result<()> {
        let visible = std::env::var("CUDA_VISIBLE_DEVICES")?;
        ensure!(
            visible.split(',').count() == 2,
            "select both inference GPUs"
        );
        let pack = PathBuf::from(std::env::var("DS4_TEST_PACK")?);
        let mut rt = Runtime::open(&pack, [0, 1])?;
        rt.layer(0)?;
        for batch in [1usize, 3] {
            let norm: Vec<_> = (0..DIM).map(|d| bf(0.7 + (d % 9) as f32 / 23.)).collect();
            let scale = [0.57, -0.83, 1.2];
            let base: Vec<_> = (0..24).map(|i| ((i * 11 % 19) as f32 - 8.) / 13.).collect();
            let mut h: Vec<_> = (0..batch * 4 * DIM)
                .map(|i| bf(((i * 17 + i / DIM * 7) % 97) as f32 / 16. - 2.))
                .collect();
            let mut pre: Vec<[f32; 4]> = (0..batch)
                .map(|t| [0.125, -0.53, 0.8 + t as f32 / 8., 0.2])
                .collect();
            let mut controls = vec![0.; batch * CONTROLS];
            for (row, p) in controls.as_chunks_mut::<CONTROLS>().0.iter_mut().zip(&pre) {
                row[..4].copy_from_slice(p);
            }
            let mut stream = rt.to_device(&h)?;
            let mut weights = rt.to_device(&controls)?;
            let dev_scale = rt.to_device(&scale)?;
            let dev_base = rt.to_device(&base)?;
            let dev_norm = rt.to_device(&norm)?;
            let mut collapsed = rt.buffer(batch * DIM)?;
            for branch in 0..2 {
                let projection: Vec<_> = (0..batch * CONTROLS)
                    .map(|i| ((i * 7 + branch) % 31) as f32 / 7. - 1.)
                    .collect();
                let projected = rt.to_device(&projection)?;
                let inputs = collapse(&h, &pre, &norm);
                let mixes: Vec<_> = h
                    .as_chunks::<{ 4 * DIM }>()
                    .0
                    .iter()
                    .zip(projection.as_chunks::<CONTROLS>().0.iter())
                    .map(|(row, p)| {
                        let inv = 1.
                            / (row.iter().map(|v| v * v).sum::<f32>() / (4 * DIM) as f32 + 1e-20)
                                .sqrt();
                        mix(p, inv, &base, &scale)
                    })
                    .collect();
                let prior_weights = rt.to_host(&weights)?;
                let mut fast_weights = rt.to_device(&prior_weights)?;
                for (candidate, fast) in [(&mut fast_weights, 1), (&mut weights, 0)] {
                    checked(
                        unsafe {
                            ds_hc_prepare(
                                rt.device,
                                stream.as_ptr(),
                                candidate.as_mut_ptr(),
                                projected.as_ptr(),
                                dev_scale.as_ptr(),
                                dev_base.as_ptr(),
                                dev_norm.as_ptr(),
                                collapsed.as_mut_ptr(),
                                batch.try_into()?,
                                fast,
                            )
                        },
                        "prepare test branch",
                    )?;
                    let actual_input = rt.to_host(&collapsed)?;
                    if fast == 0 {
                        assert_eq!(actual_input, inputs);
                    } else {
                        close_bf16(&actual_input, &inputs);
                    }
                    let actual = rt.to_host(candidate)?;
                    for (row, m) in actual.as_chunks::<CONTROLS>().0.iter().zip(&mixes) {
                        for (&a, &b) in row
                            .iter()
                            .zip(m.pre.iter().chain(&m.post).chain(m.comb.iter().flatten()))
                        {
                            let tolerance = if fast == 0 { 1e-6 } else { 1e-4 };
                            assert!((a - b).abs() <= tolerance, "mix coefficient {a} vs {b}");
                        }
                    }
                }
                // Use the normalized input as a deterministic stand-in for branch output.
                scalar_expand(&mut h, &inputs, &mixes);
                super::expand(&rt, &mut stream, &collapsed, &weights, batch)?;
                assert_eq!(rt.to_host(&stream)?, h);
                pre = mixes.into_iter().map(|m| m.pre).collect();
            }
            let rows = batch.min(2);
            let t = batch - rows;
            let expected = collapse(&h[t * 4 * DIM..], &pre[t..], &norm);
            let mut projected = rt.buffer(rows * DIM)?;
            for fast in [0, 1] {
                checked(
                    unsafe {
                        ds_hc_final(
                            rt.device,
                            stream.as_ptr(),
                            weights.as_mut_ptr(),
                            dev_norm.as_ptr(),
                            projected.as_mut_ptr(),
                            t.try_into()?,
                            rows.try_into()?,
                            fast,
                        )
                    },
                    "final test readout",
                )?;
                let actual = rt.to_host(&projected)?;
                if fast == 0 {
                    assert_eq!(actual, expected);
                } else {
                    close_bf16(&actual, &expected);
                }
            }
            let mut features = rt.buffer(batch * 3 * DIM)?;
            let mut expected = vec![0.; features.len()];
            for slot in 0..3 {
                let streams: Vec<_> = (0..batch * 4 * DIM)
                    .map(|i| {
                        bf((i / (4 * DIM)) as f32 * 11.
                            + slot as f32 * 4.
                            + ((i / DIM) % 4) as f32 / 8.
                            + (i % DIM % 7) as f32 / 8.)
                    })
                    .collect();
                for t in 0..batch {
                    for d in 0..DIM {
                        let sum: f32 = (0..4).map(|c| streams[(t * 4 + c) * DIM + d]).sum();
                        expected[(t * 3 + slot) * DIM + d] = bf(sum * 0.25);
                    }
                }
                let h = rt.to_device(&streams)?;
                capture(&rt, &h, &mut features, slot, batch)?;
            }
            assert_eq!(rt.to_host(&features)?, expected);
        }
        Ok(())
    }
}
