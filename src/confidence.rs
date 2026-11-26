use crate::{
    model::DIM,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn ds4_confidence(
        device: i32,
        h: *const f32,
        mix: *const f32,
        w: *const f32,
        embed: *const c_void,
        ids: *const f32,
        seed: i32,
        vocab: i32,
        out: *mut f32,
        rows: i32,
    ) -> i32;
}

pub(super) struct Gate {
    threshold: f32,
    trace: bool,
}
impl Gate {
    pub(super) fn configured() -> Result<Option<Self>> {
        let threshold = match std::env::var("DS4_CONFIDENCE_THRESHOLD") {
            Ok(v) => v
                .parse::<f32>()
                .context("invalid DS4_CONFIDENCE_THRESHOLD")?,
            Err(std::env::VarError::NotPresent) => 0.,
            Err(e) => return Err(e.into()),
        };
        ensure!(
            (0. ..=1.).contains(&threshold),
            "DS4_CONFIDENCE_THRESHOLD must be in [0,1]"
        );
        let trace = std::env::var_os("DS4_CONFIDENCE_TRACE").is_some_and(|v| v == "1");
        Ok((threshold > 0. || trace).then_some(Self { threshold, trace }))
    }
    pub(super) fn evaluate(
        &self,
        rt: &mut Runtime,
        h: &Dev,
        mix: &Dev,
        seed: u32,
        ids: &Dev,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
        let width = ids.len();
        ensure!(
            (1..=16).contains(&width) && h.len() == width * 4 * DIM && mix.len() == width * 24,
            "invalid DSpark confidence dimensions"
        );
        let embedding = rt.raw_bf16("mtp.2.markov_head.embed.weight", 129280, 256)?;
        let w = rt.vector_ptr("mtp.2.confidence_head.proj.weight", DIM + 256)?;
        let mut output = rt.buffer(width)?;
        let rc = unsafe {
            ds4_confidence(
                rt.device,
                h.as_ptr(),
                mix.as_ptr(),
                w,
                embedding,
                ids.as_ptr(),
                seed.try_into()?,
                129280,
                output.as_mut_ptr(),
                width.try_into()?,
            )
        };
        ensure!(rc == 0, "CUDA confidence head failed ({rc})");
        let scores = rt.to_host(&output)?;
        let mut proposed = rt.token_ids(ids)?;
        let (keep, confidence) = select(&scores, self.threshold)?;
        proposed.truncate(keep);
        Ok((proposed, self.trace.then_some(confidence)))
    }
}

fn select(logits: &[f32], threshold: f32) -> Result<(usize, Vec<f32>)> {
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite DSpark confidence logit"
    );
    let scores: Vec<_> = logits.iter().map(|v| 1. / (1. + (-v).exp())).collect();
    let keep = scores
        .iter()
        .position(|&v| v < threshold)
        .unwrap_or(scores.len());
    Ok((keep, scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_gate_keeps_contiguous_verified_candidates() -> Result<()> {
        let logits = [0., 2., -4., 3.];
        assert_eq!(select(&logits, 0.)?.0, 4);
        assert_eq!(select(&logits, 0.5)?.0, 2);
        assert_eq!(select(&logits, 0.9)?.0, 0);
        assert!(select(&[f32::NAN], 0.).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to two inference GPUs"]
    fn confidence_uses_raw_collapse_and_previous_markov_embedding() -> Result<()> {
        use std::path::PathBuf;
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(&PathBuf::from(std::env::var("DS4_TEST_PACK")?), [0, 1])?;
        rt.layer(39)?;
        let width = 3;
        let values: Vec<_> = (0..width * 4 * DIM)
            .map(|i| crate::model::bf((i * 29 % 151) as f32 / 31. - 2.))
            .collect();
        let mut mix = vec![0.; width * 24];
        for (t, p) in mix.as_chunks_mut::<24>().0.iter_mut().enumerate() {
            p[..4].copy_from_slice(&[0.3125, 0.427, -0.14, 0.15 * t as f32]);
        }
        let q = rt.weights.vector("mtp.2.confidence_head.proj.weight")?;
        let embeddings = rt.weights.tensor("mtp.2.markov_head.embed.weight")?;
        let mut expected = Vec::new();
        for (t, id) in [21usize, 1234, 981].into_iter().enumerate() {
            let mut sum = 0f64;
            for d in 0..DIM {
                let v = crate::model::bf(
                    (0..4)
                        .map(|c| values[(t * 4 + c) * DIM + d] * mix[t * 24 + c])
                        .sum(),
                );
                sum += f64::from(v) * f64::from(q[d]);
            }
            let bytes = &embeddings.data[id * 512..(id + 1) * 512];
            for (e, &v) in bytes.as_chunks::<2>().0.iter().zip(&q[DIM..]) {
                sum += f64::from(half::bf16::from_le_bytes(*e).to_f32()) * f64::from(v);
            }
            expected.push(1. / (1. + (-sum).exp()));
        }
        let h = rt.to_device(&values)?;
        let mix = rt.to_device(&mix)?;
        let ids = rt.to_device(&[1234., 981., 17.])?;
        let (tokens, actual) = Gate {
            threshold: 0.,
            trace: true,
        }
        .evaluate(&mut rt, &h, &mix, 21, &ids)?;
        assert_eq!(tokens, [1234, 981, 17]);
        for (a, b) in actual.unwrap().into_iter().zip(expected) {
            assert!((f64::from(a) - b).abs() < 2e-5, "{a} vs {b}");
        }
        Ok(())
    }
}
