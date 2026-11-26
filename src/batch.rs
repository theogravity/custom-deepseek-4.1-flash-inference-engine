//! Sequence-local histories meet at row-independent projections and the MoE.
//! Each attention call only sees rows and KV belonging to its own request.
use super::{residual, ForwardItem, Model, Target, DIM};
use crate::{
    attention,
    engram::EngramRows,
    moe,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Context, Result};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static MULTI_PASSES: AtomicU64 = AtomicU64::new(0);
static SIX_PASSES: AtomicU64 = AtomicU64::new(0);
static PEAK_BATCH: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn six_target_passes() -> u64 {
    SIX_PASSES.load(Ordering::Relaxed)
}

pub(crate) fn target_batch_stats() -> (u64, usize) {
    (
        MULTI_PASSES.load(Ordering::Relaxed),
        PEAK_BATCH.load(Ordering::Relaxed),
    )
}

fn history<'a>(item: &ForwardItem<'a>) -> Result<&'a [u32]> {
    ensure!(
        item.rows > 0 && item.logit_rows <= item.rows,
        "invalid target span"
    );
    let end = item
        .pos
        .checked_add(item.rows)
        .context("target position overflow")?;
    // Engram consults the three IDs preceding this span as well.
    let context = item
        .ids
        .get(item.pos.saturating_sub(3)..end)
        .context("invalid target history")?;
    ensure!(
        context.iter().all(|&id| id < 129280),
        "invalid target token ID"
    );
    Ok(&item.ids[item.pos..end])
}

fn attend(
    rt: &mut Runtime,
    items: &mut [&mut ForwardItem<'_>],
    errors: &mut [Option<anyhow::Error>],
    layer: usize,
    x: &Dev,
) -> Result<Dev> {
    if let [s] = items {
        return attention::forward_dev(rt, s.cache, layer, s.pos, x, s.rows);
    }
    let mut spans = items
        .iter_mut()
        .zip(errors)
        .map(|(s, error)| attention::Span {
            state: s.cache,
            pos: s.pos,
            rows: s.rows,
            error,
        })
        .collect::<Vec<_>>();
    attention::forward_group(rt, layer, x, &mut spans)
}

impl Model {
    pub(super) fn forward_batch(
        &mut self,
        items: &mut [ForwardItem<'_>],
    ) -> Result<Vec<Result<Target>>> {
        ensure!(!items.is_empty(), "empty target batch");
        if let [s] = items {
            let ids = history(s)?;
            return self.forward_valid(&mut [s], ids);
        }
        let mut ids = Vec::new();
        let mut active = Vec::with_capacity(items.len());
        let mut invalid = Vec::with_capacity(items.len());
        for s in items {
            match history(s) {
                Ok(rows) => {
                    ids.extend_from_slice(rows);
                    active.push(s);
                    invalid.push(None);
                }
                Err(e) => invalid.push(Some(e)),
            }
        }
        let outcomes = if active.is_empty() {
            Vec::new()
        } else {
            self.forward_valid(&mut active, &ids)?
        };
        let mut outcomes = outcomes.into_iter();
        Ok(invalid
            .into_iter()
            .map(|e| match e {
                Some(e) => Err(e),
                None => outcomes.next().unwrap(),
            })
            .collect())
    }

    fn forward_valid(
        &mut self,
        items: &mut [&mut ForwardItem<'_>],
        ids: &[u32],
    ) -> Result<Vec<Result<Target>>> {
        let mut errors = (0..items.len()).map(|_| None).collect::<Vec<_>>();
        let rows = ids.len();
        let capture = items.iter().any(|s| s.capture_features);
        let mut features = None;
        self.rt.layer(0)?;
        let mut h = residual::embed(&mut self.rt, ids)?;
        let mut mix = residual::initial_mix(&mut self.rt, rows)?;
        let mut gathered: Option<EngramRows> = None;
        for layer in 0..40 {
            let stem = format!("layers.{layer}");
            if layer == 20 {
                let host_h = self.rt.to_host(&h)?;
                let host_mix = self.rt.to_host(&mix)?;
                self.rt.layer(layer)?;
                h = self.rt.to_device(&host_h)?;
                mix = self.rt.to_device(&host_mix)?;
            }
            if let Some(engram) = gathered.take() {
                engram.apply(&mut self.rt, &mut h)?;
            }
            if capture && layer >= 37 {
                if layer == 37 {
                    features = Some(self.rt.buffer(rows * 3 * DIM)?);
                }
                residual::capture(&self.rt, &h, features.as_mut().unwrap(), layer - 37, rows)?;
            }
            let x = residual::prepare(&mut self.rt, &stem, "attn", &h, &mut mix, rows)?;
            let y = attend(&mut self.rt, items, &mut errors, layer, &x)?;
            if errors.iter().all(Option::is_some) {
                return Ok(errors.into_iter().map(|e| Err(e.unwrap())).collect());
            }
            residual::expand(&self.rt, &mut h, &y, &mix, rows)?;
            drop((x, y));
            if layer == 0 || layer == 13 {
                for s in items.iter() {
                    let next =
                        self.engram
                            .gather(&self.rt.weights, layer + 1, s.ids, s.pos, s.rows)?;
                    if let Some(all) = &mut gathered {
                        all.append(next);
                    } else {
                        gathered = Some(next);
                    }
                }
            }
            let x = residual::prepare(&mut self.rt, &stem, "ffn", &h, &mut mix, rows)?;
            let y = moe::forward_dev(&mut self.rt, layer, &x, rows)?;
            residual::expand(&self.rt, &mut h, &y, &mix, rows)?;
        }

        for (s, error) in items.iter_mut().zip(&errors) {
            if error.is_some() {
                s.logit_rows = 0;
                s.capture_features = false;
            }
        }
        let logits_to_return: usize = items.iter().map(|s| s.logit_rows).sum();
        // Decode uses one joint vocabulary projection. Single-stream prefill
        // needs only the last row. Discontinuous row ranges can occur on a
        // final chunk from more than one prompt.
        let mut logits = None;
        if logits_to_return > 0 {
            let inputs = if logits_to_return == rows || items.len() == 1 {
                residual::readout(
                    &mut self.rt,
                    &h,
                    &mut mix,
                    "norm.weight",
                    rows - logits_to_return,
                    logits_to_return,
                )?
            } else {
                let mut selected = self.rt.buffer(logits_to_return * DIM)?;
                let (mut end, mut dst) = (0, 0);
                for s in items.iter() {
                    end += s.rows;
                    if s.logit_rows == 0 {
                        continue;
                    }
                    let input = residual::readout(
                        &mut self.rt,
                        &h,
                        &mut mix,
                        "norm.weight",
                        end - s.logit_rows,
                        s.logit_rows,
                    )?;
                    self.rt.copy_into(&mut selected, dst, &input)?;
                    dst += input.len();
                }
                selected
            };
            logits = Some(
                self.rt
                    .linear_dev("head.weight", &inputs, logits_to_return, 1)?,
            );
        }
        if items.len() == 1 {
            PEAK_BATCH.fetch_max(1, Ordering::Relaxed);
            return Ok(vec![Ok(Target { logits, features })]);
        }
        let mut targets = Vec::with_capacity(items.len());
        let mut feature_offset = 0;
        let mut logit_offset = 0;
        for (s, error) in items.iter().zip(errors) {
            let feature_len = s.rows * 3 * DIM;
            let logit_len = s.logit_rows * 129280;
            if let Some(e) = error {
                targets.push(Err(e));
                feature_offset += feature_len;
                continue;
            }
            targets.push(Ok(Target {
                logits: if logit_len > 0 {
                    Some(
                        self.rt
                            .slice(logits.as_ref().unwrap(), logit_offset, logit_len)?,
                    )
                } else {
                    None
                },
                features: if s.capture_features {
                    Some(
                        self.rt
                            .slice(features.as_ref().unwrap(), feature_offset, feature_len)?,
                    )
                } else {
                    None
                },
            }));
            feature_offset += feature_len;
            logit_offset += logit_len;
        }
        let completed = targets.iter().filter(|r| r.is_ok()).count();
        PEAK_BATCH.fetch_max(completed, Ordering::Relaxed);
        if completed == 6 {
            SIX_PASSES.fetch_add(1, Ordering::Relaxed);
        }
        if completed > 1 {
            MULTI_PASSES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::AttentionState;

    #[test]
    fn target_input_checks_engram_history_and_token_bounds() {
        let mut cache = AttentionState::default();
        let mut item = ForwardItem {
            cache: &mut cache,
            ids: &[129280, 10, 20, 30, 40],
            pos: 4,
            rows: 1,
            logit_rows: 1,
            capture_features: false,
        };
        assert_eq!(history(&item).unwrap(), [40]);
        item.pos = 3;
        assert!(history(&item).is_err());
        item.pos = usize::MAX;
        assert!(history(&item).is_err());
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and CUDA_VISIBLE_DEVICES set to both inference GPUs"]
    fn cuda_mixed_requests_finish_around_an_invalid_kv_session() -> Result<()> {
        use std::path::PathBuf;
        use tokenizers::Tokenizer;
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask both inference GPUs"
        );
        let pack = PathBuf::from(std::env::var("DS4_TEST_PACK")?);
        let tok = Tokenizer::from_file(pack.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut model = Model::open(&pack, &tok, [0, 1], 128, 16)?;
        let inputs = [vec![123, 221, 222], vec![17, 1234], vec![44, 55, 66, 77]];
        let mut expected = Vec::new();
        for ids in [&inputs[0], &inputs[2]] {
            let mut output = Vec::new();
            model.generate(ids, 2, 0., |id| {
                output.push(id);
                true
            })?;
            expected.push(output);
        }

        let mut sessions = inputs
            .into_iter()
            .map(|ids| model.start_request(ids, 2, 0.))
            .collect::<Result<Vec<_>>>()?;
        let mut delivered = vec![Vec::new(); sessions.len()];
        for (ids, step) in delivered
            .iter_mut()
            .zip(model.step_batch(&mut sessions.iter_mut().collect::<Vec<_>>()))
        {
            let step = step?;
            ensure!(!step.finished, "test prompt emitted EOS before decode");
            ids.extend(step.tokens);
        }

        sessions[1].cache = AttentionState::default();
        let second = model.step_batch(&mut sessions.iter_mut().collect::<Vec<_>>());
        for (i, step) in second.into_iter().enumerate() {
            if i == 1 {
                assert!(attention::request_local(
                    &step.err().context("lost invalid KV fault")?
                ));
            } else {
                let step = step?;
                assert!(step.finished);
                delivered[i].extend(step.tokens);
            }
        }
        assert_eq!(delivered[0], expected[0]);
        assert_eq!(delivered[2], expected[1]);
        Ok(())
    }
}
