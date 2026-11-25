use crate::{
    attention::{self, AttentionState},
    engram::Engram,
    moe,
    runtime::Runtime,
};
use anyhow::{ensure, Result};
use rand::Rng;
use rayon::prelude::*;
use std::path::Path;
use tokenizers::Tokenizer;

pub const DIM: usize = 5120;
#[inline(always)]
pub fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

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
fn expand(h: &mut [f32], x: &[f32], mixes: &[Mix]) {
    h.par_chunks_exact_mut(4 * DIM)
        .zip(x.par_chunks_exact(DIM))
        .zip(mixes.par_iter())
        .for_each(|((old, row), m)| {
            for i in 0..DIM {
                let r = [old[i], old[DIM + i], old[2 * DIM + i], old[3 * DIM + i]];
                for c in 0..4 {
                    old[c * DIM + i] =
                        bf(row[i] * m.post[c] + (0..4).map(|j| r[j] * m.comb[j][c]).sum::<f32>());
                }
            }
        });
}

struct PromptCache {
    window_span: usize,
    tokens: Vec<u32>,
    attention: AttentionState,
}

pub struct Model {
    prefix: Option<PromptCache>,
    rt: Runtime,
    engram: Engram,
    max_ctx: usize,
    chunk: usize,
}
impl Model {
    pub fn open(
        pack: &Path,
        tok: &Tokenizer,
        devices: [i32; 2],
        max_ctx: usize,
        chunk: usize,
    ) -> Result<Self> {
        ensure!(
            chunk > 0 && chunk <= 512 && (1..=524288).contains(&max_ctx),
            "invalid chunk or context limit"
        );
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(pack.join("config.json"))?)?;
        ensure!(
            cfg["model_type"] == "deepseek_v41"
                && cfg["text_config"]["hidden_size"] == DIM
                && cfg["text_config"]["num_hidden_layers"] == 40
                && cfg["eos_token_id"] == 1,
            "unexpected model configuration"
        );
        Ok(Self {
            prefix: None,
            rt: Runtime::open(pack, devices)?,
            engram: Engram::new(tok)?,
            max_ctx,
            chunk,
        })
    }
    pub fn max_context(&self) -> usize {
        self.max_ctx
    }
    fn forward(
        &mut self,
        cache: &mut AttentionState,
        ids: &[u32],
        pos: usize,
        b: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let e = self.rt.weights.tensor("embed.weight")?;
        ensure!(
            e.dtype == "BF16" && e.shape == [129280, DIM],
            "invalid token embedding"
        );
        let mut h = Vec::with_capacity(b * 4 * DIM);
        for &id in &ids[pos..pos + b] {
            let offset = id as usize * 2 * DIM;
            let row: Vec<f32> = e.data[offset..offset + 2 * DIM]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16))
                .collect();
            for _ in 0..4 {
                h.extend_from_slice(&row);
            }
        }
        let mut pre = vec![[1., 0., 0., 0.]; b];
        for layer in 0..40 {
            self.rt.layer(layer)?;
            if layer == 1 || layer == 14 {
                self.engram.apply(&mut self.rt, layer, ids, pos, &mut h)?;
            }
            for branch in ["attn", "ffn"] {
                let stem = format!("layers.{layer}");
                let projection = self.rt.linear(&format!("{stem}.hc_{branch}_fn"), &h, b)?;
                let scale = self
                    .rt
                    .weights
                    .vector(&format!("{stem}.hc_{branch}_scale"))?;
                let base = self
                    .rt
                    .weights
                    .vector(&format!("{stem}.hc_{branch}_base"))?;
                ensure!(
                    projection.len() == 24 * b && scale.len() == 3 && base.len() == 24,
                    "invalid hyper-connection parameters"
                );
                let mixes: Vec<Mix> = h
                    .par_chunks_exact(4 * DIM)
                    .zip(projection.par_chunks_exact(24))
                    .map(|(token, logits)| {
                        let inv = 1.
                            / (token.iter().map(|x| x * x).sum::<f32>() / (4 * DIM) as f32 + 1e-20)
                                .sqrt();
                        mix(logits, inv, &base, &scale)
                    })
                    .collect();
                let norm = self
                    .rt
                    .weights
                    .vector(&format!("{stem}.{branch}_norm.weight"))?;
                ensure!(norm.len() == DIM, "invalid norm at {stem}");
                let input = collapse(&h, &pre, &norm);
                let output = if branch == "attn" {
                    attention::forward(&mut self.rt, cache, layer, pos, &input, b)?
                } else {
                    moe::forward(&mut self.rt, layer, &input, b)?
                };
                ensure!(output.len() == DIM * b, "invalid {branch} result at {stem}");
                expand(&mut h, &output, &mixes);
                for (p, m) in pre.iter_mut().zip(mixes) {
                    *p = m.pre;
                }
            }
        }
        if !want_logits {
            return Ok(vec![]);
        }
        let norm = self.rt.weights.vector("norm.weight")?;
        ensure!(norm.len() == DIM, "invalid final norm");
        let x = collapse(&h[(b - 1) * 4 * DIM..], &pre[b - 1..], &norm);
        self.rt.linear("head.weight", &x, 1)
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        temperature: f32,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.generate_cancellable(input, limit, temperature, || false, emit)
    }
    pub fn generate_cancellable(
        &mut self,
        input: &[u32],
        limit: usize,
        temperature: f32,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        ensure!(
            !input.is_empty() && input.len() <= self.max_ctx && input.iter().all(|t| *t < 129280),
            "invalid prompt length or token ID"
        );
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "invalid temperature"
        );
        let limit = limit.min(self.max_ctx - input.len());
        if limit == 0 {
            return Ok(0);
        }
        let mut ids = input.to_vec();
        // Cache a completed prefill chunk. Reusing the same chunk boundaries preserves
        // the sparse attention and projection rounding seen by a fresh request.
        let snapshot_pos = (input.len() - 1) / self.chunk * self.chunk;
        let prefix = self.prefix.take().filter(|p| {
            p.window_span == input.len().min(128)
                && p.tokens.len() <= snapshot_pos
                && input.starts_with(&p.tokens)
        });
        let (mut pos, mut cache) = prefix.map_or_else(
            || (0, AttentionState::default()),
            |p| (p.tokens.len(), p.attention),
        );
        cache.begin_prefill(input.len());
        let reused = pos;
        let started = std::time::Instant::now();
        let mut logits = vec![];
        while pos < input.len() {
            ensure!(!cancelled(), "request cancelled");
            if pos > 0 && pos == snapshot_pos {
                self.prefix = Some(PromptCache {
                    window_span: input.len().min(128),
                    tokens: input[..pos].to_vec(),
                    attention: cache.host_snapshot(),
                });
            }
            let n = self.chunk.min(input.len() - pos);
            logits = self.forward(&mut cache, &ids, pos, n, pos + n == input.len())?;
            pos += n;
        }
        eprintln!(
            "prefill tokens={} reused={} seconds={:.3}",
            input.len(),
            reused,
            started.elapsed().as_secs_f64()
        );
        let mut rng = rand::thread_rng();
        for n in 1..=limit {
            ensure!(
                logits.len() == 129280 && logits.iter().all(|x| x.is_finite()),
                "invalid logits at position {pos}"
            );
            let mut best = f32::NEG_INFINITY;
            let mut token = 0;
            for (i, v) in logits.iter().copied().enumerate() {
                let score = if temperature == 0. {
                    v
                } else {
                    let u: f32 = rng.sample(rand::distributions::Open01);
                    v - temperature * (-u.ln()).ln()
                };
                if score > best {
                    best = score;
                    token = i as u32;
                }
            }
            if token == 1 || !emit(token) || n == limit {
                return Ok(n);
            }
            ids.push(token);
            logits = self.forward(&mut cache, &ids, pos, 1, true)?;
            pos += 1;
        }
        unreachable!()
    }
}
