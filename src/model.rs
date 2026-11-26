use crate::{
    attention::{self, AttentionState},
    engram::Engram,
    moe,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Result};
use rand::Rng;
use std::path::Path;
use tokenizers::Tokenizer;

#[path = "residual.rs"]
mod residual;

pub const DIM: usize = 5120;
#[inline(always)]
pub fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
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
    fn forward_hidden(
        &mut self,
        cache: &mut AttentionState,
        ids: &[u32],
        pos: usize,
        b: usize,
    ) -> Result<(Dev, Dev)> {
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
        self.rt.layer(0)?;
        let mut h = self.rt.to_device(&h)?;
        let mut mix = residual::initial_mix(&mut self.rt, b)?;
        for layer in 0..40 {
            if layer == 20 {
                let host_h = self.rt.to_host(&h)?;
                let host_mix = self.rt.to_host(&mix)?;
                self.rt.layer(layer)?;
                h = self.rt.to_device(&host_h)?;
                mix = self.rt.to_device(&host_mix)?;
            }
            if layer == 1 || layer == 14 {
                let mut host_h = self.rt.to_host(&h)?;
                self.engram
                    .apply(&mut self.rt, layer, ids, pos, &mut host_h)?;
                h = self.rt.to_device(&host_h)?;
            }
            for branch in ["attn", "ffn"] {
                let input = residual::prepare(&mut self.rt, layer, branch, &h, &mut mix, b)?;
                let output = if branch == "attn" {
                    attention::forward_dev(&mut self.rt, cache, layer, pos, &input, b)?
                } else {
                    moe::forward_dev(&mut self.rt, layer, &input, b)?
                };
                residual::expand(&self.rt, &mut h, &output, &mix, b)?;
            }
        }
        Ok((h, mix))
    }

    fn forward(
        &mut self,
        cache: &mut AttentionState,
        ids: &[u32],
        pos: usize,
        b: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let (h, mut mix) = self.forward_hidden(cache, ids, pos, b)?;
        if !want_logits {
            return Ok(vec![]);
        }
        let input = residual::readout(&mut self.rt, &h, &mut mix, b - 1)?;
        let logits = self.rt.linear_dev("head.weight", &input, 1, 1)?;
        self.rt.to_host(&logits)
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
                    attention: cache.host_snapshot()?,
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
