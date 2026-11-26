//! DSpark's noise block is refined by three MTP layers, then a rank-256
//! Markov head conditions each proposed token on the one proposed before it.
use crate::{
    draft_attention::DraftAttention,
    model::residual,
    moe,
    runtime::{Dev, Runtime},
};
use anyhow::{ensure, Result};

pub struct DraftState {
    attention: DraftAttention,
}
impl DraftState {
    pub fn new() -> Self {
        Self {
            attention: DraftAttention::default(),
        }
    }
    pub fn ready(&self, pos: usize) -> bool {
        self.attention.ready(pos)
    }
    pub fn snapshot(&self, rt: &mut Runtime, pos: usize) -> Result<Vec<u8>> {
        self.attention.snapshot(rt, pos)
    }
    pub fn restore(rt: &mut Runtime, pos: usize, bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            attention: DraftAttention::restore(rt, pos, bytes)?,
        })
    }
    pub fn observe(
        &mut self,
        rt: &mut Runtime,
        features: &Dev,
        pos: usize,
        committed_rows: usize,
    ) -> Result<()> {
        self.attention.observe(rt, features, pos, committed_rows)
    }
    pub fn propose(
        &mut self,
        rt: &mut Runtime,
        pos: usize,
        seed: u32,
        width: usize,
    ) -> Result<Vec<u32>> {
        ensure!(
            (1..=16).contains(&width),
            "DSpark proposal length out of range"
        );
        let mut noise = vec![128799; width];
        noise[0] = seed;
        let mut h = residual::embed(rt, &noise)?;
        let mut mix = residual::initial_mix(rt, width)?;
        for stage in 0..3 {
            let stem = format!("mtp.{stage}");
            let x = residual::prepare(rt, &stem, "attn", &h, &mut mix, width)?;
            let x = self.attention.forward(rt, stage, &x, pos, width)?;
            residual::expand(rt, &mut h, &x, &mix, width)?;
            let x = residual::prepare(rt, &stem, "ffn", &h, &mut mix, width)?;
            let x = moe::draft_forward(rt, stage, &x, width)?;
            residual::expand(rt, &mut h, &x, &mix, width)?;
        }
        let x = residual::readout(rt, &h, &mut mix, "mtp.2.norm.weight", 0, width)?;
        let logits = rt.linear_dev("head.weight", &x, width, 1)?;
        rt.markov(&logits, seed, width)
    }
}
