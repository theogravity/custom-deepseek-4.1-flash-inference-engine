//! Text n-gram hashing and row-wise int4 lookup for the two Engram layers.
use crate::{
    model::{bf, DIM},
    runtime::Runtime,
};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use tokenizers::{normalizers::NormalizerWrapper, NormalizedString, Normalizer, Tokenizer};

// PCG64(10007 * layer).integers(0, floor(i64::MAX / 99092 / 2), size=4) * 2 + 1.
const MULT: [[u64; 4]; 2] = [
    [
        76632096046245,
        4839876093313,
        35959672319349,
        73987337458391,
    ],
    [
        67716810739261,
        51510806800915,
        30921347202721,
        82619226485591,
    ],
];
pub struct Engram {
    map: Vec<u32>,
    primes: [[usize; 24]; 2],
}
impl Engram {
    pub fn new(tok: &Tokenizer) -> Result<Self> {
        let normalizer: NormalizerWrapper =
            serde_json::from_value(serde_json::json!({"type":"Sequence","normalizers":[
          {"type":"NFKC"},{"type":"NFD"},{"type":"StripAccents"},{"type":"Lowercase"},
          {"type":"Replace","pattern":{"Regex":"[ \\t\\r\\n]+"},"content":" "},
          {"type":"Replace","pattern":{"Regex":"^ $"},"content":"\u{e000}"},
          {"type":"Strip","strip_left":true,"strip_right":true},
          {"type":"Replace","pattern":{"String":"\u{e000}"},"content":" "}] }))?;
        ensure!(
            tok.get_vocab_size(true) == 129280,
            "unexpected checkpoint tokenizer size"
        );
        let mut ids = HashMap::new();
        let mut map = Vec::new();
        for i in 0..tok.get_vocab_size(true) as u32 {
            ensure!(tok.id_to_token(i).is_some(), "missing token ID {i}");
            let s = tok
                .decode(&[i], false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let key = if s.contains('\u{fffd}') {
                tok.id_to_token(i).context("missing byte token")?
            } else {
                let mut n = NormalizedString::from(s.as_str());
                normalizer
                    .normalize(&mut n)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if n.get().is_empty() {
                    s
                } else {
                    n.get().to_owned()
                }
            };
            let next = ids.len() as u32;
            map.push(*ids.entry(key).or_insert(next));
        }
        ensure!(
            ids.len() == 99092,
            "Engram tokenizer normalization has {} entries, expected 99092",
            ids.len()
        );
        let mut primes = [[0; 24]; 2];
        let mut n: usize = 15999999;
        for p in primes.iter_mut().flatten() {
            loop {
                n += 1;
                if n % 2 == 1 && (3..=n.isqrt()).step_by(2).all(|d| !n.is_multiple_of(d)) {
                    break;
                }
            }
            *p = n;
        }
        ensure!(
            primes[0].iter().sum::<usize>() == 384006168
                && primes[1].iter().sum::<usize>() == 384016682,
            "Engram bucket mismatch"
        );
        Ok(Self { map, primes })
    }
    pub fn apply(
        &self,
        rt: &mut Runtime,
        layer: usize,
        history: &[u32],
        pos: usize,
        h: &mut [f32],
    ) -> Result<()> {
        let li = usize::from(layer == 14);
        let batch = h.len() / (4 * DIM);
        let prefix = format!("layers.{layer}.engram");
        let packed = rt.weights.tensor(&format!("{prefix}.embed.weight"))?;
        let scales = rt.weights.tensor(&format!("{prefix}.embed.scale"))?;
        let rows = self.primes[li].iter().sum::<usize>();
        ensure!(
            packed.dtype == "U8"
                && scales.dtype == "F16"
                && packed.shape == [rows, 128]
                && scales.shape == [rows, 8],
            "Engram layout mismatch"
        );
        let mut embeddings = vec![0.; batch * 6144];
        embeddings
            .par_chunks_exact_mut(6144)
            .enumerate()
            .for_each(|(i, out)| {
                let mut rolling = 0;
                let mut offset = 0;
                for (shift, &multiplier) in MULT[li].iter().enumerate() {
                    let id = (pos + i)
                        .checked_sub(shift)
                        .map(|at| history[at])
                        .unwrap_or(2);
                    rolling ^= self.map[id as usize] as u64 * multiplier;
                    if shift == 0 {
                        continue;
                    }
                    for j in ((shift - 1) * 8)..(shift * 8) {
                        let prime = self.primes[li][j];
                        let row = offset + (rolling % prime as u64) as usize;
                        offset += prime;
                        for (g, dst) in out[j * 256..(j + 1) * 256]
                            .as_chunks_mut::<32>()
                            .0
                            .iter_mut()
                            .enumerate()
                        {
                            let k = row * 16 + g * 2;
                            let scale =
                                half::f16::from_le_bytes([scales.data[k], scales.data[k + 1]])
                                    .to_f32();
                            for (d, y) in dst.iter_mut().enumerate() {
                                let byte = packed.data[row * 128 + (g * 32 + d) / 2];
                                *y =
                                    bf((i32::from((byte >> (4 * (d % 2))) & 15) - 8) as f32
                                        * scale);
                            }
                        }
                    }
                }
            });
        let kv = rt.linear(&format!("{prefix}.wkv.weight"), &embeddings, batch)?;
        let q = rt.weights.vector(&format!("{prefix}.q_weight"))?;
        let k = rt.weights.vector(&format!("{prefix}.k_weight"))?;
        ensure!(
            q.len() == 4 * DIM && k.len() == q.len() && kv.len() == batch * 5 * DIM,
            "Engram projection mismatch"
        );
        h.par_chunks_exact_mut(4 * DIM)
            .zip(kv.par_chunks_exact(5 * DIM))
            .for_each(|(h, kv)| {
                for c in 0..4 {
                    let start = c * DIM;
                    let old = &mut h[start..start + DIM];
                    let key = &kv[start..start + DIM];
                    let (mut hh, mut kk, mut dot) = (0., 0., 0.);
                    for d in 0..DIM {
                        hh += old[d] * old[d];
                        kk += key[d] * key[d];
                        dot += old[d] * (q[start + d] * k[start + d]) * key[d];
                    }
                    dot /=
                        ((hh / DIM as f32 + 1e-20) * (kk / DIM as f32 + 1e-20) * DIM as f32).sqrt();
                    let gate = 1. / (1. + (-dot.abs().max(1e-6).sqrt().copysign(dot)).exp());
                    for d in 0..DIM {
                        old[d] = bf(old[d] + gate * kv[4 * DIM + d]);
                    }
                }
            });
        Ok(())
    }
}
