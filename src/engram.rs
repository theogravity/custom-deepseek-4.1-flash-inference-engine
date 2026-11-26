//! Text n-gram hashing and row-wise int4 lookup for the two Engram layers.
use crate::{
    model::DIM,
    runtime::{Dev, Runtime},
    weights::Weights,
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
const HEADS: usize = 24;
const EMBED_DIM: usize = 256;
const ROW_BYTES: usize = EMBED_DIM / 2 + 8 * 2;
type PackedRow = [u8; ROW_BYTES];

unsafe extern "C" {
    fn ds_engram_unpack(
        device: i32,
        host: *const u8,
        packed: *mut u8,
        out: *mut f32,
        rows: i32,
    ) -> i32;
    fn ds_engram_apply(
        device: i32,
        h: *mut f32,
        kv: *const f32,
        q: *const f32,
        k: *const f32,
        batch: i32,
    ) -> i32;
}

pub struct EngramRows {
    prefix: String,
    records: Vec<[PackedRow; HEADS]>,
}

impl EngramRows {
    pub fn apply(&self, rt: &mut Runtime, h: &mut Dev) -> Result<()> {
        let batch = self.records.len();
        ensure!(h.len() == batch * 4 * DIM, "invalid Engram hidden size");
        let rows = batch * HEADS;
        let mut packed = rt.buffer(rows * ROW_BYTES / 4)?;
        let mut embeddings = rt.buffer(rows * EMBED_DIM)?;
        let status = unsafe {
            ds_engram_unpack(
                rt.device,
                self.records.as_ptr().cast(),
                packed.as_mut_ptr().cast(),
                embeddings.as_mut_ptr(),
                rows.try_into()?,
            )
        };
        ensure!(status == 0, "CUDA Engram unpack failed ({status})");
        drop(packed);
        let kv = rt.linear_dev(
            &format!("{}.wkv.weight", self.prefix),
            &embeddings,
            batch,
            1,
        )?;
        ensure!(kv.len() == batch * 5 * DIM, "Engram projection mismatch");
        let q = rt.vector_ptr(&format!("{}.q_weight", self.prefix), 4 * DIM)?;
        let k = rt.vector_ptr(&format!("{}.k_weight", self.prefix), 4 * DIM)?;
        let status = unsafe {
            ds_engram_apply(
                rt.device,
                h.as_mut_ptr(),
                kv.as_ptr(),
                q,
                k,
                batch.try_into()?,
            )
        };
        ensure!(status == 0, "CUDA Engram update failed ({status})");
        Ok(())
    }
}

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
    pub fn gather(
        &self,
        weights: &Weights,
        layer: usize,
        history: &[u32],
        pos: usize,
        batch: usize,
    ) -> Result<EngramRows> {
        ensure!(
            [1, 14].contains(&layer)
                && batch > 0
                && pos
                    .checked_add(batch)
                    .is_some_and(|end| end <= history.len()),
            "invalid Engram positions"
        );
        ensure!(
            history[pos.saturating_sub(3)..pos + batch]
                .iter()
                .all(|&id| (id as usize) < self.map.len()),
            "invalid Engram token id"
        );
        let li = usize::from(layer == 14);
        let prefix = format!("layers.{layer}.engram");
        let packed = weights.tensor(&format!("{prefix}.embed.weight"))?;
        let scales = weights.tensor(&format!("{prefix}.embed.scale"))?;
        let rows = self.primes[li].iter().sum::<usize>();
        ensure!(
            packed.dtype == "U8"
                && scales.dtype == "F16"
                && packed.shape == [rows, 128]
                && scales.shape == [rows, 8],
            "Engram layout mismatch"
        );
        let mut records = vec![[[0; ROW_BYTES]; HEADS]; batch];
        records.par_iter_mut().enumerate().for_each(|(i, out)| {
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
                let first = (shift - 1) * 8;
                for (j, record) in out[first..first + 8].iter_mut().enumerate() {
                    let prime = self.primes[li][first + j];
                    let row = offset + (rolling % prime as u64) as usize;
                    offset += prime;
                    record[..128].copy_from_slice(&packed.data[row * 128..(row + 1) * 128]);
                    record[128..].copy_from_slice(&scales.data[row * 16..(row + 1) * 16]);
                }
            }
        });
        Ok(EngramRows { prefix, records })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::bf;
    use std::ffi::c_void;

    unsafe extern "C" {
        fn ds_alloc_f32(dev: i32, out: *mut *mut f32, n: usize) -> i32;
        fn ds_copy_f32(dev: i32, dst: *mut f32, src: *const f32, n: usize, kind: i32) -> i32;
        fn ds_free(dev: i32, p: *mut c_void);
    }

    struct Fixture {
        device: i32,
        ptr: *mut f32,
    }

    impl Fixture {
        fn upload(device: i32, x: &[f32]) -> Self {
            let mut ptr = std::ptr::null_mut();
            assert_eq!(unsafe { ds_alloc_f32(device, &mut ptr, x.len()) }, 0);
            let b = Self { device, ptr };
            assert_eq!(
                unsafe { ds_copy_f32(device, b.ptr, x.as_ptr(), x.len(), 0) },
                0
            );
            b
        }
        fn read(&self, n: usize) -> Vec<f32> {
            let mut x = vec![0.; n];
            assert_eq!(
                unsafe { ds_copy_f32(self.device, x.as_mut_ptr(), self.ptr, n, 1) },
                0
            );
            x
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            unsafe { ds_free(self.device, self.ptr.cast()) };
        }
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to an inference GPU ordinal"]
    fn cuda_engram_matches_signed_nibbles_scales_and_scalar_update() -> Result<()> {
        let device: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let batch = 3;
        let records = batch * HEADS;
        let mut bytes = vec![[0u8; ROW_BYTES]; records];
        let mut expected = Vec::new();
        for (i, row) in bytes.iter_mut().enumerate() {
            for (d, b) in row[..128].iter_mut().enumerate() {
                *b = (i * 7 + d * 31) as u8;
            }
            for g in 0..8 {
                let scale = half::f16::from_f32((i as f32 - 17.) / (3. + g as f32));
                row[128 + 2 * g..130 + 2 * g].copy_from_slice(&scale.to_le_bytes());
                for j in 0..32 {
                    let byte = row[g * 16 + j / 2];
                    let nibble = ((byte >> ((j % 2) * 4)) & 15) as i32;
                    expected.push(bf((nibble - 8) as f32 * scale.to_f32()));
                }
            }
        }
        let packed = Fixture::upload(device, &vec![0.; records * ROW_BYTES / 4]);
        let out = Fixture::upload(device, &vec![0.; records * EMBED_DIM]);
        assert_eq!(
            unsafe {
                ds_engram_unpack(
                    device,
                    bytes.as_ptr().cast(),
                    packed.ptr.cast(),
                    out.ptr,
                    records as i32,
                )
            },
            0
        );
        assert_eq!(out.read(expected.len()), expected);
        let values = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| bf(((i * seed % 3931) as f32 - 1701.) / 917.))
                .collect()
        };
        let mut hidden = values(batch * 4 * DIM, 193);
        let kv = values(batch * 5 * DIM, 231);
        let mut q = values(4 * DIM, 511);
        q[..DIM].fill(0.);
        let k = values(4 * DIM, 643);
        let h_dev = Fixture::upload(device, &hidden);
        let kv_dev = Fixture::upload(device, &kv);
        let q_dev = Fixture::upload(device, &q);
        let k_dev = Fixture::upload(device, &k);
        assert_eq!(
            unsafe {
                ds_engram_apply(
                    device,
                    h_dev.ptr,
                    kv_dev.ptr,
                    q_dev.ptr,
                    k_dev.ptr,
                    batch as i32,
                )
            },
            0
        );
        for (h, kv) in hidden
            .as_chunks_mut::<{ 4 * DIM }>()
            .0
            .iter_mut()
            .zip(kv.as_chunks::<{ 5 * DIM }>().0)
        {
            for c in 0..4 {
                let start = c * DIM;
                let (mut hh, mut kk, mut dot) = (0., 0., 0.);
                for j in start..start + DIM {
                    hh += h[j] * h[j];
                    kk += kv[j] * kv[j];
                    dot += h[j] * (q[j] * k[j]) * kv[j];
                }
                dot /= ((hh / DIM as f32 + 1e-20) * (kk / DIM as f32 + 1e-20) * DIM as f32).sqrt();
                let gate = 1. / (1. + (-dot.abs().max(1e-6).sqrt().copysign(dot)).exp());
                for d in 0..DIM {
                    h[start + d] = bf(h[start + d] + gate * kv[4 * DIM + d]);
                }
            }
        }
        assert_eq!(h_dev.read(hidden.len()), hidden);
        Ok(())
    }
}
