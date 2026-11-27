//! Read-only tensor views into the pack's memory-mapped safetensors shards.
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    path::{Component, Path},
};

#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}
struct Entry {
    shard: usize,
    start: usize,
    end: usize,
    dtype: String,
    shape: Vec<usize>,
}
pub struct Tensor<'a> {
    pub data: &'a [u8],
    pub shape: &'a [usize],
    pub dtype: &'a str,
}
pub struct Weights {
    maps: Vec<Mmap>,
    entries: HashMap<String, Entry>,
}

fn element_bytes(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "U8" | "I8" | "F8_E8M0" | "F8_E4M3" => 1,
        "BF16" | "F16" | "I16" => 2,
        "F32" | "I32" => 4,
        "I64" => 8,
        s => bail!("unsupported checkpoint dtype {s}"),
    })
}
impl Weights {
    pub fn open(root: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let index: Index =
            serde_json::from_reader(File::open(root.join("model.safetensors.index.json"))?)?;
        let random_engram = match std::env::var("DS4_ENGRAM_RANDOM_READ").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(std::env::VarError::NotPresent) => false,
            _ => bail!("DS4_ENGRAM_RANDOM_READ must be 0 or 1"),
        };
        let shards: BTreeSet<_> = index.weight_map.values().collect();
        let mut w = Self {
            maps: vec![],
            entries: HashMap::new(),
        };
        for shard in shards {
            let name = Path::new(shard);
            ensure!(
                name.components().count() == 1
                    && matches!(name.components().next(), Some(Component::Normal(_))),
                "unexpected shard path"
            );
            let file = File::open(root.join(name)).with_context(|| format!("open {shard}"))?;
            // The checkpoint must remain immutable while mapped.
            let map = unsafe { Mmap::map(&file) }?;
            let len = u64::from_le_bytes(map.get(..8).context("short shard")?.try_into()?) as usize;
            let base = len.checked_add(8).context("header overflow")?;
            ensure!(len <= 100_000_000, "oversized safetensors header");
            let mut header: serde_json::Map<_, _> =
                serde_json::from_slice(map.get(8..base).context("truncated header")?)?;
            header.remove("__metadata__");
            for (key, value) in header {
                let h: Header = serde_json::from_value(value)?;
                ensure!(
                    index.weight_map.get(&key) == Some(shard),
                    "index mismatch for {key}"
                );
                let bytes = h
                    .shape
                    .iter()
                    .copied()
                    .try_fold(element_bytes(&h.dtype)?, usize::checked_mul)
                    .context("shape overflow")?;
                let [a, b] = h.data_offsets;
                ensure!(
                    b.checked_sub(a) == Some(bytes) && b <= map.len() - base,
                    "invalid extent for {key}"
                );
                // Engram hashes address independent 128-byte rows in tables on NVMe.
                // Restrict the mapping hint to the table so dense weights keep readahead.
                if random_engram
                    && bytes != 0
                    && (key.ends_with(".engram.embed.weight")
                        || key.ends_with(".engram.embed.scale"))
                {
                    map.advise_range(memmap2::Advice::Random, base + a, bytes)
                        .with_context(|| format!("random access hint for {key}"))?;
                }
                ensure!(
                    w.entries
                        .insert(
                            key,
                            Entry {
                                shard: w.maps.len(),
                                start: base + a,
                                end: base + b,
                                dtype: h.dtype,
                                shape: h.shape
                            }
                        )
                        .is_none(),
                    "duplicate tensor"
                );
            }
            w.maps.push(map);
        }
        ensure!(
            w.entries.len() == index.weight_map.len(),
            "missing indexed tensors"
        );
        Ok(w)
    }
    /// Swap only the three MTP shards after checking the complete overlay.
    pub fn overlay_mtp(&mut self, root: &Path) -> Result<()> {
        let overlay = Self::open(root)?;
        ensure!(
            overlay.entries.len()
                == self
                    .entries
                    .keys()
                    .filter(|k| k.starts_with("mtp."))
                    .count(),
            "incomplete MTP overlay"
        );
        for (name, item) in &overlay.entries {
            let original = self
                .entries
                .get(name)
                .context("unexpected drafter tensor")?;
            ensure!(
                name.starts_with("mtp.")
                    && original.shape == item.shape
                    && original.dtype == item.dtype,
                "incompatible drafter tensor {name}"
            );
        }
        let offset = self.maps.len();
        self.maps.extend(overlay.maps);
        self.entries
            .extend(overlay.entries.into_iter().map(|(key, mut e)| {
                e.shard += offset;
                (key, e)
            }));
        Ok(())
    }
    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let t = self
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        Ok(Tensor {
            data: &self.maps[t.shard][t.start..t.end],
            shape: &t.shape,
            dtype: &t.dtype,
        })
    }
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        Ok(match t.dtype {
            "F32" => t
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            "BF16" => t
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| half::bf16::from_le_bytes([v[0], v[1]]).to_f32())
                .collect(),
            "F16" => t
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| half::f16::from_le_bytes([v[0], v[1]]).to_f32())
                .collect(),
            s => bail!("cannot read {name} as a vector ({s})"),
        })
    }
}
