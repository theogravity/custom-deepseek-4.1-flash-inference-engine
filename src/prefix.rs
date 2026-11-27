//! Content-addressed snapshots at exact prefill boundaries. A file contains
//! target KV only. The decoder always evaluates the remaining prompt and output.
use crate::{
    attention::AttentionState,
    restore_timing::{timed, Host, Tier},
};
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::SystemTime,
};

type Key = [u8; 32];
const MAGIC: &[u8; 8] = b"DS4KV004";
const HEADER: usize = 8 + 32 + 32;
const MIB: usize = 1 << 20;

fn mib_env(name: &str, default: usize) -> Result<usize> {
    std::env::var(name)
        .ok()
        .map_or(Ok(default), |v| {
            v.parse::<usize>()
                .with_context(|| format!("invalid {name}"))
        })?
        .checked_mul(MIB)
        .context("cache budget overflows address space")
}
fn hex(key: &Key) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}
fn namespace(root: &Path, chunk: usize) -> Result<Sha256> {
    #[derive(Deserialize)]
    struct Index {
        weight_map: HashMap<String, String>,
    }
    let mut h = Sha256::new();
    h.update(MAGIC);
    h.update([
        u8::from(crate::model::residual::fast_rms()),
        u8::from(crate::runtime::packed_dense()),
        u8::from(crate::attention::fast_attention()),
        u8::from(std::env::var_os("DS4_TP_DENSE").is_some_and(|v| v == "1")),
    ]);
    if std::env::var_os("DS4_PREFILL_LT").is_some_and(|v| v == "1") {
        h.update(b"prefill cuBLASLt MXFP8 v1");
    }
    if std::env::var_os("DS4_PREFILL_BLAS").is_some_and(|v| v == "1") {
        h.update(b"wide cuBLAS FP32/BF16 prefill v1");
    }
    if crate::model::bounded_replay()? {
        h.update(b"bounded decoder tail with full source KV v2");
    }
    if crate::attention::prefill_gpu_pool_pairs() {
        h.update(b"resident two-token CUDA compressor pooling v1");
    }
    if crate::attention::flash_prefill() {
        h.update(b"FlashInfer 946200de SM120 DSV4 MG prompt attention v1");
    }
    if crate::attention::prefill_index_fp8() {
        h.update(b"SM120 prompt index E4M3 MMA v1");
    } else if crate::attention::prefill_index_mma() {
        h.update(b"SM120 prompt index BF16 MMA v1");
    }
    if std::env::var_os("DS4_RECIPE_PREFILL").is_some_and(|v| v == "1") {
        h.update(b"Diffbot S8B hot expert prefill v1");
    }
    if crate::moe::tp2_prefill_enabled()? {
        h.update(b"Diffbot S8B TP2 expert prefill v1");
    }
    h.update((chunk as u64).to_le_bytes());
    h.update(Sha256::digest(fs::read(root.join("tokenizer.json"))?));
    let config = fs::read(root.join("config.json"))?;
    h.update(Sha256::digest(config));
    let index = fs::read(root.join("model.safetensors.index.json"))?;
    h.update(Sha256::digest(&index));
    let index: Index = serde_json::from_slice(&index)?;
    for shard in index.weight_map.into_values().collect::<BTreeSet<_>>() {
        let path = Path::new(&shard);
        ensure!(
            matches!(
                path.components().next(),
                Some(std::path::Component::Normal(_))
            ) && path.components().count() == 1,
            "unexpected checkpoint shard"
        );
        let m = fs::metadata(root.join(path))?;
        h.update(shard.as_bytes());
        for v in [
            m.len(),
            m.ino(),
            m.mtime() as u64,
            m.mtime_nsec() as u64,
            m.ctime() as u64,
            m.ctime_nsec() as u64,
        ] {
            h.update(v.to_le_bytes());
        }
    }
    Ok(h)
}

pub struct PrefixCache {
    identity: Sha256,
    draft_identity: Key,
    chunk: usize,
    capacity: usize,
    ram_limit: usize,
    disk_limit: usize,
    dir: Option<PathBuf>,
    ram: VecDeque<(Key, Bytes)>,
    ram_bytes: usize,
}
impl PrefixCache {
    pub fn new(root: &Path, chunk: usize, capacity: usize) -> Result<Self> {
        let ram_limit = mib_env("DS4_PREFIX_RAM_MIB", 2048)?;
        let disk_limit = mib_env("DS4_PREFIX_DISK_MIB", 65536)?;
        let mut dir = std::env::var_os("DS4_PREFIX_DIR")
            .filter(|v| !v.is_empty() && disk_limit > 0)
            .map(PathBuf::from);
        if let Some(d) = &dir {
            if let Err(e) = fs::DirBuilder::new().recursive(true).mode(0o700).create(d) {
                eprintln!("KV disk directory unavailable: {e}");
                dir = None;
            }
        }
        let mut dh = Sha256::new();
        dh.update(b"DS4 draft window 1");
        let mode = std::env::var("DS4_DRAFTER").unwrap_or_else(|_| "none".into());
        dh.update(mode.as_bytes());
        if mode == "code" {
            let overlay = root.join("drafter-code-k7");
            dh.update(Sha256::digest(fs::read(
                overlay.join("model.safetensors.index.json"),
            )?));
            for name in [
                "model-00044-of-00048.safetensors",
                "model-00045-of-00048.safetensors",
                "model-00046-of-00048.safetensors",
            ] {
                let m = fs::metadata(overlay.join(name))?;
                for x in [
                    m.len(),
                    m.ino(),
                    m.mtime() as u64,
                    m.mtime_nsec() as u64,
                    m.ctime() as u64,
                    m.ctime_nsec() as u64,
                ] {
                    dh.update(x.to_le_bytes());
                }
            }
        }
        let mut out = Self {
            draft_identity: dh.finalize().into(),
            identity: namespace(root, chunk)?,
            chunk,
            capacity,
            ram_limit,
            disk_limit,
            dir,
            ram: VecDeque::new(),
            ram_bytes: 0,
        };
        if let Some(dir) = &out.dir {
            if let Err(e) = Self::lock_dir(dir).and_then(|_lock| out.trim_disk(0)) {
                eprintln!("KV disk store unavailable: {e:#}");
                out.dir = None;
            }
        }
        Ok(out)
    }
    pub fn enabled(&self) -> bool {
        self.ram_limit > 0 || self.dir.is_some()
    }
    pub fn can_warm(&self, pos: usize) -> bool {
        AttentionState::snapshot_bytes(pos).is_ok_and(|size| size <= self.ram_limit)
    }
    fn lock_dir(dir: &Path) -> Result<File> {
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(dir.join(".cache.lock"))?;
        f.lock()?;
        Ok(f)
    }
    fn draft_key(&self, tokens: &[u32], pos: usize, span: usize) -> Result<Key> {
        ensure!(
            pos > 0 && pos.is_multiple_of(self.chunk),
            "invalid drafter cache boundary"
        );
        let base = self
            .keys(tokens, pos, span)?
            .last()
            .context("empty drafter cache key")?
            .1;
        let mut h = Sha256::new();
        h.update(self.draft_identity);
        h.update(base);
        Ok(h.finalize().into())
    }
    pub fn read_draft(
        &mut self,
        tokens: &[u32],
        pos: usize,
        span: usize,
    ) -> Result<Option<(Tier, Vec<u8>)>> {
        if !self.enabled() {
            return Ok(None);
        }
        let key = self.draft_key(tokens, pos, span)?;
        if let Some(i) = self.ram.iter().position(|e| e.0 == key) {
            let entry = self.ram.remove(i).unwrap();
            let data = entry.1.to_vec();
            self.ram.push_back(entry);
            return Ok(Some((Tier::Ram, data)));
        }
        if let Some(path) = self.file(&key).filter(|p| p.exists()) {
            match self.read_blob(&path, &key, 2 * MIB).and_then(|data| {
                crate::draft_attention::validate_snapshot(pos, &data)?;
                Ok(data)
            }) {
                Ok(data) => {
                    self.remember(key, data.clone().into());
                    return Ok(Some((Tier::Disk, data)));
                }
                Err(e) => {
                    eprintln!("unreadable draft cache file: {e:#}");
                    self.discard(&path, &key, 2 * MIB, |b| {
                        crate::draft_attention::validate_snapshot(pos, b).is_ok()
                    });
                }
            }
        }
        Ok(None)
    }
    pub fn write_draft(
        &mut self,
        tokens: &[u32],
        pos: usize,
        span: usize,
        data: &[u8],
        ram_only: bool,
    ) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        crate::draft_attention::validate_snapshot(pos, data)?;
        if ram_only
            && AttentionState::snapshot_bytes(pos)?.saturating_add(data.len()) > self.ram_limit
        {
            return Ok(());
        }
        let key = self.draft_key(tokens, pos, span)?;
        if !self.ram.iter().any(|e| e.0 == key) {
            self.save(key, data.to_vec(), ram_only);
        }
        Ok(())
    }
    fn keys(&self, tokens: &[u32], pos: usize, span: usize) -> Result<Vec<(usize, Key)>> {
        ensure!(
            pos <= tokens.len()
                && pos <= self.capacity
                && self.chunk > 0
                && (1..=128).contains(&span),
            "invalid prefix lookup"
        );
        let mut state = self.identity.clone();
        state.update((span as u32).to_le_bytes());
        let mut output = Vec::with_capacity(pos / self.chunk);
        for (i, chunk) in tokens[..pos].chunks_exact(self.chunk).enumerate() {
            for x in chunk {
                state.update(x.to_le_bytes());
            }
            output.push(((i + 1) * self.chunk, state.clone().finalize().into()));
        }
        Ok(output)
    }
    fn file(&self, key: &Key) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|p| p.join(format!("{}.kv", hex(key))))
    }
    fn remember(&mut self, key: Key, data: Bytes) {
        if data.len() > self.ram_limit {
            return;
        }
        while self.ram_bytes + data.len() > self.ram_limit {
            if let Some((_, old)) = self.ram.pop_front() {
                self.ram_bytes -= old.len();
            } else {
                break;
            }
        }
        self.ram_bytes += data.len();
        self.ram.push_back((key, data));
    }
    fn trim_disk(&self, incoming: u64) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        let mut entries = vec![];
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(k) = name.to_str().and_then(|n| n.strip_suffix(".kv")) else {
                continue;
            };
            if k.len() != 64
                || !k.bytes().all(|b| b.is_ascii_hexdigit())
                || !entry.file_type()?.is_file()
            {
                continue;
            }
            let m = entry.metadata()?;
            entries.push((
                m.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                m.len(),
                entry.path(),
            ));
        }
        let mut total = entries
            .iter()
            .fold(incoming, |n, (_, sz, _)| n.saturating_add(*sz));
        entries.sort_unstable_by_key(|(time, _, _)| *time);
        for (_, sz, file) in entries {
            if total <= self.disk_limit as u64 {
                break;
            }
            match fs::remove_file(file) {
                Ok(()) => total = total.saturating_sub(sz),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    total = total.saturating_sub(sz)
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
    fn read_blob(&self, path: &Path, key: &Key, max_payload: usize) -> Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let total: usize = f.metadata()?.len().try_into()?;
        // The checkpoint uses fewer than 2 KiB of compressed KV per token,
        // plus at most 5 MiB for the forty sliding windows.
        ensure!(
            total >= HEADER && total <= self.disk_limit && total <= HEADER + max_payload,
            "invalid KV file size"
        );
        let mut header = [0u8; HEADER];
        f.read_exact(&mut header)?;
        ensure!(
            &header[..8] == MAGIC && &header[8..40] == key,
            "wrong KV file identity"
        );
        let mut data = vec![0u8; total - HEADER];
        f.read_exact(&mut data)?;
        ensure!(
            Sha256::digest(&data)[..] == header[40..],
            "damaged KV payload"
        );
        // Preserve recently restored prefixes when trimming the disk store.
        let _ = f.set_modified(SystemTime::now());
        Ok(data)
    }
    fn discard(&self, path: &Path, key: &Key, max_payload: usize, valid: impl Fn(&[u8]) -> bool) {
        if let Some(dir) = &self.dir {
            if let Ok(_lock) = Self::lock_dir(dir) {
                if !self
                    .read_blob(path, key, max_payload)
                    .is_ok_and(|b| valid(&b))
                {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
    fn decode_target(
        data: &Bytes,
        pos: usize,
        tier: Tier,
        timing: &mut Option<Host>,
    ) -> Result<AttentionState> {
        if let Some(profile) = timing {
            profile.backing_read(tier);
        }
        let (state, elapsed) = timed(timing.is_some(), || {
            AttentionState::decode_prefix(data, pos)
        });
        if let Some(profile) = timing {
            profile.decoded(data.len(), elapsed);
        }
        state
    }
    pub fn restore(
        &mut self,
        tokens: &[u32],
        max_pos: usize,
        span: usize,
    ) -> Result<Option<(usize, AttentionState, Option<Host>)>> {
        if self.ram_limit == 0 && self.dir.is_none() {
            return Ok(None);
        }
        let mut timing = Host::start("prefix");
        for (pos, key) in self.keys(tokens, max_pos, span)?.into_iter().rev() {
            if let Some(i) = self.ram.iter().position(|(k, _)| k == &key) {
                let entry = self.ram.remove(i).unwrap();
                let state = Self::decode_target(&entry.1, pos, Tier::Ram, &mut timing)?;
                self.ram.push_back(entry);
                eprintln!("prefix tier=ram tokens={pos}");
                return Ok(Some((pos, state, timing)));
            }
            if let Some(path) = self.file(&key) {
                if !path.exists() {
                    continue;
                }
                let read = self
                    .read_blob(&path, &key, 6 * MIB + 2048 * pos)
                    .and_then(|data| {
                        let data = Bytes::from(data);
                        let state = Self::decode_target(&data, pos, Tier::Disk, &mut timing)?;
                        Ok((state, data))
                    });
                match read {
                    Ok((state, data)) => {
                        self.remember(key, data);
                        eprintln!("prefix tier=disk tokens={pos}");
                        return Ok(Some((pos, state, timing)));
                    }
                    Err(e) => {
                        eprintln!("discarding unreadable prefix at {pos} tokens: {e:#}");
                        self.discard(&path, &key, 6 * MIB + 2048 * pos, |b| {
                            AttentionState::decode_snapshot(b, pos).is_ok()
                        });
                    }
                }
            }
        }
        Ok(None)
    }
    pub fn store(
        &mut self,
        tokens: &[u32],
        pos: usize,
        span: usize,
        state: &AttentionState,
        ram_only: bool,
    ) -> Result<()> {
        if self.ram_limit == 0 && self.dir.is_none() {
            return Ok(());
        }
        ensure!(
            pos > 0 && pos.is_multiple_of(self.chunk),
            "invalid prefill boundary"
        );
        let key = self
            .keys(tokens, pos, span)?
            .last()
            .context("empty prefix cache key")?
            .1;
        if let Some(i) = self.ram.iter().position(|(k, _)| k == &key) {
            if ram_only {
                // Keep the target ahead of unrelated entries when its drafter
                // is appended to the same bounded LRU immediately afterwards.
                let entry = self.ram.remove(i).unwrap();
                self.ram.push_back(entry);
            }
            return Ok(());
        }
        self.save(key, state.encode_snapshot(pos)?, ram_only);
        Ok(())
    }
    fn save(&mut self, key: Key, data: Vec<u8>, ram_only: bool) {
        if ram_only {
            self.remember(key, Bytes::from(data));
            return;
        }
        if let Some(path) = self
            .file(&key)
            .filter(|_| data.len() + HEADER <= self.disk_limit)
        {
            let persist = || -> Result<()> {
                let _lock = Self::lock_dir(path.parent().unwrap())?;
                if path.exists() {
                    return Ok(());
                }
                self.trim_disk((data.len() + HEADER) as u64)?;
                let mut f = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
                f.write_all(MAGIC)?;
                f.write_all(&key)?;
                f.write_all(&Sha256::digest(&data))?;
                f.write_all(&data)?;
                f.persist(&path)?;
                Ok(())
            };
            if !path.exists() {
                if let Err(e) = persist() {
                    eprintln!("KV disk store failed: {e:#}");
                }
            }
        }
        self.remember(key, Bytes::from(data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_and_tokenizer_bind_prefixes_to_their_target() -> Result<()> {
        let d = tempfile::tempdir()?;
        for (name, data) in [
            ("tokenizer.json", "{}"),
            ("config.json", "{}"),
            (
                "model.safetensors.index.json",
                r#"{"weight_map":{"x":"shard"}}"#,
            ),
            ("shard", "1"),
        ] {
            fs::write(d.path().join(name), data)?;
        }
        let original = namespace(d.path(), 64)?.finalize();
        fs::write(d.path().join("tokenizer.json"), "{ }")?;
        assert_ne!(original, namespace(d.path(), 64)?.finalize());
        fs::write(d.path().join("tokenizer.json"), "{}")?;
        assert_ne!(original, namespace(d.path(), 32)?.finalize());
        fs::write(d.path().join("shard"), "changed")?;
        assert_ne!(original, namespace(d.path(), 64)?.finalize());
        Ok(())
    }

    #[test]
    fn disk_lru_corruption_and_cross_branch_restore() -> Result<()> {
        let d = tempfile::tempdir()?;
        let make = || PrefixCache {
            identity: Sha256::new(),
            draft_identity: [0; 32],
            chunk: 1,
            capacity: 20,
            ram_limit: 0,
            disk_limit: 10 * MIB,
            dir: Some(d.path().into()),
            ram: VecDeque::new(),
            ram_bytes: 0,
        };
        let mut cache = make();
        // Position 1 contains one row in every window and pending pairs for
        // the three ratio-two sources. The remaining source has one full row.
        let mut data = vec![0u8; 16 + 40 * 512 * 2 + 3 * (2 * 512 * 4) + 584 + 132];
        data[..8].copy_from_slice(b"DS4ATT01");
        data[8] = 1;
        let state = AttentionState::decode_snapshot(&data, 1)?;
        let mut warm = make();
        warm.ram_limit = 2 * data.len();
        warm.store(&[77], 1, 128, &state, true)?;
        warm.ram_limit = data.len();
        let mut sidecar = vec![0; 16 + 3 * 512 * 2];
        sidecar[..8].copy_from_slice(b"DS4DRF01");
        sidecar[8..16].copy_from_slice(&1u64.to_le_bytes());
        warm.write_draft(&[77], 1, 128, &sidecar, true)?;
        assert!(warm.read_draft(&[77], 1, 128)?.is_none());
        assert!(warm.restore(&[77], 1, 128)?.is_some());
        assert!(cache.restore(&[77], 1, 128)?.is_none());
        assert!(!warm.file(&warm.keys(&[77], 1, 128)?[0].1).unwrap().exists());
        cache.store(&[22, 33], 1, 128, &state, false)?;
        cache.store(&[23, 88], 1, 128, &state, false)?;
        let mut restarted = make();
        restarted.ram_limit = 2 * data.len();
        let (pos, held, _) = restarted.restore(&[22, 99], 2, 128)?.unwrap();
        assert_eq!(pos, 1);
        assert_eq!(restarted.restore(&[23, 11], 2, 128)?.unwrap().0, 1);
        assert!(restarted.restore(&[22], 1, 64)?.is_none());
        let key = cache.keys(&[22], 1, 128)?[0].1;
        fs::write(cache.file(&key).unwrap(), b"damaged")?;
        assert!(cache.restore(&[22], 1, 128)?.is_none());
        assert!(restarted.restore(&[22], 1, 128)?.is_some());
        // A partially upgraded writer could produce a checksummed but incompatible snapshot.
        cache.save(key, vec![0; 24], false);
        assert!(cache.restore(&[22], 1, 128)?.is_none());
        assert!(!cache.file(&key).unwrap().exists());
        cache.store(&[22], 1, 128, &state, false)?;
        assert!(cache.restore(&[22], 1, 128)?.is_some());
        restarted.ram_limit = data.len();
        restarted.store(&[24], 1, 128, &state, false)?;
        assert_eq!(restarted.ram_bytes, data.len());
        assert!(restarted.ram.iter().all(|(k, _)| k != &key));
        assert_eq!(held.encode_snapshot(pos)?, data);
        restarted.disk_limit = 0;
        restarted.trim_disk(0)?;
        assert!(fs::read_dir(d.path())?.all(|entry| entry.unwrap().file_name() == ".cache.lock"));
        Ok(())
    }
    #[test]
    fn concurrent_writers_and_lru_keep_the_disk_budget() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let payload = |v: u8| {
            let mut b = vec![v; 16 + 3 * 512 * 2];
            b[..8].copy_from_slice(b"DS4DRF01");
            b[8..16].copy_from_slice(&1u64.to_le_bytes());
            b
        };
        let budget = 2 * (HEADER + payload(0).len());
        let make = || PrefixCache {
            identity: Sha256::new(),
            draft_identity: [1; 32],
            chunk: 1,
            capacity: 10,
            ram_limit: 0,
            disk_limit: budget,
            dir: Some(dir.path().into()),
            ram: VecDeque::new(),
            ram_bytes: 0,
        };
        let mut cache = make();
        cache.write_draft(&[1], 1, 128, &payload(1), false)?;
        cache.write_draft(&[2], 1, 128, &payload(2), false)?;
        let old = cache.file(&cache.draft_key(&[1], 1, 128)?).unwrap();
        File::open(&old)?.set_modified(SystemTime::UNIX_EPOCH)?;
        cache.write_draft(&[3], 1, 128, &payload(3), false)?;
        assert!(!old.exists());
        assert!(cache.read_draft(&[2], 1, 128)?.is_some());
        let barrier = std::sync::Barrier::new(5);
        std::thread::scope(|scope| {
            for token in 4..9 {
                let mut c = make();
                let b = &barrier;
                scope.spawn(move || {
                    b.wait();
                    c.write_draft(&[token], 1, 128, &payload(4), false).unwrap();
                });
            }
        });
        let size: u64 = fs::read_dir(dir.path())?
            .map(|e| e.unwrap().metadata().unwrap().len())
            .sum();
        assert_eq!(size, budget as u64);
        let mut distinct = make();
        distinct.draft_identity = [2; 32];
        for token in 4..9 {
            assert!(distinct.read_draft(&[token], 1, 128)?.is_none());
        }
        Ok(())
    }
}
