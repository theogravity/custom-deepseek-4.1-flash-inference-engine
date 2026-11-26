//! Backing for live, interrupted requests. Unlike reusable prefix snapshots, a
//! live image has one owner and must survive until that request resumes or exits.
use crate::restore_timing::Tier;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    fs::File,
    io::Write,
    ops::Deref,
    os::unix::fs::{DirBuilderExt, FileExt},
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Debug, Copy, Clone, Default, Serialize)]
pub(crate) struct SwapStats {
    pub(crate) ram_limit_bytes: usize,
    pub(crate) disk_limit_bytes: usize,
    pub(crate) ram_bytes: usize,
    pub(crate) disk_bytes: usize,
    pub(crate) peak_ram_bytes: usize,
    pub(crate) peak_disk_bytes: usize,
    pub(crate) ram_writes: u64,
    pub(crate) disk_writes: u64,
    pub(crate) ram_reads: u64,
    pub(crate) disk_reads: u64,
    pub(crate) write_failures: u64,
    pub(crate) read_failures: u64,
}

#[derive(Clone)]
pub(crate) struct SwapStore(Arc<Backing>);
struct Backing {
    directory: Option<PathBuf>,
    stats: Mutex<SwapStats>,
}
struct Reservation {
    store: SwapStore,
    tier: Tier,
    len: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.store.0.stats.lock().unwrap();
        match self.tier {
            Tier::Ram => s.ram_bytes -= self.len,
            Tier::Disk => s.disk_bytes -= self.len,
        }
    }
}
struct RamImage {
    data: Box<[u8]>,
    charge: Reservation,
}
struct DiskImage {
    file: File,
    hash: [u8; 32],
    charge: Reservation,
}
enum Location {
    Ram(Arc<RamImage>),
    Disk(DiskImage),
}
pub(crate) struct SwapHandle(Location);
enum ReadData {
    Shared(Arc<RamImage>),
    Disk(Vec<u8>),
}
pub(crate) struct SwapRead(ReadData);
impl Deref for SwapRead {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.0 {
            ReadData::Shared(image) => &image.data,
            ReadData::Disk(data) => data,
        }
    }
}
impl AsRef<[u8]> for SwapRead {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

fn budget(name: &str, default_mib: usize) -> Result<usize> {
    let mib = match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .with_context(|| format!("invalid {name}"))?,
        Err(env::VarError::NotPresent) => default_mib,
        Err(e) => return Err(e.into()),
    };
    mib.checked_mul(1 << 20)
        .with_context(|| format!("{name} overflows byte count"))
}
impl SwapStore {
    pub(crate) fn configured() -> Result<Self> {
        let ram = budget("DS4_SWAP_RAM_MIB", 48 * 1024)?;
        let disk = budget("DS4_SWAP_DISK_MIB", 128 * 1024)?;
        let directory = env::var_os("DS4_SWAP_DIR")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("DS4_PREFIX_DIR")
                    .filter(|p| !p.is_empty())
                    .map(|p| PathBuf::from(p).join("live"))
            })
            .or_else(|| {
                env::var_os("XDG_CACHE_HOME")
                    .filter(|p| !p.is_empty())
                    .map(PathBuf::from)
                    .or_else(|| {
                        env::var_os("HOME")
                            .filter(|p| !p.is_empty())
                            .map(|p| PathBuf::from(p).join(".cache"))
                    })
                    .map(|p| p.join("ds4-flash-rs/live"))
            });
        Self::new(ram, disk, directory)
    }
    pub(crate) fn new(ram: usize, disk: usize, directory: Option<PathBuf>) -> Result<Self> {
        if disk != 0 {
            let path = directory
                .as_deref()
                .context("live disk KV needs a directory; set DS4_SWAP_DIR")?;
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(path)
                .context("creating live KV swap directory")?;
            tempfile::tempfile_in(path).context("checking live KV swap directory is writable")?;
        }
        Ok(Self(Arc::new(Backing {
            directory,
            stats: Mutex::new(SwapStats {
                ram_limit_bytes: ram,
                disk_limit_bytes: disk,
                ..Default::default()
            }),
        })))
    }
    pub(crate) fn stats(&self) -> SwapStats {
        *self.0.stats.lock().unwrap()
    }
    fn reserve(&self, len: usize) -> Result<Reservation> {
        let mut s = self.0.stats.lock().unwrap();
        let tier = if len <= s.ram_limit_bytes - s.ram_bytes {
            s.ram_bytes += len;
            s.peak_ram_bytes = s.peak_ram_bytes.max(s.ram_bytes);
            Tier::Ram
        } else if len <= s.disk_limit_bytes - s.disk_bytes {
            s.disk_bytes += len;
            s.peak_disk_bytes = s.peak_disk_bytes.max(s.disk_bytes);
            Tier::Disk
        } else {
            bail!(
                "live KV backing full: {len} bytes requested, {} RAM and {} disk bytes available",
                s.ram_limit_bytes - s.ram_bytes,
                s.disk_limit_bytes - s.disk_bytes
            )
        };
        Ok(Reservation {
            store: self.clone(),
            tier,
            len,
        })
    }
    #[cfg(test)]
    pub(crate) fn put(&self, bytes: Vec<u8>) -> Result<SwapHandle> {
        self.put_with(bytes.len(), || Ok(bytes))
    }
    pub(crate) fn put_with(
        &self,
        len: usize,
        snapshot: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<SwapHandle> {
        let result = self.write(len, snapshot);
        if result.is_err() {
            self.0.stats.lock().unwrap().write_failures += 1;
        }
        result
    }
    fn write(&self, len: usize, snapshot: impl FnOnce() -> Result<Vec<u8>>) -> Result<SwapHandle> {
        ensure!(len > 0, "empty live KV image");
        // Reject a full store before copying a potentially gigabyte-sized target
        // from the GPUs. Other requests cannot claim the same bytes during I/O.
        let charge = self.reserve(len)?;
        let bytes = snapshot()?;
        ensure!(
            bytes.len() == len,
            "live KV snapshot length differs from its reservation"
        );
        let image = match charge.tier {
            Tier::Ram => {
                let image = Arc::new(RamImage {
                    data: bytes.into_boxed_slice(),
                    charge,
                });
                self.0.stats.lock().unwrap().ram_writes += 1;
                Location::Ram(image)
            }
            Tier::Disk => {
                let dir = self
                    .0
                    .directory
                    .as_ref()
                    .context("missing live KV backing directory")?;
                // An unnamed file lives exactly as long as this request, even if
                // the worker crashes. Publish its handle after writeback succeeds.
                let mut file =
                    tempfile::tempfile_in(dir).context("opening anonymous live KV file")?;
                let hash = Sha256::digest(&bytes).into();
                file.write_all(&bytes).context("writing live KV image")?;
                file.sync_data().context("flushing live KV image")?;
                self.0.stats.lock().unwrap().disk_writes += 1;
                Location::Disk(DiskImage { file, hash, charge })
            }
        };
        Ok(SwapHandle(image))
    }
}
impl SwapHandle {
    pub(crate) fn tier(&self) -> Tier {
        match &self.0 {
            Location::Ram(_) => Tier::Ram,
            Location::Disk(_) => Tier::Disk,
        }
    }
    pub(crate) fn read(&self) -> Result<SwapRead> {
        match &self.0 {
            Location::Ram(image) => {
                image.charge.store.0.stats.lock().unwrap().ram_reads += 1;
                Ok(SwapRead(ReadData::Shared(image.clone())))
            }
            Location::Disk(image) => {
                let result = image.read();
                let mut stats = image.charge.store.0.stats.lock().unwrap();
                if result.is_ok() {
                    stats.disk_reads += 1;
                } else {
                    stats.read_failures += 1;
                }
                result.map(|x| SwapRead(ReadData::Disk(x)))
            }
        }
    }
}
impl DiskImage {
    fn read(&self) -> Result<Vec<u8>> {
        let size = self.charge.len;
        ensure!(
            self.file.metadata()?.len() == size as u64,
            "truncated live disk KV image"
        );
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .context("allocating live KV restore buffer")?;
        bytes.resize(size, 0);
        self.file
            .read_exact_at(&mut bytes, 0)
            .context("reading live KV image")?;
        ensure!(
            Sha256::digest(&bytes)[..] == self.hash,
            "live disk KV checksum mismatch"
        );
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_images_spill_into_private_files_and_hold_capacity_until_release() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = SwapStore::new(8, 12, Some(dir.path().to_owned()))?;
        let ram = store.put(vec![1; 8])?;
        let disk = store.put(vec![2; 12])?;
        assert_eq!(fs::read_dir(dir.path())?.count(), 0);
        assert!(store.put(vec![3]).is_err());
        let lease = ram.read()?;
        drop(ram);
        assert_eq!(lease.as_ref(), [1; 8]);
        assert!(store.put(vec![4]).is_err());
        assert_eq!(disk.read()?.as_ref(), [2; 12]);
        assert_eq!(disk.read()?.as_ref(), [2; 12]);
        drop((lease, disk));
        let s = store.stats();
        assert_eq!((s.ram_bytes, s.disk_bytes), (0, 0));
        assert_eq!((s.peak_ram_bytes, s.peak_disk_bytes), (8, 12));
        assert_eq!(
            (
                s.ram_writes,
                s.disk_writes,
                s.ram_reads,
                s.disk_reads,
                s.write_failures
            ),
            (1, 1, 1, 2, 2)
        );
        assert!(store.put(vec![5; 8]).is_ok());
        Ok(())
    }
    #[test]
    fn full_store_leaves_target_on_device_and_snapshot_errors_return_the_reservation() -> Result<()>
    {
        let store = SwapStore::new(8, 0, None)?;
        let first = store.put(vec![1; 8])?;
        assert!(store
            .put_with(4, || panic!("should not copy the target"))
            .is_err());
        drop(first);
        assert!(store.put_with(4, || bail!("device read failed")).is_err());
        assert!(store.put_with(4, || Ok(vec![2; 3])).is_err());
        assert_eq!(store.stats().ram_bytes, 0);
        assert_eq!(store.stats().write_failures, 3);
        Ok(())
    }

    #[test]
    fn write_failure_releases_budget_and_damaged_images_fail_closed() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("swap");
        let store = SwapStore::new(0, 64, Some(path.clone()))?;
        fs::remove_dir(&path)?;
        assert!(store.put(vec![1; 64]).is_err());
        assert_eq!(store.stats().disk_bytes, 0);
        fs::create_dir(path)?;
        let image = store.put(vec![2; 64])?;
        let Location::Disk(d) = &image.0 else {
            unreachable!()
        };
        d.file.write_all_at(&[3], 1)?;
        assert!(image.read().is_err());
        d.file.set_len(1)?;
        assert!(image.read().is_err());
        assert_eq!(store.stats().read_failures, 2);
        drop(image);
        assert_eq!(store.stats().disk_bytes, 0);
        Ok(())
    }
}
