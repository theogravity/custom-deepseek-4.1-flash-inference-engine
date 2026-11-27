//! A worker shares at most one reclaimable peer cache per GPU. Sessions retain
//! only an identity for their append-only key history. CUDA owns the mirrors.
use super::{cuda_ok, INDEX_BYTES};
use crate::runtime::{Pool, Runtime};
use anyhow::Result;
use std::{
    ffi::c_void,
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};
const UNAVAILABLE: i32 = -1099;
unsafe extern "C" {
    fn cudaSetDevice(dev: i32) -> i32;
    fn ds4_index_peer_new(owner: i32, peer: i32, out: *mut *mut c_void) -> i32;
    fn ds4_index_peer_bytes(cache: *mut c_void) -> usize;
    fn ds4_index_peer_growth(
        cache: *mut c_void,
        id: u64,
        keys: usize,
        work: usize,
        rows: i32,
    ) -> usize;
    fn ds4_index_peer_begin(
        cache: *mut c_void,
        id: u64,
        keys: *const c_void,
        bytes: usize,
        work: usize,
        rows: i32,
    ) -> i32;
    fn ds4_index_peer_tile(
        cache: *mut c_void,
        q: *const f32,
        w: *const f32,
        width: i32,
        pos: i32,
        ratio: i32,
        rows: i32,
        picks: *mut i32,
        mode: i32,
    ) -> i32;
    fn ds4_index_peer_finish(cache: *mut c_void) -> i32;
    fn ds4_index_peer_free(cache: *mut c_void);
    #[cfg(test)]
    fn ds4_index_peer_stats(cache: *mut c_void, bytes: *mut u64, tiles: *mut u64);
}

pub(crate) struct Cache(NonNull<c_void>);
// Only the inference worker submits work. The shared transient pool serializes
// checkout and eviction, and a checked-out cache is outside the pool.
unsafe impl Send for Cache {}
impl Cache {
    pub(crate) fn bytes(&self) -> usize {
        unsafe { ds4_index_peer_bytes(self.0.as_ptr()) }
    }
    #[cfg(test)]
    pub(super) fn stats(&self) -> (u64, u64) {
        let (mut bytes, mut tiles) = (0, 0);
        unsafe { ds4_index_peer_stats(self.0.as_ptr(), &mut bytes, &mut tiles) };
        (bytes, tiles)
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        unsafe { ds4_index_peer_free(self.0.as_ptr()) };
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static UNAVAILABLE_AFTER:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
}

pub(super) struct Scan {
    pool: Arc<Mutex<Pool>>,
    cache: Option<Cache>,
    active: bool,
}
impl Scan {
    fn raw(&self) -> *mut c_void {
        self.cache.as_ref().unwrap().0.as_ptr()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        rt: &mut Runtime,
        id: u64,
        keys: *const c_void,
        positions: usize,
        work: usize,
        rows: usize,
    ) -> Result<Option<Self>> {
        let peer = rt.peer_device();
        let pool = rt.index_pool(peer);
        let cached = pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .peer_index
            .take();
        let cache = if let Some(cache) = cached {
            cache
        } else {
            let mut raw = ptr::null_mut();
            let rc = unsafe { ds4_index_peer_new(rt.device, peer, &mut raw) };
            if rc == UNAVAILABLE {
                return Ok(None);
            }
            cuda_ok(rc, "opening tensor-parallel index")?;
            Cache(NonNull::new(raw).expect("successful peer cache allocation"))
        };
        let mut scan = Self {
            pool,
            cache: Some(cache),
            active: false,
        };
        let key_bytes = positions * INDEX_BYTES;
        let rows = rows.try_into()?;
        let growth = unsafe { ds4_index_peer_growth(scan.raw(), id, key_bytes, work, rows) };
        if growth == usize::MAX {
            return Ok(None);
        }
        let result: Result<i32> = (|| {
            if growth > 0 {
                rt.moe_cache.reserve(peer, growth)?;
            }
            let rc = unsafe { ds4_index_peer_begin(scan.raw(), id, keys, key_bytes, work, rows) };
            Ok(rc)
        })();
        scan.active = matches!(result, Ok(0));
        cuda_ok(
            unsafe { cudaSetDevice(rt.device) },
            "restoring primary GPU after peer allocation",
        )?;
        let rc: i32 = result?;
        if rc == UNAVAILABLE {
            scan.cache.take();
            return Ok(None);
        }
        cuda_ok(rc, "starting tensor-parallel prompt index")?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            eprintln!(
                "TP2 prompt index active on owner {} and peer {}",
                rt.device, peer
            )
        });
        Ok(Some(scan))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn tile(
        &self,
        q: *const f32,
        w: *const f32,
        width: usize,
        pos: usize,
        ratio: usize,
        rows: usize,
        picks: *mut i32,
        mode: i32,
    ) -> Result<bool> {
        #[cfg(test)]
        if UNAVAILABLE_AFTER.with(|left| match left.get() {
            Some(0) => {
                left.set(None);
                true
            }
            Some(n) => {
                left.set(Some(n - 1));
                false
            }
            None => false,
        }) {
            return Ok(false);
        }
        let rc = unsafe {
            ds4_index_peer_tile(
                self.raw(),
                q,
                w,
                width as i32,
                pos as i32,
                ratio as i32,
                rows as i32,
                picks,
                mode,
            )
        };
        if rc == UNAVAILABLE {
            return Ok(false);
        }
        cuda_ok(rc, "scanning a prompt index tile on the peer")?;
        Ok(true)
    }
    pub(super) fn abandon(mut self) -> Result<()> {
        let rc = unsafe { ds4_index_peer_finish(self.raw()) };
        self.active = false;
        self.cache.take();
        cuda_ok(rc, "fencing peer access before owner fallback")
    }
    pub(super) fn finish(mut self) -> Result<()> {
        let rc = unsafe { ds4_index_peer_finish(self.raw()) };
        self.active = false;
        if rc != 0 {
            self.cache.take();
        }
        cuda_ok(rc, "joining tensor-parallel prompt index")
    }
}
impl Drop for Scan {
    fn drop(&mut self) {
        if self.active && unsafe { ds4_index_peer_finish(self.raw()) } != 0 {
            self.cache.take();
        }
        if let Some(cache) = self.cache.take() {
            self.pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .peer_index = Some(cache);
        }
    }
}
