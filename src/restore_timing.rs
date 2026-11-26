//! Opt-in wall-clock observations of real restore operations. Each component
//! holds a probe only until its first successful installation. Dropping a
//! cancelled or failed request releases incomplete measurements with the state.
//! Install intervals end when the existing CUDA calls return. Restored data is
//! then usable by stream-ordered work. No fence or isolated DMA timer is added.
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Tier {
    Ram,
    Disk,
}

pub(crate) fn timed<T>(enabled: bool, operation: impl FnOnce() -> T) -> (T, Duration) {
    if enabled {
        let start = Instant::now();
        let result = operation();
        (result, start.elapsed())
    } else {
        (operation(), Duration::ZERO)
    }
}

#[derive(Default, Serialize)]
struct Device {
    restored_bytes: usize,
    upload_bytes: usize,
    components: usize,
    install_ms: f64,
}

#[derive(Default, Serialize)]
struct Sample {
    event: &'static str,
    kind: &'static str,
    tier: Option<Tier>,
    position: usize,
    target_image_bytes: usize,
    target_lookup_read_ms: f64,
    target_deserialize_ms: f64,
    draft_included: bool,
    draft_tier: Option<Tier>,
    draft_image_bytes: usize,
    draft_lookup_read_ms: f64,
    draft_restore_ms: f64,
    target_restored_bytes: usize,
    target_upload_bytes: usize,
    target_gpu_install_ms: f64,
    target_gpu_components: usize,
    devices: BTreeMap<i32, Device>,
    wall_to_ready_ms: f64,
}

/// Host phases are private to one admission/resume and need no locking.
pub(crate) struct Host {
    started: Instant,
    record: Sample,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.
}

impl Host {
    pub(crate) fn start(kind: &'static str) -> Option<Self> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        ENABLED
            .get_or_init(|| std::env::var_os("DS4_TIMING_RESTORES").is_some_and(|v| v == "1"))
            .then(|| Self::new(kind))
    }

    fn new(kind: &'static str) -> Self {
        Self {
            started: Instant::now(),
            record: Sample {
                event: "kv_restore",
                kind,
                ..Default::default()
            },
        }
    }

    /// Includes matching-key search and disk integrity validation. For live
    /// requests the target and drafter share one read of the combined image.
    pub(crate) fn backing_read(&mut self, tier: Tier) {
        self.record.tier = Some(tier);
        self.record.target_lookup_read_ms = ms(self.started.elapsed());
    }

    pub(crate) fn decoded(&mut self, target_bytes: usize, duration: Duration) {
        self.record.target_image_bytes = target_bytes;
        self.record.target_deserialize_ms = ms(duration);
    }

    pub(crate) fn draft_read(&mut self, duration: Duration) {
        self.record.draft_lookup_read_ms = ms(duration);
    }

    pub(crate) fn draft_restore(
        &mut self,
        tier: Tier,
        bytes: usize,
        duration: Duration,
        included: bool,
    ) {
        self.record.draft_included = included;
        self.record.draft_tier = Some(tier);
        self.record.draft_image_bytes = bytes;
        self.record.draft_restore_ms = ms(duration);
    }

    pub(crate) fn probe(mut self, position: usize) -> Probe {
        self.record.position = position;
        Probe(Arc::new(Mutex::new(Pending {
            host: self,
            expected: None,
            emitted: false,
        })))
    }
}

struct Pending {
    host: Host,
    expected: Option<(usize, usize)>,
    emitted: bool,
}

#[derive(Clone)]
pub(crate) struct Probe(Arc<Mutex<Pending>>);

impl Probe {
    pub(crate) fn expect(&self, components: usize, bytes: usize) {
        let mut state = self.0.lock().unwrap();
        state.expected = Some((components, bytes));
        state.finish();
    }

    pub(crate) fn installed(
        &self,
        device: i32,
        restored_bytes: usize,
        transferred_bytes: usize,
        wall: Duration,
    ) {
        let mut state = self.0.lock().unwrap();
        let record = &mut state.host.record;
        record.target_restored_bytes += restored_bytes;
        record.target_upload_bytes += transferred_bytes;
        record.target_gpu_components += 1;
        record.target_gpu_install_ms += ms(wall);
        let card = record.devices.entry(device).or_default();
        card.restored_bytes += restored_bytes;
        card.upload_bytes += transferred_bytes;
        card.components += 1;
        card.install_ms += ms(wall);
        state.finish();
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Host::new("live").probe(0)
    }

    #[cfg(test)]
    pub(crate) fn card_bytes(&self, dev: i32) -> (usize, usize) {
        self.0
            .lock()
            .unwrap()
            .host
            .record
            .devices
            .get(&dev)
            .map(|card| (card.restored_bytes, card.upload_bytes))
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn progress(&self) -> (usize, usize, usize, usize, bool) {
        let s = self.0.lock().unwrap();
        let (count, bytes) = s.expected.unwrap_or_default();
        (
            count,
            s.host.record.target_gpu_components,
            bytes,
            s.host.record.target_restored_bytes,
            s.emitted,
        )
    }
}

impl Pending {
    fn finish(&mut self) {
        let record = &mut self.host.record;
        if self.emitted
            || self.expected != Some((record.target_gpu_components, record.target_restored_bytes))
        {
            return;
        }
        record.wall_to_ready_ms = ms(self.host.started.elapsed());
        self.emitted = true;
        eprintln!(
            "{}",
            serde_json::to_string(record).expect("restore timing JSON")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_record_sums_its_own_device_installs() {
        let mut host = Host::new("prefix");
        host.backing_read(Tier::Disk);
        host.decoded(100, Duration::from_millis(3));
        host.draft_read(Duration::from_millis(4));
        host.draft_restore(Tier::Ram, 5, Duration::from_millis(6), true);
        let probe = host.probe(97);
        probe.expect(2, 32);
        probe.clone().installed(0, 7, 10, Duration::from_millis(1));
        assert_eq!(probe.progress(), (2, 1, 32, 7, false));
        probe.installed(1, 25, 25, Duration::from_millis(2));
        assert_eq!(probe.progress(), (2, 2, 32, 32, true));
        assert_eq!(probe.card_bytes(0), (7, 10));
        let s = probe.0.lock().unwrap();
        let v = serde_json::to_value(&s.host.record).unwrap();
        assert_eq!(v["event"], "kv_restore");
        assert_eq!(v["position"], 97);
        assert_eq!(v["tier"], "disk");
        assert_eq!(v["draft_tier"], "ram");
        assert_eq!(v["devices"]["0"]["restored_bytes"], 7);
        assert_eq!(v["devices"]["0"]["upload_bytes"], 10);
        assert_eq!(v["target_upload_bytes"], 35);
        assert_eq!(v["devices"]["1"]["install_ms"], 2.);
        assert_eq!(v["target_gpu_install_ms"], 3.);
    }

    #[test]
    fn no_components_complete_immediately_and_abandoned_probes_release() {
        let empty = Probe::for_test();
        empty.expect(0, 0);
        assert_eq!(empty.progress(), (0, 0, 0, 0, true));
        let interrupted = Probe::for_test();
        interrupted.expect(3, 90);
        interrupted.installed(1, 20, 20, Duration::ZERO);
        let weak = Arc::downgrade(&interrupted.0);
        drop(interrupted);
        assert!(weak.upgrade().is_none());
    }
}
