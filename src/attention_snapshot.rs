//! Portable attention payload for a cache entry. The fixed layout contains
//! only chronological window rows, completed compressed rows and pending pairs.
//! The enclosing prefix file binds this payload to token IDs and the model.
use super::*;

const MAGIC: &[u8; 8] = b"DS4ATT01";
const OWNERS: [usize; 4] = [2, 8, 14, 20];

fn source_lengths(owner: usize, pos: usize) -> (usize, usize, usize) {
    let ratio = if owner == 20 { 1 } else { 2 };
    let rows = pos / ratio;
    (rows * KV_BYTES, rows * INDEX_BYTES, (pos % ratio) * DIM)
}

fn payload_size(pos: usize) -> Result<usize> {
    ensure!(pos <= MAX_POS, "snapshot position exceeds model limit");
    let mut bytes = MAGIC.len() + 8 + 40 * pos.min(WINDOW - 1) * DIM * 2;
    for owner in OWNERS {
        let (kv, keys, pending) = source_lengths(owner, pos);
        bytes += kv + keys + 2 * pending * 4;
    }
    Ok(bytes)
}

impl AttentionState {
    /// Observe initialized host payloads that still need their first GPU
    /// installation. Pending unpaired compressor rows remain on the host.
    pub(crate) fn profile_restore(&mut self, probe: Probe) {
        let (mut count, mut bytes) = (0, 0);
        for w in self.windows.values_mut().filter(|w| w.ring.is_none()) {
            w.restored = RestoredPart::attach(
                &probe,
                std::mem::size_of_val(w.rows.as_slice()),
                &mut count,
                &mut bytes,
            );
        }
        for source in self.sources.values_mut() {
            for r in [&mut source.keys, &mut source.values] {
                if r.writer.is_none() && r.devices.is_empty() {
                    r.restored =
                        RestoredPart::attach(&probe, r.packed.len(), &mut count, &mut bytes);
                }
            }
        }
        probe.expect(count, bytes);
    }

    pub(crate) fn snapshot_bytes(pos: usize) -> Result<usize> {
        payload_size(pos)
    }

    pub fn encode_snapshot(&self, pos: usize) -> Result<Vec<u8>> {
        self.encode_snapshot_with_tail(pos, 0)
    }

    /// Allow the live-session container to append its metadata and DSpark
    /// payload without copying a full compressed target cache a second time.
    pub(crate) fn encode_snapshot_with_tail(&self, pos: usize, extra: usize) -> Result<Vec<u8>> {
        let size = payload_size(pos)?
            .checked_add(extra)
            .context("snapshot size overflow")?;
        ensure!(self.verify.is_none(), "cannot persist active verification");
        let empty = pos == 0 && self.windows.is_empty() && self.sources.is_empty();
        ensure!(
            empty || (self.windows.len() == 40 && self.sources.len() == 4),
            "incomplete attention snapshot"
        );
        let mut out = Vec::new();
        out.try_reserve_exact(size)?;
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(pos as u64).to_le_bytes());
        if empty {
            return Ok(out);
        }

        let window_len = pos.min(WINDOW - 1) * DIM;
        for layer in 0..40 {
            let w = self
                .windows
                .get(&layer)
                .context("missing snapshot window")?;
            ensure!(
                w.next_pos == pos,
                "snapshot position mismatch at layer {layer}"
            );
            // A live ring owns the newest rows. A restored snapshot can also
            // be serialized directly, before its first upload to a device.
            let host;
            let rows = if w.ring.is_some() {
                host = w.host_snapshot()?;
                &host.rows
            } else {
                &w.rows
            };
            ensure!(
                rows.len() == window_len,
                "invalid snapshot window at layer {layer}"
            );
            for x in rows {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for owner in OWNERS {
            let src = self
                .sources
                .get(&owner)
                .context("missing snapshot source")?;
            let (kv, keys, pending) = source_lengths(owner, pos);
            ensure!(
                src.values.len() == kv
                    && src.keys.len() == keys
                    && src.pending_kv.len() == pending
                    && src.pending_gate.len() == pending,
                "invalid compressed snapshot shape at layer {owner}"
            );
            src.values.append_snapshot(&mut out)?;
            src.keys.append_snapshot(&mut out)?;
            for x in src.pending_kv.iter().chain(&src.pending_gate) {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Prefix reuse adopts the new request's window. Live rotation resumes
    /// the original request's prefill geometry and total position budget.
    pub(crate) fn restore_rotation(
        bytes: &[u8],
        pos: usize,
        prompt_len: usize,
        capacity_limit: usize,
    ) -> Result<Self> {
        ensure!(
            prompt_len > 0
                && prompt_len <= capacity_limit
                && pos <= capacity_limit
                && capacity_limit <= MAX_POS,
            "invalid suspended target geometry"
        );
        let mut cache = Self::decode_snapshot(bytes, pos)?;
        cache.begin_prefill_with_limit(prompt_len, capacity_limit);
        Ok(cache)
    }

    pub fn decode_snapshot(bytes: &[u8], pos: usize) -> Result<Self> {
        Self::parse_snapshot(bytes, pos, |part| PackedRows::Owned(part.to_vec()))
    }

    pub(crate) fn decode_prefix(image: &Bytes, pos: usize) -> Result<Self> {
        Self::parse_snapshot(image, pos, |part| PackedRows::Prefix(image.slice_ref(part)))
    }

    fn parse_snapshot(
        bytes: &[u8],
        pos: usize,
        packed: impl Fn(&[u8]) -> PackedRows,
    ) -> Result<Self> {
        // Derive every slice length from the caller's validated token count.
        // Check the entire file before allocating buffers or slicing its body.
        ensure!(
            bytes.len() == payload_size(pos)?,
            "invalid attention snapshot size"
        );
        ensure!(&bytes[..8] == MAGIC, "unknown attention snapshot version");
        ensure!(
            bytes[8..16] == (pos as u64).to_le_bytes(),
            "attention snapshot position mismatch"
        );
        let mut state = Self::default();
        if pos == 0 {
            return Ok(state);
        }

        let mut body = &bytes[16..];
        let mut take = |n| {
            let (part, rest) = body.split_at(n);
            body = rest;
            part
        };
        let window_bytes = pos.min(WINDOW - 1) * DIM * 2;
        for layer in 0..40 {
            let rows = take(window_bytes)
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(u16::from_le_bytes)
                .collect();
            state.windows.insert(
                layer,
                Window {
                    restored: None,
                    rows,
                    next_pos: pos,
                    ring: None,
                },
            );
        }
        for owner in OWNERS {
            let (kv, keys, pending) = source_lengths(owner, pos);
            let values = Rows {
                packed: packed(take(kv)),
                ..Default::default()
            };
            let keys = Rows {
                packed: packed(take(keys)),
                ..Default::default()
            };
            let mut floats = || {
                take(pending * 4)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect()
            };
            state.sources.insert(
                owner,
                Source {
                    values,
                    keys,
                    pending_kv: floats(),
                    pending_gate: floats(),
                },
            );
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(pos: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0; payload_size(pos)?];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..16].copy_from_slice(&(pos as u64).to_le_bytes());
        for (i, b) in bytes[16..].iter_mut().enumerate() {
            *b = (i * 13 % 251) as u8;
        }
        Ok(bytes)
    }

    #[test]
    fn prefix_rows_share_the_image_until_mutated_or_released() -> Result<()> {
        let pos = 257; // Covers a wrapped window and the pending ratio-two rows.
        let bytes = payload(pos)?;
        let image = Bytes::from(bytes);
        let mut restored = AttentionState::decode_prefix(&image, pos)?;
        let other = AttentionState::decode_prefix(&image, pos)?;
        assert_eq!(restored.encode_snapshot(pos)?, &image[..]);
        let mut offset = 16 + 40 * (WINDOW - 1) * DIM * 2;
        for owner in OWNERS {
            let (kv, keys, pending) = source_lengths(owner, pos);
            let source = restored.sources.get_mut(&owner).unwrap();
            for (rows, len) in [(&mut source.values, kv), (&mut source.keys, keys)] {
                assert_eq!(rows.packed.as_ptr(), image[offset..].as_ptr());
                rows.truncate(len - 1);
                assert_eq!(rows.packed.as_ptr(), image[offset..].as_ptr());
                let owned = rows.packed.for_append();
                owned.push(0);
                assert_ne!(owned.as_ptr(), image[offset..].as_ptr());
                assert_eq!(&owned[..len - 1], &image[offset..offset + len - 1]);
                offset += len;
            }
            offset += 2 * pending * 4;
        }
        assert_eq!(offset, image.len());
        assert_eq!(other.encode_snapshot(pos)?, &image[..]);
        drop(other);
        assert!(image.is_unique());
        Ok(())
    }

    #[test]
    fn restore_profile_counts_only_initialized_gpu_payloads() -> Result<()> {
        for (pos, parts) in [(0, 0), (1, 42), (2, 48), (257, 48)] {
            let bytes = payload(pos)?;
            let mut state = AttentionState::decode_snapshot(&bytes, pos)?;
            let probe = Probe::for_test();
            state.profile_restore(probe.clone());
            let transfer = 40 * pos.min(WINDOW - 1) * DIM * 2
                + (3 * (pos / 2) + pos) * (KV_BYTES + INDEX_BYTES);
            assert_eq!(probe.progress(), (parts, 0, transfer, 0, parts == 0));
            assert_eq!(state.encode_snapshot(pos)?, bytes);
            drop(state);
            assert_eq!(probe.progress(), (parts, 0, transfer, 0, parts == 0));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires both inference GPUs"]
    fn cuda_restore_profile_completes_once_after_the_last_successful_install() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "mask to both inference GPUs"
        );
        let pos = 257;
        let mut state = AttentionState::decode_snapshot(&payload(pos)?, pos)?;
        // A previously installed window and host-backed key cache are excluded.
        state.windows.get_mut(&0).unwrap().on(0, None)?;
        state.sources.get_mut(&2).unwrap().keys.on(0, None)?;
        let prefix_bytes = 39 * (WINDOW - 1) * DIM * 2
            + (3 * (pos / 2) + pos) * (KV_BYTES + INDEX_BYTES)
            - pos / 2 * INDEX_BYTES;
        let probe = Probe::for_test();
        state.profile_restore(probe.clone());
        assert_eq!(probe.progress(), (46, 0, prefix_bytes, 0, false));
        let w = state.windows.get_mut(&1).unwrap();
        let missing = w.rows.pop().unwrap();
        assert!(w.on(0, None).is_err());
        assert_eq!(probe.progress(), (46, 0, prefix_bytes, 0, false));
        w.rows.push(missing);
        for (layer, w) in &mut state.windows {
            w.on(i32::from(*layer >= 20), None)?;
            w.on(i32::from(*layer >= 20), None)?;
        }
        for owner in OWNERS {
            let s = state.sources.get_mut(&owner).unwrap();
            let dev = i32::from(owner >= 20);
            // The CPU compressor can append fresh host rows before uploading
            // its saved prefix. Only the original bytes belong to restoration.
            let len = s.keys.packed.len() + INDEX_BYTES;
            s.keys.packed.for_append().resize(len, 0);
            s.keys.on(dev, None)?;
            if owner == 20 {
                assert_eq!(
                    probe.progress(),
                    (46, 45, prefix_bytes, prefix_bytes - pos * KV_BYTES, false)
                );
            }
            s.values.on(dev, None)?;
            s.values.on(dev, None)?;
        }
        assert_eq!(probe.progress(), (46, 46, prefix_bytes, prefix_bytes, true));
        let window_bytes = (WINDOW - 1) * DIM * 2;
        let first = 19 * window_bytes + pos / 2 * (3 * KV_BYTES + 2 * INDEX_BYTES);
        let second = 20 * window_bytes + pos * (KV_BYTES + INDEX_BYTES);
        assert_eq!(probe.card_bytes(0), (first, first + 2 * INDEX_BYTES));
        assert_eq!(probe.card_bytes(1), (second, second + INDEX_BYTES));
        Ok(())
    }

    #[test]
    fn rotation_preserves_prefill_span_capacity_and_committed_pairs() -> Result<()> {
        for (pos, prompt, limit) in [
            (0, 63, 66),
            (3, 63, 66),
            (63, 63, 66),
            (65, 63, 66),
            (67, 200, 400),
        ] {
            let bytes = payload(pos)?;
            let mut state = AttentionState::restore_rotation(&bytes, pos, prompt, limit)?;
            assert_eq!((state.prompt_len, state.capacity_limit), (prompt, limit));
            assert_eq!(
                state.window_layout(pos, 1),
                if pos < prompt {
                    (prompt.min(WINDOW), false)
                } else {
                    (WINDOW, true)
                }
            );
            let saved = state.encode_snapshot_with_tail(pos, 911)?;
            assert_eq!(saved, bytes);
            assert!(saved.capacity() >= bytes.len() + 911);
            assert!(state.encode_snapshot_with_tail(pos, usize::MAX).is_err());
            state.begin_verify(pos, 1)?;
            assert!(state.encode_snapshot(pos).is_err());
        }
        let bytes = payload(3)?;
        for (prompt, limit) in [(4, 3), (0, 4), (1, 2), (3, MAX_POS + 1)] {
            assert!(AttentionState::restore_rotation(&bytes, 3, prompt, limit).is_err());
        }
        Ok(())
    }

    #[test]
    #[ignore = "set DS4_TEST_DEVICE to a visible inference GPU"]
    fn cuda_prefix_then_live_rotation_round_trip_wrapped_rings_and_compressed_rows() -> Result<()> {
        let dev: i32 = std::env::var("DS4_TEST_DEVICE")?.parse()?;
        let (pos, prompt, limit) = (275, 521, 612);
        let bytes = payload(pos)?;
        let image = Bytes::copy_from_slice(&bytes);
        let mut state = AttentionState::decode_prefix(&image, pos)?;
        assert!(!image.is_unique());
        for pass in 0..3 {
            for w in state.windows.values_mut() {
                w.on(dev, None)?;
            }
            for s in state.sources.values_mut() {
                for rows in [&mut s.keys, &mut s.values] {
                    rows.on(dev, None)?;
                    rows.finish_append(dev, 0);
                    assert!(rows.packed.is_empty());
                }
            }
            assert!(image.is_unique());
            let saved = state.encode_snapshot(pos)?;
            assert_eq!(saved, bytes);
            if pass < 2 {
                state = AttentionState::restore_rotation(&saved, pos, prompt, limit)?;
            }
        }
        Ok(())
    }

    #[test]
    fn snapshot_round_trip_checks_position_sizes_and_order() -> Result<()> {
        let pos = 5;
        let mut bytes = payload(pos)?;
        let mut restored = AttentionState::decode_snapshot(&bytes, pos)?;
        assert!(restored.freqs.is_empty() && restored.picks.is_empty());
        assert_eq!(restored.windows[&0].rows[0], u16::from_le_bytes([0, 13]));
        assert!(restored
            .windows
            .values()
            .all(|w| w.ring.is_none() && w.next_pos == pos));
        assert!(restored
            .sources
            .values()
            .all(|s| s.keys.devices.is_empty() && s.values.devices.is_empty()));
        assert_eq!(restored.encode_snapshot(pos)?, bytes);
        restored.sources.get_mut(&2).unwrap().pending_gate.clear();
        assert!(restored.encode_snapshot(pos).is_err());
        assert!(AttentionState::decode_snapshot(&bytes[1..], pos).is_err());
        assert!(AttentionState::decode_snapshot(&bytes, MAX_POS + 1).is_err());
        bytes[8] = 0;
        assert!(AttentionState::decode_snapshot(&bytes, pos).is_err());
        Ok(())
    }
}
