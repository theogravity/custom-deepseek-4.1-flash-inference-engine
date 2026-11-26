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
    pub fn encode_snapshot(&self, pos: usize) -> Result<Vec<u8>> {
        let size = payload_size(pos)?;
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

    pub fn decode_snapshot(bytes: &[u8], pos: usize) -> Result<Self> {
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
                    rows,
                    next_pos: pos,
                    ring: None,
                },
            );
        }
        for owner in OWNERS {
            let (kv, keys, pending) = source_lengths(owner, pos);
            let values = Rows {
                packed: take(kv).to_vec(),
                ..Default::default()
            };
            let keys = Rows {
                packed: take(keys).to_vec(),
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

    #[test]
    fn snapshot_round_trip_checks_position_sizes_and_order() -> Result<()> {
        let pos = 5;
        let mut bytes = vec![0; payload_size(pos)?];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..16].copy_from_slice(&(pos as u64).to_le_bytes());
        for (i, b) in bytes[16..].iter_mut().enumerate() {
            *b = (i * 13 % 251) as u8;
        }
        let mut restored = AttentionState::decode_snapshot(&bytes, pos)?;
        assert!(restored.scratch.is_empty() && restored.picks.is_empty());
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
