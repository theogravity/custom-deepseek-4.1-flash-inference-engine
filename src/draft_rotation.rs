//! Live DSpark images contain exactly the initialized chronological KV rows.
//! During sparse prefill the observed interval may end before the target's
//! committed position. The remaining physical slots are scratch.
use super::*;

const MAGIC: &[u8; 8] = b"DS4DRS01";
const HEADER: usize = 33;
const WORDS: usize = DIM / 2;

fn check_range(pos: usize, start: usize, end: usize) -> Result<()> {
    ensure!(
        start <= end && end <= pos && pos <= i32::MAX as usize && end - start <= WINDOW,
        "invalid live DSpark interval"
    );
    Ok(())
}

// Physical ranges covering a chronological interval of at most one window.
fn ring_parts(start: usize, end: usize) -> [(usize, usize); 2] {
    let slot = start % WINDOW;
    let first = (end - start).min(WINDOW - slot);
    [
        (slot * WORDS, first * WORDS),
        (0, (end - start - first) * WORDS),
    ]
}

struct Image<'a> {
    start: usize,
    end: usize,
    trace: Option<Vec<f32>>,
    stages: &'a [u8],
}

fn decode_image(bytes: &[u8], pos: usize) -> Result<Image<'_>> {
    ensure!(
        bytes.len() >= HEADER && &bytes[..8] == MAGIC,
        "invalid live DSpark header"
    );
    let position = |at| usize::try_from(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()));
    ensure!(position(8)? == pos, "live DSpark target position mismatch");
    let (start, end) = (position(16)?, position(24)?);
    check_range(pos, start, end)?;
    let count = usize::from(bytes[32]);
    let stage_offset = HEADER + count * 4;
    ensure!(
        count <= 16 && bytes.len() == stage_offset + (end - start) * WORDS * 4 * STAGES,
        "invalid live DSpark payload size"
    );
    let trace = (count > 0).then(|| {
        bytes[HEADER..stage_offset]
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect()
    });
    Ok(Image {
        start,
        end,
        trace,
        stages: &bytes[stage_offset..],
    })
}

impl DraftAttention {
    pub(crate) fn release_device(self) {
        for ring in self.rings.into_iter().flatten() {
            ring.release();
        }
    }

    pub(crate) fn encode_rotation(
        &self,
        rt: &Runtime,
        pos: usize,
        trace: Option<&[f32]>,
    ) -> Result<Vec<u8>> {
        let (start, end) = (self.start, self.end);
        check_range(pos, start, end)?;
        ensure!(
            trace.is_none_or(|t| (1..=16).contains(&t.len())),
            "invalid DSpark confidence trace"
        );
        let trace = trace.unwrap_or_default();
        let mut bytes =
            Vec::with_capacity(HEADER + trace.len() * 4 + (end - start) * WORDS * 4 * STAGES);
        bytes.extend_from_slice(MAGIC);
        for n in [pos, start, end] {
            bytes.extend_from_slice(&(n as u64).to_le_bytes());
        }
        bytes.push(trace.len() as u8);
        for v in trace {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if start < end {
            for ring in &self.rings {
                let ring = ring.as_ref().context("missing live DSpark ring")?;
                ensure!(ring.len() == WINDOW * WORDS, "invalid live DSpark ring");
                for (offset, len) in ring_parts(start, end) {
                    for word in rt.to_host_range(ring, offset, len)? {
                        bytes.extend_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
        Ok(bytes)
    }

    pub(crate) fn decode_rotation(
        rt: &mut Runtime,
        pos: usize,
        bytes: &[u8],
    ) -> Result<(Self, Option<Vec<f32>>)> {
        let image = decode_image(bytes, pos)?;
        let mut state = Self {
            start: image.start,
            end: image.end,
            ..Self::default()
        };
        if state.start < state.end {
            rt.layer(39)?;
            let parts = ring_parts(state.start, state.end);
            let stride = (state.end - state.start) * WORDS * 4;
            for (stage, payload) in image.stages.chunks_exact(stride).enumerate() {
                let mut ring = vec![0.; WINDOW * WORDS];
                let mut words = payload
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes);
                for (offset, len) in parts {
                    for slot in &mut ring[offset..offset + len] {
                        *slot = words.next().unwrap();
                    }
                }
                match rt.to_device(&ring) {
                    Ok(device) => state.rings[stage] = Some(device),
                    Err(err) => {
                        state.release_device();
                        return Err(err);
                    }
                }
            }
        }
        Ok((state, image.trace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pos: usize, start: usize, end: usize, trace: &[f32]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        for n in [pos, start, end] {
            bytes.extend_from_slice(&(n as u64).to_le_bytes());
        }
        bytes.push(trace.len() as u8);
        for v in trace {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for stage in 0..STAGES {
            for t in start..end {
                for col in 0..WORDS {
                    bytes.extend_from_slice(&pattern(stage, t, col).to_le_bytes());
                }
            }
        }
        bytes
    }

    fn pattern(stage: usize, t: usize, col: usize) -> u32 {
        (t as u32).wrapping_mul(0xcf340051) ^ ((stage as u32) << 26) ^ col as u32
    }

    #[test]
    fn live_draft_header_bounds_and_wrap_ranges() {
        let bytes = image(251, 120, 137, &[0.25, 0.75]);
        let decoded = decode_image(&bytes, 251).unwrap();
        assert_eq!((decoded.start, decoded.end), (120, 137));
        assert_eq!(decoded.trace.as_deref(), Some(&[0.25, 0.75][..]));
        assert_eq!(
            ring_parts(120, 137),
            [(120 * WORDS, 8 * WORDS), (0, 9 * WORDS)]
        );
        assert!(decode_image(&bytes, 252).is_err());
        for n in [0, 8, 32, bytes.len() - 1] {
            assert!(decode_image(&bytes[..n], 251).is_err());
        }
        for (at, value) in [(8, u64::MAX), (16, 9), (16, 138), (24, 252)] {
            let mut damaged = bytes.clone();
            damaged[at..at + 8].copy_from_slice(&value.to_le_bytes());
            assert!(decode_image(&damaged, 251).is_err());
        }
        let mut damaged = bytes.clone();
        damaged[32] = 17;
        assert!(decode_image(&damaged, 251).is_err());
        for (pos, start, end) in [(0, 0, 0), (70000, 0, 0), (591, 273, 273)] {
            assert!(decode_image(&image(pos, start, end, &[]), pos)
                .unwrap()
                .stages
                .is_empty());
        }
    }

    #[test]
    #[ignore = "requires DS4_TEST_PACK and both inference GPUs"]
    fn cuda_live_draft_rotation_keeps_only_observed_rows_and_their_physical_slots() -> Result<()> {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES")?.split(',').count() == 2,
            "select both inference GPUs"
        );
        let mut rt = Runtime::open(
            std::path::Path::new(&std::env::var("DS4_TEST_PACK")?),
            [0, 1],
        )?;
        let trace = [0.25, 0.5, 1.];
        for (pos, start, end) in [
            (0, 0, 0),
            (70000, 0, 0),
            (6, 0, 6),
            (1500, 121, 139),
            (300, 172, 300),
            (328, 172, 300),
        ] {
            let bytes = image(pos, start, end, &trace);
            rt.layer(0)?;
            let (state, scores) = DraftAttention::decode_rotation(&mut rt, pos, &bytes)?;
            assert_eq!(scores.as_deref(), Some(&trace[..]));
            assert_eq!(
                state.ready(pos),
                pos > 0 && end == pos && start == pos.saturating_sub(WINDOW)
            );
            for (stage, ring) in state.rings.iter().enumerate() {
                if start == end {
                    assert!(ring.is_none());
                    continue;
                }
                let physical = rt.to_host(ring.as_ref().unwrap())?;
                for t in start..end {
                    for col in 0..WORDS {
                        assert_eq!(
                            physical[(t % WINDOW) * WORDS + col].to_bits(),
                            pattern(stage, t, col)
                        );
                    }
                }
            }
            rt.layer(0)?;
            assert_eq!(state.encode_rotation(&rt, pos, scores.as_deref())?, bytes);
            state.release_device();
        }
        Ok(())
    }
}
