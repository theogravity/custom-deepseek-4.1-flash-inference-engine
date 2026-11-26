//! Per-request prompt lookup. Only committed tokens and the target's current
//! anchor enter the index. Positions refer to the first token after a four-gram.
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

pub(super) fn configured_width() -> Result<usize> {
    match std::env::var("DS4_PROMPT_LOOKUP").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => return Ok(0),
        Ok("1") => (),
        _ => bail!("DS4_PROMPT_LOOKUP must be 0 or 1"),
    }
    let width = match std::env::var("DS4_LOOKUP_K") {
        Ok(v) => v.parse().context("invalid DS4_LOOKUP_K")?,
        Err(std::env::VarError::NotPresent) => 32,
        Err(e) => return Err(e.into()),
    };
    ensure!(
        (1..=128).contains(&width),
        "DS4_LOOKUP_K must be in 1..=128"
    );
    Ok(width)
}

#[derive(Default)]
pub(super) struct PromptLookup {
    latest: HashMap<[u32; 4], usize>,
    // previous[i] points at the preceding occurrence with the same 4-token
    // suffix as the occurrence ending at 4+i. Zero means no preceding match.
    previous: Vec<u32>,
}

impl PromptLookup {
    pub(super) fn propose(&mut self, ids: &[u32], width: usize) -> Vec<u32> {
        if ids.len() <= 4 || width == 0 {
            return vec![];
        }
        let n = ids.len();
        // Each indexed four-gram has at least one observed continuation.
        for end in self.previous.len() + 4..n {
            let suffix = ids[end - 4..end].try_into().unwrap();
            let prev = self.latest.insert(suffix, end).unwrap_or(0);
            self.previous.push(prev as u32);
        }
        let suffix: &[u32; 4] = ids[n - 4..].try_into().unwrap();
        let mut end = self.latest.get(suffix).copied().unwrap_or(0);
        let mut best = (0, 0);
        let mut best_end = 0;
        // Bound work on common punctuation runs. Favor long matching contexts;
        // break ties by how many subsequent tokens we have already observed.
        for _ in 0..32 {
            if end == 0 {
                break;
            }
            let mut matched = 4;
            while matched < end.min(128) && ids[end - matched - 1] == ids[n - matched - 1] {
                matched += 1;
            }
            let score = (matched, width.min(n - end));
            if score > best {
                best = score;
                best_end = end;
            }
            end = self.previous[end - 4] as usize;
        }
        ids[best_end..best_end + best.1].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_uses_only_observed_tokens_and_prefers_longer_previous_context() {
        let mut lookup = PromptLookup::default();
        let mut history = vec![
            9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 0, 1, 2, 3, 77, 9, 8, 0, 1, 2, 3,
        ];
        assert_eq!(lookup.propose(&history, 4), [4, 5, 6, 7]);
        history.push(4);
        assert_eq!(lookup.propose(&history, 2), [5, 6]);
        history.extend([97, 98, 99]);
        assert!(lookup.propose(&history, 4).is_empty());
        assert!(PromptLookup::default().propose(&[1, 2, 3, 4], 4).is_empty());
        let repetitive = vec![3; 9];
        assert_eq!(PromptLookup::default().propose(&repetitive, 128), [3]);
    }

    #[test]
    fn lookup_can_verify_128_known_continuation_tokens() {
        let mut ids: Vec<u32> = (0..140).collect();
        ids.extend([0, 1, 2, 3]);
        let found = PromptLookup::default().propose(&ids, 128);
        assert_eq!(found, (4..132).collect::<Vec<_>>());
    }
}
