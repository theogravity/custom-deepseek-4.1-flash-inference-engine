//! Per-request source lookup. Positions refer to the first token after a four-gram.
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

#[derive(Clone, Copy, Default)]
pub(super) struct Config {
    max_width: usize,
    adaptive: bool,
}

pub(super) fn configured() -> Result<Config> {
    match std::env::var("DS4_PROMPT_LOOKUP").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => return Ok(Config::default()),
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
    let adaptive = match std::env::var("DS4_ADAPTIVE_LOOKUP").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => bail!("DS4_ADAPTIVE_LOOKUP must be 0 or 1"),
    };
    Ok(Config {
        max_width: width,
        adaptive,
    })
}

pub(super) struct PromptLookup {
    prompt_len: usize,
    config: Config,
    latest: HashMap<[u32; 4], usize>,
    previous: Vec<u32>,
}
impl PromptLookup {
    /// Restrict lookup to the source text supplied by this request. Generic
    /// code generation can reuse punctuation from its own output frequently
    /// even when the subsequent code differs, wasting a long verification pass.
    pub(super) fn new(prompt: &[u32], config: Config) -> Self {
        let mut lookup = Self {
            prompt_len: prompt.len(),
            config,
            latest: HashMap::new(),
            previous: vec![],
        };
        if config.max_width > 0 {
            for end in 4..prompt.len() {
                let key = prompt[end - 4..end].try_into().unwrap();
                lookup
                    .previous
                    .push(lookup.latest.insert(key, end).unwrap_or(0) as u32);
            }
        }
        lookup
    }
    pub(super) fn propose(&self, ids: &[u32], remaining: usize, learned_ready: bool) -> Vec<u32> {
        let width = self.config.max_width.min(remaining);
        if ids.len() < 4 || width == 0 {
            return vec![];
        }
        let prompt = &ids[..self.prompt_len];
        let n = ids.len();
        let suffix: &[u32; 4] = ids[n - 4..].try_into().unwrap();
        let mut end = self.latest.get(suffix).copied().unwrap_or(0);
        let mut best = (0, 0);
        let mut source = 0;
        for _ in 0..32 {
            if end == 0 {
                break;
            }
            let mut matched = 4;
            while matched < end.min(n).min(128) && prompt[end - matched - 1] == ids[n - matched - 1]
            {
                matched += 1;
            }
            let score = (matched, width.min(prompt.len() - end));
            if score > best {
                best = score;
                source = end;
            }
            end = self.previous[end - 4] as usize;
        }
        // Choose the source using the full available suffix, then spend more
        // verification rows only when recent output follows that source closely.
        let (matched, available) = best;
        let length = match (self.config.adaptive, matched) {
            (true, 0..16) if learned_ready => 0,
            (true, 0..32) => available.min(16),
            (true, 32..64) => available.min(64),
            _ => available,
        };
        prompt[source..source + length].to_vec()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn history(prompt: &[u32], suffix: &[u32]) -> Vec<u32> {
        [prompt, &[9001], suffix].concat()
    }

    fn config(adaptive: bool) -> Config {
        Config {
            max_width: 128,
            adaptive,
        }
    }

    #[test]
    fn lookup_width_tracks_the_contiguous_source_match() {
        let prompt: Vec<_> = (0..260).collect();
        let lookup = PromptLookup::new(&prompt, config(true));
        let fixed = PromptLookup::new(&prompt, config(false));
        let end = 100;
        for (matched, width) in [
            (4, 16),
            (15, 16),
            (16, 16),
            (31, 16),
            (32, 64),
            (63, 64),
            (64, 128),
            (95, 128),
        ] {
            let mut history = history(&prompt, &[9000; 40]);
            history.extend_from_slice(&prompt[end - matched..end]);
            assert_eq!(
                lookup.propose(&history, 150, false),
                prompt[end..end + width]
            );
            let ready_width = if matched < 16 { 0 } else { width };
            assert_eq!(
                lookup.propose(&history, 150, true),
                prompt[end..end + ready_width]
            );
            assert_eq!(fixed.propose(&history, 150, true), prompt[end..end + 128]);
            assert_eq!(lookup.propose(&history, 3, false), prompt[end..end + 3]);
        }
        // A long match at the end of the prompt has only a short continuation.
        assert_eq!(
            lookup.propose(&history(&prompt, &prompt[160..258]), 128, false),
            prompt[258..]
        );
        let capped = PromptLookup::new(
            &prompt,
            Config {
                max_width: 9,
                adaptive: true,
            },
        );
        assert_eq!(
            capped.propose(&history(&prompt, &prompt[..64]), 128, false),
            prompt[64..73]
        );
        assert!(PromptLookup::new(&prompt, Config::default())
            .propose(&history(&prompt, &prompt[..64]), 128, false)
            .is_empty());
    }

    #[test]
    fn longest_source_suffix_wins_and_never_crosses_prompt_end() {
        let prompt = [9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 0, 1, 2, 3, 77];
        let lookup = PromptLookup::new(&prompt, config(false));
        assert_eq!(
            lookup.propose(&history(&prompt, &[9, 8, 0, 1, 2, 3]), 4, false),
            [4, 5, 6, 7]
        );
        assert_eq!(
            lookup.propose(&history(&prompt, &[9, 9, 0, 1, 2, 3]), 128, false),
            [77]
        );
        assert!(lookup
            .propose(
                &history(&prompt, &[101, 102, 103, 104, 101, 102, 103, 104]),
                128,
                false
            )
            .is_empty());
        assert!(lookup
            .propose(&history(&prompt, &[1, 2, 3]), 128, false)
            .is_empty());
        let prompt: Vec<_> = (0..140).collect();
        assert_eq!(
            PromptLookup::new(&prompt, config(false)).propose(
                &history(&prompt, &[0, 1, 2, 3]),
                128,
                false
            ),
            (4..132).collect::<Vec<_>>()
        );
        assert!(PromptLookup::new(&prompt, Config::default())
            .propose(&history(&prompt, &[0, 1, 2, 3]), 128, false)
            .is_empty());
    }
}
