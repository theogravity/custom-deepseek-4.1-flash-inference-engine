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

pub(super) struct PromptLookup<'a> {
    prompt: &'a [u32],
    config: Config,
    width: usize,
    latest: HashMap<[u32; 4], usize>,
    previous: Vec<u32>,
}
impl<'a> PromptLookup<'a> {
    /// Restrict lookup to the source text supplied by this request. Generic
    /// code generation can reuse punctuation from its own output frequently
    /// even when the subsequent code differs, wasting a long verification pass.
    pub(super) fn new(prompt: &'a [u32], config: Config) -> Self {
        let mut lookup = Self {
            prompt,
            config,
            width: if config.adaptive {
                config.max_width.min(16)
            } else {
                config.max_width
            },
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
    pub(super) fn propose(&self, ids: &[u32], remaining: usize) -> Vec<u32> {
        let width = self.config.max_width.min(remaining);
        if ids.len() < 4 || width == 0 {
            return vec![];
        }
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
            while matched < end.min(n).min(128)
                && self.prompt[end - matched - 1] == ids[n - matched - 1]
            {
                matched += 1;
            }
            let score = (matched, width.min(self.prompt.len() - end));
            if score > best {
                best = score;
                source = end;
            }
            end = self.previous[end - 4] as usize;
        }
        self.prompt[source..source + best.1.min(self.width)].to_vec()
    }

    /// Feedback is request-local and applies only to verified source proposals.
    pub(super) fn observe(&mut self, proposed: usize, accepted: usize) {
        if !self.config.adaptive {
            return;
        }
        if accepted == proposed {
            // A source ending or generation limit can truncate an otherwise
            // short probe. Only grow when the current width was actually tested.
            if proposed == self.width {
                self.width = (self.width * 4).min(self.config.max_width);
            }
        } else {
            // Retain a small margin above the observed copy length and cut an
            // overlong failed probe immediately. Four rows bound near misses.
            let margin = 4.max(accepted.div_ceil(4));
            self.width = self.width.min(accepted + margin);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn config(adaptive: bool) -> Config {
        Config {
            max_width: 128,
            adaptive,
        }
    }

    #[test]
    fn lookup_width_adapts_to_verified_copies_and_resets_for_each_request() {
        let prompt: Vec<_> = (0..200).collect();
        let mut lookup = PromptLookup::new(&prompt, config(true));
        let prefix = &prompt[..4];
        for (proposed, accepted) in [(16, 16), (64, 64), (128, 10), (14, 0), (4, 4), (16, 16)] {
            assert_eq!(lookup.propose(prefix, 200), prompt[4..4 + proposed]);
            lookup.observe(proposed, accepted);
        }
        assert_eq!(lookup.propose(prefix, 200).len(), 64);
        assert_eq!(lookup.propose(prefix, 3), prompt[4..7]);
        lookup.observe(3, 3);
        assert_eq!(lookup.propose(&prompt[194..198], 200), prompt[198..]);
        lookup.observe(2, 2);
        assert_eq!(lookup.propose(prefix, 200).len(), 64);
        assert_eq!(
            PromptLookup::new(&prompt, config(true))
                .propose(prefix, 200)
                .len(),
            16
        );
        let mut fixed = PromptLookup::new(&prompt, config(false));
        fixed.observe(128, 0);
        assert_eq!(fixed.propose(prefix, 200).len(), 128);
        for cap in [1, 3, 9] {
            let mut short = PromptLookup::new(
                &prompt,
                Config {
                    max_width: cap,
                    adaptive: true,
                },
            );
            short.observe(cap, cap);
            assert_eq!(short.propose(prefix, 200).len(), cap);
            short.observe(cap, 0);
            assert_eq!(short.propose(prefix, 200).len(), cap.min(4));
        }
    }

    #[test]
    fn longest_source_suffix_wins_and_never_crosses_prompt_end() {
        let prompt = [9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 0, 1, 2, 3, 77];
        let lookup = PromptLookup::new(&prompt, config(false));
        assert_eq!(lookup.propose(&[9, 8, 0, 1, 2, 3], 4), [4, 5, 6, 7]);
        assert_eq!(lookup.propose(&[9, 9, 0, 1, 2, 3], 128), [77]);
        assert!(lookup
            .propose(&[101, 102, 103, 104, 101, 102, 103, 104], 128)
            .is_empty());
        assert!(lookup.propose(&[1, 2, 3], 128).is_empty());
        let prompt: Vec<_> = (0..140).collect();
        assert_eq!(
            PromptLookup::new(&prompt, config(false)).propose(&[0, 1, 2, 3], 128),
            (4..132).collect::<Vec<_>>()
        );
        assert!(PromptLookup::new(&prompt, Config::default())
            .propose(&[0, 1, 2, 3], 128)
            .is_empty());
    }
}
