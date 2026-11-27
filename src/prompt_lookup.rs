//! Per-request source lookup. Positions refer to the first token after a four-gram.
use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, sync::Arc};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Default)]
enum OutputLookup {
    #[default]
    Disabled,
    SingleClient,
    Always,
    Repeated,
}

#[derive(Clone, Default)]
pub(super) struct Config {
    max_width: usize,
    adaptive: bool,
    prior_output: OutputLookup,
    single_consensus: bool,
    consensus: Option<Arc<DecimalTokens>>,
    extend_repetition: bool,
}

pub(super) fn configured(tokenizer: &Tokenizer) -> Result<Config> {
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
    let prior_output = match std::env::var("DS4_OUTPUT_LOOKUP").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => OutputLookup::Disabled,
        Ok("single") => OutputLookup::SingleClient,
        Ok("1") => OutputLookup::Always,
        Ok("repeated") => OutputLookup::Repeated,
        _ => bail!("DS4_OUTPUT_LOOKUP must be 0, single, 1 or repeated"),
    };
    let single_consensus = match std::env::var("DS4_C1_OUTPUT_CONSENSUS").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => bail!("DS4_C1_OUTPUT_CONSENSUS must be 0 or 1"),
    };
    let consensus = (single_consensus || matches!(prior_output, OutputLookup::Repeated))
        .then(|| Arc::new(DecimalTokens::new(|s| tokenizer.token_to_id(s))));
    let extend_repetition = match std::env::var("DS4_EXTEND_REPETITION").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => bail!("DS4_EXTEND_REPETITION must be 0 or 1"),
    };
    Ok(Config {
        extend_repetition,
        consensus,
        single_consensus,
        prior_output,
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
    /// Index the supplied prompt once. Previous output is searched only on a
    /// prompt lookup miss and only when explicitly enabled.
    pub(super) fn new(prompt: &[u32], config: Config) -> Self {
        let mut lookup = Self {
            prompt_len: prompt.len(),
            config,
            latest: HashMap::new(),
            previous: vec![],
        };
        if lookup.config.max_width > 0 {
            for end in 4..prompt.len() {
                let key = prompt[end - 4..end].try_into().unwrap();
                lookup
                    .previous
                    .push(lookup.latest.insert(key, end).unwrap_or(0) as u32);
            }
        }
        lookup
    }
    pub(super) fn propose(
        &self,
        ids: &[u32],
        remaining: usize,
        learned_ready: bool,
        single_client: bool,
    ) -> Vec<u32> {
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
        let use_output = match self.config.prior_output {
            OutputLookup::Disabled => false,
            OutputLookup::SingleClient => single_client,
            OutputLookup::Always | OutputLookup::Repeated => true,
        };
        if length == 0 && use_output {
            let output = &ids[self.prompt_len..];
            let recent = &output[output.len().saturating_sub(2048)..];
            let (prior, matched) = prior_output_proposal(recent, width);
            let repeated = if single_client {
                self.config.single_consensus
            } else {
                matches!(self.config.prior_output, OutputLookup::Repeated)
            };
            if let (true, Some(decimals)) = (repeated, &self.config.consensus) {
                let proposed =
                    repeating_output(recent, width, decimals, self.config.extend_repetition);
                if !proposed.is_empty() && (matched < 16 || proposed.len() >= prior.len()) {
                    return proposed;
                }
                if learned_ready && matched < 16 {
                    return vec![];
                }
            }
            return prior.to_vec();
        }
        prompt[source..source + length].to_vec()
    }
}
// Repeated code edits often share generated signatures or documentation that
// never occurred in the supplied file. Read only the committed output already
// present at this scheduler boundary. The target verifies these tokens as usual.
fn prior_output_proposal(recent: &[u32], width: usize) -> (&[u32], usize) {
    const MATCH: usize = 8;
    if recent.len() <= MATCH || width == 0 {
        return (&[], 0);
    }
    let suffix = &recent[recent.len() - MATCH..];
    let mut best = (0, 0);
    let mut source = 0;
    for (start, _) in recent[..recent.len() - 1]
        .windows(MATCH)
        .enumerate()
        .rev()
        .filter(|(_, w)| *w == suffix)
        .take(16)
    {
        let end = start + MATCH;
        let mut matched = MATCH;
        while matched < end.min(64)
            && recent[end - matched - 1] == recent[recent.len() - matched - 1]
        {
            matched += 1;
        }
        let available = width.min(recent.len() - end);
        if available >= 4 && (matched, available) > best {
            best = (matched, available);
            source = end;
        }
    }
    let cap = match best.0 {
        0..8 => 0,
        8..16 => 12,
        16..32 => 24,
        _ => 48,
    };
    (&recent[source..source + best.1.min(cap)], best.0)
}

// Token IDs are not ordered by numeric value. Only canonical single-token
// decimal strings participate in extrapolation; all other tokens match exactly.
#[derive(Default)]
struct DecimalTokens {
    values: HashMap<u32, i32>,
    ids: Vec<Option<u32>>,
}

impl DecimalTokens {
    fn new(id_for: impl Fn(&str) -> Option<u32>) -> Self {
        let ids: Vec<_> = (0..1000)
            .map(|n| id_for(&n.to_string()).filter(|&id| id < 129280))
            .collect();
        let values = ids
            .iter()
            .enumerate()
            .filter_map(|(n, id)| id.map(|i| (i, n as i32)))
            .collect();
        Self { values, ids }
    }
    fn predict(&self, older: u32, newer: u32) -> Option<u32> {
        if older == newer {
            return Some(newer);
        }
        let next = 2 * self.values.get(&newer)? - self.values.get(&older)?;
        self.ids.get(usize::try_from(next).ok()?).copied().flatten()
    }
}

// Compare the current suffix against the corresponding positions in two
// completed repetitions. Advance a second period only when a third source
// agrees for each position. A changing numeric field needs a third
// previous value confirming its step before it can enter a proposal.
fn repeating_output(
    recent: &[u32],
    width: usize,
    decimals: &DecimalTokens,
    extend: bool,
) -> Vec<u32> {
    const LIMIT: usize = 96;
    let n = recent.len();
    let mut best = Vec::new();
    let mut rank = (0, 0);
    for period in 8..=256.min(n.saturating_sub(4) / 2) {
        let suffix = (1..=period.min(32).min(n - 2 * period))
            .take_while(|&offset| {
                let i = n - offset;
                decimals.predict(recent[i - 2 * period], recent[i - period]) == Some(recent[i])
            })
            .count();
        if suffix < 4 {
            continue;
        }
        let cap = if extend && suffix >= 12 && n >= 3 * period {
            LIMIT.min(2 * period)
        } else {
            period.min(if suffix < 8 { 12 } else { 48 })
        };
        let mut tokens = [0; LIMIT];
        let mut length = 0;
        for (offset, slot) in tokens[..width.min(cap)].iter_mut().enumerate() {
            let second = offset >= period;
            let phase = if second { offset - period } else { offset };
            let i = n + phase;
            let (older, newer) = (recent[i - 2 * period], recent[i - period]);
            if (second || older != newer)
                && i.checked_sub(3 * period)
                    .and_then(|j| decimals.predict(recent[j], older))
                    != Some(newer)
            {
                break;
            }
            let Some(token) = decimals.predict(older, newer) else {
                break;
            };
            *slot = if second {
                let Some(advanced) = decimals.predict(newer, token) else {
                    break;
                };
                advanced
            } else {
                token
            };
            length += 1;
        }
        // A six-token agreement competes with a learned draft of up to seven.
        let score = (suffix.min(12), length);
        if length >= 6 && score > rank {
            rank = score;
            best.clear();
            best.extend_from_slice(&tokens[..length]);
        }
    }
    best
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
            prior_output: OutputLookup::Disabled,
            single_consensus: false,
            consensus: None,
            extend_repetition: false,
        }
    }

    #[test]
    fn generated_lookup_requires_a_previous_matching_suffix_and_reads_only_committed_ids() {
        let passage: Vec<u32> = (1..=32).collect();
        let output = [passage.as_slice(), &[81, 82, 83, 84, 85], &passage].concat();
        assert_eq!(prior_output_proposal(&output, 5).0, [81, 82, 83, 84, 85]);
        assert!(prior_output_proposal(&passage, 16).0.is_empty());
        assert!(prior_output_proposal(&[12; 7], 16).0.is_empty());
        assert!(prior_output_proposal(&output, 0).0.is_empty());
        let mut cfg = config(true);
        cfg.prior_output = OutputLookup::Always;
        let prompt = [999; 32];
        let ids = [prompt.as_slice(), &output].concat();
        assert!(PromptLookup::new(&prompt, config(true))
            .propose(&ids, 5, true, true)
            .is_empty());
        assert_eq!(
            PromptLookup::new(&prompt, cfg).propose(&ids, 5, true, true),
            [81, 82, 83, 84, 85]
        );
    }

    #[test]
    fn single_client_lookup_leaves_concurrent_output_to_the_learned_drafter() {
        let passage: Vec<_> = (1..=32).collect();
        let prior = [passage.as_slice(), &[81, 82, 83, 84], &passage].concat();
        let prompt = [999; 32];
        let ids = [prompt.as_slice(), &prior].concat();
        let lookup = PromptLookup::new(
            &prompt,
            Config {
                prior_output: OutputLookup::SingleClient,
                ..config(true)
            },
        );
        assert_eq!(lookup.propose(&ids, 4, true, true), [81, 82, 83, 84]);
        assert!(lookup.propose(&ids, 4, true, false).is_empty());
    }

    #[test]
    fn consensus_drafts_from_short_suffixes_with_two_agreeing_continuations() {
        let block = |marker| vec![20, 21, 22, 23, 24, 25, marker, marker, 7, 8, 9, 10];
        let output = [block(70), block(71), block(72)].concat();
        let decimals = DecimalTokens::default();
        assert_eq!(
            repeating_output(&output, 48, &decimals, false),
            [20, 21, 22, 23, 24, 25]
        );
        assert!(repeating_output(&output, 5, &decimals, false).is_empty());
        assert!(repeating_output(&block(70), 48, &decimals, false).is_empty());
        let prompt = [9090; 8];
        let lookup = PromptLookup::new(
            &prompt,
            Config {
                prior_output: OutputLookup::SingleClient,
                single_consensus: true,
                consensus: Some(Arc::new(DecimalTokens::default())),
                ..config(true)
            },
        );
        let ids = [prompt.as_slice(), &output].concat();
        assert!(prior_output_proposal(&output, 48).0.is_empty());
        assert_eq!(
            lookup.propose(&ids, 48, true, true),
            [20, 21, 22, 23, 24, 25]
        );
        assert!(lookup.propose(&ids, 48, true, false).is_empty());
        assert!(lookup.propose(&ids, 5, true, true).is_empty());
        let mut disagreement = output;
        disagreement[12 + 2] = 800;
        assert!(repeating_output(&disagreement, 48, &decimals, false).is_empty());
    }

    #[test]
    fn concurrent_repetition_preserves_single_client_policy_and_learned_fallback() {
        let prompt = [9090; 8];
        let lookup = PromptLookup::new(
            &prompt,
            Config {
                prior_output: OutputLookup::Repeated,
                consensus: Some(Arc::new(DecimalTokens::default())),
                ..config(true)
            },
        );
        let block = |v| vec![20, 21, 22, 23, 24, 25, v, v, 7, 8, 9, 10];
        let repeated = history(&prompt, &[block(70), block(71), block(72)].concat());
        assert_eq!(
            lookup.propose(&repeated, 48, true, false),
            [20, 21, 22, 23, 24, 25]
        );
        assert!(lookup.propose(&repeated, 48, true, true).is_empty());
        for matched in [8, 20] {
            let suffix: Vec<_> = (100..100 + matched).collect();
            let tail = [81, 82, 83, 84];
            let ids = history(&prompt, &[suffix.as_slice(), &tail, &suffix].concat());
            if matched == 8 {
                assert!(lookup.propose(&ids, 4, true, false).is_empty());
                assert_eq!(lookup.propose(&ids, 4, false, false), tail);
            } else {
                assert_eq!(lookup.propose(&ids, 4, true, false), tail);
            }
        }
    }

    #[test]
    fn consensus_can_extend_a_verified_decimal_progression() {
        let decimals = DecimalTokens::new(|s| s.parse::<u32>().ok().map(|n| 2200 - n));
        let id = |n| 2200 - n;
        let block = |n: u32| {
            let mut b: Vec<u32> = (100..120).collect();
            b[10] = id(n);
            b
        };
        let mut history = [block(9), block(12), block(15)].concat();
        history.extend_from_slice(&block(18)[..8]);
        let draft = repeating_output(&history, 12, &decimals, false);
        assert_eq!(draft, block(18)[8..]);
        history[10] = id(8);
        assert!(repeating_output(&history, 12, &decimals, false).is_empty());
        assert_eq!(decimals.predict(id(12), id(9)), Some(id(6)));
        assert_eq!(decimals.predict(id(20), id(8)), None);
        let invalid = DecimalTokens::new(|s| match s {
            "8" => Some(910),
            "9" => Some(911),
            "10" => Some(129280),
            _ => None,
        });
        assert_eq!(invalid.predict(910, 911), None);
        assert_eq!(decimals.predict(id(998), id(999)), None);
    }

    #[test]
    fn extended_repetition_uses_committed_witnesses_and_stops_at_the_budget() {
        let decimals = DecimalTokens::new(|s| s.parse::<u32>().ok().map(|n| 3000 + n));
        let block = |i: u32| {
            let mut b: Vec<u32> = (10..42).collect();
            b[9] = 3000 + i;
            b[11] = 3200 - i;
            b
        };
        let history = [block(0), block(1), block(2)].concat();
        let expected = [block(3), block(4)].concat();
        for budget in [6, 32, 33, 47, 63, 64, 128] {
            assert_eq!(
                repeating_output(&history, budget, &decimals, true),
                expected[..budget.min(64)]
            );
        }
        assert_eq!(
            repeating_output(&history, 128, &decimals, false),
            expected[..32]
        );
        let mut disagreement = history;
        disagreement[1] = 7000;
        assert_eq!(
            repeating_output(&disagreement, 128, &decimals, true),
            expected[..33]
        );
    }

    #[test]
    fn consensus_respects_client_count_and_preserves_strong_single_matches() {
        let prompt = [9090; 12];
        let mut cfg = config(true);
        cfg.prior_output = OutputLookup::Always;
        cfg.single_consensus = true;
        cfg.consensus = Some(Arc::new(DecimalTokens::default()));
        let lookup = PromptLookup::new(&prompt, cfg);
        for size in [8, 16] {
            let pattern: Vec<u32> = (40..40 + size).collect();
            let ids = [prompt.as_slice(), &pattern, &[1, 2, 3, 4, 5, 6], &pattern].concat();
            assert_eq!(lookup.propose(&ids, 6, true, false), [1, 2, 3, 4, 5, 6]);
            assert_eq!(
                lookup.propose(&ids, 6, true, true).len(),
                if size == 8 { 0 } else { 6 }
            );
            assert_eq!(lookup.propose(&ids, 6, false, true), [1, 2, 3, 4, 5, 6]);
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
                lookup.propose(&history, 150, false, true),
                prompt[end..end + width]
            );
            let ready_width = if matched < 16 { 0 } else { width };
            assert_eq!(
                lookup.propose(&history, 150, true, true),
                prompt[end..end + ready_width]
            );
            assert_eq!(
                fixed.propose(&history, 150, true, true),
                prompt[end..end + 128]
            );
            assert_eq!(
                lookup.propose(&history, 3, false, true),
                prompt[end..end + 3]
            );
        }
        // A long match at the end of the prompt has only a short continuation.
        assert_eq!(
            lookup.propose(&history(&prompt, &prompt[160..258]), 128, false, true),
            prompt[258..]
        );
        let capped = PromptLookup::new(
            &prompt,
            Config {
                max_width: 9,
                adaptive: true,
                prior_output: OutputLookup::Disabled,
                single_consensus: false,
                consensus: None,
                extend_repetition: false,
            },
        );
        assert_eq!(
            capped.propose(&history(&prompt, &prompt[..64]), 128, false, true),
            prompt[64..73]
        );
        assert!(PromptLookup::new(&prompt, Config::default())
            .propose(&history(&prompt, &prompt[..64]), 128, false, true)
            .is_empty());
    }

    #[test]
    fn longest_source_suffix_wins_and_never_crosses_prompt_end() {
        let prompt = [9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 0, 1, 2, 3, 77];
        let lookup = PromptLookup::new(&prompt, config(false));
        assert_eq!(
            lookup.propose(&history(&prompt, &[9, 8, 0, 1, 2, 3]), 4, false, true),
            [4, 5, 6, 7]
        );
        assert_eq!(
            lookup.propose(&history(&prompt, &[9, 9, 0, 1, 2, 3]), 128, false, true),
            [77]
        );
        assert!(lookup
            .propose(
                &history(&prompt, &[101, 102, 103, 104, 101, 102, 103, 104]),
                128,
                false,
                true
            )
            .is_empty());
        assert!(lookup
            .propose(&history(&prompt, &[1, 2, 3]), 128, false, true)
            .is_empty());
        let prompt: Vec<_> = (0..140).collect();
        assert_eq!(
            PromptLookup::new(&prompt, config(false)).propose(
                &history(&prompt, &[0, 1, 2, 3]),
                128,
                false,
                true
            ),
            (4..132).collect::<Vec<_>>()
        );
        assert!(PromptLookup::new(&prompt, Config::default())
            .propose(&history(&prompt, &[0, 1, 2, 3]), 128, false, true)
            .is_empty());
    }
}
