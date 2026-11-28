# Decode recovery after long prompts: evict prompt shards first

Date: 2026-11-27. Image `ds4-flash-rs:tp2-round4-20261127`, the round-three profile (88 GiB expert limit, 4 GiB VRAM reserve, wide inline descriptors) plus `DS4_EVICT_RANK_FIRST=1`. Both inference GPUs ran at 375 W. Rates are `upstream_tok_s` medians from `tools/edit_scaling.py` unless stated. Every arm's output hashes matched the round-three controls.

## Problem

After one cold long prompt, six-client decode stayed slow for many minutes. In the [round-three check](c46-round3-20261127.md) the waves after a 189k prompt and a 473k prompt measured 141, 249, 277 and 290 tok/s, against a steady rate above 500, under both cache sizes.

The prefix RAM/disk tiers were not involved. They hold attention KV and drafter windows. Expert weights live in a separate GPU slot cache, backed by the mmapped checkpoint in the host page cache. A miss there is a synchronous ~13 MB host-to-GPU copy.

## Cause

TP2 prompt batches of 128 rows or more run every routed expert as two half-width rank shards, one per GPU. Decode uses full-width slots on the expert's parity GPU. LFU frequency counts are shared across these formats, so each decode access also refreshes the counts of that expert's shards. After a long prompt, the shards of decode-hot experts therefore ranked among the most valuable entries and stayed resident even though decode never reads them. Decode then thrashed through its remaining slots.

The `DS4_MOE_STATS=1` census (now including slot formats) for a fresh server, two warm C6 waves, one cold 189k-token prompt and four more C6 waves:

| | Before the prompt | After: 1st quarter | 4th quarter |
|---|---:|---:|---:|
| Full slots / rank shards on GPU 1 | 6,250 / 0 | 4,541 / 2,624 | 4,576 / 2,566 |
| Expert uploads per C6 verification pass | 1.5 | 84.2 | 72.0 |

## Fix

When a full-format (decode or draft) acquisition needs space, the cache now evicts the lowest-ranked evictable rank shard before considering full slots. Slots pinned by the current group are still skipped. Prompt batches keep plain LFU order, and the arithmetic is unchanged. Set `DS4_EVICT_RANK_FIRST=0` for the previous policy. [Implementation and unit test](../../src/moe.rs)

## Measurements

**Recovery.** Same sequence as above; each row is a fresh server and a fresh prompt seed.

| Policy | C6 before the prompt | 189k cold TTFT | C6 waves after the prompt | Uploads/pass after |
|---|---:|---:|---|---:|
| Lifetime LFU (previous) | 524 | 96.0 s | 179, 327, 331, 319 | 72–84 |
| Shards first for decode (selected) | 522 | 98.9 s | 273, **525, 529, 532** | 1.8–1.9 |
| Shards first for prompts too | 515 | **138.3 s** | 450, 523, 527, 530 | 1.8–2.5 |
| Prompts too, once shards hold ≥ 50% | 524 | 95.3 s | 312, 523, 529, 521 | 1.9–2.0 |
| Selected, clean round-4 image | 514 | 96.2 s | 319, 510, 515, 514 | — |

**Mixed traffic.** `tools/mixed_prefill_decode.py`: five edit decoders while a sixth client sends a cold ~19k-token prompt every 10 s. Five waves each.

| Policy | Decode median, tok/s | Cold TTFT median | Uploads per verification pass |
|---|---:|---:|---:|
| Lifetime LFU (previous) | 128.3 | 11.4 s | 43.6 |
| Shards first for decode, two runs | **191.0 / 189.2** | 13.2 / 10.7 s | 17.1–17.3 |
| Shards first for prompts too | 220.1 | 9.8 s | 7.4 |
| Prompts too, once shards hold ≥ 50% | 176.3 | 12.5 s | 17.1 |

Decode-only shard preference restores steady decode from the second wave after a long prompt and adds about 48% to decode under interleaved prompts. Extending the preference to prompt batches helped the mixed test further, but made the isolated 189k cold prompt 44% slower: the decode set occupies the cache and the prompt's shards stream through what is left. The 50% guard removed that TTFT cost, but showed no reliable gain over decode-only in the mixed test, so the simpler decode-only rule was selected. Mixed-test waves vary by ±20%, so differences under about 10% are not resolved.

**Steady state.** The change is neutral when no long prompt has run, because no shards are resident:

| Clients | Round-3 control | Shards first | Round-3 control |
|---:|---:|---:|---:|
| 4 | 533.9 | 531.2 | 534.7 |
| 6 | 533.2 | 533.2 | 534.4 |
| 6, later session | 514.0 | 514.6 | 512.7 |

The later C6 bracket ran about 3.5% lower for every image, including the unchanged round-three control. GPU temperatures, power limits and throttle flags were normal, and no other client sent requests. The drift is unexplained, so cross-session comparisons in this report use same-session controls.

Raw data: [`../results/decode/expert-recovery-20261127/`](../results/decode/expert-recovery-20261127/) (`recovery.jsonl`, `mixed.jsonl`, `steady-*.jsonl`).

## Remaining

- Decode under concurrent cold prompts is still far below steady state (about 190 vs 530 tok/s). Each 8,192-row prompt chunk streams rank shards for nearly every expert, about 15k per GPU, which is more than the cache can hold. A prompt path that reuses resident full slots would reduce both the uploads and the eviction pressure.
- The first wave after a long prompt still reloads the full slots the prompt evicted (273–319 tok/s).
- Not run: six concurrent 524k sessions and KV rotation with this profile.
