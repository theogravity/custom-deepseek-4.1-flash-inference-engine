# Experiment ledger

One row per performance or stability experiment. **Check this before proposing an experiment;** rerun a rejected one only with a new reason, and say what changed. Newest work first. "Kept" means the setting is in the recommended TP2 profile unless noted. Throughput is aggregate tok/s from `tools/edit_scaling.py` (`upstream_tok_s`, six measured waves); TTFT is from `tools/cold_prefill.py` on a fresh server after one ~11k warm-up. All runs: 2× RTX PRO 6000 at 375 W.

Acceptance rule (from 2026-11-27): keep a change that gains on some metrics and is flat (inside the control spread) on the rest; reject only if it is net-flat or something regresses beyond noise. Always bracket a candidate with controls from the same session: this host drifted ~3.5% within one evening.

## Cold prefill, round 2 (2026-11-27)

Cold TTFT, fresh seeds. Controls per row are the runs immediately before/after. Raw: [`results/prefill/cold-round2-20261127/`](results/prefill/cold-round2-20261127/)

| Experiment | Setting | 16k | 60k (2 prompts) | 196k | Verdict |
|---|---|---|---|---|---|
| Direct peer copy of the layer-20 residual/mix (was a host bounce at 5 GB/s) | `DS4_TP2_PEER_BOUNDARY=1` | 3.25 vs 4.11/3.54 s | 11.12, 10.45 vs 12.01, 12.14 / 11.80, 11.88 s | 39.82 vs 43.06/43.85 s | **Kept** (source of the `pb-*` runs). The copy is now completed before return, see the crash fix below |
| Async pinned upload of TP2 rank-group descriptors | `DS4_RECIPE_ASYNC_META=1` (code default) | 3.10 vs 3.42/3.20 s | 11.70, 11.06 vs 11.06, 11.28 / 12.01, 10.66 s | 38.12 vs 39.21/39.42 s | **Kept**: gains at 16k/196k, flat at 60k |
| Overlap Engram gathers with GPU work in prefill | `DS4_OVERLAP_PREFILL_ENGRAM=1` | 3.38 vs 3.33/3.17 s | 11.11, 11.33 vs 11.71, 13.48 / 10.62, 10.47 s | 39.76 vs 39.68/40.50 s | Flat; left off |
| 24 rayon threads for prompt routing/Engram (was 12) | `RAYON_NUM_THREADS=24` | 3.22 vs 3.31/3.47 s | 10.63, 10.94 vs 12.10, 11.08 / 10.56, 10.40 s | 37.12 vs 40.36/40.75 s | **Kept**. Routing 92 → 78 ms/chunk |
| Counting sort of routes by expert (was a serial comparison sort per layer) | code (`sort_by_expert`) | 4 prompts: median 3.08 vs 2.98/2.88 s (samples 2.80–3.64) | 10.40, 10.66 vs 10.63, 10.81 / 10.49, 10.26 s | 37.65 vs 37.98/38.74 s | **Kept**: identical output, routing 70 → 50 ms/chunk; 16k inconclusive (first-request noise) |
| Fill rank-shard staging slabs from 8 rayon tasks (was one thread) | `DS4_TP2_PARALLEL_STAGE=1` (default) | 4 prompts: median 3.11 vs 2.96/3.03 s | 10.69, 10.45 vs 10.30, 10.59 / 10.04, 10.08 s | 37.54 vs 39.48/43.77 s | Kept (small, identical output). Did not shorten slot acquisition: the copy is bound by NVMe page faults, not memcpy |

Per-chunk host split after these (`DS4_PREFILL_TIMING=1`, 8,192 rows): ~1.15–1.35 s total; router wait (GPU attention/dense on GPU 1) ~480 ms; rank groups 245–430 ms; collect ~110 ms; routing ~50 ms; Engram 50–80 ms. Inside rank groups, 150–500 shard uploads per chunk cost ~0.9–1.1 ms each to fill (page faults from NVMe) plus ~0.35–0.5 ms of `cudaMalloc` while the cache grows. The final <128-row tail chunk costs ~650 ms because it is the first small-batch MoE step and must upload full-format experts.

Page cache: expert shards (1–46) total ~239 GB and Engram (47–48) ~110 GB, against ~230 GB usable page cache. During this session Engram was ~96% resident and expert shards 42–89%.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Engram random-access hint, i.e. keep Engram on disk and leave page cache to experts | `DS4_ENGRAM_RANDOM_READ=1` | From a dropped page cache, six cold 60k prompts each: converged 14.0 s vs 10.6 / 12.3 s controls; Engram resident 45 vs 102 / 101 GiB, experts 126 vs 124 / 126 GiB | **Rejected**: Engram gathers are synchronous; random NVMe reads cost more than readahead pollution |
| Pinned host expert tier (L2 for rank shards, LRU, direct DMA) | `DS4_EXPERT_HOST_TIER_GIB=48` (prototype, code removed) | 60k 65.6 s / 16k 9.5 s vs 14.1 / 2.4 s (prefix off); **0 tier hits**: every shard upload in a fresh process is a first use, the GPU cache never fills during one prompt; fills into new pinned memory ~4 ms/shard | **Rejected**. Would only matter for re-misses in a long-running server; revisit with a long-running benchmark |
| Preload the encoder layers' rank shards at startup | `DS4_TP2_PRELOAD_LAYERS=20` | 16k 7.48 vs 5.31 / 4.27 s; 60k 16.15, 14.63 vs 14.48, 13.09 / 11.60, 10.43 s; 196k 46.04 vs 46.59 / 40.18 s; startup +27 s (7,680 shards, 51 GB per GPU) | **Rejected**: reading 102 GB through page cache at startup evicts Engram pages |

Protocol notes: (1) page-cache state carries over between arms and dominates TTFT after anything that drops or floods it. Bracket with controls, and after such a run re-warm with a few prompts before measuring. (2) `tools/cold_prefill.py` seeds must differ per arm: the disk prefix tier survives restarts and a reused seed restores instead of prefilling. (3) The first 16k sample after start is noisy (±15%); use four samples for 16k decisions.

Profile that motivated these (60k cold, mid-prompt, 3.7 s window): neither GPU ran a kernel 55% of the time; GPU 2 was busy 16% (decoder pruning skips layers 21–39 on most chunks). Main-thread CPU with both GPUs idle 772 ms; per-layer router waits 1.24 s; host bounce of the layer-20 activation ~470 ms; TP dense peer traffic 25.9 GB / 479 ms; TP2 descriptor copies 6,350 sync calls / 708 ms.

### Round-5 validation (2026-11-28)

Clean `ds4-flash-rs:tp2-round5-20261128`, profile defaults only, bracketed by `tp2-round4-20261127` in its own configuration (peer boundary off, 12 threads). Raw: `results/decode/validation-round5-*`, `results/prefill/cold-round2-20261127/validation-*`.

| Check | Round 4 | **Round 5** | Round 4 |
|---|---|---|---|
| C4 steady | 505.2 | **535.5** | 532.8 |
| C6 steady | 525.0 | **530.7** | 525.2 |
| Output hashes | — | identical to session baseline | — |
| Cold 16k / 60k / 196k | 9.21 / 20.39, 14.60 / 44.96 s | **3.78 / 10.48, 10.17 / 38.46 s** | Xid 31 on first request (pre-existing bug) |
| Fresh-process crash reproducer | ~44% faults | **0/6** (30/30 since the fix) | — |

## Stability: Xid 31 copy-engine fault (2026-11-27)

| Step | Result |
|---|---|
| Symptom | ~44% of fresh processes faulted (Xid 31, CE write `FAULT_PDE` on GPU 2) during the first cold request; request fails with `unspecified launch failure` |
| Pre-existing? | Yes: the original image `tp2-peer-index-prefix64-20261127` faulted too |
| Not the cause | `DS4_EVICT_RANK_FIRST` (faulted with 0), `DS4_TP2_STAGE_WEIGHTS=0` (3/6), `DS4_TP2_PEER_DMA=0` (2/6), `DS4_TP2_INDEX=0` (1/5) |
| Avoided by | Prefix tiers off (0/6, changes chunk shapes), or a device sync right after the staged MoE peer collect (0/8; syncs after peer groups 3/8, primary groups 2/8, uploads 2/8 did not help) |
| Cause | The small-batch staged MoE path queues a peer→primary copy into a pooled `compact` buffer, then drops it. After a long prefill the pool is full, so the buffer is freed at once; GPU 2's copy engine then writes freed memory |
| Fix | Keep `compact` until that device's next router read (`retired_collect` in `src/moe.rs`); also complete `ds_copy_peer_f32` before returning. 24/24 clean fresh-process runs after the fix (12 with the peer boundary on) |

Diagnostic left in place: `DS4_DEBUG_SYNC=1` synchronizes both GPUs after each prefill stage and logs the stage that faulted. Bisect log: [`xid-bisect.log`](results/prefill/cold-round2-20261127/xid-bisect.log).

## Decode recovery after long prompts (2026-11-27) — [report](reports/expert-shard-eviction-20261127.md)

| Experiment | Result | Verdict |
|---|---|---|
| Decode misses evict TP2 rank shards first (`DS4_EVICT_RANK_FIRST=1`) | C6 after a 189k prompt: 179, 327, 331, 319 → 273, 525, 529, 532; mixed decode 128 → 189–191; steady unchanged | **Kept** |
| Prompt batches also evict shards first | Mixed decode 220 and best TTFT there, but isolated 189k TTFT 96 → 138 s | Rejected |
| …only once shards hold ≥ 50% of resident bytes | No TTFT cost, no reliable gain over decode-only | Rejected (simpler rule kept) |

## C4/C6 round 3 (2026-11-27) — [report](reports/c46-round3-20261127.md)

| Experiment | C4 | C6 | Verdict |
|---|---|---|---|
| Inline MoE descriptors up to 256 rows (`DS4_MOE_INLINE_WIDE=1`) | 501.1 vs 499.2/498.7 | 508.9 vs 500.5/505.3 | **Kept** |
| Pinned-ring async small H2D uploads (attention, `to_device`) | 503.9 | 506.9 (below both controls) | Rejected, code removed |
| Expert cache 88 GiB + 4 GiB VRAM reserve (`DS4_MOE_CACHE_GIB=88`, `DS4_MOE_HEADROOM_MIB=4096`) | 535.0 vs 499.0/500.5 | 533.3 vs 511.3/507.5 | **Kept** |
| Same with a 6 GiB reserve | 503.8 | 523.9 | Rejected: C4 working set needs ~78 GiB cache (cliff) |

## C4/C6 round 3: cold prefill and memory check (2026-11-27) — [report](reports/c46-round3-20261127.md)

Two warm C6 waves, then cold 189k and 473k needle prompts (fresh seeds), then C6 again. Single samples. [Raw rows](results/context/cold-after-decode-cache-88-vs-68-20261127.jsonl)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| 88 GiB cache / 4 GiB reserve vs 68 GiB / 512 MiB, long cold prompts | `DS4_MOE_CACHE_GIB=88`, `DS4_MOE_HEADROOM_MIB=4096` | TTFT 189k 102.5 vs 95.9 s, 473k 152.1 vs 149.6 s; needles found; lowest free VRAM GPU 1/2 602/2,512 vs 2,660/4,704 MiB; no allocation failures | Kept (non-reserving prefill allocations still dip 3.5 GiB below the reserve) |
| C6 decode right after the long prompts | both profiles | 141, 249, 277, 290 (68 GiB) and 127, 256 (88 GiB) tok/s vs >500 steady | Pre-existing slow recovery; fixed by `DS4_EVICT_RANK_FIRST=1` (section above) |
| Profile of a warm C6 wave (Nsight, 2.78 s) | — | Both GPUs idle 34%; 1.02 s in router `cudaEventSynchronize`; 8,498 sync `cudaMemcpy` in `ds_moe_run_dev` = 333 ms; 637–952 expert uploads = 265–335 ms; `DS4_STEP_TIMING=1`: target forward ~90% of step | Finding (motivated rows above) |

## Earlier reports, migrated

Verdicts are as stated in each report at the time. Later rounds changed the profile (for example 64-token prefill chunks were replaced by 8,192-token physical prefill with 64-token prefix checkpoints), so "kept" here means "selected by that round". Unless noted: 375 W per GPU, 225 W CPU socket, fresh process per arm, one warm-up plus six measured waves, controls listed as before/after. "Edits" is the 25,625-token Python edit with 1,500 output tokens; "404k" is six (or one) independent 404,151–404,170-token coding histories.

## Cost-aware adaptive draft budget (2026-11-27) — [report](reports/adaptive-draft-budget-2026-11-27.md)

Metric: `whole_wave_tok_s` median. Controls: `c4/c6-shared-budget-restart-control-20261127.jsonl`.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Shared draft budget maximizing expected tokens / measured draft+target step cost | `DS4_ADAPTIVE_DRAFT_BUDGET=1` | C4 460.5 vs 490.7 (−6.2%); C6 499.5 vs 495.6 (+0.8%, within noise); hashes match | Opt-in, off by default |
| First adaptive runs | curve collection also printed every confidence vector | Throughput distorted; raw files labelled `verbose-log-contaminated` | Discarded; vectors now print only with `DS4_CONFIDENCE_TRACE=1`, lookup batches excluded from cost learning |

## vLLM and Hyperflare mechanisms (2026-11-27) — [report](reports/vllm-hyperflare-followup-2026-11-27.md)

Metric: `whole_wave_tok_s` median; control C4 490.7, C6 495.6.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| CPU host-expert overlap ported to the TP2 sharded cache | CPU path on | C4 457.8 (−6.7%), C6 499.7 (+0.8%); `upstream_tok_s` 464.9 / 504.6 | Rejected; disabled |
| Hyperflare-style per-depth confidence calibration (EWMA of predicted vs observed survival) | CPU path off | C4 468.0 (−4.6%), C6 483.9 (−2.4%) | Rejected; code removed |
| CUDA graph around each routed MoE FFN group (router reads and dispatch stay eager) | `DS4_MOE_CUDA_GRAPH=1` (`DS4_MOE_CUDA_GRAPH_TRACE=1` for logs) | C4 488.8 (−0.4%), C6 498.3 (+0.5%); hashes match | Within noise; opt-in only. Full-step capture blocked by host router sync |

## Fixed shared draft budget (2026-11-27) — [report](reports/vllm-c46-decode-path-20261127.md)

Fresh-restart controls: C4 498.8, C6 504.1 tok/s.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Rank draft positions by cumulative survival across requests, fixed budget 4 tokens/request | `DS4_BATCH_DRAFT_BUDGET_PER_REQUEST` (4) | C4 481.3 (−3.5%), C6 488.6 (−3.1%) | Rejected; off by default |
| Same, 6 tokens/request | `DS4_BATCH_DRAFT_BUDGET_PER_REQUEST` (6) | C4 459.2 (−7.9%), C6 462.9 (−8.2%) | Rejected |
| Finding: local vLLM recipe | — | Full decode-only CUDA graphs, DSpark k=5, adaptive verification **not** enabled; CPU experts are its measured advantage (485 off vs 537–561 on) | Reference |

## Batch-aware confidence threshold (2026-11-27) — [report](reports/batch-adaptive-confidence.md)

Edits at C1/C4/C6, same server process across client counts. Control: threshold 0.4 for every request (C1 408.5, C4 495.0, C6 499.1).

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Threshold 0.4 for one decoder, +0.1 (0.5) when several decode together | batch delta 0.1 | C1 410.0 (+0.37%), C4 497.8 (+0.56%), C6 503.8 (+0.93%); hashes match | **Kept** (default) |
| Batch delta 0.2 | — | C6 504.1 | Tied with 0.1; not selected |
| Global threshold 0.3 | — | C6 479.2 | Rejected |
| Global threshold 0.5 | — | C6 505.1 but C1 398.3 | Rejected (hurts C1) |
| Speculative width 8 (vs 7) | — | C6 496.4 | Rejected |
| Speculative width 9 | — | C6 487.0 | Rejected |

## Recommended-default headline and prefix-checkpoint fix (2026-11-27) — [report](reports/default-profile-headline-20261127.md), [report](reports/prefix-fix-benchmark-20261127.md)

Metric: `upstream_tok_s` (published convention).

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Recommended TP2 defaults, prefix snapshots tied to the 8,192-token prefill chunk | image `tp2-peer-index-20261127` | C1 343.1, C4 312.0, C6 261.2 (prior release 383/584/583); first output 1.71 / 6.43 / 9.69 s | Regression: repeated prompt restored only 24,576 of 25,625 tokens |
| 64-token prefix checkpoints, 8,192-token physical prefill | image `tp2-peer-index-prefix64-20261127` | C4 491.4, C6 497.8 (+57.5%, +90.6%); first output 0.219 / 0.312 s | **Kept** |
| MoE cache 92 GiB with 64-token chunks | — | C4 533.8, C6 541.8; ~1.1 GiB free on one GPU | Rejected: unsafe for full context (later 88 GiB + 4 GiB reserve kept) |
| 92 GiB cache, TP dense projection disabled | — | Stopped at 66 MiB free after 3 waves | Inconclusive: too short and memory-bound |
| Physical prefill 16,384 tokens | `DS4_PREFILL_CHUNK=16384` | 60k TTFT 22.93 s vs 27.77 / 20.60 s (8k); 902 MiB free on GPU 2 | Rejected: headroom too small |
| Physical prefill 12,288 tokens | `DS4_PREFILL_CHUNK=12288` | CUDA launch error | Rejected |
| DSpark sidecar restore error handling | — | Intermittent CUDA `unspecified launch failure` on restore, previously swallowed as a cache miss | Fixed: request fails and engine is marked unavailable (not reproduced) |
| Reference: cold prefill on defaults | timing-only, single samples | 15.8k 19.30 s, 60k 29.17 s, 196k 63.24 s, 505k 164.40 s; six cold 404k sessions 882.0 s (median TTFT 854.1 s), min free VRAM 1,200 / 2,604 MiB | Reference |

## Wide TP2 prefill and the pack's S8B kernel (2026-11-26/27) — [report](reports/local-vllm-prefill.md)

Single cold samples after an 11k warm-up; zero prefix reuse; host page cache warm.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Reference: installed vLLM vs native 64-token chunks with deferred decoder | same rendered tokens | 16k 2.91 vs 11.78 s; 60k 8.07 vs 38.91 s; 196k 20.62 vs 159.69 s; 505k 52.92 vs 438.09 s | Reference (native 4.1–8.3× slower) |
| Pack S8B expert kernel (`xmoe_do_i8`) at 128-token chunks | `--features recipe-prefill`, control `DS4_RECIPE_PREFILL=0` | 16k 14.26 vs 21.11 / 12.85 s; 60k 50.64 vs 76.21 / 50.45 s | Inconclusive: within control spread (later used for wide TP2 prompts) |
| Two-rank EXL3 expert shards (rank-local gate/up/down) | `check_recipe_tp2.py` | Bit-exact vs vLLM extension in 4/4 cases | Correctness only |
| TP2 physical step 8,192 vs 16,384 (rank shards, sharded dense, MXFP8, BF16 index) | `DS4_PREFILL_CHUNK` | 16k 8.68 vs 12.74 s; 60k 32.12 vs 29.73 s; 68k review during generation 51.89 vs 51.91 s; longest peer stall 13.24 vs 17.94 s | 8,192 kept (one run each) |
| FP8 index + BF16 peer transfers vs BF16 index (16k step) | `DS4_PREFILL_INDEX_FP8`, `DS4_TP_DENSE_BF16_WIRE` (=0 for the BF16 arm) | 16k 4.44 vs 3.77 s; 60k 13.96 vs 14.61 s; 196k 84.84 vs 93.32 s; FP8 at 8k step: 3.77 / 13.99 / 78.86 s | FP8 on in the TP2 overlay (single samples; arms also differ by indexed LFU) |
| Repeat 8,192 vs 16,384, FP8 build | `DS4_PREFILL_CHUNK` | 16k 3.70 vs 9.64 s; 60k 11.85 vs 21.38 s; review 45.23 vs 41.50 s; peer stall 12.84 vs 14.89 s | 8,192 stays default |
| 505k prompt at 16k step, BF16 index | — | 356.32 s; 66 MiB free per card at minimum | Memory limit: no headroom at 16k step |
| Profile, 170k prompt at 98k tokens (BF16 build) | — | Radix sort 688 ms, candidate scoring 194 ms, peer copies 45.03 GB, layer-boundary residual via host 302 ms D2H | Finding (later fixed by `DS4_TP2_PEER_BOUNDARY=1`) |

## Cold prefill: deferred decoder work (2026-11-26) — [report](reports/cold-prefill-optimization.md)

64-token chunks, single samples per arm. [Data](results/upstream/cold-prefill-optimization.json)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Skip expiring hidden-state work in layers 21–39 on chunks far from a snapshot or output | `DS4_PREFILL_DECODER_PRUNE=1` | 16k 15.51 vs 18.82 / 15.64 s; 60k 43.61 vs 60.83 / 63.99 s; checkpoints byte-identical | **Kept** |
| Pass larger expert routing tables as CUDA launch parameters | — | 16k 15.04 vs 16.98 / 17.05 s; 60k 57.79 vs 59.96 / 60.58 s | Inconclusive: no verdict in report (gain 3–12%) |
| 128-token chunks | — | 16k 14.21, 13.67 s | Exploratory only |
| 512-token chunks on the direct-peer path | — | 16k 35.58 s, then Xid 31 on the second request | Rejected: copy-engine fault. Possibly the same class as the staged-collect bug fixed 2026-11-27 (see Xid 31 section), but this was the direct-peer path; not re-tested |

## Headline replay on the coding release (2026-11-26) — [report](reports/headline-reference.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Upstream edit, C1/C4/C6 (median of six second passes) | 64-token chunks, K=7 + source lookup | 382.5 / 457.5 / 502.5 vs published vLLM 282 / 518 / 557 | Reference |
| Upstream cold prefill | same | 16k 23.14 s, 60k 60.9 s, 196k 204.77 s, 505k 601.78 s (12–16× vLLM) | Reference |
| Six cold ~404k prompts (`six400.py`) | same | Stopped at 2,511 s (2,211,520 tokens processed) | Inconclusive: run interrupted |

## Concurrent output repetition (2026-11-26) — [report](reports/concurrent-repetition.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Repeated-output proposals for every client while peers decode, plus numeric extrapolation | `DS4_OUTPUT_LOOKUP=repeated` (control `single`) | C4 590.2 vs 463.7 / 462.5; C6 582.9 vs 505.2 / 505.2; first output unchanged (0.181 / 0.265 s); six-way passes 1,494 → 905; hashes match | **Kept** (concurrent-coding release) |
| Upstream headline on that release | same | C1 383, C4 584, C6 583 | Reference |

## Coding-agent follow-ups (2026-11-26) — [report](reports/agent-followups.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Cache tokenized ordinary-text spans (64 entries, 32 MiB, ≥ 512 bytes) | — | First output on changing 404k history 0.482 vs 0.847 / 0.930 s; count endpoint 1.9 vs 445.1 / 466.2 ms | **Kept** |
| Interleave two independent DSpark Markov dot products per warp | tag `checkpoint-2026-11-26-agent3-02-markov` | Pi decode 178.4 vs 178.0 / 178.4; 404k decode 256.8 vs 256.0 / 256.2 tok/s | Rejected: within noise |
| Bulk SHA-256 prefix keys (terminal digest only) | tag `checkpoint-2026-11-26-agent3-03-prefix` | First output 135.0 vs 136.6 / 139.6 ms; target lookup 0.81 vs 1.53 / 1.55 ms | Rejected: end-to-end flat |
| Selected release, six 404k sessions | — | Decode 304.5 vs 299.5 / 304.7; whole-wave 299.7 vs 282.9 / 286.4 tok/s | Reference |

## Tool-return latency (2026-11-26) — [report](reports/tool-return.md)

Real Pi edit plus tool result; first-event latency, median of two full-exchange replays.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Prefill the finished turn on idle GPUs while the tool runs (≥ 512 generated tokens, after 200 ms) | `DS4_IDLE_TOOL_PREFILL_MAX_TOKENS=32768` (0 = off) | Immediate 4.177 vs 4.165 / 4.167 s; 2 s tool 2.669 vs 4.168 / 4.223 s; 6 s tool 0.097 vs 4.166 / 4.169 s | **Kept** |
| Reuse generated KV directly instead of canonical prefill | tag `checkpoint-2026-11-26-pi-decode-kv` | 0.34 vs 4.60 / 4.28 s, but reply changed (59 vs 62 tokens, different shell call) | Rejected: output drift |
| 128-token prefill chunks | — | 3.603 vs 4.064 / 4.080 s; regenerated exchange made a different edit | Not selected (output changed) |
| Overlap Engram gathers | `DS4_OVERLAP_PREFILL_ENGRAM=1` | 4.054 vs 4.066 / 4.054 s | Flat; off |
| Router upper bound to skip experts that cannot reach the top six | tag `checkpoint-2026-11-26-pi-router-bound` | 4.079 vs 4.079 / 4.080 s | Rejected (< 3%) |

## Pi coding session (2026-11-26) — [report](reports/pi-coding.md)

One Go parser bug, three session IDs per profile, 240 s deadline.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Deployed engine vs previous engine, Pi `xhigh` | — | Clean finishes 1/3 vs 1/3 and 2/3; walls 220.3 / 240.0 / 155.0 s vs 180.0 / 240.1 / 196.8 and 240.0 / 173.6 / 97.6 s | Inconclusive: three sessions, tool output varies |
| Decode on the captured 2,786-token tool-result turn | — | 178.0 vs 178.3 / 178.7 tok/s; first event 45.4 vs 46.7 / 43.6 ms | Flat |
| Pi effort `high` instead of `xhigh` | deployed engine | 2/3 correct clean finishes (141.4, 128.5 s), third timed out | Inconclusive: supports trying `high`, needs more tasks |

## Decode round 2, single client (2026-11-26) — [report](reports/decode-round-2.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Longer repetition proposals (up to two periods, ≤ 96 tokens, ≥ 12-token suffix, three periods) | tag `checkpoint-2026-11-26-c2-01-repetition` | C1 edit 385.8 vs 366.2 / 364.8 (+5.5%); C1 404k after edits 252.9 vs 255.7 / 256.4 (−1.2%, −1.96% on repeat); fresh long-only 259.23 / 260.26 / 259.07 | **Kept** (404k ordering effect unexplained) |
| DSpark Markov shortlist (rescore top two logits per 32-token group) | tag `checkpoint-2026-11-26-c2-02-markov` | Microbench 96.6 → 28.5 µs; C1 edit 366.6 vs 383.9 / 383.5 (−4.5%); 404k 242.2 vs 253.1 / 252.4 (−4.2%); more verification rounds | Rejected |
| Vocabulary-head row tiling (two or four 8-row tiles) | tag `checkpoint-2026-11-26-c2-03-head` | C1 edit 386.8 vs 384.0 / 383.4 (+0.8%); C1 404k −0.3%; C6 edits 506.4 vs 498.8 / 500.7; C6 404k 282.7 vs 276.5 / 272.2; 48 rows 1.26 vs 2.63–2.67 ms | **Kept** |

## Single-client decode (2026-11-26) — [report](reports/single-client-decode.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Wide attention ranking (recursive top 512 per 4,096-entry tile) | tag `checkpoint-2026-11-26-c1-01-wide` | Edit 294.0 vs 293.8 / 293.8; 404k 237.1 vs 237.1 / 236.9; selector faster at 8 rows, slower at 5 rows × 404k | Rejected: no gain |
| Warp-shared expert weights (neighbour-lane word reuse) | tag `checkpoint-2026-11-26-c1-02-expert` | Edit 293.8 vs 293.9 / 293.4; 404k 237.1 vs 235.9 / 236.5; repeat 239.65 vs 239.64 / 239.31 | Rejected |
| Verified output repetition for a lone decoder (copy agreeing fields, extrapolate one decimal token) | tag `checkpoint-2026-11-26-c1-03-consensus` | Edit 366.5 vs 293.5 / 293.9 (+24.8%), repeat 365.4; 404k 256.9 vs 236.0 / 237.1 (+8.6%), repeat 256.3; C6 edits +0.4%, C6 404k −0.2% | **Kept** |

## Decode round 1, six clients (2026-11-26) — [report](reports/decode-round-1.md), [top-k](reports/attention-topk.md), [drafter inputs](reports/drafter-inputs.md), [paired index](reports/paired-index.md)

404k rows: 768 output tokens, earliest first to latest last token.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Tiled batched attention top-k (4,096-key tiles, union ranked per row) | `DS4_BATCH_TOPK=1` | C6 edits 504.4 vs 461.7 / 466.1 (+8.7%); C6 404k 333.4 vs 318.7 / 320.0 (+4.4%); C1 404k 240.2 vs 234.8; full window 286.7 vs 275.5 / 276.5; selector 180.8 → 29.7 µs | **Kept** |
| Borrow drafter logits/hidden/mixture ranges instead of copying | `DS4_DRAFT_VIEWS=1` | C6 edits 503.5 vs 504.3 / 502.9; C6 404k 328.5 vs 327.3 / 333.1 | Rejected (experiment tag) |
| Reuse dequantized index-key registers for two query rows | `DS4_PAIRED_INDEX=1` | C6 404k 335.9 vs 333.4 / 335.0 (+0.5%); reversed order 336.7 / 333.1 / 331.5; C1 241.6 vs 239.7; kernel 698.0 vs 732 µs | Rejected: gain not resolvable (edits never enter the dispatch) |
| Pack two queries into one FP32 pair (alternative layout) | microbench | 1,109.9 vs 729.6 / 732.1 µs at 202,112 × 8 | Rejected |

## Selective CPU experts at 404k (2026-11-26) — [report](reports/cpu-experts.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Up to four uncached target experts per step on a 10-thread host pool (`cepf_*`, int16 activations) | `DS4_CPU_EXPERTS=1` | Six 404k, 512 out: 266.4 vs 297.2 / 298.8; 768 out: 289.4 vs 320.9 / 324.9; 0–4 of 36 outputs identical to GPU; rel. L2 0.0009–0.0010 | Rejected; experiment branch |

## Six concurrent edits vs vLLM (2026-11-26) — [report](reports/concurrent-edits.md)

Metric: published aggregate (completion tokens / longest first-token-to-end).

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| vLLM CPU experts on / off / on (same host) | `EXL3_CPU_EXPERTS` | 561.03 / 484.98 / 537.20; native 461.57 / 462.80; vLLM used 22.7 vs 3.1 cores | Reference: CPU experts give vLLM +10.8–15.7% |
| Native draft maximum 5 (vs 7) | `DS4_SPEC_K=5` | 421.2 vs 460.3 / 461.4 (−8.5 to −8.7%); 3.99 vs 4.81 accepted/round | Rejected |
| Fused gate/up expert kernel with 2 or 8 warps (vs 4) | microbench | 30 jobs 282.2 / 278.8 / 287.4 µs; 32 jobs 313.2 / 303.6 / 306.7; 8 warps 0.7% faster only at 34 jobs | Rejected: < 2% screening bar |
| Warm expert uploads on this workload | counters | 2 and 0 uploads per warm wave | Finding: little for CPU experts to take over once warm |
| Profile of a warm C6 wave (Nsight, 1.87 s) | — | Both GPUs busy 15.9%, neither 17.0%; radix sort alone 12.2% of wall | Finding (led to `DS4_BATCH_TOPK`) |

## Shared packed KV rows on prefix restore (2026-11-26) — [report](reports/shared-prefix.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Immutable views of the serialized prefix image instead of copying into new vectors | — | Six 404k, 512 out: 296.6 vs 278.2 / 278.8; 768 out: 313.2 vs 300.1 / 302.7; six 524k: 287.7 vs 202.7 / 204.4; C1 404k 234.2 vs 234.2 / 233.8; deserialization 136.61 → 0.23 ms; wall-to-ready 849 → 275 ms; anon memory 5.64 vs 8.71 / 8.57 GiB | **Kept** |

## LFU expert eviction (2026-11-26) — [report](reports/expert-cache.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| LFU with lifetime per-device counts vs LRU | `expert-cache-lfu.yml` | Six 404k, 512 out: 278.1 vs 263.8 / 258.6; 768 out: 300.0 vs 285.0 / 292.1, repeat 297.0 vs 287.1 / 285.7; C1 234.7 vs 233.8 / 233.2; uploads 2,676 → 1,191 per wave (−55.5%, −18.4 GiB) | **Kept** (opt-in at first, default from the shared-prefix round) |
| Task switch from coding to six full-window retrievals | LFU vs LRU | First wave 103.8 vs 84.2 / 109.8; next six 208.1 vs 208.5 / 213.7 | Neutral |
| Replay: segmented LRU, 25 / 50 / 75% protected | trace replay | 2,667 / 2,621.5 / 2,402 uploads per wave | Rejected (replay) |
| Replay: LFU with dynamic aging | trace replay | 1,557 uploads | Rejected (lifetime counts better) |
| Replay: farthest next use (oracle) | trace replay | 771 uploads | Lower bound |

## Expert transfers and drafter settings at 404k (2026-11-26) — [report](reports/expert-transfers.md)

Six 404k clients, 512 output tokens. Aggregate visible tok/s.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Maximum five learned proposals | — | 257.8 vs 261.4 / 260.2 (−0.9 to −1.4%) | Rejected |
| Confidence threshold 0.65 (vs 0.40) | — | 260.6 vs 260.2 / 264.2; acceptance 74% → 88%; resident 325.2 vs 317.4 / 320.9 | Rejected: delivered rate between controls, whole-wave lower |
| Profile: expert uploads during six-way decode | counters + Nsight | 2.78–2.82 GiB/s; 137 ms (9.0%) of a 1.52 s trace with neither GPU busy | Finding (led to LFU) |
| Native host run vs Docker image | — | Host-only run changed one client's output | Open: profile only in-image, output-matched |

## Long-context decode at 375 W (2026-11-25/26) — [report](reports/long-context.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| Packed index scorer with 8 or 16 lanes per dot-product group | — | 404k C1 232.9 vs 212.2 / 211.8, C4 238.4 vs 221.1 / 224.4, C6 252.1 vs 231.1 / 227.8; retrieval C1 524k 371.7 vs 321.1 / 322.3, C6 524k 118.3 vs 111.1 / 111.8; kernel −17 to −45% | **Kept** |
| Host prefix RAM tier 6 GiB (vs 2 GiB) | prefix RAM budget | Six 404k: 284.6 vs 249.4 / 248.3 (+14–15%); first-response spread 1.56 vs 3.86 / 4.03 s; six full windows 214.0 vs 129.0 / 117.4 | **Kept** |
| Share the drafter's dense projections across sessions | branch `experiment/long-context-joint-draft` | C6 228.2 vs 230.9 / 231.1 | Rejected |
| Reference: client scaling at 404k, 512 out | — | C1 233.3, C4 287.6, C6 255.8 aggregate | Reference (C4 highest) |

## Published comparison and power cap (2026-11-25/26) — [report](reports/vllm-comparison.md)

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| GPU cap 600 / 375 / 600 W, same process | power limit | Upstream edit C1 277 / 275 / 278, C4 417 / 414 / 419, C6 453 / 448 / 454 (600 W +0.9–1.2%); 524k receipts 314.4 / 314.8 / 302.6 | 375 W kept: ≤ 1.2% gain |
| Generated-output lookup under all loads (600 W) | — | C1/C4/C6 278 / 388 / 406 vs 278 / 423 / 459 | Rejected then (lone-decoder only); later superseded by `DS4_OUTPUT_LOOKUP=repeated` |
| Draft maximum 5 at C1 (600 W, function-50 edit) | — | 262.1 vs 277.7 tok/s | Rejected |
| All source lookup disabled (C1) | — | 267.9 vs 277.7 tok/s | Rejected |
| Four-tile packed projections + parallel router selection (prefill) | `DS4_PREFILL_MX_TILES=4`, `DS4_VERIFY_MX_TILES=4`, `DS4_PARALLEL_PREFILL_ROUTER=1` | ~15.8k prompts, warm cache: 1,040 → 1,176 input tok/s (+13.0%); 128-token chunks 1,319 | **Kept** (64-token chunks retained then) |
| 512-row prefill with staged expert transfers | — | Copy-engine fault twice; direct peer above 128 rows passed, 478 input tok/s | Rejected: Xid fault; production kept 64-row chunks |
| Reference: six cold independent 404k requests (64-token chunks) | 375 W | 3,714.61 s vs vLLM 187 s | Reference |

## Single-client context sweep and LiteLLM overhead (2026-11-25) — [report](reports/single-client-context.md)

128-token receipt retrieval, reasoning off, six pairs per size.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| LiteLLM proxy vs direct, OpenAI and Anthropic wire (375 W) | — | Largest route difference 0.43%; proxy adds 6–17 ms to first text | Finding: proxy overhead negligible |
| Tuned build vs control build at 600 W | 64-token chunks, `DS4_OUTPUT_LOOKUP=single`, MX tiles 4, `DS4_PARALLEL_PREFILL_ROUTER=1`, `DS4_OVERLAP_PREFILL_ENGRAM=0` | 4k 504.4 vs 357.2; 65k 447.6 vs 308.4; 524k 322.9 vs 237.8 tok/s | **Kept** (receipt workload repeats codes and benefits from output lookup) |
| Decode after a cold prompt (prefix tiers off) | `DS4_PREFIX_RAM_MIB=0 DS4_PREFIX_DISK_MIB=0` | 65k still 307.9 tok/s after ~66 s ingestion | Reference |

## Bring-up, two-card tuning and ceilings (2026-11-25) — [report](reports/workstation-history.md), [roofline](reports/roofline.md), [development history](../docs/development-history.md)

Mostly short new-code prompts (1,024 tokens) and Tetris/Go-edit screening runs, some at 600 W.

| Experiment | Setting | Result | Verdict |
|---|---|---|---|
| sm_120-compatible numerics | — | 26.11 vs 25.48 tok/s | Kept for numeric compatibility (within noise) |
| Device-resident target, then GPU sampler + packed dense kernels | no drafter | 25.75 → 44.50 tok/s; 256-token Tetris to 86 tok/s | **Kept** |
| Learned drafters, released vs code-tuned DSpark | lookup off | Early 123 vs 130 tok/s; later 269.8 vs 300.6 (target-only 84.4); both match undrafted output | **Kept** code-tuned |
| Packed MXFP8 target and GPU head | K=7 | K=7 preferred for the code overlay | **Kept** |
| Split all dense projections across both GPUs (decode) | — | Slowed the short coding control | Rejected (dense TP later used for wide prompts only) |
| Direct remote expert reads | — | Slowed the short coding control | Rejected; explicit staging instead |
| Staged target expert split across cards | — | 287.97 vs 260.14 / 259.64 tok/s (768 Tetris tokens) | **Kept** |
| Learned confidence cutoff 0.4 | — | Verification rows 929 → 824, same 634 accepted; 260 → 270 tok/s | **Kept** |
| Source lookup | — | Full Go edit 470.5 tok/s, final combined profile 846.6 | **Kept** |
| Fixed 128-token source lookup on a 25,598-token rewrite | — | 211 / 1,954 proposals accepted; 166.2 tok/s vs 308.1 for the contiguous-match hybrid | Rejected; hybrid kept |
| Fixed 16-position probe / learned-only / per-request width feedback | same rewrite | 302.7 / 331.2 / 244.0 tok/s | Width feedback rejected; hybrid scheduler selected |
| Joint six-client target with sequence-local attention and drafting | — | 336 tok/s (two screening waves) | **Kept** |
| Joint attention projections and learned draft projections | — | 399 tok/s, six coding clients, bytes match serial | **Kept** |
| Packed FP32 index scorer | — | Six full windows 242.6 → 288.9 tok/s | **Kept** |
| Radix sort dispatch by row count and index width | — | 288.9 → 361.1 tok/s; 262k × 8 selection 3.71 → 2.08 ms | **Kept** |
| Reference: decode ceiling | measured VRAM 1,702 GB/s at 375 W and 600 W | 11.03 GB/token → 6.48 ms floor = 154 tok/s target-only (95.6 with BF16-expanded dense); ideal two-card split ~309 tok/s | Reference: power cap does not change bandwidth |

## Open candidates (not yet tried)

- Cold prefill, largest remaining bucket: the per-layer router wait (~480 ms per 8k chunk) is GPU attention/dense work on GPU 1 while GPU 2 idles (decoder pruning leaves layers 21–39 mostly idle). Split attention heads across both GPUs (true TP, as vLLM does), or move part of the encoder layers to GPU 2.
- Cold prefill: a pooled slot arena instead of `cudaMalloc` per new expert slot (~0.35–0.5 ms each while the cache grows); overlap or shrink the synchronous TP2 collect (~110 ms per chunk); remove the ~650 ms tail-chunk cost (first small-batch MoE step uploads full-format experts evicted by the prompt).
- Benchmark protocol: add a long-running-server cold-prefill benchmark (warm GPU cache, mixed prior traffic). Fresh-process runs make every shard upload a first use, which is why the host expert tier showed zero hits.
- Cold prefill: reduce TP dense peer traffic (25.9 GB per 3.7 s window); a prompt path that reuses resident full expert slots instead of streaming rank shards.
- C6: two-microbatch pipelining across the two layer stages, or GPU-side routing to remove the per-layer router sync.
- Same-session vLLM comparison on the mixed and recovery workloads.
- C6: run each request's attention kernels as one batched launch in `forward_group` instead of serially (about 6× the launches at C6). [round 3](reports/c46-round3-20261127.md)
- Full decode-step CUDA graph replay (vLLM `FULL_DECODE_ONLY` style); needs capture-safe router and attention first. Only per-MoE-group graphs were tried. [follow-up](reports/vllm-hyperflare-followup-2026-11-27.md)
- Device-side causal-prefix draft allocator using profiled draft/target costs (the CPU-side cost-aware selector was tried). Revisit Hyperflare's allocation-surplus correction only after explaining why the local cost estimates and realized tokens per step diverge. [shared budget](reports/vllm-c46-decode-path-20261127.md), [follow-up](reports/vllm-hyperflare-followup-2026-11-27.md)
- Overlap known expert misses with independent expert computation via a bounded pinned staging buffer and a copy stream. [expert transfers](reports/expert-transfers.md)
- Overlap independent indexer and query projections before the attention launch (Flare); instrument the individual projection sites first. [concurrent edits](reports/concurrent-edits.md)
- Sweep the fused gate/up expert kernel's output tile width, keeping split-K partials and reduction order (only the 2/4/8-warp block size was tried). [expert transfers](reports/expert-transfers.md)
- Qualification not yet run on the round-3/4/5 profile: six concurrent 524k sessions, KV rotation, C1. [round 3](reports/c46-round3-20261127.md)
- Quality and soak: matched GSM8K, HumanEval, mixed-load interference and a 20-minute soak against the vLLM recipe. [vLLM comparison](reports/vllm-comparison.md)
- Pi reasoning effort `high` vs `xhigh` on a wider set of bugs and repositories. [Pi coding](reports/pi-coding.md)
