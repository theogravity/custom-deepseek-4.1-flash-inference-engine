# Full headline benchmark on the recommended defaults

Run on November 27, 2026 against commit `a586b10`, image `ds4-flash-rs:tp2-peer-index-20261127`, with the recommended TP2 profile active. The two inference GPUs were capped at 375 W each; the CPU socket cap read 225 W. The profile used an 8,192-token prefill step, 68 GiB MoE cache, 524,288-token context, peer-index, LFU eviction, batched top-k, C1 consensus, extended repetition, head-row tiles, repeated-output drafting, tokenized agent text caching and 32,768-token idle tool prefill.

The run covers the upstream Python code edit at one, four and six clients; four single-client cold-prefill sizes; and six independent cold 404k-token sessions. Decode runs use one warm-up followed by six measured waves. The cold-prefill tool ran with `--timing-only`, so it checked terminal completion and zero prefix reuse without grading the requested first code. The six-session run used fresh independent prompts. We did not run a cache-repeat timing or a needle-retrieval check.

## Upstream code-edit decode

Each client received a 25,625-token Python module and generated 1,500 tokens at temperature zero. Aggregate rate is the benchmark's sum-of-output-tokens divided by the longest individual first-output-to-close duration. First-output latency is the median across individual streams and six measured waves.

| Clients | Published vLLM at 600 W | Previous concurrent coding release | Recommended defaults at 375 W | Defaults vs. published vLLM |
|---:|---:|---:|---:|---:|
| 1 | 282 | 383 | **343.1** | +21.7% |
| 4 | 518 | **584** | 312.0 | -39.8% |
| 6 | 557 | **583** | 261.2 | -53.1% |

For the recommended defaults, median first output was 1.71 s at one client, 6.43 s at four clients and 9.69 s at six clients. The previous coding release measured about 0.18 s at four clients and 0.27 s at six in its matched repetition experiment. The new combined profile therefore keeps the single-client gain but does not preserve the earlier four- and six-client throughput or first-output behavior. This run shows the regression; it does not isolate which promoted setting or profile interaction caused it.

The comparison with published vLLM is directional: the native run used 375 W per GPU and vLLM's published figures used 600 W. The six-wave native medians and single published second-pass figures also have different sample sizes. [Raw client records](../results/upstream/c1-code-edit-default-20261127.jsonl) · [four clients](../results/upstream/c4-code-edit-default-20261127.jsonl) · [six clients](../results/upstream/c6-code-edit-default-20261127.jsonl)

## Regression diagnosis and fix

The 8,192-token compute batch was not itself making later decode slower. The native engine also used that value as its prefix snapshot interval. Thus the repeated 25,625-token code-edit prompt could restore only 24,576 tokens, leaving 1,049 tokens for serial prefill before decoding. The earlier concurrent coding profile used 64-token batches, which accidentally provided finer prefix checkpoints as well as smaller compute batches. vLLM keeps these controls separate: an 8k scheduler prefill budget can coexist with 64-token KV blocks.

The engine now separates them. Physical prompt batches remain 8,192 tokens; prefix checkpoints are 64-token aligned, using a sparse geometric schedule and a final reusable block to bound snapshot storage. For this workload the final checkpoint is at 25,600 tokens, leaving only 25 prompt tokens after restore. This preserves large cold-prefill batches while avoiding the 1,049-token warm tail. Targeted boundary/key tests passed, the full release suite passed (82 passed, 48 GPU-only tests ignored), and the replacement TP2 image is healthy on the recommended 68 GiB cache profile.

We did not run a cache-repeat timing or retrieval check. The one post-deploy C4 wave encountered the pre-existing 24,576-token disk snapshot, so its 8.68–9.24 s first-output observations are not a measurement of reuse from the new 25,600-token checkpoint. We therefore make no post-fix TTFT or aggregate-rate claim yet. The earlier chunk-64 ablation showed the potential of a complete shared-prefix restore (C4 452 tok/s / ~0.3 s TTFT; C6 471 tok/s / ~0.36 s TTFT), but those results are not a validation of this new implementation. [First post-deploy C4 wave](../results/upstream/c4-code-edit-prefix64-firstwave-20261127.jsonl)

Those chunk-64 runs also changed physical compute batch size, so they cannot isolate prefix reuse. Raising the MoE cache to 92 GiB alongside chunk 64 produced 533.8/541.8 tok/s at C4/C6, but left only about 1.1 GiB free on one inference GPU and is not a safe full-context default. [68 GiB C4](../results/upstream/c4-code-edit-default-prefill64-20261127.jsonl) · [68 GiB C6](../results/upstream/c6-code-edit-default-prefill64-20261127.jsonl) · [92 GiB C4](../results/upstream/c4-code-edit-default-prefill64-cache92-20261127.jsonl) · [92 GiB C6](../results/upstream/c6-code-edit-default-prefill64-cache92-20261127.jsonl)

A partial 92 GiB C4 run with TP dense projection disabled was stopped when free VRAM fell to 66 MiB; its three measured waves were similar to the TP-dense run, but the arm is too short and memory-constrained to draw a reliable comparison. [Partial raw data](../results/upstream/c4-code-edit-default-prefill64-cache92-no-tpdense-20261127.jsonl)

## Follow-up: larger physical prefill batches

Fresh 60k-token prompts were used to check whether a larger compute batch helps new input tokens. All three completed with zero prefix reuse and timing-only output validation. There was no separate warm-up, and the control timings varied substantially, so this is exploratory rather than a controlled A/B.

| Physical batch | Prompt tokens | TTFT | Free VRAM after request (GPU 1 / GPU 2) |
|---:|---:|---:|---:|
| 8,192 | 60,049 | 27.77 s | not sampled |
| 16,384 | 60,117 | 22.93 s | 3,836 / 902 MiB |
| 8,192 | 60,068 | 20.60 s | 2,822 / 4,428 MiB |

The 16k sample was faster than the first 8k sample, but slower than the later 8k sample and left less than 1 GiB free on GPU 2. A 12,288-token batch failed with a CUDA launch error. The current service was restored to 8,192 tokens and remains healthy. We did not promote a larger batch or claim a speedup; the 16k headroom is too small for the six-session/full-context target. [8k control 1](../results/upstream/cold-60k-control-8k-prefix-search-20261127.jsonl) · [16k candidate](../results/upstream/cold-60k-candidate-16k-prefix-search-20261127.jsonl) · [8k control 2 and memory sample](../results/upstream/cold-60k-control-8k-memory-20261127.jsonl)

## Cold single-client prefill

Each case was a single request with a fresh prompt, zero prefix tokens reused and one output. Inputs use the upstream word counts. Results are single samples; output correctness was not graded in timing-only mode.

| Input tokens | TTFT | Input tokens/s |
|---:|---:|---:|
| 15,824 | 19.30 s | 820 |
| 59,958 | 29.17 s | 2,055 |
| 195,786 | 63.24 s | 3,096 |
| 505,082 | 164.40 s | 3,072 |

The 505k request fit below the 524,288-token context limit. [Per-request timings and validation](../results/upstream/cold-prefill-default-20261127.jsonl)

## Six cold, independent 404k sessions

The capacity harness sent six distinct prompts with 403,998 input tokens and 128 output tokens each. Before the first stream ended, the scheduler sample showed 2,424,683 processed tokens, exceeding the 2,423,988 prompt-token total. The run completed in 882.0 s. Median TTFT was 854.1 s; observed TTFT ranged from 833.5 to 854.2 s. All six outputs completed. Prefix-reuse delta was zero; rotations, resumes and swap failures were all zero. At the proof sample, all six sessions were resident and decoding. Minimum free VRAM sampled during the run was 1,200 MiB on GPU 1 and 2,604 MiB on GPU 2; host available RAM minimum was 234,872 MiB.

This is a cold capacity and admission result, not a cache-repeat or retrieval benchmark. Its 128-token responses and synthetic review prompts are not directly comparable to the published six400 wall-time result. [Full receipt](../results/upstream/six-404k-default-20261127.json)

## Run files

The raw JSONL records preserve every decode wave, including the warm-up wave. The cold-pre-fill JSONL stores the exact seeds, prompt hashes and per-request checks. The six-session JSON contains the seed, six prompt digests, scheduler proof and local memory samples. [Combined benchmark summary](../results/upstream/default-profile-headline-20261127.json)
