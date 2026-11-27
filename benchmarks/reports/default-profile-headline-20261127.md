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
