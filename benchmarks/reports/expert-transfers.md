# Next long-context optimizations at 375 W

## Where to optimize next

1. **Reduce expert-weight transfers during six-client decode.** Per-pass engine counters showed 2.78–2.82 GiB/s of expert uploads in every warm six-client wave. Target verification reloaded roughly 0.57–0.58% of its distinct per-layer expert selections. Each reload costs 13,315,584 bytes. In the 1.52-second C6 trace, 395 pairs of copies matched the packed trellis and scale sizes, totaling 4.90 GiB. They occupied 166 ms, including 137 ms (9.0% of elapsed time) when neither GPU had a recorded kernel. The C1 trace had zero copies of those sizes. First, record the reuse distance of each evicted expert and replay alternative eviction policies at the same slot budget. Then test overlapping known misses with independent expert computation using a bounded pinned staging buffer and a copy stream. A reused slot must wait for its preceding consumer, and each projection must wait for the upload it reads. Keep the six full-context KV slots and their memory headroom throughout. [Expert cache](../../src/moe.rs) · [Weight upload](../../native/moe.cu) · [Per-wave counters and trace](../results/decode/next-optimizations-375w.json)
2. **Reduce host copying during warm prefix admission.** In a six-request 404k wave, each target prefix image was about 695 MiB. The median RAM lookup took 1.5 ms, target deserialization took 136 ms, and the host spent another 61 ms in GPU installation calls. The latter measures time until stream-ordered work can use the data and includes allocation. The decoder copies the packed KV and index bytes into eight new vectors. A prototype can share immutable slices of the serialized image through installation and copy only on host mutation. Measure first-response spread, whole-wave throughput and peak host memory, including evicted-but-referenced images. This targets admission. [Snapshot decoder](../../src/attention_snapshot.rs) · [Prefix ownership](../../src/prefix.rs) · [Restore records](../results/decode/next-optimizations-375w.json)
3. **Optimize the routed expert projections, then the long-context index scorer.** Expert projections accounted for 38% of summed kernel time at both C1 and C6. Replay the common routed batch shapes from the trace, starting with the fused gate/up projection. Sweep the output tile width while preserving the split-K partials and their reduction order. For the index scorer, which rose from 13.2% at C1 to 16.9% at C6, test loading the same compressed keys once for multiple verification rows on consecutive-key scans. Preserve each row's causal mask, FP32 reduction order and tie handling. Check raw result bits before repeating the native 404k and full-window tests. [Expert projection](../../native/moe.cu) · [Packed scorer](../../native/attention_score.cuh) · [Captured grid shapes](../results/decode/next-optimizations-375w.json)

## Drafter settings on six 404k coding clients

The stable checkpoint uses a maximum of seven learned proposals per round and a confidence threshold of 0.40. Each row below used the same six independent prompts (404,151–404,170 input tokens), 512 visible output tokens per client, greedy decoding and the native OpenAI chat endpoint. Each leg started a fresh container, warmed it with one wave and measured the next six. The controls immediately before and after each candidate used the checkpoint settings. The two GPUs each had a 375 W limit. The CPU socket cap read 225 W.

| In run order | Aggregate visible tok/s | Whole-wave tok/s | Committed tok/s while six were decoding | Proposed / accepted per learned round |
|---|---:|---:|---:|---:|
| Checkpoint A | 261.4 | 246.0 | 319.0 | 5.37 / 3.96 |
| Maximum five proposals | 257.8 | 241.5 | 305.1 | 4.29 / 3.32 |
| Checkpoint B | 260.2 | 245.2 | 317.4 | 5.37 / 3.97 |
| Confidence threshold 0.65 | 260.6 | 242.9 | 325.2 | 3.43 / 3.03 |
| Checkpoint C | 264.2 | 246.7 | 320.9 | 5.37 / 3.96 |

The shorter draft delivered 0.9–1.4% fewer tokens/s than its controls. The stricter threshold raised learned-proposal acceptance from about 74% to 88%. Its resident rate rose 1.4–2.5%, and its delivered rate fell between the controls. Both candidates had lower whole-wave throughput than either neighboring control. All 210 output digests across the 35 waves matched the respective saved checkpoint responses. Every measured wave restored 2,424,704 prefix tokens and recorded a six-session target pass.

Visible throughput counts 3,066 output tokens after the first token of each stream, from the earliest first text to the latest last text. Whole-wave throughput counts all 3,072 tokens and includes admission. The resident figure estimates committed target tokens from scheduler samples over the longest interval with all six in decode. These three columns are medians of the six measured waves. Draft counters include the warm-up. At the observed control rates, reaching 300 visible tok/s would require shortening the first-text-to-last-text span by approximately 1.4–1.6 seconds, from 11.6–11.8 seconds to 10.22 seconds. [Per-wave results](../results/decode/next-optimizations-375w.json) · [Workload and reproduction commands](long-context.md#reproduce-the-coding-workload)

## GPU timelines during resident decode

| Trace measurement at 404k | One client | Six clients |
|---|---:|---:|
| Analyzed wall time | 1.43 s | 1.52 s |
| Routed expert projections, share of summed kernel time | 38.6% | 38.4% |
| Packed dense projections, same denominator | 20.2% | 14.7% |
| Packed index scores, same denominator | 13.2% | 16.9% |
| Both GPUs executing recorded kernels, share of wall time | 17.0% | 12.6% |
| Exactly one GPU executing a recorded kernel, share of wall time | 71.7% | 62.6% |
| Neither GPU executing a recorded kernel, share of wall time | 11.3% | 24.7% |

Each Nsight Systems capture used the tagged Docker image, the two designated cards, the existing prefix store and one warm-up wave. The C1 request generated 1,024 tokens and each C6 request generated 768. The central 90% of the first-to-last kernel interval fell within scheduler observations showing every client in decode. Kernel shares sum device time across both cards. The wall-time rows use the union of recorded kernel intervals per device. Nsight reported incomplete events at capture stop. The window lengths, different output budgets and profiling overhead limit comparisons to the sampled workload. End-to-end speedups require uninstrumented measurements. The original reports are saved on this host under `~/.cache/ds4-flash-rs/profiles/375w-00de13a/` as `profile-c1.nsys-rep` and `profile-c6.nsys-rep`. The results file includes their SHA-256 digests. [Kernel, copy and scheduler measurements](../results/decode/next-optimizations-375w.json)

The separate expert telemetry run used the tagged service with per-pass MoE counters enabled, six 512-token clients, one warm-up and six measured waves. Counting completed passes within the all-six-decoding interval gave 1,635–1,676 target expert uploads per measured wave. The drafter added 32–35. All 42 responses matched the saved checkpoint. The counters measure transfers of model weights. KV rotation stayed at zero throughout the five uninstrumented tuning legs.

The first profiling attempt ran the same release executable directly on the host. One client's output differed from the checkpoint at both 512 and 768 output tokens. The tagged service reproduced the saved 768-token responses in three uninstrumented waves, and the Docker trace matched all six. The C1 Docker trace matched a separate uninstrumented Docker replay at 1,024 tokens. Only output-matched, in-image traces with kernel records contributed to the table. The cause of the host-only discrepancy remains open. [Validation records and output digests](../results/decode/next-optimizations-375w.json)
