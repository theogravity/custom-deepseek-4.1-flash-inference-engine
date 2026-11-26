# Long-context decode at 375 W

The checkpoint before these experiments is the annotated Git tag `stable-2026-11-25-pre-long-context-tuning` at `2d00037`. The parent LLM stack has the same local tag at `d8ac007`, which pins the LiteLLM integration. The local engine Docker image also carries the tag. Its image ID is `sha256:30133989d3bf9cc83ed276c56e3e04a7eb8ed7f4ab55a4949e99b4f76d69ebef`.

## Serving configuration

The deployed configuration uses the new scorer, 6 GiB of host RAM for reusable prefixes and a 375 W limit on each inference card. The table holds the coding output budget at **512 tokens per client** and uses the first one, four or six histories in the same six-history set. The three aggregate columns report medians of six wave-level rates. The individual stream column pools all client rates across those six waves. Each additional client brings its own prompt and greedy continuation.

| Clients at 404k | Aggregate output tok/s | Whole-wave output tok/s | Estimated committed target tok/s with all clients in decode | Median individual stream tok/s |
|---:|---:|---:|---:|---:|
| 1 | 233.3 | 172.0 | 229.8 | 233.3 |
| 4 | 287.6 | 260.4 | 339.6 | 91.4 |
| 6 | 255.8 | 239.3 | 309.6 | 55.3 |

Four clients delivered the highest aggregate rate for these 512-token continuations. Extending the six-client outputs to 768 tokens gave 284.6 aggregate output tok/s and 323.5 estimated committed tok/s during six-way decode. Single-client retrieval on the deployed configuration measured 459.3, 406.0 and 371.5 tok/s at 65,536, 262,144 and 524,160 input tokens respectively. The replay of the published 25,625-token code edit gave 278/422/460 at C1/C4/C6. [Per-wave data](results/long-context-tuning-375w.json) · [Published edit comparison](UPSTREAM-COMPARISON.md)

For short prompts, the three-program C1 check interleaves Python, C and Rust with 1,024 output tokens per request. Over 18 measured requests, a fresh start gave 322.2 tok/s on the tagged checkpoint and 321.9 on the selected build. A separate selected-build pass immediately after the long-context tests gave 309.9. Every program's output digest matched across the three runs. [Short-code samples](results/long-context-tuning-375w.json)

Eight simultaneous full-window requests also completed on the deployed configuration. Each used a 523,776-token shared register with a client-specific receipt and generated 512 tokens. Six sessions stayed resident at peak and two were swapped. The service completed eight RAM save/restore pairs with zero failures. All eight responses matched their serial replays. This was one validation wave; the disk tier was covered in the earlier forced-overflow test. [Eight-client run](results/long-context-tuning-375w.json) · [Forced disk overflow](results/eight-full-600w.jsonl)

## Throughput with the new attention scorer

Each cell is the median following one warm-up. Coding and single-client retrieval use six measured waves. Six-client full-window retrieval uses five measured repeats. The two control columns bracket each candidate. All three legs use a 2 GiB host prefix tier, the same 48 GiB live offload tier and the same 375 W GPU limits. The coding rows report aggregate visible output tok/s. C1 and C4 use 512 output tokens per client, and C6 uses 768. The retrieval rows each generate 128 verified tokens.

| Native workload | Clients | Input per client | Control before | New scorer | Control after |
|---|---:|---:|---:|---:|---:|
| New code | 1 | 404k | 212.2 | 232.9 | 211.8 |
| New code | 4 | 404k | 221.1 | 238.4 | 224.4 |
| New code | 6 | 404k | 231.1 | 252.1 | 227.8 |
| Receipt retrieval | 1 | 65,536 | 445.6 | 458.5 | 445.3 |
| Receipt retrieval | 1 | 262,144 | 372.8 | 404.9 | 372.4 |
| Receipt retrieval | 1 | 524,160 | 321.1 | 371.7 | 322.3 |
| Receipt retrieval | 6 | 524,160 | 111.1 | 118.3 | 111.8 |

Each of the six independent 524,160-token inputs completed its 128-token response, using the full 524,288-token allowance per request. With only 128 output tokens, staggered prefix restores account for a large part of the aggregate decode interval. The scorer passed all 84 release tests, including the real-checkpoint CUDA comparisons. The greedy output hashes matched the checkpoint on every coding and retrieval row.

## Six-client prefix RAM budget

All three legs use the new scorer and the same six code requests. Only the host prefix RAM budget changes. The GPU slot limit is six. Live request offload retains separate budgets of 48 GiB RAM and 128 GiB disk.

| 404k coding metric, six clients | 2 GiB before | 6 GiB | 2 GiB after |
|---|---:|---:|---:|
| Aggregate output tok/s, first text to last text | 249.4 | 284.6 | 248.3 |
| Aggregate output tok/s, whole request wave | 233.3 | 270.5 | 236.5 |
| Estimated committed target tok/s, all six in decode | 325.2 | 323.5 | 327.1 |
| First-response spread, seconds | 3.86 | 1.56 | 4.03 |

With 6 GiB, all six warm-up requests loaded target prefixes from disk. All 36 subsequent restores came from RAM. The surrounding 2 GiB legs served 39 and 38 of their 42 restores from the disk tier, including warm-up. The larger tier reduced staggered admission enough to raise first-text-to-last-text aggregate output throughput by 14–15%. Fully resident throughput varied by about 1% across the three runs. Disk-tier reads can hit the operating system's page cache. [Per-wave measurements and output digests](results/long-context-tuning-375w.json)

On six full-window retrieval requests (524,160 input and 128 output tokens each), the 6 GiB tier delivered **214.0 aggregate output tok/s**, versus 129.0 and 117.4 on the two 2 GiB controls. Including admission, the three medians were 101.1, **167.4** and 94.4 tok/s. All retrieved receipts and full output digests matched across the three legs.

## Workload

Each coding client starts from a distinct 404,000-token conversation. The original exchange retrieved receipts from a long document. A new user turn requests a Python, C or Rust program. The resulting inputs are 404,151–404,170 tokens. Generation uses greedy decoding with reasoning disabled on the native OpenAI chat endpoint. Checks cover the original prompt IDs, actual previous assistant reply, extended prompt IDs, distinct stream IDs, visible coding output, finish reason and exact output count. Output digests enable comparison across builds. A task-specific check recognizes each language and task in the generated prefix. The generated programs are partial at these output limits.

For coding, each number uses six measured waves following one warm-up. Every measured wave must restore the last eligible 64-token checkpoint for each client. At six clients the backend must record a six-request target pass, and all six visible streams must overlap. Aggregate decode throughput divides the generated token count, less one per client, by the interval from the earliest first visible text to the latest last visible text. Request throughput includes admission and prefix restoration from the start of the wave. Individual stream rates use each client's own first and last text events. Client stream rates count output tokens. The optional resident-decode estimate uses committed target positions from the scheduler.

The two inference cards each had a verified 375 W power limit. Their configured P0 offsets were +450 MHz core and +4,000 MHz memory. The CPU socket power cap read 225 W. [GPU settings](results/benchmark-gpu-settings.json) · [CPU readback](results/benchmark-cpu-power.json)

## Attention scorer and drafter experiments

At 524,288 positions the three early index layers each score approximately 262,144 keys, the fourth scores 524,288, and four later layers each score up to 16,384 candidates. That totals roughly 1.38 million key positions for one verification row. With 32 query heads and 128 components per head, the key dot products alone account for about 11.3 GFLOP per row, counting a fused multiply-add as two operations. The existing short-context target-only bandwidth estimate is 154 tok/s at 375 W under sequential stage execution. Speculative and concurrent decode amortize target weight reads and depend on draft acceptance. [Weight traffic and measured bandwidth](THEORY.md)

The new CUDA scorer assigns eight or sixteen physical lanes to each dot-product group and preserves the original FP32 multiply-add and reduction order. In a separate GPU test it reproduced the previous packed scorer bit-for-bit on all 20 combinations of key count, row count and addressing mode. Kernel time fell by 17–45%. For eight rows of 262,081 consecutive keys the median fell from 1.376 ms to 0.763 ms. Each timing sample averaged five launches, with six rotated repetitions per implementation. The release suite also compares the production scorer directly with the scalar reference across masks, gathered keys, masked tails and ties. [Kernel timings](results/long-context-index-kernels.csv)

Sharing the drafter's dense projections across sessions measured 228.2 aggregate coding tok/s at C6, bracketed by 230.9 and 231.1 for the checkpoint. All six outputs matched. The local experiment branch is `experiment/long-context-joint-draft`; its actual-checkpoint projection equivalence fixture passed on the GPUs.

## Reproduce the coding workload

Supply the engine credential through `DSV41_API_KEY`. The first command creates the six histories used to construct the follow-up coding requests. A first run with empty prefix tiers also pays the cold prefill cost.

```sh
uv run --with tokenizers==0.22.2 python tools/six_streams.py \
  --seed 26611502719397 --input 404000 --output 128 --save-histories histories.json
uv run --with tokenizers==0.22.2 python tools/long_decode.py \
  --histories histories.json --clients 6 --output 768 --rounds 6
```

The prefix RAM experiments add `--sample-steady` to the coding command. With this option the harness also polls the native scheduler at 200 ms intervals. It divides the change in committed KV positions by elapsed time over the longest sampled interval in which all clients were resident and decoding. It reports the underlying counts, elapsed time, sample count and longest metrics round trip. A sample needs at least one second of overlap. A resident client can sit out a step under backpressure.
