# Shared packed KV rows during prefix restoration

A warm reusable target image contains about 695 MiB at 404k tokens. The Rust decoder now gives the four compressor layers immutable views of the eight KV and index byte ranges in that image. GPU installation reads each range directly. CPU appends detach only the range being modified. The last range releases its image reference when GPU ownership transfers or the request ends. Both builds used LFU for expert eviction. [Implementation](../../src/attention_snapshot.rs) · [Per-wave results, restore events and output hashes](../results/context/shared-prefix-375w.json)

## Warm throughput

| Clients × input tokens, output per client | LFU before, delivered tok/s | Shared rows, delivered tok/s | LFU after, delivered tok/s | Whole-wave tok/s, before / shared / after |
|---|---:|---:|---:|---:|
| 1 × 404k, 512 output | 234.2 | 234.2 | 233.8 | 174.3 / 185.1 / 173.2 |
| 6 × 404k, 512 output | 278.2 | 296.6 | 278.8 | 261.7 / 280.4 / 261.3 |
| 6 × 404k, 768 output | 300.1 | 313.2 | 302.7 | 285.3 / 301.7 / 288.4 |
| 6 × 524,160, 128 output | 202.7 | 287.7 | 204.4 | 159.9 / 222.0 / 160.3 |

Each build served one warm-up and six measured waves per workload at 375 W per inference GPU. The CPU socket cap read 225 W. Each row reports medians. The coding histories had 404,151–404,170 tokens apiece. The full-window receipt requests consumed exactly 524,288 positions each. Every coding and receipt response, including warm-ups, matched its previously qualified response digest. Every measured six-client coding wave reused 2,424,704 prefix tokens, reached six-way verification and completed with zero live swaps. The first 512-token control began during candidate build preparation. The trailing control ran with the build finished. [Frozen workloads and benchmark commands](long-context.md#reproduce-the-coding-workload)

Delivered tok/s counts the tokens after each stream’s first visible token, from the first client’s first text to the last client’s last text. Whole-wave tok/s counts every generated token and includes admission. For example, the initial 512-token control’s 278.2 aggregate tok/s gives 46.4 tok/s per client over the common interval. Over each client’s own streaming interval, the six medians ranged from 53.1 to 71.8 tok/s. The shared build exceeded 300 delivered tok/s in 0/6 waves at 512 output and 6/6 at 768.

## Where the time went

For the 512-token coding waves, first-text spread fell from 1.40 and 1.43 seconds in the controls to 0.68 seconds. The median committed rate over the interval when all six clients were decoding was 340.1 / 340.3 / 341.3 tok/s in run order. The gain came from getting all six clients through restoration sooner. At one client, whole-request throughput improved while the client’s streaming rate stayed near 234 tok/s.

A separate instrumented run started each server fresh, restored six coding requests, and repeated the wave. The six RAM events per build gave the following per-request medians. GPU installation measures host time spent in allocation and upload calls. Wall-to-ready can include other scheduler work. [Restore timing instrumentation](../../src/restore_timing.rs)

| Per warm target restore | LFU control | Shared rows |
|---|---:|---:|
| RAM lookup, ms | 1.32 | 1.55 |
| Target deserialization, ms | 136.61 | 0.23 |
| GPU installation calls, ms | 64.98 | 64.97 |
| Wall-to-ready, ms | 849.10 | 274.63 |

Peak sampled container anonymous memory during the six-client 512-token workload was 8.71 / 5.64 / 8.57 GiB. Polling was approximately once per second. A live request can hold an evicted prefix image until the last view releases it. The 6 GiB prefix RAM budget counts cache entries. This process measurement also includes retained images and other anonymous allocations, and excludes filesystem page cache.

## Published 25.6k code-edit workload

The unchanged upstream script reported a 25,625-token warm-up prompt. It requests up to 1,500 tokens per client while editing a common Python module, and divides summed completion tokens by the longest individual decode duration. Our second-pass six-stream scores were 458 tok/s for LFU and 464 for shared rows. These are individual script passes. The upstream report lists 600 W for its later results and says GPU settings were unrecorded for its November 9 runs. The upstream publication reports 490 GPU-only and 574 with CPU decode experts in its ablation, and 557 at six streams in its final configuration. Its six-by-404k section reports retrieval wall time. [Published methodology](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/main/docs/BENCHMARKS.md#methodology-and-pitfalls) · [Exact script](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/362838bb8e5ae683760c96a13c4389cb7770671f/bench/decode_bench2.py)

## Full-window rotation and reproduction

With RAM as the live offload tier, eight clients each used 523,776 input and 512 output tokens. The scheduler reached 8 active sessions, 6 GPU residents and 2 offloaded. The engine completed 8 RAM writes and 8 reads, with 0 I/O failures. Every output matched both the serial replay and the saved checkpoint. The wave took 24.8 seconds.

With disk as the live offload tier, eight clients each used 523,776 input and 512 output tokens. The scheduler reached 8 active sessions, 6 GPU residents and 2 offloaded. The engine completed 8 disk writes and 8 reads, with 0 I/O failures. Every output matched both the serial replay and the saved checkpoint. The disk test set the live RAM tier to zero to force disk traffic. The wave took 36.1 seconds.

The deployed profile allows six residents, 32 logical sessions, 48 GiB RAM and 128 GiB disk for live KV. The reusable prefix tiers have 6 GiB RAM and 64 GiB disk. The checkpoint’s existing prefix files were used across image switches. All 86 release tests passed with both designated GPUs visible, including the shared-prefix GPU handoff and two consecutive live rotations. Results cover warmed greedy text workloads on this host. The 404k vLLM publication supplies retrieval elapsed times, so a vLLM decode rate on these same independent coding prompts remains unmeasured.

The measured executable’s SHA-256 begins with `ac4518d5`. After the GPU test was extended, a rebuild of the committed source matched all 33 allocatable ELF sections apart from the build ID. Per-wave logs are archived under `~/.cache/ds4-flash-rs/profiles/shared-prefix-3d8c541/` on this host.

Build the historical image from the public revision corresponding to workstation commit `3d8c541`. Run these commands from the root of the main checkout. The subshell builds the image inside a separate worktree. [Revision mapping](../source-revisions.tsv)

```sh
git worktree add --detach ../ds4-shared-prefix-build 6e5cbe4be80f4901651bbddc8ec4f03783ea9873
(cd ../ds4-shared-prefix-build && CARGO_TARGET_DIR=target cargo build --release --locked && docker build -t ds4-flash-rs:shared-prefix-3d8c541 .)
```

Launch the tested configuration from the main checkout. The LFU-only profile remains available for replaying the control.

```sh
docker compose --env-file ../models/.env -f compose.yml -f deploy/profiles/shared-prefix-lfu.yml up --no-build -d --wait
```
