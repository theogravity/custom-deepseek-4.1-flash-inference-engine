# Expert cache eviction at 375 W

Six concurrent agents used independent 404k prompts on the native chat API. Frequency-based expert eviction raised delivered throughput at the same VRAM budget. The single-client 404k workload stayed near 234 tok/s. The candidate is available as an opt-in Compose profile.

| 404k coding workload | LRU before | LFU | LRU after | LFU whole-wave tok/s | LFU committed tok/s while all clients decoded |
|---|---:|---:|---:|---:|---:|
| 6 clients × 512 output tokens | 263.8 | 278.1 | 258.6 | 260.9 | 340.5 |
| 6 clients × 768 | 285.0 | 300.0 | 292.1 | 285.9 | 345.3 |
| 6 clients × 768, independent repeat | 287.1 | 297.0 | 285.7 | 284.3 | 341.5 |
| 1 client × 512 | 233.8 | 234.7 | 233.2 | 175.0 | 231.7 |

The two six-client LFU medians at 768 output tokens were **300.015** and **296.984 tok/s**. Individual waves ranged from 277.4 to 304.2, with 3 of 12 above 300. Delivered throughput spans the first visible token across the group to the last. Whole-wave throughput includes admission and prefix restoration. The resident estimate uses scheduler positions while all clients were decoding. Each rate is a median of six measured waves after one warm-up.

## Switching from coding to full-window retrieval

Each leg began with two six-client 404k coding waves. The same service then handled six 524,160-token requests for receipts, each with a 128-token output budget. These requests extend the same original histories with distinct new receipts. The first wave includes the change of task and prefix admission. The next six waves repeat the retrieval workload.

| Aggregate delivered tok/s | LRU before | LFU | LRU after |
|---|---:|---:|---:|
| First wave after the switch | 84.2 | 103.8 | 109.8 |
| Median of the next six waves | 208.5 | 208.1 | 213.7 |

All six requests used the full 524,288-token window and matched the checkpoint responses. In a separate LFU overflow run, eight clients each used 523,776 input tokens and 512 output tokens. Peak residency was 6 on GPU and 2 offloaded. There were 8 RAM writes and 8 RAM reads. All eight outputs matched their serial replays and the saved checkpoint. This run exercised RAM offload. The earlier forced-overflow run covers the disk tier. [Disk overflow evidence](results/eight-full-600w.jsonl)

## Running the candidate on this workstation

The normal service uses the checkpoint image and LRU. The opt-in profile selects the locally tested image and enables LFU. From this repository, launch it with the command below.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.expert-cache-lfu.yml up --no-build -d --wait
```

To return to the checkpoint service, run the base Compose definition.

```sh
docker compose --env-file ../models/.env -f compose.yml up --no-build -d --wait
```

The measured image contains the LFU implementation from commit `3209b98`. After the GPU test fixture was updated, a clean release rebuild produced a different file digest. All loaded code and data sections matched byte-for-byte apart from the ELF build ID. The Compose override selects the measured image. The cache learns from all requests until the process restarts. The timing runs repeatedly served the same coding prompts, and the task-switch check used one receipt workload. Long-running agents with frequent changes of repository or task can produce different expert frequencies.

## Expert traffic and replay

Seven waves used the same six independent coding continuations at 404,151–404,170 input tokens per agent. A diagnostic build recorded the GPU, layer, routed expert, physical dispatch group, resident slot count, capacity and actual hit or miss. The two traces contain 3,004,717 expert accesses in 142,426 groups. Final engine counters matched both files, and replay reproduced every LRU hit and miss. A preliminary capture guided policy selection. A repeat interrupted by the memory watchdog was discarded. The final validated capture and live timing runs completed.

| Policy at the recorded slot budget | Median uploads per warm six-client wave |
|---|---:|
| LRU (current release) | 2,676 |
| Segmented LRU, 25% protected | 2,667 |
| Segmented LRU, 50% protected | 2,621.5 |
| Segmented LRU, 75% protected | 2,402 |
| LFU with dynamic aging | 1,557 |
| LFU with per-device lifetime counts | 1,191.5 |
| Farthest next use, with future accesses supplied | 771 |

Each replay started empty, included the warm-up and used the same recorded per-access slot budget. It protected each acquired slot through the end of its CUDA group. The median counts only the six measured waves. Actual capacities can diverge with a different live policy, since a miss can trigger a fresh free-memory query.

Only five of the 16,137 warm LRU misses were first accesses. On CUDA ordinals 0 and 1, the median number of distinct keys accessed since a missed expert's previous use was 6,552 and 6,657. The median resident slot counts on misses were 6,338 and 6,318. Giving frequent experts priority retained many of those entries across gaps longer than the LRU cache.

In a separate six-wave run, the LFU engine recorded a median of **1,191 uploads** per warm wave. Compared with the instrumented LRU control's 2,676, the reduction was **55.5%**, or **18.4 GiB per wave** at 13,315,584 bytes per expert. The timing runs disabled both traces and per-pass logging. LFU counts accesses per device, layer and expert over the lifetime of the process, including accesses before eviction. It evicts the lowest-frequency available slot, breaking ties by least-recent access. The current implementation scans resident slots on a miss.

## Reproducibility

The local tag `checkpoint-2026-11-26-before-expert-cache` identifies engine commit `38f4fc6`, parent integration commit `d8ac007` and the matching pre-experiment Docker image. The trace recorder and standalone C++17 replay tool are on `experiment/expert-cache-replay` at `f42b9de`. The original per-device trace files and wave boundaries are in `~/.cache/ds4-flash-rs/profiles/expert-cache-38f4fc6/`. To reproduce the replay from that revision on this host, run the following commands.

```sh
git show 01f95dc:tools/replay_expert_cache.cpp > /tmp/replay_expert_cache.cpp
c++ -O3 -std=c++17 /tmp/replay_expert_cache.cpp -o /tmp/replay_expert_cache
trace_dir="$HOME/.cache/ds4-flash-rs/profiles/expert-cache-38f4fc6"
for card in 0 1; do
  /tmp/replay_expert_cache "$trace_dir/gpu-$card.bin" "$trace_dir/windows.tsv"
done
```

The native throughput comparisons used the same Rust/CUDA executable in both configurations, two RTX PRO 6000 Blackwell cards capped at 375 W each and a measured 225 W CPU socket limit. Each leg started a fresh service and ran one warm-up followed by six measured waves. All measured coding waves had overlapping streams, the expected prefix reuse and no KV rotations. For the same workload, use the frozen histories and `tools/long_decode.py` commands from the long-context benchmark. [Workload](LONG-CONTEXT-BENCHMARK.md#reproduce-the-coding-workload) · [Measurements and trace digests](results/expert-cache-policy-375w.json)
