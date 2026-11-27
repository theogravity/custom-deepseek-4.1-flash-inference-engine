# Why vLLM leads on six concurrent 25.6k code edits

The published Python-edit workload ran on two RTX PRO 6000 Blackwell Workstation cards at 375 W each with a measured 225 W CPU socket limit. Each of six clients requested a different edit of the same 25,625-token module, with temperature zero, default reasoning, and a 1,500-token output budget. The two servers used their existing configurations and the same base pack. The native server uses the code-tuned DSpark overlay with up to seven learned proposals and source lookup. vLLM uses its released DSpark head with five proposals and full decode CUDA graphs. [Published methodology](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/362838bb8e5ae683760c96a13c4389cb7770671f/docs/BENCHMARKS.md#cpu-host-expert-runtime-ab-same-server-otherwise) · [Published client](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/362838bb8e5ae683760c96a13c4389cb7770671f/bench/decode_bench2.py) · [Per-wave evidence and hashes](../results/agents/edit-scaling-375w.json)

## Same-host comparison

| Run order | Aggregate decode tok/s | Whole-wave tok/s | Accepted draft tokens per proposal round | Per-session proposal rounds/s, summed | CPU cores used |
|---|---:|---:|---:|---:|---:|
| Rust/CUDA before | 461.57 | 454.69 | 4.81 | 79.44 | 1.03 |
| vLLM CPU experts on | 561.03 | 546.15 | 3.85 | 115.44 | 22.67 |
| vLLM CPU experts off | 484.98 | 475.92 | 3.88 | 100.01 | 3.13 |
| vLLM CPU experts on again | 537.20 | 523.88 | 3.83 | 112.60 | 22.76 |
| Rust/CUDA after | 462.80 | 455.74 | 4.81 | 79.68 | 1.03 |

Each leg started a fresh server, ran one warm-up wave, and measured the next six. The decode column follows the published script and divides 9,000 completion tokens by the longest individual first-token-to-end duration. Both reasoning and visible output count. Whole-wave throughput includes admission. Each cell is the median of six waves. CPU usage is container CPU time divided by elapsed time, expressed in fully occupied cores. It includes workers and helper processes. The proposal-round column uses individual sequence counters and the same decode time denominator. Multiple sequence rounds can execute in one GPU batch.

The vLLM image and settings were held fixed across the three middle legs except for `EXL3_CPU_EXPERTS`. Host prefill experts were disabled throughout. GPU-only vLLM delivered 485.0 tok/s. CPU expert execution raised delivered throughput by 52.2–76.0 tok/s, or 10.8–15.7%. The GPU-only result exceeded the two native controls by 22.2–23.4 tok/s. These rate differences describe this workload and include the effect of the continuations each stack generated. Estimating a native CPU backend’s gain requires measuring which native expert uploads could overlap other work.

The reference schedules host-resident, uncached experts on two ten-thread CPU pools while both GPUs process their resident experts. Logs confirmed that both pools processed jobs without watchdog rescues or errors in the captured summaries. The CPU implementation uses two-pass int8 activations. It changes the numerical path. Our native engine loads missing packed expert weights to GPU and executes its projections there. [CPU expert implementation and precision](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/6110453b901e4b51da66f539f3802b5c7f18e8e8/recipe/cpuexp/README.md) · [Native upload](../../native/moe.cu)

All timed responses finished at 1,500 tokens and contained the first requested Python declaration. At temperature zero, each client had one output digest per native control and six distinct response digests in each vLLM leg. Cross-engine paired waves had zero matching visible-code or reasoning hashes out of 36. vLLM’s two CPU-on controls also differed across all 36 paired responses. The vLLM timings varied across repetitions, especially immediately after warm-up. The benchmark verifies the presence of the requested edit and records text identity. Code correctness and quality require separate evaluation.

## Flare comparison

Local Inference Lab’s MXFP4 release reports a single-session Tetris peak of 1,076 tok/s and sustained end-to-end averages of 703–766 tok/s. That package uses four RTX PRO 6000s, tensor parallelism and RAM-resident Engram tables for those figures. Flare overlaps indexer and query decode on an auxiliary CUDA stream and tunes its PCIe collectives. The timed release used seven draft positions with adaptive verification. A later scheduler change, PR #515, calibrated forecasts at each draft depth against accepted tokens and improved the estimate for sharing verification slots across requests. That change postdates the reported Tetris peak. The native five-proposal experiment below tests a fixed maximum on our own six-client edits. [Flare benchmark revision](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE/blob/2f5e650ab0616ea54140e502d91ef3620d8bc0cb/README.md) · [Flare scheduler implementation](https://github.com/NVIDIA/hyperflare-inference/pull/515)

## Native experiments

The expert counters were collected separately from the throughput runs. The same six edits warmed the cache, then ran twice more and matched the control output. The two warm waves recorded **2 and 0 expert uploads** across target and draft combined (25.4 and 0.0 MiB over the whole waves). The initial wave loaded 153.8 GiB. Offloading cold experts to the CPU has little work to take over once this particular native workload is warm. The earlier six-client 404k profile measured 2.78–2.82 GiB/s with its then-current eviction policy. Miss rates depend on the active histories, available GPU slots and eviction policy. [Long-context transfer profile](expert-transfers.md)

| Native draft maximum | Combined decode tok/s | Accepted tokens/round | Proposal rounds/s, summed |
|---|---:|---:|---:|
| Seven, before | 460.3 | 4.81 | 79.2 |
| Five | 421.2 | 3.99 | 84.4 |
| Seven, after | 461.4 | 4.81 | 79.4 |

With the maximum reduced to five learned proposals, rounds ran faster but accepted fewer tokens. Combined throughput fell 8.5–8.7% against the two seven-proposal controls. All 36 paired five-versus-seven responses matched on both visible output and reasoning. The normal service retains seven. Source lookup can propose more tokens than the learned maximum, and the existing per-request confidence threshold remained at 0.40.

The CUDA microbenchmark varied the thread block between two, four and eight warps for the fused target gate/up kernel. It used valid synthetic packed weights, fixed arithmetic, eight split-K partials and a byte comparison against four warps. It checked routed job sizes covering 1, 8, 9, 15 and 16 rows. Each timed cell is the median of nine samples, each averaging 40 launches on host GPU 1. The analyzer rotated the order of all three widths. Common C6 shapes from the previous trace were selected before timing.

| Expert jobs | Two warps, µs | Four warps, µs | Eight warps, µs |
|---:|---:|---:|---:|
| 4 | 21.2 | 21.1 | 21.1 |
| 16 | 127.3 | 115.7 | 122.8 |
| 30 | 282.2 | 278.8 | 287.4 |
| 32 | 313.2 | 303.6 | 306.7 |
| 34 | 334.5 | 334.1 | 331.6 |
| 48 | 491.4 | 471.9 | 459.5 |

Four warps won at 30 and 32 jobs. Eight warps were 0.7% faster at 34 jobs. Neither alternative met the screening requirement of a 2% improvement at each of these three common shapes. The four-warp production kernel was retained.

## GPU time during the six edits

A separate Nsight Systems capture traced 1.87 seconds of the warm edit wave. Scheduler samples bracketed the entire window with all six clients decoding. The visible code and reasoning hashes from the profiled wave matched the native control for every client. Both GPUs executed recorded kernels for **15.9%** of wall time. Exactly one executed kernels for **67.1%**, and neither had a recorded kernel for **17.0%**. A gap can include copies, synchronization, CPU work or profiler overhead.

| Kernel family | Share of summed kernel time on both devices | Wall time with only this family on exactly one GPU |
|---|---:|---:|
| Routed expert projections | 38.0% | 7.1% |
| Packed dense projections | 17.7% | 17.5% |
| `DeviceSegmentedRadixSortKernel` | 12.3% | 12.2% |
| `stable_dot` | 4.6% | 4.5% |
| `fast_qk` | 4.5% | 4.4% |

The kernel shares use summed device time. The wall-time column counts only intervals with one GPU executing that family and no other recorded kernel family on either device. The capture recorded 2 complete H2D copies matching the two packed expert upload sizes in the analysis window. The profiler trace measures native execution only. The rates in the tables above came from separate uninstrumented runs.

Nsight reported potentially incomplete events at capture stop. The analysis clips activity to the central 90% of the observed kernel interval.

The most localized next experiment is in the attention index. The short-row path sorts all rank keys and then keeps a subset. Its segmented radix-sort kernel ran alone on one GPU for 12.2% of observed wall time. Benchmark exact batched top-k selection at those sizes, preserving causal masking, tie-breaking and chronological output order. [Current sort and top-k dispatch](../../native/attention_index.cuh)

Flare also suggests testing overlap between independent indexer and query projections before the attention launch. Dense projections ran alone on one GPU for 17.5% of observed wall time. The trace groups multiple projection sites together, so instrument the individual sites before choosing which to overlap. The other Flare experiment is a shared verification budget allocated by measured acceptance at each draft depth, retaining seven as the maximum. [Attention projections](../../src/attention.rs) · [Flare scheduler](https://github.com/NVIDIA/hyperflare-inference/pull/515)

## Reproduction

Run the following from the engine repository against an idle authenticated server with its credential in `DSV41_API_KEY` or `API_KEY`. The command emits one JSON object per wave with per-stream timing, token counts and content digests. The reference legs used the original vLLM Compose file with CPU experts set to 1, then 0, then 1. A separate LMCache test directory preserved the regular store. The native controls used the shared-prefix LFU profile. The native width experiment set `DS4_SPEC_K=5` only on its middle leg.

```sh
python3 tools/edit_scaling.py --clients 6 --waves 7
```

The raw per-wave JSONL, GPU samples, Nsight report, CUDA microbenchmark source and its kernel patch are archived on the benchmark host under `~/.cache/ds4-flash-rs/profiles/edit-scaling-0fb9857/`. The tracked results include input and output hashes, resource samples’ hashes, and the Nsight report hash.
