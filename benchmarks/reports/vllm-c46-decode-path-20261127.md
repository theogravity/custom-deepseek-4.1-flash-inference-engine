# What gives vLLM its C4/C6 decode advantage

## What the local vLLM recipe runs

The current recipe uses full decode-only CUDA graphs, captures the request/token shapes up through `max_num_seqs × (1 + speculative_width)`, and runs the released DSpark draft at `k=5`. For six concurrent sequences, its maximum captured verification shape is 36 tokens. The recipe does **not** enable vLLM's newer adaptive-verification option. [Local launcher](</mnt/models/dsv41-flash-exl3/recipe/serve/serve.sh>)

The strongest measured differentiator is host expert execution. On this host and at 375 W per inference GPU, vLLM measured 485 tok/s with CPU experts disabled and 537–561 tok/s with them enabled. Its CPU use rose from about three to 23 fully occupied cores. The native engine's recent warm C4/C6 controls were 499/504 tok/s: ahead of vLLM GPU-only, but behind vLLM with CPU experts. The local CPU-expert prototype previously lost throughput on six independent 404k contexts, so it needs a careful current-TP2 port and a coding-workload A/B before promotion. [Same-host measurements](concurrent-edits.md#same-host-comparison) · [CPU-expert prototype result](cpu-experts.md)

Full CUDA graphs are another difference. vLLM captures replayable decode execution for its batch shapes; the native engine launches its own CUDA kernels from the Rust worker and has no full target-decode graph capture. A graph implementation could cut launch and scheduling overhead, but the local evidence has not isolated its gain from CPU experts.

## Adaptive verification is a separate opportunity

vLLM's newer DSpark adaptive verifier computes cumulative survival probabilities per request, prices verification by profiled draft/target step time, and chooses a shared draft-token budget. It assigns positions on the GPU and supports variable-length decode graphs. That is materially more than clipping every request to the same fixed number. The [vLLM design note](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md) describes the cost model and GPU allocation. The local vLLM recipe does not enable it, so it does not explain the measured lead in our current comparison.

Hyperflare's merged [PR #515](https://github.com/NVIDIA/hyperflare-inference/pull/515) adds online per-depth confidence calibration and a correction for how GPU top-budget allocation gains from confidence differences across concurrent requests. It reinforces that a fixed cap based on average confidence can misprice crowded heterogeneous batches.

## Native fixed-budget prototype: rejected

I added an opt-in `DS4_BATCH_DRAFT_BUDGET_PER_REQUEST` prototype. It turns each request's conditional confidence scores into cumulative survival probabilities, globally ranks those draft positions, and retains a causal prefix of each request. It intentionally omits vLLM's profiled cost tables and GPU-side allocator, so it is a screening experiment rather than a reproduction of vLLM adaptive verification.

Each arm used the 25,625-token shared coding prompt, 1,500 output tokens at temperature zero, one warm-up wave and six measured waves. The original engine control was restarted before the matched control series so the GPU expert cache began cold in both control and candidate runs. GPUs were capped at 375 W. Every request completed and all reasoning/content hashes matched the fresh control.

| Clients | Fresh control | Shared budget 4/request | Change | Shared budget 6/request | Change |
|---:|---:|---:|---:|---:|---:|
| 4 | 498.8 tok/s | 481.3 tok/s | -3.5% | 459.2 tok/s | -7.9% |
| 6 | 504.1 tok/s | 488.6 tok/s | -3.1% | 462.9 tok/s | -8.2% |

This fixed-budget policy is rejected and stays disabled by default. The results do not disprove adaptive verification: the policy lacks the step-cost model, depth calibration and GPU allocation that make vLLM's method effective. A better next experiment is to profile target/draft costs and implement a device-side causal-prefix allocator, or to forward-port the CPU expert overlap path against the current TP2 sharded cache.

Raw data: [C4 control](../results/decode/c4-shared-budget-restart-control-20261127.jsonl) · [C6 control](../results/decode/c6-shared-budget-restart-control-20261127.jsonl) · [C4 budget 4](../results/decode/c4-shared-budget-4-20261127.jsonl) · [C6 budget 4](../results/decode/c6-shared-budget-4-20261127.jsonl) · [C4 budget 6](../results/decode/c4-shared-budget-6-20261127.jsonl) · [C6 budget 6](../results/decode/c6-shared-budget-6-20261127.jsonl)
