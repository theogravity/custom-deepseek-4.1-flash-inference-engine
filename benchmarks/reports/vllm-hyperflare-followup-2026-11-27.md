# vLLM and Hyperflare follow-up: C4/C6 decode

Date: 2026-11-27. Workload: the repository's agentic code-edit scaling replay, four or six clients, seven waves (one warm-up and six measured), 1,500 generated tokens per client. Reported throughput is aggregate `whole_wave_tok_s`. The native server was rebuilt from the working tree for each experiment. GPU power cap was left at the user's benchmark setting; CPU expert work was isolated from the calibration arm.

## Results

| Experiment | C4 aggregate tok/s | C6 aggregate tok/s | Decision |
|---|---:|---:|---|
| Stable GPU-only control | 490.7 | 495.6 | Keep as baseline |
| CPU-expert overlap port, CPU path on | 457.8 | 499.7 | Reject; disabled |
| Hyperflare-style depth calibration, CPU path off | 468.0 | 483.9 | Reject; removed |

Control results are the six measured waves in `c4-shared-budget-restart-control-20261127.jsonl` and `c6-shared-budget-restart-control-20261127.jsonl`. Experiment data, including every wave and per-request completion fields, is in the adjacent `c4/c6-cpu-experts-20261127.jsonl` and `c4/c6-depth-calibration-20261127.jsonl` files.

The CPU route was exercised during its run. It regressed whole-wave C4 by 6.7% and was effectively flat at C6 (+0.8%), so the vLLM CPU-expert result did not transfer to this engine's TP2 sharded cache/dispatch path. Keep this approach out of the default build until its dispatch and synchronization costs are addressed.

Calibration regressed whole-wave C4 by 4.6% and C6 by 2.4%. It updates a bounded EWMA of predicted versus observed draft survival and uses that ratio to alter later draft widths. This workload did not benefit from a process-wide calibration signal. It was removed rather than left as an experimental runtime toggle. The benchmark build briefly logged confidence vectors; that run was discarded and repeated after logging was decoupled from calibration. The values above are from the clean repeat.

Earlier status messages quoted the separate `upstream_tok_s` metric (C4/C6 CPU arm 464.9/504.6 tok/s); the table consistently uses `whole_wave_tok_s`, measured from earliest first output through latest last output. All scored waves completed all client requests, generated 1,500 tokens each, and included the requested function call. The CPU route may produce different text because its host floating-point kernel is approximate; throughput is not a quality-equivalence claim.

## What transfers from vLLM

The strongest remaining vLLM mechanism to test is full CUDA-graph replay for steady-state decode. The current native decode path launches many small kernels and uses host-side scheduling between model stages; vLLM captures the decode step and replays it with stable buffers. vLLM documents `FULL_DECODE_ONLY` capture specifically to speed decode while preserving memory for KV. Its current DeepSeek V4 attention path also has dedicated sparse MLA backends. The native server has neither equivalent full-step CUDA-graph replay nor a vLLM-style graph-size grid, so launch and host scheduling overhead is a plausible source of the residual C4/C6 gap.

The next implementation experiment should first establish which native decode stages are capture-safe (no per-step host reads, allocations, or variable shapes), then capture one warmed C4 and C6 graph shape. Measure graph replay with identical requests and verify output parity before expanding shape coverage. This is a larger architectural step than the rejected CPU/calibration toggles.

## What transfers from Hyperflare

Hyperflare PR #515 combines per-depth calibration with a correction for surplus from packed, variable-length allocations. The calibration half alone did not help this fixed code-edit replay. The surplus half is aimed at crowded rounds where requests get unequal draft budgets; our current fixed-width speculative path does not yet have the same packed causal-prefix allocator, so copying that correction alone would be mathematically ungrounded. Revisit it only alongside a measured variable-length packed scheduler and per-request calibration data.

References: [vLLM decode-only CUDA graph guidance](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md), [vLLM DSpark adaptive verification](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md), [Hyperflare calibration and allocation-surplus PR #515](https://github.com/NVIDIA/hyperflare-inference/pull/515).
