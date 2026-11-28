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

## CUDA-graph replay experiment (2026-11-27)

A narrow graph around each routed MoE FFN group was added as an opt-in path (`DS4_MOE_CUDA_GRAPH=1`); CPU router-score reads and per-layer dispatch remain eager. The graph uses a dedicated nonblocking stream with default-stream event handoffs, pinned host descriptors, and eager fallback while a prior graph owns the staging buffers. Trace logging is disabled unless `DS4_MOE_CUDA_GRAPH_TRACE=1`. The feature remains off by default.

| Experiment | C4 aggregate tok/s | C6 aggregate tok/s | Output hashes | Decision |
|---|---:|---:|---|---|
| Stable GPU-only control | 490.7 | 495.6 | baseline | keep |
| MoE FFN CUDA-graph groups | 488.8 | 498.3 | exact match | no reliable improvement; keep opt-in only |

Each result uses one warm-up and six measured waves, 1,500 generated tokens per client. All scored requests completed and matched reasoning/content hashes with their controls. Raw waves are in `../results/decode/c4-moe-cuda-graphs-20261127.jsonl` and `../results/decode/c6-moe-cuda-graphs-20261127.jsonl`. Candidate changes: `DS4_MOE_CUDA_GRAPH`/trace toggles and graph capture around expert FFN groups. The small C6 delta (+0.5%) is within ordinary run variation and does not justify enabling this path by default. This result does not test full-step replay.

## Variable-length scheduler assessment

The target batch already supports ragged verification lengths: `plan_decode` records `proposal.len() + 1` rows per request, `forward_batch` concatenates each request's actual history, and the attention batch passes each span's own row count. Confidence gating can also shorten individual DSpark drafts before packing. The missing piece is the *adaptive shared budget* that chooses how many draft positions to schedule across a cohort using calibrated survival and measured draft/verification costs.

A simple shared top-budget prototype exists, but the 4- and 6-token-per-request variants both reduced throughput (see `vllm-c46-decode-path-20261127.md`). So I do not recommend coupling a new allocator to the narrow MoE graph path: that graph did not improve C4/C6 reliably, and the service still reads router scores on the host between transformer layers, preventing full decode-step capture. vLLM's variable-length graph design assumes attention backends support always-capturable variable query lengths and profiles verification cost by token count; its design note describes those prerequisites and the shared cost objective. [vLLM adaptive verification design](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md)

The standalone cost-aware selector is implemented and measured in [the adaptive draft-budget report](adaptive-draft-budget-2026-11-27.md). It regressed C4 and C6, so it remains disabled by default. Full-step CUDA graph capture remains a separate prerequisite-heavy effort because of host router synchronization and attention capture support.

## What transfers from Hyperflare

Hyperflare PR #515 combines per-depth calibration with a correction for surplus from packed, variable-length allocations. The calibration half alone did not help this fixed code-edit replay. The surplus correction is aimed at crowded rounds where requests get unequal draft budgets; the local cost-aware selector already creates such ragged batches but regressed throughput. Revisit Hyperflare's correction only after identifying why the local scheduler's cost estimates and realized tokens-per-step diverge.

References: [vLLM decode-only CUDA graph guidance](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md), [vLLM DSpark adaptive verification](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md), [Hyperflare calibration and allocation-surplus PR #515](https://github.com/NVIDIA/hyperflare-inference/pull/515).
