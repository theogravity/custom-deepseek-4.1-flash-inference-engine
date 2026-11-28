# Adaptive shared draft-budget experiment

Date: 2026-11-27. Workload: agentic code-edit scaling replay, 25,625-token prompt, four or six clients, seven waves (one warm-up and six measured), 1,500 generated tokens per client. Reported throughput is aggregate `whole_wave_tok_s`. Both RTX PRO 6000 Blackwell GPUs ran at the user's 375 W cap; the CPU remained limited to 180 W.

## Implementation

`DS4_ADAPTIVE_DRAFT_BUDGET=1` enables an opt-in shared budget selector. It ranks proposed positions by cumulative survival probability, then chooses the budget that maximizes expected tokens divided by observed draft-plus-target step cost. Costs are rolling medians keyed by context bucket and drafter/target request shape. Verification costs are monotonically smoothed and piecewise interpolated. Until enough samples exist for the active shapes, the selector uses the existing confidence threshold. It skips cost learning and shared allocation when a batch contains nonempty lookup proposals, whose expected yield is not represented by the learned confidence curves.

The profile uses live wall-clock timings: grouped learned-draft generation and the whole target `decode_group` including per-request scoring and commit. Failed grouped-drafter attempts that fall back to separate drafting are excluded from draft-cost samples. Confidence-curve collection is separate from diagnostic logging; `DS4_CONFIDENCE_TRACE=1` is required to print per-step score vectors. The feature is disabled by default and cannot be combined with the fixed `DS4_BATCH_DRAFT_BUDGET_PER_REQUEST` experiment.

## Corrected results

| Clients | Stable control | Adaptive budget | Change | Reasoning/content hashes |
|---:|---:|---:|---:|---|
| 4 | 490.7 tok/s | 460.5 tok/s | −6.2% | exact match |
| 6 | 495.6 tok/s | 499.5 tok/s | +0.8% | exact match |

Each result is the median of six scored waves. All requests completed 1,500 tokens and contained the requested function. C4 waves were 452.1, 461.1, 458.2, 461.5, 464.1 and 459.8 tok/s; C6 waves were 490.6, 494.2, 501.5, 501.4, 499.0 and 500.0 tok/s. Corrected raw data is in `../results/decode/c4-adaptive-draft-budget-corrected-20261127.jsonl` and `c6-adaptive-draft-budget-corrected-20261127.jsonl`; controls are `c4/c6-shared-budget-restart-control-20261127.jsonl`.

The first candidate measurements were invalid: adaptive mode collected confidence curves using a flag that also printed every confidence vector, even with diagnostic tracing off. Those raw runs remain labeled `verbose-log-contaminated` and are excluded from the table. A subsequent trace also found lookup-shaped verification timings in the cost profile; lookup batches are now excluded from cost learning.

The clean result is mixed: C4 regresses, and the C6 gain is within normal run variation. The selector remains opt-in and should not be enabled by default. Output hashes match, so the measured difference is throughput rather than output drift. The cost-based policy has not shown a consistent improvement on this workload.
