# Adaptive shared draft-budget experiment

Date: 2026-11-27. Workload: agentic code-edit scaling replay, 25,625-token prompt, four or six clients, seven waves (one warm-up and six measured), 1,500 generated tokens per client. Reported throughput is aggregate `whole_wave_tok_s`. Both RTX PRO 6000 Blackwell GPUs ran at the user's 375 W cap; the CPU remained limited to 180 W.

## Implementation

`DS4_ADAPTIVE_DRAFT_BUDGET=1` enables an opt-in shared budget selector. It ranks each proposed token by cumulative survival probability, then chooses the budget that maximizes expected tokens divided by observed draft-plus-target step cost. Costs are rolling medians keyed by context bucket, draft batch shape, verification row count, and request count. Verification costs are monotonically smoothed and piecewise interpolated. Until the profile has repeated samples for the active shapes, the selector uses the existing confidence threshold. It skips global allocation on batches that contain nonempty prompt/output lookup proposals, whose expected yield is not represented by the learned confidence curves.

The profile uses live wall-clock timings: grouped learned-draft generation and the whole target `decode_group` including per-request scoring/commit. Request count is included in the target-cost key to account for that per-request work. Failed grouped-drafter attempts that fall back to separate drafting are excluded from draft-cost samples. The feature is disabled by default and cannot be combined with the fixed `DS4_BATCH_DRAFT_BUDGET_PER_REQUEST` experiment.

## Results

| Clients | Stable control | Adaptive budget | Change | Reasoning/content hashes |
|---:|---:|---:|---:|---|
| 4 | 490.7 tok/s | 467.0 tok/s | −4.8% | exact match |
| 6 | 495.6 tok/s | 454.3 tok/s | −8.3% | exact match |

Each value is the median of six scored waves. All requests completed 1,500 tokens and contained the requested function. The adaptive results were 469.5, 479.0, 418.7, 483.4, 464.5 and 428.4 tok/s at C4; 460.9, 439.4, 446.8, 455.1, 453.4 and 457.1 tok/s at C6. Raw per-request data is in `../results/decode/c4-adaptive-draft-budget-20261127.jsonl` and `c6-adaptive-draft-budget-20261127.jsonl`; controls are the adjacent `c4/c6-shared-budget-restart-control-20261127.jsonl` files.

The cost-aware selector did not improve this workload and is rejected for default use. Keep it opt-in for further diagnosis only. Its output was deterministic against the controls, so the regression is throughput rather than output drift. The measured cost objective did not predict the best proposal budget for this workload; further work should inspect the per-step budget/acceptance distribution before changing the cost model or adding online calibration.
