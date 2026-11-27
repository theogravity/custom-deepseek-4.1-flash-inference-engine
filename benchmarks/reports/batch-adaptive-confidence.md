# Batch-aware speculative confidence

The native engine trailed the installed vLLM CPU-expert configuration by 6–10% on the six-client coding edit, while its single-client rate was already higher. A global draft-confidence threshold of 0.5 improved C6 but reduced C1. This experiment applies the existing 0.4 threshold to one active decode session and raises it by 0.1 only when multiple requests are decoding together.

Each leg replayed the upstream 25,625-token Python edit with a 1,500-token completion at temperature zero. It used one warm-up and six measured waves, the same 375 W GPU caps, the same 8k prefill and LFU/peer-index profile, and the same server process across client counts. The control threshold was 0.4 for every request. The adaptive leg used 0.4 for C1 and 0.5 for C4/C6. The benchmark checked every completion length and requested function. Reasoning and visible-output hashes matched between control and adaptive results for every function at each concurrency.

| Clients | Threshold 0.4 control | Batch-aware threshold | Change |
|---:|---:|---:|---:|
| 1 | 408.5 tok/s | **410.0 tok/s** | +0.37% |
| 4 | 495.0 tok/s | **497.8 tok/s** | +0.56% |
| 6 | 499.1 tok/s | **503.8 tok/s** | +0.93% |

The C6 median remains 9.5% below the published 557 tok/s vLLM figure, which used 600 W GPU caps. On the same host at 375 W, the prior vLLM CPU-expert-on controls measured 537–561 tok/s; the adaptive native result is still 6.2–10.2% behind them. It is 3.9% faster than the same-host vLLM CPU-expert-off median of 485 tok/s. [Same-host control details](concurrent-edits.md) · [Published comparison details](prefix-fix-benchmark-20261127.md)

## Rejected variants

The fixed speculative width was raised from the production 7 to 8 and 9. Neither helped C6: medians were 496.4 and 487.0 tok/s. A global confidence threshold of 0.3 measured 479.2 tok/s at C6; global 0.5 measured 505.1 at C6 but 398.3 at C1 versus the matched 408.5 control. A batch delta of 0.2 measured 504.1 tok/s at C6, effectively tied with the smaller 0.1 delta. The 0.1 batch-aware delta is retained because it preserves the single-session behavior and gave the best measured balance across client counts.

All candidate and control waves are in [decode results](../results/decode/). The runner was `tools/edit_scaling.py --clients N --waves 7`.
