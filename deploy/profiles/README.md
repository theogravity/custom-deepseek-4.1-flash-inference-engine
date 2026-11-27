# Compose profiles

The root [compose.yml](../../compose.yml) now enables the measured recommended TP2 coding defaults directly. Use it on its own for a local build, or add one of the overlays below with `-f` to reproduce an earlier or experimental image/configuration. The recommended profile overlay selects the pinned checkpoint image used for published benchmark results.

| Profile | Overlays after the base | Build and measurements |
|---|---|---|
| **Recommended TP2 coding profile** | [tp2-prefill.yml](tp2-prefill.yml) | Enables the promoted decode/coding settings and peer-index split; 68 GiB MoE cache, 524k context. [Full profile results](../../benchmarks/reports/default-profile-headline-20261127.md) |
| Concurrent coding with deferred cold prefill | Same three concurrent coding overlays, then [cold-prefill.yml](cold-prefill.yml) | [Cold-prefill experiment and build](../../benchmarks/reports/cold-prefill-optimization.md#build-and-activate) |
| Concurrent coding release | [decode-round2.yml](decode-round2.yml), [coding.yml](coding.yml), [concurrent-coding.yml](concurrent-coding.yml) | [Concurrent decoding](../../benchmarks/reports/concurrent-repetition.md) |
| Previous coding release | [decode-round2.yml](decode-round2.yml), [coding.yml](coding.yml) | [Coding release](../../benchmarks/reports/agent-followups.md#build-and-rollback) |
| Previous coding release with idle prefill | [decode-round2.yml](decode-round2.yml), [agentic.yml](agentic.yml) | [Tool-return latency](../../benchmarks/reports/tool-return.md#capacity-and-deployment) |
| Decode round two | [decode-round2.yml](decode-round2.yml) | [Round two](../../benchmarks/reports/decode-round-2.md#build-and-rollback) |
| Single-client output drafting | [c1-optimized.yml](c1-optimized.yml) | [Single-client decode](../../benchmarks/reports/single-client-decode.md#run-reproduce-or-roll-back) |
| Tiled attention ranking | [decode-optimized.yml](decode-optimized.yml) | [Round one](../../benchmarks/reports/decode-round-1.md#run-or-roll-back) |
| Shared prefixes with LFU | [shared-prefix-lfu.yml](shared-prefix-lfu.yml) | [Shared prefixes](../../benchmarks/reports/shared-prefix.md#full-window-rotation-and-reproduction) |
| LFU expert cache | [expert-cache-lfu.yml](expert-cache-lfu.yml) | [Expert cache](../../benchmarks/reports/expert-cache.md#running-the-candidate-on-this-workstation) |

For a native comparison using the workstation vLLM drafter and prefix-tier limits, append [reference-drafter.yml](reference-drafter.yml). It selects the released DSpark with five proposals, disables source and output lookup, and sets the native prefix tiers to 48 GiB RAM and 128 GiB disk. The [recipe-prefill.yml](recipe-prefill.yml) overlay selects the optional S8B build and 128-token physical prefill chunks. [Local reference, build and controls](../../benchmarks/reports/local-vllm-prefill.md)

On the workstation, pass `--env-file ../models/.env` to Compose from the checkout next to the stack's `models/` directory. For a fresh clone, follow the [build and launch instructions](../../README.md#build-and-serve-from-a-fresh-clone).
