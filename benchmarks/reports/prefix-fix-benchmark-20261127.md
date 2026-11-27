# Prefix checkpoint fix: concurrent code-edit benchmark

The benchmark replays the published 25,625-token Python edit. Every client requests 1,500 output tokens at temperature zero with default reasoning. It used engine commit `b47aa30` (`checkpoint-2026-11-27-prefill-batch-sweep`), image `ds4-flash-rs:tp2-peer-index-prefix64-20261127` (`c623b9507e7e`), an 8,192-token prefill chunk, the 64-token prefix block schedule, a warm 25,600-token shared prefix, and two RTX PRO 6000 GPUs configured at 375 W. Each run has one warm-up wave followed by six measured waves. `upstream_tok_s` uses the published convention: total generated tokens divided by the longest per-client interval from first output to stream close.

| Clients | Published vLLM, 600 W | Previous native default | Post-fix native median | Shared-span median | Whole-wave median | Median first output |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 518 | 312.0 | **491.4** | 486.8 | 483.6 | 0.219 s |
| 6 | 557 | 261.2 | **497.8** | 491.8 | 490.8 | 0.312 s |

Against the published vLLM values, the post-fix Rust run is 5.1% behind at C4 and 10.6% behind at C6. It improves 57.5% and 90.6% over the previous recommended-default measurements. The published vLLM figures used 600 W GPU caps; this native run used 375 W. The GPU limits were read back at 375 W after the benchmark. The host CPU power cap reads 225 W after the run; it was not sampled per wave.

All 24 measured C4 streams and 36 measured C6 streams returned the requested 1,500 tokens and the requested starting function. Prompt counts were 25,625 for every stream. Reasoning and visible-output hashes were stable across all seven waves for each function range. C1 was not remeasured as part of this run.

The cold C4 warm-up after restarting the native service restored the shared target snapshot and DSpark sidecar from disk for one client and from RAM for the other three. All four completed. The six measured C4 waves and six measured C6 waves then restored from RAM. No cache retrieval/needle test was run.

An earlier C4 attempt returned an incomplete stream. A retry then logged a CUDA `unspecified launch failure` during a DSpark host-to-device restore copy. This failure did not recur after restarting the native engine: the disk-cold C4 wave and both full C4/C6 series passed. The exact originating CUDA operation remains unproven. The engine previously swallowed that restore error as a cache miss and could continue issuing work on a failed CUDA context. It now propagates DSpark snapshot/restore failures, marks the runtime unavailable for later requests, and reports SSE error events explicitly in the benchmark client.

Raw waves: [C4](../results/upstream/c4-code-edit-post-prefix-fix-20261127.jsonl) · [C6](../results/upstream/c6-code-edit-post-prefix-fix-20261127.jsonl). Reference values and prior run definitions are in [vLLM comparison](vllm-comparison.md).
