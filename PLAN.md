# Runtime and qualification log

Target: use the existing calibrated DeepSeek V4.1 Flash EXL3 checkpoint and coding clients through LiteLLM on this workstation. Preserve the 512k context, six agent slots and model outputs. Use host GPUs 1 and 2 at the current 375 W limits. The 5080 serves the desktop and Winnow.

The base stack is parent revision `9f9dadd51f1cc3da0378b1c24b4b37b15fcc370e`. The serving image ID is `sha256:3689a42b8a151bf85d72d03b733ec6f353dba5936ff2558b577687c4f5908aa0`. The local Hugging Face download records shard 1 at `451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae` and the mounted EXL3 plugin at `6110453b901e4b51da66f539f3802b5c7f18e8e8`. These identify the local snapshot components.

On 2026-11-25, the first stock control delivered 236.1 output tok/s at concurrency 1 and 637.8 aggregate at concurrency 6. Each number is the median of six warm runs of tools/measure.py after a discarded warm-up. The code-edit prompt had 25,598 tokens and each response had 512 tokens. Historical tests used other prompts, output lengths, power limits and timing conventions. Raw runs are in results/ledger.jsonl.

After the model drained, a simultaneous two-card copy and GEMM benchmark ran with the model stopped. See THEORY.md and results/roofline-hardware.json.

The candidate changes one component, the Engram execution path for batches up to 64 tokens. Speculative decoding, kernel choices for other layers, KV capacity and precision match the stock Compose configuration.

The fixed-ID Engram microbenchmark initially measured ~40 vs 66 µs at six tokens, but ~78 vs 66 µs at 36 tokens. Repeated IDs issued redundant prefetch hints. The final callback deduplicates page hints within the callback and retains the GPU block budget on the background stream. The same warm-row graph comparison then measured 27.7 vs 69.4 µs and 32.4 vs 72.2 µs at six and 36 tokens, respectively. Those are per-lookup medians for 35 replays. Real-shard tests compared BF16 bits against an independent CPU reference on fresh IDs. See [graph test evidence](results/engram-graph.json).

The initial single-stream stock samples rose from 193 to 258 tok/s through the first six timed requests. Later comparison runs discard seven warm-up batches at each concurrency before recording six timed batches. Keep the initial control for reference and use the matched warm-up runs to compare full-model rates.

A clean clone of commit `a5cf7c2` reproduced the generated module, native bridge and shared library byte-for-byte. Both tensor-parallel workers mapped the resulting library in the live model process. On the native build, all four client paths returned their own randomly generated tool receipt, and a streamed request returned separate thinking and text with the correct final answer. The real binaries were Codex CLI 0.172.0, Claude Code 2.1.296 and Pi 1.4.0. Pi used both the Anthropic and DeepSeek aliases. [Client evidence](results/client-acceptance.jsonl)

The native build recovered three random receipts at positions near 3%, 50% and 96% in each of the 261,107- and 518,814-token cold prompts. End-to-end times were 42.5 and 85.1 seconds, including HTTP and generation. [Context evidence](results/context.jsonl)

All six concurrent requests of 411,169–411,266 prompt tokens returned all three receipts correctly (2,467,359 input tokens total). Elapsed time for the batch was 417.43 seconds. Individual requests took 76.71–417.18 s from submission to answer, including queueing. The sampler recorded peaks of 95,964 MiB per GPU across the text and vision tests. Four checkpoint vision probes also passed, including repeated and multi-image inputs. [Context evidence](results/context.jsonl) · [Vision evidence](results/vision.txt)

## Measured throughput at 375 W per card

| Warm code edit | Original, matched control | Native, first boot | Native, second boot |
| --- | ---: | ---: | ---: |
| One stream | 292.2 tok/s | 320.4 tok/s | 304.0 tok/s |
| Six streams, aggregate | 683.6 tok/s | 710.0 tok/s | 683.8 tok/s |

Each cell is the median of six timed batches after seven warm-up batches at that concurrency. The three matched builds ran in the order native, original, native. All timed streams returned 512 completion tokens with a length stop. These are warm 25,598-token code edits with temperature zero and thinking disabled. The three short greedy answers also matched byte-for-byte between native and original. [Throughput ledger](results/ledger.jsonl) · [Speculative counters](results/draft-counters.jsonl) · [Greedy outputs](results/greedy.jsonl)

The native kernel improved the isolated Engram operation. End-to-end one-stream medians rose on both candidate boots, with a smaller difference on the repeat. Six-stream throughput on the repeat was effectively equal to stock. The run-to-run spread limits the speedup that can be attributed to Engram. The accepted-draft rate on this workload is unusually high because the prompts ask for repetitive code edits.

Resampling six timed batches with replacement 20,000 times per build (median native divided by median stock, percentile interval) gives 95% intervals of 0.98–1.19 and 0.95–1.13 for the two single-stream runs, and 0.99–1.08 and 0.95–1.05 at six streams. Each interval includes parity. These are within-boot sample intervals and do not account for every source of variation between boots.

Final system state on 2026-11-25: the native build is running and healthy on port 8001 behind the existing LiteLLM proxy on 4000. GPU 1 and 2 remain at 375 W. All six acceptance checks passed again after the final native restart. [Final client checks](results/client-final.jsonl)
