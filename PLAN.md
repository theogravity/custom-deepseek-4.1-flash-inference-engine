# Runtime and qualification log

Target: use the existing calibrated DeepSeek V4.1 Flash EXL3 checkpoint and coding clients through LiteLLM on this workstation. Preserve the 512k context, six agent slots and model outputs. Use host GPUs 1 and 2 at the current 375 W limits. The 5080 serves the desktop and Winnow.

The base stack is parent revision `9f9dadd51f1cc3da0378b1c24b4b37b15fcc370e`. The serving image ID is `sha256:3689a42b8a151bf85d72d03b733ec6f353dba5936ff2558b577687c4f5908aa0`. The local Hugging Face download records shard 1 at `451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae` and the mounted EXL3 plugin at `6110453b901e4b51da66f539f3802b5c7f18e8e8`. These identify the local snapshot components.

On 2026-11-25, the first stock control delivered 236.1 output tok/s at concurrency 1 and 637.8 aggregate at concurrency 6. Each number is the median of six warm runs of tools/measure.py after a discarded warm-up. The code-edit prompt had 25,598 tokens and each response had 512 tokens. Historical tests used other prompts, output lengths, power limits and timing conventions. Raw runs are in results/ledger.jsonl.

After the model drained, a simultaneous two-card copy and GEMM benchmark ran with the model stopped. See THEORY.md and results/roofline-hardware.json.

The candidate changes one component, the Engram execution path for batches up to 64 tokens. Speculative decoding, kernel choices for other layers, KV capacity and precision match the stock Compose configuration.

The fixed-ID Engram microbenchmark initially measured ~40 vs 66 µs at six tokens, but ~78 vs 66 µs at 36 tokens. Repeated IDs issued redundant prefetch hints. The final callback deduplicates page hints within the callback and retains the GPU block budget on the background stream. The same warm-row graph comparison then measured 27.7 vs 69.4 µs and 32.4 vs 72.2 µs at six and 36 tokens, respectively. Those are per-lookup medians for 35 replays. Real-shard tests compared BF16 bits against an independent CPU reference on fresh IDs. See [graph test evidence](results/engram-graph.json).

The initial single-stream stock samples rose from 193 to 258 tok/s through the first six timed requests. Later comparison runs discard seven requests at each concurrency before recording six timed requests. Keep the initial control for reference and use the matched warm-up runs to compare full-model rates.
