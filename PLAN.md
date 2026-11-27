# Implementation and qualification log, 2026-11-25

The target is one native text inference engine for the installed DeepSeek V4.1 Flash 3-bit EXL3 pack, with the existing authenticated LiteLLM routes and coding clients. All model execution uses Rust, CUDA C++ and cuBLAS. Testing used the two RTX PRO 6000s. The RTX 5080 continued to serve its existing workload.

1. Captured token IDs and next-token log probabilities from the previous DeepSeek service for four raw prompts, then stopped it. The local fixture can be checked offline. [Reference](tests/reference.json)
2. Implemented the checkpoint reader, 40-layer target, sparse attention and compressed cache, n-gram Engram lookup, quantized expert execution, and HTTP adapters. The first end-to-end build matched the first token on all four prompts. Exact full continuations initially matched 0/4. The auth and streaming checks passed and each of the three installed coding clients read a fresh receipt. [Initial client checks](results/initial-clients.jsonl)
3. Compared the expert kernel to an independent dense decode of real EXL3 weights. Relative L2 error was 0.000233 for one row and 0.000282 for 18 rows across two experts. A forced two-slot GPU cache reloaded evicted experts with identical output. Tested tokenizer prompts against both checkpoint goldens, state snapshots across chunk boundaries, cancellation, bounded streaming, and FP8 quantization. [Expert check](tools/check_moe.py)
4. Matched the deployed sm_120 attention format. Each compressed row is 584 bytes with 448 FP8 channels, 64 BF16 rotary channels, and scale bytes. Index keys use 128 FP8 channels and an FP32 scale. The production CUDA writer and this engine produced identical bytes for a real checkpoint row. Adjusted dense activations, compressor precision, and the FP8 vocabulary head to the workstation recipe. All four prompts predicted the reference first token, and one full greedy continuation matched. A punctuation or stopping choice can change the rest of a continuation. [Token-by-token continuations](results/greedy-comparison.txt)
5. Measured six single-stream decode samples after one warm-up at temperature zero. The previous native build measured a median of 25.48 tok/s. The sm_120-compatible build measured 26.11 tok/s. The 2.5% difference is within run variation. These changes were kept for numeric compatibility. All runs returned the same sequence of prime numbers. The final rate is roughly 27% of this engine's short-context 95.6 tok/s weight-bandwidth estimate. [Baseline](results/baseline-decode.json) · [Final samples and generated text](results/decode.json) · [Ceiling](THEORY.md)
6. Qualified cache reuse using distinct requests through LiteLLM. A first attempt repeated identical HTTP content and LiteLLM served the second answer from its own cache. A changed suffix produced two measured Rust prefills, with 2,880 of 2,895 input tokens reused on the second. Both returned the correct random receipt. [Responses](results/retrieval.json) · [Worker measurements](results/retrieval-prefill.txt)
7. An early Docker stop killed the idle process after its timeout. Added SIGTERM and SIGINT handling and joined the inference worker before process exit. Stopped the final server while it streamed a real response. The client received the terminal event and Docker reported exit code 0. The service was restarted and its deployed executable matched the local release hash. [Shutdown check](results/shutdown.json) · [Build and tests](results/release.json)

## Speculative engine and two-card tuning

The optimization target was more than 300 target-verified output tokens/s for a single coding client. Each new-code sample uses 1,024 generated tokens, temperature zero and a short prompt. The three prompts were taken from the model pack. The source-edit task returns the complete updated Go file and can benefit from matching tokens in its input. Direct engine requests bypass the proxy's optional answer cache.

| Experiment | Result and decision |
|---|---|
| Device-resident target, no drafter | The initial 25.75 tok/s control rose to 44.50 on the same workload. The GPU sampler and packed dense kernels subsequently took a 256-token Tetris control to 86 tok/s. |
| Learned drafters | At an early build, original DSpark reached 123 tok/s and the code-tuned overlay reached 130 on a 128-token Tetris continuation. Kept both. |
| Packed MXFP8 target and GPU head | The retiled packed matrix multiplications cut weight traffic and significantly improved verification. K=7 was the preferred tested setting for the code overlay. |
| Splitting all dense projections, direct remote expert reads | Both slowed the short coding control. Dense tensor parallelism stays disabled. Replaced direct remote accesses with explicit staging. |
| Staged target expert split | 287.97 vs 260.14 and 259.64 tok/s neighboring controls on 768 Tetris tokens. Kept. Draft-layer experts remain on their owning card. |
| Learned confidence | A 0.4 cutoff shortened proposed verification from 929 to 824 rows with the same 634 accepted, increasing 260 to 270 tok/s before staged parallelism. |
| Source lookup | Full 1,700-token Go edit reached 470.5 tok/s across six warm runs on an intermediate profile. The final combined profile reached 846.6 over two on-wire samples. All edits passed the Go regression. |
| Combined profile | Six interleaved warm runs per new-code prompt at 303.4 / 351.3 / 330.6 tok/s. All three prompts were shorter than the cache's 64-token chunk. |

The confidence and staging experiments used shorter screening runs. The on-wire suite exercised the combination for six warm runs per prompt, with a first warm-up for each. Historical trials and their continuation digests are in the ledger. The numerical profile uses FP8 and BF16 rounding and grouped reductions, so generated text can differ from the original vLLM recipe. [Raw trials](results/ledger.jsonl) · [Confidence controls](results/confidence.json) · [Timing protocol](tools/code_speed.py)

The pinned serving binary then repeated the 18-sample suite at 303.7, 351.1 and 330.5 tok/s (median 330.5). Its six warm full-edit trials reached 837.5 tok/s median with identical generated source. Both checkpoint drafters reproduced the undrafted 256-token target output at this profile. [Repeat new-code samples](results/clean-code.jsonl) · [Repeat edits](results/clean-edit.jsonl) · [Parity](results/drafter-parity.json)

At 25,598 input tokens and a 512-generated-token cap, the Python rewrite exposed a cost hidden by short-context benchmarks. Fixed 128-token source lookup accepted 211 of 1,954 proposals. The learned drafter accepted 245 of 253. A fixed 16-position probe reached 302.7 tok/s, learned-only reached 331.2, and per-request width feedback shrank probes too far and reached 244.0. Every fixed and adaptive Python run reached the same 512-output-token limit. The selected scheduler uses the amount of contiguous matching source text to decide between a learned proposal and a copy. Its direct HTTP medians were 308.1 for the partial rewrite and 866.5 for the full Go edit, with identical output hashes to their fixed-profile controls. The scheduler is request-local and the same persisted KV files serve either mode. [Controlled trials](results/lookup-tuning.json) · [Partial rewrite](results/hybrid-long.jsonl) · [Full edit](results/hybrid-go.jsonl)

## Six concurrent 524,288-token contexts

The new serving target is six independent active requests with up to 524,288 positions each. The previous native service queued six callers behind a single request. The separate vLLM recipe had a reported 2,769,546-token pool and demonstrated six ~404k histories. The native six-slot target allows 3,145,728 positions across the two inference cards. Shared target projections, DSpark proposals and grouped experts amortize weight reads across ready requests. KV, sampling, draft attention, speculative acceptance and session errors remain per request. The worker advances cold prefill in 64-token chunks to preserve numerical settings and compatible prefix snapshots.

| Change | Warm six-stream decode | Workload and decision |
|---|---:|---|
| Joint target with sequence-local attention and drafting | 336 tok/s | Initial six coding clients, two warm screening waves |
| Joint attention projections and learned draft projections | 399 tok/s | Six coding clients, six warm waves. All generated bytes matched the serial service |
| Packed FP32 index scorer | 289 tok/s | Full restored windows, up from 243 on the same six streams; exact scalar score parity |
| Radix sort dispatch by row count and index width | 361 tok/s | Full restored windows, six warm waves. Identical completion and selected-index fixtures; short coding held at 395 tok/s |

At 262,144 index positions × 8 verification rows, full GPU index selection fell from 3.71 ms to 2.08 ms after the sort change. The CUDA workspace fell from 64.0 to 50.1 MiB. At 524,288 × 4 the timing held at 1.79 ms. The worker and device tests include the failure of a middle KV session while both healthy requests finish normally. [GPU microbenchmark](results/packed-index-timing.txt) · [CUB controls](results/radix-shapes.txt) · [Concurrency checks](results/six-full-restored.jsonl) · [Release tests](results/test-six-release.txt)

## 2026-11-25 live KV and client qualification

Six independent agents reached 524,288 positions each and retrieved receipts from their original prompts and extensions. The default six-resident, 32-admitted scheduler supports 48 GiB of live RAM offload and 128 GiB of live disk offload. Eight full-window clients through forced RAM and disk overflow matched all serial replays. Near-full live restores took median 257 ms of measured operations from RAM and 869 ms from disk. The first 32-client run reached six resident and 26 parked but missed overlap across all 32 deliveries. All 32 results matched the serial controls. The warm repeat passed overlap, finished in 27.90 s and matched all 32 again. [Full-window continuations](results/six-independent-full.jsonl) · [Two-tier rotation](results/eight-full-ram-disk.jsonl) · [32-client result](results/thirtytwo-12k-rotation.jsonl)

The release validates context limits before GPU admission, preserves valid partial output budgets, and returns protocol-specific HTTP 400 errors. The patched proxy retains the symbolic context error needed by Codex. The live boundary test passed all 22 cases. An isolated replay using the installed Codex verified its machine-readable classification. [Boundary tests](results/context-limits-live.jsonl) · [Codex behavior](results/codex-context-classification.json)

With warm prefixes, 224 single-client requests compared the native and LiteLLM routes over the same eight context lengths and two wire protocols. The largest difference between route median output rates was 0.43%. Rates exceeded 300 tok/s through the 65,536-token case on this receipt-generation workload and dropped to about 238 tok/s at 524,160. [C1 comparison](C1-BENCHMARK.md)

Future performance runs use the native OpenAI Chat endpoint on port 8001. The completed transport comparison remains available in [the C1 results](C1-BENCHMARK.md).

## Long-context decode and prefix RAM at 375 W

Tagged the stable engine and serving image before changing the CUDA scorer. Reducing packed index scoring to eight or sixteen lanes preserved raw scores and improved single-client full-window retrieval from 321–322 to 372 tok/s. At 404k, six coding clients delivered 252 aggregate tok/s with the scorer alone and 285 with a 6 GiB host prefix tier. The joint drafter projection experiment measured 228 against controls of 231 and remained on its experiment branch. [Controls, workloads and gates](LONG-CONTEXT-BENCHMARK.md)

## Expert uploads and draft limits at 404k

Six 512-token coding clients delivered 260–264 aggregate tok/s on the 375 W checkpoint. A maximum draft length of five measured 258, and a confidence threshold of 0.65 measured 261. Neither improved delivery against both controls. The next experiments target repeated expert uploads (2.78–2.82 GiB/s during six-way decode), host copies during warm prefix admission, and the routed expert projection kernel. [Ranked experiments, GPU timelines and controls](NEXT-OPTIMIZATIONS.md)

## Frequency-based expert eviction at 375 W

The local checkpoint tag `checkpoint-2026-11-26-before-expert-cache` marks the start of this experiment. At the same slot budget, LFU reduced median six-client expert uploads from 2,676 to 1,191 per warm 404k coding wave. The live candidate delivered 278.1 tok/s at 512 output tokens per client and 300.0 and 297.0 tok/s in two independently controlled 768-token runs. The normal service uses the LRU checkpoint. [Full-window gates, rates, replay evidence and opt-in launch](EXPERT-CACHE-EXPERIMENT.md)

## Shared target rows on warm prefix restore

Selected shared packed rows with LFU for the 375 W service. At six independent 404k contexts the 512-token coding median rose from 278.2–278.8 to 296.6 aggregate tok/s. At 768 output tokens it reached 313.2, with all six measured waves above 300. Six full-window receipt clients rose from 202.7–204.4 to 287.7. The unchanged published 25.6k edit script reached 464 tok/s across six. Eight full-window clients matched the checkpoint through both RAM and forced disk rotation. [Experiment, per-client timing and deployment](SHARED-PREFIX-EXPERIMENT.md)

## vLLM CPU experts and native speculative verification

On the same six 25,625-token edits, vLLM delivered 537–561 aggregate tok/s with CPU experts and 485 with GPU experts alone. The native shared-prefix LFU controls delivered 462–463. Native expert uploads fell to two and then zero across two warm measured waves. Lowering the native learned draft maximum to five lost 8.5–8.7% against both seven-proposal controls with identical continuations. Both the seven-proposal default and the four-warp expert kernel remain selected. [Controlled comparison and native CUDA trace](EDIT-SCALING-EXPERIMENT.md)

## Selective CPU experts

At six 404k coding sessions, CPU experts delivered 266 and 289 aggregate tok/s at output budgets of 512 and 768. GPU-only controls delivered 297–299 and 321–325 respectively. The experimental host path stays disabled. [Performance and numerical checks](CPU-EXPERT-EXPERIMENT.md)

## Single-client decode at 375 W, 2026-11-26

Wide attention ranking and warp-shared expert weights produced sub-percent single-client differences. Verified output repetition raised the 25,625-token edit from 293–294 to 366.5 tok/s and 404,151-token coding from 236–237 to 256.9 tok/s. Both gains repeated across fresh server starts. The selected profile retains the six-resident scheduler and RAM/disk KV rotation. [All three experiments, holdouts, checks and Git checkpoints](C1-DECODE-EXPERIMENTS.md)

## Second decode round at 375 W, 2026-11-26

Longer repetition proposals and vocabulary-head row tiling reached 386.7 tok/s on the deployed single-client edit. The Markov shortlist made the draft scan faster but reduced accepted tokens per verification round. Six concurrent 404k coding sessions reached 282.7 tok/s aggregate with the wider head. [Measured controls, qualifications and three checkpoints](DECODE-ROUND2.md)

## Returning from Pi tools at 375 W, 2026-11-26

Preparing canonical history on idle GPUs reduced the wait after a simulated two-second tool from 4.17–4.22 s to 2.67 s. With six seconds available it fell to 0.097 s. Immediate replies took 4.18 s. All 54 captured-response replays matched. [Experiment, full-window checks and deployment](TOOL-RETURN.md)

## Coding-agent optimization round, 2026-11-26

Caching ordinary text tokens cut the median wait for first output on changing 404k histories from 0.85–0.93 s to 0.48 s. Six warm 404k coding streams delivered 299.7 aggregate tok/s over the whole request wave. Two-score interleaving in the DSpark head and bulk prefix-key hashing remain experimental. [Results and three checkpoints](AGENT-ROUND3.md)
