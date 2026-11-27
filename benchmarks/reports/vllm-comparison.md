# Native engine and published dual-RTX 6000 benchmarks

The later concurrent coding release reached 383 / 584 / 583 aggregate tok/s at one, four and six clients on the upstream edit. The measurements below document earlier profiles. [Current comparison](concurrent-repetition.md)

For the preceding coding release, see the [headline benchmark replay](headline-reference.md). The experiments below used earlier native checkpoints.

The two engines use the same workstation class and EXL3 model pack. The first table uses the upstream Python client unmodified, against port 8001. The benchmark first submits a 25,625-token Python module, then rewrites a different 41-function span per client. Each stream generates 1,500 tokens at temperature zero with default thinking. The published aggregate divides the total generated tokens by the longest individual post-first-token interval. [Upstream code](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/decode_bench2.py) · [Published benchmark](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)

## Six-client code edits on the same host

At 25,625 input tokens per request, the shared-prefix LFU engine delivered 462–463 aggregate tok/s. The local vLLM recipe delivered 485 with CPU experts disabled and 537–561 with them enabled. Each figure is a six-wave median at 375 W per GPU. [On/off/on experiment, output checks and GPU profile](concurrent-edits.md)

## 375 W long-context release

The selected attention scorer and 6 GiB prefix RAM tier measured **278 / 422 / 460 aggregate tok/s** at C1/C4/C6 on the unchanged published 25,625-token edit script's second pass. The tagged 375 W checkpoint previously measured 275 / 414 / 448. The published comparison points are 282 / 518 / 557; the largest remaining gaps on this edit are at four and six clients. These single script passes use the upstream timing convention. The 404k coding and full-window retrieval results use repeated waves and explicit cache checks. [Long-context measurements](long-context.md) · [Upstream replay](../results/upstream/long-context-published-375w.txt)

## Matched GPU power test on the tuned release

The same server process ran at 600 W, then 375 W, then 600 W again on both inference cards. The table gives the second pass of the unchanged upstream code-edit script at each setting. All entries are output tok/s, aggregate across active clients. The measured CPU socket cap stayed at 225 W. [Runs, hardware readbacks and output hashes](../results/workstation/power-ab-tuned.json)

| Clients | 600 W before | 375 W | 600 W after |
|---:|---:|---:|---:|
| 1 | 277 | 275 | 278 |
| 4 | 417 | 414 | 419 |
| 6 | 453 | 448 | 454 |

Averaging the two 600 W code-edit passes gives a 0.9% improvement at C1, 1.0% at C4, and 1.2% at C6 relative to the 375 W pass. At full context the two 600 W receipt medians varied by 11.8 tok/s, and neither exceeded the 375 W observation. These three power legs give a workload-specific comparison.

On the warm receipt test, each cap leg used three measured native requests per input length after one warm-up. Each request generated 128 verified output tokens. The two 600 W columns bracket the 375 W run.

| Input tokens | 600 W before | 375 W | 600 W after |
|---:|---:|---:|---:|
| 4,096 | 503.5 | 501.0 | 505.3 |
| 65,536 | 446.5 | 445.1 | 447.2 |
| 524,160 | 314.4 | 314.8 | 302.6 |

## Published code-edit workload

| Concurrent clients | Published recipe | Native control 375 W | Native control 600 W | Optimized native 600 W | Gap to published |
|---:|---:|---:|---:|---:|---:|
| 1 | 282 | 275 | 277 | 278 | −1.4% |
| 4 | 518 | 413 | 420 | 423 | −18.3% |
| 6 | 557 | 446 | 452 | 459 | −17.6% |

Values are aggregate output tok/s. On the 600 W optimized build, the 1,500-token function-50 edit repeated six times measured 277.7 tok/s median. The visible and reasoning outputs matched the control for all eight requests across three function ranges. The scheduler reserves generated-output lookup for a lone decoder. Enabling it under all loads measured 278/388/406 tok/s on the same C1/C4/C6 script. [Optimized upstream run](../results/upstream/published-decode-final-600w.txt) · [Repeated edit and hashes](../results/upstream/published-code-final-600w.jsonl) · [Always-on lookup](../results/upstream/published-decode-always-lookup-600w.txt)

On five warm repeats of the function-50 edit, the selected seven-token DSpark plus lookup policy measured 277.7 tok/s, the same code-tuned DSpark checkpoint limited to five proposals measured 262.1, and seven-token DSpark with all lookup disabled measured 267.9. Each variant reproduced the control's visible and reasoning text on all eight requests. [Drafter ablation](../results/decode/drafter-edit-tuning-600w.json)

These are second-pass figures from one complete script invocation per profile. Raising the power limit coincided with a 0.7–1.7% higher warm rate on the native control. Expert-cache warm-up differed between service starts. The November 15 report records 600 W, +350 MHz core and +4,000 MHz VRAM from one card’s screenshot and assumes its peer used the same settings. Both native profiles used +450 MHz core and +4,000 MHz VRAM. [375 W run](../results/upstream/published-decode-375w.txt) · [600 W run](../results/upstream/published-decode-600w.txt) · [Driver and LACT readback](../results/workstation/benchmark-gpu-settings-600w.json)

The full direct-versus-LiteLLM context sweep uses a retrieval prompt with reasoning disabled. On the 375 W control, it reached 309 tok/s direct and 308 via LiteLLM at 65,536 input tokens, and about 238 on each route at 524,160. The separate new-code benchmark produced 330 tok/s median through LiteLLM over 18 measured requests. Draft acceptance can vary with the continuation. [Full C1 sweep](single-client-context.md) · [New-code samples](../results/clients/c1-code-litellm-375w.jsonl)

At 600 W, six serial repetitions of the 1,500-token edit starting at function 50 had a median of 277.1 tok/s. A separate two-pair OpenAI sweep measured direct/proxy medians of 357.2/356.6 at 4,096 input tokens, 308.4/308.1 at 65,536, and 237.8/238.0 at 524,160. [Repeated edit timings and output hashes](../results/upstream/published-code-control-600w.jsonl) · [Paired 600 W context checks](../results/clients/c1-context-control-600w.jsonl)

For the same native control binary, changing the GPU cap made little difference to warm C1 receipt decode. The 375 W medians use six route pairs at each size and the 600 W control uses two. The tuned 600 W column shows the later software configuration. [375 W samples](../results/clients/c1-overhead-warm.jsonl) · [600 W control](../results/clients/c1-context-control-600w.jsonl) · [Tuned 600 W samples](../results/clients/c1-context-final-600w.jsonl)

| Input tokens | 375 W control direct / LiteLLM | 600 W control direct / LiteLLM | Tuned at 600 W direct / LiteLLM |
|---:|---:|---:|---:|
| 4,096 | 356.6 / 356.4 | 357.2 / 356.6 | 504.4 / 496.5 |
| 65,536 | 309.2 / 307.9 | 308.4 / 308.1 | 447.6 / 441.7 |
| 524,160 | 237.7 / 237.6 | 237.8 / 238.0 | 322.9 / 322.0 |

Power sampling covered 97 seconds-scale readings per card across part of the first pass and the entire warm pass. GPU 1 peaked at 347.7 W and GPU 2 at 414.1 W. GPU 2 exceeded the old 375 W limit in 30 samples. Both reported a 600 W enforced limit throughout the sampled interval. [Sampled power](../results/upstream/published-decode-power-600w.json)

The owner recalled a 180 W CPU limit throughout benchmarking. The Threadripper PRO 9965WX boot service specifies 180 W. On November 25, systemd skipped that service because of an ordering cycle. At 20:58 PST AMD HSMP reported a 225 W socket cap during the native benchmark session. Earlier timings have no simultaneous CPU-cap capture. [CPU configuration and kernel readback](../results/workstation/benchmark-cpu-power.json)

## Cold prompt processing

The native probe uses the upstream random-word generator and first-code instruction, includes the first reasoning or text delta in time to first token, checks the returned code and confirms zero prefix reuse. Seeds differ across the published and native cold requests. The published single-client figures come from November 9, when GPU clock and power settings were unrecorded. The recipe's 187-second six-client run is from November 15. [Published prefill and concurrency methodology](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)

| Workload | Published recipe | Native engine control |
|---|---:|---:|
| First streamed token, about 16k input | 1.4 s at 15,826 tokens | 17.14 s at 15,884 tokens and 375 W · 26.05 s at 15,846 tokens and 600 W |
| First streamed token, about 60k input | 5.0 s at 60,046 tokens | 72.02 s at 59,997 tokens and 600 W |
| Six independent cold requests, about 404k input each | 187 s | 3,714.61 s at 375 W |

The six-client native receipt run uses distinct transcripts of exactly 404,000 input tokens and generates 128 tokens per client. This compares wall time at the same load scale with different prompts and output lengths. The published setup processes 8,192-token chunks and uses a bulk expert kernel. The native control processes 64-token chunks. The native 60k request that overlapped the power change was excluded. [375 W single-client capture](../results/upstream/published-prefill-375w.jsonl) · [600 W single-client capture](../results/upstream/published-prefill-600w.jsonl) · [Six independent requests](../results/concurrency/six-independent-404k.jsonl)

On the same two fresh ~15.8k-token prompts repeated after the expert cache warmed, the 600 W control processed 1,040 input tok/s (median of two). Four-tile packed projections and parallel router selection increased that to 1,176 tok/s at the same 64-token chunk size, up 13.0%. A 128-token test reached 1,319 tok/s on those seeds. All three variants ingested every input token and returned the same visible and reasoning text. These two-seed steady-state tests use different cache and seed conditions from the reference’s first-token table. [Matched prefill timings and hashes](../results/context/cold-prefill-tuning-600w.jsonl)

An experimental 512-row prefill twice triggered a GPU copy-engine memory fault with staged expert transfers. The release uses direct peer access for target prefill batches above 128 rows. All six requests in the reproducer then passed across each of two fresh service processes, with matching receipts and reasoning digests. The four repeated ~16k inputs reached 478 input tok/s median in that mode. Production retains 64-row chunks. The failing DMA operation remains unidentified. [Wide prefill tests](../results/decode/wide-prefill-600w.json) · [Release test suite](../results/decode/test-600w-release.txt)

At C1 the optimized build generated new code at **314.8 tok/s median** through LiteLLM (18 measured requests across three prompts). On two six-client full-window receipt waves through LiteLLM, each client supplied 524,160 input tokens and generated 128. End-to-end wall times, including prefix restoration, were 10.02 s and 8.71 s. All six output digests matched their earlier per-client results. [New code](../results/clients/c1-code-600w.jsonl) · [Full-window waves](../results/concurrency/six-full-600w.jsonl)

## Capacity and validation coverage

The native engine has demonstrated six independent 524,288-position sessions and can rotate additional sessions through host RAM and disk. It served 32 concurrent 12,000-token clients through LiteLLM. The reference recipe reports a 2.77M-token GPU pool, six independent ~404k prompts and a separate 1M-window configuration. The native production window is 524,288 tokens. At 600 W, the eight-client full-context test forced disk overflow by setting the live RAM tier to 1 GiB and the rotation quantum to four steps. Production uses 48 GiB and 32 steps. The test completed seven RAM and 13 disk save/restore pairs with zero cache errors. All eight outputs matched their serial replays. The 22 context-limit cases also passed. [Eight-client overflow](../results/context/eight-full-600w.jsonl) · [Boundary cases](../results/clients/context-limits-600w.jsonl) [Six independent native contexts](../results/concurrency/six-independent-full-proxy.jsonl) · [32-client run](../results/context/thirtytwo-12k-rotation.jsonl)

The upstream release reports GSM8K 197/200, HumanEval 159/164, vision support and long-context needle tests. The native engine serves text, passes real Codex, Claude Code and Pi tool-use checks, and has retrieved codes from independent long prompts. Matched GSM8K, HumanEval, mixed-load interference and 20-minute soak results remain open for the native engine. [600 W client checks](../results/clients/clients-600w.jsonl) · [Published quality and soak results](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)
