# Native engine and published dual-RTX 6000 benchmarks

The two engines use the same workstation class and EXL3 model pack. The first table uses the upstream Python client unmodified, against port 8001. The benchmark first submits a 25,625-token Python module, then rewrites a different 41-function span per client. Each stream generates 1,500 tokens at temperature zero with default thinking. The published aggregate divides the total generated tokens by the longest individual post-first-token interval. [Upstream code](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/decode_bench2.py) · [Published benchmark](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)

## Published code-edit workload

| Concurrent clients | Published recipe | Native at 375 W | Same native build at 600 W | Native 600 W gap |
|---:|---:|---:|---:|---:|
| 1 | 282 tok/s | 275 tok/s | 277 tok/s | −1.8% |
| 4 | 518 tok/s | 413 tok/s | 420 tok/s | −18.9% |
| 6 | 557 tok/s | 446 tok/s | 452 tok/s | −18.9% |

These are second-pass figures from one complete script invocation per profile. Raising the power limit coincided with a 0.7–1.7% higher warm rate on the native control. Expert-cache warm-up differed between service starts. The November 15 report records 600 W, +350 MHz core and +4,000 MHz VRAM from one card’s screenshot and assumes its peer used the same settings. Both native profiles used +450 MHz core and +4,000 MHz VRAM. [375 W run](results/published-decode-375w.txt) · [600 W run](results/published-decode-600w.txt) · [Driver and LACT readback](results/benchmark-gpu-settings-600w.json)

The full direct-versus-LiteLLM context sweep uses a retrieval prompt with reasoning disabled. On the 375 W control, it reached 309 tok/s direct and 308 via LiteLLM at 65,536 input tokens, and about 238 on each route at 524,160. The separate new-code benchmark produced 330 tok/s median through LiteLLM over 18 measured requests. Draft acceptance can vary with the continuation. [Full C1 sweep](C1-BENCHMARK.md) · [New-code samples](results/c1-code-litellm-375w.jsonl)

At 600 W, six serial repetitions of the 1,500-token edit starting at function 50 had a median of 277.1 tok/s. A separate two-pair OpenAI sweep measured direct/proxy medians of 357.2/356.6 at 4,096 input tokens, 308.4/308.1 at 65,536, and 237.8/238.0 at 524,160. [Repeated edit timings and output hashes](results/published-code-control-600w.jsonl) · [Paired 600 W context checks](results/c1-context-control-600w.jsonl)

Power sampling covered 97 seconds-scale readings per card across part of the first pass and the entire warm pass. GPU 1 peaked at 347.7 W and GPU 2 at 414.1 W. GPU 2 exceeded the old 375 W limit in 30 samples. Both reported a 600 W enforced limit throughout the sampled interval. [Sampled power](results/published-decode-power-600w.json)

## Cold prompt processing

The native probe uses the upstream random-word generator and first-code instruction, includes the first reasoning or text delta in time to first token, checks the returned code and confirms zero prefix reuse. Seeds differ across the published and native cold requests. The published single-client figures come from November 9, when GPU clock and power settings were unrecorded. The recipe's 187-second six-client run is from November 15. [Published prefill and concurrency methodology](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)

| Workload | Published recipe | Native engine control |
|---|---:|---:|
| First streamed token, about 16k input | 1.4 s at 15,826 tokens | 17.14 s at 15,884 tokens and 375 W · 26.05 s at 15,846 tokens and 600 W |
| First streamed token, about 60k input | 5.0 s at 60,046 tokens | 72.02 s at 59,997 tokens and 600 W |
| Six independent cold requests, about 404k input each | 187 s | 3,714.61 s at 375 W |

The six-client native receipt run uses distinct transcripts of exactly 404,000 input tokens and generates 128 tokens per client. This compares wall time at the same load scale with different prompts and output lengths. The published setup processes 8,192-token chunks and uses a bulk expert kernel. The native control processes 64-token chunks. The native 60k request that overlapped the power change was excluded. [375 W single-client capture](results/published-prefill-375w.jsonl) · [600 W single-client capture](results/published-prefill-600w.jsonl) · [Six independent requests](results/six-independent-404k.jsonl)

## Capacity and validation coverage

The native engine has demonstrated six independent 524,288-position sessions and can rotate additional sessions through host RAM and disk. It served 32 concurrent 12,000-token clients through LiteLLM. The reference recipe reports a 2.77M-token GPU pool, six independent ~404k prompts and a separate 1M-window configuration. The native production window is 524,288 tokens. [Six independent native contexts](results/six-independent-full-proxy.jsonl) · [32-client run](results/thirtytwo-12k-rotation.jsonl)

The upstream release reports GSM8K 197/200, HumanEval 159/164, vision support and long-context needle tests. The native engine serves text, passes real Codex, Claude Code and Pi tool-use checks, and has retrieved codes from independent long prompts. Matched GSM8K, HumanEval, mixed-load interference and 20-minute soak results remain open for the native engine. [Client checks](results/c1-clients-375w.jsonl) · [Published quality and soak results](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md)
