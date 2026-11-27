# Headline benchmark replay on the coding release

We replayed the upstream benchmarks against the deployed Rust/CUDA engine on November 26, 2026 (Pacific time). Warm code-edit decode and single-client cold prefill completed. The six-client long-context run was stopped during cold prefill. Both use two RTX PRO 6000 Blackwell workstation GPUs and the Diffbot EXL3 pack. The vLLM figures below come from the original publication. All native figures are from this replay. [Upstream measurements](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md) · [Per-run data and seeds](../results/upstream/headline-reference.json) · [Benchmark stdout](../results/upstream/headline-stdout.txt)

## Warm code-edit decode

| Concurrent clients | Published vLLM tok/s | Native tok/s | Native vs. vLLM |
|---:|---:|---:|---:|
| 1 | 282 | 382.5 | +35.6% |
| 4 | 518 | 457.5 | -11.7% |
| 6 | 557 | 502.5 | -9.8% |

These are aggregate output tokens per second, including reasoning tokens. The benchmark warms a 25,625-token Python module, then each client requests a rewrite of 41 functions and generates 1,500 tokens. It divides total completion tokens by the longest individual interval from first output to stream close. We ran the upstream script six times and took the median of its six second-pass rates at each concurrency. The upstream figures are one published second pass. The script rounds rates to whole tok/s. Every native invocation restored 25,600 prefix tokens for each of its 23 requests, including warm-up. The timing client records token counts and timings only. Code correctness was ungraded. [Exact decode client](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/decode_bench2.py)

## Single-client cold prefill

| Published input tokens | Native input tokens | vLLM cold TTFT | Native cold TTFT | Native / vLLM time |
|---:|---:|---:|---:|---:|
| 15,826 | 15,858 | 1.4 s | 23.14 s | 16.5× |
| 60,046 | 60,017 | 5 s | 60.9 s | 12.2× |
| 195,600 | 195,702 | 15.7 s | 204.77 s | 13.0× |
| 505,214 | 505,143 | 41.6 s | 601.78 s | 14.5× |

Each row is one streamed request with a fresh random seed and zero native prefix tokens restored. The serving process had already run other requests. Filesystem caches were left in place. TTFT runs from sending the request to the first visible or reasoning delta. Both engines use the upstream word counts of 5,000, 19,000, 62,000 and 160,000. Actual token counts vary with the seed. The upstream client reported the expected first code in all four visible answers. [Exact TTFT client](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/lmc_ask.py)

## Six concurrent long conversations

| Six independent prompts | Published vLLM wall time | Native wall time |
|---|---:|---:|
| Cold, ~404k input tokens per client | 187 s | Interrupted |
| Same six prompts immediately repeated | 2 s | Skipped |

All six native requests were active at the last progress sample. At 2,511 s after script launch, the engine reported 2,211,520 processed tokens across their live sessions. The run ended before the cold wave returned its answers. The native wall time and retrieval outcome are therefore unavailable. [Upstream six-client benchmark](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/six400.py)

## Reproduce the completed workloads

Check out the linked upstream revision and export `API_KEY` for the direct service on `localhost:8001`. Run `python3 -I bench/decode_bench2.py` six times. For each word count above, run `python3 -I bench/lmc_ask.py <words> <fresh_seed>`. The per-run data records this replay’s seeds. Choose fresh ones when testing cold prefill against a server that may still hold those prefixes.

## Serving conditions

The native run used the deployed code-tuned DSpark drafter with up to seven learned proposals, source lookup, 64-token prefill chunks, 375 W per GPU and a measured 225 W CPU socket cap. The vLLM release used its five-proposal drafter, CPU decode experts and 8,192-token prefill chunks. Its November 15 decode and six-client measurements used a reported 600 W GPU profile with a +350 MHz core offset, versus +450 MHz on the native profile. The GPU settings for its November 9 single-client prefill table were unrecorded. All three scripts sent temperature zero and omitted an explicit thinking-effort setting. The native renderer selects 75/100 by default. The upstream report describes its default as approximately 50/100. Inspection of the installed vLLM tokenizer on November 26 found that it maps omitted effort to 75/100, matching the native renderer. The historical rendered prompt and output tokens are unavailable. These runs compare complete serving configurations on different dates. The published November 15 GPU profile was recorded on one card and assumed to match the second. [Upstream setup and dates](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md) · [Native settings](../../docs/engine.md)
