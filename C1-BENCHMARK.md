# Single-client decode throughput by context length

The native Rust/CUDA engine and LiteLLM both cleared **300 output tok/s** through **65,536 input tokens** on this workload. The next measured size, 131,072 tokens, delivered about 296 tok/s on each route. At 524,160 input tokens, both delivered **238 tok/s**. Across eight warm context lengths and both API formats, the largest difference between direct and proxied median decode rates was **0.43%**. [Warm samples and backend counters](results/c1-overhead-warm.jsonl)

## Warm decode throughput

Values are streamed output tokens per second for one active client. Each column reports the median of six measured runs at that context length. Each run produced 128 visible tokens from a greedy receipt-retrieval prompt with reasoning disabled. The window includes the generated tokens. The separately measured new-code workload uses different prompts. [New-code benchmark](results/live-offload-c1-coding.jsonl)

| Input tokens | OpenAI direct | OpenAI via LiteLLM | Anthropic direct | Anthropic via LiteLLM |
|---:|---:|---:|---:|---:|
| 256 | 379.0 | 378.0 | 378.6 | 378.7 |
| 4,096 | 356.6 | 356.4 | 357.4 | 358.1 |
| 16,384 | 328.4 | 327.1 | 327.5 | 327.2 |
| 65,536 | 309.2 | 307.9 | 308.5 | 308.0 |
| 131,072 | 295.7 | 295.5 | 296.1 | 296.4 |
| 262,144 | 269.1 | 268.4 | 268.7 | 269.0 |
| 404,000 | 263.6 | 262.8 | 263.6 | 262.7 |
| 524,160 | 237.7 | 237.6 | 237.8 | 238.1 |

![Warm direct and LiteLLM single-client decode rates, first text and request overhead](results/c1-overhead.png)

## Latency through LiteLLM

Each value below is the median of six paired differences in milliseconds. Positive numbers mean a longer wait through the proxy. At 524,160 input tokens the OpenAI route added 16.9 ms to the first visible text and 36.0 ms to stream completion. The Anthropic route added 12.4 ms and 12.8 ms. [Absolute timings and interquartile ranges](results/c1-overhead-summary.csv)

| Input tokens | OpenAI added first text (ms) | OpenAI added total (ms) | Anthropic added first text (ms) | Anthropic added total (ms) |
|---:|---:|---:|---:|---:|
| 256 | 7.6 | 20.0 | 8.4 | 9.1 |
| 4,096 | 6.4 | 19.4 | 7.2 | 6.2 |
| 16,384 | 7.3 | 20.5 | 6.4 | 6.8 |
| 65,536 | 9.7 | 22.0 | 9.5 | 9.3 |
| 131,072 | 11.9 | 40.1 | 7.0 | 6.7 |
| 262,144 | 12.6 | 31.7 | 6.2 | 7.3 |
| 404,000 | 9.7 | 28.7 | 7.2 | 8.6 |
| 524,160 | 16.9 | 36.0 | 12.4 | 12.8 |

For OpenAI Chat, the extra time *after the last visible text* also grew with prompt size, from 11.3 ms at 256 tokens to 19.3 ms at 524,160. That is the time until the client received the terminal SSE frame. For Anthropic Messages the paired median for this interval stayed within 0.02 ms of zero at each size. These are client-observed intervals. [Stream timings and paired samples](results/c1-overhead-warm.jsonl)

## Decode after a cold prompt

Each rate is the median of two measured runs after a warm-up pair. Both native prefix tiers were disabled and every request ingested the entire prompt. The model and OS page cache were warm. At 65,536 input tokens, cold ingestion took about 66 seconds on either route. Once output began, both routes still delivered over 306 tok/s. Small differences in first-token times include variation in ingestion time. [Cold samples and native execution counters](results/c1-overhead-cold.jsonl)

| Input tokens | OpenAI direct | OpenAI via LiteLLM | Anthropic direct | Anthropic via LiteLLM | OpenAI direct / proxy first text (s) |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 356.5 | 355.3 | 357.5 | 355.3 | 3.762 / 3.770 |
| 16,384 | 326.9 | 324.9 | 326.3 | 326.7 | 15.190 / 15.198 |
| 65,536 | 307.9 | 306.4 | 307.5 | 307.9 | 66.154 / 66.307 |

## Measurement conditions

The test used the two RTX PRO 6000s at the workstation owner’s 375 W power limit, +450 MHz core and +4,000 MHz memory offsets. Each route opened a fresh local HTTP connection. Client timing began after JSON encoding and included request transfer, server work, proxy processing when applicable, and SSE delivery. Time to first text ended at the first visible content. Decode rate used 127 tokens divided by the interval from first to last visible text. Total latency ended at the terminal SSE frame. [Hardware profile](results/benchmark-gpu-settings.json) · [Benchmark harness](tools/c1_proxy.py)

The harness alternated which route ran first, excluded one warm-up pair per wire and length, and verified a native execution with zero concurrent work for every request. It checked input and output counts, expected receipts, termination reason and identical output hashes. Warm measured requests restored exactly the preceding 64-token chunk boundary. LiteLLM response-cache reads and writes were disabled. Both routes used the checkpoint’s own tokenizer counts. [Serving profile](results/serving-profile.json)

## Reproduce the comparison

Run from the engine repository with the native service, LiteLLM, and an existing API key available to the harness. The first command creates the six independent 404,000-token histories. The warm comparison uses the first history at 404,000 tokens and its verified continuation at 524,160.

```sh
uv run --with tokenizers==0.22.2 python tools/six_streams.py --seed 26611502719397 --input 404000 --output 128 --save-histories histories.json
uv run --with tokenizers==0.22.2 python tools/c1_proxy.py --wire both --histories histories.json > warm.jsonl
```

For the cold comparison, temporarily set the native service's prefix RAM and disk budgets to zero. These commands restore the normal Compose settings when the run exits. The full warm and cold sweeps sent 260 generation requests, including warm-ups. [Per-phase GPU settings](results/c1-hardware.jsonl)

```sh
(
  set -e
  trap 'docker compose --env-file ../models/.env up --no-build -d --wait' EXIT
  DS4_PREFIX_RAM_MIB=0 DS4_PREFIX_DISK_MIB=0 docker compose --env-file ../models/.env up --no-build -d --wait
  uv run --with tokenizers==0.22.2 python tools/c1_proxy.py --wire both --mode cold --contexts 4096 16384 65536 --pairs 2 > cold.jsonl
)
```
