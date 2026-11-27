# Benchmarks

| What do you want to measure? | Reports |
|---|---|
| Coding-agent latency and task completion | [Changing 404k conversations](reports/agent-followups.md) · [Pi coding task](reports/pi-coding.md) · [Time after tool calls](reports/tool-return.md) |
| Context, concurrency and KV | [Long-context decode](reports/long-context.md) · [Concurrent code edits](reports/concurrent-edits.md) · [Shared prefixes](reports/shared-prefix.md) |
| Throughput ceilings and comparisons | [Hardware roofline](reports/roofline.md) · [vLLM comparison](reports/vllm-comparison.md) · [Single-client context sweep](reports/single-client-context.md) |
| Decode optimizations | [Single-client decode](reports/single-client-decode.md) · [Round one](reports/decode-round-1.md) · [Round two](reports/decode-round-2.md) |
| Attention, drafters and experts | [Tiled attention](reports/attention-topk.md) · [Paired index scoring](reports/paired-index.md) · [Drafter inputs](reports/drafter-inputs.md) · [Expert transfers](reports/expert-transfers.md) · [Expert cache](reports/expert-cache.md) · [CPU experts](reports/cpu-experts.md) |

The [workstation benchmark history](reports/workstation-history.md) includes the original power, cache-restore and full-window measurements. Raw results are grouped by workload in [results/](results/).

## Replay the long-history coding benchmark

The latest latency experiment replays seven different follow-ups against the same 404,000-token synthetic conversation. The first request warms the process. The six timed requests vary the final instruction. The driver checks the exact serialized prompt, independent token count, generated output digest and absence of competing inference. It reports time to count and time to the first nonempty generated delta separately. [Results and measurement conditions](reports/agent-followups.md) · [Individual requests, waves and cache-tier checks](agent-followups/measurements/)

Run these commands from the repository root with the engine running and its API key in `LITELLM_MASTER_KEY`. The first command creates the six synthetic histories from the benchmark seed. The same histories work with the long-decode and full-window tools.

```sh
uv run --with tokenizers==0.22.2 python tools/six_streams.py --seed 26611502719397 --input 404000 --output 128 --save-histories /tmp/ds4-histories.json
python3 tools/agent_followups.py --histories /tmp/ds4-histories.json --prime
python3 tools/agent_followups.py --histories /tmp/ds4-histories.json
uv run --with tokenizers==0.22.2 python tools/long_decode.py --histories /tmp/ds4-histories.json --clients 6 --output 512 --rounds 6
```

For the cache comparison, start fresh processes with `DS4_AGENT_TEXT_CACHE=0`, then `1`, then `0` in the coding Compose profile. Keep the same disk prefix directory and power settings. Prime the seven requests before the comparison. Check that each timed prompt restored the same number of KV tokens in all three arms. The raw A/B/A records use `a`, `on` and `b` in that order. The independent count measurements are in the `changing` directory.

The Pi task includes the driver, starting tree and held-out grader. A separate set of CUDA microbenchmarks tests kernels from their corresponding historical revisions. The revision table maps workstation commit IDs in older benchmark receipts to their published Git revisions. Source trees differ only where identifying metadata or receipt fields were sanitized. [Pi coding task](pi-edit/) · [CUDA microbenchmarks](micro/) · [Revision table](source-revisions.tsv)

The [workstation campaign archive](https://github.com/theogravity/custom-deepseek-4.1-flash-inference-engine/tree/archive/benchmark-campaigns/campaigns) preserves the earlier experiment drivers, including the rejected variants and per-kernel timing tables.

## Other benchmark commands

The roofline tool reads the model pack on disk. The direct HTTP benchmarks use a running native engine and its exported `DSV41_API_KEY` or `LITELLM_MASTER_KEY`. Saving or continuing frozen histories also needs the checkpoint tokenizer and the Python `tokenizers` package, as in the replay commands above.

```sh
python3 tools/roofline.py --packed
python3 tools/code_speed.py --rounds 7
python3 tools/six_streams.py --output 256 --rounds 7
python3 tools/six_streams.py --input 524160 --output 128 --rounds 2 --seed 417129047
python3 tools/kv_rotation.py --agents 8 --input 12000 --output 512 --expect-tier ram
```

The managed workstation also has `model-cli`, LiteLLM on port 4000, and the installed coding clients. The Go edit benchmark reads a fixture from the sibling `model-cli` checkout.

```sh
python3 tools/clients.py
python3 tools/retrieval.py --records 1600
python3 tools/code_speed.py --rounds 7 --tokens 3000 --edit-source ../model-cli/internal/selfupdate/selfupdate.go
```

`code_speed.py` sends one request at a time, with reasoning disabled and temperature zero. It consumes the SSE stream, uses the engine's completion token count and measures from first to last visible content. The benchmark reports first-token latency separately and excludes one warm-up per prompt. Each short new-code prompt fits inside a single prefill chunk. Source edits can use cached prefix state and source-token proposals. `six_streams.py` releases six clients together, checks output and completion IDs separately, and measures the interval when all six streams are producing output. Its aggregate decode rate uses the earliest first content and latest last content across the entire wave. Its wall rate also includes prompt ingestion. A fresh seed gives the six long histories distinct contents from the beginning.
