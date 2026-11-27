# Benchmark materials

The latest latency experiment replays seven different follow-ups against the same 404,000-token synthetic conversation. The first request warms the process. The six timed requests vary the final instruction. The driver checks the exact serialized prompt, independent token count, generated output digest and absence of competing inference. It reports time to count and time to the first nonempty generated delta separately. [Results and measurement conditions](../AGENT-ROUND3.md) · [Individual requests, waves and cache-tier checks](agent-followups/measurements/)

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
