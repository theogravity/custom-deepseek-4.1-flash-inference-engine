# Single-client decode on two RTX PRO 6000s

The reference service at `stable-2026-11-26-decode-375w` delivered **293.6 tok/s** on the 25,625-token Python edit and **235.9 tok/s** on the 404,151-token coding history. These are warm single-client rates measured through the native streaming endpoint. Each 6000 had a 375 W limit. The CPU socket reported a 225 W cap throughout the comparisons.

The edit rewrites functions 50–90 and emits 1,500 tokens including reasoning. Its published metric divides 1,500 by the time from the first generated chunk to stream completion. The long-context workload emits 512 visible tokens and measures the 511 intervals between the first and last text chunks. Both rates exclude time to first token. Each control and candidate started in a fresh process with LFU, the same prefix-cache capacity and the code-tuned DSpark model. Each primary rate is the median of six warm requests after one warm-up. All timed experiment and confirmation waves reused the expected prefix, generated the reference output, and completed with zero KV rotations. The initial reference edit matched the saved reasoning and code hashes. Its per-request prefix reuse was not recorded.

## Three experiments

Rates are control / enabled / control in tok/s. Percentages compare the enabled median with the mean of its two control medians.

| Implementation | 25.6k Python edit | 404k coding |
|---|---:|---:|
| Wide attention ranking | 293.8 / **294.0** / 293.8 (+0.1%) | 237.1 / **237.1** / 236.9 (+0.0%) |
| Warp-shared expert weights | 293.9 / **293.8** / 293.4 (+0.1%) | 235.9 / **237.1** / 236.5 (+0.4%) |
| Verified output repetition | 293.5 / **366.5** / 293.9 (+24.8%) | 236.0 / **256.9** / 237.1 (+8.6%) |

1. **Wide attention ranking.** Recursively select the strongest 512 candidates in each 4,096-entry tile. With eight rows of width 202,112 the isolated selection fell from about 111 to 82 µs, but at five rows of width 404,224 it rose from about 82 to 91 µs. The server showed no material gain. [Checkpoint evidence](results/c1-wide.json)
2. **Warp-shared expert weights.** Routed projections occupied 40.8% of summed GPU kernel time in a fresh one-client edit trace. The new CUDA kernel loads a packed word and gets the previous word from a neighboring lane. All tested partial outputs matched bit for bit. The initial build changed instruction scheduling outside its target range. The final revision preserved the original GPU instruction sequence in all 12 unaffected specializations. A separate long-only repeat measured 239.64 / 239.65 / 239.31 tok/s. The deployed expert kernel remains selected. [Checkpoint evidence](results/c1-expert-r2.json)
3. **Verified output repetition.** When one client is decoding and a prompt lookup misses, the Rust drafter looks for repeated blocks in committed output. It can copy fields that agree and extrapolate a single decimal token when three earlier values establish the same increment. It defers weak, unconfirmed output matches to the learned DSpark drafter. Each proposed token passes through the usual target verification. [Checkpoint evidence](results/c1-lookup.json)

## Confirmation and other clients

The independent 25.6k edit repeat measured 293.8 / **365.4** / 293.6 tok/s. The independent 404k coding repeat measured 235.6 / **256.3** / 236.5 tok/s.

Across seven edit requests, the total number of verification rounds fell from 1,736 to 1,225. The target accepted 6,293 of 7,042 lookup proposals under the new policy. At 404k the baseline accepted just 56 of 588 lookup proposals. All speculative proposals on that continuation came from the learned drafter.

Completion lengths and prompt identities were fixed within each comparison. The short prompts used a 512-token output budget. The six-client figures are aggregate rates for the same workloads as the primary single-client trials, using six different edits or six independent coding histories. Each six-client leg ran the edits first and then the long histories in the same process, so the populated expert cache carried over.

| Workload | Control before | Enabled | Control after | Change |
|---|---:|---:|---:|---:|
| C1 · Python Tetris | 295.8 | 295.0 | 295.8 | -0.2% |
| C1 · C JSON parser | 339.3 | 347.7 | 338.8 | +2.6% |
| C1 · Rust TTL cache | 327.3 | 327.8 | 327.2 | +0.2% |
| C6 · 25.6k edits | 494.8 | 498.7 | 498.7 | +0.4% |
| C6 · 404k coding | 275.8 | 276.4 | 277.8 | -0.2% |

Exactly three receipts were recovered at each of 266,174 and 520,362 input tokens. Six independent 524,288-token windows completed three waves with matching prompt and output hashes. Eight full-window clients using a shared register with distinct receipts completed with six GPU residents and two swapped sessions. The RAM and forced-disk tests each recorded eight writes and eight reads with zero I/O failures. All 89 release tests passed with the actual checkpoint and both GPUs available.

## Run, reproduce or roll back

The selected local image is `ds4-flash-rs:c1-consensus-20261126`. To rebuild that image, use source tag `checkpoint-2026-11-26-c1-03-consensus`, run `cargo build --release --locked` from the engine checkout, then run `docker build -t ds4-flash-rs:c1-consensus-20261126 .`. The serving and rollback commands use the main engine checkout.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.c1-optimized.yml up --no-build -d --wait
```

To restore the previous qualified image and drafter policy, run the following command.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-optimized.yml up --no-build -d --wait
```

For the short edit, run `python3 tools/edit_scaling.py --clients 1 --waves 7` with the existing engine credential in the environment. For long coding, generate the frozen histories using the documented seed, then run `python tools/long_decode.py --histories histories.json --clients 1 --output 512 --rounds 6` in the documented Python environment. Both clients stream from the native endpoint. The raw measurements, CUDA microbenchmarks and profiler capture are archived locally in `~/.cache/ds4-flash-rs/profiles/c1-20261126/`. [Long coding reproduction](LONG-CONTEXT-BENCHMARK.md#reproduce-the-coding-workload)

Each experiment has an annotated Git tag.

- `checkpoint-2026-11-26-c1-01-wide`
- `checkpoint-2026-11-26-c1-02-expert`
- `checkpoint-2026-11-26-c1-03-consensus`

## Active service

The selected Compose profile returned the expected answer to the smoke request and reported six resident slots, 32 logical sessions, 48 GiB of RAM swap and 128 GiB of disk swap. On that running service the six warm single-client requests per workload measured **366.0 tok/s** on the 25.6k edit and **256.3 tok/s** at 404k. [Image identity, limits and live results](results/c1-selected-service.json)
