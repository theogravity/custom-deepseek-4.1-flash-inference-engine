# Decode experiments at 375 W

The starting service was `stable-2026-11-26-c1-375w`. A fresh readback on its native streaming endpoint measured **365.6 tok/s** for a single 25,625-token Python edit and **256.45 tok/s** for a single 404,151-token coding history. Both RTX PRO 6000s had 375 W power limits. The AMD socket reported 225 W. The clock offsets and the target weights, code-tuned DSpark weights, LFU policy and KV quotas stayed fixed.

The edit emits 1,500 tokens including reasoning. Its rate divides 1,500 by the interval from the first generated chunk to stream completion. The coding continuation emits 512 visible tokens and measures 511 first-to-last intervals. Each leg starts a fresh engine, warms the workload once, then measures six warm requests. The primary runs issue the edit before the long continuation. The timed edit and long-coding waves check the cached prefix and the saved output hashes. Six-client figures use aggregate throughput. [Benchmark definitions](C1-DECODE-EXPERIMENTS.md)

At these clocks, the measured bandwidth and 11.03 GB of active weights imply a 6.48 ms sequential weight-read floor for a single target token (154 target-only tok/s). The speculative engine can emit multiple verified tokens per pass. The number varies with the continuation. [Ceiling assumptions and measurements](THEORY.md)

## Experiments

1. **Longer repetition proposals.** On a single-client lookup miss, the Rust drafter can now project up to two periods, capped at 96 tokens. It requires a matching suffix of at least 12 tokens and three committed periods. Every field in the second projected period requires a third historical witness, and the target verifies the result. Across seven edits the lookup performed 112 verification rounds, down from 175. The 25.6k gain repeated at +4.88%. The 404k history following edits lost 1.23% in the initial comparison and 1.96% on repeat. With that history alone, fresh processes measured 259.23 / 260.26 / 259.07 tok/s. The reason for the ordering effect remains open. All three short coding holdouts stayed within 1% of their bracket midpoints. [Experiment, repeat, holdouts and capacity](results/c1-cycle.json)
2. **DSpark Markov shortlist.** The CUDA drafter rescored the top two raw logits in each 32-token group. The target still verified every proposal. The seven-position rank-256 microbenchmark fell from roughly 96.6 to 28.5 µs. At 404k the learned drafter needed 637 verification rounds across seven requests, up from 602, and throughput fell 4.19%. Edit throughput fell 4.46%. The selected engine keeps the full vocabulary scan. [Experiment, GPU checks and weight bounds](results/c1-markov.json)
3. **Share vocabulary-head weights across verification rows.** The packed CUDA head uses two eight-row tiles for 9–16 verification rows and four for wider batches. The target keeps each row’s original tensor-core accumulation and reduction order. The full-size synthetic GPU comparison checked 233,867,520 values across 17 shapes up through 512 rows. Every candidate output matched the single-tile reference bit for bit. At 48 rows the projection took 1.26 ms with tiling and 2.63–2.67 ms in the controls. The 404k single-client continuation used the learned drafter and verified at most eight rows. Six concurrent continuations could share the head weights. [Experiment, serving comparisons and capacity](results/c1-head.json)

Each cell lists disabled / enabled / disabled medians in tok/s. The percentage uses the average of the two control medians. The shortlist and head experiments both started with longer repetitions enabled. The shortlist stayed off in the head experiment.

| Experiment | C1 edit at 25.6k | C1 coding at 404k |
|---|---:|---:|
| Longer repetitions | 366.2 / **385.8** / 364.8 (+5.5%) | 255.7 / **252.9** / 256.4 (-1.2%) |
| Markov shortlist | 383.9 / **366.6** / 383.5 (-4.5%) | 253.1 / **242.2** / 252.4 (-4.2%) |
| Head row tiling | 384.0 / **386.8** / 383.4 (+0.8%) | 253.5 / **251.9** / 251.8 (-0.3%) |
| Head row tiling, new starts | 383.2 / **386.7** / 383.9 (+0.8%) | 252.8 / **252.4** / 253.0 (-0.2%) |

## Selected service

The cumulative comparisons use the original deployed image on either side of the final candidate. The six-client comparisons change only the vocabulary head while keeping longer repetitions enabled.

| Workload | Control / selected / control, tok/s |
|---|---:|
| C1 edit at 25.6k | 365.6 / **387.8** / 365.4 (+6.1%) |
| C1 coding at 404k after the edit | 257.0 / **251.6** / 256.0 (-1.9%) |
| C1 coding at 404k, fresh long-only processes | 259.2 / **258.6** / 258.7 (-0.1%) |
| Six edits at 25.6k, aggregate | 498.8 / **506.4** / 500.7 (+1.3%) |
| Six coding sessions at 404k, aggregate | 276.5 / **282.7** / 272.2 (+3.1%) |

The final head change was also tested on three short coding prompts with 512 output tokens. Their control / enabled / control medians in tok/s were Python Tetris 295.3 / 294.9 / 294.6, C JSON parser 348.2 / 348.3 / 347.8, Rust TTL cache 327.9 / 328.1 / 328.2. All matched their frozen hashes.

On the same workstation, the earlier six-client vLLM edit benchmark measured 485.0 aggregate tok/s with CPU experts disabled and 537.2–561.0 with them enabled. The selected Rust engine’s 506.4 tok/s is 5.7–9.7% below those CPU-enabled runs. Both stacks used the same edit requests and power limits. They used different drafters and could generate different continuations. [Earlier same-host comparison](EDIT-SCALING-EXPERIMENT.md#same-host-comparison)

Both retained candidates recovered every receipt at 266,174 and 520,362 prompt tokens and served three six-client waves filling six independent 524,288-token windows. Eight 523,776-input + 512-output requests returned the same outputs as serial controls with six resident and two swapped sessions. RAM and forced-disk tests each recorded eight reads and eight writes with zero I/O failures. During the first experiment, one disk trial counted nine requests. Its eight test-client outputs matched, and a clean eight-request rerun passed. The final head build passed all 90 release tests with both GPUs and the checkpoint.

On the selected service, six warm single-client requests per workload measured **386.7 tok/s on edits** and **252.8 tok/s at 404k after edits**. The service reported six resident slots, 32 logical sessions, 48 GiB RAM swap and 128 GiB disk swap. [Live image, flags and throughput](results/decode-round2-selected.json)

## Build and rollback

The stable release tag is `stable-2026-11-26-c2-375w`. The local selected image is `ds4-flash-rs:c2-verified-20261126`. To rebuild, check out `checkpoint-2026-11-26-c2-03-head` in a separate worktree, run `cargo build --release --locked`, then `docker build -t ds4-flash-rs:c2-verified-20261126 .` there. The rebuild with expanded test coverage matched every loadable section of the tested executable except its build ID. The ELF symbol string table also differed. Use the engine's main checkout for the serving commands.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-round2.yml up --no-build -d --wait
```

To restore the original 366 tok/s single-client profile, run the following command.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.c1-optimized.yml up --no-build -d --wait
```

The experiment checkpoints are `checkpoint-2026-11-26-c2-01-repetition`, `checkpoint-2026-11-26-c2-02-markov` and `checkpoint-2026-11-26-c2-03-head`. The second tag preserves the rejected implementation. Raw waves, checksums, GPU microbenchmarks and measured settings are archived on this machine in `~/.cache/ds4-flash-rs/profiles/decode-round2-20261126/`. [Edit benchmark](tools/edit_scaling.py) · [Frozen coding-history reproduction](LONG-CONTEXT-BENCHMARK.md#reproduce-the-coding-workload)
