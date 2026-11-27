# Decode tuning on two 375 W RTX PRO 6000s

Throughput below is aggregate for six clients. Each control pair brackets the enabled run on the same executable. The 25,625-token edits generated 1,500 tokens per client. The independent 404k coding sessions generated 768 each. The CPU socket read back a 225 W cap.

| Experiment | 25.6k edits, enabled vs. controls (tok/s) | 404k coding, enabled vs. controls (tok/s) | Profile decision |
|---|---:|---:|---|
| [Tiled attention ranking](attention-topk.md) | 504.4 vs. 461.7 / 466.1 | 333.4 vs. 318.7 / 320.0 | Enabled |
| [Borrowed drafter inputs](drafter-inputs.md) | 503.5 vs. 504.3 / 502.9 | 328.5 vs. 327.3 / 333.1 | Experiment branch |
| [Reused index-key registers](paired-index.md) | Outside dispatch range | 335.9 vs. 333.4 / 335.0 | Experiment branch |

A second 404k comparison reversed the register-reuse run order. Enabled, disabled and enabled again measured 336.7, 333.1 and 331.5 aggregate tok/s. Tiled ranking was the only change with a consistent end-to-end gain across its workload comparisons.

The preceding selective CPU-expert experiment delivered 289.4 tok/s on the same 404k workload versus 320.9 and 324.9 for the GPU controls. The CPU implementation remains available at its experiment checkpoint. [CPU performance and precision](cpu-experts.md)

## Run or roll back

The selected image is `ds4-flash-rs:attention-topk-checkpoint`. From a clean checkout at `checkpoint-2026-11-26-decode-01-attention-topk`, build the pinned engine executable and image. The optimized Compose overlay is in the current main checkout.

```sh
cargo build --release --locked
docker build -t ds4-flash-rs:attention-topk-checkpoint .
```

From the engine directory on main, start the selected profile.

```sh
docker compose --env-file ../models/.env -f compose.yml -f deploy/profiles/decode-optimized.yml up --no-build -d --wait
```

To restore the previous shared-prefix LFU GPU image, run the original overlay.

```sh
docker compose --env-file ../models/.env -f compose.yml -f deploy/profiles/shared-prefix-lfu.yml up --no-build -d --wait
```

## Local Git checkpoints

- `checkpoint-2026-11-26-decode-00-cpu-experts`
- `checkpoint-2026-11-26-decode-01-attention-topk`
- `checkpoint-2026-11-26-decode-02-draft-views`
- `checkpoint-2026-11-26-decode-03-paired-index`

## Selected service validation

Six 404k coding clients with 512 output tokens each delivered 308.6 aggregate tok/s over three warm waves. In the same process, six clients then filled their 524,288-token windows in three waves. The two warm receipt waves delivered 248.9 and 265.5 aggregate tok/s. All 18 full-window prompt-token and output-text hashes matched the saved references. The active profile retains six residents, 32 logical sessions and 48 GiB RAM / 128 GiB disk for live KV. [Service validation](../results/decode/decode-selected-service.json)

A separate full-window timing repeated the transition from six coding sessions before each leg. With tiled ranking disabled, enabled and disabled again, the six warm full-window receipt waves in each leg had medians of **275.5 / 286.7 / 276.5 aggregate tok/s**. The inputs were 524,160 tokens with 128 output tokens each. All 126 full-window responses matched the references, with six residents and zero rotations during the waves. The preceding 404k coding medians at 512 output tokens were 297.2 / 306.9 / 302.9 aggregate tok/s.

Across the two selected-profile coding-to-full-window checks, the lowest sampled free VRAM on either inference GPU was 78 MiB. Eight full-window clients were verified separately with six residents and two rotated through RAM or disk.
