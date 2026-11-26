# Tile attention candidate selection

The short-edit profile attributed 12.3% of summed GPU kernel time to the segmented radix sort. The experimental CUDA kernel selects the best candidates in 4,096-key tiles and ranks their union in one block per query row. The 64-bit key retains the score and position tie break. For the second pass, tile winners use the memory that previously held input IDs. Single-row and wide-sort shapes retain their existing dispatch.

| Six concurrent requests | GPU control before | Enabled | GPU control after | Change vs. midpoint |
|---|---:|---:|---:|---:|
| 25,625-token edits, 1,500 output | 461.7 | 504.4 | 466.1 | +8.7% |
| 404k coding, 768 output | 318.7 | 333.4 | 320.0 | +4.4% |

Rates are aggregate generated tokens per second across six clients. Each row uses one binary with the option disabled, enabled and disabled again, a fresh process per leg and one warm-up wave. The short edits use the published aggregate-rate formula. The 404k rate runs from the earliest first token to the latest last token, excluding each client’s first token. The two workloads use different prompts, output budgets and timing conventions.

Both RTX PRO 6000 GPUs were capped at 375 W. The CPU socket reported a 225 W cap. All measured waves exercised six-way target verification, reused the full expected prefix and matched all six reference responses. The service recorded zero KV rotations during these throughput tests.

The isolated 8-row GPU benchmark measured the selector alone, using fixed synthetic 64-bit ranking keys on host GPU 1. With 16,384 input keys and 512 winners per row, elapsed time fell from 180.8 to 29.7 microseconds. Each number is the median of nine CUDA-event samples of 150 launches after warming the kernels. The baseline includes its required offset kernel.

Select tiled ranking for the optimized serving profile. Its enabled median exceeded both controls on both workloads, and the six individual enabled 404k waves each exceeded 332 aggregate tok/s.

The option for this experiment is `DS4_BATCH_TOPK=1`. Per-wave throughput, whole-wave throughput, image identity and raw capture hashes are in the [measurement record](results/attention-topk.json).

Rebuild from `checkpoint-2026-11-26-decode-01-attention-topk` in an engine checkout beside the stack’s `models/` directory. This override selects both the rebuilt binary and the enabled environment inside the container.

```sh
cargo build --release --locked && docker build -t ds4-flash-rs:attention-topk-checkpoint .
cat > /tmp/ds4-decode-override.json <<'EOF'
{
  "services": {
    "dsv41-flash-exl3": {
      "image": "ds4-flash-rs:attention-topk-checkpoint",
      "environment": {
        "DS4_BATCH_TOPK": "1"
      }
    }
  }
}
EOF
docker compose --env-file ../models/.env -f compose.yml -f compose.shared-prefix-lfu.yml -f /tmp/ds4-decode-override.json up --no-build -d --wait
```

Six 524,160-token prompts each completed 128 output tokens in three waves, matching the reference prompt-token and output-text hashes. Eight 523,776-token prompts each completed 512 output tokens through RAM rotation and again through forced disk rotation, matching all eight reference digests both times.

One 404k coding client at a 512-token output budget delivered 234.8 tok/s with the option disabled and 240.2 enabled (three measured warm waves each).

The full release suite passed all 86 tests on the two inference GPUs with the option enabled.
