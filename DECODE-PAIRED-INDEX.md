# Share compressed index keys between verification rows

The 404k profile attributed 16.9% of summed GPU kernel time to the compressed key scorer. The selected CUDA variant keeps both lanes of a packed FP32 instruction on adjacent key columns. It evaluates two consecutive queries in sequence using the same dequantized key registers. The dispatch uses eight-lane groups on full, physically backed scans with 4–8 query rows and at least 65,536 keys. It preserves each row’s causal mask, dot-product reduction and head weights. Candidate-filtered rows retain the existing kernel because their block lists can differ.

| Six concurrent requests | GPU control before | Enabled | GPU control after | Change vs. midpoint |
|---|---:|---:|---:|---:|
| 404k coding, 768 output | 333.4 | 335.9 | 335.0 | +0.5% |

Rates are aggregate generated tokens per second across six clients. Each row uses one binary with the option disabled, enabled and disabled again, a fresh process per leg and one warm-up wave. The 404k rate runs from the earliest first token to the latest last token, excluding each client’s first token.

Both RTX PRO 6000 GPUs were capped at 375 W. The CPU socket reported a 225 W cap. All measured waves exercised six-way target verification, reused the full expected prefix and matched all six reference responses. The service recorded zero KV rotations during these throughput tests.

On GPU 1, two packing layouts were screened at 65,537, 202,112 and 404,224 columns, each with 4, 5, 7 and 8 query rows. Each cell used seven CUDA-event samples of 20 launches. For the observed 202,112-column, eight-row shape, packing two queries into one FP32 pair took 1,109.9 µs with eight lanes, versus 729.6 and 732.1 µs in the controls. Keeping adjacent key columns paired and reusing their registers across queries took 698.0 µs, versus 732.2 and 734.7 µs. This second layout went into the serving test. The 25.6k edit workload cannot enter the new dispatch. The first implementation is recorded at commit `5a333bd`.

A second 404k comparison reversed the run order, with a fresh process, one warm-up and four measured waves each time. Enabled / disabled / enabled delivered **336.7 / 333.1 / 331.5** aggregate tok/s. All 72 measured outputs matched their references.

Keep the existing packed index scorer in the selected profile. Register reuse reduced time for the isolated long scans, and its first 404k serving median was 0.5% above the control midpoint. In the reverse-order run, one enabled leg beat the control and the other fell below it. The change stays available on its experiment branch. The measurements support a kernel speedup, while the six-client application gain is too small to distinguish reliably here.

The option for this experiment is `DS4_PAIRED_INDEX=1`. Per-wave throughput, whole-wave throughput, image identity and raw capture hashes are in the [measurement record](results/paired-index.json).

Rebuild from `checkpoint-2026-11-26-decode-03-paired-index` in an engine checkout beside the stack’s `models/` directory. This override selects both the rebuilt binary and the enabled environment inside the container.

```sh
cargo build --release --locked && docker build -t ds4-flash-rs:paired-index-checkpoint .
cat > /tmp/ds4-decode-override.json <<'EOF'
{
  "services": {
    "dsv41-flash-exl3": {
      "image": "ds4-flash-rs:paired-index-checkpoint",
      "environment": {
        "DS4_PAIRED_INDEX": "1",
        "DS4_BATCH_TOPK": "1"
      }
    }
  }
}
EOF
docker compose --env-file ../models/.env -f compose.yml -f compose.shared-prefix-lfu.yml -f /tmp/ds4-decode-override.json up --no-build -d --wait
```

Six 524,160-token prompts each completed 128 output tokens in three waves, matching the reference prompt-token and output-text hashes. Eight 523,776-token prompts each completed 512 output tokens through RAM rotation and again through forced disk rotation, matching all eight reference digests both times.

One 404k coding client at a 512-token output budget delivered 239.7 tok/s with the option disabled and 241.6 enabled (three measured warm waves each).

The full release suite passed all 86 tests on the two inference GPUs with the option enabled.
