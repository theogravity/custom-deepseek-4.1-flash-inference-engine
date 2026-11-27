# Read drafter logits in place

The batched DSpark head produced one contiguous logits tensor, then copied each client’s rows to another allocation for Markov selection. At seven draft rows the logits copy moves 3.45 MiB per client. Including the confidence inputs, six seven-row proposals copy 24.0 MiB. The experiment borrows each client’s logits, hidden and mixture ranges directly from their batch allocations. The individual Markov chains, confidence calculations and per-client widths retain their existing arithmetic.

| Six concurrent requests | GPU control before | Enabled | GPU control after | Change vs. midpoint |
|---|---:|---:|---:|---:|
| 25,625-token edits, 1,500 output | 504.3 | 503.5 | 502.9 | -0.02% |
| 404k coding, 768 output | 327.3 | 328.5 | 333.1 | -0.5% |

Rates are aggregate generated tokens per second across six clients. Each row uses one binary with the option disabled, enabled and disabled again, a fresh process per leg and one warm-up wave. The short edits use the published aggregate-rate formula. The 404k rate runs from the earliest first token to the latest last token, excluding each client’s first token. The two workloads use different prompts, output budgets and timing conventions.

Both RTX PRO 6000 GPUs were capped at 375 W. The CPU socket reported a 225 W cap. All measured waves exercised six-way target verification, reused the full expected prefix and matched all six reference responses. The service recorded zero KV rotations during these throughput tests.

Keep copied drafter inputs in the selected serving profile. Eliminating the copies did not establish a repeatable throughput gain across these coding workloads. The hardware fixture with one-, three- and two-row clients verified exact proposal IDs and confidence scores for nonoverlapping row ranges. The experimental path is available at the checkpoint.

The option for this experiment is `DS4_DRAFT_VIEWS=1`. Per-wave throughput, whole-wave throughput, image identity and raw capture hashes are in the [measurement record](../results/decode/draft-views.json).

Rebuild from `checkpoint-2026-11-26-decode-02-draft-views` in an engine checkout beside the stack’s `models/` directory. Run the commands below in that tagged checkout, which has its Compose overlays at the root. This override selects both the rebuilt binary and the enabled environment inside the container.

```sh
cargo build --release --locked && docker build -t ds4-flash-rs:draft-views-checkpoint .
cat > /tmp/ds4-decode-override.json <<'EOF'
{
  "services": {
    "dsv41-flash-exl3": {
      "image": "ds4-flash-rs:draft-views-checkpoint",
      "environment": {
        "DS4_DRAFT_VIEWS": "1",
        "DS4_BATCH_TOPK": "1"
      }
    }
  }
}
EOF
docker compose --env-file ../models/.env -f compose.yml -f compose.shared-prefix-lfu.yml -f /tmp/ds4-decode-override.json up --no-build -d --wait
```

The full release suite passed all 87 tests on the two inference GPUs with the option enabled.
