# Generation and drafters

The 40-layer target runs in two consecutive stages. Attention, hidden states, shared projections and the vocabulary head stay on the owning GPU. Target expert batches run across both cards. Decode and prompt batches of up to 128 rows use staged peer transfers. Wider prompt batches use direct peer access. Six routed experts are selected per layer. The CUDA kernels multiply the checkpoint's 3-bit EXL3 trellises and packed MXFP8 dense weights in place. The int4 Engram tables are memory-mapped on the host. Expert slots grow on demand and give back VRAM for temporary buffers and attention state.

The deployed profile uses the code-tuned DSpark drafter with up to seven proposals. It reads the three MTP shards from the pack's `drafter-code-k7` overlay and checks their tensor names, shapes and types. The released DSpark weights in the main checkpoint are also supported. A learned confidence head can shorten a draft when later positions are unlikely to match. When an edit copies text from the input, prompt lookup can propose up to 128 source tokens. The target scores the proposals, commits the matching prefix and supplies the next token. Speculative output is streamed to the client as it is verified.

| Setting | Deployed value | Purpose |
|---|---|---|
| `DS4_DRAFTER` | `code` | `released` selects the pack's original weights, `none` disables learned drafting |
| `DS4_MAX_SEQS` | 6 | Maximum GPU-resident requests, from 1 through 6 |
| `DS4_MAX_LIVE_SEQS` | 32 | Maximum admitted requests sharing the GPU slots, configurable up to 64 |
| `DS4_SWAP_QUANTUM_STEPS` | 32 | Number of prefill or decode steps a runnable request receives before it can yield a GPU slot |
| `DS4_SWAP_RAM_MIB`, `DS4_SWAP_DISK_MIB` | 49152, 131072 | On-demand host RAM and NVMe budgets for suspended live requests |
| `DS4_TIMING_RESTORES` | 0 | Set to 1 at launch to log one JSON event per completed target KV restore, including timings for each GPU and cache tier |
| `DS4_BATCH_DRAFT` | 1 | Evaluate the learned drafter for multiple ready requests together |
| `DS4_BATCH_TOPK` | 1 | Selects exact attention candidates in tiles for the qualified multirow shapes. [Benchmark](../benchmarks/reports/attention-topk.md) |
| `DS4_EXPERT_EVICTION` | lfu | Keeps frequently used experts on each card. Counts span the process lifetime. Ties use least-recent access. [Tested image and launch command](../benchmarks/reports/expert-cache.md#running-the-candidate-on-this-workstation) |
| `DS4_SPEC_K` | automatic | Defaults to 7 for code weights and 5 for released weights |
| `DS4_CONFIDENCE_THRESHOLD` | 0.4 | Use 0 to evaluate the whole learned draft |
| `DS4_PROMPT_LOOKUP`, `DS4_LOOKUP_K` | 1, 128 | Copy proposals from source tokens already present in the prompt |
| `DS4_OUTPUT_LOOKUP` | single | Use recent verified output for a single client. The concurrent coding overlay selects `repeated` to also draft repeated patterns from each active client’s own output. Weak concurrent matches fall back to DSpark. Select `0` to disable output lookup or `1` for unrestricted output lookup at all loads |
| `DS4_C1_OUTPUT_CONSENSUS` | 1 in the selected overlay | Use repeated output structure for a lone decoder. Set to 0 to disable single-client consensus. The concurrent mode is selected independently by `DS4_OUTPUT_LOOKUP=repeated`. |
| `DS4_EXTEND_REPETITION` | 1 in the selected overlay | Extend strongly supported repetition proposals to two periods and up to 96 tokens for a single client and for concurrent clients using `repeated`. |
| `DS4_HEAD_ROW_TILES` | 1 in the selected overlay | Reuse vocabulary-head weights across multiple eight-row tiles during target verification. |
| `DS4_PREFILL_MX_TILES`, `DS4_VERIFY_MX_TILES` | 4, 4 | Number of packed projection tiles per block during prompt ingestion and multi-token target verification |
| `DS4_PARALLEL_PREFILL_ROUTER` | 1 | Select routed experts across prompt rows using the CPU worker pool |
| `DS4_OVERLAP_PREFILL_ENGRAM` | 0 | Optionally overlap Engram reads with target GPU work during prefill |
| `DS4_ADAPTIVE_LOOKUP` | 1 | Select learned vs. source proposals from contiguous matches. Use 0 for fixed-width source lookup |
| `DS4_PREFIX_RAM_MIB`, `DS4_PREFIX_DISK_MIB` | 6144, 65536 | Compose prefix budgets in MiB. A standalone process defaults to 2048 MiB RAM. |
| `DS4_AGENT_TEXT_CACHE` | 1 | Reuse token IDs for unchanged text in later requests. [Long-history replay](../benchmarks/reports/agent-followups.md) |
| `DS4_IDLE_TOOL_PREFILL_MAX_TOKENS` | 32768 | Maximum tool-call history length eligible for prefill while the engine is idle. [Tool-return benchmark](../benchmarks/reports/tool-return.md) |
| `DS4_HOST_PREFIX_DIR` | `~/.cache/ds4-flash-rs/prefix` | Private host bind mount for restart-persistent KV and drafter windows |

From the repository root on the workstation, select the released drafter with the following command. Omit the environment overrides to return to the deployed profile.

```sh
DS4_DRAFTER=released docker compose --env-file ../models/.env -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml up --no-build -d --wait
```

For a target-only control, set both `DS4_DRAFTER=none` and `DS4_PROMPT_LOOKUP=0`. Standalone commands also accept `--drafter code|released|none` and `--spec-k N`. Compose sets the numerical and hardware switches used in the benchmarks. [Deployment configuration](../compose.yml)
