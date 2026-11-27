# Coding-agent latency and decode on two RTX PRO 6000s

The native server reuses token IDs for ordinary text that appears unchanged in a later message. On the six timed follow-ups to a 404k-token conversation, the median wait for the first output fell to **0.482 s**. The two controls measured **0.847 s** and **0.930 s**. Each of the seven prompts, including warm-up, had a different final instruction. All 21 generated replies matched across runs. [Machine-readable results](results/agent-round3.json)

## Three experiments

| Change | Measurement | Control / candidate / return control | Decision |
|---|---|---:|---|
| Cache tokenized message text | First output on changing 404k history, seconds | 0.847 / 0.482 / 0.930 | Enable |
| Interleave independent DSpark Markov scores | Captured Pi reasoning, decode tok/s | 178.0 / 178.4 / 178.4 | Keep original scorer |
| Compute KV keys from bulk token bytes | First output on repeated 404k coding request, ms | 136.6 / 135.0 / 139.6 | Keep original lookup |

**Tokenized text.** The server caches whole ordinary-text spans separately from chat control tokens. It uses an exact text match within the serving tokenizer. It stores up to 64 entries and 32 MiB of text and token-ID payload in host RAM. Text shorter than 512 bytes bypasses the cache. New or edited text still uses the original tokenizer. The cache is process-local and starts empty after a restart.

The 404k replay extended a frozen two-message history with seven different user instructions. Every native token count matched the independent checkpoint tokenizer, from 404,137 to 404,172 tokens. Each count request preceded generation on that body. Median count endpoint latency in milliseconds was 445.1 / 1.9 / 466.2. The metric includes local HTTP, rendering and tokenization. Generation timing starts at its own POST. All measured arms restored the same number of KV tokens for each prompt and completed the same 16-token replies. An initial run was excluded because its first control restored 64 fewer KV tokens on two prompts. The reported run started after those disk entries existed for all arms.

**DSpark Markov head.** The CUDA kernel interleaved two independent dot products in one warp, keeping the original eight FMAs and reduction order for each vocabulary ID. Both paths passed a dedicated GPU check for chained predictions, ties across thread blocks, a partial final tile and nonfinite scores. The 404k decode rates were 256.0 / 256.8 / 256.2 tok/s. The Pi rate was within the variation between runs. Its code is preserved on the second experiment tag.

**KV prefix keys.** The Rust implementation updates SHA-256 in blocks and computes only the terminal digest for a saved target or drafter checkpoint. It tries the final RAM key first and falls back to the existing longest-prefix search. Target lookup and read measured 1.53 / 0.81 / 1.55 ms. Drafter lookup and read measured 1.55 / 0.81 / 1.56 ms. These phase measurements used six warm RAM restores of 404,096 tokens. Tests compare both target keys and drafter keys with the original byte encoding and include existing disk entries.

## Selected release

| One client | Previous release | Selected release | Previous release repeated |
|---|---:|---:|---:|
| Captured Pi turn, decode tok/s | 178.5 | 178.8 | 178.4 |
| 404,151-token coding, first output (ms) | 627.3 | 136.8 | 622.2 |
| 404,151-token coding, decode tok/s | 256.4 | 256.6 | 254.0 |

Six concurrent 404k coding histories measured 299.5 / 304.5 / 304.7 aggregate decode tok/s (previous / selected / previous) and 282.9 / 299.7 / 286.4 aggregate output tok/s including the whole request wave. Each client generated 512 tokens. Every output matched the saved per-agent reference.

A separate mixed sequence replayed the full Pi edit, two long retrievals and the structured edit before switching to 404k coding. The initial qualification measured 241.5 tok/s on that transition. Repeating the sequence on fresh processes gave 241.8 / 241.2 / 241.4 tok/s on 404k coding and 383.1 / 385.6 / 383.7 tok/s on the structured edit. The earlier throughput table uses the shorter Pi turn immediately before the 404k workload.

All throughput figures are medians of six warm native requests or waves per fresh engine. The captured Pi request has 2,786 input tokens and a 256-token reasoning continuation. The long request generates 512 tokens. Decode measures first-to-last generated output. Counting the entire single-client 404k request, throughput was 195.2 / 240.4 / 193.5 output tok/s. The 404k decode rate uses 511 intervals. The Pi rate uses 255. The cards ran at 375 W each and the CPU socket at 225 W. The existing 154 tok/s sequential weight-read ceiling applies to one target token per pass. The speculative rates above depend on accepted proposals. [Ceiling and assumptions](THEORY.md)

An earlier release run was discarded when its test list exposed a stale executable in a shared Cargo build directory. The reported release was rebuilt from the tagged source in an empty target directory. Its server and test executable hashes were checked before qualification. The clean release build reproduced the complete 4,103-token Pi edit, its 62-token tool-return reply and the 542-token following reply. It passed 95 tests with the model and both GPUs. It recovered the exact receipts at 266,174 and 520,362 tokens. It served three waves of six independent full windows, then eight agents with six resident and two swapped on each of the RAM and forced-disk tiers. Both tiers returned the reference replies with zero read or write failures. The 25,625-token edit measured 383.5 tok/s. Previous tool-return idle prefill remains enabled. [Prior tool-return measurements](TOOL-RETURN.md)

The three source checkpoints are `checkpoint-2026-11-26-agent3-01-text`, `checkpoint-2026-11-26-agent3-02-markov`, and `checkpoint-2026-11-26-agent3-03-prefix`. Raw samples and per-run settings are archived on this workstation under `~/.cache/ds4-flash-rs/profiles/agent-round3-20261126/`.

## Build and rollback

From the engine checkout, build the release in a separate worktree and start it with the selected profile.

```sh
git worktree add --detach ../ds4-agent-round3-build checkpoint-2026-11-26-agent3-release
(cd ../ds4-agent-round3-build && CARGO_TARGET_DIR=target cargo build --release --locked && docker build -t ds4-flash-rs:agent-coding-20261126 .)
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-round2.yml -f compose.coding.yml up --no-build -d --wait
```

To restore the preceding release, replace `compose.coding.yml` with `compose.agentic.yml` in the launch command.
