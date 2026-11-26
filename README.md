# ds4-flash-rs

A standalone Rust and CUDA inference engine for the Diffbot 3-bit EXL3 pack of DeepSeek V4.1 Flash on two RTX PRO 6000 Blackwell workstation cards. The server opens the original 48 safetensors shards and generates tokens locally. Rust owns checkpoint loading, tokenization, scheduling, sampling, and the model loop. The GPU runtime uses CUDA C++, tensor-core instructions and cuBLAS. The HTTP server uses Axum.

## Run on this workstation

The release build needs Rust, a C++ host compiler and CUDA 13 with sm_120 support. CUDA 13.5 and Rust 1.100 were used here. The Docker image packages the host-built executable and mounts the host toolkit read-only.

```sh
cd ~/llm/ds4-flash-rs
cargo build --release --locked
install -d -m 700 "${DS4_HOST_PREFIX_DIR:-$HOME/.cache/ds4-flash-rs/prefix}"
docker compose --env-file ../models/.env up --build -d --wait
```

This selects the Rust engine for the existing DeepSeek service on port 8001 and uses its existing secret for authentication. It replaces the DeepSeek container in the `models` Compose project. The Rust server drains accepted requests on shutdown. Compose allows up to four minutes before forcing a stop. Finish active requests before switching. If another rotation model holds the GPUs, stop that model with `model-cli stop` first. The Compose file selects host GPUs 1 and 2. The executable checks both model names. LiteLLM continues to serve clients on port 4000.

Launch a coding client from its project directory using one of these commands.

```sh
eval "$(model-cli codex dsv41-flash-exl3)"
eval "$(model-cli claude dsv41-flash-exl3)"
eval "$(model-cli pi dsv41-flash-exl3)"
```

The rotation menu displays the original DeepSeek label for both implementations. To see which binary is serving, run `docker inspect -f '{{.Config.Image}}' dsv41-flash-exl3`. The regular DeepSeek model entry points to the original serving recipe. A later `model-cli switch dsv41-flash-exl3` can start that recipe. Use the Compose command above to select Rust again. Use this command to restore the original service.

```sh
cd ~/llm
docker compose -p models -f models/dsv41-flash-exl3.yml up -d --wait
```

## Generation and drafters

The 40-layer target runs in two consecutive stages. Attention, hidden states, shared projections and the vocabulary head stay on the owning GPU. Large target expert batches run across both cards using staged peer transfers. Six routed experts are selected per layer. The CUDA kernels multiply the checkpoint's 3-bit EXL3 trellises and packed MXFP8 dense weights in place. The int4 Engram tables are memory-mapped on the host. Expert slots grow on demand and give back VRAM for temporary buffers and attention state.

The deployed profile uses the code-tuned DSpark drafter with up to seven proposals. It reads the three MTP shards from the pack's `drafter-code-k7` overlay and checks their tensor names, shapes and types. The released DSpark weights in the main checkpoint are also supported. A learned confidence head can shorten a draft when later positions are unlikely to match. When an edit copies text from the input, prompt lookup can propose up to 128 source tokens. The target scores the proposals, commits the matching prefix and supplies the next token. Speculative output is streamed to the client as it is verified.

| Setting | Deployed value | Purpose |
|---|---|---|
| `DS4_DRAFTER` | `code` | `released` selects the pack's original weights, `none` disables learned drafting |
| `DS4_SPEC_K` | automatic | Defaults to 7 for code weights and 5 for released weights |
| `DS4_CONFIDENCE_THRESHOLD` | 0.4 | Use 0 to evaluate the whole learned draft |
| `DS4_PROMPT_LOOKUP`, `DS4_LOOKUP_K` | 1, 128 | Copy proposals from source tokens already present in the prompt |
| `DS4_PREFIX_RAM_MIB`, `DS4_PREFIX_DISK_MIB` | 2048, 65536 | Separate RAM and disk cache budgets in MiB |
| `DS4_HOST_PREFIX_DIR` | `~/.cache/ds4-flash-rs/prefix` | Private host bind mount for restart-persistent KV and drafter windows |

For example, select the released drafter with the following command. Omit the environment overrides to return to the deployed profile.

```sh
DS4_DRAFTER=released docker compose --env-file ../models/.env up -d --wait
```

For a target-only control, set both `DS4_DRAFTER=none` and `DS4_PROMPT_LOOKUP=0`. Standalone commands also accept `--drafter code|released|none` and `--spec-k N`. Compose sets the numerical and hardware switches used in the benchmarks. [Deployment configuration](compose.yml)

## Prefix caching and API

The native prefix cache saves target attention state and the drafter's last 128 positions at completed prefill boundaries. Keys cover the token prefix, attention layout and numerical settings, checkpoint files and tokenizer. Draft sidecars also bind to the drafter weights. Entries carry checksums, disk writes use atomic renames, and the store bounds RAM and disk independently. A request can restore a shared prefix after a tool call or a process restart. The engine evaluates the suffix and generates the new completion. Set both cache budgets to 0 to disable persistence and cross-request memoization.

The worker executes one text request at a time and bounds the queue. It exposes OpenAI chat completions, Anthropic messages and token counting, model discovery and health checks. Both APIs support streaming, reasoning and DSML tool calls. Codex uses LiteLLM's Responses translation. The original recipe also supports vision and continuous batching. The Rust profile currently serves text with a configured limit of 524,288 tokens.

## Reproduce the measurements

```sh
python3 tools/clients.py
python3 tools/retrieval.py --records 1600
python3 tools/roofline.py --packed
# With DSV41_API_KEY or LITELLM_MASTER_KEY set in your shell
python3 tools/code_speed.py --rounds 7
python3 tools/code_speed.py --rounds 7 --tokens 3000 --edit-source ../model-cli/internal/selfupdate/selfupdate.go
```

The HTTP benchmark sends one request at a time directly to the authenticated inference server, with reasoning disabled and temperature zero. It consumes the SSE stream, uses the engine's completion token count and measures from first to last visible content. The benchmark reports first-token latency separately and excludes one warm-up per prompt. Each short new-code prompt fits inside a single prefill chunk. Source edits can use cached prefix state and source-token proposals. All reported tokens are generated and verified by the target.

To run the full Rust test suite, drain the server and select the two inference GPU UUIDs. The suite includes checkpoint tokenizer, real EXL3, attention, peer-copy, speculative rollback and sampling fixtures.

```sh
docker compose --env-file ../models/.env stop
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX PRO 6000/{print $2}' | paste -sd,)"
DS4_TEST_DEVICE=1 DS4_TEST_CUDA_DEVICE=1 DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3 cargo test --release --locked -- --include-ignored --test-threads=1
docker compose --env-file ../models/.env up -d --wait
```

## Measurements on this workstation

The serving binary and tested model revision are recorded in the [serving profile](results/serving-profile.json). The original native engine measured 26.1 tok/s on a 32-token prime-number continuation. Its four short raw prompts matched the original recipe's first token and one entire continuation. The optimized profile uses grouped tensor-core attention, FP8 dense arithmetic and parallel reductions. Greedy equivalence checks below compare drafting against the optimized target on specific prompts. [Initial benchmark](results/decode.json) · [Original reference comparison](results/greedy-comparison.txt) · [Hardware and bandwidth model](THEORY.md)

| Authenticated single-stream HTTP workload | Warm median output tok/s | Measurement |
|---|---:|---|
| New Python Tetris implementation | 303.7 | Six runs of 1,024 tokens |
| New C JSON parser | 351.1 | Six runs of 1,024 tokens |
| New Rust TTL cache | 330.5 | Six runs of 1,024 tokens |
| Go source edit, returned in full | 837.5 | Six runs of 1,700 tokens after warm-up |

Across the 18 warm new-code samples the median was **330.5 tok/s**. Each of these generations reached its length limit. The Go edit had a natural stop, changed the requested behavior and passed the original and case-insensitivity tests. Before and after restarting the service, every Go output had the same SHA-256. On the first request after restart both target and drafter restored 1,728 of 1,734 prompt tokens from disk. First-token latency was 1.48 s with cold model allocations and 22 ms on the next request. [New-code samples](results/clean-code.jsonl) · [Full edits](results/clean-edit.jsonl) · [After restart](results/api-disk-edit.jsonl) · [KV counters](results/restart-kv.txt)

Throughput depends on what the drafter can predict. The source edit retained most of its input and 1,652 tokens were accepted from source lookup. An uncached opening, sampling at a different temperature, an essay, and a long transcript can produce very different rates. A first Tetris request in a fresh process measured 90.5 tok/s in the initial HTTP qualification. [Experiment ledger](results/ledger.jsonl)

Both DSpark variants reproduced the identical 256-token Tetris continuation as the undrafted target under the same deployment arithmetic. With source lookup disabled, one warm sample measured 84.4 tok/s target-only, 269.8 with the released weights, and 300.6 with the code-tuned weights. The engine passed all 48 release tests, including the real CUDA and checkpoint fixtures. [Drafter parity](results/drafter-parity.json) · [Pinned serving configuration](results/serving-profile.json)

Codex, Claude Code, and Pi each read a fresh random file through a real tool call and returned the exact contents through LiteLLM. Native authentication, Anthropic thinking streams, and Pi's alternative DeepSeek route passed as well. A 28,538-token cold request retrieved a randomly chosen receipt in 37.23 s. A follow-up with a different instruction and 28,546 input tokens returned the same receipt in 0.319 s. [Client checks](results/drafter-clients.jsonl) · [Proxy retrieval](results/agent-retrieval.json)
