# ds4-flash-rs

A standalone Rust and CUDA inference engine for the Diffbot 3-bit EXL3 pack of DeepSeek V4.1 Flash on two RTX PRO 6000 Blackwell workstation cards. The server opens the original 48 safetensors shards and generates tokens locally. Rust owns checkpoint loading, tokenization, scheduling, sampling, and the model loop. The GPU runtime uses CUDA C++, tensor-core instructions and cuBLAS. The HTTP server uses Axum.

The published 1-, 4- and 6-client workload has been replayed on this engine at both power limits. [Native and published benchmarks](UPSTREAM-COMPARISON.md)

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
| `DS4_SPEC_K` | automatic | Defaults to 7 for code weights and 5 for released weights |
| `DS4_CONFIDENCE_THRESHOLD` | 0.4 | Use 0 to evaluate the whole learned draft |
| `DS4_PROMPT_LOOKUP`, `DS4_LOOKUP_K` | 1, 128 | Copy proposals from source tokens already present in the prompt |
| `DS4_OUTPUT_LOOKUP` | single | Draft from recent verified output when one client is decoding. Select 0 to disable or 1 to enable at all loads |
| `DS4_PREFILL_MX_TILES`, `DS4_VERIFY_MX_TILES` | 4, 4 | Number of packed projection tiles per block during prompt ingestion and multi-token target verification |
| `DS4_PARALLEL_PREFILL_ROUTER` | 1 | Select routed experts across prompt rows using the CPU worker pool |
| `DS4_OVERLAP_PREFILL_ENGRAM` | 0 | Optionally overlap Engram reads with target GPU work during prefill |
| `DS4_ADAPTIVE_LOOKUP` | 1 | Select learned vs. source proposals from contiguous matches. Use 0 for fixed-width source lookup |
| `DS4_PREFIX_RAM_MIB`, `DS4_PREFIX_DISK_MIB` | 2048, 65536 | Separate RAM and disk cache budgets in MiB |
| `DS4_HOST_PREFIX_DIR` | `~/.cache/ds4-flash-rs/prefix` | Private host bind mount for restart-persistent KV and drafter windows |

For example, select the released drafter with the following command. Omit the environment overrides to return to the deployed profile.

```sh
DS4_DRAFTER=released docker compose --env-file ../models/.env up -d --wait
```

For a target-only control, set both `DS4_DRAFTER=none` and `DS4_PROMPT_LOOKUP=0`. Standalone commands also accept `--drafter code|released|none` and `--spec-k N`. Compose sets the numerical and hardware switches used in the benchmarks. [Deployment configuration](compose.yml)

## Prefix caching and API

The native prefix cache saves target attention state and the drafter's last 128 positions at completed prefill boundaries. Keys cover the token prefix, attention layout and numerical settings, checkpoint files and tokenizer. Draft sidecars also bind to the drafter weights. Entries carry checksums, disk writes use atomic renames, and the store bounds RAM and disk independently. A request can restore a shared prefix after a tool call or a process restart. The engine evaluates the suffix and generates the new completion. Set both cache budgets to 0 to disable persistence and cross-request memoization.

The worker serves up to **six concurrent text requests of 524,288 tokens each**, for **3,145,728 total context positions**. Input and generated output share each request's window. Each request has independent sliding-window KV, compressed KV, drafter state and sampling. At full size the six sets of compressed target KV and sliding windows occupy 3,252 MiB on the first inference GPU and 2,178 MiB on the second. Drafter windows, index buffers, activations, model weights and expert slots use additional VRAM. The 2 GiB RAM and 64 GiB disk prefix tiers store reusable snapshots separately.

The scheduler advances each GPU-resident runnable prompt by up to 64 input tokens per round and batches ready requests for drafting and target verification. A small request can start decoding while another is ingesting a long prompt. At the default settings, up to 32 requests can participate in rotation and another 32 can wait in the bounded intake queue. A full intake queue returns HTTP 429. The authenticated `/metrics` endpoint exposes resident, waiting and suspended requests, processed GPU positions, and successful six-request target passes.

### Live KV offload

When another runnable client needs a GPU slot, the worker saves one request's committed target KV and learned drafter state. It keeps that request's sampler, token history, stream and pending output alive. It releases the GPU buffers and lets the waiting client advance. On the next turn the suspended request resumes at the same token position. With six or fewer runnable clients the worker can serve each on a resident slot.

The backing store places each image in RAM if space is available, then uses an anonymous file on the NVMe drive mounted at `/prefix/live`. It checks disk writeback before releasing the GPU state and verifies the checksum on restore. Completion and cancellation release the backing space. These files belong to ongoing connections and disappear when their owner exits. Reusable prompt snapshots survive a service restart in the separate prefix cache.

The live tiers use the original LMCache budgets of **48 GiB RAM and 128 GiB disk**. With the separate prefix tier budgets, the combined payload limits are 50 GiB RAM and 192 GiB disk. A full 524,288-position suspended target image is about 900 MiB, plus drafter state. GPU staging, live token histories and the operating system's reclaimable file cache use additional host memory. If both live tiers fill, resident requests can finish and release their slots. The `swap` object in `/metrics` reports charged/peak bytes, reads, writes and failures for the two tiers; `rotations` and `resumes` count successful slot changes.

The server exposes OpenAI chat completions, Anthropic messages and token counting, model discovery and health checks. Both chat APIs support streaming, reasoning and DSML tool calls. Codex uses LiteLLM's Responses translation. The Rust profile serves text. The original recipe also supports vision.

### Requests at the context limit

The server tokenizes the fully rendered request, including role markers, tool definitions and reasoning instructions. The **524,288 positions** include the prompt and generated tokens. Context validation runs before model admission and before a streaming response starts.

| Request | Result |
|---|---|
| 524,287 input, 1 requested output | HTTP 200, one output token fits |
| 524,287 input, 2 requested output | HTTP 200, output capped at one token |
| 524,288 or more input | HTTP 400 with the rendered length and the maximum usable generation prompt of 524,287 |
| Explicit output budget greater than 524,288 | HTTP 400 with the requested budget and available room |
| Native `POST /v1/messages/count_tokens` with 524,289 input | HTTP 200 with the full count, zero inference admissions |

For a valid prompt, the native service caps an otherwise valid output budget at the remaining room, preserving the original serving recipe's behavior. When it caps the budget, it sends `x-ds4-effective-max-tokens` and either `x-ds4-requested-max-tokens` or `x-ds4-default-max-tokens`. LiteLLM delivered the capped output and usage but omitted those custom response headers in the tested routes.

OpenAI chat and LiteLLM Responses report `error.code=context_length_exceeded`. Anthropic Messages reports `invalid_request_error` and includes the measured length in the message. Stock LiteLLM v1.109.0 converted the symbolic OpenAI code to `"400"`. The local proxy image preserves the code so Codex recognizes the error. On the workstation, all 12 rejected cases returned HTTP 400 before streaming and produced zero native admissions, batch steps, prefix reuse or swaps. All 10 valid cases returned one generated token. [Direct and proxy boundary checks](results/context-limits-live.jsonl) · [Installed Codex classification](results/codex-context-classification.json) · [Release tests](results/test-context-release.txt)

## Reproduce the measurements

```sh
python3 tools/clients.py
python3 tools/retrieval.py --records 1600
python3 tools/roofline.py --packed
# With DSV41_API_KEY or LITELLM_MASTER_KEY set in your shell
python3 tools/code_speed.py --rounds 7
python3 tools/code_speed.py --rounds 7 --tokens 3000 --edit-source ../model-cli/internal/selfupdate/selfupdate.go
python3 tools/six_streams.py --output 256 --rounds 7
python3 tools/six_streams.py --input 524160 --output 128 --rounds 2 --seed 417129047
python3 tools/kv_rotation.py --agents 8 --input 12000 --output 512 --expect-tier ram
```

`code_speed.py` sends one request at a time, with reasoning disabled and temperature zero. It consumes the SSE stream, uses the engine's completion token count and measures from first to last visible content. The benchmark reports first-token latency separately and excludes one warm-up per prompt. Each short new-code prompt fits inside a single prefill chunk. Source edits can use cached prefix state and source-token proposals. `six_streams.py` releases six clients together, checks output and completion IDs separately, and measures the interval when all six streams are producing output. Its aggregate decode rate uses the earliest first content and latest last content across the entire wave. Its wall rate also includes prompt ingestion. A fresh seed gives the six long histories distinct contents from the beginning.

To run the full Rust test suite, drain the server and select the two inference GPU UUIDs. The suite includes checkpoint tokenizer, real EXL3, attention, peer-copy, speculative rollback and sampling fixtures.

```sh
docker compose --env-file ../models/.env stop
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX PRO 6000/{print $2}' | paste -sd,)"
DS4_TEST_DEVICE=1 DS4_TEST_CUDA_DEVICE=1 DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3 cargo test --release --locked -- --include-ignored --test-threads=1
docker compose --env-file ../models/.env up -d --wait
```

## Measurements on this workstation

At the 600 W GPU setting, one client generating 128 verified receipt tokens exceeded 300 output tok/s across all eight prompt lengths. With 524,160 input tokens it reached 322.9 tok/s direct and 322.0 via LiteLLM on OpenAI Chat. On the upstream Python edit, the native service reached 278 tok/s at C1 and 459 tok/s aggregate at C6. [Direct and proxied rates by context](C1-BENCHMARK.md) · [Upstream comparison](UPSTREAM-COMPARISON.md)

The original workstation benchmarks and the full C1 proxy sweep used both RTX PRO 6000 Blackwell cards with a **375 W limit per card**, **+450 MHz P0 core offset**, **+4,000 MHz P0 VRAM offset**, **0% voltage boost**, and **PowerMizer Auto**. GPU and VRAM clock locking were disabled. The workstation owner confirmed the historical configuration. The saved LACT settings and a live power-limit readback agree on both cards. The screenshot’s 3,015 MHz core, 15,365 MHz VRAM, 320.1 W and temperature values were instantaneous readings. [GPU settings and saved voltage/frequency points](results/benchmark-gpu-settings.json) · [Owner’s screenshot](results/rtx-pro-6000-lact.png)

On November 25 at about 20:24 Pacific, the workstation owner raised both inference cards to **600 W** with the same clock offsets. The driver reported 600 W requested and enforced on each. [600 W settings](results/benchmark-gpu-settings-600w.json) · [Updated screenshot](results/rtx-pro-6000-lact-600w.png)

The serving binary and tested model revision are recorded in the [serving profile](results/serving-profile.json). The original native engine measured 26.1 tok/s on a 32-token prime-number continuation. Its four short raw prompts matched the original recipe's first token and one entire continuation. The optimized profile uses grouped tensor-core attention, packed MXFP8 weights with BF16 tensor-core multiplication and FP32 accumulation, and parallel reductions. Greedy equivalence checks below compare drafting against the optimized target on specific prompts. [Initial benchmark](results/decode.json) · [Original reference comparison](results/greedy-comparison.txt) · [Hardware and bandwidth model](THEORY.md)

| Single-stream workload | Route | Warm median output tok/s | Measurement |
|---|---|---:|---|
| New Python Tetris implementation | LiteLLM | 303.6 | Six runs of 1,024 output tokens |
| New C JSON parser | LiteLLM | 350.9 | Six runs of 1,024 output tokens |
| New Rust TTL cache | LiteLLM | 331.0 | Six runs of 1,024 output tokens |
| Python rewrite, 25,598 input tokens | Direct | 308.1 | Six runs capped at 512 output tokens |
| Go source edit, returned in full | Direct | 866.5 | Six runs of 1,700 output tokens to a natural stop |

Across the 18 warm new-code samples through LiteLLM the median was **331.0 tok/s**. Each of these generations reached its length limit. The Go edit had a natural stop, changed the requested behavior and passed the original and case-insensitivity tests. Before and after restarting the service, every Go output had the same SHA-256. On the first request after restart both target and drafter restored 1,728 of 1,734 prompt tokens from disk. First-token latency was 1.48 s with cold model allocations and 22 ms on the next request. [New-code samples](results/hybrid-proxy.jsonl) · [Full edits](results/hybrid-go.jsonl) · [After restart](results/api-disk-edit.jsonl) · [KV counters](results/restart-kv.txt)

Throughput depends on what the drafter can predict. The source edit retained most of its input. On the adaptive profile, 1,625 output tokens matched source proposals. An uncached opening, sampling at a different temperature, an essay, and a long transcript can produce very different rates. A first Tetris request in a fresh process measured 90.5 tok/s in the initial HTTP qualification. [Experiment ledger](results/ledger.jsonl)

Both DSpark variants reproduced the identical 256-token Tetris continuation as the undrafted target under the same deployment arithmetic. With source lookup disabled, one warm sample measured 84.4 tok/s target-only, 269.8 with the released weights, and 300.6 with the code-tuned weights. The single-request build passed all 49 release tests, including the real CUDA and checkpoint fixtures. [Drafter parity](results/drafter-parity.json) · [Pinned serving configuration](results/serving-profile.json)

Codex, Claude Code, and Pi each read a fresh random file through a real tool call and returned the exact contents through LiteLLM. Native authentication, Anthropic thinking streams, and Pi's alternative DeepSeek route passed as well. A 28,538-token cold request retrieved a randomly chosen receipt in 37.23 s. A follow-up with a different instruction and 28,546 input tokens returned the same receipt in 0.319 s. [Client checks](results/hybrid-clients.jsonl) · [Proxy retrieval](results/agent-retrieval.json)

For the LiteLLM OpenAI route measurements, response-cache reads and writes were disabled. All 21 requests, including warm-ups, had a corresponding new backend prefill. Their prompt sizes, completion lengths and output hashes matched the direct control in order. [Proxy timings](results/hybrid-proxy.jsonl) · [Native execution counters](results/hybrid-proxy-backend.txt) · [Direct control](results/clean-code.jsonl)

```sh
python3 tools/code_speed.py --base http://127.0.0.1:4000/v1 --model 'DeepSeek V4.1 Flash (openai)' --bypass-proxy-cache --rounds 7
```

### Six concurrent agents

Six independent coding requests each generated 256 tokens. Across the six warm waves, aggregate decode measured **395.5 tok/s** and median per-stream decode was **84.7 tok/s**. The overlap and target-pass counters showed genuine simultaneous decoding, and all 42 outputs across the warm and first waves matched the serial controls byte for byte. A separate single-client test of the six-slot scheduler through LiteLLM measured 330.5 tok/s across 18 warm new-code samples. [Concurrent coding](results/six-coding.jsonl) · [Serial controls](results/six-serial-control.jsonl) · [Single-client proxy](results/six-slot-single-client.jsonl)

Six requests with a restored common document and distinct late receipts each consumed **524,160 input + 128 output = 524,288 tokens**. Every wave had 18 successful six-request target passes. Each request returned its own correct receipts. The six warm waves had a median aggregate decode rate of **361.1 tok/s**, a median individual stream rate of **60.9 tok/s**, and a median end-to-end wall time of **5.36 s** for the wave, including cache restore. The first wave in the freshly started process took 13.53 s. The aggregate rate including restore was 143.4 tok/s across warm waves. [Full-window streams and timings](results/six-full-restored.jsonl)

In the deployed offload and context-validation build, the first full-window stream started returning text after a median **1.11 s**, down from 3.22 s in the earlier run. The other five began later as their prefixes restored. This wider delivery interval brought the first-text-to-last-text aggregate to **177.0 tok/s**. Median per-stream delivery was **61.2 tok/s**, the wave took **5.43 s** including restore, and every output matched the earlier build. Six short coding clients delivered **394.4 tok/s** aggregate on this release. [Final full-window run](results/final-six-full-shared.jsonl) · [Final coding run](results/final-six-coding.jsonl) · [Actual coding-client checks](results/context-clients.jsonl)

With six unrelated 404,000-token inputs, the engine ingested all **2,424,000 input tokens from scratch**. All six agents retrieved their early and late receipts, emitted 128 verified tokens apiece and overlapped during decode. Total wall time was **3,714.6 s** (61 min 55 s), about **653 input tok/s** including the final answers. Each worker log reported zero reused prompt tokens. [Cold independent wave](results/six-independent-404k.jsonl) · [Per-request cache counters](results/six-independent-404k-backend.txt)

After a process restart, each agent continued its own 404,000-token transcript with its actual prior response. The engine restored **403,968 positions per request** from disk, then evaluated a combined 721,152 new prompt positions. Each of the six generated 128 tokens at the exact 524,288-token limit and returned a previously undisclosed receipt from its original prompt plus a new receipt from its own extension. The wave took **1,588.1 s** (26 min 28 s); an identical repeat took **9.37 s** after restoring 524,096 positions per agent. Both waves executed six-request target passes. In the repeat, one stream started 5.18 s ahead of the others, so its delivery interval included their restores. Aggregate delivery from the first visible text to the last measured 100.0 tok/s; five of the six streams individually delivered 52–54 tok/s. [Independent full-window continuations](results/six-independent-full.jsonl) · [Restored positions](results/six-independent-full-backend.txt) · [Three LiteLLM waves, all 18 outputs identical to direct](results/six-independent-full-proxy.jsonl)

An eight-client cold test at the default 32-step quantum kept six requests on the GPUs and rotated two through host RAM. Eight 12,000-token prompts generated 512 tokens each. The worker performed 50 saves and 50 restores. All eight streams matched their serial controls byte for byte and overlapped for 3.80 s. The complete wave took 160.6 s, including cold model and prompt work. [Eight independent clients](results/eight-independent-12k-rotation.jsonl) · [GPU and checkpoint tests](results/test-rotation-release.txt)

With the default offload settings, eight clients each used 523,776 input and 512 output tokens on a shared document. All eight matched their serial replays over 19 RAM saves and restores. The wave took 97.18 s, with 10.22 s of overlap across all eight client delivery intervals. [Eight full windows on the default profile](results/eight-full-default-rotation.jsonl)

Through LiteLLM, 32 clients each submitted 12,000 input tokens and generated 512. The engine kept six resident and parked up to 26. In both waves, each of the 32 clients returned its expected receipt sequence and matched its serial digest. The initial wave took 150.11 s. The last client began receiving output 35.14 s after the earliest stream delivered its last visible text, so that wave missed the all-stream overlap check. On the warm repeat, all 32 delivery intervals overlapped for 3.02 s, and the wave finished in 27.90 s after 33 RAM saves and restores. Neither wave reported cache failures or retained live state after completion. [32-client waves and individual timings](results/thirtytwo-12k-rotation.jsonl)

The packed FP32 GPU index scorer and shape-dependent radix sort produced identical receipt outputs across all 42 responses in each configuration. Warm aggregate full-window decode increased from 242.6 tok/s with scalar scoring to 288.9 with packed scoring and 361.1 after changing the sort dispatch. All 59 release tests passed on the two inference GPUs, including rollback, mixed-request failure isolation, token and block selection, cache reuse and per-session memory sizing. [Scalar control](results/six-full-scalar.jsonl) · [Packed scoring control](results/six-full-packed-index.jsonl) · [GPU index timings](results/packed-index-timing.txt) · [Release tests](results/test-six-release.txt)

### KV restore and rotation measurements

Eight requests reached 524,288 tokens apiece while sharing six GPU slots. Six resumed independent prior conversations, and two used a common document with distinct late receipts. We set the live RAM tier to 1 GiB and the rotation quantum to four steps to exercise disk overflow. The first wave finished in 44.04 s and the second in 39.63 s. All 16 responses matched their individual serial replays. Across the two concurrent waves the engine saved and restored 14 live states through RAM and 27 through disk, with zero save or restore failures. [Eight full-window clients and serial controls](results/eight-full-ram-disk.jsonl)

| Restored state | Source | Restores | Median time in restore operations | Median wall time to last GPU install call returning |
|---|---|---:|---:|---:|
| 403,968-token reusable prefix | Disk | 6 | 747 ms | 3.78 s |
| Near-full reusable prefix | RAM | 3 | 300 ms | 453 ms |
| Near-full reusable prefix | Disk | 29 | 904 ms | 1.12 s |
| Near-full suspended session | RAM | 14 | 257 ms | 454 ms |
| Near-full suspended session | Disk | 27 | 869 ms | 1.08 s |

The six 403,968-token prefix samples came from the preceding six-client run. The 32 near-full prefix samples came from 16 concurrent requests and their 16 serial replays. All 41 live restores came from the two concurrent waves. Timing starts at lookup and ends when the host returns from the final GPU install call. Subsequent computation follows the uploads on the CUDA streams. The measurements use the existing synchronization in the serving path. The operations column sums lookup/read, deserialization, drafter restore and GPU installation within each sample. The wall column also includes scheduling and inference interleaved with this work. At near-full context a suspended target image contains about 900 MiB. Disk samples used the normal filesystem page cache. The service defaults are 48 GiB live RAM, 128 GiB live disk and a 32-step quantum. [Per-phase medians and ranges](results/kv-restore-summary.json) · [Individual restore events](results/kv-restore-events.jsonl)

### Long-context retrieval

A cold 266,174-token request returned three randomly generated vault receipts in the requested order in 447.8 s. A 520,362-token continuation reused the first 266,112 prompt tokens from RAM, processed the suffix and returned its three receipts in 802.9 s. After a process restart, a changed suffix forced each request through the engine again. Both still returned all three receipts. The 266,179-token request took 27.0 s with cold weights. The 520,367-token request took 7.21 s end to end and restored 520,320 target and DSpark positions from disk. The final hybrid build repeated exact retrieval in 1.87 s and 2.59 s with warm weights, restoring both prefixes from disk. [Hybrid retrieval](results/hybrid-context.jsonl) · [Hybrid worker counters](results/hybrid-context-backend.txt)

Peak sampled memory was 96,828 and 96,888 MiB on GPUs 1 and 2. These measurements are single-request text retrieval; cold ingestion at long context remains expensive. [Cold and extended prompts](results/context.jsonl) · [Restart timing](results/context-restart.jsonl) · [Restore counters](results/long-restart-kv.txt) · [Memory samples summary](results/context-telemetry.json)

### Choosing between the learned and source drafters

When DSpark is ready, the scheduler uses it if fewer than 16 consecutive tokens agree with a source occurrence. With 16–31 matches it can propose the next 16 source tokens, with 32–63 it can propose 64, and with at least 64 it can propose the configured maximum. The target verifies every proposal. The Python rewrite used 25,598 input tokens and reached its 512-output-token limit in both modes. It measured 166.2 tok/s with fixed 128-token lookup and 308.1 tok/s after this change, with identical generated text. The complete Go edit measured 866.5 tok/s and passed the same tests. [Partial rewrite before](results/long-edit-fixed.jsonl) · [Partial rewrite after](results/hybrid-long.jsonl) · [Source lookup controls](results/lookup-tuning.json) · [Generated Go diff and tests](results/edit-validation.json)
