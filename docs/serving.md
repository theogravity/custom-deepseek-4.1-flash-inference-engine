# Serving the engine

## Run on this workstation

The coding profile uses the locally built release image. It uses longer verified output repetition for single-client drafting and reuses vocabulary-head weights across verification rows. It retains tiled attention ranking, shared prefix restoration and LFU expert eviction. After a coding tool call, idle GPUs prepare up to 32,768 tokens of history for the next request. The HTTP server reuses tokenized text when an agent sends the same file or tool result in a later turn. [Performance and checkpoints](../benchmarks/reports/decode-round-2.md)

Run these commands from the root of this checkout beside the workstation stack’s `models/` directory.

```sh
export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
install -d -m 700 "${DS4_HOST_PREFIX_DIR:-$HOME/.cache/ds4-flash-rs/prefix}"
docker compose --env-file ../models/.env -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml up --no-build -d --wait
```

For a fresh installation, build the named image first using Rust, a C++ host compiler and CUDA 13 with sm_120 support. CUDA 13.5 and Rust 1.100 were used here. The Docker image packages the host-built executable and mounts the host toolkit read-only. [Build and rollback instructions](../benchmarks/reports/agent-followups.md#build-and-rollback)

This selects the Rust engine for the existing DeepSeek service on port 8001 and uses its existing secret for authentication. It replaces the DeepSeek container in the `models` Compose project. The Rust server drains accepted requests on shutdown. Compose allows up to four minutes before forcing a stop. Finish active requests before switching. If another rotation model holds the GPUs, stop that model with `model-cli stop` first. The Compose file selects host GPUs 1 and 2. LiteLLM continues to serve clients on port 4000. The prior LFU and base LRU images have separate launch commands. [Checkpoint profiles](../benchmarks/reports/expert-cache.md#running-the-candidate-on-this-workstation)

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

## Prefix caching and API

The native prefix cache saves target attention state and the drafter's last 128 positions at completed prefill boundaries. Keys cover the token prefix, attention layout and numerical settings, checkpoint files and tokenizer. Draft sidecars also bind to the drafter weights. Entries carry checksums, disk writes use atomic renames, and the store bounds RAM and disk independently. A request can restore a shared prefix after a tool call or a process restart. The engine evaluates the suffix and generates the new completion. Set both cache budgets to 0 to disable persistence and cross-request memoization.

The worker serves up to **six concurrent text requests of 524,288 tokens each**, for **3,145,728 total context positions**. Input and generated output share each request's window. Each request has independent sliding-window KV, compressed KV, drafter state and sampling. At full size the six sets of compressed target KV and sliding windows occupy 3,252 MiB on the first inference GPU and 2,178 MiB on the second. Drafter windows, index buffers, activations, model weights and expert slots use additional VRAM. The 6 GiB RAM prefix tier can hold the last prefill checkpoint for six full-window requests. At 524,096 cached positions their six target images and drafter sidecars total 5.27 GiB. The disk prefix tier holds another 64 GiB.

The scheduler advances each GPU-resident runnable prompt by up to 64 input tokens per round and batches ready requests for drafting and target verification. A small request can start decoding while another is ingesting a long prompt. At the default settings, up to 32 requests can participate in rotation and another 32 can wait in the bounded intake queue. A full intake queue returns HTTP 429. The authenticated `/metrics` endpoint exposes resident, waiting and suspended requests, processed GPU positions, and successful six-request target passes.

### Live KV offload

When another runnable client needs a GPU slot, the worker saves one request's committed target KV and learned drafter state. It keeps that request's sampler, token history, stream and pending output alive. It releases the GPU buffers and lets the waiting client advance. On the next turn the suspended request resumes at the same token position. With six or fewer runnable clients the worker can serve each on a resident slot.

The backing store places each image in RAM if space is available, then uses an anonymous file on the NVMe drive mounted at `/prefix/live`. It checks disk writeback before releasing the GPU state and verifies the checksum on restore. Completion and cancellation release the backing space. These files belong to ongoing connections and disappear when their owner exits. Reusable prompt snapshots survive a service restart in the separate prefix cache.

The live tiers use the original LMCache budgets of **48 GiB RAM and 128 GiB disk**. With the separate prefix tier budgets, the combined payload limits are 54 GiB RAM and 192 GiB disk. A full 524,288-position suspended target image is about 900 MiB, plus drafter state. GPU staging, live token histories and the operating system's reclaimable file cache use additional host memory. If both live tiers fill, resident requests can finish and release their slots. The `swap` object in `/metrics` reports charged/peak bytes, reads, writes and failures for the two tiers; `rotations` and `resumes` count successful slot changes.

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

OpenAI chat and LiteLLM Responses report `error.code=context_length_exceeded`. Anthropic Messages reports `invalid_request_error` and includes the measured length in the message. Stock LiteLLM v1.109.0 converted the symbolic OpenAI code to `"400"`. The local proxy image preserves the code so Codex recognizes the error. On the workstation, all 12 rejected cases returned HTTP 400 before streaming and produced zero native admissions, batch steps, prefix reuse or swaps. All 10 valid cases returned one generated token. [Direct and proxy boundary checks](../benchmarks/results/clients/context-limits-live.jsonl) · [Installed Codex classification](../benchmarks/results/clients/codex-context-classification.json) · [Release tests](../benchmarks/results/context/test-context-release.txt)


The optional LiteLLM patch preserves oversized-request errors for Codex. The earlier Engram prototype is archived separately. [Model pack](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) · [LiteLLM patch](../integrations/litellm-context-errors/) · [Archived prototype](https://github.com/theogravity/custom-deepseek-4.1-flash-inference-engine/tree/archive/vllm-engram-prototype)
