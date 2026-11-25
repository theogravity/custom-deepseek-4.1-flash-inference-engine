# ds4-flash-rs

A standalone Rust and CUDA inference engine for the Diffbot 3-bit EXL3 pack of DeepSeek V4.1 Flash on two RTX PRO 6000 Blackwell workstation cards. The server opens the original 48 safetensors shards and generates tokens locally. Rust owns checkpoint loading, tokenization, scheduling, sampling, and the model loop. The GPU runtime uses CUDA C++, tensor-core instructions and cuBLAS. The HTTP server uses Axum.

## Run on this workstation

The release build needs Rust, a C++ host compiler and CUDA 13 with sm_120 support. CUDA 13.5 and Rust 1.100 were used here. The Docker image packages the host-built executable and mounts the host toolkit read-only.

```sh
cd ~/llm/ds4-flash-rs
cargo build --release --locked
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

## Model execution

Layers 0–19 run on the first RTX card and layers 20–39 on the second. The engine expands dense MXFP8 weights into BF16 on demand for matrix multiplication. Short chunks use BF16 activations. The attention output is rounded through block-scaled FP8 before its grouped output projection. A custom CUDA kernel packs the vocabulary head into per-row, 32-channel FP8 groups and another computes its logits. Each layer selects six routed experts out of 384. CUDA decodes their packed trellises during multiplication. GPU slots hold recently used experts, and the mmap-backed checkpoint supplies misses. Each Engram layer gathers only its hashed int4 rows. Sliding-window and compressed sparse attention use request-local KV state. A snapshot of the previous prompt's last complete chunk permits prefix reuse across tool-result turns that share the same tokens and attention layout.

One request executes at a time. Other requests enter a bounded queue. The server accepts text, reasoning and DSML tools, supports both streamed and nonstreamed responses, and exposes `/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/models` and `/health`. Codex uses LiteLLM's Responses translation. Vision, DSpark speculation, LMCache and continuous batching are outside this implementation. The Compose context cap is 524,288 tokens, matching the existing model-cli metadata. See the qualification section for lengths actually exercised.

## Validation and measurements

The following standalone commands own the GPUs. Stop the serving process before using them.

```sh
cd ~/llm/ds4-flash-rs
docker compose --env-file ../models/.env stop
RAYON_NUM_THREADS=12 target/release/ds4-flash-rs infer 'The capital of France is' --max-tokens 8
RAYON_NUM_THREADS=12 target/release/ds4-flash-rs --chunk-size 64 bench
RAYON_NUM_THREADS=12 target/release/ds4-flash-rs --chunk-size 64 verify
```

`bench` runs one warm-up and six measured greedy generations and checks that every continuation agrees. The rate excludes prefill and the first token. `verify` is a strict comparison with the *other engine's* saved greedy token IDs. It returns a nonzero exit code on any divergence. The serving recipe uses tensor parallelism, alternate expert and reduction kernels, and speculative verification. Small logit differences can change which token wins. The qualification results below record the observed agreement.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID DS4_TEST_DEVICE=1 DS4_TEST_CUDA_DEVICE=1 cargo test --release --locked -- --include-ignored
docker compose --env-file ../models/.env up -d --wait
python3 tools/clients.py
python3 tools/roofline.py
```

The client checks create temporary workspaces and require the assistant to read a fresh random receipt using a tool. The check accepts only an exact final reply. The credential comes from the existing model-cli launcher.

## Qualification on the two RTX PRO 6000s

At temperature zero, the engine produced the same 32-token continuation on six measured runs after one warm-up. Median decode was **26.1 tok/s**, with a range of 25.7–26.5. Median time to the first output token on this 18-token raw prompt was 0.16 seconds once its weights were warm. The first run in a fresh process took 2.39 seconds to the first token after checkpoint setup. Performance varies with expert residency, prompt length, and the tokens generated. [Timing samples and output](results/decode.json) · [Bandwidth estimates](THEORY.md)

The four raw prompts matched the previous server's first token. One entire continuation matched exactly. The three differing continuations and their expected token IDs are recorded in the comparison file. The installed Codex, Claude Code, and Pi clients also have independent tool-result checks. [Greedy comparison](results/greedy-comparison.txt) · [Client results](results/clients.jsonl) · [Client prefill timings](results/client-prefill.txt)

A retrieval test asked for a randomly generated receipt in the first half of 160 records. The answer was exact on the cold 2,887-token prompt and a 2,895-token follow-up with a different final instruction. The engine logged 2,880 reused tokens on the follow-up. Prefill fell from 41.52 to 0.64 seconds. The local coding-client tests exercised prompts of up to 13,635 tokens. The 524,288-token cap and the old recipe's six-session throughput have not been qualified for this engine. [Retrieval responses](results/retrieval.json) · [Engine prefill counters](results/retrieval-prefill.txt)

The release passed all 18 tests, including the real tokenizer and CUDA checks. During a live service stop, a 40-delta stream delivered its final message event and the container exited with code 0. [Release details](results/release.json) · [Shutdown check](results/shutdown.json)
