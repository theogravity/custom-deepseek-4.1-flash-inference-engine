# ds4-flash-rs

A standalone Rust and CUDA inference engine for the Diffbot 3-bit EXL3 pack of DeepSeek V4.1 Flash on two RTX PRO 6000 Blackwell workstation cards. The server opens the original 48 safetensors shards and generates tokens locally. Rust owns checkpoint loading, tokenization, scheduling, sampling, and the model loop. The GPU runtime uses CUDA C++, tensor-core instructions and cuBLAS. The HTTP server uses Axum. [Download the Diffbot 3-bit EXL3 model on Hugging Face](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)

## Benchmarks

These warm workload measurements use the previous coding profile on the native Rust/CUDA engine with the code-tuned DSpark drafter and two RTX PRO 6000 Blackwell cards at 375 W each. Rates are medians of six requests or six concurrent batches.

| Coding workload | Clients | Input tokens per client | Output tokens per client | Decode tok/s | End-to-end tok/s |
|---|---:|---:|---:|---:|---:|
| Captured Pi reasoning turn | 1 | 2,786 | 256 | 178.8 | 174.2 |
| Structured Python edit | 1 | 25,625 | 1,500 | 383.5 | 379.0 |
| Code generation on a long history | 1 | 404,151 | 512 | 256.6 | 240.4 |
| Code generation on six independent histories | 6 | ~404,000 | 512 | **304.5 aggregate** | **299.7 aggregate** |

Decode spans the first through last output. The Python edit follows the upstream convention of counting all 1,500 output tokens from first output to stream close. End-to-end includes the wait for first output and runs until the request or batch finishes.

On six different follow-ups to a warm 404k-token conversation, median time to first output was **0.48 s** with tokenized-text caching enabled. The engine supports six independent 524,288-token windows and can rotate additional sessions through RAM and disk. [Methodology](benchmarks/reports/agent-followups.md) · [Raw measurements](benchmarks/agent-followups/measurements/) · [All benchmarks](benchmarks/)

### Same workloads as the vLLM reference

| Concurrent clients | Published vLLM tok/s | Previous native tok/s | Current native tok/s | Current vs. vLLM |
|---:|---:|---:|---:|---:|
| 1 | 282 | 382.5 | 383 | +35.8% |
| 4 | 518 | 457.5 | **584** | **+12.7%** |
| 6 | 557 | 502.5 | **583** | **+4.7%** |

Rates are aggregate output tok/s on the upstream 25,625-token Python edit. At six clients, dividing 583 by six gives about 97 tok/s per client over the longest decode interval. Individual streams finish at different times. Native rates are medians of six warm second passes at 375 W per GPU. The published vLLM figure is one warm second pass at 600 W. The concurrent coding profile lets each client draft repeated patterns from its own verified output. In the separate matched four- and six-client runs, every measured stream reproduced the control’s visible code and reasoning. [Concurrent decoding experiment](benchmarks/reports/concurrent-repetition.md) · [Original headline comparison](benchmarks/reports/headline-reference.md)

### Cold first output on the same workstation

Single-client time to first streamed reasoning or visible output with the released five-proposal DSpark drafter. Each engine received the same rendered inputs, a separate 11k warm-up and zero reused tokens. Both GPUs were capped at 375 W. Each entry is one request.

| Input tokens | Installed vLLM, 8k step | Earlier native, 64 step | Native TP2, 8k step | Native TP2, 16k step |
|---:|---:|---:|---:|---:|
| 15,864 | 2.91 s | 11.78 s | 3.77 s | 4.44 s |
| 60,013 | 8.07 s | 38.91 s | 13.99 s | 13.96 s |
| 195,803 | 20.62 s | 159.69 s | 78.86 s | 84.84 s |
| 505,000 | 52.92 s | 438.09 s | — | 356.32 s¹ |

The two wide-step columns use the same build through 195,803 tokens. ¹The 505k result is from the preceding 16k build, with BF16 index scoring. Free VRAM on that run fell to 66 MiB per GPU. All four input lengths fit the 524,288-token serving limit. Six simultaneous 404k requests have yet to be qualified on this wide-step profile. [Protocol, builds and agent overlap](benchmarks/reports/local-vllm-prefill.md#wide-tp2-checkpoint) · [Raw native results](benchmarks/results/upstream/tp2-wide-prefill.json)

## Build and serve from a fresh clone

Install the CUDA toolkit at `/usr/local/cuda`, a C++ host compiler, Rust, Docker Compose and the NVIDIA Container Toolkit. Download the checkpoint including its code-tuned drafter to the model mount. From this checkout, with the same `LITELLM_MASTER_KEY` as your proxy exported in the shell, run the following commands. The Compose profile reserves host GPU indices 1 and 2 and serves on port 8001.

```sh
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --revision 451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae --local-dir /mnt/models/dsv41-flash-exl3
cargo build --release --locked
docker build -t ds4-flash-rs:concurrent-coding-20261126 .
export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
install -d -m 700 "${DS4_HOST_PREFIX_DIR:-$HOME/.cache/ds4-flash-rs/prefix}"
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml up --no-build -d --wait
curl -fsS http://127.0.0.1:8001/health
```

## Project guide

| Task | Start here |
|---|---|
| Connect Codex, Claude Code or Pi, configure KV offload, or check context limits | [Serving guide](docs/serving.md) |
| Understand the model loop, drafters and runtime settings | [Engine guide](docs/engine.md) · [Rust server](src/) · [CUDA kernels](native/) |
| Reproduce a measurement or compare coding, context and concurrency results | [Benchmark index](benchmarks/) |
| Run the host and GPU tests | [Testing guide](docs/testing.md) |
| Trace an experiment or restore an earlier profile | [Development history](docs/development-history.md) · [Compose overlays](deploy/profiles/) |

## Benchmark workstation

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 9965WX, 24 cores / 48 threads |
| System memory | 256 GB DDR5-5600, approximately 247 GiB visible to Linux |
| Inference GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB GDDR7 each, host GPU indices 1 and 2 |
| Desktop GPU | GeForce RTX 5080, excluded from engine benchmarks |
| Storage | Samsung 9100 PRO 2 TB NVMe drives for the model pack and the host cache |
| Software used | Ubuntu 26.04.1 LTS, NVIDIA driver 615.71.09, CUDA Toolkit 13.5.37, Rust 1.100.0 |

The current coding profile was benchmarked at **375 W per inference GPU** and a measured **225 W CPU socket cap**. The GPU P0 offsets were +450 MHz core and +4,000 MHz VRAM. Separate 600 W GPU runs are labeled in the results. The workstation's 180 W CPU boot setting was prepared for a later boot. [GPU settings](benchmarks/results/workstation/benchmark-gpu-settings.json) · [CPU power measurements](benchmarks/results/workstation/benchmark-cpu-power.json) · [Boot unit](deploy/workstation/cpu-power-cap.service)

## License

The engine, tools and documentation are licensed under MIT. The bundled DeepSeek tool prompt retains its original copyright and MIT notice. Downloaded weights, CUDA libraries and other dependencies retain their respective licenses. [Project license](LICENSE) · [DeepSeek prompt notice](src/tool_prompt.LICENSE)
