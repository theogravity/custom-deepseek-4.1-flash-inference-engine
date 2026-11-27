# ds4-flash-rs

A ground-up Rust and CUDA inference engine for [DeepSeek V4.1 Flash, Diffbot EXL3 3bpw](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), tuned for two NVIDIA RTX PRO 6000 Blackwell GPUs. The engine provides Anthropic Messages and OpenAI Chat Completions APIs and supports long contexts, concurrent sessions, prefix caching, and RAM/disk KV offload.

## Headline benchmarks

Warm coding-edit decode throughput. Concurrent rates are aggregate across clients.

| Clients | Native engine | vLLM reference |
|---:|---:|---:|
| 1 | 343 tok/s¹ | 282 tok/s |
| 4 | 491 tok/s² | 518 tok/s |
| 6 | 498 tok/s² | 557 tok/s |

¹ Earlier full-profile run; the latest prefix-fix run did not measure C1. ² Latest run, 25.6k-token prompt with restored shared prefix. Six independent ~404k-token sessions measured 304.5 aggregate tok/s in a separate run. The published vLLM numbers used a different run and power limit; see the detailed reports for methodology and raw data.

- [Latest C4/C6 comparison and run data](benchmarks/reports/prefix-fix-benchmark-20261127.md)
- [Long-context, cold-prefill and session-capacity results](benchmarks/reports/default-profile-headline-20261127.md)
- [All benchmark reports and raw data](benchmarks/)

## Run

Requirements: Linux, Rust, CUDA Toolkit, Docker Compose, NVIDIA Container Toolkit, and the model checkpoint downloaded to `/mnt/models/dsv41-flash-exl3`.

From this repository:

```sh
git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer
git -C /tmp/flashinfer checkout 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1
DS4_FLASHINFER_DIR=/tmp/flashinfer cargo build --release --locked --features recipe-prefill,flashinfer-prefill
docker build -t ds4-flash-rs:tp2-peer-index-prefix64-20261127 .
export LITELLM_MASTER_KEY='your-local-litellm-key'
export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
install -d -m 700 "$HOME/.cache/ds4-flash-rs/prefix"
docker compose -f compose.yml -f deploy/profiles/tp2-prefill.yml up --no-build -d --wait
curl -fsS http://127.0.0.1:8001/health
```

The root Compose file enables the recommended measured TP2 coding defaults: the 8k prefill chunk, LFU expert eviction, peer-index split, speculative drafting, batched top-k, output consensus, repeated-output lookup, and text caching. It serves six resident sessions with a 512k context and RAM/disk KV rotation. The `tp2-prefill.yml` overlay selects the pinned benchmark image. For this workstation's existing `model-cli` setup, build that image and select it with `model-cli switch dsv41-flash-rust`; the vLLM profile remains available as `dsv41-flash-exl3`.

## Project guide

| Topic | Documentation |
|---|---|
| API clients, context limits, KV offload | [Serving guide](docs/serving.md) |
| Reproduce benchmarks | [Benchmark index](benchmarks/) |
| Architecture and tuning options | [Engine guide](docs/engine.md) |
| Build and test | [Testing guide](docs/testing.md) |
| Historical experiments and profiles | [Development history](docs/development-history.md) · [Compose profiles](deploy/profiles/) |

## Benchmark workstation

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 9965WX, 24 cores / 48 threads |
| Memory | 256 GB DDR5-5600 |
| Inference GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each |
| Desktop GPU | NVIDIA GeForce RTX 5080, excluded from benchmarks |
| Storage | Samsung 9100 PRO NVMe drives |

The reported native coding results use 375 W per inference GPU. Detailed CPU/GPU settings and their benchmark dates are in [workstation measurements](benchmarks/results/workstation/).

## License

The engine, tools, and documentation are licensed under MIT. Model weights, CUDA libraries, and third-party dependencies retain their respective licenses. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
