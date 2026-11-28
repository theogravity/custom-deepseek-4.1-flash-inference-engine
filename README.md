# ds4-flash-rs

A ground-up Rust and CUDA inference engine for [DeepSeek V4.1 Flash, Diffbot EXL3 3bpw](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), tuned for two NVIDIA RTX PRO 6000 Blackwell GPUs. The engine provides Anthropic Messages and OpenAI Chat Completions APIs and supports long contexts, concurrent sessions, prefix caching, and RAM/disk KV offload.

## Headline benchmarks

Warm 25.6k-token coding-edit decode throughput. Concurrent rates are aggregate across clients.

| Clients | Native engine | vLLM reference |
|---:|---:|---:|
| 1 | **410 tok/s** | 282 tok/s |
| 4 | **534 tok/s** | 518 tok/s |
| 6 | 533 tok/s | 557 tok/s |

The native rates are medians of six waves at 375 W per GPU. C4/C6 use the round-three defaults (wide inline MoE descriptors, an 88 GiB expert cache limit with a 4 GiB VRAM reserve); C1 predates them and was not remeasured. Decode returns to these rates right after long cold prompts ([shard eviction](benchmarks/reports/expert-shard-eviction-20261127.md)). Cold first output on the round-5 image: 3.8 s at 16k, 10.2–10.5 s at 60k and 38.5 s at 196k tokens (vLLM on this host: 2.9 / 8.1 / 20.6 s); see the [experiment ledger](benchmarks/experiment-ledger.md). vLLM reference rates were published at 600 W per GPU. Six independent ~404k-token sessions measured 304.5 aggregate tok/s in a separate run. See the reports for methods and raw data.

- [Round three C4/C6: host stalls and expert residency](benchmarks/reports/c46-round3-20261127.md)
- [Prefix-fix C4/C6 comparison and run data](benchmarks/reports/prefix-fix-benchmark-20261127.md)
- [Batch-aware decode optimization and matched controls](benchmarks/reports/batch-adaptive-confidence.md)
- [vLLM decode path investigation and rejected shared-budget trial](benchmarks/reports/vllm-c46-decode-path-20261127.md)
- [Long-context, cold-prefill and session-capacity results](benchmarks/reports/default-profile-headline-20261127.md)
- [All benchmark reports and raw data](benchmarks/)

## Run

Requirements: Linux, Rust, CUDA Toolkit, Docker Compose, NVIDIA Container Toolkit, and the model checkpoint downloaded to `/mnt/models/dsv41-flash-exl3`.

From this repository:

```sh
git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer
git -C /tmp/flashinfer checkout 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1
DS4_FLASHINFER_DIR=/tmp/flashinfer cargo build --release --locked --features recipe-prefill,flashinfer-prefill
docker build -t ds4-flash-rs:tp2-round5-20261128 .
export LITELLM_MASTER_KEY='your-local-litellm-key'
export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
install -d -m 700 "$HOME/.cache/ds4-flash-rs/prefix"
docker compose -f compose.yml -f deploy/profiles/tp2-prefill.yml up --no-build -d --wait
curl -fsS http://127.0.0.1:8001/health
```

The root Compose file enables the recommended measured TP2 coding defaults: the 8k prefill chunk, LFU expert eviction, peer-index split, batch-aware speculative confidence, batched top-k, output consensus, repeated-output lookup, and text caching. It serves six resident sessions with a 512k context and RAM/disk KV rotation. The `tp2-prefill.yml` overlay selects the pinned benchmark image. For this workstation's existing `model-cli` setup, build that image and select it with `model-cli switch dsv41-flash-rust`; the vLLM profile remains available as `dsv41-flash-exl3`.

## Project guide

| Topic | Documentation |
|---|---|
| API clients, context limits, KV offload | [Serving guide](docs/serving.md) |
| Reproduce benchmarks | [Benchmark index](benchmarks/) |
| Architecture and tuning options | [Engine guide](docs/engine.md) |
| Build and test | [Testing guide](docs/testing.md) |
| Historical experiments and profiles | [Experiment ledger](benchmarks/experiment-ledger.md) · [Development history](docs/development-history.md) · [Compose profiles](deploy/profiles/) |

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
