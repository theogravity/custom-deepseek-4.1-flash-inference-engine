# Testing

Run the host-side Rust and Python tests from the root of the engine checkout. The Rust build compiles the CUDA sources and needs the CUDA toolkit.

```sh
cargo test --release --locked
python3 -m unittest discover -s tests -p 'test_*.py'
```

## GPU and checkpoint tests

The GPU suite needs exclusive use of both inference GPUs. Let active requests finish before stopping the coding profile. Run the commands below from the checkout root with the same exported `LITELLM_MASTER_KEY` used at startup. On the managed workstation, you can instead add `--env-file ../models/.env` immediately after `docker compose` in both commands. The suite includes checkpoint tokenizer, real EXL3, attention, peer-copy, speculative rollback and sampling fixtures.

```sh
export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml stop
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX PRO 6000/{print $2}' | paste -sd,)"
DS4_DRAFTER=code DS4_PREFIX_RAM_MIB=1024 DS4_PREFIX_DISK_MIB=0 DS4_PREFIX_DIR= DS4_PROMPT_LOOKUP=1 DS4_OUTPUT_LOOKUP=repeated DS4_ADAPTIVE_LOOKUP=1 DS4_LOOKUP_K=128 DS4_C1_OUTPUT_CONSENSUS=1 DS4_AGENT_TEXT_CACHE=1 DS4_IDLE_TOOL_PREFILL_MAX_TOKENS=32768 DS4_BATCH_TOPK=1 DS4_EXTEND_REPETITION=1 DS4_HEAD_ROW_TILES=1 DS4_EXPERT_EVICTION=lfu DS4_TEST_DEVICE=0 DS4_TEST_CUDA_DEVICE=0 DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3 cargo test --release --locked -- --include-ignored --test-threads=1
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml up --no-build -d --wait
```

### FlashInfer SM120 prompt attention

The optional `flashinfer-prefill` feature compiles the DSV4 single- and dual-cache kernels from FlashInfer commit `946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`. Set `DS4_FLASHINFER_DIR` to the checkout root or installed FlashInfer package's `data` directory containing `csrc/` and `include/`. The build verifies the source hashes. On the serving process set `DS4_FLASHINFER_PREFILL=1` to use the kernels during target prompt evaluation. Its prefix snapshots have a separate namespace.

```bash
cargo test --release --features flashinfer-prefill flash_prefill_workspace
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" DS4_TEST_DEVICE=0 \
  cargo test --release --features flashinfer-prefill cuda_flash_mla -- --ignored --nocapture
```

Build the two-rank expert and FlashInfer kernels into the serving image from the engine checkout. The build uses the model pack's `recipe/` folder and needs the pinned FlashInfer source tree.

```sh
cargo build --release --locked --features recipe-prefill,flashinfer-prefill
docker build -t ds4-flash-rs:tp2-prefill-20261127 .
export DS4_FLASHINFER_PREFILL=1
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml -f deploy/profiles/tp2-prefill.yml up -d --force-recreate --no-build --wait
```

### SM120 E4M3 prompt index scores

`DS4_PREFILL_INDEX_FP8=1` selects the native FP8 MMA scorer for target prompt evaluation and gives those prefix snapshots their own namespace. The scorer prepares each query row once, checks that every component round-trips through E4M3 with a per-head scale, and uses BF16 MMA for incompatible rows. The GPU selection fixture checks signed near-cutoff scores, earliest tie winners, complete position and block selections, and full selection timing at up to 524,288 positions.

```bash
CUDA_VISIBLE_DEVICES=<inference-gpu-uuid> DS4_TEST_DEVICE=0 \
  cargo test --release cuda_index -- --ignored --test-threads=1 --nocapture
```

### Resident ratio-two compressor pooling

`DS4_PREFILL_GPU_POOL_PAIRS=1` computes two-token gate-weighted pooling on the inference GPU during resident target prompt evaluation. At odd chunk boundaries the engine keeps the unpaired projected KV and gate row in the snapshot format. Verification uses the host pooling and capture path. The opt-in prefix namespace accounts for CPU and CUDA exponential rounding.

```bash
DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3 DS4_PREFILL_GPU_POOL_PAIRS=1 \
  CUDA_VISIBLE_DEVICES=<first-inference-uuid>,<second-inference-uuid> DS4_TEST_DEVICE=0 \
  cargo test --release pool_pairs -- --ignored --test-threads=1 --nocapture
```
