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
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml stop
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX PRO 6000/{print $2}' | paste -sd,)"
DS4_PROMPT_LOOKUP=1 DS4_OUTPUT_LOOKUP=single DS4_ADAPTIVE_LOOKUP=1 DS4_LOOKUP_K=128 DS4_C1_OUTPUT_CONSENSUS=1 DS4_AGENT_TEXT_CACHE=1 DS4_IDLE_TOOL_PREFILL_MAX_TOKENS=32768 DS4_BATCH_TOPK=1 DS4_EXTEND_REPETITION=1 DS4_HEAD_ROW_TILES=1 DS4_EXPERT_EVICTION=lfu DS4_TEST_DEVICE=0 DS4_TEST_CUDA_DEVICE=0 DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3 cargo test --release --locked -- --include-ignored --test-threads=1
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml up --no-build -d --wait
```
