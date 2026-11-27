# Cold first output against the installed vLLM image

We ran the workstation’s vLLM image and the standalone Rust/CUDA engine on the same two RTX PRO 6000 cards. Each process received one separate 11k-token warm-up before the timed prompts. Each timed case used one client, temperature zero, a 400-token output limit, and measured wall time through the first nonempty reasoning or visible text. Both GPUs had a 375 W cap and the CPU socket had a measured 225 W cap. Each reported timing is a completed request with zero restored prefix tokens. The order was vLLM, three 128-token kernel trial arms, then the native 64-token run. The host filesystem cache stayed warm across arms, so these timings establish cold KV behavior. Each reported timing is one sample. [Per-request data and effective settings](../results/upstream/local-vllm-prefill.json)

| Input tokens | Installed vLLM | Rust/CUDA, deferred decoder | Native / vLLM time |
|---:|---:|---:|---:|
| 15,864 | 2.91 s | 11.78 s | 4.05× |
| 60,013 | 8.07 s | 38.91 s | 4.82× |
| 195,803 | 20.62 s | 159.69 s | 7.74× |
| 505,000 | 52.92 s | 438.09 s | 8.28× |

Both servers used the same model pack, released five-proposal DSpark drafter, 524,288-token context limit, six resident session slots and 48 GiB RAM / 128 GiB disk prefix budgets. Both render omitted thinking effort as 75/100. We compared SHA-256 digests of all rendered token IDs for the warm-up and all four timed requests. Every digest matched. The native comparison disabled prompt and output lookup. The local vLLM run used the stack’s generated Compose file, installed plugins, mounted overrides and image `3689a42b8a15`. It used an isolated listener and a fresh LMCache disk directory. GPU and external prefix-hit counters were zero. This was single-client cold prefill. [Historical results](headline-reference.md)

## Why the paths differ

vLLM uses tensor parallelism across both cards, FP8 prefill all-reduce and a scheduling budget of 8,192 tokens. Its 64-token KV block controls cache allocation. The native run used 64-token physical prefill batches, layer placement across the two cards, expert parallelism and exact deferral of unobserved local decoder work. At 60k, an earlier native trace placed both GPUs in compute kernels together for 14.19% of wall time. These measurements do not isolate the contribution of tensor parallelism or the collective. [Native trace and decoder experiment](cold-prefill-optimization.md)

The installed recipe enables its S8B GPU expert kernel for eligible prefill batches of 128 tokens or more. It reuses decoded weights for up to 128 rows assigned to the same expert. At 128 input tokens, six routes per token across 384 experts give an average of two rows per expert. At 8,192 input tokens that average is 128. Actual loads depend on routing, placement and cache residency. CPU experts are enabled for decode and disabled for prefill in the local vLLM configuration. LMCache served zero tokens on these requests. [Model pack and kernel recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/tree/451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae/recipe/kernels/xmoe_do_i8)

## Run the same expert kernel from Rust

The optional Cargo feature compiles the pack’s pinned CUDA device code. The standalone engine supplies the host-side routing and buffers. PyTorch is used by the independent parity fixture only. The adapter uses the pack’s S8B variant 0, compiles its two projections for the native full intermediate width, and assigns a separate prefix-cache identity. The installed vLLM worker runs a tensor-parallel shard of the expert. S8B uses quantized activations and weights with different arithmetic from native’s customary FP16 expert path.

The adapter matched the packaged extension bit-for-bit on 16 real-weight fixtures spanning two model layers, 127–134 rows, repeated token indices, sparse output destinations and both GPU peer directions. These fixtures test the expert projection at the same shape. Model-wide accuracy was ungraded. At 128-token native chunks the installed kernel was within the spread of the two controls at both lengths. Each arm used a fresh process, an isolated cache and the same seeded prompts. The integrated host-side feature build passed 67 Rust tests.

| Input tokens | Native kernel, before | Pack S8B | Native kernel, after |
|---:|---:|---:|---:|
| 15,864 | 21.11 s | 14.26 s | 12.85 s |
| 60,013 | 76.21 s | 50.64 s | 50.45 s |

From the repository root, with the model downloaded at the documented path and the stack’s authentication key exported, use the following profile. The final overlay uses physical 128-token chunks and enables S8B. For each timing arm, set `DS4_HOST_PREFIX_DIR` to a new empty directory, restart the process to clear its RAM tier, warm it with the separate 11k input, and check that the prefix-reuse counter increases by zero for each timed request. Set `DS4_RECIPE_PREFILL=0` for the control. The 64-token comparison in the upper table uses the cold-prefill overlay followed by the reference-drafter overlay. The recipe and ExLlamaV3 license notices are included in the image. [Build prerequisites](../../README.md#build-and-serve-from-a-fresh-clone) · [Third-party notices](../../THIRD_PARTY_NOTICES.md)

```sh
cargo build --release --locked --features recipe-prefill
docker build -t ds4-flash-rs:recipe-prefill-20261126 .
export DS4_HOST_PREFIX_DIR="$(mktemp -d)"
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml -f deploy/profiles/reference-drafter.yml -f deploy/profiles/recipe-prefill.yml up --no-build --force-recreate -d --wait
```

### Reproduce projection parity

On the model workstation, build a small test library from the same two CUDA objects as the binary. Use a separate Cargo target directory to identify this build’s objects. Run this fixture with both inference cards available. The model’s compiled `xmoe_do_i8` module must be present in the pack’s `recipe/kernels/build` directory.

```sh
export CARGO_TARGET_DIR="$(mktemp -d)"
cargo build --features recipe-prefill --release --locked
obj="$(dirname "$(find "$CARGO_TARGET_DIR" -name recipe_prefill.o -print -quit)")"
/usr/local/cuda/bin/nvcc -shared "$obj/moe.o" "$obj/recipe_prefill.o" -o "$CARGO_TARGET_DIR/librecipe.so"
docker run --rm --network none --gpus '"device=1,2"' -e CUDA_VISIBLE_DEVICES=0,1 \
  -v "$CARGO_TARGET_DIR/librecipe.so:/tmp/librecipe.so:ro" -v "$PWD/tools:/check:ro" \
  -v /mnt/models/dsv41-flash-exl3:/model:ro --entrypoint python3 \
  dsv41-flash-1017-exl3-3bpw-sm120-lmcache /check/check_recipe_prefill.py --model /model --library /tmp/librecipe.so
```

## Experimental two-rank expert projection

This branch adds the rank-local CUDA entry points for a two-rank EXL3 expert. For the gate and up projections, the loader splits the output dimension of the trellis and its output scale vector. For the down projection, it splits the input dimension of the trellis and its input scale vector. Each rank processes 1,152 intermediate channels. The other scales are replicated. The input is still the original quantized model pack.

To compare each rank with the installed vLLM extension, use the projection parity commands above with `/check/check_recipe_tp2.py` in the final Docker invocation. The test evaluates expert 3 at layers 0 and 20 on both ranks using 129 routed rows in two jobs. It checks the partial outputs before reduction and compares their host-side sum. That projection-only checkpoint tested the rank-local CUDA calculations. The later serving experiment below measures the integrated expert and dense paths. It used the native attention head layout and did not grade model-wide output fidelity.

Both ranks matched the installed extension bit-for-bit in all four cases. The fixture ran with Python assertions disabled and verified that seven invalid job layouts per case were rejected before allocation. [Rank-local test results](../results/upstream/tp2-expert-shards.json)

## Physical prefill steps of 8,192 and 16,384 tokens

This experiment uses rank-local EXL3 expert shards, sharded large dense projections, cuBLASLt MXFP8, BF16 tensor-core index scoring, and layer-20 bounded decoder replay. Both arms used the released K=5 drafter, identical requests, separate empty prefix directories, six resident slots, and the same 375 W GPU and 225 W CPU limits as the installed reference. Each result is one run. [Per-request data](../results/upstream/tp2-prefill-chunks.json)

| Workload | 8,192 step | 16,384 step |
|---|---:|---:|
| Cold first output, 15,864 input tokens | 8.68 s | 12.74 s |
| Cold first output, 60,013 input tokens | 32.12 s | 29.73 s |
| First output for a 68,049-token code review arriving during a Rust generation | 51.89 s | 51.91 s |
| Longest full interval between existing-client output events intersecting the arriving client’s wait | 13.24 s | 17.94 s |

The client submitted the review after the Rust stream emitted eight nonempty text or reasoning events. The Rust request used 70 input and 1,500 output tokens. The review used 68,049 input and 64 output tokens. Some events contained multiple tokens. The published longest gap uses the **whole** interval between consecutive Rust events if it intersected the time from review submission through first review output. Either end of that interval can fall outside the review’s wait. Review TTFT starts when the worker opens the HTTP request. Both timings include scheduling and delivery. The historical JSON field `first_client_e2e_s` ends at the last nonempty Rust event.

Both historical runs began and ended with idle scheduler snapshots, reported zero reused prefix tokens, and recorded usage and finish reasons. They did not retain the full event timeline or the mixed pair’s admission count. The current driver also requires exactly two admissions, matching token counts, zero reuse or rotations, both terminal SSE markers, and at least one Rust output strictly between the review worker starting its request and receiving its first output. The Rust stream must continue past that first review output. New receipts report the full gap that intersects this worker-to-first-output interval, its endpoints relative to worker start, and the Rust stream’s time to its last content event and terminal marker separately.

### Reproduce the two arms

Use the current checkout for the profiles and benchmark drivers. Build the measured engine revision in a separate worktree. Its historical image config SHA-256 was `3d3aac20569105af552341eb2553423ced2574c9514a115b880457941e6e42c1`. It predates the optional FlashInfer backend. Later engine revisions change the kernels. Use the build prerequisites and model pack from the [main setup](../../README.md#build-and-serve-from-a-fresh-clone).

```bash
bench_dir="$(mktemp -d)"
git worktree add --detach "$bench_dir/engine" ecd8e2a96e1eee48c49100d1fe6c8e58eb0c8ce4
(
  cd "$bench_dir/engine" &&
  CARGO_TARGET_DIR="$PWD/target" cargo build --release --locked --features recipe-prefill &&
  docker build -t ds4-flash-rs:tp2-measured-ecd8e2a .
)
```

On an idle benchmark host, with the API key exported, this Bash block recreates the service on port 8001 for each arm. Keep the 375 W cap on each inference GPU and the 225 W CPU socket cap to match the original. A new process and empty prefix directory per arm precede the 11,080-token warm-up, both isolated cold cases and finally the mixed requests. The warm-up is excluded from the tables.

```bash
(
  set -e
  export DS4_RUN_UID="$(id -u)" DS4_RUN_GID="$(id -g)"
  export DS4_TP2_IMAGE=ds4-flash-rs:tp2-measured-ecd8e2a
  export DS4_PREFILL_DECODER_PRUNE=1 DS4_IDLE_TOOL_PREFILL_MAX_TOKENS=0 DS4_FLASHINFER_PREFILL=0
  stack=(-f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml
         -f deploy/profiles/concurrent-coding.yml -f deploy/profiles/reference-drafter.yml
         -f deploy/profiles/tp2-prefill.yml)
  for step in 8192 16384; do
    export DS4_PREFILL_CHUNK="$step" DS4_HOST_PREFIX_DIR="$bench_dir/prefix-$step"
    mkdir -m 700 "$DS4_HOST_PREFIX_DIR"
    docker compose "${stack[@]}" up -d --force-recreate --no-build --wait
    python3 tools/cold_prefill.py --timing-only 3500,91570510 5000,91570511 19000,91570512 \
      > "$bench_dir/cold-$step.jsonl"
    python3 tools/prefill_overlap.py --out "$bench_dir/mixed-$step.json"
  done
)
```

The configured context limit is 524,288 in both arms. Restore the original serving image and prefix directory using the same Compose files and environment that were active before the experiment.
