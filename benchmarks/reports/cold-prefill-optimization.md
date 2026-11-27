# Single-client cold prefill on the dual RTX PRO 6000

Deferring old decoder work shortened a fresh 60k-token request to **43.61 seconds** until first output. The same image with the option disabled took 60.83 s before and 63.99 s after. At 16k the candidate was about 0.14 s faster than the faster control. Each cell is a single HTTP stream with a fresh seed, after a different 11k-token request warmed each process. Each run restored zero prefix tokens. The native engines ran at 375 W per GPU and a 225 W CPU socket cap. [Seeds, token counts, controls and CUDA trace summary](../results/upstream/cold-prefill-optimization.json)

| Native experiment, 64-token chunks | Input | Control before | Enabled | Control after |
|---|---:|---:|---:|---:|
| Defer local decoder layers | ~16k | 18.82 s | 15.51 s | 15.64 s |
| Defer local decoder layers | ~60k | 60.83 s | **43.61 s** | 63.99 s |
| Send larger expert routing tables with the CUDA launch | ~16k | 16.98 s | 15.04 s | 17.05 s |
| Send larger expert routing tables with the CUDA launch | ~60k | 59.96 s | 57.79 s | 60.58 s |

Each experiment changes one setting, starts a fresh process for each arm and measures through the first visible or reasoning delta. The historical published vLLM times were 1.4 s at 16k and 5.0 s at 60k. Those published runs did not record the GPU power limit. [Published reference and workload](headline-reference.md)

## What the native change preserves

The engine computes all shared compressed attention sources through layer 20. Layers 21–39 use their own 128-token sliding windows. On a prompt chunk sufficiently far from the next prefix snapshot or output, the engine advances those windows without computing the hidden state that will expire. It resumes full computation early enough to rebuild all 19 layers and the last 128 drafter feature rows. The cutoff rounds each layer’s dependency up to a whole physical chunk because rows in the same routed-expert batch can share a split-K plan.

In a model-backed test with the code drafter at 64- and 127-token chunks, each reusable target and drafter checkpoint matched the full forward byte-for-byte. The test rotated a live request during deferral and compared continuation tokens after restoring the candidate’s own saved prefix. The optimization is opt-in via `DS4_PREFILL_DECODER_PRUNE=1`. Set it to 0 for the full forward. The normal context and session admission limits apply. [Model and parity tests](../../src/model.rs)

The three-arm experiment and the model-backed checkpoint test used the pruning worktree based on `1d7f142`. The integrated source also includes the concurrent coding checkpoint. Its host-side release suite passed 67 tests. [GPU test procedure](../../docs/testing.md)

### Build and activate

For existing installations, build the new executable and image from this checkout, then recreate the container with the cold-prefill overlay last. The preceding concurrent-coding image predates this setting. New installations also need the model and prefix directories from the [build instructions](../../README.md#build-and-serve-from-a-fresh-clone).

```sh
cargo build --release --locked
docker build -t ds4-flash-rs:uniform-coding-20261126 .
docker compose -f compose.yml -f deploy/profiles/decode-round2.yml -f deploy/profiles/coding.yml -f deploy/profiles/concurrent-coding.yml -f deploy/profiles/cold-prefill.yml up --no-build --force-recreate -d --wait
```

Use the same exported authentication key as the serving stack or supply its Compose env file. To roll back, omit the last overlay and recreate the container with the preceding concurrent-coding image.

## What the local vLLM profile does

The workstation’s generated Compose configuration selects TP2, a 524,288-token served window, six sequences, an 8,192-token scheduling budget, 64-token KV blocks, FP8 KV with an 11 GiB budget per worker, and the released five-proposal DSpark drafter. The native coding release uses a code-tuned seven-proposal drafter and 64-token prefill chunks. A vLLM scheduling budget and a KV block size govern different operations.

For eager prefill batches of at least 128 tokens, eligible GPU experts use the pack’s `xmoe_do_i8` grouped kernel. The locally built extension reports S8B mode, variant 0. It quantizes activations at one scale per row and 128 features, and reuses decoded EXL3 weights across up to 128 rows per job. CPU expert execution is disabled for prefill. The mounted recipe also trims decoder replay, enables 8-bit prefill all-reduce, and uses specialized dense projections and fused routing. LMCache supplies RAM and disk tiers for prefix reuse. The mounted tokenizer maps omitted effort to high, which the image maps to 75/100, the same default as this native engine. The historical benchmark's rendered inputs were not retained. [Workstation configuration](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/docs/BENCHMARKS.md) · [Diffbot model recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae/recipe/README.md)

A 1.90-second Nsight window during a fresh 60k native prompt, around positions 6,400–8,000, had compute kernels on both GPUs for 14.19% of wall time. Routed-expert projection accounted for 46.01% of summed GPU kernel time. These two percentages have different denominators. Two 128-token-chunk requests took 14.21 and 13.67 s at 16k. At 512-token chunks the existing direct-peer path took 35.58 s on the first measured 16k request and hit NVIDIA Xid 31 on the second. The diagnostic did not establish the faulting instruction. Raising the native chunk limit to 8,192 requires further implementation and qualification. [Native trace and width results](../results/upstream/cold-prefill-optimization.json)
