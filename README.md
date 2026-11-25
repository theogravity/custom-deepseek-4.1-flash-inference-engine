# DeepSeek V4.1 Flash local engine

This is a custom build of the two-GPU EXL3 inference stack for Theo's workstation. It runs the calibrated 3-bit checkpoint on the two RTX PRO 6000s and serves its existing LiteLLM routes. It builds on the Diffbot vLLM executor, DSpark speculative decoder and EXL3 kernels. The new C++ and Triton backend executes the model's two NVMe-backed Engram layers during decode.

For small batches, CUDA graph replay calls a native host function that gathers packed rows and scales directly into their final staging slots. A GPU kernel then decodes the two int4 values per byte into BF16. This removes Python, PyTorch sorting and the compact-index transfer from those graph nodes. Large prefills use the existing parallel CPU lookup. See [the integration](integration/) and [native implementation](native/engram.cpp).

## Run on this workstation

The parent LLM checkout provides the generated model Compose file, its per-project environment, the installed recipe image, built EXL3 kernels, model weights and LiteLLM. See the [parent setup guide](../AGENTS.md) for provisioning. The child repository contains the runtime modifications and builds them locally. GPU 1 and 2 must be available. If another rotation model occupies them, stop it with model-cli first.

```sh
cd ~/llm/dsv41-engine
./engine build     # compile the native library inside the installed recipe image
./engine start     # switch the existing DeepSeek service and wait for health
./engine status    # expect "running healthy native-engram"
```

For a separate checkout, set `LLM_STACK_DIR` to the parent stack directory. Build artifacts live under `.build/` in directories keyed by the source and image. Keep that directory in place while the native service is running. The builder checks the upstream Engram module hash. If the parent runtime changes, review the integration before updating the pin.

The running service uses the same model name, container, host port 8001, authentication, 524,288-token context, six sequence slots and RAM/disk prefix caches. The LiteLLM proxy runs on port 4000. Start clients from your normal working directory with the installed launchers.

```sh
eval "$(model-cli codex)"
eval "$(model-cli claude)"
eval "$(model-cli pi)"
```

Run one launcher per shell. `model-cli pi --alias deepseek` also uses the OpenAI-compatible route. A Docker restart retains the active build. If activation fails, the launcher attempts to restore the previously running build. After switching to another model with model-cli, use `./engine start` to select this build again once the other model has stopped. Run `./engine stock` for the original implementation, or `./engine stop` to release both inference GPUs. If the DeepSeek worker is stuck and you intend to interrupt active requests, use `./engine stop --interrupt` followed by `./engine stock`. The permissive client options printed by model-cli are the parent stack's defaults.

## Reproduce the measurements

`python3 tools/measure.py LABEL results/ledger.jsonl` benchmarks the engine directly on localhost:8001. It discards seven warm-up runs at each concurrency, then collects six timed runs at each concurrency using 512 generated tokens per stream. Rates divide server-reported completion tokens by wall time including the warm prompt and stream setup. The ledger records TTFT and a digest of each generated answer. Compare runs on the same hardware and power limits.

`python3 tools/acceptance.py` exercises the local LiteLLM proxy and installed coding clients. It creates a temporary file with a random receipt for each client, asks the client to read it with a tool, and checks its final answer.

See [THEORY.md](THEORY.md) for the throughput model and [PLAN.md](PLAN.md) for measured results, workload limits and the tested system state.
