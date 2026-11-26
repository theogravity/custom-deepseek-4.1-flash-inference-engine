# Selective CPU target experts at six independent 404k contexts

| Output tokens per client | GPU before, tok/s | CPU enabled, tok/s | GPU after, tok/s | CPU experts per warm wave | CPU responses identical to GPU |
|---|---:|---:|---:|---:|---:|
| 512 | 297.2 | 266.4 | 298.8 | 452–755 | 4/36 |
| 768 | 320.9 | 289.4 | 324.9 | 765–1112 | 0/36 |

All rates are combined visible decode throughput from the first client's first output to the last client's last output, excluding one token per client. The six clients sent independent frozen coding prompts of 404,151–404,170 tokens. Each configuration used the same executable, a fresh process, one warm-up and six measured waves, with 2,424,704 prefix tokens reused per measured wave. Every wave had six-way verification, six successful code-format checks and zero live KV rotations. Both GPUs were capped at 375 W and the CPU socket at 225 W.

The opt-in Rust path schedules up to four uncached target experts (up to two routes per expert) on a ten-thread host pool when the owning GPU cache is under pressure. The next eligible miss admits that expert to GPU. The CPU runs the pack's host-matched `cepf_*` sparse job through its C ABI while the inference thread issues GPU work. This kernel uses int16 activation quantization and emits weighted FP16 rows. Prefill, drafter experts and the shared expert stay on GPU. The original vLLM decode runtime uses a different int8 kernel and mailbox. The tested bridge synchronously copies selected activation rows to the host and completed rows back to their original rank slots.

A fixed-route comparison exercised three target layers across both GPU owners, one- and two-row experts, two different weights for the same expert, resident and CPU routes and untouched output slots. Relative L2 error on the nine CPU-computed rows ranged from 0.000903 to 0.001040. Both GPU-only controls matched all saved coding responses. The default-cap receipt workload returned all 12 expected early/late sequences but used zero CPU experts. A separate retrieval check reduced the expert cache to 4 GiB per GPU to exercise CPU execution. It completed 6238 CPU jobs over two warm waves and returned all 12 expected sequences. The stress run is a correctness check only. The code checks validate output format and the receipt check validates retrieval; neither measures general coding accuracy.

The experiment branch exposes `DS4_CPU_EXPERTS=1` on its own rebuilt binary. `cargo build --release --locked` followed by `docker build -t ds4-flash-rs:cpu-expert-experiment .` builds it from the experiment checkout. The base Compose template forwards the option and the GPU-only default is `0`. The deployment image must contain that binary to exercise it. The `cpu_expert_jobs` and `cpu_expert_rows` metrics count completed expert groups and routed token rows.

The CPU option remains on its experiment branch. The shared-prefix LFU GPU service is the selected baseline. The measured image predates a constructor mutex and a correction to optional route diagnostics. The timed path used one model runtime per process, with those diagnostics disabled. [Per-wave results](results/selective-cpu-experts.json)
