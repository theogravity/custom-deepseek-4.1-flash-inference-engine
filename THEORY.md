# Throughput on two RTX PRO 6000s

At 375 W per GPU, the short-context weight-streaming calculation gives **307 output tokens/s for one stream without speculation** using the recipe's FP8 output head. DSpark can emit several verified tokens per step. With five draft positions, independent routing and acceptance fractions observed on this workstation, the bandwidth calculation gives **408–601 output tokens/s for one stream** or **693–1,021 tokens/s in aggregate for six**. These are conditional engineering estimates. The first stock control measured 236 and 638 end-to-end output tokens/s respectively on the warm 25,598-token code-edit benchmark. [Control measurements](results/ledger.jsonl)

## Hardware measurements

Both RTX PRO 6000 Blackwell Workstation cards ran the same operation simultaneously on 2026-11-25. Each has 97,887 MiB of usable memory reported by the driver (95.6 GiB). They communicate over PCIe through the host bridges. The host is a 24-core Threadripper PRO 9965WX with 246 GiB of RAM. The desktop RTX 5080 was outside this test. NVIDIA rates each workstation card at 1,792 GB/s at its reference settings. [NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)

| GPU operation | GPU 1 | GPU 2 | Equal-work TP2 reference |
| --- | ---: | ---: | ---: |
| 512 MiB device copy, read + write GB/s | 1,702.39 | 1,702.74 | 3,404.78 GB/s |
| BF16, 4,096 cubed | 300.25 | 269.71 | 539.42 TFLOP/s |
| BF16, 8,192 cubed | 274.86 | 246.81 | 493.62 TFLOP/s |
| FP8, 4,096 cubed | 552.31 | 497.60 | 995.20 TFLOP/s |
| FP8, 8,192 cubed | 543.15 | 489.43 | 978.86 TFLOP/s |

Each entry is the median of seven CUDA-event measurements after warm-up. GEMM counts a multiply-add as two FLOPs. The final column doubles the slower card's rate because equal shards synchronize. Memory clocks reached 15,365 MHz during the sustained samples. The 8,192-cubed GEMMs ran at roughly 1.49–1.50 GHz SM clock on GPU 1 and 1.35–1.37 GHz on GPU 2. Both cards reached their 375 W limits. Copy bandwidth measures mixed reads and writes. Decoder weight reads, tensor-core kernels and communication have different access patterns. [Individual samples, clocks, power and device identifiers](results/roofline-hardware.json)

## Ordinary decode

The model has 40 target layers. Each token selects six of 384 routed experts per layer. The following byte counts come from the tensor offsets in the local pack's 48 safetensors headers. GB uses decimal units. The table assumes one read of each selected matrix, balanced partitioning over both cards, warm Engram pages and negligible short-context KV traffic. [Checkpoint configuration](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae/config.json)

| Component | GB per single target token |
| --- | ---: |
| Attention projections, compressors, indexers and scales | 5.152 |
| Shared experts | 1.417 |
| Routers, normalization and hyper-connections | 0.316 |
| Engram projections | 0.315 |
| EXL3 routed experts, 204.528 GB × 6/384, including scales and side tensors | 3.196 |
| FP8 vocabulary head, 129,280 × 5,120 with one scale byte per 32 values | 0.683 |
| One token-embedding row plus 48 packed Engram rows with scales | 0.000017 |
| **Total** | **11.077** |

3,404.78 GB/s ÷ 11.077 GB gives 307 tokens/s, or 3.25 ms per token at the bandwidth reference. The checkpoint's original BF16 output head increases the total to 11.719 GB and gives 291 tokens/s. For six streams with independent expert choices, a joint step reads an estimated 26.32 GB with the FP8 head, giving 776 aggregate tokens/s without speculation. Actual rank replication, expanded scales, intermediate reads and writes, collectives, scheduling and variable batch occupancy affect the result. CPU execution of some experts and reuse in GPU L2 also change how much of this traffic reaches GDDR7.

The 110.60 GB Engram tables contribute 6,912 logical row bytes per target token. Random-page faults and synchronization cost time even when the byte count is small. The model also has 204.53 GB of target expert weights. The deployed plan places 5,296 of 15,360 experts in host RAM, with 512 cache slots on each GPU. The plan's training sessions routed 4.4% of decode choices to host experts before caching. That fraction depends on the text. For scale, 4.4% of 3.196 GB is 141 MB, about 0.56 ms at the previously measured 251.8 GB/s host-memory bandwidth before unpacking and multiplication. GPU and CPU execution can overlap within each layer. [Upstream routing plan](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/6110453b901e4b51da66f539f3802b5c7f18e8e8/recipe/plans/README.md) · [Workstation tuning measurements](../docs/dsv41-flash-exl3-tuning-2026-11.md)

## DSpark decode

One step runs the three-layer draft backbone once across five positions, then verifies six positions together. Target and draft each project through the shared vocabulary head. The draft also applies five sequential low-rank Markov projections. The calculation charges a full read on each GPU for each replicated Markov projection. L2 reuse can reduce that cost. For an expert selected with probability q in a row, the expected proportion of experts used by m independent rows is 1 − (1 − q)ᵐ. With acceptance p equal to accepted draft tokens divided by all proposed draft tokens, the mean output per full step is 1 + 5p per stream. [DSpark implementation in the checkpoint](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae/inference/model.py)

| Assumed routes within each layer | Target + draft per step | Ideal time | 0% accepted | 48.3% | 80.6% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One stream, independent uniform | 26.32 + 2.18 GB | 8.37 ms | 119 | 408 | 601 | 717 |
| Six streams, independent uniform, aggregate | 96.39 + 4.21 GB | 29.55 ms | 203 | 693 | 1,021 | 1,218 |
| One stream, identical expert choices across all positions | 11.08 + 1.72 GB | 3.76 ms | 266 | 908 | 1,338 | 1,596 |

The final row isolates the sensitivity to expert reuse under extreme locality. Rates in the four rightmost columns are estimated output tokens/s. At fixed acceptance, route correlations can materially change step time. Consecutive tokens and related code prompts frequently share experts. Different prompts also move different portions of the working set into host memory. The historical 48.3% and 80.6% acceptance came from reasoning and code-edit workloads respectively. The acceptance of the current control run was unmeasured. Previous free-form, temperature-0.6 output measured 60–95 tokens/s on one stream. Comparing client-facing performance requires the same prompts, sampling settings, output lengths, context and timing boundary. [Historical workloads and acceptance measurements](../docs/dsv41-flash-exl3-tuning-2026-11.md)

## Cold text prefill

The target executes about 15.47 billion active **linear** parameters per token across all 40 layers, including 8.49 billion routed expert parameters (40 × 6 × 3 × 5,120 × 2,304). The first 20 layers contain about 7.89 billion. The recipe computes full prompt rows through layers 0–19, writes the source KV at layer 20, then can replay the last 128 positions through layers 20–39 for ordinary generation. This roughly halves the arithmetic of a long prompt. Returning full prompt log probabilities can require the full-depth path and a vocabulary projection per prompt row. The full-depth column covers the backbone alone. The figures below assume no cached prefix. [Upstream prefill implementation](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/6110453b901e4b51da66f539f3802b5c7f18e8e8/recipe/patches/vllm/decoder_replay_layers_skip.py)

| Prompt tokens | Full-depth GFLOP/token | Tail-replay GFLOP/token | Tail replay at 50% BF16 reference | Tail replay at 50% FP8 reference |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 33.72 | 18.10 | 13,638 tok/s | 27,045 tok/s |
| 40,000 | 34.97 | 17.63 | 14,002 tok/s | 27,767 tok/s |
| 524,288 | 40.05 | 20.57 | 12,001 tok/s | 23,799 tok/s |

The numerator is half the measured 8,192-cubed TP2 rate. The denominator counts linear multiply-adds, QK and AV over the 128-token window plus up to 512 indexed positions, and indexer dot products. At 40k, using the measured GEMM rates at 100% would give about 28.0k BF16-equivalent or 55.5k FP8-equivalent tokens/s with replay. The 50% rows illustrate how assumed compute utilization changes the estimate. A production step mixes FP8 projections, EXL3 dequantization, integer or BF16 expert math and sparse attention. Small or skewed expert batches, PCIe reads, softmax, routing, packing, tokenization and tail handling reduce achieved throughput. Extra replay rows at chunk boundaries also affect the total. The original local recipe processed a cold 498k prompt in 39.9 s (about 12.5k tok/s) in a previous 600 W run. GPU clock and power differences preclude a direct comparison to the current test. [Historical cold prefill measurement](../docs/dsv41-flash-exl3-tuning-2026-11.md)

## Reproduce

From this repository, use the recorded probe and read only local checkpoint headers to recalculate the table.

```sh
python3 tools/roofline.py /mnt/models/dsv41-flash-exl3
```

With both inference GPUs free, repeat the GPU probe in the locally installed image. It uses about 2 GiB per card, writes the report to stdout and takes about 30 seconds.

```sh
docker run --rm --gpus '"device=1,2"' \
  --entrypoint /usr/bin/python3 \
  -v "$PWD/tools/roofline_probe.py:/probe.py:ro" \
  dsv41-flash-1017-exl3-3bpw-sm120-lmcache /probe.py > /tmp/dsv41-roofline-hardware.json
python3 tools/roofline.py /mnt/models/dsv41-flash-exl3 --probe /tmp/dsv41-roofline-hardware.json
```
