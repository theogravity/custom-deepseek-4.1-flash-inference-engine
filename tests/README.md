# Real-checkpoint Engram parity

From the engine checkout, build the overlay and run the test on an available inference card. The test uses an ephemeral container and the existing checkpoint. It does not need the model server. The measurements in `results/engram-graph.json` used host GPU 1 with both inference GPUs idle.

```bash
cd ~/llm/dsv41-engine
overlay=$(./engine build)
common=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common
pack=/mnt/models/dsv41-flash-exl3
docker run --rm --network none --gpus device=1 \
  -e DSV41_ENGRAM_INT4=1 -e DSV41_ENGRAM_NVME=1 \
  -v "$pack:$pack:ro" -v "$PWD/tests:/tests:ro" \
  -v "$overlay/engram.py:$common/engram.py:ro" \
  -v "$overlay/dsv41_native_engram.py:$common/dsv41_native_engram.py:ro" \
  -v "$overlay/libdsv41_engram.so:$common/libdsv41_engram.so:ro" \
  --entrypoint python3 dsv41-flash-1017-exl3-3bpw-sm120-lmcache \
  -I /tests/engram_graph.py
```

The test independently reads packed rows and FP16 scales from both Engram shards, computes a CPU reference, and compares BF16 bits for all output elements. It checks both TP2 head partitions, TP1, a padded tail, and strided hash inputs. Each nonempty shape runs eagerly and replays a CUDA graph with three fresh sets of IDs. The 64/65-token cases exercise both sides of the native cutoff. The 6- and 36-token cases also time full graph replays against the Python host-node implementation on fixed, warm IDs.
