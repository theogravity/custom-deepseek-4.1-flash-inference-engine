# Historical CUDA microbenchmarks

Each driver selects the same kernel revision used for its experiment. The two early MoE trials also require the included patch against the pre-experiment tree. The build helper extracts sources into a temporary directory. It accepts a local experiment branch or its remote-tracking branch after a standard clone. Use Rust engine sources from this repository and CUDA 13.5 with sm_120f support.

```sh
python3 benchmarks/micro/build.py --help
python3 benchmarks/micro/build.py decode-round2-20261126-head-micro --output /tmp/ds4-head-micro
```

The executable runs on the first visible CUDA device. On the measured workstation, run during a dedicated profiling window and select an inference GPU explicitly.

```sh
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader) /tmp/ds4-head-micro
```
