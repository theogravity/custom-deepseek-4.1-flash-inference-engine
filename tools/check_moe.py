#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Compare the routed CUDA FFN to independently reconstructed checkpoint matrices.

    uv run tools/check_moe.py /mnt/models/dsv41-flash-exl3 --device 2

The device argument uses CUDA's visible ordinals. The CUDA toolkit is required.
"""
import argparse
import ctypes as C
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def tensors(pack):
    """The first text layer is stored together in one safetensors shard."""
    key = "layers.0.ffn.experts.0.w1.trellis"
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    shard = pack / index["weight_map"][key]
    with shard.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))

    def load(key):
        entry = header[key]
        dtype = {"I16": "<i2", "F16": "<f2"}[entry["dtype"]]
        return np.memmap(shard, dtype, mode="r", shape=tuple(entry["shape"]),
                         offset=8 + size + entry["data_offsets"][0])

    return load


def dense_trellis(trellis):
    # Advance each tile's state by appending its next 3 bits. This recurrence
    # checks the GPU's direct extraction of cyclic overlapping 16-bit windows.
    words = trellis.view(np.uint32)
    tiles_k, tiles_n, _ = words.shape
    matrix = np.empty((tiles_k, tiles_n, 16, 16), dtype=np.float16)
    products = np.arange(65536, dtype=np.uint32) * np.uint32(0x83DCD12D)
    byte_sums = sum((products >> i) & 255 for i in (0, 8, 16, 24)).astype("f4")
    scale = np.uint16(0x1EEE).view("f2").astype("f4")
    bias = np.uint16(0xC931).view("f2").astype("f4")
    codebook = ((byte_sums + 1024) * scale + bias).astype("f2")
    state = words[:, :, -1] & 65535
    for i in range(256):
        word, bit = divmod(i * 3, 32)
        if bit <= 29:
            following = words[:, :, word] >> (29 - bit)
        else:
            following = ((words[:, :, word] << (bit - 29))
                         | (words[:, :, (word + 1) % 24] >> (61 - bit)))
        state = ((state << 3) | (following & 7)) & 65535
        lane, j = divmod(i, 8)
        row = (lane % 4) * 2 + (0, 1, 8, 9)[j % 4]
        col = lane // 4 + 8 * (j // 4)
        matrix[:, :, row, col] = codebook[state]
    return np.ascontiguousarray(matrix.transpose(0, 2, 1, 3)
                                .reshape(tiles_k * 16, tiles_n * 16), dtype="f4")


def hadamard(x):
    blocks = x.astype("f4").reshape(-1, 128)
    for step in (1, 2, 4, 8, 16, 32, 64):
        pairs = blocks.reshape(-1, 2, step)
        a, b = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
        pairs[:, 0, :], pairs[:, 1, :] = a + b, a - b
    return blocks.reshape(x.shape) * np.float32(128**-0.5)


def reference(load, expert, x):
    projections = {}
    for name in ("w1", "w3", "w2"):
        base = f"layers.0.ffn.experts.{expert}.{name}"
        projections[name] = tuple(load(f"{base}.{s}") for s in ("trellis", "suh", "svh"))

    def linear(src, name):
        trellis, input_scale, output_scale = projections[name]
        matrix = dense_trellis(trellis)
        rotated = hadamard(src.astype("f2") * input_scale).astype("f2").astype("f4")
        result = hadamard((rotated @ matrix).astype("f2"))
        if name == "w2":
            return result * output_scale
        return (result.astype("f2") * output_scale).astype("f4")

    gate = linear(x, "w1").astype("f2")
    up = linear(x, "w3").astype("f2").clip(-10, 10)
    # The routed kernel evaluates SiLU in half, then applies the gate limit.
    with np.errstate(over="ignore"):
        activated = gate * np.reciprocal(np.float16(1) + np.exp(-gate))
    expected = linear(activated.clip(max=10) * up, "w2")
    parts = [parts[0] for parts in projections.values()]
    parts.extend(scale for _, suh, svh in projections.values() for scale in (suh, svh))
    return expected, parts


class Job(C.Structure):
    _fields_ = [("expert", C.c_void_p), ("begin", C.c_int), ("count", C.c_int)]


def compare(lib, device, x, expected, experts):
    ptr = C.c_void_p
    lib.ds_moe_error.restype = C.c_char_p
    lib.ds_moe_alloc.argtypes = [C.POINTER(ptr)]
    lib.ds_moe_upload.argtypes = [ptr, C.POINTER(ptr)]
    lib.ds_moe_free.argtypes = [C.c_int, ptr]
    lib.ds_moe_run.argtypes = [C.POINTER(ptr), C.POINTER(C.c_size_t), C.POINTER(Job),
                               C.c_int, C.POINTER(ptr), ptr, ptr, C.c_int]

    def check(status):
        if status:
            raise RuntimeError(lib.ds_moe_error(status).decode())

    slabs = []
    scratch = ptr()
    size = C.c_size_t()
    try:
        for parts in experts:
            allocation = ptr()
            check(lib.ds_moe_alloc(C.byref(allocation)))
            slabs.append(allocation)
            check(lib.ds_moe_upload(allocation, (ptr * 9)(*(p.ctypes.data for p in parts))))
        first, second = slabs
        for jobs, slots in (
            ([Job(first, 0, 1)], [first]),
            ([Job(first, 0, 16), Job(first, 16, 1), Job(second, 17, 1)], [first] * 17 + [second]),
        ):
            n = len(slots)
            output = np.empty_like(x[:n])
            check(lib.ds_moe_run(C.byref(scratch), C.byref(size), (Job * len(jobs))(*jobs),
                                 len(jobs), (ptr * n)(*slots), x.ctypes.data, output.ctypes.data, n))
            truth = expected[:n]
            relative_l2 = np.linalg.norm(output - truth) / np.linalg.norm(truth)
            cosine = np.vdot(output, truth) / (np.linalg.norm(output) * np.linalg.norm(truth))
            print(f"{n:2} rows   relative L2 {relative_l2:.8f}   cosine {cosine:.8f}")
            if not (relative_l2 < 0.004 and cosine > 0.99999):
                raise ValueError("expert parity check failed")
    finally:
        for p in [scratch, *slabs]:
            lib.ds_moe_free(device, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    runtime = C.CDLL(str(cuda / "lib64/libcudart.so"))
    status = runtime.cudaSetDevice(args.device)
    if status:
        raise RuntimeError(f"cudaSetDevice({args.device}) failed with code {status}")
    x = np.random.default_rng(13).normal(size=(18, 5120)).astype("f4")
    load = tensors(args.pack)
    first, first_parts = reference(load, 0, x[:17])
    second, second_parts = reference(load, 9, x[17:])
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "moe.so"
        source = Path(__file__).resolve().parents[1] / "native/moe.cu"
        subprocess.run([str(cuda / "bin/nvcc"), "-std=c++17", "-O3", "-arch=sm_120",
                        "-Xcompiler", "-fPIC", "-shared", source, "-o", binary], check=True)
        compare(C.CDLL(str(binary)), args.device, x, np.concatenate([first, second]),
                [first_parts, second_parts])


if __name__ == "__main__":
    main()
