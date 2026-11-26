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
from contextlib import ExitStack
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def tensors(pack, layer, overlay):
    """Every expert of the requested layer lives in the same indexed shard."""
    key = f"{layer}.ffn.experts.0.w1.trellis"
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    shard = (overlay or pack) / index["weight_map"][key]
    with shard.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))

    def load(key):
        entry = header[key]
        dtype = {"I16": "<i2", "F16": "<f2"}[entry["dtype"]]
        return np.memmap(shard, dtype, mode="r", shape=tuple(entry["shape"]),
                         offset=8 + size + entry["data_offsets"][0])

    return load


def dense_trellis(trellis, mcg):
    # Advance each tile's state by appending its next 3 bits. This recurrence
    # checks the GPU's direct extraction of cyclic overlapping 16-bit windows.
    words = trellis.view(np.uint32)
    tiles_k, tiles_n, _ = words.shape
    matrix = np.empty((tiles_k, tiles_n, 16, 16), dtype=np.float16)
    states = np.arange(65536, dtype=np.uint32)
    if mcg:
        products = states * np.uint32(0xCBAC1FED)
        halves = ((products & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)).view("f2")
        codebook = halves[::2] + halves[1::2]
    else:
        products = states * np.uint32(0x83DCD12D)
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


def reference(load, layer, expert, x):
    projections = {}
    for name in ("w1", "w3", "w2"):
        base = f"{layer}.ffn.experts.{expert}.{name}"
        projections[name] = tuple(load(f"{base}.{s}") for s in ("trellis", "suh", "svh"))

    def linear(src, name):
        trellis, input_scale, output_scale = projections[name]
        matrix = dense_trellis(trellis, layer.startswith("mtp."))
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


class RoutedRow(C.Structure):
    _fields_ = [("expert", C.c_void_p), ("token", C.c_int), ("output", C.c_int),
                ("weight", C.c_float)]


def compare(lib, runtime, device, x, expected, experts, mcg):
    ptr = C.c_void_p
    lib.ds_moe_error.restype = C.c_char_p
    lib.ds_moe_alloc.argtypes = [C.POINTER(ptr)]
    lib.ds_moe_upload.argtypes = [ptr, C.POINTER(ptr)]
    lib.ds_moe_free.argtypes = [C.c_int, ptr]
    lib.ds_moe_run_dev.argtypes = [C.POINTER(ptr), C.POINTER(C.c_size_t), C.POINTER(Job),
                                   C.c_int, C.POINTER(RoutedRow), C.c_int, ptr, ptr, C.c_int, C.c_int, C.c_int, C.c_int]
    lib.ds_moe_finish.argtypes = [ptr, ptr, C.c_size_t, C.c_int]
    lib.ds_moe_shared_swiglu.argtypes = [ptr, ptr, C.c_size_t]
    runtime.cudaMalloc.argtypes = [C.POINTER(ptr), C.c_size_t]
    runtime.cudaMemcpy.argtypes = [ptr, ptr, C.c_size_t, C.c_int]
    runtime.cudaFree.argtypes = [ptr]

    def check(status):
        if status:
            raise RuntimeError(lib.ds_moe_error(status).decode())

    def upload(stack, host):
        assert host.flags.c_contiguous
        dev = ptr()
        check(runtime.cudaMalloc(C.byref(dev), host.nbytes))
        stack.callback(runtime.cudaFree, dev)
        check(runtime.cudaMemcpy(dev, host.ctypes.data, host.nbytes, 1))
        return dev

    def download(dev, host):
        check(runtime.cudaMemcpy(host.ctypes.data, dev, host.nbytes, 2))

    # The shared gate is a BF16 projection evaluated with host FP32 expf.
    gate = (np.arange(65536, dtype="u4") << 16).view("f4")
    up = np.linspace(-16, 16, len(gate), dtype="f4")
    expf = C.CDLL(None).expf
    expf.argtypes = [C.c_float]
    expf.restype = C.c_float
    clipped = np.fmin(gate, np.float32(10))
    with np.errstate(invalid="ignore"):
        activation = clipped / (1 + np.array([expf(-g) for g in clipped], dtype="f4"))
        expected_activation = activation * np.clip(up, -10, 10)
    with ExitStack() as stack:
        dev_gate = upload(stack, gate)
        check(lib.ds_moe_shared_swiglu(dev_gate, upload(stack, up), len(gate)))
        download(dev_gate, gate)
        finite = ~np.isnan(expected_activation)
        np.testing.assert_array_equal(np.isnan(gate), ~finite)
        np.testing.assert_array_equal(gate.view("u4")[finite], expected_activation.view("u4")[finite])

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
            # Shuffle both row mappings and use exactly representable FP16 weights.
            # This checks gather and weighted scatter alongside the expert matrix.
            tokens = np.roll(np.arange(n), 1)
            destinations = np.arange(n)[::-1]
            inputs = np.empty_like(x[:n])
            inputs[tokens] = x[:n]
            weights = np.linspace(.125, 1., n).astype("f2").astype("f4")
            rows = (RoutedRow * n)(*(RoutedRow(p, t, dst, w)
                                     for p, t, dst, w in zip(slots, tokens, destinations, weights)))
            output = np.zeros_like(inputs)
            truth = np.empty_like(output)
            truth[destinations] = expected[:n] * weights[:, None]
            with ExitStack() as stack:
                weighted = upload(stack, output)
                inp = upload(stack, inputs)
                for fused in (0, 1, 2):
                    check(lib.ds_moe_run_dev(C.byref(scratch), C.byref(size), (Job * len(jobs))(*jobs),
                                             len(jobs), rows, n, inp, weighted, mcg, 8, 1, fused))
                    download(weighted, output)
                    if fused == 0:
                        unfused_output = output.copy()
                    else:
                        np.testing.assert_array_equal(output.view("u4"), unfused_output.view("u4"))
                if n > 1:
                    top_k = 3 if mcg else 6
                    routed = output.reshape(n // top_k, top_k, -1)
                    shared = inputs[:len(routed)].copy()
                    dev_shared = upload(stack, shared)
                    sums = np.zeros_like(shared)
                    for rank in range(top_k):
                        sums += routed[:, rank]
                    bits = (sums + shared).view("u4")
                    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
                    check(lib.ds_moe_finish(weighted, dev_shared, shared.size, top_k))
                    download(dev_shared, shared)
                    np.testing.assert_array_equal(shared.view("u4"), rounded)
            relative_l2 = np.linalg.norm(output - truth) / np.linalg.norm(truth)
            cosine = np.vdot(output, truth) / (np.linalg.norm(output) * np.linalg.norm(truth))
            print(f"{n:2} rows   relative L2 {relative_l2:.8f}   cosine {cosine:.8f}")
            if not (relative_l2 < 0.004 and cosine > 0.99999):
                raise ValueError("expert parity check failed")
        # A 32-job verification group crosses the normal prefill split-K
        # threshold. Every weighted row must remain bit-identical to decode.
        if not mcg:
            n = 36
            with ExitStack() as stack:
                inputs = np.ascontiguousarray(x[[0, -1]])
                output = np.zeros((n, x.shape[1]), dtype="f4")
                inp, dst = upload(stack, inputs), upload(stack, output)
                rows = (RoutedRow * n)(*(RoutedRow(slabs[i % 2], i % 2, i, .5) for i in range(n)))
                jobs = (Job * n)(*(Job(slabs[i % 2], i, 1) for i in range(n)))
                check(lib.ds_moe_run_dev(C.byref(scratch), C.byref(size), jobs, n,
                                         rows, n, inp, dst, 0, 8, 1, 2))
                download(dst, output)
                for i in range(2):
                    row = RoutedRow(slabs[i], i, 0, .5)
                    job = Job(slabs[i], 0, 1)
                    decode = np.empty_like(inputs[:1])
                    check(lib.ds_moe_run_dev(C.byref(scratch), C.byref(size), C.byref(job), 1,
                                             C.byref(row), 1, inp, dst, 0, 8, 0, 0))
                    download(dst, decode)
                    np.testing.assert_array_equal(output[i::2], np.broadcast_to(decode, output[i::2].shape))
                # Compare a two-split logical prefill group with the same jobs
                # issued in two smaller physical groups. The 18-job subsets
                # must retain the full 36-job group's prescribed reduction.
                check(lib.ds_moe_run_dev(C.byref(scratch), C.byref(size), jobs, n,
                                         rows, n, inp, dst, 0, 2, 1, 0))
                download(dst, output)
                for parity in range(2):
                    n_sub = n // 2
                    subset_rows = (RoutedRow * n_sub)(*(rows[i] for i in range(parity, n, 2)))
                    subset_jobs = (Job * n_sub)(*(Job(slabs[parity], i, 1) for i in range(n_sub)))
                    check(lib.ds_moe_run_dev(C.byref(scratch), C.byref(size), subset_jobs, n_sub,
                                             subset_rows, n_sub, inp, dst, 0, 2, 1, 2))
                partitioned = np.empty_like(output)
                download(dst, partitioned)
                np.testing.assert_array_equal(partitioned.view("u4"), output.view("u4"))
    finally:
        for p in [scratch, *slabs]:
            lib.ds_moe_free(device, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--draft-stage", type=int, choices=range(3), help="check a DSpark MCG layer")
    parser.add_argument("--overlay-dir", type=Path, help="read drafter weights from an alternate shard directory")
    args = parser.parse_args()
    if args.overlay_dir and args.draft_stage is None:
        parser.error("--overlay-dir requires --draft-stage")
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    runtime = C.CDLL(str(cuda / "lib64/libcudart.so"))
    status = runtime.cudaSetDevice(args.device)
    if status:
        raise RuntimeError(f"cudaSetDevice({args.device}) failed with code {status}")
    x = np.random.default_rng(13).normal(size=(18, 5120)).astype("f4")
    layer = "layers.0" if args.draft_stage is None else f"mtp.{args.draft_stage}"
    load = tensors(args.pack, layer, args.overlay_dir)
    first, first_parts = reference(load, layer, 0, x[:17])
    second, second_parts = reference(load, layer, 9, x[17:])
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "moe.so"
        source = Path(__file__).resolve().parents[1] / "native/moe.cu"
        subprocess.run([str(cuda / "bin/nvcc"), "-std=c++17", "-O3", "-arch=sm_120",
                        "-Xcompiler", "-fPIC", "-shared", source, "-o", binary], check=True)
        compare(C.CDLL(str(binary)), runtime, args.device, x, np.concatenate([first, second]),
                [first_parts, second_parts], args.draft_stage is not None)


if __name__ == "__main__":
    main()
