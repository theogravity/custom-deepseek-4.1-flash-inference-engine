# SPDX-License-Identifier: Apache-2.0
"""Native decode gather and contiguous dequantization for the local EXL3 pack."""
import ctypes as C
import os
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton


# The runner owns one model per worker process. Like the upstream Python host
# slots, keep callback arguments and their backing mappings and tensors alive
# even if a Python table reference is dropped before a replay completes.
_CAPTURE_OWNERS = []


class Gather(C.Structure):
    _fields_ = [
        ("ids", C.c_void_p),
        ("weights", C.c_void_p),
        ("scales", C.c_void_p),
        ("staged_weights", C.c_void_p),
        ("staged_scales", C.c_void_p),
        ("rows", C.c_int64),
        ("vocab_start", C.c_int64),
        ("vocab_end", C.c_int64),
        ("weight_bytes", C.c_int64),
        ("scale_bytes", C.c_int64),
        ("weight_offset", C.c_int64),
        ("scale_offset", C.c_int64),
        ("weight_fd", C.c_int),
        ("scale_fd", C.c_int),
    ]


def _cudart():
    # Select the same major runtime already loaded by PyTorch.
    major = torch.version.cuda.split(".")[0]
    lib = C.CDLL(f"libcudart.so.{major}")
    lib.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    lib.cudaLaunchHostFunc.restype = C.c_int
    return lib


@triton.jit
def _dequantize_rows(weights, scales, out, count, QUANT_BLOCK: tl.constexpr,
                     BLOCK: tl.constexpr):
    for start in tl.range(tl.program_id(0) * BLOCK, count, tl.num_programs(0) * BLOCK):
        cols = start + tl.arange(0, BLOCK)
        packed = tl.load(weights + cols // 2, mask=cols < count, other=136).to(tl.int32)
        scale = tl.load(scales + cols // QUANT_BLOCK, mask=cols < count, other=0).to(tl.float32)
        quantized = ((packed >> ((cols % 2) * 4)) & 0xf) - 8
        tl.store(out + cols, (quantized.to(tl.float32) * scale).to(tl.bfloat16), mask=cols < count)


class NativeDecode:
    def __init__(self):
        self.max_tokens = int(os.environ.get("DSV41_NATIVE_ENGRAM_MAX_TOKENS", "64"))
        if self.max_tokens < 0:
            raise ValueError("DSV41_NATIVE_ENGRAM_MAX_TOKENS must be non-negative")
        self.records = {}
        if self.max_tokens:
            self.lib = C.CDLL(str(Path(__file__).with_name("libdsv41_engram.so")))
            self.lib.dsv41_engram_gather_size.restype = C.c_size_t
            if self.lib.dsv41_engram_gather_size() != C.sizeof(Gather):
                raise RuntimeError("native Engram ABI mismatch")
            self.cudart = _cudart()

    def run(self, table, tokens, local_heads, vocab_start, vocab_end, quant_block, out, num_sms, background):
        # The capture/enqueue path runs in Python. Each graph replay calls C++
        # directly on CUDA's host worker, between the D2H and H2D copies.
        rows = tokens * local_heads
        key = (rows, vocab_start, vocab_end)
        record = self.records.get(key)
        if record is None:
            if not (0 <= vocab_start <= vocab_end <= table.num_rows):
                raise ValueError("Engram shard range exceeds the checkpoint table")
            if local_heads != table.ids_pin.shape[1] or quant_block * table.sb != table.dim:
                raise ValueError("Engram staging layout mismatch")
            record = Gather(
                table.ids_pin.data_ptr(), table.w.data_ptr(), table.s.data_ptr(),
                table.w_pin.data_ptr(), table.s_pin.data_ptr(),
                rows, vocab_start, vocab_end, table.dim // 2, table.sb * 2,
                table.w_base, table.s_base, table.w_fd, table.s_fd,
            )
            if not self.records:
                _CAPTURE_OWNERS.append(table)
            self.records[key] = record
        err = self.cudart.cudaLaunchHostFunc(
            torch.cuda.current_stream().cuda_stream, self.lib.dsv41_engram_gather, C.byref(record)
        )
        if err:
            raise RuntimeError(f"cudaLaunchHostFunc(native Engram) returned {err}")
        table.w_gpu[:rows].copy_(table.w_pin[:rows], non_blocking=True)
        table.s_gpu[:rows].copy_(table.s_pin[:rows], non_blocking=True)
        count = rows * table.dim
        block = 512
        # Preserve the background prefetch SM budget so this layer can overlap
        # its dequantization with main-stream matrix multiplication.
        budget = max(1, num_sms // 2) if background else num_sms
        grid = min(triton.cdiv(count, block), budget)
        _dequantize_rows[(grid,)](
            table.w_gpu, table.s_gpu, out, count, quant_block, block,
        )
