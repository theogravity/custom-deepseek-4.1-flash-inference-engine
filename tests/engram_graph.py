"""Exercise the installed overlay on checkpoint bytes and actual CUDA graph replays."""
import argparse
import functools
import json
from pathlib import Path
import statistics
import struct
import time

import torch
from vllm.models.deepseek_v41.common import engram


DIM = 256
QUANT_BLOCK = 32
HEADS = 24


def row_reader(directory, name, width):
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    with (directory / weight_map[name]).open("rb") as file:
        length, = struct.unpack("<Q", file.read(8))
        start = 8 + length + json.loads(file.read(length))[name]["data_offsets"][0]

    @functools.lru_cache(4096)
    def read(row):
        with (directory / weight_map[name]).open("rb") as file:
            file.seek(start + row * width)
            return bytearray(file.read(width))
    return read


def reference(lookup_weights, lookup_scales, indices, start, local, low, high):
    columns = torch.arange(DIM)
    result = torch.zeros((indices.shape[0], local, DIM), dtype=torch.bfloat16)
    for token in range(indices.shape[0]):
        for h in range(min(local, HEADS - start)):
            row = indices[token, h + start].item()
            if low <= row < high:
                weight = torch.frombuffer(lookup_weights(row), dtype=torch.uint8)
                scales = torch.frombuffer(lookup_scales(row), dtype=torch.float16)
                nibble = (weight[columns // 2].int() >> ((columns % 2) * 4)) & 15
                result[token, h] = (nibble - 8).float() * scales[columns // QUANT_BLOCK].float()
    return result


def check_case(table, readers, tokens, start, local, isolate_native, timed=False):
    maximum = table.native_decode.max_tokens
    low, high = 0, table.num_rows
    if start:
        low, high = 110_000_000, 249_000_000
    candidates = torch.tensor([-1, low - 1, low, low + 1, low + 3, high - 1, high, high + 1,
                               low + 230794, low + 120063, low + 86532, low + 9385257])

    def fresh():
        return candidates[torch.randint(len(candidates), (tokens, HEADS))]

    # A strided view of all layers' hashes, as supplied by real decode.
    storage = torch.zeros(tokens, 2, HEADS, device="cuda", dtype=torch.int64)
    indices = storage[:, 1]
    output = torch.empty(tokens, local, DIM, device="cuda", dtype=torch.bfloat16)
    info = dict(head_start=start, local_heads=local, total_heads=HEADS,
                vocab_start=low, vocab_end=high, quant_block=QUANT_BLOCK,
                num_sms=torch.cuda.get_device_properties(0).multi_processor_count, background=True)
    def run():
        table.lookup(indices, output, **info)

    ids = fresh()
    indices.copy_(ids)
    run()
    torch.cuda.synchronize()
    if isolate_native:
        assert (tokens * local, low, high) in table.native_decode.records
    ref = reference(*readers, ids, start, local, low, high)
    assert torch.equal(output.cpu().view(torch.int16), ref.view(torch.int16)), "eager BF16 mismatch"
    if tokens == 0:
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        ids = fresh()
        indices.copy_(ids)
        output.fill_(float("nan"))
        graph.replay()
        ref = reference(*readers, ids, start, local, low, high)
        assert torch.equal(output.cpu().view(torch.int16), ref.view(torch.int16)), "graph BF16 mismatch"
    if timed:
        def replay_time(graph):
            samples = []
            for _ in range(35):
                t0 = time.perf_counter_ns()
                graph.replay()
                torch.cuda.synchronize()
                samples.append((time.perf_counter_ns() - t0) / 1000)
            return round(statistics.median(samples), 1)

        native_us = replay_time(graph)
        table.native_decode.max_tokens = 0
        try:
            run()
            torch.cuda.synchronize()
            fallback_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fallback_graph):
                run()
            fallback_us = replay_time(fallback_graph)
            assert torch.equal(output.cpu().view(torch.int16), ref.view(torch.int16)), "baseline mismatch"
        finally:
            table.native_decode.max_tokens = maximum
        print(f"warm fixed-ID graph median of 35: {tokens} tokens native={native_us} us "
              f"original={fallback_us} us", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/models/dsv41-flash-exl3"))
    args = parser.parse_args()
    torch.manual_seed(419)
    torch.set_num_threads(1)
    for layer, local, start in [(1, 12, 0), (1, 12, 12), (14, 12, 18), (14, 24, 0)]:
        with torch.device("cuda"):
            table = engram._NvmeEngramTable(str(args.model), layer, DIM, QUANT_BLOCK,
                                            max_tokens=65, local_heads=local, total_heads=HEADS,
                                            device=torch.device("cuda"))
        assert table.native_decode.max_tokens == 64, "test expects default decode threshold"
        key = f"layers.{layer}.engram.embed."
        readers = (row_reader(args.model, key + "weight", DIM // 2),
                   row_reader(args.model, key + "scale", DIM // QUANT_BLOCK * 2))
        for tokens in (0, 1, 6, 36, 64, 65):
            check_case(table, readers, tokens, start, local, 0 < tokens <= 64,
                       timed=(layer, start) == (1, 12) and tokens in (6, 36))
        print(f"PASS layer={layer} start={start} local={local} tokens=0,1,6,36,64,65 "
              "(eager and graph for nonempty shapes)", flush=True)


if __name__ == "__main__":
    main()
