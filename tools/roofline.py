#!/usr/bin/env python3
"""Single-stream weight bandwidth model for the native engine at short context."""
import argparse
import json
import struct
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=Path("/mnt/models/dsv41-flash-exl3"))
    parser.add_argument("--gbps", nargs=2, type=float, default=[1702.39, 1702.74], help="Measured memory bandwidth for each stage, decimal GB/s")
    args = parser.parse_args()
    # At <=512 compressed rows every candidate fits. The indexer can skip search.
    stages = [Counter(), Counter()]
    for path in sorted(args.checkpoint.glob("model-*-of-*.safetensors")):
        with path.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, row in header.items():
            if name == "__metadata__":
                continue
            elements = 1
            for n in row["shape"]:
                elements *= n
            if name == "head.weight":
                stages[1]["Packed FP8 vocabulary head with scales"] += elements + elements//32
                continue
            parts = name.split(".", 2)
            if len(parts) != 3 or parts[0] != "layers":
                continue
            layer = int(parts[1])
            if layer >= 40:
                continue
            field = parts[2]
            stage = stages[layer//20]
            if field.startswith("ffn.experts."):
                if field.endswith((".trellis",".suh",".svh")):
                    a,b = row["data_offsets"]
                    stage["Selected EXL3 experts"] += (b-a)*6/384
            elif field in ("hc_attn_fn", "hc_ffn_fn"):
                stage["FP32 residual projections"] += 4*elements
            elif field in ("attn.indexer.wq_b.weight","attn.indexer.weights_proj.weight"):
                continue
            elif field.endswith(".weight") and len(row["shape"]) == 2 and field != "engram.embed.weight":
                width = 4 if row["dtype"] == "F32" else 2
                # The key projection is applied once per complete compression group.
                rate = 0.5 if field == "attn.indexer.wk.weight" and layer in (2,8,14) else 1
                stage["Other dense projections"] += width*elements*rate
    assert all(stage["Selected EXL3 experts"] > 1e9 for stage in stages)
    seconds = sum(sum(stage.values()) / (gbps*1e9) for stage,gbps in zip(stages,args.gbps))
    print(json.dumps({"stage_GB": [{k:round(v/1e9,4) for k,v in stage.items()} for stage in stages],
                      "assumed_GB_per_s": args.gbps, "weight_read_floor_ms":round(1000*seconds,3),
                      "ideal_decode_tok_s":round(1/seconds,2)},indent=2))

if __name__ == "__main__":
    main()
