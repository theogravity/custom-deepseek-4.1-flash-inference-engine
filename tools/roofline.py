"""Estimate text-only TP2 rooflines from the 48-shard EXL3 checkpoint and a local probe."""
import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path


def read_inventory(pack, cfg):
    table = Counter()
    matrix_params = Counter()
    layer20_kv = 0
    expert_weight = ".ffn.experts."
    shard_names = set(json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"].values())
    for name in sorted(shard_names):
        with (pack / name).open("rb") as f:
            header_length, = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_length))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            size = end - start
            shape = info["shape"]
            if key.startswith(("vision.", "aligner.", "image_")):
                table["vision"] += size
            elif key == "embed.weight":
                table["token_embedding"] += size
                table["token_row"] = size // shape[0]
            elif key == "head.weight":
                table["lm_head_bf16"] += size
                table["lm_head_fp8"] += math.prod(shape) * 33 // 32
            elif key.startswith("mtp."):
                if expert_weight in key:
                    table["draft_expert_pool"] += size
                elif "markov_head" in key:
                    if ".head.weight" in key:
                        # Hook converts all ParallelLMHead weights incl. the replicated Markov projection.
                        table["markov_fp8"] = math.prod(shape) * 33 // 32
                        table["markov_bf16"] = size
                else:
                    table["draft_dense"] += size
            elif ".engram.embed." in key:
                table["engram_tables"] += size
                rows = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
                table["engram_rows"] += size // shape[0] * rows
            elif expert_weight in key:
                table["target_expert_pool"] += size
            elif key.startswith("layers."):
                layer = int(key.split(".")[1])
                if ".ffn.shared_experts." in key:
                    group = "shared"
                elif ".ffn.gate." in key:
                    group = "router"
                elif ".attn." in key:
                    group = "attention"
                elif ".engram." in key:
                    group = "engram_projections"
                else:
                    group = "norm_mhc"
                table[group] += size
                # Stored matrices that perform learned linear projections. Skip index tables, scale arrays, norms.
                if (key.endswith(".weight") and len(shape) == 2 and
                        not key.endswith((".k_weight", ".q_weight"))) or key.endswith(("_attn_fn", "_ffn_fn")):
                    params = math.prod(shape)
                    matrix_params[layer] += params
                    if layer == max(cfg["kv_source_layer_ids"]) and key.endswith(
                            ("hc_attn_fn", "attn.wkv.weight", "compressor.wkv.weight", "indexer.wk.weight")):
                        layer20_kv += params
            elif key == "norm.weight":
                table["norm_mhc"] += size
            else:
                raise ValueError(f"unexpected checkpoint tensor {key}")
    active_expert_params = (3 * cfg["hidden_size"] * cfg["moe_intermediate_size"]
                            * cfg["num_experts_per_tok"])
    for layer in matrix_params:
        matrix_params[layer] += active_expert_params
    return table, matrix_params, layer20_kv


def union_fraction(top_k, experts, rows, identical):
    """Mean fraction of distinct experts if rows share every choice or are independent."""
    if identical:
        return top_k / experts
    return 1 - (1 - top_k / experts) ** rows


def mean_visible(start, stop, ratio, cap):
    """Mean min(position / ratio, cap) over (start, stop], continuous approximation."""
    def integrated(x):
        cut = min(x, cap * ratio)
        return cut ** 2 / (2 * ratio) + (x - cut) * cap
    return (integrated(stop) - integrated(start)) / (stop - start)


def prefill_flops(cfg, matrix_params, kv_params, length, trim_replay):
    """FMA=2. Include QK/AV sparse attention and indexer dots, omit activation and reduction ops."""
    total = 0
    replay_layer = max(cfg["kv_source_layer_ids"])
    for layer, params in matrix_params.items():
        start = 0
        if trim_replay and layer >= replay_layer:
            start = max(0, length - cfg["sliding_window"])
        fraction = (length - start) / length
        ratio = cfg["compress_ratios"][layer]
        positions = mean_visible(start, length, 1, cfg["sliding_window"])
        if ratio:
            positions += mean_visible(start, length, ratio, cfg["index_topk"])
        attention = 4 * cfg["num_attention_heads"] * cfg["head_dim"] * positions
        indexer = 0
        if layer in cfg["index_source_layer_ids"]:
            cap = length
            if layer > cfg["candidate_source_layer_id"]:
                cap = cfg["candidate_topk_blocks"] * cfg["candidate_block_size"]
            positions = mean_visible(start, length, ratio, cap)
            indexer = 2 * cfg["index_n_heads"] * cfg["index_head_dim"] * positions
        total += fraction * (2 * params + attention + indexer)
    if trim_replay:
        # Full-batch KV write for first replay layer; final replay rows can be recomputed.
        total += 2 * kv_params
    return total


def calculate(pack, probe):
    cfg = json.loads((pack / "config.json").read_text())["text_config"]
    table, matrix_params, kv_params = read_inventory(pack, cfg)
    measured = json.loads(probe.read_text())["runs"]
    # Equal TP2 work synchronizes at the slower card.
    rates = {(r["kind"], r["size"]): 2 * min(device["median"] for device in r["devices"])
             for r in measured}
    bandwidth = rates[("copy", 512 * 2**20)] * 1e9
    dense = sum(table[k] for k in ["attention", "shared", "router", "norm_mhc", "engram_projections"])
    lookups = table["engram_rows"] + table["token_row"]
    active = table["target_expert_pool"] * cfg["num_experts_per_tok"] / cfg["n_routed_experts"]
    decode = {}
    for head in ["lm_head_bf16", "lm_head_fp8"]:
        target_bytes = dense + active + table[head] + lookups
        decode[head] = {"bytes": target_bytes, "tokens_s": bandwidth / target_bytes}
    n = 6
    target_union = union_fraction(cfg["num_experts_per_tok"], cfg["n_routed_experts"], n, False)
    c6_bytes = dense + table["lm_head_fp8"] + table["target_expert_pool"] * target_union + n * lookups
    decode["c6_fp8_independent"] = {"bytes": c6_bytes, "tokens_s": n * bandwidth / c6_bytes}
    # k=5 DSpark uses one parallel 3-layer draft, one backbone verify and five sequential Markov projections.
    k = cfg["dspark_block_size"]
    spec = {}
    for streams, identical, label in [(1, False, "independent_routes"), (1, True, "same_routes"),
                                       (6, False, "c6_independent_routes")]:
        target_rows, draft_rows = streams * (k + 1), streams * k
        target_union = union_fraction(cfg["num_experts_per_tok"], cfg["n_routed_experts"], target_rows, identical)
        draft_union = union_fraction(cfg["dspark_num_experts_per_tok"], cfg["dspark_n_routed_experts"],
                                     draft_rows, identical)
        verify = dense + table["lm_head_fp8"] + table["target_expert_pool"] * target_union + target_rows * lookups
        # Markov projection is replicated on each rank; its matrix can serve all streams at one position.
        draft = (table["draft_dense"] + table["lm_head_fp8"] + table["draft_expert_pool"] * draft_union
                 + 2 * k * table["markov_fp8"] + draft_rows * (table["token_row"] + cfg["dspark_markov_rank"] * 2))
        step = verify + draft
        spec[label] = {
            "verify_bytes": verify, "draft_bytes": draft, "step_ms": step / bandwidth * 1e3,
            "tokens_s_at_fraction_accepted": {
                p: streams * (1 + k * p) * bandwidth / step for p in (0, .483, .806, 1)}}
    prefill = []
    for length in [2048, 40000, 524288]:
        for trim in (False, True):
            flops = prefill_flops(cfg, matrix_params, kv_params, length, trim)
            prefill.append({"length": length, "tail_replay": trim, "gflops_per_token": flops / 1e9,
                            "half_bf16_tok_s": 0.5 * rates[("bf16", 8192)] * 1e12 / flops,
                            "half_fp8_tok_s": 0.5 * rates[("fp8", 8192)] * 1e12 / flops})
    return {"tensor_bytes": dict(table), "target_active_routed_bytes": active, "target_dense_bytes": dense,
            "target_linear_params": sum(matrix_params.values()), "first20_linear_params":
            sum(v for l, v in matrix_params.items() if l < 20), "no_spec": decode, "dspark": spec,
            "prefill": prefill}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--probe", type=Path, default=Path(__file__).resolve().parents[1] / "results/roofline-hardware.json")
    args = parser.parse_args()
    print(json.dumps(calculate(args.pack, args.probe), indent=2))
