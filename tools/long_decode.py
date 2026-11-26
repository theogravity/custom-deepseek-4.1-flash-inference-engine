#!/usr/bin/env python3
"""Measure native C1–C6 decode on the frozen independent long agent histories."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
from statistics import median
import threading
from time import monotonic, sleep
from types import SimpleNamespace

from six_streams import PROMPTS, api_keys, chat_ids, check_output, id_digest, load_tokenizer, payload, query, stream


def prepare(frozen, workload, clients, generation, base, key, tokenizer):
    programs = list(PROMPTS.values())
    prompts, proofs = [], []
    sessions = frozen["sessions"]
    if frozen["version"] != 1 or len(sessions) != 6 or len({h["prompt_ids_sha256"] for h in sessions}) != 6:
        raise ValueError("expected six distinct frozen histories")
    for agent, history in enumerate(sessions):
        turns = history["messages"]
        if history["agent"] != agent or [m["role"] for m in turns] != ["user", "assistant"]:
            raise ValueError(f"invalid transcript for agent {agent}")
        reply = turns[1]["content"]
        original = chat_ids(tokenizer, turns[:1])
        if (id_digest(original) != history["prompt_ids_sha256"] or len(original) != frozen["input"]
                or hashlib.sha256(reply.encode()).hexdigest() != history["reply_sha256"]
                or not check_output(reply, history["receipts"], agent)[0]):
            raise ValueError(f"frozen prompt or response mismatch for agent {agent}")
        if agent >= clients:
            continue
        if workload == "code":
            ask = programs[agent % len(programs)]
            if agent >= len(programs):
                ask += " Include explanatory code comments."
            messages = turns + [{"role": "user", "content": ask}]
            receipts = []
        else:
            messages = turns[:1]
            receipts = history["receipts"]
        size = query(base, "/v1/messages/count_tokens", key, payload(messages, generation))["input_tokens"]
        ids = chat_ids(tokenizer, messages) if workload == "code" else original
        if size != len(ids) or not frozen["input"] <= size <= 524288 - generation:
            raise ValueError(f"unexpected prompt size for agent {agent}: {size}")
        prompts.append((messages, receipts, size))
        proofs.append(dict(agent=agent, original_ids_sha256=history["prompt_ids_sha256"],
                           original_reply_sha256=history["reply_sha256"], prompt_ids_sha256=id_digest(ids),
                           input_tokens=size))
    return prompts, proofs


def sample_resident_decode(futures, base, key, clients, started):
    """Estimate committed target tokens/s while every request is in decode."""
    pending = set(futures)
    spans, current = [], []
    while pending:
        _, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        if len(pending) != clients:
            break
        begin = monotonic()
        state = query(base, "/metrics", key)
        end = monotonic()
        if (state["requests_started"] == started + clients
                and state["logical_active"] == state["active"] == state["decoding"] == clients
                and state["prefilling"] == 0 and state["swapped"] == 0):
            current.append(((begin + end) / 2, state["processed_tokens"], end - begin))
        elif current:
            spans.append(current)
            current = []
    spans.append(current)
    samples = max((s for s in spans if len(s) >= 2),
                  key=lambda s: s[-1][0] - s[0][0], default=[])
    if not samples:
        return None
    seconds = samples[-1][0] - samples[0][0]
    if any(b[1] < a[1] for a, b in zip(samples, samples[1:])):
        raise RuntimeError("committed decode position moved backwards")
    tokens = samples[-1][1] - samples[0][1]
    if seconds < 1 or not tokens:
        return None
    return dict(committed_tokens=tokens, seconds=seconds, tok_s=tokens / seconds,
                samples=len(samples), max_metrics_rtt_ms=max(s[2] for s in samples) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--workload", choices=("code", "receipt"), default="code")
    parser.add_argument("--clients", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--output", type=int, default=768)
    parser.add_argument("--rounds", type=int, default=6, help="measured waves after one warm-up")
    parser.add_argument("--tokenizer", type=Path, default=Path("/mnt/models/dsv41-flash-exl3/tokenizer.json"))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--sample-steady", action="store_true",
                        help="sample committed positions every 200 ms during all-client decode")
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--allow-output-variation", action="store_true", help="allow varying output hashes for approximate numerical experiments")
    args = parser.parse_args()
    if args.rounds < 1 or args.output < 2 or args.chunk_size < 1:
        parser.error("need positive rounds and chunk size, and at least two output tokens")
    key, _ = api_keys(False)
    frozen = json.loads(args.histories.read_text())
    prompts, proofs = prepare(frozen, args.workload, args.clients, args.output, args.base, key,
                              load_tokenizer(args.tokenizer))
    expected_reuse = sum((size - 1) // args.chunk_size * args.chunk_size for _, _, size in prompts)
    stream_args = SimpleNamespace(base=args.base, output=args.output, litellm=None, save_histories=None)
    identities, rates, wall_rates = None, [], []
    for wave in range(args.rounds + 1):
        gate = threading.Barrier(args.clients)
        before = query(args.base, "/metrics", key)
        if before["logical_active"]:
            raise RuntimeError("another request is active")
        start = monotonic()
        with ThreadPoolExecutor(args.clients) as pool:
            futures = [pool.submit(stream, stream_args, key, msg, receipts, gate, i)
                       for i, (msg, receipts, _) in enumerate(prompts)]
            steady = (sample_resident_decode(futures, args.base, key, args.clients, before["requests_started"])
                      if args.sample_steady else None)
            rows = [future.result() for future in futures]
        after = query(args.base, "/metrics", key)
        deadline = monotonic() + 3
        while after["logical_active"] and monotonic() < deadline:
            sleep(0.01)
            after = query(args.base, "/metrics", key)
        hashes = [row["sha256"] for row in rows]
        valid = all(row["exact"] and row["prompt"] == spec[2] and row["output"] == args.output
                    and row["finish"] == "length" for row, spec in zip(rows, prompts, strict=True))
        delta = {k: after[k] - before[k] for k in ("requests_started", "prefix_reused_tokens",
                 "multi_target_passes", "six_target_passes", "rotations")}
        for k in ("cpu_expert_jobs", "cpu_expert_rows"):
            if k in after:
                delta[k] = after[k] - before[k]
        overlap = min(r["last"] for r in rows) - max(r["first"] for r in rows)
        if (not valid or after["logical_active"] or delta["requests_started"] != args.clients
                or len({r["stream_id"] for r in rows}) != args.clients or delta["rotations"]
                or (not args.allow_output_variation and identities is not None and hashes != identities) or overlap <= 0
                or (wave and delta["prefix_reused_tokens"] != expected_reuse)
                or (args.clients > 1 and delta["multi_target_passes"] <= 0)
                or (args.clients == 6 and delta["six_target_passes"] <= 0)):
            raise RuntimeError(f"invalid wave {wave}: {valid=}, {overlap=}, {delta=}, {hashes=}")
        identities = hashes
        first_spread = max(r["first"] for r in rows) - min(r["first"] for r in rows)
        span = max(r["last"] for r in rows) - min(r["first"] for r in rows)
        elapsed = max(r["end"] for r in rows) - start
        rate = sum(r["output"]-1 for r in rows) / span
        wall_rate = args.clients * args.output / elapsed
        if wave:
            rates.append(rate)
            wall_rates.append(wall_rate)
        for row in rows:
            for k in ("first", "last", "end"):
                row[k+"_s"] = row.pop(k) - start
        print(json.dumps(dict(wave=wave, warmup=not wave, aggregate_decode_tok_s=rate,
              wall_tok_s=wall_rate, seconds=elapsed, resident_decode=steady, first_response_spread_s=first_spread, decode_span_s=span, overlap_s=overlap, delta=delta, rows=rows)), flush=True)
    print(json.dumps(dict(clients=args.clients, workload=args.workload, output=args.output,
          seed=frozen["seed"], prompts=proofs, expected_reused_tokens=expected_reuse, chunk_size=args.chunk_size,
          allow_output_variation=args.allow_output_variation, measured_waves=len(rates), median_decode_tok_s=median(rates), median_wall_tok_s=median(wall_rates))))


if __name__ == "__main__":
    main()
