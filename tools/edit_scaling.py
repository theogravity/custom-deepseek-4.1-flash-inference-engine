#!/usr/bin/env python3
"""Repeat the published 25.6k code edits and record per-stream timing and output identity."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from threading import Barrier
from time import monotonic
from urllib.request import Request, urlopen


def prompt(start):
    functions = [
        f"def func_{i}(x, y):\n    result = x * {i} + y\n    if result > {i * 3}:\n"
        f"        return result - {i}\n    return result\n"
        for i in range(700)
    ]
    return ("Here is a Python module:\n\n" + "\n".join(functions)
            + f"\n\nRewrite func_{start} through func_{start + 40} with type hints and a one-line docstring each. Output only code.")


def stream(url, key, start, text, barrier):
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 1500, "temperature": 0, "messages": [{"role": "user", "content": text}]}
    req = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    usage = finish = first = None
    done = False
    chunks = {"reasoning": [], "content": []}
    barrier.wait(timeout=30)
    began = monotonic()
    with urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(f"inference stream error at function {start}: {event['error']!r}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice["delta"]
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                answer = delta.get("content")
                if first is None and (answer or thought):
                    first = monotonic()
                if answer:
                    chunks["content"].append(answer)
                if thought:
                    chunks["reasoning"].append(thought)
                finish = choice.get("finish_reason") or finish
    end = monotonic()
    if first is None or not done or finish != "length" or not usage or usage["completion_tokens"] != 1500:
        raise ValueError(f"incomplete edit at function {start}, finish={finish!r}, usage={usage!r}")
    result = dict(start_function=start, prompt_sha256=sha256(text.encode()).hexdigest(),
                  usage=usage, first=first, end=end, began=began)
    for k, parts in chunks.items():
        encoded = "".join(parts).encode()
        result[k + "_sha256"] = sha256(encoded).hexdigest()
        result[k + "_bytes"] = len(encoded)
    result["first_function_present"] = f"def func_{start}(".encode() in "".join(chunks["content"]).encode()
    if usage["prompt_tokens"] != 25625 or not result["first_function_present"]:
        raise ValueError(f"mismatched prompt length or missing requested function {start}")
    return result


def wave(url, key, clients):
    starts = [50 + 60*i for i in range(clients)]
    barrier = Barrier(clients)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(stream, url, key, i, prompt(i), barrier) for i in starts]
        rows = [f.result() for f in futures]
    tokens = sum(r["usage"]["completion_tokens"] for r in rows)
    earliest = min(r["began"] for r in rows)
    latest = max(r["end"] for r in rows)
    durations = [r["end"] - r["first"] for r in rows]
    shared = latest - min(r["first"] for r in rows)
    overlap = min(r["end"] for r in rows) - max(r["first"] for r in rows)
    if overlap <= 0:
        raise RuntimeError("streams did not overlap")
    for r in rows:
        for k in ["began", "first", "end"]:
            r[k + "_s"] = r.pop(k) - earliest
    return dict(upstream_tok_s=tokens/max(durations), shared_tok_s=(tokens-clients)/shared,
                whole_wave_tok_s=tokens/(latest-earliest), overlap_s=overlap, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--waves", type=int, default=7, help="first wave is warm-up")
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.waves < 1:
        parser.error("--waves must be positive")
    key = os.environ.get("DSV41_API_KEY") or os.environ["API_KEY"]
    for idx in range(args.waves):
        print(json.dumps(dict(wave=idx, warmup=not idx, **wave(args.base.rstrip("/")+"/v1/chat/completions", key, args.clients))), flush=True)
