#!/usr/bin/env python3
"""Measure one streaming client on the pack's three codegen prompts or a Go source edit."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = {
    "game": "Write a complete, playable Tetris game in Python using pygame. Output only the code.",
    "parser": "Write a JSON parser in C (objects, arrays, strings with escapes, numbers, true/false/null) with a small test main(). Output only the code.",
    "cache": "Implement a thread-safe LRU cache with TTL expiry in Rust, with unit tests. Output only the code.",
}
NON_TEXT_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "tool_calls",
                   "function_call", "refusal", "audio")


def request(args, prompt, limit, key, *, bypass_proxy_cache=False):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limit, "temperature": args.temperature, "reasoning_effort": "none",
        "stream": True, "stream_options": {"include_usage": True},
    }
    if bypass_proxy_cache:
        payload["cache"] = {"no-cache": True, "no-store": True}
    req = Request(args.base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    start = time.perf_counter()
    first = last = usage = finish = None
    chunks = []
    done = False
    with urlopen(req, timeout=args.timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                if any(delta.get(key) for key in NON_TEXT_FIELDS):
                    raise ValueError("throughput benchmark received a non-text output block")
                content = delta.get("content")
                if content:
                    last = time.perf_counter()
                    first = last if first is None else first
                    chunks.append(content)
    end = time.perf_counter()
    if not done or not usage or first is None or last <= first or finish not in ("length", "stop"):
        raise ValueError(f"incomplete stream, finish reason {finish!r}")
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") or details.get("audio_tokens"):
        raise ValueError("throughput benchmark includes hidden or audio tokens")
    # Requests have no string stops. On a natural stop the target's sampled
    # EOS counts in API usage but has no visible text, so remove that token.
    n = usage["completion_tokens"] - int(finish == "stop")
    if n < 2:
        raise ValueError("fewer than two generated tokens")
    result = "".join(chunks)
    return {
        "prompt_tokens": usage["prompt_tokens"], "visible_tokens": n, "finish": finish,
        "ttft_s": first-start, "wall_s": end-start, "decode_tok_s": (n-1)/(last-first),
        "sha256": hashlib.sha256(result.encode()).hexdigest(),
        "preview": result[:120],
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="DeepSeek V4.1 Flash")
    parser.add_argument("--rounds", type=int, default=3, help="first round warms up, later rounds are measured")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--bypass-proxy-cache", action="store_true",
                        help="disable both LiteLLM response cache reads and writes")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--edit-source", type=Path, help="benchmark the selfupdate.go edit instead of new code")
    parser.add_argument("--save-output", type=Path, help="save each complete output under this directory")
    args = parser.parse_args()
    if args.rounds < 2 or args.tokens < 2:
        parser.error("need at least two rounds and tokens")
    key = os.getenv("DSV41_API_KEY") or os.environ["LITELLM_MASTER_KEY"]
    prompts = PROMPTS
    if args.edit_source:
        prompts = {"edit": 'Update this Go source file so SkipUpdate also recognizes the value "true" case-insensitively. Keep the other behavior, comments, and formatting. Return the complete updated source as a single Go code block.\n\n```go\n' + args.edit_source.read_text() + '\n```'}
    if args.save_output:
        args.save_output.mkdir(parents=True, exist_ok=True)
    rates = []
    for rep in range(args.rounds):
        for name, prompt in prompts.items():
            row, text = request(args, prompt, args.tokens, key, bypass_proxy_cache=args.bypass_proxy_cache)
            print(json.dumps(dict(workload=name, warmup=rep == 0, **row)), flush=True)
            if args.save_output:
                (args.save_output / f"{name}-{rep}.txt").write_text(text)
            if rep:
                rates.append(row["decode_tok_s"])
    print(json.dumps({"samples": len(rates), "median_decode_tok_s": statistics.median(rates)}))


if __name__ == "__main__":
    main()
