#!/usr/bin/env python3
"""Replay seven different follow-ups to the frozen 404k synthetic history."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median
from urllib.request import Request, urlopen

from six_streams import MODEL, api_keys, query

CASES = Path(__file__).resolve().parents[1] / "benchmarks/agent-followups/requests.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_case(base, key, history, case, prime):
    body = {
        "model": MODEL, "reasoning_effort": "none", "temperature": 0, "max_tokens": 16,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": history + [{"role": "user", "content": case["instruction"]}],
    }
    wire = json.dumps(body).encode()
    if digest(wire) != case["wire"]:
        raise ValueError("The supplied history differs from the published fixture")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    count = Request(base + "/v1/messages/count_tokens", data=wire, headers=headers)
    chat = Request(base + "/v1/chat/completions", data=wire, headers=headers)
    before = query(base, "/metrics", key)
    if before["logical_active"]:
        raise ValueError("The engine already has a request in progress")
    start = time.monotonic()
    with urlopen(count, timeout=300) as response:
        tokens = json.load(response)["input_tokens"]
    count_s = time.monotonic() - start
    if tokens != case["tokens"]:
        raise ValueError("The token count differs from the independent tokenizer reference")

    first = usage = finish = None
    pieces = []
    done = False
    start = time.monotonic()
    with urlopen(chat, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError("The engine reported a streaming error")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice["delta"]
                if delta.get("reasoning_content") or delta.get("tool_calls"):
                    raise ValueError("Expected a text-only fixture response")
                if text := delta.get("content"):
                    if first is None:
                        first = time.monotonic() - start
                    pieces.append(text)
                finish = choice.get("finish_reason") or finish
    if not (done and first is not None and finish == "length" and usage
            and usage["prompt_tokens"] == tokens and usage["completion_tokens"] == 16):
        raise ValueError("Incomplete fixture response")
    output_digest = digest(json.dumps(["", "".join(pieces), []]).encode())
    if output_digest != case["reply"]:
        raise ValueError("The generated answer differs from the reference")
    after = query(base, "/metrics", key)
    deadline = time.monotonic() + 3
    while after["logical_active"] and time.monotonic() < deadline:
        time.sleep(0.05)
        after = query(base, "/metrics", key)
    if after["logical_active"]:
        raise ValueError("The engine did not drain")
    if after["requests_started"] != before["requests_started"] + 1 or after["rotations"] != before["rotations"]:
        raise ValueError("Concurrent requests or KV rotations affected the sample")
    reused = after["prefix_reused_tokens"] - before["prefix_reused_tokens"]
    if not prime and "warm_prefix_tokens" in case and reused != case["warm_prefix_tokens"]:
        raise ValueError(f"Restored {reused} tokens on a timed request. Run --prime before measuring.")
    return {"input_tokens": tokens, "count_s": count_s, "ttft_s": first,
            "prefix_reused_tokens": reused,
            "output_sha256": output_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--prime", action="store_true", help="populate KV entries without reporting warm medians")
    args = parser.parse_args()
    histories = json.loads(args.histories.read_text())
    key, _ = api_keys(None)
    cases = json.loads(CASES.read_text())
    rows = [run_case(args.base.rstrip("/"), key, histories["sessions"][0]["messages"], case, args.prime)
            for case in cases]
    report = {"requests": rows}
    if not args.prime:
        report.update(warm_median_count_s=median(v["count_s"] for v in rows[1:]),
                      warm_median_ttft_s=median(v["ttft_s"] for v in rows[1:]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
