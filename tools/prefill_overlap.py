"""Measure an arriving repository review and the coding stream already in flight."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time
from urllib.request import Request, urlopen

from six_streams import MODEL, api_keys, query


def stream(base, key, prompt, limit, ready=None):
    payload = dict(model=MODEL, temperature=0, stream=True, stream_options={"include_usage": True},
                   max_tokens=limit, messages=[{"role": "user", "content": prompt}])
    req = Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(payload).encode(),
                  {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    events = []
    usage, finish, done = None, None, False
    with urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                done = True
                break
            message = json.loads(body)
            if "error" in message:
                raise RuntimeError("server reported an inference error")
            usage = message.get("usage") or usage
            for choice in message.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content")):
                    events.append(time.monotonic())
                    if ready and len(events) == 8:
                        ready.set()
    if not (events and usage and done and finish in ("length", "stop")):
        raise RuntimeError("incomplete coding stream")
    return started, events, usage


def run(base, key):
    rng = random.Random(43718)
    code = "\n".join(f"def rule_{i}(x): return (x * {rng.randrange(13,900)} + {rng.randrange(1300,33000)}) % 65519"
                     for i in range(3000))
    review = "Review this Python module and implement a deduplicated version with identical outputs. Explain your choice briefly.\n" + code
    coding = ("Implement a small Rust JSON lexer from scratch with zero allocations. Show every token variant, handling for "
              "escape sequences and numbers, and 25 tests. Provide the complete implementation and tests in one response.")
    before = query(base, "/metrics", key)
    if before["logical_active"]:
        raise RuntimeError("run the probe on an idle engine")
    ready = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(stream, base, key, coding, 1500, ready)
        if not ready.wait(120):
            raise RuntimeError("first coding client did not emit eight events")
        arrival = time.monotonic()
        second = pool.submit(stream, base, key, review, 64)
        a, ta, ua = first.result()
        b, tb, ub = second.result()
    after = query(base, "/metrics", key)
    gaps = [end-start for start, end in zip(ta, ta[1:]) if start < tb[0] and end >= arrival]
    if not gaps or ta[-1] <= arrival or after["prefix_reused_tokens"] != before["prefix_reused_tokens"]:
        raise RuntimeError("expected overlapping streams with zero reused prefix tokens")
    return dict(first_client=ua, second_client=ub, first_client_stream_events=len(ta), second_ttft_s=tb[0]-b,
                peer_prefill_overlap=True, overlap_gap_observed=True, max_peer_interevent_s=max(gaps),
                first_client_e2e_s=ta[-1]-a, prefix_reused_tokens=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key, _ = api_keys(None)
    with args.out.open("x") as output:
        json.dump(run(args.base, key), output, indent=2)
        output.write("\n")
