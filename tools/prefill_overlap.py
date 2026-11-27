"""Measure an arriving repository review and the coding stream already in flight."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

from six_streams import MODEL, api_keys, query


@dataclass
class Stream:
    started: float
    events: list[float]
    done: float
    usage: dict
    finish: str | None = None


def stream(base, key, prompt, limit, ready=None, timeout=120):
    payload = dict(model=MODEL, temperature=0, stream=True, stream_options={"include_usage": True},
                   max_tokens=limit, messages=[{"role": "user", "content": prompt}])
    req = Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(payload).encode(),
                  {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    events = []
    usage, finish, done = None, None, None
    with urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                done = time.monotonic()
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
    if not (events and usage and done is not None and finish in ("length", "stop")):
        raise RuntimeError("incomplete coding stream")
    return Stream(started, events, done, usage, finish)


def idle(base, key):
    deadline = time.monotonic() + 20
    while True:
        state = query(base, "/metrics", key)
        if all(state[k] == 0 for k in ("logical_active", "resident", "waiting", "swapped")):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError("engine did not drain")
        time.sleep(0.05)


def summarize(first, second, before, after):
    counters = {k: after[k] - before[k] for k in (
        "requests_started", "prefix_reused_tokens", "rotations", "resumes", "suspend_failures", "resume_failures")}
    if counters["requests_started"] != 2 or any(v for k, v in counters.items() if k != "requests_started"):
        raise RuntimeError(f"unexpected workload counters: {counters}")
    for result, inputs, outputs in ((first, 70, 1500), (second, 68049, 64)):
        expected = (inputs, outputs, inputs + outputs)
        actual = tuple(result.usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if actual != expected:
            raise RuntimeError(f"expected workload token counts {expected}, got {actual}")

    start, end = second.started, second.events[0]
    peer = first.events
    if not (len(peer) >= 8 and peer[7] <= start < end < peer[-1]):
        raise RuntimeError("the first stream must span the arriving request through its first output")
    inside = sum(start < event < end for event in peer)
    gaps = [(left, right) for left, right in zip(peer, peer[1:]) if left < end and right > start]
    left, right = max(gaps, key=lambda pair: pair[1] - pair[0])
    return dict(first_client=first.usage, second_client=second.usage,
                first_client_stream_events=len(peer), second_ttft_s=end-start,
                peer_events_during_second_ttft=inside, max_peer_interevent_s=right-left,
                # Endpoints are relative to the arriving HTTP request starting.
                max_gap_start_from_second_start_s=left-start, max_gap_end_from_second_start_s=right-start,
                first_client_to_last_event_s=peer[-1]-first.started,
                first_client_to_done_s=first.done-first.started, counters=counters)


def run(base, key):
    rng = random.Random(43718)
    code = "\n".join(f"def rule_{i}(x): return (x * {rng.randrange(13,900)} + {rng.randrange(1300,33000)}) % 65519"
                     for i in range(3000))
    review = "Review this Python module and implement a deduplicated version with identical outputs. Explain your choice briefly.\n" + code
    coding = ("Implement a small Rust JSON lexer from scratch with zero allocations. Show every token variant, handling for "
              "escape sequences and numbers, and 25 tests. Provide the complete implementation and tests in one response.")
    before = idle(base, key)
    ready = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(stream, base, key, coding, 1500, ready)
        if not ready.wait(120):
            raise RuntimeError("first coding client did not emit eight events")
        second = pool.submit(stream, base, key, review, 64)
        a, b = first.result(), second.result()
    return summarize(a, b, before, idle(base, key))


def measure(base, output):
    if output.exists():
        raise FileExistsError(output)
    # A socket read timeout resets on each SSE heartbeat. Bound the entire pair,
    # including executor shutdown and metrics calls, in a separate process.
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--base", base],
                           stdout=subprocess.PIPE, check=True, timeout=600)
    receipt = json.loads(child.stdout)
    with output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        key, _ = api_keys(None)
        print(json.dumps(run(args.base, key)))
    else:
        measure(args.base, args.out)
