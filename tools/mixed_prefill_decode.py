#!/usr/bin/env python3
"""Five code-edit decoders while a sixth client sends a cold ~19k-token prompt every 10 s.

Each wave prints the decoders' aggregate rate and the cold prompts' time to first generated token.
"""
import argparse
import json
import os
import random
import threading
import time
from urllib.request import Request, urlopen

import edit_scaling


def cold(url, key, seed, words):
    rng = random.Random(seed)
    text = " ".join(f"{rng.choice(('alpha', 'beta', 'gamma', 'delta'))}{rng.randint(0, 99999)}" for _ in range(words))
    body = {"model": "DeepSeek V4.1 Flash", "stream": True, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with the FIRST code only.\n" + text}]}
    req = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    started, first = time.monotonic(), None
    with urlopen(req, timeout=600) as response:
        for line in response:
            if first is not None or not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            choices = json.loads(line[5:]).get("choices") or [{}]
            delta = choices[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                first = time.monotonic() - started
    return first


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True, help="use a fresh seed so prompts miss the prefix cache")
    parser.add_argument("--words", type=int, default=6000)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    key = os.environ.get("DSV41_API_KEY") or os.environ["API_KEY"]
    url = args.base.rstrip("/") + "/v1/chat/completions"
    for w in range(args.waves):
        stop, ttfts = threading.Event(), []

        def side():
            i = 0
            while not stop.is_set():
                ttfts.append(cold(url, key, args.seed + w * 100 + i, args.words))
                i += 1
                stop.wait(args.interval)

        thread = threading.Thread(target=side)
        thread.start()
        result = edit_scaling.wave(url, key, 5)
        stop.set()
        thread.join()
        print(json.dumps(dict(wave=w, upstream_tok_s=result["upstream_tok_s"], whole=result["whole_wave_tok_s"],
                              cold_ttft=[round(x, 2) for x in ttfts if x])), flush=True)
