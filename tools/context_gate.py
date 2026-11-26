#!/usr/bin/env python3
"""Check three independently generated receipts near half and full deployed context."""

import argparse
import json
import os
import random
import secrets
import time
from urllib.request import Request, urlopen


def post(base, endpoint, key, payload):
    request = Request(
        base.rstrip("/") + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urlopen(request, timeout=7200) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="DeepSeek V4.1 Flash")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    key = os.getenv("DSV41_API_KEY") or os.environ["LITELLM_MASTER_KEY"]
    seed = args.seed if args.seed is not None else secrets.randbits(48)
    rng = random.Random(seed)
    words = [
        rng.choice(("alpha", "beta", "gamma", "delta", "omega")) + str(rng.randrange(100000))
        for _ in range(174000)
    ]
    positions = {"A": 2600, "B": 44600, "C": 83500, "D": 166000}
    receipts = {}
    for name, position in positions.items():
        receipts[name] = f"R{rng.getrandbits(48):012x}"
        words[position] = f"\nVAULT {name} receipt = {receipts[name]}\n"
    passed = True
    for label, count, names, low, high in [
        ("half", 89000, "ABC", 255000, 273000),
        ("full", len(words), "ACD", 510000, 524200),
    ]:
        for _ in range(4):
            text = f"Vault register {seed}.\n" + " ".join(words[:count])
            text += "\nReturn only the three receipt strings for VAULT " + ", ".join(names) + ", in that order, separated by spaces."
            body = {
                "model": args.model, "messages": [{"role": "user", "content": text}],
                "temperature": 0, "max_tokens": 128, "reasoning_effort": "none",
            }
            tokens = post(args.base, "/v1/messages/count_tokens", key, body)["input_tokens"]
            if low <= tokens <= high:
                missing = [name for name in names if positions[name] >= count]
                if missing:
                    raise ValueError(f"{label} token sizing removed vaults {missing}")
                break
            count = min(len(words), round(count * ((low+high)/2) / tokens))
        else:
            raise ValueError(f"could not size the {label} request ({tokens} tokens)")
        print(json.dumps({"starting": label, "prompt_tokens": tokens, "seed": seed}), flush=True)
        start = time.monotonic()
        response = post(args.base, "/v1/chat/completions", key, body)
        choice = response["choices"][0]
        received = choice["message"]["content"].split()
        expected = [receipts[n] for n in names]
        ok = choice["finish_reason"] == "stop" and received == expected and response["usage"]["prompt_tokens"] == tokens
        print(json.dumps({
            "context": label, "prompt_tokens": tokens, "output_tokens": response["usage"]["completion_tokens"],
            "seed": seed, "wall_s": round(time.monotonic()-start, 3), "exact": ok,
            "expected": expected, "received": received, "finish": choice["finish_reason"],
        }), flush=True)
        passed &= ok
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
