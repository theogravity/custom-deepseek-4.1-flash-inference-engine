#!/usr/bin/env python3
"""Exercise sparse attention and repeated-prompt reuse through the existing LiteLLM proxy."""
import argparse
import json
import secrets
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from clients import launch


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--model-id", default="dsv41-flash-exl3")
    args.add_argument("--proxy-port", type=int, default=4000)
    args.add_argument("--timeout", type=int, default=1200)
    args.add_argument("--records", type=int, default=160)
    args = args.parse_args()
    if args.records < 2:
        raise SystemExit("at least two records required")
    with tempfile.TemporaryDirectory() as cache:
        env, _ = launch("claude", args, Path(cache))
        model = env["ANTHROPIC_DEFAULT_SONNET_MODEL"]
        key = env["ANTHROPIC_API_KEY"]
        base = env["ANTHROPIC_BASE_URL"]
        entries = [secrets.token_hex(6) for _ in range(args.records)]
        index = secrets.randbelow(args.records // 2)
        prompt = "\n".join(f"Record {n:04d}. The audit receipt is {code}." for n, code in enumerate(entries))
        prompt += f"\nReturn only the audit receipt for record {index:04d}."
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 24, "temperature": 0, "thinking": {"type": "disabled"}}
        records = []
        for i in range(2):
            # A distinct suffix forces LiteLLM to call the engine on both requests.
            suffix = " Output the receipt on a single line." if i else ""
            payload["messages"][0]["content"] = prompt + suffix
            start = time.monotonic()
            request = Request(base + "/v1/messages", data=json.dumps(payload).encode(),
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"})
            with urlopen(request, timeout=args.timeout) as r:
                result = json.load(r)
            answer = "".join(b["text"] for b in result["content"] if b["type"] == "text").strip()
            records.append({"repeat": bool(i), "records": args.records, "target_record": index,
                            "seconds": round(time.monotonic()-start, 3),
                            "usage": result["usage"], "exact_retrieval": answer == entries[index]})
        print(json.dumps(records, indent=2))
        return 0 if all(r["exact_retrieval"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
