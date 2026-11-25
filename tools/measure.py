"""Warm code-edit workload for comparing two builds serving the same checkpoint."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

MODEL = "DeepSeek V4.1 Flash"


def credential():
    key = os.environ.get("DSV41_API_KEY") or os.environ.get("LITELLM_MASTER_KEY")
    if key:
        return key
    for line in (Path.home() / "llm/models/.env").read_text().splitlines():
        if line.startswith("LITELLM_MASTER_KEY="):
            return line.partition("=")[2].strip().strip("\"'")
    raise ValueError("Set DSV41_API_KEY")


def request(prefix, index, limit):
    prompt = prefix + f"\nRewrite func_{index} through func_{index + 22} with type hints and one-line docstrings. Output only Python code."
    body = dict(model=MODEL, temperature=0, stream=True,
                stream_options={"include_usage": True}, max_tokens=limit,
                chat_template_kwargs={"thinking": False},
                messages=[dict(role="user", content=prompt)])
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + credential()})
    start = time.monotonic()
    first = None
    content = []
    usage = None
    done = False
    finish = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            if raw.strip() == b"data: [DONE]":
                done = True
                break
            chunk = json.loads(raw[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {}).get("content")
                if delta:
                    first = first or time.monotonic()
                    content.append(delta)
    duration = time.monotonic() - start
    answer = "".join(content)
    if not (done and usage and first and f"def func_{index}(" in answer):
        raise RuntimeError("Incomplete stream or missing requested function")
    tokens = usage["completion_tokens"]
    if finish != "length" or tokens != limit:
        raise RuntimeError(f"Expected {limit} output tokens at length stop, got {tokens} ({finish})")
    return dict(prompt=usage["prompt_tokens"], tokens=tokens, seconds=duration,
                ttft_s=first-start, output_sha256=hashlib.sha256(answer.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    prefix = "Here is a Python module:\n\n" + "\n".join(
        f"def func_{i}(x, y):\n    result = x * {i} + y\n    if result > {i * 3}:\n        return result - {i}\n    return result\n"
        for i in range(700))
    for n in (1, 6):
        rates = []
        with concurrent.futures.ThreadPoolExecutor(n) as pool:
            for rep in range(13):
                start = time.monotonic()
                rows = list(pool.map(lambda k: request(prefix, 50 + 60*k, 512), range(n)))
                wall = time.monotonic() - start
                rate = sum(r["tokens"] for r in rows) / wall
                if rep >= 7:
                    rates.append(rate)
                with args.ledger.open("a") as f:
                    print(json.dumps(dict(label=args.label, streams=n, warmup=rep<7, rep=rep,
                                          wall_s=wall, aggregate_tps=rate, rows=rows)), file=f)
        print(n, "streams, median end-to-end output tok/s", round(statistics.median(rates), 1), flush=True)


if __name__ == "__main__":
    main()
