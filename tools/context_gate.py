"""Check unique near-capacity prompts and six concurrent compaction-length prompts."""
import concurrent.futures
import json
import random
import secrets
import time
import urllib.request

from measure import MODEL, credential


def check(words, seed):
    rng = random.Random(seed)
    target = [f"R{rng.getrandbits(48):012x}" for _ in range(3)]
    text = [f"{rng.choice(('alpha','beta','gamma','delta','omega'))}{rng.randrange(100000)}"
            for _ in range(words)]
    for pct, key, receipt in zip((3, 50, 96), ("A", "B", "C"), target):
        text[words * pct // 100] = f"\nVAULT {key} receipt = {receipt}\n"
    prompt = (f"Document {seed}. Return the three VAULT receipts A, B and C in that order. "
              "Return only the three receipt strings separated by spaces.\n"
              + " ".join(text) + "\nReturn the three VAULT receipts.")
    body = dict(model=MODEL, messages=[dict(role="user", content=prompt)],
                temperature=0, max_tokens=256, chat_template_kwargs={"thinking": False})
    request = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + credential()})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    answer = result["choices"][0]["message"]["content"]
    return dict(seed=seed, prompt_tokens=result["usage"]["prompt_tokens"],
                finish=result["choices"][0]["finish_reason"],
                exact=answer.split()==target, seconds=round(time.monotonic()-start, 2))


def main():
    passed = True
    for streams, words, minimum in [(1, 87300, 250000), (1, 173500, 515000), (6, 137500, 400000)]:
        seeds = [secrets.randbits(48) for _ in range(streams)]
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(streams) as pool:
            rows = list(pool.map(lambda seed: check(words, seed), seeds))
        print(json.dumps(dict(streams=streams, wall_s=round(time.monotonic()-start, 2), rows=rows)), flush=True)
        passed &= all(minimum <= row["prompt_tokens"] < 524288 and row["exact"] and row["finish"]=="stop" for row in rows)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
