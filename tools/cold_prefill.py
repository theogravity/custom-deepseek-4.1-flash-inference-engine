"""Time the published random-code prefill workload and check native execution counters."""
import argparse
import hashlib
import json
import random
import time

from urllib.request import Request, urlopen

from six_streams import MODEL, api_keys, query


def probe(base, key, word_count, seed, timing_only=False):
    rng = random.Random(seed)
    words = [f"{rng.choice(('alpha', 'beta', 'gamma', 'delta', 'omega', 'kappa'))}{rng.randint(0, 99999)}"
             for _ in range(word_count)]
    prompt = "Here is a list of codes. Reply with only the FIRST code, nothing else.\n" + " ".join(words)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    before = query(base, "/metrics", key)
    if before["logical_active"]:
        raise RuntimeError("run the probe on an idle engine")

    first = None
    text, reasoning = [], []
    usage, finish, done = {}, None, False
    request = Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                      {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(request, timeout=21600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice["delta"]
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if first is None and (content or thought):
                    first = time.perf_counter() - start
                text.append(content)
                reasoning.append(thought)
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    deadline = time.monotonic() + 20
    after = query(base, "/metrics", key)
    while after["logical_active"] and time.monotonic() < deadline:
        time.sleep(0.03)
        after = query(base, "/metrics", key)
    reused = after["prefix_reused_tokens"] - before["prefix_reused_tokens"]
    started = after["requests_started"] - before["requests_started"]
    answer = "".join(text).strip()
    valid = (done and first is not None and finish == "stop" and usage.get("completion_tokens", 0) > 0 and
             started == 1 and after["logical_active"] == reused == 0 and
             all(after[k] == before[k] for k in ("multi_sequence_steps", "rotations", "resumes",
                                                 "suspend_failures", "resume_failures")))
    if not timing_only:
        valid = valid and answer == words[0]
    tokens = usage["prompt_tokens"]
    row = dict(seed=seed, words=word_count, prompt_tokens=tokens, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
               output=usage["completion_tokens"], finish=finish, ttft_s=first, total_s=elapsed,
               input_tok_s=tokens / first if first is not None else None,
               reasoning_sha256=hashlib.sha256("".join(reasoning).strip().encode()).hexdigest(),
               reused=reused, passed=valid, terminal=done, requests_started=started, logical_active=after["logical_active"])
    if timing_only:
        row.update(answer_graded=False, content_sha256=hashlib.sha256("".join(text).encode()).hexdigest())
    else:
        row.update(answer=answer, needle=words[0])
    print(json.dumps(row), flush=True)
    if not valid:
        raise RuntimeError("cold prefill check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--timing-only", action="store_true",
                        help="check completion and isolated cold execution without grading the answer")
    parser.add_argument("cases", nargs="+", metavar="WORDS,SEED")
    args = parser.parse_args()
    key, _ = api_keys(False)
    for case in args.cases:
        words, seed = map(int, case.split(","))
        if words < 1:
            parser.error("word counts must be positive")
        probe(args.base, key, words, seed, args.timing_only)


if __name__ == "__main__":
    main()
