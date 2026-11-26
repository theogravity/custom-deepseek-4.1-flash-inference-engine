#!/usr/bin/env python3
"""Qualify generation and error handling at the deployed context boundary.

The default window is 524288. Add --litellm to check the OpenAI, Anthropic and
Codex Responses routes. Use --histories with the staged 404k fixture to reuse
the 524160-token native prefix. Each JSONL row includes the HTTP/SSE outcome
and native backend counter changes. Run on an otherwise idle service.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from c1_proxy import Completion, frames, require
from six_streams import (LITELLM_MODEL, MODEL, api_keys, continued_documents,
                         load_tokenizer, payload, query)


def sized_messages(args, key):
    def count(messages, codex):
        body = payload(messages, 1)
        if codex:
            # model-cli's Codex Responses request uses the Anthropic model's
            # default reasoning. Count its native template, including history.
            del body["reasoning_effort"]
        return query(args.base, "/v1/messages/count_tokens", key, body)["input_tokens"]

    base = [{"role": "user", "content": "Continue writing the sequence 1, 2, 3,"}]
    if args.histories:
        frozen = json.loads(args.histories.read_text())
        spec = SimpleNamespace(base=args.base, input=args.window - 128, output=128, seed=frozen["seed"])
        docs, _ = continued_documents(spec, key, load_tokenizer(args.tokenizer), frozen, agents=[0])
        base = docs[0][0]
    result = {}
    for codex in ([False, True] if args.litellm else [False]):
        base_count = count(base, codex)
        targets = [args.window + 1] if codex else [args.window - 1, args.window, args.window + 1]
        for target in targets:
            pad = target - base_count
            for _ in range(4):
                require(pad >= 0, "context is smaller than the fixture")
                messages = base[:-1] + [dict(base[-1], content=base[-1]["content"] + " x" * pad)]
                actual = count(messages, codex)
                if actual == target:
                    break
                pad += target - actual
            require(actual == target, f"failed to size prompt to {target} (got {actual})")
            result[codex, target] = messages
    return result


def call(base, key, wire, proxy, messages, budget, stream, timeout):
    headers = {"Content-Type": "application/json"}
    body = {"model": LITELLM_MODEL if proxy and wire == "openai" else MODEL, "stream": stream}
    if wire == "responses":
        path = "/v1/responses"
        body.update(input=messages, store=False, max_output_tokens=budget)
    else:
        path = "/v1/messages" if wire == "anthropic" else "/v1/chat/completions"
        body.update(messages=messages, temperature=0)
        if budget is not None:
            body["max_tokens"] = budget
    if wire == "anthropic":
        body["thinking"] = {"type": "disabled"}
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = "Bearer " + key
        if wire == "openai":
            body["reasoning_effort"] = "none"
            if stream:
                body["stream_options"] = {"include_usage": True}
    if proxy:
        headers["Cache-Control"] = "no-cache, no-store"
        body["cache"] = {"no-cache": True, "no-store": True}
    req = Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers)
    try:
        response = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        mime = response.headers.get_content_type()
        row = dict(http_status=response.status, content_type=mime,
                   limit_headers={k.lower(): v for k, v in response.headers.items()
                                  if k.lower().startswith("x-ds4-")})
        if mime == "text/event-stream":
            completion = Completion()
            for frame in frames(response):
                event = {} if frame == b"[DONE]" else json.loads(frame)
                if event.get("error") or event.get("type") in ("error", "response.failed"):
                    row["sse_event"] = event.get("type", "error")
                    row["error"] = event.get("error") or (event.get("response") or {}).get("error") or event
                    break
                if wire != "responses":
                    parser = completion.anthropic if wire == "anthropic" else completion.openai
                    parser(frame)
            if "error" not in row:
                row.update(ended=completion.ended, input_tokens=completion.prompt,
                           output_tokens=completion.output, finish=completion.finish)
        else:
            data = json.load(response)
            if "error" in data:
                row.update(error=data["error"], envelope=data.get("type"))
            elif wire == "anthropic":
                row.update(ended=True, finish=data["stop_reason"], **data["usage"])
            elif wire == "openai":
                row.update(ended=True, finish=data["choices"][0]["finish_reason"],
                           input_tokens=data["usage"]["prompt_tokens"],
                           output_tokens=data["usage"]["completion_tokens"])
        return row


def snapshot(args, key, wait_for_finish=False):
    m = query(args.base, "/metrics", key)
    deadline = time.monotonic() + 5
    while wait_for_finish and m["logical_active"] and time.monotonic() < deadline:
        time.sleep(0.01)
        m = query(args.base, "/metrics", key)
    require(m["logical_active"] == 0, "native service must be idle")
    counts = {k: m[k] for k in ("requests_started", "batch_steps", "prefix_reused_tokens", "rotations", "resumes")}
    counts.update({k: m["swap"][k] for k in ("ram_writes", "disk_writes", "ram_reads", "disk_reads")})
    return counts


def validate(row, window):
    input_size, requested = row["prompt_tokens"], row["requested_max"]
    invalid = input_size >= window or (requested or 0) > window
    proxy, wire = row["route"] == "litellm", row["wire"]
    delta = row["backend_delta"]
    require(delta["requests_started"] == int(not invalid), f"unexpected native admission: {delta}")
    require(all(delta[k] == 0 for k in ("rotations", "resumes", "ram_writes", "disk_writes", "ram_reads", "disk_reads")),
            "unexpected live swap activity")
    if invalid:
        require(delta["batch_steps"] == delta["prefix_reused_tokens"] == 0, "rejected request ran inference")
        error = row.get("error") or {}
        message = error.get("message", "") if isinstance(error, dict) else str(error)
        require(row["http_status"] == 400 and row["content_type"] == "application/json",
                "expected HTTP 400 JSON before streaming")
        require(f"maximum context length is {window} tokens" in message and MODEL in message,
                "client error omitted the native model or window")
        if input_size >= window:
            require(f"prompt is too long: {input_size} tokens > {window - 1} maximum" in message,
                    "client error omitted exact prompt count")
        else:
            require(f"prompt contains {input_size} tokens" in message and
                    f"requests {requested} output tokens" in message, "incorrect output budget error")
        if wire == "responses":
            require(error.get("code") == "context_length_exceeded",
                    "Codex requires error.code=context_length_exceeded to recognize context overflow")
        elif wire == "openai":
            require(error.get("code") == "context_length_exceeded", "missing OpenAI context_length_exceeded code")
        if not proxy:
            require(error["type"] == "invalid_request_error", "wrong native error type")
            if wire == "openai":
                require(error["code"] == "context_length_exceeded"
                        and error["param"] == ("messages" if input_size >= window else "max_tokens"),
                        "wrong OpenAI error code/param")
            else:
                require(row.get("envelope") == "error", "wrong Anthropic error envelope")
    else:
        require(row["http_status"] == 200 and row.get("ended") and not row.get("error")
                and row["input_tokens"] == input_size and row["output_tokens"] == 1
                and delta["batch_steps"] > 0, "expected exactly one sampled token at the context boundary")
        require(row["finish"] in (("max_tokens", "end_turn") if wire == "anthropic" else ("length", "stop")),
                "invalid terminal reason")
        if not proxy:
            expected = {}
            if requested != 1:
                expected = {"x-ds4-effective-max-tokens": "1",
                            ("x-ds4-default-max-tokens" if requested is None else "x-ds4-requested-max-tokens"):
                                str(4096 if requested is None else requested)}
            require(row["limit_headers"] == expected, "incorrect cap disclosure headers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--litellm", nargs="?", const="http://127.0.0.1:4000", metavar="ROOT")
    parser.add_argument("--histories", type=Path, help="saved six-session 404k transcript JSON")
    parser.add_argument("--tokenizer", type=Path, default=Path("/mnt/models/dsv41-flash-exl3/tokenizer.json"))
    parser.add_argument("--window", type=int, default=524288)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    keys = api_keys(args.litellm)
    before_count = snapshot(args, keys[0])
    prompts = sized_messages(args, keys[0])
    require(snapshot(args, keys[0]) == before_count, "token counting admitted inference")
    print(json.dumps(dict(counted=[{"template":"codex_default" if codex else "thinking_disabled",
                                    "tokens":tokens} for codex,tokens in prompts], admissions=0)), flush=True)
    routes = [(False, "anthropic"), (False, "openai")]
    if args.litellm:
        routes += [(True, "anthropic"), (True, "openai"), (True, "responses")]
    passed = True
    for proxy, wire in routes:
        cases = [(args.window + 1, 1, True)]
        if wire == "responses":
            cases.append((args.window + 1, 1, False))
        else:
            cases += [(args.window, 1, True), (args.window-1, 1, False), (args.window-1, 2, True)]
        if not proxy:
            cases += [(args.window-1, None, False), (args.window-1, args.window+1, True)]
        for size, budget, stream in cases:
            before = snapshot(args, keys[0])
            row = dict(route="litellm" if proxy else "native", wire=wire, prompt_tokens=size,
                       requested_max=budget, stream=stream)
            row.update(call(args.litellm if proxy else args.base, keys[proxy], wire, proxy,
                            prompts[wire == "responses", size], budget, stream, args.timeout))
            after = snapshot(args, keys[0], wait_for_finish=True)
            row["backend_delta"] = {k: after[k] - v for k, v in before.items()}
            try:
                validate(row, args.window)
                row["passed"] = True
            except (KeyError, ValueError) as exc:
                row.update(passed=False, failure=str(exc))
                passed = False
            print(json.dumps(row), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
