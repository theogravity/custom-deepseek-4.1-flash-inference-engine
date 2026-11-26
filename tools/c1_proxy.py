#!/usr/bin/env python3
"""Measure single-client native versus LiteLLM overhead at matched context lengths.

Warm sweep with the workstation's existing prefix cache (OpenAI wire):
  python3 tools/c1_proxy.py > warm-openai.jsonl

Reuse the actual 404k first turn and full-window continuation previously qualified:
  uv run --offline --no-project --with tokenizers==0.22.2 python tools/c1_proxy.py --histories histories.json > warm.jsonl

Same-wire Anthropic comparison:
  python3 tools/c1_proxy.py --wire anthropic > warm-anthropic.jsonl

With BOTH native prefix cache budgets already set to zero:
  python3 tools/c1_proxy.py --mode cold --contexts 4096 16384 65536 --pairs 2 > cold.jsonl

All routes use fresh HTTP connections. Timers exclude JSON encoding, prompt sizing
and metrics reads. TTFT ends at the first visible text event; total ends at receipt
of the terminal SSE frame. Decode rate estimates client-visible delivery rate,
assuming one token in the first visible frame. A fully coalesced stream has
an undefined decode rate. Reported differences are proxy minus direct.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from code_speed import NON_TEXT_FIELDS
from six_streams import (DEFAULT_SEED, LITELLM_MODEL, MODEL, api_keys, chat_ids, check_output,
                         continued_documents, document, fit_text, id_digest, load_tokenizer, payload, query)

CONTEXTS = (256, 4096, 16384, 65536, 131072, 262144, 404000, 524160)
MEASUREMENTS = ("headers_ms", "first_sse_ms", "ttft_ms", "decode_ms", "total_ms",
                "decode_tok_s", "sse_frames", "text_chunks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def workload(args, native_key, size, frozen, tokenizer):
    def count(text):
        return query(args.native, "/v1/messages/count_tokens", native_key,
                     payload(text, args.output))["input_tokens"]

    if frozen and size >= frozen["input"]:
        history = frozen["sessions"][0]
        require(frozen["version"] == 1 and history["agent"] == 0, "invalid saved C1 history")
        if size == frozen["input"]:
            first = history["messages"][:1]
            ids = chat_ids(tokenizer, first)
            require(len(ids) == count(first) == size and id_digest(ids) == history["prompt_ids_sha256"],
                    "frozen first-turn checkpoint differs from native prompt")
            return first, history["receipts"], "frozen_first_turn"
        spec = SimpleNamespace(base=args.native, input=size, output=args.output, seed=frozen["seed"])
        requests, _ = continued_documents(spec, native_key, tokenizer, frozen, agents=[0])
        return *requests[0], "frozen_continuation"
    if size >= 12000:
        # Byte-identical to agent zero from six_streams.py --shared. A previously
        # qualified 524160-token register reuses its full native checkpoint.
        spec = SimpleNamespace(base=args.native, input=size, output=args.output,
                               seed=args.seed, shared=True, save_histories=None)
        text, receipts = document(spec, native_key, 0)
    else:
        rng = random.Random(args.seed)
        receipts = [f"R{rng.getrandbits(48):012x}" for _ in range(2)]
        head = f"Vault register {args.seed}.\nVAULT A receipt = {receipts[0]}\n"
        suffix = (f"\nVAULT E receipt = {receipts[1]}\n"
                  "Print the VAULT A receipt then the VAULT E receipt separated by a space. "
                  "Repeat that exact pair on each line for 100 lines, starting immediately.")
        words = [f"entry{rng.randrange(100000)}" for _ in range(size // 3)]
        text = fit_text(head, words, suffix, len(words), size, count)
    require(count(text) == size, f"native prompt did not size to {size} input tokens")
    return [{"role": "user", "content": text}], receipts, "shared_register" if size >= 12000 else "short_register"


def frames(response):
    parts = []
    for line in response:
        line = line.rstrip(b"\r\n")
        if not line:
            if parts:
                yield b"\n".join(parts)
                parts.clear()
        elif line.startswith(b"data:"):
            parts.append(line[5:].removeprefix(b" "))


class Completion:
    def __init__(self):
        self.identity = None
        self.finish = None
        self.prompt = None
        self.output = None
        self.block = None
        self.started = self.ended = False

    def openai(self, data):
        if data == b"[DONE]":
            self.ended = True
            return ""
        event = json.loads(data)
        require(not event.get("error"), "OpenAI stream error")
        identity = event.get("id")
        require(identity and (self.identity is None or identity == self.identity), "invalid OpenAI completion ID")
        self.identity = identity
        if usage := event.get("usage"):
            details = usage.get("completion_tokens_details") or {}
            require(not details.get("reasoning_tokens") and not details.get("audio_tokens"), "non-text output usage")
            self.prompt, self.output = usage["prompt_tokens"], usage["completion_tokens"]
        choices = event.get("choices", [])
        if not choices:
            return ""
        require(len(choices) == 1 and choices[0].get("index", 0) == 0, "unexpected completion choices")
        choice = choices[0]
        delta = choice.get("delta") or {}
        require(not any(delta.get(field) for field in NON_TEXT_FIELDS), "non-text OpenAI delta")
        self.finish = choice.get("finish_reason") or self.finish
        return delta.get("content") or ""

    def anthropic(self, data):
        event = json.loads(data)
        kind = event["type"]
        if kind == "ping":
            return ""
        if kind == "message_start":
            msg = event["message"]
            require(not self.started and msg.get("id") and not msg.get("content"), "invalid Anthropic message start")
            self.started = True
            self.identity = msg["id"]
            self.prompt = msg["usage"]["input_tokens"]
        else:
            require(self.started, "Anthropic data before message start")
            if kind == "content_block_start":
                block = event["content_block"]
                require(self.block is None and block["type"] == "text", "unexpected Anthropic output block")
                self.block = event["index"]
                return block["text"]
            if kind == "content_block_delta":
                require(self.block == event["index"] and event["delta"]["type"] == "text_delta",
                        "unexpected Anthropic output delta")
                return event["delta"]["text"]
            if kind == "content_block_stop":
                require(self.block == event["index"], "wrong block stop")
                self.block = None
            elif kind == "message_delta":
                self.finish = event["delta"].get("stop_reason") or self.finish
                if "output_tokens" in event["usage"]:
                    self.output = event["usage"]["output_tokens"]
            elif kind == "message_stop":
                require(self.block is None, "unclosed Anthropic content block")
                self.ended = True
            else:
                raise ValueError(f"unexpected Anthropic SSE event {kind!r}")
        return ""


def stream_request(args, wire, route, key, messages, expected, size):
    proxy = route == "proxy"
    body = {"model": LITELLM_MODEL if proxy and wire == "openai" else MODEL,
            "messages": messages, "temperature": 0, "max_tokens": args.output, "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if wire == "openai":
        path = "/v1/chat/completions"
        body.update(reasoning_effort="none", stream_options={"include_usage": True})
        headers["Authorization"] = "Bearer " + key
    else:
        path = "/v1/messages"
        body["thinking"] = {"type": "disabled"}
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    if proxy:
        body["cache"] = {"no-cache": True, "no-store": True}
        headers["Cache-Control"] = "no-cache, no-store"
    url = (args.proxy if proxy else args.native).rstrip("/") + path
    req = Request(url, data=json.dumps(body, separators=(",", ":")).encode(), headers=headers)
    msg = Completion()
    parse = msg.openai if wire == "openai" else msg.anthropic
    chunks = []
    first = last = first_frame = None
    nframes = 0
    start = time.perf_counter()
    with urlopen(req, timeout=args.timeout) as response:
        got_headers = time.perf_counter()
        require(response.headers.get_content_type() == "text/event-stream", f"{route} returned a non-SSE response")
        for frame in frames(response):
            now = time.perf_counter()
            nframes += 1
            if first_frame is None:
                first_frame = now
            text = parse(frame)
            if text:
                require(isinstance(text, str), "stream returned non-string text")
                first = now if first is None else first
                last = now
                chunks.append(text)
            if msg.ended:
                end = now
                break
    require(msg.ended and first is not None, f"incomplete {wire} SSE output")
    require(msg.prompt == size and msg.output == args.output, f"unexpected {wire} token usage {(msg.prompt,msg.output)}")
    require(msg.finish == ("length" if wire == "openai" else "max_tokens"), f"unexpected finish {msg.finish!r}")
    answer = "".join(chunks)
    require(check_output(answer, expected, 0)[0], "incorrect receipt sequence")
    return {"route": route, "model": body["model"], "stream_id": msg.identity,
            "prompt_tokens": msg.prompt, "output_tokens": msg.output, "finish": msg.finish,
            "sha256": hashlib.sha256(answer.encode()).hexdigest(), "sse_frames": nframes,
            "text_chunks": len(chunks), "headers_ms": (got_headers-start)*1000,
            "first_sse_ms": (first_frame-start)*1000, "ttft_ms": (first-start)*1000,
            "decode_ms": (last-first)*1000, "total_ms": (end-start)*1000,
            "decode_tok_s": (msg.output-1)/(last-first) if last > first else None}


def checked_call(args, wire, route, keys, messages, receipts, size, warmup):
    def metrics():
        return query(args.native, "/metrics", keys[0])
    before = metrics()
    require(before["logical_active"] == 0, "another native request is active")
    row = stream_request(args, wire, route, keys[route == "proxy"], messages, receipts, size)
    after = metrics()
    deadline = time.monotonic() + 5
    while after["logical_active"] and time.monotonic() < deadline:
        time.sleep(0.01)
        after = metrics()
    delta = {k: after[k]-before[k] for k in ("requests_started", "prefix_reused_tokens", "batch_steps",
                                             "multi_sequence_steps", "multi_target_passes", "rotations")}
    require(after["logical_active"] == 0 and delta["requests_started"] == 1 and delta["batch_steps"] > 0,
            f"expected exactly one completed native request: {delta}")
    require(not any(delta[k] for k in ("multi_sequence_steps", "multi_target_passes", "rotations")),
            f"concurrent native work detected: {delta}")
    reused = delta["prefix_reused_tokens"]
    require(0 <= reused < size, "invalid native prefix reuse")
    if args.mode == "cold":
        require(reused == 0, "cold pair restored a prefix; disable both native prefix-cache budgets")
    elif not warmup:
        boundary = ((size-1)//args.chunk_size)*args.chunk_size
        require(reused == boundary, f"warm pair restored {reused}, expected {boundary}")
    row["backend"] = delta
    return row


def spread(values):
    usable = [v for v in values if v is not None]
    result = dict(samples=len(usable), missing=len(values)-len(usable))
    if not usable:
        return dict(result, median=None, q1=None, q3=None, min=None, max=None)
    q1, _, q3 = statistics.quantiles(usable, n=4, method="inclusive") if len(usable)>1 else [usable[0]]*3
    return dict(result, median=statistics.median(usable), q1=q1, q3=q3, min=min(usable), max=max(usable))


def run(args, write):
    keys = api_keys(True)
    frozen = json.loads(args.histories.read_text()) if args.histories else None
    tokenizer = load_tokenizer(args.tokenizer) if frozen else None
    for context_index, size in enumerate(args.contexts):
        messages, receipts, source = workload(args, keys[0], size, frozen, tokenizer)
        digest = hashlib.sha256(json.dumps(messages, separators=(",", ":")).encode()).hexdigest()
        identity = None
        for wire_index, wire in enumerate(["openai","anthropic"] if args.wire == "both" else [args.wire]):
            label = {"wire": wire, "mode":args.mode, "input":size, "output":args.output, "source":source,
                     "seed":frozen["seed"] if source.startswith("frozen") else args.seed,
                     "messages_sha256":digest}
            pairs = []
            offset = (context_index + wire_index) % 2
            for rep in range(-1, args.pairs):
                order = ["native", "proxy"] if (rep + offset) % 2 == 0 else ["proxy", "native"]
                pair = {}
                for route in order:
                    row = checked_call(args, wire, route, keys, messages, receipts, size, rep < 0)
                    require(identity is None or row["sha256"] == identity,
                            f"greedy output changed at context {size} ({wire}, {route}, pair {rep})")
                    identity = row["sha256"]
                    pair[route] = row
                    write(dict(label, kind="sample", pair=rep, warmup=rep<0, **row))
                if rep >= 0:
                    delta = {k: pair["proxy"][k] - pair["native"][k]
                             if pair["proxy"][k] is not None and pair["native"][k] is not None else None
                             for k in MEASUREMENTS}
                    pairs.append((pair, delta))
                    write(dict(label, kind="pair", pair=rep, order=order, proxy_minus_native=delta))
            summaries = {route: {k: spread([p[route][k] for p, _ in pairs]) for k in MEASUREMENTS}
                         for route in ["native","proxy"]}
            write(dict(label, kind="summary", pairs=len(pairs), **summaries,
                       proxy_minus_native={k: spread([d[k] for _, d in pairs]) for k in MEASUREMENTS}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", default="http://127.0.0.1:8001", help="native HTTP root")
    ap.add_argument("--proxy", default="http://127.0.0.1:4000", help="LiteLLM HTTP root")
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--output", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=6, help="measured pairs per context and wire (must be even for counterbalancing)")
    ap.add_argument("--mode", choices=("warm","cold"), default="warm")
    ap.add_argument("--wire", choices=("openai","anthropic","both"), default="openai")
    ap.add_argument("--chunk-size", type=int, default=64, help="native prefill chunk for exact warm reuse checks")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--histories", type=Path, help="use agent zero's saved first turn and continuation at or above its prompt length")
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/models/dsv41-flash-exl3/tokenizer.json"),
                    help="checkpoint tokenizer for verifying saved histories (requires Python tokenizers)")
    ap.add_argument("--timeout", type=float, default=21600)
    args = ap.parse_args()
    if args.output < 2 or any(not 128 <= size <= 524288-args.output for size in args.contexts):
        ap.error("each context must be at least 128 input tokens and input plus output must fit 524288")
    if args.pairs < 2 or args.pairs % 2 or args.chunk_size < 1 or args.timeout <= 0:
        ap.error("use a positive even number of pairs, a positive chunk size and a positive timeout")
    run(args, lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
