#!/usr/bin/env python3
"""Exercise six overlapping native streams, including independently sized near-window prompts.

With the local API key in the environment, run both phases in the same checkpoint configuration:

  uv run --with tokenizers==0.22.2 python tools/six_streams.py --seed 26611502719397 --input 404000 --output 128 --save-histories histories.json
  uv run --with tokenizers==0.22.2 python tools/six_streams.py --input 524160 --output 128 --continue-histories histories.json

Choose a fresh seed for a cold first phase. At 64-token chunks, six existing
403,968-token checkpoints leave 721,152 input tokens to process in phase 2.
Processing both sizes then needs 3,145,152 input tokens and two decode rounds.
One direct cold full-window run needs 3,144,960 input tokens and one decode round.

Add --litellm to either command to stream completions through the local OpenAI
alias with LiteLLM response cache reads and writes disabled. Sizing and batch
counters still use the authenticated native server specified by --base.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from code_speed import NON_TEXT_FIELDS, PROMPTS

MODEL = "DeepSeek V4.1 Flash"
LITELLM_MODEL = "DeepSeek V4.1 Flash (openai)"
STREAMS = 6
DEFAULT_SEED = 268815548685955


def http(base, path, key, body=None):
    data = None if body is None else json.dumps(body).encode()
    return urlopen(Request(base.rstrip("/") + path, data=data,
                           headers={"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"}), timeout=21600)


def query(base, path, key, body=None):
    with http(base, path, key, body) as resp:
        return json.load(resp)


def payload(prompt, limit):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    return {"model": MODEL, "messages": messages,
            "temperature": 0, "max_tokens": limit, "reasoning_effort": "none"}


def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    # Frozen independently by the checkpoint's encoding.py. The same fixture
    # covers the serving adapter's Rust chat::render + Prompt::encode.
    fixture = json.loads((Path(__file__).resolve().parents[1] / "tests/receipt_prefix.json").read_text())
    first = fixture["first_ids"]
    if (chat_ids(tok, fixture["messages"][:1]) != first
            or chat_ids(tok, fixture["messages"]) != first + fixture["additional_ids"]):
        raise ValueError("checkpoint tokenizer does not match the frozen receipt transcript")
    return tok


def chat_ids(tok, messages):
    # Exact plain-text, thinking-disabled template, also exercised by Rust's
    # checkpoint golden test. Keep content encoding separate from controls.
    def control(value):
        token_id = tok.token_to_id(value)
        if token_id is None:
            raise ValueError(f"checkpoint has no token {value!r}")
        return token_id
    ids = [control("<｜begin▁of▁sentence｜>")]
    for i, message in enumerate(messages):
        role = "assistant" if i % 2 else "user"
        if message["role"] != role:
            raise ValueError("expected alternating user and assistant turns")
        if role == "user":
            ids.append(control("<｜User｜>"))
        text = message["content"]
        if not text.isascii() or any(marker in text for marker in ["<think>", "</think>"]):
            raise ValueError("unexpected content in staged receipt fixture")
        ids.extend(tok.encode(text, add_special_tokens=False).ids)
        if role == "user":
            ids.extend([control("<｜Assistant｜>"), control("</think>")])
        else:
            ids.append(control("<｜end▁of▁sentence｜>"))
    return ids


def id_digest(ids):
    from array import array
    import sys
    packed = array("I", ids)
    if packed.itemsize != 4:
        raise RuntimeError("expected 32-bit unsigned token IDs")
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def save_histories(path, args, specs, rows, tok):
    sessions = []
    for i, ((prompt, receipts), result) in enumerate(zip(specs, rows, strict=True)):
        messages = payload(prompt, args.output)["messages"]
        ids = chat_ids(tok, messages)
        if len(ids) != result["prompt"]:
            raise ValueError("local checkpoint and server disagree on prompt tokens")
        sessions.append({"agent": i, "messages": messages + [
            {"role": "assistant", "content": result.pop("reply")}], "receipts": receipts,
            "prompt_ids_sha256": id_digest(ids), "reply_sha256": result["sha256"]})
    data = {"version": 1, "seed": args.seed, "input": args.input, "sessions": sessions}
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(data, f)
            f.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def fit_text(head, words, suffix, nwords, target, size, min_words=0):
    while nwords >= min_words:
        prefix = head + " ".join(words[:nwords])
        count = size(prefix + suffix)
        if count <= target:
            break
        nwords -= max(1, (count - target + 1) // 2)
    else:
        raise ValueError("requested input would discard a required receipt or instruction")
    pad = target - count
    for _ in range(4):
        text = prefix + " x" * pad + suffix
        count = size(text)
        if count == target:
            return text
        pad += target - count
        if pad < 0:
            break
    raise ValueError(f"failed to size transcript: {count}, expected {target}")


def document(args, key, agent):
    # The shared case reuses a known real, persisted prefix. The independent
    # case changes the seed and every generated document from its beginning.
    seed = args.seed if args.shared else args.seed + agent
    rng = random.Random(seed)
    words = [rng.choice(("alpha", "beta", "gamma", "delta", "omega")) + str(rng.randrange(100000))
             for _ in range(174000)]
    early_index = 2600
    for name, index in (("A", early_index), ("B", 44600), ("C", 83500), ("D", 166000)):
        words[index] = f"\nVAULT {name} receipt = R{rng.getrandbits(48):012x}\n"
    if args.save_histories:
        # Keep an undisclosed early receipt even in the smallest staged dry run.
        words[early_index // 2] = f"\nVAULT G receipt = R{rng.getrandbits(48):012x}\n"
    receipt = f"R{random.Random((args.seed+agent)*997).getrandbits(48):012x}"
    # Ensure an unrelated, early needle in the independent-context gate. The
    # late marker belongs to this request even when multiple agents share text.
    early = words[early_index].split(" = ")[1].strip()
    suffix = (f"\nVAULT E receipt = {receipt}\n"
              "Print the VAULT A receipt then the VAULT E receipt separated by a space. "
              "Repeat that exact pair on each line for 100 lines, starting immediately.")
    head = f"Vault register {seed}.\n"
    def size(text):
        return query(args.base, "/v1/messages/count_tokens", key,
                     payload(text, args.output))["input_tokens"]
    nwords = min(len(words), args.input // 3)
    return fit_text(head, words, suffix, nwords, args.input, size, early_index+1), [early, receipt]


def continued_documents(args, key, tok, frozen):
    if frozen["version"] != 1 or frozen["seed"] != args.seed:
        raise ValueError("staged history version or seed mismatch")
    sessions = frozen["sessions"]
    if len(sessions) != STREAMS or len({s["prompt_ids_sha256"] for s in sessions}) != STREAMS:
        raise ValueError("expected six distinct frozen histories")
    specs, proofs = [], []
    for agent, s in enumerate(sessions):
        turns = s["messages"]
        if s["agent"] != agent or [m["role"] for m in turns] != ["user", "assistant"]:
            raise ValueError(f"invalid frozen transcript for agent {agent}")
        reply = turns[1]["content"]
        if hashlib.sha256(reply.encode()).hexdigest() != s["reply_sha256"] or not check_output(
            reply, s["receipts"], agent
        )[0]:
            raise ValueError(f"invalid frozen assistant response for agent {agent}")
        # B is deep in the initial 404k prompt. On smaller controls the G
        # marker supplies the same challenge. Neither was named in the first reply.
        hidden = re.search(r"\nVAULT B receipt = (R[0-9a-f]{12})\n", turns[0]["content"])
        name = "B"
        if not hidden:
            hidden = re.search(r"\nVAULT G receipt = (R[0-9a-f]{12})\n", turns[0]["content"])
            name = "G"
        if not hidden or hidden[1] in reply:
            raise ValueError(f"history lacks an undisclosed receipt for agent {agent}")
        original = chat_ids(tok, turns[:1])
        digest = id_digest(original)
        if digest != s["prompt_ids_sha256"] or len(original) != frozen["input"]:
            raise ValueError(f"frozen input token digest or length changed for agent {agent}")
        prior = chat_ids(tok, turns)
        room = args.input - len(prior)
        rng = random.Random(f"continuation-{args.seed}-{agent}")
        receipt = f"R{rng.getrandbits(48):012x}"
        suffix = (f"\nVAULT F receipt = {receipt}\n"
                  f"Print the VAULT {name} receipt then the VAULT F receipt separated by a space. "
                  "Repeat that exact pair on each line for 100 lines, starting immediately.")
        head = f"Vault register supplement {args.seed + agent}.\n"
        # In this template, the new turn contributes User + text + Assistant
        # + </think>. Size only that content until the final verification.
        budget = room - 3
        if budget <= 0:
            raise ValueError(f"no room for a continuation at {args.input} input tokens")
        words = [rng.choice(("zeta", "delta", "beta", "eta", "gamma")) + str(rng.randrange(100000))
                 for _ in range(budget // 3)]
        def size(text):
            return len(tok.encode(text, add_special_tokens=False).ids)
        text = fit_text(head, words, suffix, len(words), budget, size)
        messages = turns + [{"role": "user", "content": text}]
        ids = chat_ids(tok, messages)
        remote = query(args.base, "/v1/messages/count_tokens", key,
                       payload(messages, args.output))["input_tokens"]
        if (len(ids) != args.input or remote != len(ids)
                or ids[:len(original)] != original or ids[:len(prior)] != prior):
            raise ValueError(f"continuation length or exact token prefix mismatch for agent {agent}")
        specs.append((messages, [hidden[1], receipt]))
        proofs.append({"verified_prefix_tokens": len(original), "prefix_ids_sha256": digest,
                       "prior_transcript_tokens": len(prior), "prior_transcript_sha256": id_digest(prior),
                       "undisclosed_vault": name, "extended_ids_sha256": id_digest(ids)})
    return specs, proofs


def check_output(text, expected, idx):
    if expected:
        words = text.split()
        # A length stop may cut the final receipt short. Any extra prose,
        # malformed interior receipt or receipt from another request fails.
        valid = words[:2] == expected and all(
            word == expected[i % 2]
            or (i == len(words) - 1 and expected[i % 2].startswith(word))
            for i, word in enumerate(words)
        )
        receipts = [word for word in words if re.fullmatch(r"R[0-9a-f]{12}", word)]
        return valid, receipts, "receipt_sequence"
    # The coding prompts deliberately request programs longer than the output
    # limit. Check for task-specific code in the visible prefix.
    patterns = (
        (r"\bpygame\b", r"\b(?:import|from|class|def)\s+\w+"),
        (r"#\s*include\s*[<\"]", r"(?i)\b(?:json\w*|parse\w*|token\w*)\b"),
        (r"\b(?:fn|struct|impl|use)\s+\w+", r"(?i)\b(?:lru\w*|\w*cache\w*|HashMap|Mutex|Instant)\b"),
    )
    valid = all(re.search(pattern, text) for pattern in patterns[idx % len(patterns)])
    return valid, [], "code_prefix"


def stream(args, key, prompt, expected, barrier, idx):
    body = payload(prompt, args.output)
    if args.litellm:
        body.update(model=LITELLM_MODEL, cache={"no-cache": True, "no-store": True})
    body.update(stream=True, stream_options={"include_usage": True})
    barrier.wait(timeout=120)
    start = time.monotonic()
    first = last = usage = finish = None
    content = []
    done = False
    stream_id = None
    with http(args.litellm or args.base, "/v1/chat/completions", key, body) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if not event.get("id") or (stream_id and event["id"] != stream_id):
                raise ValueError(f"agent {idx} changed completion identity")
            stream_id = event["id"]
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                if any(delta.get(field) for field in NON_TEXT_FIELDS):
                    raise ValueError("expected plain visible output")
                if text := delta.get("content"):
                    last = time.monotonic()
                    if first is None:
                        first = last
                    content.append(text)
    end = time.monotonic()
    if not done or not usage or not first or last <= first:
        raise ValueError(f"agent {idx} returned an incomplete stream")
    text = "".join(content)
    tokens = usage["completion_tokens"] - int(finish == "stop")
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") or details.get("audio_tokens"):
        raise ValueError("unexpected non-text tokens in throughput")
    valid, found, check = check_output(text, expected, idx)
    row = {"agent": idx, "stream_id": stream_id, "quality_check": check,
            "complete_receipts": len(found), "prompt": usage["prompt_tokens"], "output": tokens,
            "finish": finish, "exact": valid, "expected": expected, "found": found[:2],
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "first": first, "last": last, "end": end, "ttft_s": first-start, "decode_tok_s": (tokens-1)/(last-first)}
    if args.save_histories:
        row["reply"] = text
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001", help="native server root (default http://127.0.0.1:8001)")
    ap.add_argument("--litellm", nargs="?", const="http://127.0.0.1:4000", metavar="ROOT",
                    help="send completions to the LiteLLM OpenAI alias (default proxy root http://127.0.0.1:4000)")
    ap.add_argument("--seed", type=int, help=f"document seed (default {DEFAULT_SEED} or the frozen histories' seed)")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--input", type=int, help="exact target context input token count")
    ap.add_argument("--output", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=1)
    stage = ap.add_mutually_exclusive_group()
    stage.add_argument("--save-histories", type=Path, help="freeze six independent initial user turns, actual completions and token digests")
    stage.add_argument("--continue-histories", type=Path, help="extend the six frozen transcripts with a new receipt and instruction")
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/models/dsv41-flash-exl3/tokenizer.json"),
                    help="checkpoint tokenizer used to verify staged input IDs")
    ap.add_argument("--baseline", action="store_true", help="launch the same six clients against the serial control, without requiring concurrent engine metrics")
    args = ap.parse_args()
    if args.input is not None and not 12000 <= args.input <= 524288-args.output:
        ap.error("input plus output must fit the context and include the early needle")
    if args.rounds < 1 or args.output < 2:
        ap.error("need at least one round and two output tokens")
    if (args.save_histories or args.continue_histories) and (args.shared or not args.input):
        ap.error("staged runs require six independent histories and --input")
    if args.save_histories and args.rounds != 1:
        ap.error("freeze a single round of histories")
    if args.litellm and args.baseline:
        ap.error("--litellm requires native concurrency metrics")
    key = os.environ.get("DSV41_API_KEY") or os.environ["LITELLM_MASTER_KEY"]
    completion_key = (os.environ.get("LITELLM_MASTER_KEY") or key) if args.litellm else key
    tok = load_tokenizer(args.tokenizer) if args.save_histories or args.continue_histories else None
    frozen = json.loads(args.continue_histories.read_text()) if args.continue_histories else None
    if args.seed is None:
        args.seed = frozen["seed"] if frozen is not None else DEFAULT_SEED
    proofs = None
    if args.continue_histories:
        specs, proofs = continued_documents(args, key, tok, frozen)
    elif args.input:
        specs = [document(args, key, i) for i in range(STREAMS)]
    else:
        specs = [(list(PROMPTS.values())[i%3] + ("" if i<3 else " Include explanatory code comments."), []) for i in range(STREAMS)]
    expected_inputs = [args.input] * STREAMS
    if args.litellm and args.input is None:
        expected_inputs = [query(args.base, "/v1/messages/count_tokens", key,
                                 payload(prompt, args.output))["input_tokens"] for prompt, _ in specs]
    ok = True
    for rep in range(args.rounds):
        barrier = threading.Barrier(STREAMS)
        before = {} if args.baseline else query(args.base,"/metrics",key)
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(STREAMS) as pool:
            pending = [pool.submit(stream,args,completion_key,*s,barrier,i) for i,s in enumerate(specs)]
            peak_observed_tokens = 0
            if not args.baseline:
                unfinished = pending
                while unfinished:
                    _, unfinished = concurrent.futures.wait(unfinished, timeout=0.1)
                    if unfinished:
                        sample = query(args.base, "/metrics", key)
                        if sample["active"] == STREAMS:
                            peak_observed_tokens = max(peak_observed_tokens, sample["processed_tokens"])
                        # Once residency has been directly observed, avoid polling
                        # through the remainder of a long decode.
                        if args.input and peak_observed_tokens >= args.input * STREAMS:
                            break
            rows = [f.result() for f in pending]
        elapsed = max(r.pop("end") for r in rows) - start
        overlap = min(r["last"] for r in rows)-max(r["first"] for r in rows)
        after = {} if args.baseline else query(args.base,"/metrics",key)
        valid = all(r["exact"] and r["finish"] == "length" and r["output"] == args.output
                    and (expected is None or r["prompt"] == expected)
                    for r, expected in zip(rows, expected_inputs, strict=True))
        valid &= len({r["stream_id"] for r in rows}) == STREAMS
        if proofs:
            for row, proof in zip(rows, proofs, strict=True):
                row.update(proof)
        if args.save_histories:
            if valid:
                save_histories(args.save_histories, args, specs, rows, tok)
            for row in rows:
                row.pop("reply", None)
        six_passes = 0 if args.baseline else after["six_target_passes"] - before["six_target_passes"]
        running_together = overlap > 0 and six_passes > 0
        decode_span = max(r["last"] for r in rows) - min(r["first"] for r in rows)
        decode_rate = sum(r["output"] - 1 for r in rows) / decode_span
        ok &= valid and (args.baseline or running_together)
        for r in rows:
            r["first_s"] = r.pop("first")-start
            r["last_s"] = r.pop("last")-start
        print(json.dumps({"round":rep,"seed":args.seed,"shared_prefix":args.shared,"requested_input":args.input,
             "completion_route":"litellm" if args.litellm else "native",
             "completion_model":LITELLM_MODEL if args.litellm else MODEL,
             "seconds":elapsed,"aggregate_wall_tok_s":sum(r["output"] for r in rows)/elapsed,
             "six_stream_overlap_s":overlap,"six_target_passes_delta":six_passes,
             "observed_six_processed_tokens":peak_observed_tokens,"aggregate_decode_tok_s":decode_rate,"all_valid":valid,"concurrent":running_together,
             "metrics_before":before,"metrics_after":after,"rows":rows}),flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
