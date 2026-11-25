#!/usr/bin/env python3
"""Exercise the running engine through this workstation's installed coding clients.

Run python3 tools/clients.py with the candidate on port 8001 and LiteLLM on
port 4000. Each check emits a JSON record. A failed check sets the exit code.
"""

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


CHECKS = ("auth", "stream", "codex", "claude", "pi", "pi-deepseek")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def launch(client, args, cache):
    """Read model-cli's provider settings without evaluating or logging shell text."""
    name = "pi" if client == "pi-deepseek" else client
    command = ["model-cli", name, args.model_id, "--port", str(args.proxy_port)]
    if client == "pi-deepseek":
        command.extend(["--alias", "deepseek"])
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=args.timeout,
        env={**os.environ, "XDG_CACHE_HOME": str(cache), "MODEL_CLI_NO_UPDATE": "1"},
    )
    require(result.returncode == 0, f"model-cli {name} failed (exit {result.returncode})")
    tokens = shlex.split(result.stdout.replace("\\\n", " "))
    position = tokens.index(name)
    environment = dict(token.split("=", 1) for token in tokens[:position])
    return environment, tokens[position:]


def check_auth(args, base, key, model):
    for url in (base, f"http://localhost:{args.engine_port}"):
        models_url = url + "/v1/models"
        try:
            with urlopen(models_url, timeout=args.timeout):
                raise ValueError(f"{url} accepted an anonymous models request")
        except HTTPError as error:
            require(error.code in (401, 403), f"{url} returned {error.code} without credentials")
            error.close()
        request = Request(models_url, headers={"Authorization": "Bearer " + key})
        with urlopen(request, timeout=args.timeout) as response:
            models = json.load(response)["data"]
        require(any(item["id"] == model for item in models), f"{url} did not advertise {model!r}")
    return {"proxy_port": args.proxy_port, "engine_port": args.engine_port, "model": model}


def sse_events(response):
    data = []
    for raw in response:
        line = raw.decode().rstrip("\r\n")
        if not line and data:
            yield json.loads("\n".join(data))
            data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


def check_stream(args, base, key, model):
    a, b = (secrets.randbelow(80) + 11 for _ in range(2))
    payload = {
        "model": model, "stream": True, "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": f"Multiply {a} by {b}. Answer with only the integer."}],
    }
    request = Request(
        base + "/v1/messages", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Api-Key": key, "Anthropic-Version": "2023-06-01"},
    )
    with urlopen(request, timeout=args.timeout) as response:
        require(response.headers.get_content_type() == "text/event-stream", "expected Anthropic SSE")
        thinking, answer, events = [], [], []
        stop_reason = None
        for event in sse_events(response):
            kind = event["type"]
            events.append(kind)
            if kind == "error":
                raise ValueError(f"Anthropic stream error ({event['error'].get('type')})")
            if kind in ("content_block_start", "content_block_delta"):
                block = event.get("content_block") or event["delta"]
                thinking.append(block.get("thinking", ""))
                answer.append(block.get("text", ""))
            if kind == "message_delta":
                stop_reason = event["delta"].get("stop_reason") or stop_reason
    require("message_start" in events and events[-1] == "message_stop", "incomplete message stream")
    require(stop_reason == "end_turn", f"unexpected stop reason {stop_reason!r}")
    require("".join(thinking).strip(), "missing streamed thinking")
    require("".join(answer).strip() == str(a * b), "incorrect streamed final answer")
    return {"model": model, "events": len(events), "thinking_chars": len("".join(thinking))}


def check_client(client, args, folder):
    # The prompt never includes the receipt. Returning it requires reading the
    # file locally and processing the tool result in a second inference turn.
    settings, command = launch(client, args, folder / "cache")
    workspace = folder / "workspace"
    workspace.mkdir()
    receipt = secrets.token_hex(16)
    witness = workspace / "receipt.txt"
    witness.write_text(receipt + "\n")
    prompt = f"Use a tool to read {witness}. Reply with only the complete contents of that file."
    env = {**os.environ, **settings}
    if client == "codex":
        command.remove("--dangerously-bypass-approvals-and-sandbox")
        command += ["exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                    "--output-last-message", str(folder / "answer"), prompt]
        env["CODEX_HOME"] = str(folder / "codex")
        Path(env["CODEX_HOME"]).mkdir()
    elif client == "claude":
        command.remove("--dangerously-skip-permissions")
        command.remove("--chrome")
        command += ["--bare", "--strict-mcp-config", "--no-session-persistence",
                    "--model", settings["ANTHROPIC_DEFAULT_SONNET_MODEL"], "--tools", "Read",
                    "--allowedTools", "Read", "--output-format", "json", "--print", prompt]
        env.update(CLAUDE_CONFIG_DIR=str(folder / "claude"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    else:
        command += ["--offline", "--no-extensions", "--no-context-files", "--no-skills",
                    "--no-mcp", "--no-session", "--tools", "read", "--mode", "text", "--print", prompt]
        env.update(PI_CODING_AGENT_DIR=str(folder / "pi"), PI_TELEMETRY="0")
    version = subprocess.check_output([command[0], "--version"], env=env, text=True, timeout=15).strip()
    run = subprocess.run(command, env=env, cwd=workspace, capture_output=True,
                         stdin=subprocess.DEVNULL, text=True, timeout=args.timeout)
    if run.returncode:
        # Restrict failure diagnostics to the tail and redact launcher credentials.
        detail = run.stderr.strip() or run.stdout.strip()
        for name, value in settings.items():
            if value and ("KEY" in name or "TOKEN" == name.rsplit("_", 1)[-1]):
                detail = detail.replace(value, "[redacted]")
        raise ValueError(f"{client} exited {run.returncode}: {detail[-1200:]}")
    if client == "codex":
        reply = (folder / "answer").read_text()
    elif client == "claude":
        result = json.loads(run.stdout)
        require(result.get("is_error") is False, "Claude Code returned an error")
        reply = result["result"]
    else:
        reply = run.stdout
    require(reply.strip() == receipt, f"{client} final reply did not match the tool result")
    return {"version": version, "tool_result_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="dsv41-flash-exl3")
    parser.add_argument("--proxy-port", type=int, default=4000)
    parser.add_argument("--engine-port", type=int, default=8001)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=CHECKS)
    args = parser.parse_args()
    passed = True
    for check in args.checks:
        started = time.monotonic()
        record = {"check": check}
        try:
            with tempfile.TemporaryDirectory(prefix="dsv41-acceptance-") as directory:
                folder = Path(directory)
                if check in ("auth", "stream"):
                    env, _ = launch("claude", args, folder / "cache")
                    test = check_auth if check == "auth" else check_stream
                    record.update(test(args, env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_API_KEY"],
                                       env["ANTHROPIC_DEFAULT_SONNET_MODEL"]))
                else:
                    record.update(check_client(check, args, folder))
            record["ok"] = True
        except subprocess.TimeoutExpired:
            record.update(ok=False, error="timeout")
            passed = False
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            record.update(ok=False, error=str(error))
            passed = False
        record["seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(record, ensure_ascii=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
