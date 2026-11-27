"""Qualify six resident 404k-token code reviews in one cold native serving pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from prefill_overlap import idle, stream
from six_streams import STREAMS, api_keys, query


OUTPUT = 128
COUNTERS = ("requests_started", "prefix_reused_tokens", "rotations", "resumes",
            "suspend_failures", "resume_failures")
STATE = ("logical_active", "resident", "prefilling", "decoding", "waiting", "swapped", "processed_tokens")
PROMPT_TOKENS = 404_000


def review(seed, agent):
    rng = random.Random((seed << 3) + agent)
    header = (f"Review patch {seed:x}.{agent} for a Python rules engine. Find correctness and "
              "maintenance issues and propose a refactor with regression tests.\n")
    lines = (f"def rule_{i}(x): return (x * {rng.randrange(13, 900)} + "
             f"{rng.randrange(1300, 33000)}) % 65519" for i in range(17606))
    return header + "\n".join(lines)


def local_memory(sample_gpu):
    """Read the client workstation's available RAM and physical inference GPUs."""
    sample = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                sample["host_available_mib"] = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError):
        pass
    if sample_gpu:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-i", "1,2", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True, timeout=3)
            for line in output.splitlines():
                gpu, free = line.split(",")
                sample[f"gpu_{int(gpu)}_free_mib"] = int(free)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return sample


def validate(rows, samples, before, after):
    issues = []
    delta = {k: after[k] - before[k] for k in COUNTERS} if after is not None else {}
    if not delta or delta["requests_started"] != STREAMS or any(delta[k] != 0 for k in COUNTERS[1:]):
        issues.append("unexpected admission, prefix reuse or swap counters")
    if len(rows) != STREAMS or len({r["prompt_sha256"] for r in rows}) != STREAMS:
        issues.append("expected six distinct input documents")
    complete = [r for r in rows if "usage" in r]
    if len(complete) != STREAMS:
        issues.append("one or more streams failed")
    for row in complete:
        usage = row["usage"]
        count, out, total = (usage.get(k, -1) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if (abs(count - PROMPT_TOKENS) > 100 or not 0 < out <= OUTPUT or total != count + out
                or row["finish"] not in ("length", "stop") or (row["finish"] == "length" and out != OUTPUT)):
            issues.append(f"unexpected usage or finish reason for agent {row['agent']}")
    # Only a live same-snapshot observation before the first client terminates
    # can establish that all six full prompts were resident together.
    first_done = min((r["done_s"] for r in rows), default=0)
    witnessed = [s for s in samples if s["elapsed_s"] < first_done and s["logical_active"] == STREAMS
                 and s["resident"] == STREAMS and s["decoding"] == STREAMS
                 and s["prefilling"] == s["waiting"] == s["swapped"] == 0]
    proof = max(witnessed, key=lambda s: s["processed_tokens"], default=None)
    total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in complete)
    if not proof or len(complete) != STREAMS or proof["processed_tokens"] < total_prompt:
        issues.append("did not observe all six complete prompts resident before the first stream ended")
    return issues, delta, proof


def run(base, key, seed):
    prompts = [review(seed, i) for i in range(STREAMS)]
    before = idle(base, key)
    start = time.monotonic()
    barrier = threading.Barrier(STREAMS)

    def client(i):
        prompt = prompts[i]
        row = {"agent": i, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        barrier.wait(timeout=30)
        try:
            result = stream(base, key, prompt, OUTPUT, timeout=21600)
            row.update(usage=result.usage, finish=result.finish, ttft_s=result.events[0]-result.started,
                       started_s=result.started-start, done_s=result.done-start)
        except Exception as exc:
            row.update(error=str(exc), done_s=time.monotonic()-start)
        return row

    local = urlsplit(base).hostname in ("127.0.0.1", "::1", "localhost")
    gpu = local and shutil.which("nvidia-smi") is not None
    next_memory = 0
    minima = {}
    samples = []
    peak = None
    monitor_error = None
    with ThreadPoolExecutor(STREAMS) as pool:
        clients = [pool.submit(client, i) for i in range(STREAMS)]
        pending = clients
        while pending:
            try:
                state = query(base, "/metrics", key)
            except (OSError, ValueError) as exc:
                # Keep per-client failures and the last confirmed residency if the worker dies.
                monitor_error = str(exc)
                state = {}
            observed = time.monotonic()-start
            if state.get("resident") == STREAMS:
                snap = {"elapsed_s": observed} | {k: state[k] for k in STATE}
                if peak is None or snap["processed_tokens"] > peak["processed_tokens"]:
                    peak = snap
                if state["decoding"] == STREAMS:
                    samples.append(snap)
            if local and observed >= next_memory:
                for k, value in local_memory(gpu).items():
                    minima[k] = min(minima.get(k, value), value)
                next_memory = observed + 2
            _, pending = wait(pending, timeout=0.1)
        rows = [future.result() for future in clients]
    try:
        after = idle(base, key)
    except (OSError, ValueError, RuntimeError) as exc:
        after, monitor_error = None, str(exc)
    issues, delta, proof = validate(rows, samples, before, after)
    if monitor_error is not None:
        issues.append(f"scheduler monitor failed: {monitor_error}")
    return dict(passed=not issues, issues=issues, seed=seed, requested_output=OUTPUT, rows=rows,
                elapsed_s=max(r["done_s"] for r in rows), counters=delta, peak_six_resident=peak,
                full_six_resident=proof, sampled_local_memory_minimum_mib=minima)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--seed", type=int, help="document seed (default: a fresh random seed)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    seed = secrets.randbits(48) if args.seed is None else args.seed
    if not 0 <= seed < (1 << 64):
        ap.error("seed must be an unsigned 64-bit integer")
    if args.worker:
        key, _ = api_keys(False)
        print(json.dumps(run(args.base, key, seed)))
        return 0
    # SSE heartbeats can reset a socket's read timeout indefinitely.
    if args.out.exists():
        raise FileExistsError(args.out)
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--base",
                             args.base, "--seed", str(seed)], capture_output=True, text=True, timeout=21600)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    receipt = json.loads(result.stdout)
    with args.out.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
