"""Time large independent GPU operations on both TP cards under simultaneous load."""
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import torch


def hardware():
    query = "uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit"
    args = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    return [line.split(", ") for line in subprocess.check_output(args, text=True).splitlines()]


def benchmark(rank, kind, n, start):
    with torch.inference_mode(), torch.cuda.device(rank):
        if kind == "copy":
            # Random payload exceeds L2 and avoids any compressible patterns.
            src = torch.empty(n, dtype=torch.uint8, device="cuda").random_(0, 256)
            dst = torch.empty_like(src)
            run = lambda: dst.copy_(src)
            work = 2 * n
            units = 1e9
        else:
            a = torch.randn((n, n), dtype=torch.bfloat16, device="cuda")
            b = torch.randn_like(a).T
            if kind == "bf16":
                out = torch.empty_like(a)
                run = lambda: torch.mm(a, b, out=out)
            else:
                a = a.to(torch.float8_e4m3fn)
                b = b.to(torch.float8_e4m3fn)
                one = torch.ones((), dtype=torch.float32, device="cuda")
                run = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one,
                                                out_dtype=torch.bfloat16)
            work = 2 * n ** 3
            units = 1e12
        run()
        torch.cuda.synchronize()
        start.wait(timeout=60)

        def sample(repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                run()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) / (1000 * repeats)

        # Ramp clocks, then target 200 ms per measurement.
        since = time.monotonic()
        while time.monotonic() - since < 1.5:
            seconds = sample(16)
        repeats = max(1, min(8000, int(0.20 / seconds)))
        timings = [sample(repeats) for _ in range(7)]
        rates = [round(work / elapsed / units, 2) for elapsed in timings]
        props = torch.cuda.get_device_properties(rank)
        return {"cuda_device": rank, "name": props.name, "uuid": str(props.uuid),
                "repeats": repeats, "samples": rates, "median": statistics.median(rates)}


def main():
    if torch.cuda.device_count() != 2 or any("RTX PRO 6000" not in torch.cuda.get_device_name(i)
                                              for i in range(2)):
        raise SystemExit("Expose exactly the two RTX PRO 6000 GPUs.")
    data = {"utc": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "telemetry_columns": ["uuid", "pstate", "sm_mhz",
            "memory_mhz", "power_w", "limit_w"], "before": hardware(), "runs": []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for kind, n in [("copy", 512 * 2**20), ("bf16", 4096), ("bf16", 8192),
                        ("fp8", 4096), ("fp8", 8192)]:
            start = threading.Barrier(2)
            started = time.monotonic()
            futures = [pool.submit(benchmark, i, kind, n, start) for i in range(2)]
            telemetry = []
            while not all(f.done() for f in futures):
                telemetry.append({"second": round(time.monotonic() - started, 2), "gpus": hardware()})
                time.sleep(0.4)
            results = [f.result() for f in futures]
            data["runs"].append({"kind": kind, "size": n, "unit": "GB/s" if kind == "copy" else "TFLOP/s",
                                 "devices": results, "telemetry": telemetry})
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
