"""Build a profiling executable against the engine revision used for its trial."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = {
    "long-context-scorebench": ("stable-2026-11-25-pre-long-context-tuning", None),
    "edit-scaling-0fb9857-micro": ("experiment/edit-gate-tiles", "gate-tile.patch"),
    "c1-20261126-expert-micro": ("experiment/edit-gate-tiles", "first-revision.patch"),
    "c1-20261126-expert-r2-micro": ("experiment/c1-expert-load", None),
    "c1-20261126-wide-micro": ("experiment/c1-wide-ranking", None),
    "decode-attention-topk-20261126-micro": ("experiment/decode-attention-topk", None),
    "decode-paired-index-20261126-micro": ("experiment/decode-paired-index", None),
    "decode-paired-index-20261126-cross-row-micro": ("754d14583391755f53d5a96b3441cf833e94c262", None),
    "decode-round2-20261126-head-micro": ("experiment/c2-head-rows", None),
    "decode-round2-20261126-markov-micro": ("experiment/c2-markov-shortlist", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref, patch = EXPERIMENTS[args.experiment]
    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
                      cwd=ROOT, stdout=subprocess.DEVNULL).returncode:
        ref = "origin/" + ref
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.check_output(["git", "archive", ref, "native"], cwd=ROOT)
        subprocess.run(["tar", "xf", "-"], input=archive, cwd=tmp, check=True)
        if patch:
            subprocess.run(["git", "apply", str(HERE / patch)], cwd=tmp, check=True)
        toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        subprocess.run([toolkit / "bin/nvcc", "-arch=sm_120f", "-std=c++17", "-O3",
                        "-I" + str(Path(tmp) / "native"), HERE / (args.experiment + ".cu"),
                        "-lcublas", "-o", args.output.resolve()], check=True)


if __name__ == "__main__":
    main()
