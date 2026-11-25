#!/usr/bin/env python3
"""Build a native Engram overlay against the pinned local NVMe implementation."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PINNED_SHA256 = "07b43f726fc66ecfc43967750bd0dcea50d39a99850f3b2ec6f12b817a81f495"
HERE = Path(__file__).resolve().parent


def insert_after(source, anchor, addition):
    if source.count(anchor) != 1:
        raise ValueError(f"expected exactly one occurrence of {anchor!r}")
    return source.replace(anchor, anchor + addition)


def build(source_path, output):
    original = source_path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != PINNED_SHA256:
        raise ValueError(f"unsupported Engram source at {source_path}: expected {PINNED_SHA256}, got {digest}")
    if output.resolve() == source_path.parent.resolve():
        raise ValueError("use a separate output directory to preserve the pinned source")

    source = insert_after(original.decode(), "        self._pool = None\n", """
        from .dsv41_native_engram import NativeDecode

        self.native_decode = NativeDecode()
""")
    source = insert_after(source, '        assert t <= self.max_tokens, f"engram lookup of {t} tokens > staging {self.max_tokens}"\n', """        if t == 0:
            return
""")
    source = insert_after(source, "        self.ids_pin[:t].copy_(ids, non_blocking=True)\n", """        if 0 < t <= self.native_decode.max_tokens:
            self.native_decode.run(self, t, local_heads, vocab_start, vocab_end,
                                   quant_block, out, num_sms, background)
            return
""")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        tmp = Path(temporary)
        compiled = tmp / "libdsv41_engram.so"
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-fPIC", "-shared",
            "-Wall", "-Wextra", "-Werror", "-fno-exceptions", "-fno-rtti",
            str(HERE.parent / "native/engram.cpp"), "-o", str(compiled),
        ], check=True)
        (tmp / "engram.py").write_text(source)
        shutil.copyfile(HERE / "dsv41_native_engram.py", tmp / "dsv41_native_engram.py")
        for name in ("libdsv41_engram.so", "dsv41_native_engram.py", "engram.py"):
            (tmp / name).replace(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
