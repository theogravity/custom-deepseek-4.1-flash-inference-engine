"""Use the pinned model recipe's device code without its PyTorch host bindings."""
from hashlib import sha256
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
data = source.read_bytes()
if sha256(data).hexdigest() != "847e54ad1768b85b42740a3c00d29d565d5b5128169c5594d241a399ada3cd63":
    raise SystemExit("recipe-prefill requires the Diffbot S8B kernel from the pinned model revision")
headers = {
    "compat.cuh": "dd6038fa6eb8b28df56b184e358e5633421af76eeb40bad4abc522ce6ba54f57",
    "ptx.cuh": "8ceacb1b321af5877a9a3d651e038fb1f44b543c3011e2f814f448830efe27ee",
    "quant/codebook.cuh": "0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a",
    "quant/exl3_dq.cuh": "7e3009b0f70635e8a98edf8b929832ca0fe16b6a5ad20f13b77ebfa72723feac",
    "quant/exl3_moe_common.cuh": "76b7c96b6ed88ca15e855791129a3f278b46f6dc54c1c72133f57f2daf060842",
    "quant/hadamard_inner.cuh": "63ca981a4a706924b4ab87047a903bc51ed86d6abe713b8d8034d8812d3467f3",
    "util.cuh": "1907cea115260db7c3b0de540e375e7733b7ffc9019bec190c1d92abea1d964e",
    "util.h": "ba89ac6793bf31cfe123d13baa2e34531a18a24bc3b2ff4add8cf4856b75a1c8",
}
include = source.parents[1] / "xmoe"
for header, expected in headers.items():
    if sha256((include / header).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Diffbot kernel header differs from the pinned recipe: {header}")
device = data.decode().split("}  // namespace dof", 1)[0] + "}  // namespace dof\n"
device = "\n".join(line for line in device.splitlines()
                   if not line.startswith(("#include <torch/", "#include <c10/")))
output.write_text(device + "\n")
