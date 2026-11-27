"""Compare two native EXL3 tensor-parallel expert shards to the installed model pack."""
from pathlib import Path
import argparse
import ctypes as C
import json
import sys

import torch
from safetensors import safe_open
from check_moe import Job, RoutedRow


def checked(status):
    if status:
        raise RuntimeError(f"CUDA call failed with {status}")


def compare(root, library):
    sys.path.insert(0, str(root / "recipe/kernels/build/xmoe_do_i8"))
    import xmoe_do_i8 as pack

    ptr = C.c_void_p
    if "default mode 1 " not in pack.info() or "variant 0" not in pack.info():
        raise ValueError(f"Expected S8B variant 0, got {pack.info()}")
    lib = C.CDLL(str(library))
    lib.ds_recipe_rank_slot_bytes.restype = C.c_size_t
    lib.ds_recipe_upload_rank.argtypes = [ptr, C.c_size_t, C.POINTER(ptr), C.c_int]
    lib.ds_recipe_rank_group.argtypes = [C.POINTER(ptr), C.POINTER(C.c_size_t), C.POINTER(Job),
                                        C.c_int, C.POINTER(RoutedRow), C.c_int, ptr, ptr]
    lib.ds_moe_free.argtypes = [C.c_int, ptr]
    lib.ds_moe_free.restype = None
    wc = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    count = 129  # two native jobs for one expert
    rank_width = 1152
    features = 5120
    for layer in (0, 20):
        suffix = f"layers.{layer}.ffn.experts.3"
        with safe_open(str(root / wc[suffix+".w1.trellis"]), framework="pt", device="cpu") as tensors:
            projections = [[tensors.get_tensor(f"{suffix}.{matrix}.{tensor}").contiguous()
                            for tensor in ("trellis","suh","svh")] for matrix in ("w1","w3","w2")]
        params = [v[0] for v in projections] + [arr for v in projections for arr in v[1:]]
        upload = (ptr*9)(*(x.data_ptr() for x in params))
        native_parts, pack_parts = [], []
        for rank in (0,1):
            torch.cuda.set_device(rank)
            slot = torch.empty(lib.ds_recipe_rank_slot_bytes(), device=rank, dtype=torch.uint8)
            checked(lib.ds_recipe_upload_rank(slot.data_ptr(), slot.numel(), upload, rank))
            blocks = []
            for proj, (tr, suh, svh) in enumerate(projections):
                dim = 0 if proj == 2 else 1
                tr = tr.chunk(2, dim=dim)[rank].contiguous()
                if proj == 2: suh = suh.chunk(2, dim=0)[rank]
                else: svh = svh.chunk(2, dim=0)[rank]
                blocks.extend([x.to(rank).contiguous() for x in (tr,suh,svh)])
            def ints(values, dtype=torch.int64):
                return torch.tensor(values, device=rank, dtype=dtype)
            planar = [ints([a.data_ptr()]) for a in blocks]
            x = (torch.arange(count*features,device=rank).float()*0.003).sin().reshape(count,features)
            order = list(range(count-1,-1,-1))
            weights = (torch.arange(count,device=rank).remainder(7)+1).half()/8
            ref = torch.zeros_like(x)
            scratch_ref = [torch.empty((count,cols),device=rank,dtype=torch.float16)
                           for cols in (features,features,rank_width,rank_width)]
            pack.do_moe(x.half(),ref,ints(order),weights,ints([0]*count,torch.int32),
                        ints([0,128],torch.int32),ints([128,1],torch.int32),ints([0,0]),
                        *planar,*scratch_ref,3,2,10.,0)
            jobs = (Job*2)(Job(slot.data_ptr(),0,128),Job(slot.data_ptr(),128,1))
            routes = (RoutedRow*count)(*(RoutedRow(slot.data_ptr(),row,row,(1+i%7)/8)
                                        for i,row in enumerate(order)))
            scratch = ptr()
            capacity = C.c_size_t()
            out = torch.full_like(x, float("nan"))
            try:
                for layout in [[(0,129)], [(0,127)], [(1,128),(128,1)], [(0,128),(128,0)],
                               [(0,128),(128,2)], [(-1,128),(128,1)], [(0,128),(127,1)]]:
                    invalid = (Job*len(layout))(*(Job(slot.data_ptr(), start, length) for start,length in layout))
                    status = lib.ds_recipe_rank_group(C.byref(scratch),C.byref(capacity),invalid,len(layout),
                                                      routes,count,x.data_ptr(),out.data_ptr())
                    if status != 1 or scratch.value is not None:
                        raise RuntimeError(f"Invalid job layout was processed: {layout} (CUDA {status})")
                checked(lib.ds_recipe_rank_group(C.byref(scratch),C.byref(capacity),jobs,2,
                                                 routes,count,x.data_ptr(),out.data_ptr()))
                torch.cuda.synchronize(rank)
                bit_equal = torch.equal(out.view(torch.int32), ref.view(torch.int32))
                if not bit_equal or not torch.isfinite(out).all():
                    raise ValueError(f"rank {rank} layer {layer} differs from the pack extension")
                native_parts.append(out.cpu())
                pack_parts.append(ref.cpu())
            finally:
                lib.ds_moe_free(rank,scratch)
            print(json.dumps(dict(layer=layer, rank=rank, channels=rank_width, rows=count,
                                  bit_equal=bit_equal)),flush=True)
        if torch.equal(native_parts[0], native_parts[1]):
            raise ValueError("test did not distinguish the two tensor-parallel shards")
        if not torch.equal((native_parts[0]+native_parts[1]).view(torch.int32),
                           (pack_parts[0]+pack_parts[1]).view(torch.int32)):
            raise ValueError("host sum of the two rank outputs differs")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    args=parser.parse_args()
    compare(args.model, args.library)
