"""Compare the standalone CUDA adapter with the installed Diffbot Torch extension.

Run inside the matching vLLM image with both inference cards visible, the model
pack mounted and the two native adapter objects linked as a shared library.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from check_moe import Job, RoutedRow

def check(code):
    if code:
        raise RuntimeError(f"native CUDA error {code}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model / "recipe/kernels/build/xmoe_do_i8"))
    import xmoe_do_i8 as reference
    if "default mode 1 " not in reference.info() or "variant 0" not in reference.info():
        raise ValueError(reference.info())
    lib = C.CDLL(str(args.library.resolve()))
    ptr = C.c_void_p
    lib.ds_moe_run_dev.argtypes = [C.POINTER(ptr), C.POINTER(C.c_size_t), C.POINTER(Job), C.c_int,
                                    C.POINTER(RoutedRow), C.c_int, ptr, ptr] + [C.c_int]*4
    lib.ds_moe_alloc.argtypes = [C.POINTER(ptr)]
    lib.ds_moe_upload.argtypes = [ptr,C.POINTER(ptr)]
    lib.ds_moe_free.argtypes = [C.c_int,ptr]
    lib.ds_moe_free.restype = None
    cudart = C.CDLL("libcudart.so." + torch.version.cuda.split(".")[0])
    cudart.cudaDeviceEnablePeerAccess.argtypes = [C.c_int, C.c_uint]
    weight_map = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    for device, layer in [(0,0),(1,20)]:
        torch.cuda.set_device(device)
        scalars = lambda values, dtype: torch.tensor(values,device=device,dtype=dtype)
        ind = lambda values: scalars(values,torch.int64)
        parts = []
        slots = []
        scratch = ptr()
        size = C.c_size_t()
        try:
            shard = weight_map[f"layers.{layer}.ffn.experts.0.w1.trellis"]
            with safe_open(str(args.model/shard), framework="pt", device="cpu") as tensors:
                for expert in [0,3]:
                    blocks = [[tensors.get_tensor(f"layers.{layer}.ffn.experts.{expert}.{proj}.{name}").contiguous()
                               for name in ("trellis","suh","svh")] for proj in ("w1","w3","w2")]
                    slot = ptr()
                    check(lib.ds_moe_alloc(C.byref(slot)))
                    slots.append(slot)
                    upload = [b[0] for b in blocks]+[v for b in blocks for v in b[1:]]
                    check(lib.ds_moe_upload(slot,(ptr*9)(*(v.data_ptr() for v in upload))))
                    parts.append([[v.cuda(device) for v in b] for b in blocks])
            planar = [ind([parts[e][q][i].data_ptr() for e in range(2)]) for q in range(3) for i in range(3)]
            for runs in [(127,), (128,), (129,), (5,129)]:
                n = sum(runs)
                source_rows = [(n - i - 1) // 2 for i in range(n)]
                destinations = [2 * (n - i - 1) + 1 for i in range(n)]
                tok = ind(range(n))
                wt = (torch.arange(n,device=device).remainder(7)+1).half()/8
                jobs = []
                owners = []
                offset=0
                for e,count in enumerate(runs):
                    jobs.extend((offset+i,min(128,count-i),e) for i in range(0,count,128))
                    owners.extend([e]*count)
                    offset+=count
                x = (torch.arange(n*5120,device=device,dtype=torch.float32).reshape(n,5120)*0.003).sin().contiguous()
                xt=x.index_select(0,ind(source_rows)).half()
                ref=torch.zeros_like(x)
                row_exp=scalars(owners,torch.int32)
                starts=ind([a for a,_,_ in jobs]).int()
                counts=ind([b for _,b,_ in jobs]).int()
                experts=ind([e for _,_,e in jobs])
                scratch_ref = [torch.empty((n,cols),device=device,dtype=torch.float16) for cols in [5120,5120,2304,2304]]
                reference.do_moe(xt,ref,tok,wt,row_exp,starts,counts,experts,*planar,*scratch_ref,3,2,10.,0)
                js=(Job*len(jobs))(*(Job(slots[e],a,b) for a,b,e in jobs))
                routes=(RoutedRow*n)(*(RoutedRow(slots[e],source_rows[i],destinations[i],float(1+i%7)/8) for i,e in enumerate(owners)))
                expected = torch.full((2*n,5120), -12345., dtype=torch.float32, device=device)
                expected.index_copy_(0,ind(destinations),ref)
                # Mirror the native direct-peer case, preserving untouched output rows.
                for owner in (device,1-device):
                    if owner != device:
                        status = cudart.cudaDeviceEnablePeerAccess(owner,0)
                        if status == 704:  # cudaErrorPeerAccessAlreadyEnabled
                            cudart.cudaGetLastError()
                        else:
                            check(status)
                    out = torch.full((2*n,5120), -12345., dtype=torch.float32, device=owner)
                    source = x.to(owner)
                    torch.cuda.synchronize(owner)
                    torch.cuda.synchronize(device)
                    check(lib.ds_moe_run_dev(C.byref(scratch),C.byref(size),js,len(js),routes,n,source.data_ptr(),out.data_ptr(),0,2,1,3))
                    torch.cuda.synchronize(device)
                    actual = out.to(device)
                    equal = torch.equal(actual.view(torch.int32),expected.view(torch.int32))
                    print(json.dumps(dict(device=device,output_owner=owner,layer=layer,rows=n,expert_rows=runs,bit_equal=equal,max_abs=(actual-expected).abs().max().item())),flush=True)
                    if not equal or not torch.isfinite(actual).all():
                        raise ValueError("adapter and packaged kernel differ")
        finally:
            lib.ds_moe_free(device,scratch)
            for slot in slots:lib.ds_moe_free(device,slot)
if __name__=="__main__":
    main()
