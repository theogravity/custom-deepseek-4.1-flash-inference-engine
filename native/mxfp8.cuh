#pragma once
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Store the four FP8 pairs that feed the A operand of m16n8k16 together
// for each lane. Adjacent warps share one 64-output tile.
__global__ void mx_retile(const unsigned char *src, uint2 *dst, int rows,int cols) {
    const int warp=threadIdx.x/32, lane=threadIdx.x%32;
    const int n=blockIdx.x*64+warp*16+lane/4;
    const int k=blockIdx.y*16+2*(lane%4);
    unsigned short a[4];
    a[0]=*reinterpret_cast<const unsigned short*>(src+size_t(n)*cols+k);
    a[1]=*reinterpret_cast<const unsigned short*>(src+size_t(n+8)*cols+k);
    a[2]=*reinterpret_cast<const unsigned short*>(src+size_t(n)*cols+k+8);
    a[3]=*reinterpret_cast<const unsigned short*>(src+size_t(n+8)*cols+k+8);
    const size_t i=(size_t(blockIdx.x)*(cols/16)+blockIdx.y)*128+threadIdx.x;
    dst[i]=make_uint2((unsigned(a[1])<<16)|a[0],(unsigned(a[3])<<16)|a[2]);
}
__device__ __forceinline__ unsigned mx_bf_pair(unsigned short q, __nv_bfloat162 scale) {
    const auto raw=__nv_cvt_fp8x2_to_bf162raw(q,__NV_E4M3);
    __nv_bfloat162_raw pair=__hmul2(__nv_bfloat162(raw),scale);
    return (unsigned(pair.y)<<16)|pair.x;
}
template<bool PerRowScale, int TokenTiles = 1>
__global__ void mx_project(const uint2 *weights,const unsigned char *scales,
        const __nv_bfloat16 *input,float *partial,int rows,int cols,int batch,int groups,int splits) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int row=blockIdx.x*64+warp*16;
    int group=row/(rows/groups);
    const int first_token = blockIdx.y * (8 * TokenTiles);
    int kt=cols/16;
    int kbeg=(blockIdx.z*kt/splits),kend=((blockIdx.z+1)*kt/splits);
    float c[TokenTiles][4] = {};
    for(int tile=kbeg;tile<kend;++tile) {
        uint2 w=weights[(size_t(blockIdx.x)*kt+tile)*128+threadIdx.x];
        const int scale_row=PerRowScale ? row+lane/4 : row/32;
        unsigned exp=scales[scale_row*(cols/32)+tile/2];
        __nv_bfloat16 scale=__ushort_as_bfloat16(exp==0 ? 0x40u : exp<<7);
        __nv_bfloat162 s=__halves2bfloat162(scale,scale), s8=s;
        if constexpr(PerRowScale) {
            unsigned exp8=scales[(scale_row+8)*(cols/32)+tile/2];
            __nv_bfloat16 v8=__ushort_as_bfloat16(exp8==0 ? 0x40u : exp8<<7);
            s8=__halves2bfloat162(v8,v8);
        }
        unsigned a0=mx_bf_pair(w.x,s),a1=mx_bf_pair(w.x>>16,s8);
        unsigned a2=mx_bf_pair(w.y,s),a3=mx_bf_pair(w.y>>16,s8);
        // Separate accumulators share the expanded weights. Each token retains
        // the same MMA instruction sequence and split-K reduction order.
#pragma unroll
        for (int t = 0; t < TokenTiles; ++t) {
            int token = first_token + 8 * t + lane / 4;
            unsigned b0 = 0, b1 = 0;
            if (token < batch) {
                const auto *v = input + (size_t(token) * groups + group) * cols
                                + tile * 16 + 2 * (lane % 4);
                b0 = *reinterpret_cast<const unsigned *>(v);
                b1 = *reinterpret_cast<const unsigned *>(v + 8);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                         : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    row += lane / 4;
#pragma unroll
    for (int t = 0; t < TokenTiles; ++t) {
        const int token = first_token + 8 * t + 2 * (lane % 4);
        if (token < batch) {
            float *out = partial + (size_t(blockIdx.z) * batch + token) * rows + row;
            out[0] = c[t][0];
            out[8] = c[t][2];
            if (token + 1 < batch) {
                out[rows] = c[t][1];
                out[rows + 8] = c[t][3];
            }
        }
    }
}
__global__ void mx_sum(const float *parts,float *result,size_t count,int splits,bool round) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){
        float v=0;
        for(int k=0;k<splits;++k)v+=parts[size_t(k)*count+i];
        result[i]=round?__bfloat162float(__float2bfloat16_rn(v)):v;
    }
}

// Router, HC and compressor weights are small enough to use a deterministic
// split dot product. Splits and FP32 reduction order depend only on K.
template<typename W>
__global__ void stable_dot(const W *weights,const float *x,float *parts,int rows,
                           int cols,int batch,int groups,int splits) {
    constexpr int T=4;
    int row=blockIdx.x,first=blockIdx.y*T,tid=threadIdx.x;
    int group=row/(rows/groups);
    int start=blockIdx.z*1024,end=min(start+1024,cols);
    float acc[T]={};
    for(int j=start+tid;j<end;j+=blockDim.x) {
        float w=float(weights[size_t(row)*cols+j]);
        #pragma unroll
        for(int t=0;t<T;++t) if(first+t<batch) {
            float a=x[(size_t(first+t)*groups+group)*cols+j];
            if constexpr(sizeof(W)==2) a=__bfloat162float(__float2bfloat16_rn(a));
            acc[t]=fmaf(a,w,acc[t]);
        }
    }
    __shared__ float partial[T][4];
    #pragma unroll
    for(int t=0;t<T;++t) {
        for(int d=16;d;d/=2) acc[t]+=__shfl_down_sync(0xffffffff,acc[t],d);
        if((tid&31)==0)partial[t][tid/32]=acc[t];
    }
    __syncthreads();
    if(tid<T && first+tid<batch){
        float v=(partial[tid][0]+partial[tid][1])+(partial[tid][2]+partial[tid][3]);
        parts[(size_t(blockIdx.z)*batch+first+tid)*rows+row]=v;
    }
}
