#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <curand_kernel.h>

// Negative IDs propagate nonfinite logits. Equal scores prefer the smaller token.
struct Winner { float value; int id; };
__device__ Winner best(Winner a, Winner b) {
    if (a.id < 0 || b.id < 0) return {0.f,-1};
    return b.value > a.value || (b.value == a.value && b.id < a.id) ? b : a;
}
__device__ Winner warp_best(Winner v) {
    for(int d=16; d; d/=2) {
        Winner n{__shfl_down_sync(0xffffffff,v.value,d),__shfl_down_sync(0xffffffff,v.id,d)};
        v=best(v,n);
    }
    return v;
}
__device__ void block_best(Winner v, Winner *out, float *id_out) {
    __shared__ Winner warps[8];
    v=warp_best(v);
    if ((threadIdx.x&31)==0) warps[threadIdx.x/32]=v;
    __syncthreads();
    if (threadIdx.x<32) {
        v=threadIdx.x<8 ? warps[threadIdx.x] : Winner{-INFINITY,INT_MAX};
        v=warp_best(v);
        if (threadIdx.x==0) {
            if (out) *out=v;
            if (id_out) *id_out=float(v.id);
        }
    }
}
template<bool Sample>
__global__ void score_parts(const float *scores, Winner *parts, int vocab, float temperature, unsigned long long seed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    Winner v{-INFINITY,INT_MAX};
    if (i<vocab) {
        float s=scores[size_t(blockIdx.y)*vocab+i];
        if constexpr(Sample) {
            uint4 bits=curand_Philox4x32_10(make_uint4(i,blockIdx.y,0,0),make_uint2(unsigned(seed),unsigned(seed>>32)));
            // Midpoints of 2^23 bins, strictly inside (0,1) in binary32.
            float u=(float(bits.x>>9)+0.5f)*0x1p-23f;
            float noise=logf(-logf(u));
            s=temperature>=1.f ? s/temperature-noise : s-temperature*noise;
        }
        v={s,isfinite(s)?i:-1};
    }
    block_best(v,&parts[size_t(blockIdx.y)*gridDim.x+blockIdx.x],nullptr);
}
__global__ void finish_parts(const Winner *parts,int count,float *ids) {
    Winner w{-INFINITY,INT_MAX};
    for (int i=threadIdx.x;i<count;i+=blockDim.x)
        w=best(w,parts[size_t(blockIdx.x)*count+i]);
    block_best(w,nullptr,ids+blockIdx.x);
}
extern "C" int ds4_argmax(int device,const float *logits,float *scratch,float *ids,int rows,int cols) {
    auto e=cudaSetDevice(device); if (e!=cudaSuccess) return e;
    int parts=(cols+255)/256;
    score_parts<false><<<dim3(parts,rows),256>>>(logits,(Winner*)scratch,cols,0,0);
    e=cudaGetLastError(); if (e!=cudaSuccess) return e;
    finish_parts<<<rows,256>>>((Winner*)scratch,parts,ids);
    return cudaGetLastError();
}

extern "C" int ds4_sample(int device,const float *logits,float *scratch,float *ids,int rows,int cols,float temperature,unsigned long long seed) {
    auto e=cudaSetDevice(device); if (e!=cudaSuccess) return e;
    int parts=(cols+255)/256;
    score_parts<true><<<dim3(parts,rows),256>>>(logits,(Winner*)scratch,cols,temperature,seed);
    e=cudaGetLastError(); if(e!=cudaSuccess)return e;
    finish_parts<<<rows,256>>>((Winner*)scratch,parts,ids); return cudaGetLastError();
}
// Each warp computes a rank-256 Markov bias for eight vocabulary entries.
// Previous predictions and both BF16 Markov tables stay on the device across K.
template<int Tracks>
__global__ void markov_parts(const __nv_bfloat16 *emb,const __nv_bfloat16 *head,
        const float *base,int vocab,int seed,const float *prior,Winner *out) {
    int source=prior ? int(*prior) : seed;
    if (source<0 || source>=vocab) {
        if (threadIdx.x==0) out[blockIdx.x]={0.f,-1};
        return;
    }
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    float x[8];
    #pragma unroll
    for(int j=0;j<8;++j) x[j]=__bfloat162float(emb[size_t(source)*256+lane+j*32]);
    Winner w{-INFINITY,INT_MAX};
    // Keep independent vocabulary dot products in flight. Every id retains
    // its original eight FMAs and lane reduction; a partial final tile is legal.
    for (int t = 0; t < 8; t += Tracks) {
        float sum[Tracks] = {};
        const int first = blockIdx.x * 64 + t * 8 + warp;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int k = 0; k < Tracks; ++k) {
                int id = first + 8 * k;
                if (id < vocab)
                    sum[k] = fmaf(x[j], __bfloat162float(head[size_t(id)*256 + lane + j*32]), sum[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < Tracks; ++k) {
            int id = first + 8 * k;
            if (id < vocab) {
                for (int d = 16; d; d /= 2)
                    sum[k] += __shfl_down_sync(0xffffffff, sum[k], d);
                if (!lane) {
                    float value = sum[k] + base[id];
                    w = best(w, {value, isfinite(value) ? id : -1});
                }
            }
        }
    }
    block_best(w,out+blockIdx.x,nullptr);
}
extern "C" int ds4_markov(int device,const void *emb,const void *head,const float *base,
        float *scratch,float *ids,int seed,int rows,int vocab,int paired) {
    auto e=cudaSetDevice(device); if (e!=cudaSuccess) return e;
    int parts=(vocab+63)/64;
    for(int i=0;i<rows;++i) {
        if (paired)
            markov_parts<2><<<parts,256>>>((const __nv_bfloat16*)emb,(const __nv_bfloat16*)head,
                base+size_t(i)*vocab,vocab,seed,i ? ids+i-1 : nullptr,(Winner*)scratch);
        else
            markov_parts<1><<<parts,256>>>((const __nv_bfloat16*)emb,(const __nv_bfloat16*)head,
            base+size_t(i)*vocab,vocab,seed,i ? ids+i-1 : nullptr,(Winner*)scratch);
        e=cudaGetLastError(); if (e!=cudaSuccess) return e;
        finish_parts<<<1,256>>>((Winner*)scratch,parts,ids+i);
        e=cudaGetLastError(); if (e!=cudaSuccess) return e;
    }
    return 0;
}
