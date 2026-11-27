// Block-scaled MXFP8 GEMM for prompt batches. Decode keeps the retiled W8A16 weight.
#pragma once
#include <cublasLt.h>
#include <map>
#include <tuple>

namespace prefill_dense {

static size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

struct Workspace {
    size_t w, sw, x, sx, y, blas, bytes;
    static constexpr size_t algorithm_bytes = 16u << 20;
    Workspace(int outputs, int cols, int tokens) {
        size_t ktiles = (cols + 127) / 128, padded = (tokens + 15) / 16 * 16;
        w = 0;
        sw = align256(size_t(outputs) * cols);
        x = sw + align256(((outputs + 127) / 128) * ktiles * 512);
        sx = x + align256(padded * cols);
        y = sx + align256(((padded + 127) / 128) * ktiles * 512);
        blas = y + align256(padded * outputs * sizeof(float));
        bytes = blas + algorithm_bytes;
    }
};

// NVIDIA 128-row by four-scale tiles. K is the contiguous contraction dimension.
__device__ size_t scale_offset(int row, int slab, int cols) {
    size_t tile = size_t(row / 128) * ((cols + 127) / 128) + slab / 4;
    return tile * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + slab % 4;
}

__global__ void unpack_weight(const uint2 *packed, const uint8_t *scales,
                               uint8_t *plain, uint8_t *swizzled, int cols) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int row = blockIdx.x * 64 + warp * 16 + lane / 4;
    int col = blockIdx.y * 16 + (lane % 4) * 2;
    uint2 q = packed[(size_t(blockIdx.x) * (cols / 16) + blockIdx.y) * 128 + threadIdx.x];
    auto *at = plain + size_t(row) * cols + col;
    *reinterpret_cast<uint16_t *>(at) = uint16_t(q.x);
    *reinterpret_cast<uint16_t *>(at + 8 * cols) = uint16_t(q.x >> 16);
    *reinterpret_cast<uint16_t *>(at + 8) = uint16_t(q.y);
    *reinterpret_cast<uint16_t *>(at + 8 * cols + 8) = uint16_t(q.y >> 16);
    if ((lane % 4) == 0 && (blockIdx.y % 2) == 0) {
        int slab = col / 32;
        uint8_t sf = scales[(row / 32) * (cols / 32) + slab];
        swizzled[scale_offset(row, slab, cols)] = sf;
        swizzled[scale_offset(row + 8, slab, cols)] = sf;
    }
}

__global__ void quantize_input(const float *x, uint8_t *quant, uint8_t *scales,
                               int cols, int tokens, int groups) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    float a = col < cols && row < tokens
        ? __bfloat162float(__float2bfloat16_rn(x[size_t(row) * groups * cols + col])) : 0.f;
    float peak = fabsf(a);
    for (int d = 16; d; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff,peak,d));
    if (col >= cols) return;
    int exponent = int(ceilf(log2f(fmaxf(peak, 0.0001f) / 448.f)));
    quant[size_t(row)*cols+col] = __nv_fp8_e4m3(ldexpf(a,-exponent)).__x;
    if (col % 32 == 0) scales[scale_offset(row,col/32,cols)] = uint8_t(exponent + 127);
}

__global__ void scatter_result(const float *parts, float *output, size_t count,
                               int rows, int group_rows, bool round_output) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float v = parts[i];
        output[(i / group_rows) * rows + i % group_rows] =
            round_output ? __bfloat162float(__float2bfloat16_rn(v)) : v;
    }
}

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool initialized = false;
    ~Plan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
};

struct Executor {
    Buffer scratch;
    cublasLtHandle_t lt = nullptr;
    std::map<std::tuple<int,int,int>,Plan> plans;
    ~Executor() { plans.clear(); if (lt) cublasLtDestroy(lt); }
    int multiply(const void *weight, const float *input, float *out, int rows, int cols,
                 int tokens, int groups, bool round_output) {
        const int group_rows = rows / groups, padded = (tokens+15)/16*16;
        Workspace work(group_rows,cols,tokens);
        if (scratch.resize(work.bytes)) return -1;
        auto *buf = static_cast<uint8_t *>(scratch.ptr);
        void *sa = buf + work.sw, *sb = buf + work.sx;
        auto *result = reinterpret_cast<float *>(buf+work.y);
        if (!lt) BLAS(cublasLtCreate(&lt));
        auto key = std::make_tuple(group_rows,cols,padded);
        auto [it, inserted] = plans.try_emplace(key);
        Plan &p = it->second;
        if (!p.initialized) {
            // A transient initialization failure leaves no partially constructed plan on retry.
            struct Rollback {
                decltype(plans) &cache; decltype(key) id; bool active = true;
                ~Rollback() { if (active) cache.erase(id); }
            } rollback{plans,key};
            BLAS(cublasLtMatmulDescCreate(&p.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
            cublasOperation_t transpose = CUBLAS_OP_T;
            BLAS(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
            auto scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            for (auto attr : {CUBLASLT_MATMUL_DESC_A_SCALE_MODE,CUBLASLT_MATMUL_DESC_B_SCALE_MODE})
                BLAS(cublasLtMatmulDescSetAttribute(p.op,attr,&scale,sizeof(scale)));
            BLAS(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)));
            BLAS(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)));
            BLAS(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_8F_E4M3,cols,group_rows,cols));
            BLAS(cublasLtMatrixLayoutCreate(&p.b,CUDA_R_8F_E4M3,cols,padded,cols));
            BLAS(cublasLtMatrixLayoutCreate(&p.c,CUDA_R_32F,group_rows,padded,group_rows));
            struct Preference {
                cublasLtMatmulPreference_t handle = nullptr;
                ~Preference() { if (handle) cublasLtMatmulPreferenceDestroy(handle); }
            } pref;
            BLAS(cublasLtMatmulPreferenceCreate(&pref.handle));
            size_t bytes = work.algorithm_bytes;
            BLAS(cublasLtMatmulPreferenceSetAttribute(pref.handle,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
            cublasLtMatmulHeuristicResult_t chosen{};
            int count = 0;
            BLAS(cublasLtMatmulAlgoGetHeuristic(lt,p.op,p.a,p.b,p.c,p.c,pref.handle,1,&chosen,&count));
            if (!count) {
                snprintf(error_message,sizeof(error_message),"no MXFP8 GEMM for N=%d K=%d M=%d",group_rows,cols,padded);
                return -1;
            }
            p.algorithm = chosen.algo;
            p.initialized = true;
            rollback.active = false;
        }
        BLAS(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)));
        BLAS(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)));
        float alpha=1.f,beta=0.f;
        size_t block_elements=size_t(group_rows)*cols;
        for (int g=0;g<groups;++g) {
            auto *q=static_cast<const uint8_t *>(weight)+g*block_elements;
            auto *sf=static_cast<const uint8_t *>(weight)+size_t(rows)*cols+size_t(g)*block_elements/1024;
            unpack_weight<<<dim3(group_rows/64,cols/16),128>>>(reinterpret_cast<const uint2 *>(q),sf,buf,buf+work.sw,cols);
            CUDA(cudaGetLastError());
            quantize_input<<<dim3((cols+255)/256,padded),256>>>(input+g*cols,buf+work.x,buf+work.sx,cols,tokens,groups);
            CUDA(cudaGetLastError());
            BLAS(cublasLtMatmul(lt,p.op,&alpha,buf,p.a,buf+work.x,p.b,&beta,result,p.c,result,p.c,
                               &p.algorithm,buf+work.blas,work.algorithm_bytes,nullptr));
            size_t count=size_t(tokens)*group_rows;
            scatter_result<<<(count+255)/256,256>>>(result,out+g*group_rows,count,rows,group_rows,round_output);
            CUDA(cudaGetLastError());
        }
        return 0;
    }
};
} // namespace prefill_dense
