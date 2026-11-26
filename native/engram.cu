// Decode only the 24 packed n-gram rows gathered for each input token.
// Each row is 128 bytes of int4, followed by eight IEEE FP16 scales.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int dim = 5120, tile = 512, channels = 4, row_bytes = 144;

__global__ void unpack(const uint8_t *rows, float *out) {
    const int j = threadIdx.x;
    const uint8_t *row = rows + size_t(blockIdx.x) * row_bytes;
    int value = (row[j / 2] >> (4 * (j % 2)) & 15) - 8;
    const auto *scales = reinterpret_cast<const __half *>(row + 128);
    out[size_t(blockIdx.x) * 256 + j] =
        __bfloat162float(__float2bfloat16_rn(float(value) * __half2float(scales[j / 32])));
}

// Preserve the independent, ordered FP32 sums of the scalar implementation.
// Three lanes sum each shared tile while the full block calculates products.
__global__ void apply(float *states, const float *kv, const float *q, const float *k) {
    const int tid = threadIdx.x, c = blockIdx.x, t = blockIdx.y;
    float *h = states + (size_t(t) * channels + c) * dim;
    const float *key = kv + (size_t(t) * (channels + 1) + c) * dim;
    const float *value = kv + (size_t(t) * (channels + 1) + channels) * dim;
    q += c * dim;
    k += c * dim;
    __shared__ float products[3][tile+1], sums[3], gate;
    float sum = 0.f;
    for (int start = 0; start < dim; start += tile) {
        for (int d = tid; d < tile; d += blockDim.x) {
            const int i = start + d;
            products[0][d] = __fmul_rn(h[i], h[i]);
            products[1][d] = __fmul_rn(key[i], key[i]);
            products[2][d] = __fmul_rn(__fmul_rn(h[i], __fmul_rn(q[i], k[i])), key[i]);
        }
        __syncthreads();
        if (tid < 3) {
#pragma unroll 16
            for (int d = 0; d < tile; ++d) sum = __fadd_rn(sum, products[tid][d]);
        }
        __syncthreads();
    }
    if (tid < 3) sums[tid] = sum;
    __syncthreads();
    if (!tid) {
        float hh = __fadd_rn(__fdiv_rn(sums[0], float(dim)), 1e-20f);
        float kk = __fadd_rn(__fdiv_rn(sums[1], float(dim)), 1e-20f);
        float dot = __fdiv_rn(sums[2], __fsqrt_rn(__fmul_rn(__fmul_rn(hh, kk), float(dim))));
        float arg = -copysignf(__fsqrt_rn(fmaxf(fabsf(dot), 1e-6f)), dot);
        gate = __fdiv_rn(1.f, __fadd_rn(1.f, float(exp(double(arg)))));
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) {
        float updated = __fadd_rn(h[i], __fmul_rn(gate, value[i]));
        h[i] = __bfloat162float(__float2bfloat16_rn(updated));
    }
}
}

extern "C" int ds_engram_unpack(int device, const uint8_t *host, uint8_t *packed,
                                float *out, int rows) {
    auto e = cudaSetDevice(device);
    if (e != cudaSuccess) return e;
    e = cudaMemcpy(packed, host, size_t(rows) * row_bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return e;
    unpack<<<rows, 256>>>(packed, out);
    return cudaGetLastError();
}

extern "C" int ds_engram_apply(int device, float *h, const float *kv, const float *q,
                               const float *k, int batch) {
    auto e = cudaSetDevice(device);
    if (e != cudaSuccess) return e;
    apply<<<dim3(channels, batch), 256>>>(h, kv, q, k);
    return cudaGetLastError();
}
