#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
constexpr int dim = 512, heads = 64, window = 128, tile = 64, threads = 256;

__global__ void store(const float *kv, __nv_bfloat16 *ring, int pos) {
    const int row = blockIdx.x;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        ring[((pos + row) % window) * dim + j] = __float2bfloat16_rn(kv[row * dim + j]);
}

__device__ float warp_sum(float v) {
    for (int offset = 16; offset; offset /= 2)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// The checkpoint scans the ring in physical slot order, then all the fresh
// positions of the noise block. A prefix restore may leave a shorter valid span.
__device__ float value(const __nv_bfloat16 *ring, const float *fresh, int start,
                       int valid, int row, int column) {
    if (row >= valid)
        return fresh[(row - valid) * dim + column];
    int wrapped = max(0, start + valid - window);
    int slot = row < wrapped ? row : start + row - wrapped;
    return __bfloat162float(ring[slot * dim + column]);
}

// One CTA per query/head. The 64-key tiles mirror the reference's online
// softmax: probabilities are rounded to BF16 only for the value multiplication.
__global__ void attend(const float *q, const __nv_bfloat16 *ring, const float *fresh,
                       int start, int valid, int width, const float *sink, float *out) {
    int head = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int offset = (blockIdx.y * heads + head) * dim;
    __shared__ float query[dim], scores[tile], probs[tile], old_scale, max_score, denom;
    query[tid] = q[offset + tid];
    query[tid + threads] = q[offset + tid + threads];
    if (tid == 0) {
        max_score = -1e30f;
        denom = 0;
    }
    __syncthreads();
    float a = 0, b = 0;
    for (int base = 0; base < valid + width; base += tile) {
        const int n = min(tile, valid + width - base);
        for (int j = warp; j < n; j += threads / 32) {
            float dot = 0;
            for (int c = lane; c < dim; c += 32)
                dot = fmaf(query[c], value(ring, fresh, start, valid, base + j, c), dot);
            dot = warp_sum(dot);
            if (lane == 0) scores[j] = dot * 0.04419417382415922f; // 1 / sqrt(512)
        }
        __syncthreads();
        if (tid == 0) {
            float next = max_score;
            for (int j = 0; j < n; ++j) next = fmaxf(next, scores[j]);
            old_scale = expf(max_score - next);
            max_score = next;
        }
        __syncthreads();
        if (tid < n) {
            float p = expf(scores[tid] - max_score);
            scores[tid] = p;
            probs[tid] = __bfloat162float(__float2bfloat16_rn(p));
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0;
            for (int j = 0; j < n; ++j) sum += scores[j];
            denom = fmaf(denom, old_scale, sum);
        }
        a *= old_scale;
        b *= old_scale;
        for (int j = 0; j < n; ++j) {
            float p = probs[j];
            a = fmaf(p, value(ring, fresh, start, valid, base+j, tid), a);
            b = fmaf(p, value(ring, fresh, start, valid, base+j, tid+threads), b);
        }
        __syncthreads();
    }
    if (tid == 0) denom += expf(sink[head] - max_score);
    __syncthreads();
    out[offset + tid] = __bfloat162float(__float2bfloat16_rn(a / denom));
    out[offset + tid + threads] = __bfloat162float(__float2bfloat16_rn(b / denom));
}
} // namespace

extern "C" int ds_draft_store(const float *kv, void *ring, int pos, int rows) {
    store<<<rows, threads>>>(kv, static_cast<__nv_bfloat16 *>(ring), pos);
    return int(cudaGetLastError());
}
extern "C" int ds_draft_attend(const float *q, const void *ring, const float *fresh,
                               int start, int valid, int width, const float *sink, float *out) {
    attend<<<dim3(heads,width), threads>>>(q, static_cast<const __nv_bfloat16 *>(ring), fresh,
                                          start, valid, width, sink, out);
    return int(cudaGetLastError());
}
