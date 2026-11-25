#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>

namespace {
constexpr int dim = 512, heads = 64, win = 128, top = 512;
constexpr int ih = 32, idim = 128;
constexpr int block = 256;
constexpr int align = 256;

#define TRY(call) do { cudaError_t rc_ = (call); if (rc_ != cudaSuccess) return (int)rc_; } while (0)

struct Buffer {
    void *ptr = nullptr;
    size_t capacity = 0;
    ~Buffer() { if (ptr) cudaFree(ptr); }

    cudaError_t grow(size_t bytes, size_t preserve = 0) {
        if (bytes <= capacity) return cudaSuccess;
        size_t n = std::max(bytes, capacity + capacity / 2);
        void *next = nullptr;
        cudaError_t rc = cudaMalloc(&next, n);
        if (rc != cudaSuccess) return rc;
        if (preserve) {
            rc = cudaMemcpy(next, ptr, preserve, cudaMemcpyDeviceToDevice);
            if (rc != cudaSuccess) { cudaFree(next); return rc; }
        }
        if (ptr) cudaFree(ptr);
        ptr = next;
        capacity = n;
        return cudaSuccess;
    }
};

// A single scratch allocation per session/device, with 256-byte aligned slices.
struct Layout {
    size_t used = 0;
    template<typename T> size_t slice(size_t count) {
        const size_t offset = used;
        used += (sizeof(T) * count + align - 1) / align * align;
        return offset;
    }
};

template<typename T> T *at(Buffer *b, size_t offset) {
    return reinterpret_cast<T *>(static_cast<char *>(b->ptr) + offset);
}

__device__ float fp8_value(uint8_t code) {
    __nv_fp8_e4m3 v; v.__x = code; return float(v);
}

__device__ float unpack_value(const uint8_t *row, int col) {
    if (col < 448) {
        const float scale = __uint_as_float(unsigned(row[576+col/64]) << 23);
        return fp8_value(row[col])*scale;
    }
    const auto *rope = reinterpret_cast<const __nv_bfloat16 *>(row + 448);
    return __bfloat162float(rope[col-448]);
}

__global__ void gather(const uint8_t *compressed, const int *picks, int width,
                       __nv_bfloat16 *selected, int count) {
    int row = blockIdx.x;
    if (row >= count) return;
    int id = picks[row];
    if (id < 0) return;
    const uint8_t *src = compressed + size_t(id) * 584;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        selected[size_t(row) * dim + j] = __float2bfloat16(unpack_value(src, j));
}

__device__ float warp_sum(float v) {
    for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}

__global__ void attend(const float *queries, const __nv_bfloat16 *window, int history, int span, bool decode,
                       const __nv_bfloat16 *selected, const int *picks, int width,
                       const float *sink, float *out) {
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int lane = tid % 32, warp = tid / 32;
    const int nwindow = min(win, history + t + 1);
    // The reference reserves window columns before compressed columns, even
    // while the ring is filling. Prefill pads on the right. Decode traverses
    // the ring oldest-first, putting unfilled slots on the left.
    const int first = decode ? span - nwindow : 0;
    const int n = span + width;
    const __nv_bfloat16 *w = window + size_t(history + t + 1 - nwindow) * dim;
    const __nv_bfloat16 *c = selected + size_t(t) * width * dim;
    const size_t qoff = (size_t(t) * heads + h) * dim;
    __shared__ float q[dim], scores[win + top], probs[win + top], divisor;
    q[tid] = queries[qoff + tid]; q[tid + block] = queries[qoff + tid + block];
    __syncthreads();
    for (int p = warp; p < n; p += block / 32) {
        float dot = 0.f;
        bool valid = (p >= first && p < first + nwindow) || (p >= span && picks[t * width + p - span] >= 0);
        if (valid) {
            const __nv_bfloat16 *k = p < span ? w + (p-first) * dim : c + (p-span) * dim;
            for (int j = lane; j < dim; j += 32) dot = fmaf(q[j], __bfloat162float(k[j]), dot);
        }
        dot = warp_sum(dot);
        if (lane == 0) scores[p] = valid ? dot * rsqrtf(float(dim)) : -INFINITY;
    }
    __syncthreads();
    // Match the reference's blockwise BF16 probability casts and FP32 denominator.
    // The extra sink is a logit with a zero value vector.
    __shared__ float maxima[(win + top + 63) / 64], final_max;
    if (tid < (n + 63) / 64) {
        float m = -INFINITY;
        for (int p = tid * 64; p < min(n, (tid + 1) * 64); ++p) m = fmaxf(m, scores[p]);
        maxima[tid] = m;
    }
    __syncthreads();
    if (!tid) {
        float m = -1e30f;
        for (int i = 0; i < (n + 63) / 64; ++i) { m = fmaxf(m, maxima[i]); maxima[i] = m; }
        final_max = fmaxf(m, sink[h]);
    }
    __syncthreads();
    float sum = 0;
    for (int p = tid; p < n; p += block) {
        const float max_tile = maxima[p / 64];
        const float e = expf(scores[p] - max_tile);
        probs[p] = __bfloat162float(__float2bfloat16(e)) * expf(max_tile - final_max);
        sum += expf(scores[p] - final_max);
    }
    __shared__ float partial[block / 32];
    sum = warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (!tid) {
        sum = expf(sink[h] - final_max);
        for (int i = 0; i < block / 32; ++i) sum += partial[i];
        divisor = sum;
    }
    __syncthreads();
    float a = 0, b = 0;
    for (int p = 0; p < n; ++p) {
        if (probs[p] == 0) continue;
        const __nv_bfloat16 *v = p < span ? w + (p-first) * dim : c + (p-span) * dim;
        a = fmaf(probs[p], __bfloat162float(v[tid]), a);
        b = fmaf(probs[p], __bfloat162float(v[tid + block]), b);
    }
    out[qoff + tid] = __bfloat162float(__float2bfloat16(a / divisor));
    out[qoff + tid + block] = __bfloat162float(__float2bfloat16(b / divisor));
}

// One warp scores one compressed index position against all 32 query heads.
__global__ void score_keys(const uint8_t *keys, const float *queries, const float *weights,
                           const int *ids, int width, float *scores) {
    const int t = blockIdx.y, lane = threadIdx.x % 32;
    const int slot = blockIdx.x * (block / 32) + threadIdx.x / 32;
    if (slot >= width) return;
    const int k = ids ? ids[t * width + slot] : slot;
    if (k < 0) { if (!lane) scores[t * width + slot] = -INFINITY; return; }
    const uint8_t *row = keys + size_t(k) * 132;
    float key[4];
    for (int j = 0; j < 4; ++j) key[j] = fp8_value(row[lane + 32*j]) * *reinterpret_cast<const float *>(row + 128);
    float score = 0;
    for (int h = 0; h < ih; ++h) {
        const float *q = queries + (size_t(t) * ih + h) * idim;
        float dot = 0;
        for (int j = 0; j < 4; ++j) dot = fmaf(key[j], q[lane + j*32], dot);
        dot = warp_sum(dot);
        if (!lane) {
            score = fmaf(fmaxf(0, dot), weights[t * ih + h], score);
        }
    }
    if (!lane) scores[t * width + slot] = score;
}
} // namespace

extern "C" void *ds4_attn_buffer_new() { return new(std::nothrow) Buffer; }
extern "C" void ds4_attn_buffer_free(int dev, void *buffer) {
    cudaSetDevice(dev); delete static_cast<Buffer *>(buffer);
}
extern "C" int ds4_attn_upload(int dev, void *buffer, const uint8_t *bytes,
                                size_t old, size_t n) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(buffer);
    TRY(b->grow(old + n, old));
    if (n) TRY(cudaMemcpy(at<uint8_t>(b, old), bytes, n, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int ds4_attn_sparse(int dev, void *scratch, const float *queries,
        const uint16_t *window, int history, int span, int decode, const void *cache, const int *ids, int width,
        const float *sinks, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(scratch);
    Layout l;
    const size_t nq = size_t(batch) * heads * dim, nw = size_t(batch + history) * dim;
    const size_t np = size_t(batch) * width;
    auto qo=l.slice<float>(nq), wo=l.slice<uint16_t>(nw), io=l.slice<int>(np);
    auto so=l.slice<float>(heads), co=l.slice<uint16_t>(np * dim), oo=l.slice<float>(nq);
    TRY(b->grow(l.used));
    TRY(cudaMemcpy(at<float>(b,qo),queries,nq*4,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(at<uint16_t>(b,wo),window,nw*2,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(at<float>(b,so),sinks,heads*4,cudaMemcpyHostToDevice));
    if (np) {
        TRY(cudaMemcpy(at<int>(b,io),ids,np*4,cudaMemcpyHostToDevice));
        const auto *kv = static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
        gather<<<unsigned(np),block>>>(kv,at<int>(b,io),width,at<__nv_bfloat16>(b,co),int(np));
        TRY(cudaGetLastError());
    }
    attend<<<dim3(heads,batch),block>>>(at<float>(b,qo),at<__nv_bfloat16>(b,wo),history,span,decode != 0,
        at<__nv_bfloat16>(b,co),at<int>(b,io),width,at<float>(b,so),at<float>(b,oo));
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(out,at<float>(b,oo),nq*4,cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int ds4_attn_scores(int dev, void *scratch, const void *cache, const float *queries,
        const float *weights, const int *ids, int width, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(scratch);
    Layout l;
    const size_t nq = size_t(batch) * ih * idim, nw = size_t(batch) * ih, ns = size_t(batch) * width;
    auto qo=l.slice<float>(nq), wo=l.slice<float>(nw), io=l.slice<int>(ids ? ns : 0), so=l.slice<float>(ns);
    TRY(b->grow(l.used));
    TRY(cudaMemcpy(at<float>(b,qo),queries,nq*4,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(at<float>(b,wo),weights,nw*4,cudaMemcpyHostToDevice));
    if (ids) TRY(cudaMemcpy(at<int>(b,io),ids,ns*4,cudaMemcpyHostToDevice));
    auto *keys = static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
    score_keys<<<dim3((width+block/32-1)/(block/32),batch),block>>>(keys,at<float>(b,qo),
        at<float>(b,wo),ids ? at<int>(b,io) : nullptr,width,at<float>(b,so));
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(out,at<float>(b,so),ns*4,cudaMemcpyDeviceToHost));
    return 0;
}
