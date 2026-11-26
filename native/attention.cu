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
// Retains up to 129 tentative rows alongside the 127 committed rows needed by attention.
constexpr int ring_rows = 2 * win;
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

__device__ float bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// The CPU reference accumulates squares in feature order without contraction.
// Compute the products in parallel, then preserve that order for the single sum.
__global__ void normalize_rows(float *x, const float *gamma, int width) {
    extern __shared__ float squares[];
    __shared__ float inv;
    float *row = x + size_t(blockIdx.x) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const float v = bf16(row[j]);
        squares[j] = __fmul_rn(v, v);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int j = 0; j < width; ++j) sum = __fadd_rn(sum, squares[j]);
        inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(width)), 1e-20f)));
    }
    __syncthreads();
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        row[j] = bf16(__fmul_rn(__fmul_rn(bf16(row[j]), inv), gamma[j]));
}

// One block per head. The sine/cosine pairs come from the same host sin_cos
// implementation as the compressed source. Quantization groups span whole warps.
__global__ void transform_heads(float *data, __nv_bfloat16 *window, const float2 *freqs,
                                int nheads, int cols, int group, bool inverse) {
    __shared__ float row[dim];
    const size_t off = size_t(blockIdx.x) * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) row[j] = bf16(data[off + j]);
    __syncthreads();
    if (threadIdx.x < 32) {
        const int k = cols - 64 + threadIdx.x * 2;
        const float2 f = freqs[size_t(blockIdx.x / nheads) * 32 + threadIdx.x];
        const float s = inverse ? -f.x : f.x;
        const float a = row[k], b = row[k + 1];
        row[k] = bf16(__fsub_rn(__fmul_rn(a, f.y), __fmul_rn(b, s)));
        row[k + 1] = bf16(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, f.y)));
    }
    __syncthreads();
    const int quant_cols = window ? cols - 64 : cols;
    if (group) {
        const int lane = threadIdx.x % 32;
        for (int start = (threadIdx.x / 32) * group; start < quant_cols;
             start += (blockDim.x / 32) * group) {
            float peak = 1e-4f;
            for (int j = lane; j < group; j += 32) peak = fmaxf(peak, fabsf(row[start+j]));
            for (int d = 16; d; d /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
            const unsigned bits = __float_as_uint(__fmul_rn(peak, 1.f / 448.f));
            const float scale = __uint_as_float((bits + 0x7fffffu) & 0x7f800000u);
            for (int j = lane; j < group; j += 32) {
                const float v = __fdiv_rn(row[start+j], scale);
                row[start+j] = float(__nv_fp8_e4m3(v)) * scale;
            }
        }
        __syncthreads();
    }
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        if (window) window[off+j] = __float2bfloat16_rn(row[j]);
        else data[off+j] = row[j];
    }
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

__device__ const __nv_bfloat16 *window_row(const __nv_bfloat16 *prior, const __nv_bfloat16 *current,
                                        int pos, int token) {
    return token < pos ? prior + (token % ring_rows) * dim : current + size_t(token - pos) * dim;
}

__global__ void attend(const float *queries, const __nv_bfloat16 *prior,
                       const __nv_bfloat16 *current, int pos, int tile, int span, bool decode,
                       const __nv_bfloat16 *selected, const int *picks, int width,
                       const float *sink, float *out) {
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int lane = tid % 32, warp = tid / 32;
    const int end = pos + tile + t + 1;
    const int nwindow = min(win, end);
    // The reference reserves window columns before compressed columns, even
    // while the ring is filling. Prefill pads on the right. Decode traverses
    // the ring oldest-first, putting unfilled slots on the left.
    const int first = decode ? span - nwindow : 0;
    const int n = span + width;
    const __nv_bfloat16 *c = width ? selected + size_t(t) * width * dim : nullptr;
    const size_t qoff = (size_t(t) * heads + h) * dim;
    __shared__ float q[dim], scores[win + top], probs[win + top], divisor;
    q[tid] = queries[qoff + tid]; q[tid + block] = queries[qoff + tid + block];
    __syncthreads();
    for (int p = warp; p < n; p += block / 32) {
        float dot = 0.f;
        bool valid = (p >= first && p < first + nwindow) || (p >= span && picks[t * width + p - span] >= 0);
        if (valid) {
            const __nv_bfloat16 *k = p < span ? window_row(prior, current, pos, end - nwindow + p - first) : c + (p-span) * dim;
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
        const __nv_bfloat16 *v = p < span ? window_row(prior, current, pos, end - nwindow + p - first) : c + (p-span) * dim;
        a = fmaf(probs[p], __bfloat162float(v[tid]), a);
        b = fmaf(probs[p], __bfloat162float(v[tid + block]), b);
    }
    out[qoff + tid] = __bfloat162float(__float2bfloat16(a / divisor));
    out[qoff + tid + block] = __bfloat162float(__float2bfloat16(b / divisor));
}

#include "attention_fast.cuh"

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
            score = fmaf(fmaxf(0, dot), bf16(bf16(weights[t * ih + h]) / 64.f), score);
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

extern "C" void *ds4_attn_data(void *handle) {
    return static_cast<Buffer *>(handle)->ptr;
}

extern "C" int ds4_attn_read(int dev, const void *src, void *dst, size_t bytes) {
    TRY(cudaSetDevice(dev));
    TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// The persistent BF16 window is indexed by absolute token position modulo ring_rows.
// Only its live suffix is serialized. Prefill chunks may exceed ring capacity.
extern "C" int ds4_attn_window(int dev, void *handle, int start, int rows,
                               void *data, cudaMemcpyKind kind) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(handle);
    constexpr size_t row_bytes = dim * sizeof(__nv_bfloat16);
    TRY(b->grow(ring_rows * row_bytes));
    auto *ring = at<uint8_t>(b, 0);
    auto *chunk = static_cast<uint8_t *>(data);
    const int slot = start % ring_rows;
    const int first = std::min(rows, ring_rows - slot);
    auto copy = [&](int offset, int slot, int n) {
        if (!n) return cudaSuccess;
        void *r = ring + slot * row_bytes, *c = chunk + offset * row_bytes;
        if (kind == cudaMemcpyDeviceToHost) return cudaMemcpy(c, r, n * row_bytes, kind);
        if (kind == cudaMemcpyDeviceToDevice) return cudaMemcpyAsync(r, c, n * row_bytes, kind);
        return cudaMemcpy(r, c, n * row_bytes, kind);
    };
    TRY(copy(0, slot, first));
    TRY(copy(first, 0, rows - first));
    return 0;
}

extern "C" int ds4_attn_normalize(int dev, float *x, const float *gamma, int width, int batch) {
    TRY(cudaSetDevice(dev));
    if (width < 1 || width > 5120) return (int)cudaErrorInvalidValue;
    normalize_rows<<<batch,256,width * sizeof(float)>>>(x, gamma, width);
    return (int)cudaGetLastError();
}

extern "C" int ds4_attn_transform(int dev, float *x, uint16_t *window, const float *freqs,
                                   int nheads, int cols, int group, int inverse, int batch) {
    TRY(cudaSetDevice(dev));
    transform_heads<<<batch * nheads,128>>>(x, reinterpret_cast<__nv_bfloat16 *>(window),
        reinterpret_cast<const float2 *>(freqs), nheads, cols, group, inverse != 0);
    return (int)cudaGetLastError();
}

extern "C" int ds4_attn_sparse(int dev, void *scratch, const float *queries, const uint16_t *prior,
        const uint16_t *current, int pos, int tile, int span, int decode, const void *cache,
        const int *ids, int width, const float *sinks, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(scratch);
    const size_t np = size_t(batch) * width;
    __nv_bfloat16 *selected = nullptr;
    if (np) {
        TRY(b->grow(np * dim * sizeof(__nv_bfloat16)));
        selected = at<__nv_bfloat16>(b, 0);
        const auto *kv = static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
        gather<<<unsigned(np),block>>>(kv, ids, width, selected, int(np));
        TRY(cudaGetLastError());
    }
    attend<<<dim3(heads,batch),block>>>(queries, reinterpret_cast<const __nv_bfloat16 *>(prior),
        reinterpret_cast<const __nv_bfloat16 *>(current), pos, tile, span, decode != 0,
        selected, ids, width, sinks, out);
    return (int)cudaGetLastError();
}

extern "C" int ds4_attn_sparse_fast(int dev, void *scratch, const float *queries, const uint16_t *prior,
        const uint16_t *current, int pos, int tile, int span, int decode, const void *cache,
        const int *ids, int width, const float *sinks, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *buf = static_cast<Buffer *>(scratch);
    int tiles = (span + width + 63) / 64, cols = tiles * 64;
    Layout l;
    size_t nh = size_t(batch)*heads;
    auto ko = l.slice<__nv_bfloat16>(size_t(batch)*cols*dim);
    auto so = l.slice<float>(nh*cols), po = l.slice<__nv_bfloat16>(nh*cols);
    auto mo = l.slice<float>(nh*tiles), no = l.slice<float>(nh);
    TRY(buf->grow(l.used));
    auto *keys = at<__nv_bfloat16>(buf,ko), *probs = at<__nv_bfloat16>(buf,po);
    auto *scores = at<float>(buf,so), *scales = at<float>(buf,mo), *norm = at<float>(buf,no);
    const auto *kv = width ? static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr) : nullptr;
    fast_keys<<<dim3(cols,batch),block>>>(reinterpret_cast<const __nv_bfloat16 *>(prior),
        reinterpret_cast<const __nv_bfloat16 *>(current),kv,ids,width,pos,tile,span,decode,keys);
    TRY(cudaGetLastError());
    fast_qk<<<dim3(tiles,heads/16,batch),block>>>(queries,keys,cols,scores);
    TRY(cudaGetLastError());
    fast_softmax<<<dim3(heads,batch),block>>>(scores,ids,width,pos,tile,span,decode,cols,sinks,probs,scales,norm);
    TRY(cudaGetLastError());
    fast_pv<<<dim3(dim/64,heads/16,batch),block>>>(keys,probs,scales,norm,cols,out);
    return (int)cudaGetLastError();
}

extern "C" int ds4_attn_scores(int dev, void *scratch, const void *cache, const float *queries,
        const float *weights, const int *ids, int width, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(scratch);
    Layout l;
    const size_t ns = size_t(batch) * width;
    auto io=l.slice<int>(ids ? ns : 0), so=l.slice<float>(ns);
    TRY(b->grow(l.used));
    if (ids) TRY(cudaMemcpy(at<int>(b,io),ids,ns*4,cudaMemcpyHostToDevice));
    auto *keys = static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
    score_keys<<<dim3((width+block/32-1)/(block/32),batch),block>>>(keys,queries,
        weights,ids ? at<int>(b,io) : nullptr,width,at<float>(b,so));
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(out,at<float>(b,so),ns*4,cudaMemcpyDeviceToHost));
    return 0;
}
