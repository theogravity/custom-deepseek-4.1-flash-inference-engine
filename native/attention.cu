#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
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

    cudaError_t grow(size_t bytes, size_t preserve = 0, bool exact = false) {
        if (bytes <= capacity) return cudaSuccess;
        size_t n = exact ? bytes : std::max(bytes, capacity + capacity / 2);
        void *next = nullptr;
        cudaError_t rc = cudaMalloc(&next, n);
        if (rc != cudaSuccess) {
            // An allocation failure is recoverable for another live session.
            // Consume only the matching last error to keep later launches clean.
            if (rc == cudaErrorMemoryAllocation && cudaPeekAtLastError() == rc) cudaGetLastError();
            return rc;
        }
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
template<bool RoundInput>
__global__ void normalize_rows(float *x, const float *gamma, int width) {
    extern __shared__ float squares[];
    __shared__ float inv;
    float *row = x + size_t(blockIdx.x) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const float v = RoundInput ? bf16(row[j]) : row[j];
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
        row[j] = bf16(__fmul_rn(__fmul_rn(RoundInput ? bf16(row[j]) : row[j], inv), gamma[j]));
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

// Match the checkpoint's record layout. One warp quantizes a 128-column
// index key or a 64-column non-rotary key/value group.
__global__ void pack_compressor(const float *values,const float *keys,uint8_t *value_bytes,uint8_t *key_bytes) {
    int row=blockIdx.x, warp=threadIdx.x/32, lane=threadIdx.x%32;
    auto *value=value_bytes+size_t(row)*584;
    auto *index=key_bytes+size_t(row)*132;
    const float *src=warp==7 ? keys+size_t(row)*128 : values+size_t(row)*512+warp*64;
    int width=warp==7 ? 128 : 64;
    float peak=1e-4f;
    for(int j=lane;j<width;j+=32) peak=fmaxf(peak,fabsf(src[j]));
    for(int d=16;d;d/=2) peak=fmaxf(peak,__shfl_xor_sync(0xffffffffu,peak,d));
    unsigned bits=(__float_as_uint(__fmul_rn(peak,1.f/448.f))+0x7fffffu)&0x7f800000u;
    float scale=__uint_as_float(bits);
    uint8_t *dst=warp==7 ? index : value+warp*64;
    for(int j=lane;j<width;j+=32) dst[j]=__nv_fp8_e4m3(__fdiv_rn(src[j],scale)).__x;
    if (!lane) {
        if (warp==7) { *reinterpret_cast<float *>(index+128)=scale; value[583]=0; }
        else value[576+warp]=bits>>23;
    }
    if(threadIdx.x<64) reinterpret_cast<__nv_bfloat16 *>(value+448)[threadIdx.x]=
        __float2bfloat16_rn(values[size_t(row)*512+448+threadIdx.x]);
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
#include "attention_score.cuh"
#include "attention_index.cuh"
} // namespace

extern "C" void *ds4_attn_buffer_new() { return new(std::nothrow) Buffer; }
extern "C" void ds4_attn_buffer_free(int dev, void *buffer) {
    int prior;
    if (cudaGetDevice(&prior) != cudaSuccess) return;
    if (cudaSetDevice(dev) != cudaSuccess) return;
    delete static_cast<Buffer *>(buffer);
    if (prior != dev) cudaSetDevice(prior);
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
    normalize_rows<true><<<batch,256,width * sizeof(float)>>>(x, gamma, width);
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

struct FastWorkspace {
    Layout layout;
    int tiles, cols;
    size_t ko, so, po, mo, no;
    FastWorkspace(int span, int width, int batch) {
        tiles = (span + width + 63) / 64;
        cols = tiles * 64;
        const size_t nh = size_t(batch)*heads;
        ko = layout.slice<__nv_bfloat16>(size_t(batch)*cols*dim);
        so = layout.slice<float>(nh*cols);
        po = layout.slice<__nv_bfloat16>(nh*cols);
        mo = layout.slice<float>(nh*tiles);
        no = layout.slice<float>(nh);
    }
};
extern "C" size_t ds4_attn_sparse_bytes(int fast, int span, int width, int batch) {
    return fast ? FastWorkspace(span,width,batch).layout.used : size_t(batch)*width*dim*sizeof(__nv_bfloat16);
}
extern "C" int ds4_attn_sparse_fast(int dev, void *scratch, const float *queries, const uint16_t *prior,
        const uint16_t *current, int pos, int tile, int span, int decode, const void *cache,
        const int *ids, int width, const float *sinks, int batch, float *out) {
    TRY(cudaSetDevice(dev));
    auto *buf = static_cast<Buffer *>(scratch);
    FastWorkspace work(span,width,batch);
    int cols=work.cols, tiles=work.tiles;
    TRY(buf->grow(work.layout.used));
    auto *keys = at<__nv_bfloat16>(buf,work.ko), *probs = at<__nv_bfloat16>(buf,work.po);
    auto *scores = at<float>(buf,work.so), *scales = at<float>(buf,work.mo), *norm = at<float>(buf,work.no);
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
        const float *weights, const int *ids, int width, int batch, float *out, int packed_scores) {
    TRY(cudaSetDevice(dev));
    auto *b = static_cast<Buffer *>(scratch);
    Layout l;
    const size_t ns = size_t(batch) * width;
    auto io=l.slice<int>(ids ? ns : 0), so=l.slice<float>(ns);
    TRY(b->grow(l.used));
    if (ids) TRY(cudaMemcpy(at<int>(b,io),ids,ns*4,cudaMemcpyHostToDevice));
    auto *keys = static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
    launch_index_scores(packed_scores,keys,queries,weights,ids ? at<int>(b,io) : nullptr,
                        width,batch,at<float>(b,so));
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(out,at<float>(b,so),ns*4,cudaMemcpyDeviceToHost));
    return 0;
}

// The Rust caller reserves the full next allocation before growth. Both copies
// coexist until the preserved prefix has moved to the larger allocation.
extern "C" size_t ds4_attn_capacity(void *handle) {
    return static_cast<const Buffer *>(handle)->capacity;
}
extern "C" size_t ds4_attn_growth(void *handle, size_t request) {
    auto capacity=static_cast<const Buffer *>(handle)->capacity;
    return request <= capacity ? 0 : std::max(request,capacity+capacity/2);
}

extern "C" int ds4_attn_index_bytes(int width, int rows, size_t *bytes) {
    IndexWorkspace s;
    TRY(s.plan(width,rows));
    *bytes=s.bytes;
    return 0;
}

extern "C" int ds4_attn_index_ids(int dev, int *out, int width, int pos, int ratio, int rows) {
    TRY(cudaSetDevice(dev));
    index_ids<<<dim3((width+block-1)/block,rows),block>>>(nullptr,out,width,pos,ratio);
    return cudaGetLastError();
}

extern "C" int ds4_attn_index_select(int dev, uint8_t *work, const void *cache, const float *queries,
        const float *weights, const int *candidates, int width, int pos, int ratio, int rows,
        int *picks, int *out_blocks, int packed_scores) {
    TRY(cudaSetDevice(dev));
    IndexWorkspace s;
    TRY(s.plan(width,rows));
    auto *ids=reinterpret_cast<int *>(work+s.ids);
    auto *scores=reinterpret_cast<float *>(work+s.scores);
    auto *rank=reinterpret_cast<unsigned long long *>(work+s.ranks);
    auto *sorted=reinterpret_cast<unsigned long long *>(work+s.sorted);
    auto *offsets=reinterpret_cast<int *>(work+s.offsets);
    auto *packed=static_cast<const uint8_t *>(static_cast<const Buffer *>(cache)->ptr);
    dim3 grid((width+block-1)/block,rows);
    index_ids<<<grid,block>>>(candidates,ids,width,pos,ratio);
    TRY(cudaGetLastError());
    launch_index_scores(packed_scores,packed,queries,weights,ids,width,rows,scores);
    TRY(cudaGetLastError());
    auto sort=[&](int stride)->cudaError_t {
        if (!wide_index_sort(stride,rows)) {
            index_bounds<<<(rows+block)/block,block>>>(offsets,stride,rows);
            auto e=cudaGetLastError();
            if (e!=cudaSuccess) return e;
        }
        size_t available=s.temp_bytes;
        return sort_index_rows(work+s.temp,available,rank,sorted,stride,rows,offsets);
    };
    rank_positions<<<grid,block>>>(scores,ids,rank,width);
    TRY(cudaGetLastError());
    TRY(sort(width));
    chronological_ids<top><<<rows,block>>>(sorted,width,picks);
    TRY(cudaGetLastError());
    if(out_blocks) {
        int count=(width+index_block-1)/index_block;
        rank_blocks<<<dim3((count+block-1)/block,rows),block>>>(scores,rank,width,pos,ratio);
        TRY(cudaGetLastError());
        TRY(sort(count));
        chronological_ids<candidate_limit><<<rows,block>>>(sorted,count,out_blocks);
        TRY(cudaGetLastError());
    }
    return 0;
}

extern "C" int ds4_attn_normalize_raw(int dev, float *x, const float *g, int n) {
    TRY(cudaSetDevice(dev));
    normalize_rows<false><<<n,block,dim*sizeof(float)>>>(x,g,dim);
    return cudaGetLastError();
}
extern "C" int ds4_attn_prepare_tail(int dev,void *cache,size_t old,size_t bytes) {
    TRY(cudaSetDevice(dev));
    return static_cast<Buffer *>(cache)->grow(old+bytes,old,true);
}
extern "C" int ds4_attn_pack_compressed(int dev,const float *values,const float *keys,
        uint8_t *value_bytes,uint8_t *key_bytes,int n) {
    TRY(cudaSetDevice(dev));
    pack_compressor<<<n,block>>>(values,keys,value_bytes,key_bytes);
    return cudaGetLastError();
}

extern "C" int ds4_attn_zero(int dev,float *dst,size_t len) {
    TRY(cudaSetDevice(dev));
    return cudaMemsetAsync(dst,0,len*sizeof(float));
}
