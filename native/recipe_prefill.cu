#ifdef DS4_RECIPE_PREFILL
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstdint>
#include "moe_types.cuh"
#include "diffbot_device.cuh"
#define CHECK(call) do { auto rc = (call); if (rc != cudaSuccess) return int(rc); } while(0)
// Host adapter for the model pack's S8B (per-row, per-128-column scale) kernel.
// The engine owns allocation, routing, expert slots and the final reduction.
namespace {
constexpr int H = 5120, intermediate = 2304;
template<int I> constexpr size_t trellis_bytes = size_t(H) * I * 3 / 8;
// Physical projection order follows the checkpoint's contiguous w1, w2, w3.
enum Projection { Gate = 0, Down = 1, Up = 2 };
template<int I> __device__ const half *suh(const char *expert, int projection) {
    return reinterpret_cast<const half *>(expert + 3 * trellis_bytes<I>) + projection * (H + I);
}
template<int I> __device__ const half *svh(const char *expert, int projection) {
    return suh<I>(expert, projection) + (projection == Down ? I : H);
}


struct RecipeLayout {
    size_t bytes = 0;
    template<typename T> size_t take(size_t count) {
        size_t offset = bytes;
        bytes += (sizeof(T) * count + 255) / 256 * 256;
        return offset;
    }
};
struct RecipeRows {
    int64_t *tokens, *outputs;
    int *experts;
    half *weights;
    int64_t *gate_in, *up_in, *gate_out, *up_out, *down_in, *down_out;
};
template<int I> __global__ void recipe_rows(const RoutedRow *routes, const float *x,
                            half *input, RecipeRows meta) {
    int r = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    RoutedRow route = routes[r];
    if (col < H) {
        input[size_t(r) * H + col] = __float2half_rn(x[size_t(route.token) * H + col]);
    }
    if (!col) {
        meta.tokens[r] = r;
        meta.outputs[r] = route.output;
        meta.experts[r] = r;
        meta.weights[r] = __float2half_rn(route.weight);
        meta.gate_in[r] = reinterpret_cast<int64_t>(suh<I>(route.expert, Gate));
        meta.up_in[r] = reinterpret_cast<int64_t>(suh<I>(route.expert, Up));
        meta.gate_out[r] = reinterpret_cast<int64_t>(svh<I>(route.expert, Gate));
        meta.up_out[r] = reinterpret_cast<int64_t>(svh<I>(route.expert, Up));
        meta.down_in[r] = reinterpret_cast<int64_t>(suh<I>(route.expert, Down));
        meta.down_out[r] = reinterpret_cast<int64_t>(svh<I>(route.expert, Down));
    }
}
// Every route owns a unique output rank. The recipe uses atomic adds; keep
// those local even when the caller owns the final buffer on the other GPU.
__global__ void recipe_scatter(const float *compact, const int64_t *outputs, float *weighted) {
    int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < H) weighted[size_t(outputs[row])*H + col] = compact[size_t(row)*H + col];
}
template<int I> __global__ void recipe_jobs(const Job *jobs, int n, int *starts, int *counts, int64_t *trellis) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    Job j = jobs[i];
    starts[i] = j.begin;
    counts[i] = j.count;
    for (int projection = 0; projection != 3; ++projection)
        trellis[projection*n+i] = reinterpret_cast<int64_t>(j.expert + projection * trellis_bytes<I>);
}

int recipe_gemm(dof::GemmArgs args) {
    constexpr int smem = 2*dof::stage_bytes<DOF_S8B,3>() + 4*sizeof(dof::ItemSm);
    const auto kernel = dof::dof_gemm_kernel<DOF_S8B,3,2,2,1>;
    int device = 0;
    CHECK(cudaGetDevice(&device));
    static std::atomic<int> sm_count[64]{};
    static std::mutex init;
    if (device < 0 || device >= 64) return int(cudaErrorInvalidDevice);
    int sms = sm_count[device].load(std::memory_order_acquire);
    if (!sms) {
        std::lock_guard<std::mutex> lock(init);
        sms = sm_count[device].load(std::memory_order_relaxed);
        if (!sms) {
            CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            sm_count[device].store(sms, std::memory_order_release);
        }
    }
    CHECK(cudaMemsetAsync(args.ctr, 0, sizeof(int)));
    args.wscale = DOF_WS_MUL1;
    args.never = 0x3fffffff;
    kernel<<<std::min(sms, args.G*args.J*(args.N/128)),256,smem>>>(args);
    return int(cudaGetLastError());
}
} // namespace

template<int I> static int recipe_group(void **scratch, size_t *capacity, const Job *jobs, int nj,
                 const RoutedRow *routes, int rows, const float *x, float *weighted) {
    size_t r = size_t(rows);
    constexpr int hs = H/128, is = I/128;
    RecipeLayout layout;
    auto jobs_off = layout.take<Job>(nj), routes_off = layout.take<RoutedRow>(r);
    auto compact_off = layout.take<float>(r*H);
    auto input_off = layout.take<half>(r*H), activation_off = layout.take<uint8_t>(2*r*H);
    auto inter_off = layout.take<half>(2*r*I), down_off = layout.take<half>(r*H);
    auto scale_off = layout.take<float>(r*(2*hs+is)), weight_off = layout.take<half>(r);
    auto ids_off = layout.take<int>(r), counter_off = layout.take<int>(1);
    auto rowmeta_off = layout.take<int64_t>(8*r);
    auto jobmeta_off = layout.take<int>(2*nj), tr_off = layout.take<int64_t>(3*nj);
    if (*capacity < layout.bytes) {
        CHECK(cudaFree(*scratch));
        *scratch = nullptr; *capacity = 0;
        CHECK(cudaMalloc(scratch, layout.bytes));
        *capacity = layout.bytes;
    }
    auto base = static_cast<char *>(*scratch);
    auto j = reinterpret_cast<Job *>(base+jobs_off);
    auto rr = reinterpret_cast<RoutedRow *>(base+routes_off);
    CHECK(cudaMemcpy(j,jobs,size_t(nj)*sizeof(Job),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(rr,routes,r*sizeof(RoutedRow),cudaMemcpyHostToDevice));
    auto input = reinterpret_cast<half *>(base+input_off);
    auto quant = reinterpret_cast<uint8_t *>(base+activation_off);
    auto inter = reinterpret_cast<half *>(base+inter_off);
    auto down = reinterpret_cast<half *>(base+down_off);
    auto sc = reinterpret_cast<float *>(base+scale_off);
    auto ptrs = reinterpret_cast<int64_t *>(base+rowmeta_off);
    RecipeRows meta{ptrs, ptrs+r, reinterpret_cast<int *>(base+ids_off),
        reinterpret_cast<half *>(base+weight_off), ptrs+2*r,ptrs+3*r,ptrs+4*r,ptrs+5*r,ptrs+6*r,ptrs+7*r};
    auto starts = reinterpret_cast<int *>(base+jobmeta_off);
    auto tr = reinterpret_cast<int64_t *>(base+tr_off);
    recipe_rows<I><<<dim3((H+255)/256,rows),256>>>(rr,x,input,meta);
    CHECK(cudaGetLastError());
    recipe_jobs<I><<<(nj+127)/128,128>>>(j,nj,starts,starts+nj,tr);
    CHECK(cudaGetLastError());
    auto blocks = [&](int slabs) {return (rows*slabs+7)/8;};
    dof::k_gather_blk<<<blocks(hs),256>>>(input,meta.tokens,meta.experts,meta.gate_in,meta.up_in,
            quant,quant+r*H,sc,sc+r*hs,rows,H,true);
    CHECK(cudaGetLastError());
    dof::GemmArgs a{};
    a.job_row0 = starts; a.job_rows = starts+nj; a.R = rows; a.J = nj;
    a.ctr = reinterpret_cast<int *>(base+counter_off);
    a.K=H; a.N=I; a.G=2;
    a.A[0]=quant; a.A[1]=quant+r*H; a.As[0]=sc; a.As[1]=sc+r*hs;
    a.C[0]=inter; a.C[1]=inter+r*I; a.tr[0]=tr+Gate*nj; a.tr[1]=tr+Up*nj;
    int status=recipe_gemm(a); if(status) return status;
    dof::k_guad_blk<<<blocks(is),256>>>(inter,inter+r*I,quant,sc+2*r*hs,meta.experts,
        meta.gate_out,meta.up_out,meta.down_in,rows,I,10.f,MOE_ACT_SILU);
    CHECK(cudaGetLastError());
    a.K=I; a.N=H; a.G=1; a.A[0]=quant; a.As[0]=sc+2*r*hs; a.C[0]=down; a.tr[0]=tr+Down*nj;
    status=recipe_gemm(a); if(status) return status;
    auto compact = reinterpret_cast<float *>(base+compact_off);
    CHECK(cudaMemsetAsync(compact,0,r*H*sizeof(float)));
    dof::k_d_out<<<blocks(hs),256,8*128*sizeof(float)>>>(down,compact,meta.tokens,meta.weights,
        meta.experts,meta.down_out,rows,H);
    CHECK(cudaGetLastError());
    recipe_scatter<<<dim3((H+255)/256,rows),256>>>(compact,meta.outputs,weighted);
    return int(cudaGetLastError());
}

extern "C" int ds_recipe_group(void **scratch, size_t *capacity, const Job *jobs, int nj,
        const RoutedRow *routes, int rows, const float *x, float *weighted) {
    return recipe_group<intermediate>(scratch, capacity, jobs, nj, routes, rows, x, weighted);
}

extern "C" int ds_recipe_rank_group(void **scratch, size_t *capacity, const Job *jobs, int nj,
        const RoutedRow *routes, int rows, const float *x, float *weighted) {
    if (!scratch || !capacity || !jobs || !routes || !x || !weighted ||
        rows <= 0 || rows > 65535 || nj <= 0 || nj > rows) return int(cudaErrorInvalidValue);
    int covered = 0;
    for (int j = 0; j < nj; ++j) {
        const Job &job = jobs[j];
        if (!job.expert || job.begin != covered || job.count <= 0 ||
            job.count > 128 || job.count > rows - covered) return int(cudaErrorInvalidValue);
        covered += job.count;
    }
    if (covered != rows) return int(cudaErrorInvalidValue);
    return recipe_group<intermediate / 2>(scratch, capacity, jobs, nj, routes, rows, x, weighted);
}

// Each rank holds half of the expert intermediate channels in the checkpoint's
// native trellis format. Input scales on gate/up and output scales on down are replicated.
extern "C" size_t ds_recipe_rank_slot_bytes() {
    return 3 * (trellis_bytes<intermediate / 2> + (H + intermediate / 2) * sizeof(half));
}

extern "C" int ds_recipe_upload_rank(void *storage, size_t capacity, const void *const *parts, int rank) {
    if (!storage || capacity < ds_recipe_rank_slot_bytes() || !parts || rank < 0 || rank > 1)
        return int(cudaErrorInvalidValue);
    for (int j = 0; j < 9; ++j) if (!parts[j]) return int(cudaErrorInvalidValue);
    constexpr int shard = intermediate / 2;
    constexpr size_t packed = trellis_bytes<shard>;
    auto *slot = static_cast<char *>(storage);
    // Checkpoint trellises are [input/16, output/16, 48] uint16 tiles.
    constexpr size_t pitch = (intermediate / 16) * 48 * sizeof(uint16_t);
    for (auto proj : {Gate, Up}) {
        int input_part = proj == Gate ? 0 : 1;
        const auto *src = static_cast<const char *>(parts[input_part]) + rank * (pitch / 2);
        CHECK(cudaMemcpy2D(slot + proj * packed, pitch / 2, src, pitch,
                           pitch / 2, H / 16, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(slot + Down * packed, static_cast<const char *>(parts[2]) + rank * packed,
                     packed, cudaMemcpyHostToDevice));
    auto *scales = reinterpret_cast<half *>(slot + 3 * packed);
    for (auto proj : {Gate, Down, Up}) {
        int first = proj == Gate ? 3 : proj == Up ? 5 : 7;
        int in_width = proj == Down ? shard : H, out_width = proj == Down ? H : shard;
        const auto *input = static_cast<const half *>(parts[first]) + (proj == Down ? rank * shard : 0);
        const auto *output = static_cast<const half *>(parts[first+1]) + (proj == Down ? 0 : rank * shard);
        auto *dst = scales + proj * (H + shard);
        CHECK(cudaMemcpy(dst, input, in_width * sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dst + in_width, output, out_width * sizeof(half), cudaMemcpyHostToDevice));
    }
    return 0;
}

// Ordered, bounded bridge between two local device buffers. Host synchronization
// completes each DMA before the single pinned tile can be reused on either GPU.
// The final H2D leaves the destination's context current for its consumer.
namespace {
constexpr size_t bridge_bytes = 2u << 20;

int transfer_tile(void *dst, int dst_device, const void *src, int src_device, void *host, size_t bytes) {
    if (!dst || !src || !host || !bytes || bytes > bridge_bytes)
        return int(cudaErrorInvalidValue);
    CHECK(cudaSetDevice(src_device));
    CHECK(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaStreamSynchronize(nullptr));
    CHECK(cudaSetDevice(dst_device));
    CHECK(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(nullptr));
    return 0;
}

__global__ void pack_input(const float *src, half *dst, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half_rn(src[i]);
}

__global__ void expand_input(const half *src, float *dst, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(src[i]);
}
} // namespace

extern "C" int ds_recipe_transfer(float *dst, int dst_device, const float *src, int src_device,
                                  float *host, size_t count) {
    if (count > bridge_bytes / sizeof(float)) return int(cudaErrorInvalidValue);
    return transfer_tile(dst, dst_device, src, src_device, host, count * sizeof(float));
}

// The S8B gather casts each selected input to FP16. Moving that cast to the
// sender cuts the outbound bytes in half; widening on the peer is exact.
// The two tiles borrow the owner's receive buffer and the peer's still-empty
// routed-output buffer. Each new H2D follows the previous expand on stream 0.
extern "C" int ds_recipe_broadcast(const float *src, int owner, float *dst, int peer,
                                   float *host, half *send, half *receive, size_t count, int packed) {
    if (!count || !src || !dst || !host || (packed && (!send || !receive)))
        return int(cudaErrorInvalidValue);
    const size_t tile = bridge_bytes / (packed ? sizeof(half) : sizeof(float));
    for (size_t start = 0; start < count; start += tile) {
        size_t n = std::min(count - start, tile);
        if (packed) {
            CHECK(cudaSetDevice(owner));
            pack_input<<<(n+255)/256,256>>>(src + start, send, n);
            CHECK(cudaGetLastError());
            int status = transfer_tile(receive, peer, send, owner, host, n * sizeof(half));
            if (status) return status;
            expand_input<<<(n+255)/256,256>>>(receive, dst + start, n);
            CHECK(cudaGetLastError());
        } else {
            int status = transfer_tile(dst + start, peer, src + start, owner, host, n * sizeof(float));
            if (status) return status;
        }
    }
    return 0;
}

namespace {
__device__ float sum_routes(const float *routed, size_t i) {
    size_t base = (i / H) * 6 * H + i % H;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 6; ++k)
        sum = __fadd_rn(sum, routed[base + size_t(k) * H]);
    return sum;
}

__global__ void reduce_rank(const float *routed, float *tokens, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) tokens[i] = sum_routes(routed, i);
}

__global__ void finish_ranks(const float *routed, const float *peer, float *shared, size_t start, size_t count) {
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < count) {
        size_t i = start + j;
        float sum = __fadd_rn(sum_routes(routed, i), peer[j]);
        shared[i] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(sum, shared[i])));
    }
}
} // namespace

extern "C" int ds_recipe_sum_rank(const float *routed, float *tokens, size_t count) {
    reduce_rank<<<(count + 255)/256,256>>>(routed, tokens, count);
    return int(cudaGetLastError());
}

extern "C" int ds_recipe_finish_ranks(const float *routed, const float *peer, float *shared,
                                      size_t start, size_t count) {
    finish_ranks<<<(count + 255)/256,256>>>(routed, peer, shared, start, count);
    return int(cudaGetLastError());
}
#endif
