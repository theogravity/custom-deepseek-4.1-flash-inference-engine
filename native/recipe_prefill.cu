#ifdef DS4_RECIPE_PREFILL
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
constexpr int H = 5120, I = 2304;
constexpr size_t TRELLIS = size_t(H) * I * 3 / 8;
constexpr size_t EXPERT = 3 * (TRELLIS + (H + I) * sizeof(half));
constexpr float HAD_SCALE = 0.08838834764831845f;
// Physical projection order follows the checkpoint's contiguous w1, w2, w3.
enum Projection { Gate = 0, Down = 1, Up = 2 };
__device__ const half *suh(const char *expert, int projection) {
    return reinterpret_cast<const half *>(expert + 3 * TRELLIS) + projection * (H + I);
}
__device__ const half *svh(const char *expert, int projection) {
    return suh(expert, projection) + (projection == Down ? I : H);
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
__global__ void recipe_rows(const RoutedRow *routes, const float *x,
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
        meta.gate_in[r] = reinterpret_cast<int64_t>(suh(route.expert, Gate));
        meta.up_in[r] = reinterpret_cast<int64_t>(suh(route.expert, Up));
        meta.gate_out[r] = reinterpret_cast<int64_t>(svh(route.expert, Gate));
        meta.up_out[r] = reinterpret_cast<int64_t>(svh(route.expert, Up));
        meta.down_in[r] = reinterpret_cast<int64_t>(suh(route.expert, Down));
        meta.down_out[r] = reinterpret_cast<int64_t>(svh(route.expert, Down));
    }
}
// Every route owns a unique output rank. The recipe uses atomic adds; keep
// those local even when the caller owns the final buffer on the other GPU.
__global__ void recipe_scatter(const float *compact, const int64_t *outputs, float *weighted) {
    int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < H) weighted[size_t(outputs[row])*H + col] = compact[size_t(row)*H + col];
}
__global__ void recipe_jobs(const Job *jobs, int n, int *starts, int *counts, int64_t *trellis) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    Job j = jobs[i];
    starts[i] = j.begin;
    counts[i] = j.count;
    for (int projection = 0; projection != 3; ++projection)
        trellis[projection*n+i] = reinterpret_cast<int64_t>(j.expert + projection * TRELLIS);
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

extern "C" int ds_recipe_group(void **scratch, size_t *capacity, const Job *jobs, int nj,
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
    recipe_rows<<<dim3((H+255)/256,rows),256>>>(rr,x,input,meta);
    CHECK(cudaGetLastError());
    recipe_jobs<<<(nj+127)/128,128>>>(j,nj,starts,starts+nj,tr);
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
#endif
