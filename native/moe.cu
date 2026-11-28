// EXL3 MUL1 (backbone) and MCG (drafter). Each 16x16 checkpoint tile is a cyclic stream of 256 three-bit
// transitions. Its eight consecutive states per lane are already arranged as
// the B operands of two m16n8k16 tensor-core instructions.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <mutex>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "moe_types.cuh"

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

template <bool mcg>
__device__ __forceinline__ half decode(unsigned state) {
    if constexpr (mcg) {
        unsigned product = (state & 0xffffu) * 0xcbac1fedu;
        unsigned bits = (product & 0x8fff8fffu) ^ 0x3b603b60u;
        return __hadd(__ushort_as_half(bits), __ushort_as_half(bits >> 16));
    }
    unsigned product = (state & 0xffffu) * 0x83dcd12du;
    unsigned sum = __dp4a(product, 0x01010101u, 0x6400u);
    return __hfma(__ushort_as_half(sum), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

__device__ __forceinline__ unsigned pair_bits(half a, half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}

// Decode both consecutive trellis states directly into an MMA B register.
// Low and high half lanes reproduce the scalar codebook arithmetic independently.
template <bool mcg>
__device__ __forceinline__ unsigned decode_pair(unsigned a, unsigned b) {
    constexpr unsigned multiplier = mcg ? 0xcbac1fedu : 0x83dcd12du;
    a = (a & 0xffffu) * multiplier;
    b = (b & 0xffffu) * multiplier;
    unsigned pair;
    if constexpr (mcg) {
        a = (a & 0x8fff8fffu) ^ 0x3b603b60u;
        b = (b & 0x8fff8fffu) ^ 0x3b603b60u;
        unsigned low = __byte_perm(a, b, 0x5410);
        unsigned high = __byte_perm(a, b, 0x7632);
        asm("add.rn.f16x2 %0, %1, %2;" : "=r"(pair) : "r"(low), "r"(high));
    } else {
        a = __dp4a(a, 0x01010101u, 0x6400u);
        b = __dp4a(b, 0x01010101u, 0x6400u);
        unsigned sums = __byte_perm(a, b, 0x5410);
        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(pair) : "r"(sums), "r"(0x1eee1eeeu), "r"(0xc931c931u));
    }
    return pair;
}

__device__ __forceinline__ void mma(const unsigned (&a)[4], unsigned b0, unsigned b1,
                                    float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Four warps cover 64 output columns. Each job covers up to 16 inputs routed
// to one expert, sharing the decoded weights. Split K keeps a one-token job
// large enough to distribute across the SMs. Partials are reduced in order.
template <int projection, bool mcg, bool paired = false, bool packed_decode = false>
__global__ void project(const Job *jobs, const half *x, float *partial, int rows, int splits) {
    constexpr int K = projection == Down ? I : H;
    constexpr int N = projection == Down ? H : I;
    constexpr int KT = K / 16;
    int lane = threadIdx.x % 32;
    int nt = blockIdx.x * 4 + threadIdx.x / 32;
    Job job = jobs[blockIdx.y];
    int split = blockIdx.z;
    int which = projection;
    if constexpr (paired) {
        static_assert(projection == Gate);
        if (split >= splits) {
            which = Up;
            split -= splits;
            x += size_t(rows) * K;
            partial += size_t(splits) * rows * N;
        }
    }
    const uint32_t *tiles = reinterpret_cast<const uint32_t *>(job.expert + which * TRELLIS);
    // Lane l owns stream bits [24*l,24*l+24), preceded by 13 history bits.
    // This interval always fits in two 32-bit words. Lane zero wraps to word 23.
    int end = (lane + 1) * 24;
    int last = (end - 1) / 32;
    int first = ((end - 37 + 768) / 32) % 24;
    int shift = (32 - (end & 31)) & 31;
    int ar0 = lane / 4, ar1 = ar0 + 8, ak = 2 * (lane % 4);
    float c[2][4] = {};
    int kb = split * KT / splits;
    int ke = (split + 1) * KT / splits;
    for (int kt = kb; kt < ke; ++kt) {
        const uint32_t *tile = tiles + (kt * (N / 16) + nt) * 24;
        uint64_t windows = ((uint64_t(tile[first]) << 32) | tile[last]) >> shift;
        unsigned b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned left = unsigned(windows >> (21 - j * 6));
            unsigned right = unsigned(windows >> (18 - j * 6));
            if constexpr (packed_decode)
                b[j] = decode_pair<mcg>(left, right);
            else
                b[j] = pair_bits(decode<mcg>(left), decode<mcg>(right));
        }
        unsigned a[4] = {};
        const half *start = x + size_t(job.begin) * K + kt * 16 + ak;
        if (ar0 < job.count) {
            a[0] = *reinterpret_cast<const unsigned *>(start + ar0 * K);
            a[2] = *reinterpret_cast<const unsigned *>(start + ar0 * K + 8);
        }
        if (ar1 < job.count) {
            a[1] = *reinterpret_cast<const unsigned *>(start + ar1 * K);
            a[3] = *reinterpret_cast<const unsigned *>(start + ar1 * K + 8);
        }
        mma(a, b[0], b[1], c[0]);
        mma(a, b[2], b[3], c[1]);
    }
    float *out = partial + (size_t(split) * rows + job.begin) * N + nt * 16 + ak;
#pragma unroll
    for (int side = 0; side < 2; ++side) {
        if (ar0 < job.count)
            *reinterpret_cast<float2 *>(out + ar0 * N + side * 8) =
                make_float2(c[side][0], c[side][1]);
        if (ar1 < job.count)
            *reinterpret_cast<float2 *>(out + ar1 * N + side * 8) =
                make_float2(c[side][2], c[side][3]);
    }
}

// Each warp rotates 128 consecutive features; each lane owns four features.
__device__ __forceinline__ void had(float (&v)[4]) {
    float a = v[0] + v[1], b = v[0] - v[1];
    float c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
    int lane = threadIdx.x % 32;
#pragma unroll
    for (int step = 1; step < 32; step *= 2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float other = __shfl_xor_sync(0xffffffffu, v[i], step);
            v[i] = other + ((lane & step) ? -v[i] : v[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] *= HAD_SCALE;
}

template <int projection, bool paired = false>
__global__ void rotate_input(const RoutedRow *routes, const float *in, half *out) {
    constexpr int K = projection == Down ? I : H;
    int row = blockIdx.y, k = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= K)
        return;
    const int which = paired && blockIdx.z != 0 ? Up : projection;
    if constexpr (paired) out += size_t(blockIdx.z) * gridDim.y * K;
    const half *scale = suh(routes[row].expert, which);
    const int input_row = projection == Down ? row : routes[row].token;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = __half2float(__hmul(__float2half(in[size_t(input_row) * K + k + i]), scale[k + i]));
    had(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        out[row * K + k + i] = __float2half(v[i]);
}

template <int projection>
__global__ void rotate_output(const RoutedRow *routes, const float *partial, float *out,
                              int rows, int splits) {
    constexpr int N = projection == Down ? H : I;
    int row = blockIdx.y, n = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N)
        return;
    int offset = row * N + n;
    float v[4] = {};
    for (int split = 0; split < splits; ++split) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] += partial[(size_t(split) * rows) * N + offset + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = __half2float(__float2half(v[i]));
    had(v);
    const half *scale = svh(routes[row].expert, projection);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (projection == Down) {
            const float value = __fmul_rn(v[i], __half2float(scale[n + i]));
            out[size_t(routes[row].output) * N + n + i] = __fmul_rn(routes[row].weight, value);
        } else
            out[offset + i] = __half2float(__hmul(__float2half(v[i]), scale[n + i]));
    }
}

__device__ __forceinline__ half routed_activation(half g, half u) {
    const half limit = __float2half(10.f);
    const half silu = __hmul(g, hrcp(__hadd(__float2half(1.f), hexp(__hneg(g)))));
    return __hmul(__hmin(silu, limit), __hmin(limit, __hmax(__hneg(limit), u)));
}

__global__ void swiglu(float *gate, const float *up, int length) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < length)
        gate[t] = __half2float(routed_activation(__float2half(gate[t]), __float2half(up[t])));
}

// Finish both output transforms and immediately run the input transform of
// the down projection. Every Hadamard spans the same warp and 128 features.
__global__ void prepare_down(const RoutedRow *routes, const float *partial, half *down,
                             int rows, int splits) {
    int row = blockIdx.y, col = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= I) return;
    size_t off = size_t(row) * I + col;
    size_t stride = size_t(rows) * I;
    const float *gp = partial + off, *up = gp + size_t(splits) * stride;
    float g[4] = {}, u[4] = {};
    for (int split = 0; split < splits; ++split) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g[j] += gp[size_t(split) * stride + j];
            u[j] += up[size_t(split) * stride + j];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        g[j] = __half2float(__float2half(g[j]));
        u[j] = __half2float(__float2half(u[j]));
    }
    had(g);
    had(u);
    const char *expert = routes[row].expert;
    const half *gs = svh(expert, Gate), *us = svh(expert, Up), *ds = suh(expert, Down);
    float d[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        half gate = __hmul(__float2half(g[j]), gs[col + j]);
        half up_value = __hmul(__float2half(u[j]), us[col + j]);
        d[j] = __half2float(__hmul(routed_activation(gate, up_value), ds[col + j]));
    }
    had(d);
#pragma unroll
    for (int j = 0; j < 4; ++j) down[off+j] = __float2half(d[j]);
}

// Shared experts use FP32 SiLU on the clipped gate. Widen the exponential
// before rounding to f32 to match host expf on the BF16 gate values.
__global__ void shared_swiglu(float *gate, const float *up, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float g = fminf(gate[i], 10.f);
        float u = up[i];
        u = u > 10.f ? 10.f : (u < -10.f ? -10.f : u);
        float denominator = __fadd_rn(1.f, float(exp(-double(g))));
        gate[i] = __fmul_rn(__fdiv_rn(g, denominator), u);
    }
}

template <int top_k>
__global__ void finish(const float *routed, float *shared_out, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        size_t base = (i / H) * top_k * H + i % H;
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < top_k; ++r)
            sum = __fadd_rn(sum, routed[base + size_t(r) * H]);
        shared_out[i] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(sum, shared_out[i])));
    }
}

#define CHECK(op)                                                                                  \
    do {                                                                                           \
        cudaError_t e = (op);                                                                      \
        if (e != cudaSuccess)                                                                      \
            return int(e);                                                                         \
    } while (0)

template <int projection, bool mcg>
void launch_projection(const Job *jobs, const RoutedRow *routes, int n_jobs, int rows,
                       int splits, const float *src, half *rotated, float *partial, float *dst,
                       cudaStream_t stream = nullptr) {
    constexpr int K = projection == Down ? I : H, N = projection == Down ? H : I;
    rotate_input<projection><<<dim3((K + 511) / 512, rows), 128, 0, stream>>>(routes, src, rotated);
    project<projection, mcg>
        <<<dim3(N / 64, n_jobs, splits), 128, 0, stream>>>(jobs, rotated, partial, rows, splits);
    rotate_output<projection>
        <<<dim3((N + 511) / 512, rows), 128, 0, stream>>>(routes, partial, dst, rows, splits);
}
// Small decode and verification descriptors fit in the kernel parameter block.
// The CUDA launch copies this trivially-copyable argument before returning, so
// the CPU can release its vectors and queue later kernels immediately.
template <int capacity>
struct Descriptors {
    Job jobs[capacity];
    RoutedRow routes[capacity];
};
template <int capacity>
__global__ void write_descriptors(const __grid_constant__ Descriptors<capacity> plan,
                                  int njobs, int rows, Job *jobs, RoutedRow *routes) {
    int i = threadIdx.x;
    if (i < njobs) jobs[i] = plan.jobs[i];
    if (i < rows) routes[i] = plan.routes[i];
}
template <int capacity>
void stage_descriptors(const Job *jobs, int njobs, const RoutedRow *routes, int rows,
                       Job *dst_jobs, RoutedRow *dst_routes, cudaStream_t stream = nullptr) {
    Descriptors<capacity> plan{};
    std::memcpy(plan.jobs, jobs, njobs * sizeof(Job));
    std::memcpy(plan.routes, routes, rows * sizeof(RoutedRow));
    write_descriptors<<<1, capacity, 0, stream>>>(plan, njobs, rows, dst_jobs, dst_routes);
}

template <bool mcg>
void launch_ffn(Job *device_jobs, RoutedRow *device_rows, int n_jobs, int rows, int splits,
                const float *x, half *rotated, float *partial, float *gate, float *up, float *weighted,
                cudaStream_t stream = nullptr) {
    launch_projection<Gate, mcg>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial,
                            gate, stream);
    launch_projection<Up, mcg>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial, up, stream);
    swiglu<<<(rows * I + 255) / 256, 256, 0, stream>>>(gate, up, rows * I);
    launch_projection<Down, mcg>(device_jobs, device_rows, n_jobs, rows, splits, gate, rotated,
                            partial, weighted, stream);
}
template <bool mcg, bool packed_decode>
void launch_fused(const Job *jobs, const RoutedRow *routes, int n_jobs, int rows, int splits,
                  const float *x, half *rotated, float *partial, float *weighted,
                  cudaStream_t stream = nullptr) {
    // The partial allocation is sized for H down columns, sufficient for
    // 2*I gate/up columns. The contiguous gate/up/rotated buffers hold 2*H halves.
    static_assert(2 * I <= H && 2 * I * sizeof(float) >= H * sizeof(half));
    rotate_input<Gate, true><<<dim3((H+511)/512,rows,2),128,0,stream>>>(routes, x, rotated);
    project<Gate, mcg, true, packed_decode><<<dim3(I/64,n_jobs,2*splits),128,0,stream>>>(jobs, rotated, partial, rows, splits);
    prepare_down<<<dim3((I+511)/512,rows),128,0,stream>>>(routes, partial, rotated, rows, splits);
    project<Down, mcg, false, packed_decode><<<dim3(H/64,n_jobs,splits),128,0,stream>>>(jobs, rotated, partial, rows, splits);
    rotate_output<Down><<<dim3((H+511)/512,rows),128,0,stream>>>(routes, partial, weighted, rows, splits);
}

struct GraphShape {
    void *scratch;
    const float *input;
    float *output;
    int jobs, rows, splits, mcg, fused, device;
    bool operator==(const GraphShape &o) const {
        return std::tie(scratch,input,output,jobs,rows,splits,mcg,fused,device) ==
               std::tie(o.scratch,o.input,o.output,o.jobs,o.rows,o.splits,o.mcg,o.fused,o.device);
    }
};
struct GraphShapeHash {
    size_t operator()(const GraphShape &s) const {
        size_t h = std::hash<void *>{}(s.scratch);
        h ^= std::hash<const float *>{}(s.input) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<float *>{}(s.output) + 0x9e3779b9 + (h<<6) + (h>>2);
        auto mix = [&h](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(s.jobs); mix(s.rows); mix(s.splits); mix(s.mcg); mix(s.fused); mix(s.device);
        return h;
    }
};
struct GraphEntry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Job *host_jobs = nullptr;
    RoutedRow *host_rows = nullptr;
    bool replay_logged = false;
    ~GraphEntry() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (host_jobs) cudaFreeHost(host_jobs);
        if (host_rows) cudaFreeHost(host_rows);
    }
};
struct GraphContext {
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t before = nullptr, done = nullptr;
    bool launched = false;
    int status = 0;
    std::unordered_map<GraphShape, GraphEntry, GraphShapeHash> graphs;
    ~GraphContext() {
        int old = -1;
        cudaGetDevice(&old);
        if (device >= 0) cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        graphs.clear();
        if (before) cudaEventDestroy(before);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
        if (old >= 0) cudaSetDevice(old);
    }
};
static std::mutex graph_mutex;
static std::unordered_map<void *, std::unique_ptr<GraphContext>> graph_contexts;
static bool graph_trace_enabled() {
    const char *value = std::getenv("DS4_MOE_CUDA_GRAPH_TRACE");
    return value && value[0] == '1' && value[1] == '\0';
}
static int graph_error(const char *stage, cudaError_t error) {
    std::fprintf(stderr, "moe-cuda-graph error stage=%s code=%d message=%s\n",
                 stage, int(error), cudaGetErrorString(error));
    return int(error);
}

static GraphContext *graph_context(void *scratch, int device) {
    auto &entry = graph_contexts[scratch];
    if (!entry) {
        entry = std::make_unique<GraphContext>();
        entry->device = device;
        entry->status = int(cudaStreamCreateWithFlags(&entry->stream, cudaStreamNonBlocking));
        if (entry->status == 0)
            entry->status = int(cudaEventCreateWithFlags(&entry->before, cudaEventDisableTiming));
        if (entry->status == 0)
            entry->status = int(cudaEventCreateWithFlags(&entry->done, cudaEventDisableTiming));
    }
    if (entry->device != device && entry->status == 0) entry->status = int(cudaErrorInvalidDevice);
    return entry.get();
}

extern "C" void ds_moe_graph_release(void *scratch) {
    if (!scratch) return;
    std::lock_guard<std::mutex> lock(graph_mutex);
    auto it = graph_contexts.find(scratch);
    if (it != graph_contexts.end()) graph_contexts.erase(it);
}

static int graph_run(void *scratch, int device, const Job *jobs, int n_jobs,
                     const RoutedRow *routes, int rows, const float *x, float *weighted,
                     int mcg, int splits, int fused, char *base,
                     size_t partial_size, size_t inter_size, size_t rotated_size,
                     size_t meta_size) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GraphContext *ctx = graph_context(scratch, device);
    if (ctx->status) return ctx->status;
    auto ready = ctx->launched ? cudaEventQuery(ctx->done) : cudaSuccess;
    if (ready == cudaErrorNotReady) return -1; // An in-flight graph owns the staging descriptors.
    if (ready != cudaSuccess) return graph_error("event-query", ready);
    GraphShape shape{scratch,x,weighted,n_jobs,rows,splits,mcg,fused,device};
    auto found = ctx->graphs.find(shape);
    if (found == ctx->graphs.end() && ctx->graphs.size() >= 256) return -1;
    auto [it, inserted] = ctx->graphs.try_emplace(shape);
    GraphEntry &entry = it->second;
    char *meta = base + partial_size + 2 * inter_size + rotated_size;
    Job *device_jobs = reinterpret_cast<Job *>(meta);
    RoutedRow *device_rows = reinterpret_cast<RoutedRow *>(meta + size_t(n_jobs) * sizeof(Job));
    size_t jobs_bytes = size_t(n_jobs) * sizeof(Job), routes_bytes = size_t(rows) * sizeof(RoutedRow);
    const float *graph_x = x;
    float *graph_y = weighted;
    if (inserted) {
        auto rc = cudaMallocHost(&entry.host_jobs, jobs_bytes);
        if (rc != cudaSuccess) return graph_error("cuda-graph-operation", rc);
        rc = cudaMallocHost(&entry.host_rows, routes_bytes);
        if (rc != cudaSuccess) return graph_error("cuda-graph-operation", rc);
    }
    std::memcpy(entry.host_jobs, jobs, jobs_bytes);
    std::memcpy(entry.host_rows, routes, routes_bytes);

    auto rc = cudaEventRecord(ctx->before, nullptr);
    if (rc != cudaSuccess) return graph_error("record-before", rc);
    rc = cudaStreamWaitEvent(ctx->stream, ctx->before, 0);
    if (rc != cudaSuccess) return graph_error("wait-before", rc);

    if (!entry.exec) {
        rc = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal);
        if (rc != cudaSuccess) return graph_error("begin-capture", rc);
        auto abort_capture = [&]() {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(ctx->stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
        };
        rc = cudaMemcpyAsync(device_jobs, entry.host_jobs, jobs_bytes, cudaMemcpyHostToDevice, ctx->stream);
        if (rc != cudaSuccess) graph_error("jobs-copy", rc);
        if (rc == cudaSuccess) rc = cudaMemcpyAsync(device_rows, entry.host_rows, routes_bytes, cudaMemcpyHostToDevice, ctx->stream);
        if (rc != cudaSuccess) graph_error("routes-copy", rc);
        if (rc == cudaSuccess) {
            if (fused) {
                auto launch = mcg ? (fused == 2 ? launch_fused<true,true> : launch_fused<true,false>)
                                 : (fused == 2 ? launch_fused<false,true> : launch_fused<false,false>);
                launch(device_jobs, device_rows, n_jobs, rows, splits, graph_x,
                       reinterpret_cast<half *>(base + partial_size),
                       reinterpret_cast<float *>(base), graph_y, ctx->stream);
            } else if (mcg) {
                launch_ffn<true>(device_jobs, device_rows, n_jobs, rows, splits, graph_x,
                    reinterpret_cast<half *>(base + partial_size + 2 * inter_size),
                    reinterpret_cast<float *>(base), reinterpret_cast<float *>(base + partial_size),
                    reinterpret_cast<float *>(base + partial_size + inter_size), graph_y, ctx->stream);
            } else {
                launch_ffn<false>(device_jobs, device_rows, n_jobs, rows, splits, graph_x,
                    reinterpret_cast<half *>(base + partial_size + 2 * inter_size),
                    reinterpret_cast<float *>(base), reinterpret_cast<float *>(base + partial_size),
                    reinterpret_cast<float *>(base + partial_size + inter_size), graph_y, ctx->stream);
            }
            rc = cudaGetLastError();
        }
        if (rc == cudaSuccess) {
            rc = cudaStreamEndCapture(ctx->stream, &entry.graph);
            if (rc != cudaSuccess) return graph_error("end-capture", rc);
        } else {
            graph_error("captured-op", rc);
            abort_capture();
            return int(cudaErrorNotSupported);
        }
        rc = cudaGraphInstantiate(&entry.exec, entry.graph, 0);
        if (rc != cudaSuccess) return graph_error("instantiate", rc);
        if (graph_trace_enabled())
            std::fprintf(stderr, "moe-cuda-graph capture device=%d rows=%d jobs=%d splits=%d mcg=%d fused=%d\n",
                         device, rows, n_jobs, splits, mcg, fused);
        // The first launch executes the freshly instantiated graph below.
    } else if (!entry.replay_logged && graph_trace_enabled()) {
        std::fprintf(stderr, "moe-cuda-graph replay device=%d rows=%d jobs=%d splits=%d\n",
                     device, rows, n_jobs, splits);
        entry.replay_logged = true;
    }
    rc = cudaGraphLaunch(entry.exec, ctx->stream);
    if (rc == cudaSuccess) rc = cudaEventRecord(ctx->done, ctx->stream);
    if (rc == cudaSuccess) rc = cudaStreamWaitEvent(nullptr, ctx->done, 0);
    if (rc == cudaSuccess) ctx->launched = true;
    return rc == cudaSuccess ? 0 : graph_error("launch-graph", rc);
}

} // namespace

extern "C" const char *ds_moe_error(int e) { return cudaGetErrorString(cudaError_t(e)); }
extern "C" int ds_moe_free_bytes(int device, size_t *free_bytes) {
    auto status = cudaSetDevice(device);
    if (status != cudaSuccess) return int(status);
    size_t total;
    return int(cudaMemGetInfo(free_bytes, &total));
}
extern "C" int ds_moe_alloc(void **allocation) { return int(cudaMalloc(allocation, EXPERT)); }
extern "C" void ds_moe_free(int device, void *allocation) {
    if (!allocation)
        return;
    int old;
    if (cudaGetDevice(&old) != cudaSuccess)
        return;
    if (cudaSetDevice(device) == cudaSuccess)
        cudaFree(allocation);
    cudaSetDevice(old);
}

// The public part array is w1/w3/w2, followed by each matching scale pair.
// Safetensors groups the three trellises and all six scales in w1/w2/w3 order.
// Coalesce adjacent extents only; independently allocated inputs work too.
extern "C" int ds_moe_upload_parts(char *slot, const void *const *parts, int coalesce) {
    constexpr int indices[] = {0, 2, 1, 3, 4, 7, 8, 5, 6};
    constexpr size_t sizes[] = {TRELLIS, TRELLIS, TRELLIS, H*2, I*2, I*2, H*2, H*2, I*2};
    size_t offset = 0;
    for (int p = 0; p < 9;) {
        const void *src = parts[indices[p]];
        size_t bytes = sizes[p++];
        while (coalesce && p < 9 &&
               reinterpret_cast<uintptr_t>(src) + bytes == reinterpret_cast<uintptr_t>(parts[indices[p]]))
            bytes += sizes[p++];
        CHECK(cudaMemcpy(slot + offset, src, bytes, cudaMemcpyHostToDevice));
        offset += bytes;
    }
    return 0;
}

extern "C" int ds_moe_upload(char *slot, const void *const *parts) {
    return ds_moe_upload_parts(slot, parts, 1);
}

// Queue the small router read before launching the shared expert branch. The
// host then waits just for these logits and ranks them during the shared GEMMs.
extern "C" int ds_moe_router_start(float **host, size_t *capacity, cudaEvent_t *ready,
                                   const float *logits, size_t count) {
    if (!*ready) CHECK(cudaEventCreateWithFlags(ready, cudaEventDisableTiming));
    if (*capacity < count) {
        if (*host) {
            CHECK(cudaEventSynchronize(*ready));
            CHECK(cudaFreeHost(*host));
            *host = nullptr;
            *capacity = 0;
        }
        CHECK(cudaMallocHost(host, count * sizeof(float)));
        *capacity = count;
    }
    CHECK(cudaMemcpyAsync(*host, logits, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(*ready));
    return 0;
}
extern "C" int ds_moe_router_wait(cudaEvent_t ready) { return int(cudaEventSynchronize(ready)); }
extern "C" void ds_moe_router_free(int device, float *host, cudaEvent_t ready) {
    if (!ready && !host) return;
    int old;
    if (cudaGetDevice(&old) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
    if (ready) cudaEventSynchronize(ready);
    if (host) cudaFreeHost(host);
    if (ready) cudaEventDestroy(ready);
    cudaSetDevice(old);
}

extern "C" int ds_moe_shared_swiglu(float *gate, const float *up, size_t count) {
    shared_swiglu<<<(count + 255) / 256, 256>>>(gate, up, count);
    return int(cudaGetLastError());
}
extern "C" int ds_moe_finish(const float *routed, float *shared_out, size_t count, int top_k) {
    if (top_k == 3)
        finish<3><<<(count + 255) / 256, 256>>>(routed, shared_out, count);
    else if (top_k == 6)
        finish<6><<<(count + 255) / 256, 256>>>(routed, shared_out, count);
    else
        return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}

// Only the job and route descriptors cross PCIe. Input, rotated activations,
// split-K partials and weighted outputs stay on the current GPU. The next
// group also uses the default stream, including uploads into reused slots.
extern "C" int ds_moe_run_dev(void **scratch, size_t *scratch_bytes, const Job *jobs, int n_jobs,
                              const RoutedRow *routes, int rows, const float *x, float *weighted,
                              int mcg, int splits, int inline_metadata, int fused, int use_graph) {
    // Verifying several positions must use the same K-reduction boundaries as
    // serial decoding, even if the block touches 32 or more distinct experts.
    if (splits != 2 && splits != 8) return int(cudaErrorInvalidValue);
    if (n_jobs <= 0 || n_jobs > rows) return int(cudaErrorInvalidValue);
    if (fused == 3) {
#ifdef DS4_RECIPE_PREFILL
        if (mcg) return int(cudaErrorInvalidValue);
        return ds_recipe_group(scratch, scratch_bytes, jobs, n_jobs, routes, rows, x, weighted);
#else
        return int(cudaErrorNotSupported);
#endif
    }
    size_t partial_size = size_t(rows) * H * splits * sizeof(float);
    size_t inter_size = size_t(rows) * I * sizeof(float);
    size_t rotated_size = size_t(rows) * H * sizeof(half);
    size_t jobs_size = size_t(n_jobs) * sizeof(Job);
    size_t meta_size = jobs_size + size_t(rows) * sizeof(RoutedRow);
    size_t size = partial_size + 2 * inter_size + rotated_size + meta_size;
    if (*scratch_bytes < size) {
        if (*scratch) {
            ds_moe_graph_release(*scratch);
            CHECK(cudaFree(*scratch));
        }
        *scratch = nullptr;
        *scratch_bytes = 0;
        CHECK(cudaMalloc(scratch, size));
        *scratch_bytes = size;
    }
    char *scratch_base = static_cast<char *>(*scratch);
    char *base = scratch_base;
    float *partial = reinterpret_cast<float *>(base);
    base += partial_size;
    float *gate = reinterpret_cast<float *>(base);
    base += inter_size;
    float *up = reinterpret_cast<float *>(base);
    base += inter_size;
    half *rotated = reinterpret_cast<half *>(base);
    base += rotated_size;
    Job *device_jobs = reinterpret_cast<Job *>(base);
    auto device_rows = reinterpret_cast<RoutedRow *>(base + jobs_size);
    if (use_graph && rows <= 64) {
        int device = 0;
        CHECK(cudaGetDevice(&device));
        int graph_status = graph_run(*scratch, device, jobs, n_jobs, routes, rows, x, weighted,
                                     mcg, splits, fused, scratch_base, partial_size, inter_size,
                                     rotated_size, meta_size);
        if (graph_status == 0) return 0;
        if (graph_status != -1) return graph_status;
        // A busy graph keeps its metadata pinned until its completion event.
        // Use the existing eager path rather than waiting for it on the host.
    }
    // Wide groups (concurrent verification) pass up to 256 descriptors as a
    // 10 KiB kernel parameter instead of a synchronous copy that drains the stream.
    static const bool wide_inline = [] {
        const char *v = std::getenv("DS4_MOE_INLINE_WIDE");
        return !v || v[0] != '0';
    }();
    if (inline_metadata && (rows <= 64 || (wide_inline && rows <= 256))) {
        if (rows <= 8)
            stage_descriptors<8>(jobs, n_jobs, routes, rows, device_jobs, device_rows);
        else if (rows <= 64)
            stage_descriptors<64>(jobs, n_jobs, routes, rows, device_jobs, device_rows);
        else
            stage_descriptors<256>(jobs, n_jobs, routes, rows, device_jobs, device_rows);
        CHECK(cudaGetLastError());
    } else {
        CHECK(cudaMemcpy(device_jobs, jobs, jobs_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(device_rows, routes, size_t(rows) * sizeof(RoutedRow), cudaMemcpyHostToDevice));
    }
    if (fused) {
        half *work = reinterpret_cast<half *>(gate);
        // Mode 1 is the original fused reference, 2 packs codebook arithmetic by pairs.
        auto launch = mcg ? (fused == 2 ? launch_fused<true,true> : launch_fused<true,false>)
                          : (fused == 2 ? launch_fused<false,true> : launch_fused<false,false>);
        launch(device_jobs, device_rows, n_jobs, rows, splits, x, work, partial, weighted, nullptr);
    } else if (mcg)
        launch_ffn<true>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial, gate, up, weighted);
    else
        launch_ffn<false>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial, gate, up, weighted);
    return int(cudaGetLastError());
}

// Host ABI used by the standalone matrix-reconstruction diagnostic. Identity
// gather/scatter and a unit weight expose a plain expert output for each input.
extern "C" int ds_moe_run(void **scratch, size_t *scratch_bytes, const Job *jobs, int n_jobs,
                          const char *const *experts, const float *x, float *y, int rows) {
    std::vector<RoutedRow> routes(rows);
    for (int i = 0; i < rows; ++i)
        routes[i] = {experts[i], i, i, 1.f};
    struct Allocation {
        float *ptr = nullptr;
        ~Allocation() { if (ptr) cudaFree(ptr); }
    } io;
    size_t count = size_t(rows) * H;
    CHECK(cudaMalloc(&io.ptr, 2 * count * sizeof(float)));
    CHECK(cudaMemcpy(io.ptr, x, count * sizeof(float), cudaMemcpyHostToDevice));
    int code = ds_moe_run_dev(scratch, scratch_bytes, jobs, n_jobs, routes.data(), rows,
                              io.ptr, io.ptr + count, 0, n_jobs < 32 ? 8 : 2, 1, 0, 0);
    if (code)
        return code;
    return int(cudaMemcpy(y, io.ptr + count, count * sizeof(float), cudaMemcpyDeviceToHost));
}

// The primary stream owns the activation pool. Record readiness before the
// shared branch so the second card can begin immediately once routes arrive.
struct PeerEvents {
    int primary, peer;
    cudaEvent_t ready = nullptr, done = nullptr;
    float *staging = nullptr;
    size_t capacity = 0;
};

extern "C" void ds_moe_peer_free(PeerEvents *p) {
    if (!p) return;
    int old;
    if (cudaGetDevice(&old) == cudaSuccess) {
        cudaSetDevice(p->primary);
        if (p->staging) cudaStreamSynchronize(nullptr); // retire the return DMA
        if (p->ready) cudaEventDestroy(p->ready);
        cudaSetDevice(p->peer);
        if (p->done) cudaEventDestroy(p->done);
        if (p->staging) cudaFree(p->staging);
        cudaSetDevice(old);
    }
    delete p;
}

extern "C" int ds_moe_peer_begin(PeerEvents **link, int primary, int peer) {
    struct Restore { int dev; ~Restore() { cudaSetDevice(dev); } } restore{primary};
    if (!*link) {
        *link = new (std::nothrow) PeerEvents{primary, peer};
        if (!*link) return int(cudaErrorMemoryAllocation);
    }
    auto *p = *link;
    if (!p->done) {
        int supported = 0;
        CHECK(cudaDeviceCanAccessPeer(&supported, peer, primary));
        if (!supported) return int(cudaErrorPeerAccessUnsupported);
        CHECK(cudaSetDevice(peer));
        auto enabled = cudaDeviceEnablePeerAccess(primary, 0);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else CHECK(enabled);
        CHECK(cudaEventCreateWithFlags(&p->done, cudaEventDisableTiming));
    }
    CHECK(cudaSetDevice(primary));
    if (!p->ready) CHECK(cudaEventCreateWithFlags(&p->ready, cudaEventDisableTiming));
    CHECK(cudaEventRecord(p->ready));
    CHECK(cudaSetDevice(peer));
    CHECK(cudaStreamWaitEvent(nullptr, p->ready, 0));
    return 0;
}

// The primary consumes the peer's disjoint output slots and can then reuse
// both tensors. Even an unsuccessful event submission must retire peer work
// before Rust drops the owning buffers or retries on this inference worker.
extern "C" int ds_moe_peer_end(PeerEvents *p) {
    auto status = cudaSetDevice(p->peer);
    if (status == cudaSuccess) status = cudaEventRecord(p->done);
    if (status == cudaSuccess) status = cudaSetDevice(p->primary);
    if (status == cudaSuccess) status = cudaStreamWaitEvent(nullptr, p->done, 0);
    if (status != cudaSuccess) {
        if (cudaSetDevice(p->peer) == cudaSuccess) cudaStreamSynchronize(nullptr);
        cudaSetDevice(p->primary);
    }
    return int(status);
}

extern "C" size_t ds_moe_peer_capacity(const PeerEvents *p) { return p->capacity; }

extern "C" int ds_moe_peer_stage(PeerEvents *p, const float *input, size_t nx,
                                  size_t rows, float **peer_x, float **peer_output) {
    struct Restore { int dev; ~Restore() { cudaSetDevice(dev); } } restore{p->primary};
    CHECK(cudaSetDevice(p->peer));
    size_t size = nx + rows * H;
    if (p->capacity < size) {
        if (p->staging) {
            // The ready event also follows the previous return DMA on primary.
            CHECK(cudaStreamSynchronize(nullptr));
            CHECK(cudaFree(p->staging));
        }
        p->staging = nullptr;
        p->capacity = 0;
        CHECK(cudaMalloc(&p->staging, size * sizeof(float)));
        p->capacity = size;
    }
    // The peer stream already waits on the primary activation's ready event.
    CHECK(cudaMemcpyPeerAsync(p->staging, p->peer, input, p->primary, nx * sizeof(float)));
    *peer_x = p->staging;
    *peer_output = p->staging + nx;
    return 0;
}

struct Destinations { int index[256]; };
__global__ void scatter_peer(const __grid_constant__ Destinations rows, const float *compact, float *weighted) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < H) weighted[size_t(rows.index[blockIdx.y]) * H + c] = compact[size_t(blockIdx.y) * H + c];
}

extern "C" int ds_moe_peer_collect(PeerEvents *p, const float *peer_output, float *compact,
                                    const int *ids, int rows, float *weighted) {
    CHECK(cudaSetDevice(p->primary));
    // ds_moe_peer_end has made every peer result visible to the primary stream.
    CHECK(cudaMemcpyPeerAsync(compact, p->primary, peer_output, p->peer, size_t(rows) * H * sizeof(float)));
    for (int begin = 0; begin < rows; begin += 256) {
        int count = std::min(256, rows-begin);
        Destinations targets{};
        std::memcpy(targets.index, ids + begin, count*sizeof(int));
        scatter_peer<<<dim3((H+255)/256,count),256>>>(targets, compact + size_t(begin)*H, weighted);
        CHECK(cudaGetLastError());
    }
    return 0;
}
