// EXL3 MUL1 (backbone) and MCG (drafter). Each 16x16 checkpoint tile is a cyclic stream of 256 three-bit
// transitions. Its eight consecutive states per lane are already arranged as
// the B operands of two m16n8k16 tensor-core instructions.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

struct Job {
    const char *expert;
    int begin, count;
};

// The output row encodes both the token and its reduction position. Production
// assigns consecutive top-k output rows per token, ordered by expert index.
struct RoutedRow {
    const char *expert;
    int token, output;
    float weight;
};

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
template <int projection, bool mcg>
__global__ void project(const Job *jobs, const half *x, float *partial, int rows, int splits) {
    constexpr int K = projection == Down ? I : H;
    constexpr int N = projection == Down ? H : I;
    constexpr int KT = K / 16;
    int lane = threadIdx.x % 32;
    int nt = blockIdx.x * 4 + threadIdx.x / 32;
    Job job = jobs[blockIdx.y];
    const uint32_t *tiles = reinterpret_cast<const uint32_t *>(job.expert + projection * TRELLIS);
    // Lane l owns stream bits [24*l,24*l+24), preceded by 13 history bits.
    // This interval always fits in two 32-bit words. Lane zero wraps to word 23.
    int end = (lane + 1) * 24;
    int last = (end - 1) / 32;
    int first = ((end - 37 + 768) / 32) % 24;
    int shift = (32 - (end & 31)) & 31;
    int ar0 = lane / 4, ar1 = ar0 + 8, ak = 2 * (lane % 4);
    float c[2][4] = {};
    int kb = int(blockIdx.z) * KT / splits;
    int ke = (int(blockIdx.z) + 1) * KT / splits;
    for (int kt = kb; kt < ke; ++kt) {
        const uint32_t *tile = tiles + (kt * (N / 16) + nt) * 24;
        uint64_t windows = ((uint64_t(tile[first]) << 32) | tile[last]) >> shift;
        unsigned b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            b[j] = pair_bits(decode<mcg>(unsigned(windows >> (21 - j * 6))),
                             decode<mcg>(unsigned(windows >> (18 - j * 6))));
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
    float *out = partial + (size_t(blockIdx.z) * rows + job.begin) * N + nt * 16 + ak;
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

template <int projection>
__global__ void rotate_input(const RoutedRow *routes, const float *in, half *out) {
    constexpr int K = projection == Down ? I : H;
    int row = blockIdx.y, k = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= K)
        return;
    const half *scale = suh(routes[row].expert, projection);
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

__global__ void swiglu(float *gate, const float *up, int length) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < length) {
        const half g = __float2half(gate[t]);
        const half limit = __float2half(10.f);
        const half silu = __hmul(g, hrcp(__hadd(__float2half(1.f), hexp(__hneg(g)))));
        const half u = __hmin(limit, __hmax(__hneg(limit), __float2half(up[t])));
        gate[t] = __half2float(__hmul(__hmin(silu, limit), u));
    }
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
                       int splits, const float *src, half *rotated, float *partial, float *dst) {
    constexpr int K = projection == Down ? I : H, N = projection == Down ? H : I;
    rotate_input<projection><<<dim3((K + 511) / 512, rows), 128>>>(routes, src, rotated);
    project<projection, mcg>
        <<<dim3(N / 64, n_jobs, splits), 128>>>(jobs, rotated, partial, rows, splits);
    rotate_output<projection>
        <<<dim3((N + 511) / 512, rows), 128>>>(routes, partial, dst, rows, splits);
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
                       Job *dst_jobs, RoutedRow *dst_routes) {
    Descriptors<capacity> plan{};
    std::memcpy(plan.jobs, jobs, njobs * sizeof(Job));
    std::memcpy(plan.routes, routes, rows * sizeof(RoutedRow));
    write_descriptors<<<1, capacity>>>(plan, njobs, rows, dst_jobs, dst_routes);
}

template <bool mcg>
void launch_ffn(Job *device_jobs, RoutedRow *device_rows, int n_jobs, int rows, int splits,
                const float *x, half *rotated, float *partial, float *gate, float *up, float *weighted) {
    launch_projection<Gate, mcg>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial,
                            gate);
    launch_projection<Up, mcg>(device_jobs, device_rows, n_jobs, rows, splits, x, rotated, partial, up);
    swiglu<<<(rows * I + 255) / 256, 256>>>(gate, up, rows * I);
    launch_projection<Down, mcg>(device_jobs, device_rows, n_jobs, rows, splits, gate, rotated,
                            partial, weighted);
}
} // namespace

extern "C" const char *ds_moe_error(int e) { return cudaGetErrorString(cudaError_t(e)); }
extern "C" int ds_moe_free_bytes(size_t *free_bytes) {
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
                              int mcg, int verify, int inline_metadata) {
    // Verifying several positions must use the same K-reduction boundaries as
    // serial decoding, even if the block touches 32 or more distinct experts.
    int splits = verify || n_jobs < 32 ? 8 : 2;
    size_t partial_size = size_t(rows) * H * splits * sizeof(float);
    size_t inter_size = size_t(rows) * I * sizeof(float);
    size_t rotated_size = size_t(rows) * H * sizeof(half);
    size_t jobs_size = size_t(n_jobs) * sizeof(Job);
    size_t meta_size = jobs_size + size_t(rows) * sizeof(RoutedRow);
    size_t size = partial_size + 2 * inter_size + rotated_size + meta_size;
    if (*scratch_bytes < size) {
        if (*scratch)
            CHECK(cudaFree(*scratch));
        *scratch = nullptr;
        *scratch_bytes = 0;
        CHECK(cudaMalloc(scratch, size));
        *scratch_bytes = size;
    }
    char *base = static_cast<char *>(*scratch);
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
    if (inline_metadata && rows <= 64) {
        if (rows <= 8)
            stage_descriptors<8>(jobs, n_jobs, routes, rows, device_jobs, device_rows);
        else
            stage_descriptors<64>(jobs, n_jobs, routes, rows, device_jobs, device_rows);
        CHECK(cudaGetLastError());
    } else {
        CHECK(cudaMemcpy(device_jobs, jobs, jobs_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(device_rows, routes, size_t(rows) * sizeof(RoutedRow), cudaMemcpyHostToDevice));
    }
    if (mcg)
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
                              io.ptr, io.ptr + count, 0, 0, 1);
    if (code)
        return code;
    return int(cudaMemcpy(y, io.ptr + count, count * sizeof(float), cudaMemcpyDeviceToHost));
}
