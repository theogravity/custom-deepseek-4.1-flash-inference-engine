// EXL3 MUL1. Each 16x16 checkpoint tile is a cyclic stream of 256 three-bit
// transitions. Its eight consecutive states per lane are already arranged as
// the B operands of two m16n8k16 tensor-core instructions.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>
#include <cstddef>

struct Job {
    const char *expert;
    int begin, count;
};

namespace {
constexpr int H = 5120, I = 2304;
constexpr size_t TRELLIS = size_t(H) * I * 3 / 8;
constexpr size_t EXPERT = 3 * (TRELLIS + (H + I) * sizeof(half));
constexpr float HAD_SCALE = 0.08838834764831845f;
enum Projection { Gate = 0, Up = 1, Down = 2 };
__device__ const half *suh(const char *expert, int projection) {
    return reinterpret_cast<const half *>(expert + 3 * TRELLIS) + projection * (H + I);
}
__device__ const half *svh(const char *expert, int projection) {
    return suh(expert, projection) + (projection == Down ? I : H);
}

__device__ __forceinline__ half mul1(unsigned state) {
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
template <int projection>
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
            b[j] = pair_bits(mul1(unsigned(windows >> (21 - j * 6))),
                             mul1(unsigned(windows >> (18 - j * 6))));
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
__global__ void rotate_input(const char *const *experts, const float *in, half *out) {
    constexpr int K = projection == Down ? I : H;
    int row = blockIdx.y, k = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= K)
        return;
    const half *scale = suh(experts[row], projection);
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = __half2float(__hmul(__float2half(in[row * K + k + i]), scale[k + i]));
    had(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        out[row * K + k + i] = __float2half(v[i]);
}

template <int projection>
__global__ void rotate_output(const char *const *experts, const float *partial, float *out,
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
    const half *scale = svh(experts[row], projection);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (projection == Down)
            out[offset + i] = v[i] * __half2float(scale[n + i]);
        else
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

#define CHECK(op)                                                                                  \
    do {                                                                                           \
        cudaError_t e = (op);                                                                      \
        if (e != cudaSuccess)                                                                      \
            return int(e);                                                                         \
    } while (0)

template <int projection>
void launch_projection(const Job *jobs, const char *const *experts, int n_jobs, int rows,
                       int splits, const float *src, half *rotated, float *partial, float *dst) {
    constexpr int K = projection == Down ? I : H, N = projection == Down ? H : I;
    rotate_input<projection><<<dim3((K + 511) / 512, rows), 128>>>(experts, src, rotated);
    project<projection>
        <<<dim3(N / 64, n_jobs, splits), 128>>>(jobs, rotated, partial, rows, splits);
    rotate_output<projection>
        <<<dim3((N + 511) / 512, rows), 128>>>(experts, partial, dst, rows, splits);
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

// Packed order: three trellises then per-projection input and output scales.
extern "C" int ds_moe_upload(char *slot, const void *const *parts) {
    size_t offset = 0;
    for (int p = 0; p < 9; ++p) {
        size_t bytes = TRELLIS;
        if (p >= 3) {
            int projection = (p - 3) / 2;
            bool input = (p - 3) % 2 == 0;
            bytes = sizeof(half) * ((projection == Down) == input ? I : H);
        }
        CHECK(cudaMemcpy(slot + offset, parts[p], bytes, cudaMemcpyHostToDevice));
        offset += bytes;
    }
    return 0;
}

// Input and output rows are expert-major and correspond one-to-one. All jobs
// and GPU slots remain owned by the Rust caller until the final D2H completes.
extern "C" int ds_moe_run(void **scratch, size_t *scratch_bytes, const Job *jobs, int n_jobs,
                          const char *const *experts, const float *x, float *y, int rows) {
    int splits = n_jobs < 32 ? 8 : 2;
    size_t partial_size = size_t(rows) * H * splits * sizeof(float);
    size_t input_size = size_t(rows) * H * sizeof(float),
           inter_size = size_t(rows) * I * sizeof(float);
    size_t rotated_size = size_t(rows) * H * sizeof(half), meta_size = size_t(n_jobs) * sizeof(Job);
    size_t size = partial_size + input_size + 2 * inter_size + rotated_size + meta_size +
                  rows * sizeof(char *);
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
    float *io = reinterpret_cast<float *>(base);
    base += input_size;
    float *gate = reinterpret_cast<float *>(base);
    base += inter_size;
    float *up = reinterpret_cast<float *>(base);
    base += inter_size;
    half *rotated = reinterpret_cast<half *>(base);
    base += rotated_size;
    Job *device_jobs = reinterpret_cast<Job *>(base);
    base += meta_size;
    const char **device_experts = reinterpret_cast<const char **>(base);
    CHECK(cudaMemcpy(io, x, input_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_jobs, jobs, meta_size, cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(device_experts, experts, size_t(rows) * sizeof(char *), cudaMemcpyHostToDevice));
    launch_projection<Gate>(device_jobs, device_experts, n_jobs, rows, splits, io, rotated, partial,
                            gate);
    launch_projection<Up>(device_jobs, device_experts, n_jobs, rows, splits, io, rotated, partial,
                          up);
    swiglu<<<(rows * I + 255) / 256, 256>>>(gate, up, rows * I);
    launch_projection<Down>(device_jobs, device_experts, n_jobs, rows, splits, gate, rotated,
                            partial, io);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(y, io, input_size, cudaMemcpyDeviceToHost));
    return 0;
}
