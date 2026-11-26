// SM120 grouped-head sparse attention. The logical key columns, sink and
// 64-column running maxima match the scalar kernel. Each MMA computes sixteen
// heads against eight keys (QK), or produces eight value components (PV).

__device__ bool fast_valid(int p, int t, int span, int first, int nwindow,
                           const int *picks, int width) {
    if (p < span) return p >= first && p < first + nwindow;
    return p < span + width && picks[t * width + p - span] >= 0;
}

// Window rows and compressed rows are combined in a single gather. Pad the
// final tile with zeros so all MMA lanes may read complete fragments.
__global__ void fast_keys(const __nv_bfloat16 *prior, const __nv_bfloat16 *current,
                          const uint8_t *cache, const int *picks, int width,
                          int pos, int tile, int span, bool decode,
                          __nv_bfloat16 *packed) {
    int p = blockIdx.x, t = blockIdx.y, end = pos + tile + t + 1;
    int nwindow = min(win, end), first = decode ? span - nwindow : 0;
    auto *out = packed + (size_t(t) * gridDim.x + p) * dim;
    bool valid = fast_valid(p, t, span, first, nwindow, picks, width);
    if (valid && p < span) {
        const auto *v = window_row(prior, current, pos, end - nwindow + p - first);
        for (int d = threadIdx.x; d < dim; d += blockDim.x) out[d] = v[d];
    } else if (valid) {
        const auto *v = cache + size_t(picks[t * width + p - span]) * 584;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            out[d] = __float2bfloat16_rn(unpack_value(v, d));
    } else {
        for (int d = threadIdx.x; d < dim; d += blockDim.x) out[d] = __float2bfloat16_rn(0.f);
    }
}

__device__ __forceinline__ unsigned fast_bf_pair(float a, float b) {
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
           (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
}

__device__ __forceinline__ void fast_mma(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1, float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void fast_qk(const float *queries, const __nv_bfloat16 *keys,
                        int cols, float *scores) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, t = blockIdx.z;
    const int head = blockIdx.y * 16 + lane / 4, p = blockIdx.x * 64 + warp * 8;
    const auto *q = queries + (size_t(t) * heads + head) * dim + 2 * (lane % 4);
    const auto *k = keys + (size_t(t) * cols + p + lane / 4) * dim + 2 * (lane % 4);
    float c[4] = {};
#pragma unroll 4
    for (int d = 0; d < dim; d += 16) {
        const float *a = q+d, *b = a+8*dim;
        fast_mma(fast_bf_pair(a[0],a[1]), fast_bf_pair(b[0],b[1]),
                 fast_bf_pair(a[8],a[9]), fast_bf_pair(b[8],b[9]),
                 *reinterpret_cast<const unsigned *>(k+d),
                 *reinterpret_cast<const unsigned *>(k+d+8), c);
    }
    auto *out = scores + (size_t(t) * heads + head) * cols + p + 2 * (lane % 4);
    out[0] = c[0] * rsqrtf(float(dim)); out[1] = c[1] * rsqrtf(float(dim));
    out[8*cols] = c[2] * rsqrtf(float(dim)); out[8*cols+1] = c[3] * rsqrtf(float(dim));
}

__global__ void fast_softmax(const float *scores, const int *picks, int width, int pos,
                             int tile, int span, bool decode, int cols, const float *sink,
                             __nv_bfloat16 *probs, float *scales, float *divisors) {
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int end = pos + tile + t + 1, nwindow = min(win, end);
    const int first = decode ? span - nwindow : 0;
    const int tiles = cols / 64;
    const size_t row = size_t(t) * heads + h;
    __shared__ float s[win+top], maxima[(win+top)/64], final_max, partial[block/32];
    for (int p = tid; p < cols; p += block)
        s[p] = fast_valid(p, t, span, first, nwindow, picks, width) ? scores[row*cols+p] : -INFINITY;
    __syncthreads();
    if (tid < tiles) {
        float m = -INFINITY;
        for (int j = 0; j < 64; ++j) m = fmaxf(m, s[tid*64+j]);
        maxima[tid] = m;
    }
    __syncthreads();
    if (!tid) {
        float m = -1e30f;
        for (int j = 0; j < tiles; ++j) { m = fmaxf(m, maxima[j]); maxima[j] = m; }
        final_max = fmaxf(m, sink[h]);
    }
    __syncthreads();
    if (tid < tiles) scales[row*tiles+tid] = expf(maxima[tid] - final_max);
    float sum = 0.f;
    for (int p = tid; p < cols; p += block) {
        probs[row*cols+p] = __float2bfloat16_rn(expf(s[p] - maxima[p/64]));
        sum += expf(s[p] - final_max);
    }
    sum = warp_sum(sum);
    if (tid % 32 == 0) partial[tid/32] = sum;
    __syncthreads();
    if (!tid) {
        sum = expf(sink[h] - final_max);
        for (int j = 0; j < block/32; ++j) sum += partial[j];
        divisors[row] = sum;
    }
}

__device__ __forceinline__ unsigned fast_v_pair(const __nv_bfloat16 *v) {
    return unsigned(__bfloat16_as_ushort(v[0])) | (unsigned(__bfloat16_as_ushort(v[dim])) << 16);
}
__global__ void fast_pv(const __nv_bfloat16 *keys, const __nv_bfloat16 *probs,
                        const float *scales, const float *divisors, int cols, float *output) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, t = blockIdx.z;
    const int f = blockIdx.x * 64 + warp * 8, head = blockIdx.y * 16 + lane / 4;
    const size_t row = size_t(t) * heads + head;
    const int tiles = cols / 64;
    const auto *p = probs + row * cols + 2*(lane%4);
    const auto *v = keys + (size_t(t)*cols + 2*(lane%4))*dim + f + lane/4;
    float acc[4] = {};
    for (int g = 0; g < tiles; ++g) {
        float c[4] = {};
#pragma unroll
        for (int j = 0; j < 64; j += 16) {
            const auto *a = p + g*64+j, *b = a+8*cols;
            const auto *k = v + (g*64+j)*dim;
            fast_mma(*reinterpret_cast<const unsigned *>(a), *reinterpret_cast<const unsigned *>(b),
                     *reinterpret_cast<const unsigned *>(a+8), *reinterpret_cast<const unsigned *>(b+8),
                     fast_v_pair(k), fast_v_pair(k+8*dim), c);
        }
        float sa = scales[row*tiles+g], sb = scales[(row+8)*tiles+g];
        acc[0] = fmaf(c[0],sa,acc[0]); acc[1] = fmaf(c[1],sa,acc[1]);
        acc[2] = fmaf(c[2],sb,acc[2]); acc[3] = fmaf(c[3],sb,acc[3]);
    }
    auto *out = output + row*dim + f + 2*(lane%4);
    out[0] = bf16(acc[0]/divisors[row]); out[1] = bf16(acc[1]/divisors[row]);
    out[8*dim] = bf16(acc[2]/divisors[row+8]); out[8*dim+1] = bf16(acc[3]/divisors[row+8]);
}
