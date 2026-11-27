// Blackwell's packed FP32 instructions round each half independently. Every
// half follows score_keys' four FMAs, warp tree and ordered sum over heads.
using ScorePair = unsigned long long;
constexpr int score_pairs = 4, score_threads = 128;

__device__ __forceinline__ ScorePair pack_scores(float low, float high) {
    ScorePair result;
    asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "f"(low), "f"(high));
    return result;
}
__device__ __forceinline__ ScorePair pair_fma(ScorePair a, ScorePair b, ScorePair sum) {
    ScorePair result;
    asm("fma.rn.f32x2 %0,%1,%2,%3;" : "=l"(result) : "l"(a), "l"(b), "l"(sum));
    return result;
}
__device__ __forceinline__ ScorePair pair_add(ScorePair a, ScorePair b) {
    ScorePair result;
    asm("add.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}
__device__ __forceinline__ float low_score(ScorePair x) {
    return __uint_as_float(static_cast<unsigned>(x));
}
__device__ __forceinline__ float high_score(ScorePair x) {
    return __uint_as_float(static_cast<unsigned>(x >> 32));
}

// Each group computes the same 32 virtual lanes as the scalar reference.
// The first reduction levels happen in registers; the remaining levels use
// subwarp shuffles. Incomplete groups stay active in every warp instruction.
template<int lanes>
__global__ void score_packed_keys(const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, float *scores) {
    static_assert(lanes == 8 || lanes == 16);
    constexpr int fragments = 32 / lanes;
    constexpr int pairs = score_pairs / fragments;
    const int row = blockIdx.y;
    const int lane = threadIdx.x % lanes;
    const int first = (blockIdx.x * (score_threads / lanes) + threadIdx.x / lanes) * (2 * pairs);
    int indices[pairs][2];
    ScorePair key[pairs][fragments][4], sums[pairs] = {};
    #pragma unroll
    for (int p = 0; p < pairs; ++p) {
        const int slot = first + p * 2;
        auto index = [&](int col) {
            return col < width ? (ids ? ids[size_t(row) * width + col] : col) : -1;
        };
        const int low = indices[p][0] = index(slot);
        const int high = indices[p][1] = index(slot + 1);
        const uint8_t *a = keys + size_t(max(0, low)) * 132;
        const uint8_t *b = keys + size_t(max(0, high)) * 132;
        const float scale_a = *reinterpret_cast<const float *>(a + idim);
        const float scale_b = *reinterpret_cast<const float *>(b + idim);
        #pragma unroll
        for (int f = 0; f < fragments; ++f) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = lane + f * lanes + j * 32;
                key[p][f][j] = pack_scores(fp8_value(a[k]) * scale_a, fp8_value(b[k]) * scale_b);
            }
        }
    }
    #pragma unroll 2
    for (int h = 0; h < ih; ++h) {
        const float *q = queries + (size_t(row) * ih + h) * idim + lane;
        ScorePair dots[pairs][fragments] = {};
        #pragma unroll
        for (int f = 0; f < fragments; ++f) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v = q[f * lanes + j * 32];
                const ScorePair query = pack_scores(v, v);
                #pragma unroll
                for (int p = 0; p < pairs; ++p)
                    dots[p][f] = pair_fma(key[p][f][j], query, dots[p][f]);
            }
        }
        #pragma unroll
        for (int p = 0; p < pairs; ++p) {
            #pragma unroll
            for (int distance = fragments / 2; distance; distance /= 2) {
                #pragma unroll
                for (int f = 0; f < distance; ++f)
                    dots[p][f] = pair_add(dots[p][f], dots[p][f+distance]);
            }
            ScorePair sum = dots[p][0];
            #pragma unroll
            for (int delta = lanes / 2; delta; delta /= 2)
                sum = pair_add(sum, __shfl_down_sync(0xffffffffu, sum, delta, lanes));
            if (!lane) {
                const float weight = bf16(bf16(weights[row * ih + h]) / 64.f);
                const ScorePair positive = pack_scores(fmaxf(0.f, low_score(sum)), fmaxf(0.f, high_score(sum)));
                sums[p] = pair_fma(positive, pack_scores(weight, weight), sums[p]);
            }
        }
    }
    if (!lane) {
        #pragma unroll
        for (int p = 0; p < pairs; ++p) {
            const int slot = first + 2 * p;
            if (slot < width)
                scores[size_t(row) * width + slot] = indices[p][0] < 0 ? -INFINITY : low_score(sums[p]);
            if (slot + 1 < width)
                scores[size_t(row) * width + slot+1] = indices[p][1] < 0 ? -INFINITY : high_score(sums[p]);
        }
    }
}

// Each warp scores 16 candidate keys against all 32 query heads. A prompt
// query has already been rounded to E4M3 times a power of two, so BF16
// represents it exactly. Keep each key unscaled through the MMA and apply
// its FP32 scale before ReLU. Production key scales are also powers of two.
constexpr int mma_warps = 8, mma_tiles = 2;
__device__ __forceinline__ unsigned index_key_pair(const uint8_t *p) {
    const auto value = __nv_cvt_fp8x2_to_bf162raw(*reinterpret_cast<const uint16_t *>(p), __NV_E4M3);
    return (unsigned(value.y) << 16) | value.x;
}

__global__ void score_mma_keys(const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, float *scores) {
    const int t = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // Arrange query pairs in the order of MMA's B registers. All warps
    // then load contiguous words for each 8-head, 16-feature fragment.
    __shared__ unsigned q[4][8][2][32];
    __shared__ float weight[ih], dots[mma_warps][ih][20];
    for (int i = threadIdx.x; i < ih * idim / 2; i += blockDim.x) {
        const int h = i / (idim / 2), k = i % (idim / 2);
        const float2 pair = reinterpret_cast<const float2 *>(queries + size_t(t) * ih * idim)[i];
        q[h / 8][k / 8][(k / 4) % 2][(h % 8) * 4 + k % 4] = fast_bf_pair(pair.x, pair.y);
    }
    if (threadIdx.x < ih)
        weight[threadIdx.x] = bf16(bf16(weights[t * ih + threadIdx.x]) / 64.f);
    __syncthreads();

    for (int tile = 0; tile < mma_tiles; ++tile) {
        const int first = ((blockIdx.x * mma_tiles + tile) * mma_warps + warp) * 16;
        if (first >= width) return;
        const int slot = first + lane;
        const int id = lane < 16 && slot < width ? (ids ? ids[size_t(t)*width + slot] : slot) : -1;
        const int lo = __shfl_sync(0xffffffffu, id, lane / 4);
        const int hi = __shfl_sync(0xffffffffu, id, lane / 4 + 8);
        // Masked lanes read the first allocated row for their MMA fragment.
        // A masked result is never committed as a score or used by another key.
        const auto *a = keys + size_t(max(0, lo)) * 132 + 2 * (lane % 4);
        const auto *b = keys + size_t(max(0, hi)) * 132 + 2 * (lane % 4);
        float c[4][4] = {};
        #pragma unroll
        for (int k = 0; k < idim / 16; ++k) {
            unsigned a0 = index_key_pair(a + k*16), a1 = index_key_pair(b + k*16);
            unsigned a2 = index_key_pair(a + k*16+8), a3 = index_key_pair(b + k*16+8);
            #pragma unroll
            for (int g = 0; g < ih / 8; ++g)
                fast_mma(a0,a1,a2,a3,q[g][k][0][lane],q[g][k][1][lane],c[g]);
        }
        // Transpose the output within the warp so one lane accumulates each
        // key's heads in reference order. A stride of 20 makes the writes
        // by the four two-head lane groups land in distinct shared banks.
        #pragma unroll
        for (int g = 0; g < ih / 8; ++g) {
            const int h = 8*g + 2*(lane%4), key = lane / 4;
            dots[warp][h][key] = c[g][0];     dots[warp][h+1][key] = c[g][1];
            dots[warp][h][key+8] = c[g][2];   dots[warp][h+1][key+8] = c[g][3];
        }
        __syncwarp();
        if (lane < 16 && slot < width) {
            float score = -INFINITY;
            if (id >= 0) {
                const float scale = *reinterpret_cast<const float *>(keys + size_t(id)*132 + idim);
                score = 0.f;
                #pragma unroll
                for (int h = 0; h < ih; ++h)
                    score = fmaf(fmaxf(0.f, dots[warp][h][lane] * scale), weight[h], score);
            }
            scores[size_t(t)*width + slot] = score;
        }
        __syncwarp();
    }
}

void launch_index_scores(int mode, const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, int rows, float *scores) {
    if (mode == 2) {
        constexpr int slots = mma_warps * mma_tiles * 16;
        score_mma_keys<<<dim3((width + slots - 1) / slots, rows), mma_warps * 32>>>(
            keys, queries, weights, ids, width, scores);
    } else if (mode == 1) {
        constexpr int slots = score_threads / 32 * 2 * score_pairs;
        dim3 grid((width + slots - 1) / slots, rows);
        // Long scans amortize the extra query loads in the smaller group.
        if (rows >= 4 && width >= 65536 && int64_t(width) * rows >= 524288)
            score_packed_keys<8><<<grid, score_threads>>>(keys, queries, weights, ids, width, scores);
        else
            score_packed_keys<16><<<grid, score_threads>>>(keys, queries, weights, ids, width, scores);
    } else {
        constexpr int slots = block / 32;
        score_keys<<<dim3((width + slots - 1) / slots, rows), block>>>(
            keys, queries, weights, ids, width, scores);
    }
}
