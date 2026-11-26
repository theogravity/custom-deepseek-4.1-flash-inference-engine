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

__global__ void score_packed_keys(const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, float *scores) {
    const int row = blockIdx.y, lane = threadIdx.x % 32;
    const int first = (blockIdx.x * (score_threads / 32) + threadIdx.x / 32) * score_pairs * 2;
    if (first >= width) return;
    int indices[score_pairs][2];
    ScorePair key[score_pairs][4], sums[score_pairs] = {};
    #pragma unroll
    for (int p = 0; p < score_pairs; ++p) {
        const int slot = first + p * 2;
        auto index = [&](int col) {
            return col < width ? (ids ? ids[row * width + col] : col) : -1;
        };
        const int low = indices[p][0] = index(slot);
        const int high = indices[p][1] = index(slot + 1);
        // Causal padding is masked on output. Every lane must participate in
        // the warp tree, even when only one of the two candidate IDs is valid.
        const uint8_t *a = keys + size_t(max(0, low)) * 132;
        const uint8_t *b = keys + size_t(max(0, high)) * 132;
        const float scale_a = *reinterpret_cast<const float *>(a + idim);
        const float scale_b = *reinterpret_cast<const float *>(b + idim);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            key[p][j] = pack_scores(fp8_value(a[lane + j * 32]) * scale_a,
                                    fp8_value(b[lane + j * 32]) * scale_b);
        }
    }
    #pragma unroll 2
    for (int h = 0; h < ih; ++h) {
        const float *q = queries + (size_t(row) * ih + h) * idim + lane;
        ScorePair dots[score_pairs] = {};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const ScorePair query = pack_scores(q[j * 32], q[j * 32]);
            #pragma unroll
            for (int p = 0; p < score_pairs; ++p)
                dots[p] = pair_fma(key[p][j], query, dots[p]);
        }
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1) {
            #pragma unroll
            for (int p = 0; p < score_pairs; ++p)
                dots[p] = pair_add(dots[p], __shfl_down_sync(0xffffffffu, dots[p], delta));
        }
        if (!lane) {
            const float weight = bf16(bf16(weights[row * ih + h]) / 64.f);
            const ScorePair pair_weight = pack_scores(weight, weight);
            #pragma unroll
            for (int p = 0; p < score_pairs; ++p) {
                const ScorePair positive = pack_scores(fmaxf(0.f, low_score(dots[p])),
                                                      fmaxf(0.f, high_score(dots[p])));
                sums[p] = pair_fma(positive, pair_weight, sums[p]);
            }
        }
    }
    if (!lane) {
        #pragma unroll
        for (int p = 0; p < score_pairs; ++p) {
            const int slot = first + 2 * p;
            if (slot < width)
                scores[row * width + slot] = indices[p][0] < 0 ? -INFINITY : low_score(sums[p]);
            if (slot + 1 < width)
                scores[row * width + slot + 1] = indices[p][1] < 0 ? -INFINITY : high_score(sums[p]);
        }
    }
}

void launch_index_scores(bool packed, const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, int rows, float *scores) {
    if (packed) {
        constexpr int slots = score_threads / 32 * 2 * score_pairs;
        score_packed_keys<<<dim3((width + slots - 1) / slots, rows), score_threads>>>(
            keys, queries, weights, ids, width, scores);
    } else {
        constexpr int slots = block / 32;
        score_keys<<<dim3((width + slots - 1) / slots, rows), block>>>(
            keys, queries, weights, ids, width, scores);
    }
}
