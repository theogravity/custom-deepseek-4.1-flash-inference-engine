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
template<int lanes, bool reuse_keys = false>
__global__ void score_packed_keys(const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, int rows, float *scores) {
    static_assert(lanes == 8 || lanes == 16);
    constexpr int fragments = 32 / lanes;
    constexpr int pairs = score_pairs / fragments;
    constexpr int query_rows = reuse_keys ? 2 : 1;
    const int first_row = blockIdx.y * query_rows;
    const int lane = threadIdx.x % lanes;
    const int first = (blockIdx.x * (score_threads / lanes) + threadIdx.x / lanes) * (2 * pairs);
    int indices[pairs][2];
    ScorePair key[pairs][fragments][4];
    #pragma unroll
    for (int p = 0; p < pairs; ++p) {
        const int slot = first + p * 2;
        auto index = [&](int col) {
            // Only full scans use the two-row specialization. Each visible ID
            // equals its column, so the packed key is shared even when the
            // first row masks a position that the second row can see.
            return col < width ? (!reuse_keys && ids ? ids[size_t(first_row) * width + col] : col) : -1;
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
    // Keep both packed halves on adjacent columns for a single query. The
    // next query uses the same dequantized key registers and its own weights.
    for (int r = 0; r < query_rows; ++r) {
        const int row = first_row + r;
        if (reuse_keys && row >= rows) break;
        ScorePair sums[pairs] = {};
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
                auto visible = [&](int col, int side) {
                    return reuse_keys ? (!ids || ids[size_t(row) * width + col] >= 0) : indices[p][side] >= 0;
                };
                if (slot < width)
                    scores[size_t(row) * width + slot] = visible(slot, 0) ? low_score(sums[p]) : -INFINITY;
                if (slot + 1 < width)
                    scores[size_t(row) * width + slot+1] = visible(slot + 1, 1) ? high_score(sums[p]) : -INFINITY;
            }
        }
    }
}

void launch_index_scores(bool packed, const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, int rows, float *scores, bool contiguous_ids) {
    if (packed) {
        constexpr int slots = score_threads / 32 * 2 * score_pairs;
        static const bool pair_rows = [] {
            const auto *flag = std::getenv("DS4_PAIRED_INDEX");
            return flag && std::strcmp(flag, "1") == 0;
        }();
        if (pair_rows && contiguous_ids && rows >= 4 && rows <= 8 && width >= 65536) {
            dim3 paired_grid((width + slots - 1) / slots, (rows + 1) / 2);
            score_packed_keys<8, true><<<paired_grid, score_threads>>>(keys, queries, weights, ids, width, rows, scores);
            return;
        }
        dim3 grid((width + slots - 1) / slots, rows);
        // Long scans amortize the extra query loads in the smaller group.
        if (rows >= 4 && width >= 65536 && int64_t(width) * rows >= 524288)
            score_packed_keys<8><<<grid, score_threads>>>(keys, queries, weights, ids, width, rows, scores);
        else
            score_packed_keys<16><<<grid, score_threads>>>(keys, queries, weights, ids, width, rows, scores);
    } else {
        constexpr int slots = block / 32;
        score_keys<<<dim3((width + slots - 1) / slots, rows), block>>>(
            keys, queries, weights, ids, width, scores);
    }
}
