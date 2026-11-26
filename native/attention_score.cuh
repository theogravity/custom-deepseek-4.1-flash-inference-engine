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
template<int lanes, bool adjacent_rows = false>
__global__ void score_packed_keys(const uint8_t *keys, const float *queries,
        const float *weights, const int *ids, int width, int rows, float *scores) {
    static_assert(lanes == 8 || lanes == 16);
    constexpr int fragments = 32 / lanes;
    constexpr int pairs = score_pairs / fragments;
    constexpr int columns_per_pair = adjacent_rows ? 1 : 2;
    const int row = blockIdx.y * (adjacent_rows ? 2 : 1);
    const int other = adjacent_rows ? min(row + 1, rows - 1) : row;
    const int lane = threadIdx.x % lanes;
    const int first = (blockIdx.x * (score_threads / lanes) + threadIdx.x / lanes) * (columns_per_pair * pairs);
    int indices[pairs][2];
    ScorePair key[pairs][fragments][4], sums[pairs] = {};
    #pragma unroll
    for (int p = 0; p < pairs; ++p) {
        const int slot = first + p * columns_per_pair;
        auto index = [&](int query_row, int col) {
            return col < width ? (ids ? ids[size_t(query_row) * width + col] : col) : -1;
        };
        const int low = indices[p][0] = index(row, slot);
        const int high = indices[p][1] = index(other, adjacent_rows ? slot : slot + 1);
        // On a full scan each valid ID equals its column. Adjacent causal
        // rows can disagree on visibility, including at compression boundaries.
        const int shared_id = max(low, high);
        const uint8_t *a = keys + size_t(max(0, adjacent_rows ? shared_id : low)) * 132;
        const uint8_t *b = adjacent_rows ? a : keys + size_t(max(0, high)) * 132;
        const float scale_a = *reinterpret_cast<const float *>(a + idim);
        const float scale_b = *reinterpret_cast<const float *>(b + idim);
        #pragma unroll
        for (int f = 0; f < fragments; ++f) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = lane + f * lanes + j * 32;
                const float v = fp8_value(a[k]) * scale_a;
                key[p][f][j] = pack_scores(v, adjacent_rows ? v : fp8_value(b[k]) * scale_b);
            }
        }
    }
    #pragma unroll 2
    for (int h = 0; h < ih; ++h) {
        const float *q = queries + (size_t(row) * ih + h) * idim + lane;
        const float *q_other = queries + (size_t(other) * ih + h) * idim + lane;
        ScorePair dots[pairs][fragments] = {};
        #pragma unroll
        for (int f = 0; f < fragments; ++f) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v = q[f * lanes + j * 32];
                const ScorePair query = pack_scores(v, adjacent_rows ? q_other[f * lanes + j * 32] : v);
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
                const float high_weight = adjacent_rows ? bf16(bf16(weights[other * ih + h]) / 64.f) : weight;
                sums[p] = pair_fma(positive, pack_scores(weight, high_weight), sums[p]);
            }
        }
    }
    if (!lane) {
        #pragma unroll
        for (int p = 0; p < pairs; ++p) {
            const int slot = first + columns_per_pair * p;
            const int high_slot = adjacent_rows ? slot : slot + 1;
            if (slot < width)
                scores[size_t(row) * width + slot] = indices[p][0] < 0 ? -INFINITY : low_score(sums[p]);
            if (high_slot < width && (!adjacent_rows || other != row))
                scores[size_t(other) * width + high_slot] = indices[p][1] < 0 ? -INFINITY : high_score(sums[p]);
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
        if (pair_rows && contiguous_ids && rows >= 4 && width >= 65536) {
            constexpr int columns = slots / 2;
            dim3 paired_grid((width + columns - 1) / columns, (rows + 1) / 2);
            if (int64_t(width) * rows >= 524288)
                score_packed_keys<8, true><<<paired_grid, score_threads>>>(keys, queries, weights, ids, width, rows, scores);
            else
                score_packed_keys<16, true><<<paired_grid, score_threads>>>(keys, queries, weights, ids, width, rows, scores);
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
