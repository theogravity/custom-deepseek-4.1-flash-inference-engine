// Each group computes the same 32 virtual lanes as the scalar reference.
// The first reduction levels happen in registers; the remaining levels use
// subwarp shuffles. Incomplete groups stay active in every warp instruction.
template<int lanes>
__global__ void score_grouped_keys(const uint8_t *keys, const float *queries,
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
