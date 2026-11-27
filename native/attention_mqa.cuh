// Four-query, split-KV port of DeepGEMM sm120_fp8_mqa_logits for the
// engine's interleaved 132-byte index records. See THIRD_PARTY_NOTICES.md.
// Each of eight warps owns sixteen keys. Keep their four K fragments in
// registers throughout the query tile. Keep queries shared across KV tiles.
constexpr int mqa_queries = 4, mqa_keys = mma_warps * 16, mqa_key_tiles = 4;

__global__ void score_batched_mqa(const uint8_t *keys, const IndexFp8Query *queries,
        const float *weights, const int *causal_ids, int width, int rows, float *scores) {
    const int first_q = blockIdx.y * mqa_queries;
    const int count = min(mqa_queries, rows - first_q);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ IndexFp8Query q[mqa_queries];
    __shared__ float weight[mqa_queries][ih], dots[mma_warps][ih][20];

    for (int r = 0; r < count; ++r) {
        const auto &src = queries[first_q + r];
        for (int i = threadIdx.x; i < ih * idim / 4; i += blockDim.x)
            reinterpret_cast<unsigned *>(q[r].q)[i] = reinterpret_cast<const unsigned *>(src.q)[i];
        if (threadIdx.x < ih) {
            q[r].scales[threadIdx.x] = src.scales[threadIdx.x];
            weight[r][threadIdx.x] = bf16(bf16(weights[(first_q+r)*ih+threadIdx.x]) / 64.f);
        }
        if (threadIdx.x == 0) q[r].compatible = src.compatible;
    }
    __syncthreads();

    for (int tile = 0; tile < mqa_key_tiles; ++tile) {
        const int first = (blockIdx.x * mqa_key_tiles + tile) * mqa_keys + warp * 16;
        if (first >= width) break;
        const int key = first + lane / 4;
        const auto *low = reinterpret_cast<const unsigned *>(keys + size_t(min(key,width-1)) * 132) + lane % 4;
        const auto *high = reinterpret_cast<const unsigned *>(keys + size_t(min(key+8,width-1)) * 132) + lane % 4;
        const int slot = first + lane;
        const float key_scale = lane < 16 && slot < width
            ? *reinterpret_cast<const float *>(keys + size_t(slot) * 132 + idim) : 0.f;
        unsigned a[idim/32][4];
        #pragma unroll
        for (int k = 0; k < idim / 32; ++k) {
            a[k][0] = low[k*8];   a[k][1] = high[k*8];
            a[k][2] = low[k*8+4]; a[k][3] = high[k*8+4];
        }
        for (int r = 0; r < count; ++r) {
            if (!q[r].compatible) continue; // the BF16 launch handles this row
            #pragma unroll
            for (int g = 0; g < ih/8; ++g) {
                float d[4] = {};
                #pragma unroll
                for (int k = 0; k < idim/32; ++k)
                    index_fp8_mma(a[k][0],a[k][1],a[k][2],a[k][3],
                                  q[r].q[g][k][0][lane],q[r].q[g][k][1][lane],d);
                const int h = g*8 + 2*(lane%4), dst = lane/4;
                dots[warp][h][dst] = d[0]*q[r].scales[h];
                dots[warp][h+1][dst] = d[1]*q[r].scales[h+1];
                dots[warp][h][dst+8] = d[2]*q[r].scales[h];
                dots[warp][h+1][dst+8] = d[3]*q[r].scales[h+1];
            }
            __syncwarp();
            if (lane < 16 && slot < width) {
                const size_t offset = size_t(first_q+r)*width + slot;
                float score = -INFINITY;
                if (!causal_ids || causal_ids[offset] >= 0) {
                    score = 0.f;
                    #pragma unroll
                    for (int h = 0; h < ih; ++h)
                        score = fmaf(fmaxf(0.f,dots[warp][h][lane]*key_scale),weight[r][h],score);
                }
                scores[offset] = score;
            }
            __syncwarp();
        }
    }
}
