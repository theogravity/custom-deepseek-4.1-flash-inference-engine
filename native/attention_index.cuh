// Keys sort in descending Rust f32::total_cmp order, then earliest index.
// Zero is reserved for suppressed or invisible positions.
__device__ unsigned long long index_order(float score, int id) {
    unsigned bits = __float_as_uint(score);
    bits ^= (bits >> 31) ? 0xffffffffu : 0x80000000u;
    return (static_cast<unsigned long long>(bits) << 32) | ~unsigned(id);
}

constexpr int index_block = 8, candidate_limit = 2048;
__global__ void index_ids(const int *blocks, int *ids, int width, int pos, int ratio) {
    int t = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    int k = blocks ? blocks[t*candidate_limit + col/index_block]*index_block + col%index_block : col;
    ids[size_t(t)*width + col] = k < (pos + t + 1)/ratio ? k : -1;
}

__global__ void index_bounds(int *offsets, int width, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i <= n) offsets[i] = i*width;
}

__global__ void rank_positions(const float *scores,const int *ids,unsigned long long *rank,int width) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= width) return;
    size_t off = size_t(blockIdx.y)*width + col;
    rank[off] = ids[off] < 0 ? 0 : index_order(scores[off],ids[off]);
}

__global__ void rank_blocks(const float *scores,unsigned long long *rank,int width,int pos,int ratio) {
    int id = blockIdx.x*blockDim.x + threadIdx.x, t=blockIdx.y;
    int blocks=(width+index_block-1)/index_block;
    if (id >= blocks) return;
    int visible=(pos+t+1)/ratio, first=id*index_block;
    float peak=-INFINITY;
    for (int j=first; j<min(first+index_block,visible); ++j)
        peak=fmaxf(peak,scores[size_t(t)*width+j]);
    if (first < visible && first+index_block >= visible) peak=INFINITY;
    rank[size_t(t)*blocks+id] = first < visible ? index_order(peak,id) : 0;
}

template<int Limit>
__global__ void chronological_ids(const unsigned long long *rank, int width, int *out) {
    // Blocked items sort into ascending chronological order; masked entries
    // sort last, including near the first 512-token compression threshold.
    constexpr int items=Limit/block;
    using Sort=cub::BlockRadixSort<unsigned,block,items>;
    __shared__ typename Sort::TempStorage storage;
    unsigned keys[items];
    for (int j=0;j<items;++j) {
        int slot=threadIdx.x*items+j;
        auto k=slot < width ? rank[size_t(blockIdx.x)*width+slot] : 0ull;
        keys[j]=k ? ~unsigned(k) : 0xffffffffu;
    }
    Sort(storage).Sort(keys);
    for (int j=0;j<items;++j)
        out[size_t(blockIdx.x)*Limit+threadIdx.x*items+j]=int(keys[j]);
}

// On sm_120, segmented sort wins for batches of short rows. Separate device
// sorts recover their launch cost when a row has about 8192 keys per batch row
// (32k x 4, 64k x 8, 128k x 16). Small decode batches need only 8192 keys.
inline bool wide_index_sort(int stride, int rows) {
    return stride >= 8192 && (rows <= 2 || stride >= int64_t(rows) * 8192);
}
// Each tile contributes at most keep keys to the global winners. Ranking keys
// already encode the tie break, so selecting again from their union is exact.
constexpr int sort_tile = 4096;
inline bool use_tiled_selection(int width, int rows, int keep) {
    static const bool enabled = [] {
        const char *v = std::getenv("DS4_BATCH_TOPK");
        return v && std::strcmp(v, "1") == 0;
    }();
    static const bool wide_enabled = [] {
        const char *v = std::getenv("DS4_WIDE_TOPK");
        return v && std::strcmp(v, "1") == 0;
    }();
    if (enabled && wide_enabled && keep == top && rows <= 16 && width <= 524288)
        return true;
    return enabled && rows > 1 && !wide_index_sort(width, rows) &&
           (width <= sort_tile || (keep == top && width <= sort_tile * (sort_tile / top)));
}

template<int Items>
__global__ void select_tile(const unsigned long long *in, unsigned long long *out,
        int width, int in_stride, int out_stride, int keep) {
    using Sort = cub::BlockRadixSort<unsigned long long, block, Items>;
    __shared__ typename Sort::TempStorage storage;
    const int first = blockIdx.x * block * Items;
    const int count = min(block * Items, width - first);
    const auto *src = in + size_t(blockIdx.y) * in_stride + first;
    auto *dst = out + size_t(blockIdx.y) * out_stride + blockIdx.x * keep;
    unsigned long long keys[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x + i * block;
        keys[i] = j < count ? src[j] : 0;
    }
    Sort(storage).SortDescendingBlockedToStriped(keys);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x + i * block;
        if (j < min(count, keep)) dst[j] = keys[i];
    }
}

inline cudaError_t tiled_selection(const unsigned long long *input, unsigned long long *output,
        unsigned long long *tile_winners, int width, int rows, int keep) {
    int stride = width, count = width;
    auto *next = tile_winners;
    unsigned long long *second = nullptr;
    while (count > sort_tile) {
        const int tiles = (count + sort_tile - 1) / sort_tile;
        const int next_stride = tiles * keep;
        select_tile<sort_tile / block><<<dim3(tiles, rows), block>>>(
            input, next, count, stride, next_stride, keep);
        auto rc = cudaGetLastError();
        if (rc != cudaSuccess) return rc;
        count = (tiles - 1) * keep + std::min(keep, count - (tiles - 1) * sort_tile);
        stride = next_stride;
        input = next;
        if (count > sort_tile) {
            // For keep=512, multiple passes require at least 32,769 keys:
            // the first two winner arrays together fit in the dead 32-bit IDs.
            if (!second) second = tile_winners + size_t(rows) * stride;
            next = next == tile_winners ? second : tile_winners;
        }
    }
    dim3 grid(1, rows);
    if (count <= 1024)
        select_tile<4><<<grid, block>>>(input, output, count, stride, width, keep);
    else if (count <= 2048)
        select_tile<8><<<grid, block>>>(input, output, count, stride, width, keep);
    else
        select_tile<16><<<grid, block>>>(input, output, count, stride, width, keep);
    return cudaGetLastError();
}

inline cudaError_t sort_index_rows(void *work,size_t &bytes,const unsigned long long *input,
        unsigned long long *output,int stride,int rows,const int *offsets,int keep) {
    if (!wide_index_sort(stride,rows)) return cub::DeviceSegmentedRadixSort::SortKeysDescending(
        work,bytes,input,output,rows*stride,rows,offsets,offsets ? offsets+1 : nullptr);
    for(int t=0;t<(work ? rows : 1);++t) {
        auto in = input ? input + size_t(t) * stride : nullptr;
        auto out = output ? output + size_t(t) * stride : nullptr;
        cudaError_t e;
#if __has_include(<cub/device/device_topk.cuh>)
        // Rank keys include the index as a tie breaker. Masked keys are all zero.
        // chronological_ids sorts the selected IDs independently of their order here.
        auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                           cuda::execution::output_ordering::unsorted);
        e = cub::DeviceTopK::MaxKeys(work, bytes, in, out, stride, std::min(stride, keep), env);
#else
        e = cub::DeviceRadixSort::SortKeysDescending(work, bytes, in, out, stride);
#endif
        if (e!=cudaSuccess) return e;
    }
    return cudaSuccess;
}

struct IndexWorkspace {
    size_t ids, scores, ranks, sorted, offsets, temp, temp_bytes, bytes;
    cudaError_t plan(int width,int rows) {
        Layout l;
        size_t count=size_t(width)*rows;
        ids=l.slice<int>(count);
        scores=l.slice<float>(count);
        ranks=l.slice<unsigned long long>(count);
        sorted=l.slice<unsigned long long>(count);
        offsets=l.slice<int>(rows+1);
        temp_bytes=0;
        for (int stage = 0; stage < 2; ++stage) {
            int stride = stage == 0 ? width : (width + index_block - 1) / index_block;
            int keep = stage == 0 ? top : candidate_limit;
            if (use_tiled_selection(stride,rows,keep)) continue;
            size_t size=0;
            auto e=sort_index_rows(nullptr,size,nullptr,nullptr,stride,rows,nullptr,keep);
            if(e!=cudaSuccess)return e;
            temp_bytes=std::max(temp_bytes,size);
        }
        temp=l.slice<uint8_t>(temp_bytes);
        bytes=l.used;
        return cudaSuccess;
    }
};
