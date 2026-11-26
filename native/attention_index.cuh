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
inline cudaError_t sort_index_rows(void *work,size_t &bytes,const unsigned long long *input,
        unsigned long long *output,int stride,int rows,const int *offsets) {
    if (!wide_index_sort(stride,rows)) return cub::DeviceSegmentedRadixSort::SortKeysDescending(
        work,bytes,input,output,rows*stride,rows,offsets,offsets ? offsets+1 : nullptr);
    for(int t=0;t<(work ? rows : 1);++t) {
        auto e=cub::DeviceRadixSort::SortKeysDescending(work,bytes,input ? input+size_t(t)*stride : nullptr,
            output ? output+size_t(t)*stride : nullptr,stride);
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
        for(int stride : {width,(width+index_block-1)/index_block}) {
            size_t size=0;
            auto e=sort_index_rows(nullptr,size,nullptr,nullptr,stride,rows,nullptr);
            if(e!=cudaSuccess)return e;
            temp_bytes=std::max(temp_bytes,size);
        }
        temp=l.slice<uint8_t>(temp_bytes);
        bytes=l.used;
        return cudaSuccess;
    }
};
