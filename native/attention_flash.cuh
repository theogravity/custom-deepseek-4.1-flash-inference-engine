// Transient page views for FlashInfer's SM120 sparse MLA prefill. Physical
// slot zero is an initialized finite dummy for masked speculative V loads.
constexpr int flash_page = 64, flash_payload = 576, flash_record = 584;
constexpr int flash_tile = 256;

struct FlashWorkspace {
    Layout layout;
    int history, queries, extra_width;
    size_t swa, compressed, q, result, lse, swa_ids, extra_ids, swa_len, extra_len;
    FlashWorkspace(int pos, int batch, int compressed_rows, int width) {
        history = std::min(win - 1, pos);
        queries = std::min(batch, flash_tile);
        extra_width = (width + 63) / 64 * 64;
        auto pages = [](int rows) { return size_t((rows + 1 + 63) / 64) * flash_page * flash_record; };
        swa = layout.slice<uint8_t>(pages(history + batch));
        compressed = layout.slice<uint8_t>(width ? pages(compressed_rows) : 0);
        size_t features = size_t(queries) * heads * dim;
        q = layout.slice<__nv_bfloat16>(features);
        result = layout.slice<__nv_bfloat16>(features);
        lse = layout.slice<float>(queries * heads);
        swa_ids = layout.slice<int>(queries * win);
        extra_ids = layout.slice<int>(queries * extra_width);
        swa_len = layout.slice<int>(queries);
        extra_len = layout.slice<int>(queries);
    }
};

__device__ void flash_page_row(uint8_t *pages, int slot, uint8_t *&payload, uint8_t *&footer) {
    auto page = pages + size_t(slot / flash_page) * flash_page * flash_record;
    payload = page + (slot % flash_page) * flash_payload;
    footer = page + flash_page * flash_payload + (slot % flash_page) * 8;
}

__global__ void flash_compressed_pages(const uint8_t *records, uint8_t *pages) {
    int slot = blockIdx.x;
    uint8_t *dst, *footer;
    flash_page_row(pages, slot, dst, footer);
    auto *src = slot ? records + size_t(slot-1) * flash_record : nullptr;
    for (int i = threadIdx.x * 4; i < flash_payload; i += blockDim.x * 4)
        *reinterpret_cast<unsigned *>(dst+i) = slot ? *reinterpret_cast<const unsigned *>(src+i) : 0;
    if (threadIdx.x < 8) footer[threadIdx.x] = slot ? src[flash_payload+threadIdx.x] : 0;
}

__global__ void flash_window_pages(const __nv_bfloat16 *prior, const __nv_bfloat16 *current,
        int pos, int history, uint8_t *pages) {
    int slot = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    uint8_t *dst, *footer;
    flash_page_row(pages, slot, dst, footer);
    if (!slot) {
        for (int i = threadIdx.x; i < flash_payload; i += blockDim.x) dst[i] = 0;
        if (threadIdx.x < 8) footer[threadIdx.x] = 0;
        return;
    }
    auto row = window_row(prior, current, pos, pos - history + slot - 1);
    if (warp == 7) {
        for (int i = lane; i < 64; i += 32)
            reinterpret_cast<__nv_bfloat16 *>(dst + 448)[i] = row[448+i];
        if (!lane) footer[7] = 0;
    } else {
        int d = warp * 64 + lane;
        float a = __bfloat162float(row[d]), b = __bfloat162float(row[d+32]);
        float peak = fmaxf(1e-4f, fmaxf(fabsf(a), fabsf(b)));
        for (int i = 16; i; i /= 2)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, i));
        unsigned bits = (__float_as_uint(__fmul_rn(peak, 1.f/448.f)) + 0x7fffffu) & 0x7f800000u;
        auto scale = __uint_as_float(bits);
        dst[d] = __nv_fp8_e4m3(__fdiv_rn(a, scale)).__x;
        dst[d+32] = __nv_fp8_e4m3(__fdiv_rn(b, scale)).__x;
        if (!lane) footer[warp] = bits >> 23;
    }
}

__global__ void flash_indices(int history, int tile, int width, int padded, int compressed_rows,
        const int *picks, int *swa_ids, int *extra_ids, int *swa_len, int *extra_len) {
    int t = blockIdx.x, lane = threadIdx.x;
    int end = history + tile + t + 1, nwindow = min(win, end);
    for (int j = lane; j < win; j += 32)
        swa_ids[t*win+j] = j < nwindow ? 1 + end - nwindow + j : 0;
    if (!lane) swa_len[t] = nwindow;
    int count = 0;
    auto extra = extra_ids + t*padded;
    for (int i = lane; i < padded; i += 32) extra[i] = 0;
    __syncwarp();
    for (int col = 0; col < width; col += 32) {
        int id = col+lane < width ? picks[size_t(tile+t)*width+col+lane] : -1;
        unsigned valid = __ballot_sync(0xffffffffu, id >= 0 && id < compressed_rows);
        if (valid & (1u << lane))
            extra[count + __popc(valid & ((1u << lane)-1))] = id+1;
        count += __popc(valid);
    }
    if (!lane) extra_len[t] = count;
}

template<typename T, typename U>
__global__ void flash_cast(const T *src, U *dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = static_cast<U>(src[i]);
}
