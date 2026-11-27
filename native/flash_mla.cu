// Framework-independent adapter for FlashInfer's two DSV4 BF16 launchers.
#include <cuda_runtime.h>
#include <mutex>
#define CUDA_CHECK(call) do { cudaError_t rc = (call); if (rc != cudaSuccess) throw rc; } while (0)
#include "flashinfer_launchers.cuh"

extern "C" cudaError_t ds4_flash_launch(const __nv_bfloat16 *q,
        const uint8_t *swa, const int *swa_ids, const int *swa_len,
        const uint8_t *compressed, const int *extra_ids, const int *extra_len,
        int extra_width, int rows, const float *sink, __nv_bfloat16 *out, float *lse) {
    using namespace flashinfer::sparse_mla_sm120;
    constexpr size_t stride = 64 * 584;
    constexpr float scale = 0.04419417382415922f; // 1/sqrt(512)
    // Upstream caches dynamic shared-memory configuration per device in
    // ordinary static arrays. Serialize host launch setup across runtimes.
    static std::mutex lock;
    const std::lock_guard<std::mutex> guard(lock);
    try {
        if (extra_width) {
            launch_prefill_mg_dual<ModelType::DSV4, ComputeMode::BF16, 64, 64, 64>(
                q, swa, swa_ids, compressed, extra_ids, sink, out, lse, scale,
                rows, 128, extra_width, stride, stride, 64, swa_len, extra_len, nullptr);
        } else {
            launch_prefill_mg<ModelType::DSV4, ComputeMode::BF16, 64, 64>(
                q, swa, swa_ids, sink, out, lse, scale, rows, 128, stride, 64, swa_len, nullptr);
        }
    } catch (cudaError_t error) { return error; }
    return cudaSuccess;
}
