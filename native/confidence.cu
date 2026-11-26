// Released DSpark confidence logit: linear([unnormalized collapsed hidden,
// Markov embedding of the previous proposal]). One block per proposal.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
namespace {
__global__ void confidence(const float *states, const float *mix, const float *w,
                           const __nv_bfloat16 *embedding, const float *ids,
                           int seed, int vocab, float *out) {
    constexpr int dim = 5120, rank = 256;
    int t = blockIdx.x, tid = threadIdx.x;
    float previous = t ? ids[t-1] : float(seed);
    if (!(previous >= 0.f && previous < vocab)) {
        if (!tid) out[t] = CUDART_NAN_F;
        return;
    }
    const float *h = states + size_t(t) * 4 * dim;
    const float *pre = mix + t * 24;
    const __nv_bfloat16 *markov = embedding + size_t(previous) * rank;
    float sum = __bfloat162float(markov[tid]) * w[dim + tid];
    for (int i = tid; i < dim; i += rank) {
        float value = 0.f;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            value = __fadd_rn(value, __fmul_rn(h[c*dim+i], pre[c]));
        value = __bfloat162float(__float2bfloat16_rn(value));
        sum = fmaf(value, w[i], sum);
    }
    for (int d=16;d;d/=2) sum += __shfl_down_sync(0xffffffff,sum,d);
    __shared__ float partial[rank/32];
    if (!(tid%32)) partial[tid/32] = sum;
    __syncthreads();
    if (!tid) {
        float answer = 0.f;
        for (float v : partial) answer += v;
        out[t] = answer;
    }
}
}
extern "C" int ds4_confidence(int device, const float *h, const float *mix, const float *w,
                              const void *embedding, const float *ids, int seed,
                              int vocab, float *out, int rows) {
    auto e = cudaSetDevice(device);
    if (e != cudaSuccess) return e;
    confidence<<<rows,256>>>(h,mix,w,static_cast<const __nv_bfloat16*>(embedding),ids,seed,vocab,out);
    return cudaGetLastError();
}
