#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// One scale for each (vocabulary row, group of 32 hidden channels).
__global__ void pack_head(const __nv_bfloat16 *input, __nv_fp8_e4m3 *weights,
                          unsigned char *scales, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return; // Head width is a multiple of the warp size.
  const float v = __bfloat162float(input[i]);
  float peak = fabsf(v);
  for (int offset = 16; offset; offset /= 2)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
  const int exponent =
      max(-127, min(127, int(ceilf(log2f(fmaxf(peak, 1e-30f) / 448.f)))));
  weights[i] = __nv_fp8_e4m3(ldexpf(v, -exponent));
  if ((threadIdx.x & 31) == 0)
    scales[i / 32] = exponent + 127;
}

// Share each vocabulary weight load across a small tile of speculative rows.
template<int Tile>
__global__ void head_gemv(const __nv_fp8_e4m3 *weights,
                          const unsigned char *scales, const float *input,
                          float *output, int rows, int cols, int batch) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int first = blockIdx.y * Tile;
  const size_t start = size_t(row) * cols;
  float acc[Tile] = {};
  for (int i = lane; i < cols; i += 32) {
    const size_t j = start + i;
    const float w = ldexpf(float(weights[j]), int(scales[j / 32]) - 127);
    #pragma unroll
    for (int t=0; t<Tile; ++t)
      if (first+t < batch) acc[t] = fmaf(input[size_t(first+t)*cols+i], w, acc[t]);
  }
  #pragma unroll
  for (int t=0; t<Tile; ++t) {
    for (int d=16; d; d/=2) acc[t] += __shfl_down_sync(0xffffffff, acc[t], d);
    if (!lane && first+t < batch) output[size_t(first+t)*rows+row] = __bfloat162float(__float2bfloat16_rn(acc[t]));
  }
}
