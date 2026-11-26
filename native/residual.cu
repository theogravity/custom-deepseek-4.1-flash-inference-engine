// Four-stream manifold hyper-connections. Each token keeps its previous
// pre-weights until its branch input has been collapsed.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr int width = 5120;
constexpr int threads = 256;
constexpr int channels = 4;
constexpr int controls = 24;
constexpr float eps = 1e-6f;

__device__ float bf(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ float inverse_rms(float energy, int count) {
  return 1.f / sqrtf(energy / count + 1e-20f);
}

// Input magnitudes and their squares are FP32. Rust accumulates in channel
// and feature order; a warp-tree sum can change BF16 rounding downstream.
__device__ float add_squares(float sum, const float *squares) {
#pragma unroll 16
  for (int d = 0; d < width; ++d)
    sum = __fadd_rn(sum, squares[d]);
  return sum;
}

__device__ float logit(const float *projection, float inverse,
                       const float *scale, const float *base, int i, int group) {
  return __fadd_rn(__fmul_rn(__fmul_rn(projection[i], inverse), scale[group]),
                   base[i]);
}

// Lanes 0..15 hold one 4x4 matrix, row major. Use the same four-term
// addition order as the scalar implementation for each row and column.
template <bool Column>
__device__ float sum4(float x, int i) {
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < channels; ++j) {
    int source = Column ? j * channels + i % channels : (i / channels) * channels + j;
    sum += __shfl_sync(0xffff, x, source);
  }
  return sum;
}

__device__ void update_mix(const float *projection, float inverse,
                           const float *scale, const float *base, float *mix) {
  const int tid = threadIdx.x;
  if (tid < 8) {
    const float v = logit(projection, inverse, scale, base, tid, tid / 4);
    const float denom = 1.f + float(exp(-double(v)));
    mix[tid] = tid < 4 ? 1.f / denom + eps : 2.f / denom;
  }
  if (tid >= 32 && tid < 48) {
    const int i = tid - 32;
    const float v = logit(projection, inverse, scale, base, 8 + i, 2);
    float mx = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < channels; ++j)
      mx = fmaxf(mx, __shfl_sync(0xffff, v, i / channels * channels + j));
    float a = float(exp(double(v - mx)));
    a = a / sum4<false>(a, i) + eps;
    for (int round = 0; round < 20; ++round) {
      if (round)
        a /= sum4<false>(a, i) + eps;
      a /= sum4<true>(a, i) + eps;
    }
    mix[8 + i] = a;
  }
}

template <bool UpdateMix>
__global__ void prepare(const float *h, float *mix, const float *projection,
                        const float *scale, const float *base,
                        const float *norm, float *out) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  h += t * channels * width;
  mix += t * controls;
  out += t * width;
  __shared__ float squares[width];
  __shared__ float h_inv, collapsed_inv;
  float values[width / threads] = {};
  float energy = 0.f;
#pragma unroll
  for (int c = 0; c < channels; ++c) {
    const float p = mix[c];
#pragma unroll
    for (int j = 0; j < width / threads; ++j) {
      int d = tid + j * threads;
      float v = h[c * width + d];
      values[j] = __fadd_rn(values[j], __fmul_rn(p, v));
      if constexpr (UpdateMix)
        squares[d] = __fmul_rn(v, v);
    }
    if constexpr (UpdateMix) {
      __syncthreads();
      if (tid == 0)
        energy = add_squares(energy, squares);
      __syncthreads();
    }
  }
#pragma unroll
  for (int j = 0; j < width / threads; ++j) {
    float v = bf(values[j]);
    values[j] = v;
    squares[tid + j * threads] = __fmul_rn(v, v);
  }
  __syncthreads();
  if (tid == 0) {
    if constexpr (UpdateMix)
      h_inv = inverse_rms(energy, channels * width);
    collapsed_inv = inverse_rms(add_squares(0.f, squares), width);
  }
  __syncthreads();
  if constexpr (UpdateMix)
    update_mix(projection + t * controls, h_inv, scale, base, mix);
#pragma unroll
  for (int j = 0; j < width / threads; ++j) {
    int d = tid + j * threads;
    out[d] = bf(__fmul_rn(__fmul_rn(values[j], collapsed_inv), norm[d]));
  }
}

__global__ void expand(float *h, const float *x, const float *mix) {
  const int t = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= width)
    return;
  h += t * channels * width + d;
  const float y = x[t * width + d];
  mix += t * controls;
  float old[channels];
#pragma unroll
  for (int j = 0; j < channels; ++j)
    old[j] = h[j * width];
#pragma unroll
  for (int c = 0; c < channels; ++c) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < channels; ++j)
      sum = __fadd_rn(sum, __fmul_rn(old[j], mix[8 + channels * j + c]));
    h[c * width] = bf(__fadd_rn(__fmul_rn(y, mix[4 + c]), sum));
  }
}
} // namespace

extern "C" int ds_hc_prepare(int device, const float *h, float *mix, const float *projection,
                             const float *scale, const float *base,
                             const float *norm, float *out, int batch) {
  if (auto status = cudaSetDevice(device); status != cudaSuccess)
    return status;
  prepare<true><<<batch, threads>>>(h, mix, projection, scale, base, norm, out);
  return cudaGetLastError();
}

extern "C" int ds_hc_expand(int device, float *h, const float *x, const float *mix, int batch) {
  if (auto status = cudaSetDevice(device); status != cudaSuccess)
    return status;
  expand<<<dim3((width + threads - 1) / threads, batch), threads>>>(h, x, mix);
  return cudaGetLastError();
}

extern "C" int ds_hc_final(int device, const float *h, float *mix, const float *norm,
                           float *out, int last) {
  if (auto status = cudaSetDevice(device); status != cudaSuccess)
    return status;
  prepare<false><<<1, threads>>>(h + last * channels * width, mix + last * controls,
                                 nullptr, nullptr, nullptr, norm, out);
  return cudaGetLastError();
}
