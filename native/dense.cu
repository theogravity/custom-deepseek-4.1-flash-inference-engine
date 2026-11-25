// Rust owns descriptors and model ordering. This unit implements dense
// projections with cuBLAS.
#include "head.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <new>

static thread_local char error_message[256];
extern "C" const char *ds_dense_error() { return error_message; }
#define CUDA(call)                                                             \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != cudaSuccess) {                                                    \
      snprintf(error_message, sizeof(error_message), "%s: %s", #call,          \
               cudaGetErrorString(e));                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)
#define BLAS(call)                                                             \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != CUBLAS_STATUS_SUCCESS) {                                          \
      snprintf(error_message, sizeof(error_message), "%s: %s", #call,          \
               cublasGetStatusString(e));                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)
struct Buffer {
  void *ptr = nullptr;
  size_t cap = 0;
  ~Buffer() {
    if (ptr)
      cudaFree(ptr);
  }
  int resize(size_t bytes) {
    if (cap >= bytes)
      return 0;
    if (ptr) {
      CUDA(cudaFree(ptr));
      ptr = nullptr;
      cap = 0;
    }
    CUDA(cudaMalloc(&ptr, bytes));
    cap = bytes;
    return 0;
  }
};
struct Context {
  int device;
  cublasHandle_t blas = nullptr;
  Buffer input, short_input, output;
  ~Context() {
    if (blas)
      cublasDestroy(blas);
  }
};
extern "C" int ds_dense_create(int device, void **out) {
  CUDA(cudaSetDevice(device));
  cudaDeviceProp gpu{};
  CUDA(cudaGetDeviceProperties(&gpu, device));
  if (!std::strstr(gpu.name, "RTX PRO 6000") || gpu.major != 12 ||
      gpu.minor != 0) {
    snprintf(error_message, sizeof(error_message),
             "CUDA device %d (%s) is not an RTX PRO 6000 Blackwell", device,
             gpu.name);
    return -1;
  }
  auto c = new (std::nothrow) Context{};
  if (!c)
    return -1;
  c->device = device;
  auto e = cublasCreate(&c->blas);
  if (e != CUBLAS_STATUS_SUCCESS) {
    delete c;
    snprintf(error_message, sizeof(error_message), "cublasCreate status %d",
             (int)e);
    return -1;
  }
  *out = c;
  return 0;
}
extern "C" void ds_dense_destroy(Context *c) {
  cudaSetDevice(c->device);
  delete c;
}
extern "C" void ds_free(int device, void *ptr) {
  cudaSetDevice(device);
  cudaFree(ptr);
}
extern "C" int ds_set_device(int device) {
  CUDA(cudaSetDevice(device));
  return 0;
}

__global__ void unpack(const __nv_fp8_e4m3 *w, const unsigned char *s,
                       __nv_bfloat16 *out, size_t n, int k) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t scale = (i / k / 32) * ((k + 31) / 32) + (i % k) / 32;
    out[i] = __float2bfloat16_rn(ldexpf(float(w[i]), int(s[scale]) - 127));
  }
}
// mode 0: raw FP32, 1: raw BF16, 2: checkpoint MXFP8 expanded once into BF16.
extern "C" int ds_upload(int device, void **out, const void *src,
                         const void *sc, int rows, int cols, int mode) {
  CUDA(cudaSetDevice(device));
  size_t n = size_t(rows) * cols;
  Buffer result;
  if (mode == 3) {
    Buffer bf16;
    if (result.resize(n + n / 32) || bf16.resize(n * 2))
      return -1;
    CUDA(cudaMemcpy(bf16.ptr, src, n * 2, cudaMemcpyHostToDevice));
    pack_head<<<(n + 255) / 256, 256>>>((__nv_bfloat16 *)bf16.ptr,
                                        (__nv_fp8_e4m3 *)result.ptr,
                                        (unsigned char *)result.ptr + n, n);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    *out = result.ptr;
    result.ptr = nullptr;
    return 0;
  }
  if (result.resize(n * (mode == 0 ? 4 : 2)))
    return -1;
  if (mode != 2) {
    CUDA(cudaMemcpy(result.ptr, src, result.cap, cudaMemcpyHostToDevice));
  } else {
    Buffer data, scales;
    size_t scale_n = size_t((rows + 31) / 32) * ((cols + 31) / 32);
    if (data.resize(n) || scales.resize(scale_n))
      return -1;
    CUDA(cudaMemcpy(data.ptr, src, n, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(scales.ptr, sc, scale_n, cudaMemcpyHostToDevice));
    unpack<<<(n + 255) / 256, 256>>>((const __nv_fp8_e4m3 *)data.ptr,
                                     (const unsigned char *)scales.ptr,
                                     (__nv_bfloat16 *)result.ptr, n, cols);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
  }
  *out = result.ptr;
  result.ptr = nullptr;
  return 0;
}
__global__ void prepare(const float *x, __nv_bfloat16 *y, size_t n,
                        bool quant) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = i < n ? __bfloat162float(__float2bfloat16_rn(x[i])) : 0;
  if (quant) {
    float peak = fabsf(a);
    for (int d = 16; d; d >>= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, d));
    float scale = exp2f(ceilf(log2f(fmaxf(peak, 0.0001f) / 448.f)));
    a = float(__nv_fp8_e4m3(a / scale)) * scale;
  }
  if (i < n)
    y[i] = __float2bfloat16_rn(a);
}
extern "C" int ds_gemm(Context *c, const void *w, const float *x, float *y,
                       int rows, int cols, int batch, int groups, int mode) {
  CUDA(cudaSetDevice(c->device));
  size_t nx = size_t(cols) * groups * batch, ny = size_t(rows) * batch;
  if (c->input.resize(nx * 4) || c->output.resize(ny * 4))
    return -1;
  CUDA(cudaMemcpy(c->input.ptr, x, nx * 4, cudaMemcpyHostToDevice));
  if (mode == 3) {
    head_gemv<<<(rows + 7) / 8, 256>>>(
        (const __nv_fp8_e4m3 *)w,
        (const unsigned char *)w + size_t(rows) * cols,
        (const float *)c->input.ptr, (float *)c->output.ptr, rows, cols);
    CUDA(cudaGetLastError());
    CUDA(cudaMemcpy(y, c->output.ptr, ny * 4, cudaMemcpyDeviceToHost));
    return 0;
  }
  void *a = c->input.ptr;
  auto kind = CUDA_R_32F;
  if (mode) {
    if (c->short_input.resize(nx * 2))
      return -1;
    prepare<<<(nx + 255) / 256, 256>>>((float *)a,
                                       (__nv_bfloat16 *)c->short_input.ptr, nx,
                                       mode == 2 && batch > 128);
    CUDA(cudaGetLastError());
    a = c->short_input.ptr;
    kind = CUDA_R_16BF;
  }
  int block = rows / groups;
  float alpha = 1, beta = 0;
  for (int g = 0; g < groups; ++g) {
    const size_t item = mode ? 2 : 4;
    BLAS(cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, block, batch, cols,
                      &alpha, (char *)w + size_t(g) * block * cols * item, kind,
                      cols, (char *)a + size_t(g) * cols * item, kind,
                      groups * cols, &beta, (float *)c->output.ptr + g * block,
                      CUDA_R_32F, rows, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
  }
  CUDA(cudaMemcpy(y, c->output.ptr, ny * 4, cudaMemcpyDeviceToHost));
  return 0;
}
