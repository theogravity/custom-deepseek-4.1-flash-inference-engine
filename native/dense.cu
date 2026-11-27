#include <cstring>
#include <cstdlib>
// Rust owns descriptors and model ordering. This unit implements dense
// projections with cuBLAS.
#include "head.cuh"
#include "mxfp8.cuh"
#include <algorithm>
#include <cstdio>
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
    auto allocated = cudaMalloc(&ptr, bytes);
    if (allocated == cudaErrorMemoryAllocation && cudaPeekAtLastError() == allocated)
      cudaGetLastError();
    CUDA(allocated);
    cap = bytes;
    return 0;
  }
};
#include "prefill_dense.cuh"
struct Context {
  prefill_dense::Executor lt;
  bool wide_mx = false;
  bool wide_blas = false;
  int device;
  cublasHandle_t blas = nullptr;
  Buffer input, short_input, output, partial;
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
  const char *wide = std::getenv("DS4_PREFILL_LT");
  c->wide_mx = wide && std::strcmp(wide, "1") == 0;
  const char *wide_blas = std::getenv("DS4_PREFILL_BLAS");
  c->wide_blas = wide_blas && std::strcmp(wide_blas, "1") == 0;
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
  int current;
  if (cudaGetDevice(&current) != cudaSuccess)
    return;
  if (current != device)
    cudaSetDevice(device);
  cudaFree(ptr);
  if (current != device)
    cudaSetDevice(current);
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
  if (mode == 4) {
    Buffer raw;
    size_t scale_n=size_t(rows/32)*(cols/32);
    if (result.resize(n+scale_n) || raw.resize(n)) return -1;
    CUDA(cudaMemcpy(raw.ptr,src,n,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(static_cast<unsigned char*>(result.ptr)+n,sc,scale_n,cudaMemcpyHostToDevice));
    mx_retile<<<dim3(rows/64,cols/16),128>>>(static_cast<const unsigned char*>(raw.ptr),
        static_cast<uint2*>(result.ptr),rows,cols);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    *out=result.ptr;result.ptr=nullptr;return 0;
  }
  if (mode == 3 || mode == 7) {
    Buffer bf16, packed;
    if (result.resize(n + n / 32) || bf16.resize(n * 2))
      return -1;
    CUDA(cudaMemcpy(bf16.ptr, src, n * 2, cudaMemcpyHostToDevice));
    if (mode==7 && packed.resize(n)) return -1;
    auto raw = mode==7 ? packed.ptr : result.ptr;
    pack_head<<<(n + 255) / 256, 256>>>((__nv_bfloat16 *)bf16.ptr,
                                        (__nv_fp8_e4m3 *)raw,
                                        (unsigned char *)result.ptr + n, n);
    CUDA(cudaGetLastError());
    if(mode==7){
      mx_retile<<<dim3(rows/64,cols/16),128>>>(static_cast<const unsigned char*>(raw),
          static_cast<uint2*>(result.ptr),rows,cols);
      CUDA(cudaGetLastError());
    }
    CUDA(cudaDeviceSynchronize());
    *out = result.ptr;
    result.ptr = nullptr;
    return 0;
  }
  if (result.resize(n * (mode == 0 || mode == 5 ? 4 : 2)))
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
extern "C" int ds_alloc_f32(int device, float **out, size_t n) {
  CUDA(cudaSetDevice(device));
  auto rc = cudaMalloc(out, n * sizeof(float));
  if (rc != cudaSuccess) {
    // An out-of-memory restore can release its request and keep serving peers.
    if (rc == cudaErrorMemoryAllocation && cudaPeekAtLastError() == rc)
      cudaGetLastError();
    snprintf(error_message, sizeof(error_message), "cudaMalloc(%zu bytes): %s",
             n * sizeof(float), cudaGetErrorString(rc));
    return -1;
  }
  return 0;
}
extern "C" int ds_copy_f32(int device, float *dst, const float *src, size_t n,
                           int to_host) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaMemcpy(dst, src, n * sizeof(float),
                  to_host == 2 ? cudaMemcpyDeviceToDevice :
                  to_host == 1 ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice));
  return 0;
}
__global__ void round_bf16(float *data, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] = __bfloat162float(__float2bfloat16_rn(data[i]));
}
static int mx_splits(int rows, int cols) {
  return std::max(1, std::min(cols / 32, (512 * 64 + rows - 1) / rows));
}
// Stable verification kernels and dense cuBLAS use the same weight storage.
static int projection_mode(Context *c, int mode, int batch, int prefill) {
  if (c->wide_blas && prefill && batch > 128 && (mode == 5 || mode == 6))
    return mode == 5 ? 0 : 1;
  return mode;
}
// Scratch buffers release their old allocation before growing. Reserve the
// net increase on this device while packed expert slots can still be evicted.
extern "C" size_t ds_dense_growth(Context *c, int rows, int cols, int batch,
                                  int groups, int mode, int full_rows, int prefill) {
  mode = projection_mode(c, mode, batch, prefill);
  const bool stable = mode == 5 || mode == 6;
  const bool packed = mode == 4 || mode == 7;
  size_t input = mode != 0 && mode != 3 && !stable
                     ? size_t(batch) * groups * cols * sizeof(__nv_bfloat16)
                     : 0;
  int splits = stable ? (cols + 1023) / 1024
                      : (packed ? mx_splits(full_rows, cols) : 0);
  size_t parts = size_t(batch) * rows * splits * sizeof(float);
  if (c->wide_mx && mode == 4 && prefill && batch > 128) {
    prefill_dense::Workspace work(rows / groups,cols,batch);
    return work.bytes > c->lt.scratch.cap ? work.bytes - c->lt.scratch.cap : 0;
  }
  return (input > c->short_input.cap ? input - c->short_input.cap : 0) +
         (parts > c->partial.cap ? parts - c->partial.cap : 0);
}

extern "C" int ds_project(Context *c, const void *w, const float *x, float *y,
                          int rows, int cols, int batch, int groups, int mode,
                          int round_output, int prefill, int mx_tiles, int full_rows) {
  CUDA(cudaSetDevice(c->device));
  mode = projection_mode(c, mode, batch, prefill);
  size_t nx = size_t(cols) * groups * batch, ny = size_t(rows) * batch;
  if (c->wide_mx && mode==4 && prefill && batch>128)
    return c->lt.multiply(w,x,y,rows,cols,batch,groups,round_output);
  if (mode == 3) {
    auto weights = static_cast<const __nv_fp8_e4m3 *>(w);
    auto scales = static_cast<const unsigned char *>(w) + size_t(rows) * cols;
    if (batch == 1)
      head_gemv<1><<<(rows + 7) / 8, 256>>>(weights, scales, x, y, rows, cols, batch);
    else
      head_gemv<4><<<dim3((rows + 7) / 8, (batch+3)/4), 256>>>(weights, scales, x, y, rows, cols, batch);
    CUDA(cudaGetLastError());
    return 0;
  }
  if(mode==5 || mode==6){
    int splits=(cols+1023)/1024;
    if(c->partial.resize(size_t(splits)*ny*sizeof(float)))return -1;
    dim3 grid(rows,(batch+3)/4,splits);
    if(mode==5)stable_dot<<<grid,128>>>(static_cast<const float*>(w),x,static_cast<float*>(c->partial.ptr),rows,cols,batch,groups,splits);
    else stable_dot<<<grid,128>>>(static_cast<const __nv_bfloat16*>(w),x,static_cast<float*>(c->partial.ptr),rows,cols,batch,groups,splits);
    CUDA(cudaGetLastError());
    mx_sum<<<(ny+255)/256,256>>>(static_cast<const float*>(c->partial.ptr),y,ny,splits,round_output);
    CUDA(cudaGetLastError());
    return 0;
  }
  const void *a = x;
  auto kind = CUDA_R_32F;
  if (mode) {
    if (c->short_input.resize(nx * 2))
      return -1;
    prepare<<<(nx + 255) / 256, 256>>>((float *)a,
                                       (__nv_bfloat16 *)c->short_input.ptr, nx,
                                       prefill && (mode == 2 || mode == 4) && batch > 128);
    CUDA(cudaGetLastError());
    a = c->short_input.ptr;
    kind = CUDA_R_16BF;
  }
  if (mode == 4 || mode == 7) {
    int splits=mx_splits(full_rows, cols);
    if(c->partial.resize(size_t(splits)*ny*sizeof(float)))return -1;
    auto weights=static_cast<const uint2*>(w);
    auto scales=static_cast<const unsigned char*>(w)+size_t(rows)*cols;
    auto values=static_cast<const __nv_bfloat16*>(a);
    auto parts=static_cast<float*>(c->partial.ptr);
    // Head verification and large backbone batches can each reuse a decoded
    // weight across independent eight-token fragments.
    const int tiles = (mode == 7 || batch >= 32) ? mx_tiles : 1;
    dim3 grid(rows / 64, (batch + tiles * 8 - 1) / (tiles * 8), splits);
    if (mode == 7 && tiles == 4)
        mx_project<true,4><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    else if (mode == 7 && tiles == 2)
        mx_project<true,2><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    else if (mode == 7)
        mx_project<true><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    else if (tiles == 4)
        mx_project<false,4><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    else if (tiles == 2)
        mx_project<false,2><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    else
        mx_project<false><<<grid,128>>>(weights,scales,values,parts,rows,cols,batch,groups,splits);
    CUDA(cudaGetLastError());
    mx_sum<<<(ny+255)/256,256>>>(static_cast<const float*>(c->partial.ptr),y,ny,splits,round_output || mode==7);
    CUDA(cudaGetLastError());
    return 0;
  }
  int block = rows / groups;
  float alpha = 1, beta = 0;
  for (int g = 0; g < groups; ++g) {
    const size_t item = mode ? 2 : 4;
    BLAS(cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, block, batch, cols,
                      &alpha, (char *)w + size_t(g) * block * cols * item, kind,
                      cols, (char *)a + size_t(g) * cols * item, kind,
                      groups * cols, &beta, y + g * block, CUDA_R_32F, rows,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  if (round_output) {
    round_bf16<<<(ny + 255) / 256, 256>>>(y, ny);
    CUDA(cudaGetLastError());
  }
  return 0;
}

// Host entry point used by the small standalone numeric test.
extern "C" int ds_gemm(Context *c, const void *w, const float *x, float *y,
                       int rows, int cols, int batch, int groups, int mode) {
  CUDA(cudaSetDevice(c->device));
  size_t nx = size_t(cols) * groups * batch, ny = size_t(rows) * batch;
  if (c->input.resize(nx * 4) || c->output.resize(ny * 4))
    return -1;
  CUDA(cudaMemcpy(c->input.ptr, x, nx * 4, cudaMemcpyHostToDevice));
  if (ds_project(c, w, (const float *)c->input.ptr, (float *)c->output.ptr,
                 rows, cols, batch, groups, mode, 0, mode != 5 && mode != 6, 1, rows))
    return -1;
  CUDA(cudaMemcpy(y, c->output.ptr, ny * 4, cudaMemcpyDeviceToHost));
  return 0;
}
