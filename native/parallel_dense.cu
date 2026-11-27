// Coordinate two output-row shards without synchronizing the inference thread.
// Dense kernels run on each device's legacy default stream. A nonblocking copy
// stream on the destination of each transfer keeps the other shard runnable.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <new>

struct Context;
extern "C" int ds_project(Context *ctx, const void *w, const float *x, float *y,
                          int rows, int cols, int batch, int groups, int mode,
                          int round_output, int prefill, int mx_tiles, int full_rows);
namespace {
struct Fence {
  int device = -1;
  cudaEvent_t done = nullptr;
  cudaStream_t transfer = nullptr;
  ~Fence() {
    if (device < 0) return;
    cudaSetDevice(device);
    if (transfer) cudaStreamDestroy(transfer);
    if (done) cudaEventDestroy(done);
  }
  cudaError_t init(int d) {
    device = d;
    auto e = cudaSetDevice(d);
    if (e == cudaSuccess)
      e = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
    if (e == cudaSuccess)
      e = cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking);
    return e;
  }
};
struct Link {
  Fence owner, peer;
  cudaEvent_t input_ready = nullptr, output_ready = nullptr;
  ~Link() {
    if (input_ready) { cudaSetDevice(peer.device); cudaEventDestroy(input_ready); }
    if (output_ready) { cudaSetDevice(owner.device); cudaEventDestroy(output_ready); }
  }
};
cudaError_t enable(int dev, int peer) {
  int available = 0;
  auto e = cudaDeviceCanAccessPeer(&available, dev, peer);
  if (e != cudaSuccess) return e;
  if (!available) return cudaErrorPeerAccessUnsupported;
  if ((e = cudaSetDevice(dev)) != cudaSuccess) return e;
  e = cudaDeviceEnablePeerAccess(peer, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return cudaSuccess;
  }
  return e;
}

__global__ void pack_bf16(const float *src, __nv_bfloat16 *dst, size_t count) {
  size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i<count) dst[i]=__float2bfloat16_rn(src[i]);
}
__global__ void unpack_bf16(const __nv_bfloat16 *src, float *dst, size_t count) {
  size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i<count) dst[i]=__bfloat162float(src[i]);
}
__global__ void interleave(const float *parts, float *out, int half, int elements, bool packed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  int token = i / half;
  size_t at = size_t(token) * 2 * half + i % half;
  out[at] = parts[i];
  out[at + half] = packed ? __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(parts+elements)[i]) : parts[elements + i];
}
} // namespace

extern "C" int ds_tp_new(int owner, int peer, void **output) {
  auto *l = new (std::nothrow) Link;
  if (!l) return cudaErrorMemoryAllocation;
  auto e = enable(owner, peer);
  if (e == cudaSuccess) e = enable(peer, owner);
  if (e == cudaSuccess) e = l->peer.init(peer);
  if (e == cudaSuccess) e = cudaEventCreateWithFlags(&l->input_ready, cudaEventDisableTiming);
  if (e == cudaSuccess) e = l->owner.init(owner);
  if (e == cudaSuccess) e = cudaEventCreateWithFlags(&l->output_ready, cudaEventDisableTiming);
  if (e != cudaSuccess) { delete l; cudaSetDevice(owner); return e; }
  *output = l;
  return cudaSetDevice(owner);
}
extern "C" void ds_tp_destroy(Link *l) {
  int current;
  if (cudaGetDevice(&current) != cudaSuccess) return;
  delete l;
  cudaSetDevice(current);
}

#define TRY(call) do { cudaError_t e = (call); if (e != cudaSuccess) return int(e); } while (0)
extern "C" int ds_tp_project(Link *l, Context *home_ctx, Context *peer_ctx,
                            const void *w_home, const void *w_peer,
                            const float *x, float *out, float *parts_home,
                            float *x_peer, float *out_peer,
                            __nv_bfloat16 *home_wire, __nv_bfloat16 *peer_wire, int rows, int cols, int batch, int mode,
                            int round_output, int prefill, int mx_tiles) {
  const int home = l->owner.device, peer = l->peer.device, half = rows / 2;
  const size_t x_bytes = size_t(batch) * cols * sizeof(float);
  const int n = batch * half;
  auto enqueue = [&]() -> int {
    TRY(cudaSetDevice(home));
    const bool packed = home_wire != nullptr;
    const bool packed_output = packed && round_output;
    if (packed) {
      pack_bf16<<<(x_bytes/4+255)/256,256>>>(x,home_wire,x_bytes/4);
      TRY(cudaGetLastError());
    }
    TRY(cudaEventRecord(l->owner.done, nullptr));
    TRY(cudaSetDevice(peer));
    TRY(cudaStreamWaitEvent(l->peer.transfer, l->owner.done));
    if (packed)
      TRY(cudaMemcpyPeerAsync(peer_wire,peer,home_wire,home,x_bytes/2,l->peer.transfer));
    else
      TRY(cudaMemcpyPeerAsync(x_peer,peer,x,home,x_bytes,l->peer.transfer));
    TRY(cudaEventRecord(l->input_ready, l->peer.transfer));
    TRY(cudaStreamWaitEvent(nullptr, l->input_ready));
    if (packed) {
      unpack_bf16<<<(x_bytes/4+255)/256,256>>>(peer_wire,x_peer,x_bytes/4);
      TRY(cudaGetLastError());
    }
    if (ds_project(peer_ctx, w_peer, x_peer, out_peer, half, cols, batch, 1,
                   mode, round_output, prefill, mx_tiles, rows)) return -1;
    if (packed_output) {
      pack_bf16<<<(n+255)/256,256>>>(out_peer,peer_wire,n);
      TRY(cudaGetLastError());
    }
    TRY(cudaEventRecord(l->peer.done, nullptr));

    TRY(cudaSetDevice(home));
    TRY(cudaStreamWaitEvent(l->owner.transfer, l->peer.done));
    if (packed_output)
      TRY(cudaMemcpyPeerAsync(parts_home+n,home,peer_wire,peer,size_t(n)*2,l->owner.transfer));
    else
      TRY(cudaMemcpyPeerAsync(parts_home+n,home,out_peer,peer,size_t(n)*4,l->owner.transfer));
    TRY(cudaEventRecord(l->output_ready, l->owner.transfer));
    if (ds_project(home_ctx, w_home, x, parts_home, half, cols, batch, 1,
                   mode, round_output, prefill, mx_tiles, rows)) return -1;
    TRY(cudaStreamWaitEvent(nullptr, l->output_ready));
    interleave<<<(n+255)/256,256>>>(parts_home,out,half,n,packed_output);
    return cudaGetLastError();
  };
  int result = enqueue();
  if (result != 0) {
    // No queued access may survive an error when Rust returns activations to
    // the shared buffer pool. Normal dispatch only installs stream dependencies.
    cudaSetDevice(peer); cudaDeviceSynchronize();
    cudaSetDevice(home); cudaDeviceSynchronize();
  }
  return result;
}
