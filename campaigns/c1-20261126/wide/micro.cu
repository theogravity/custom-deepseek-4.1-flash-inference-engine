#include "attention.cu"
#include <cstdio>
#include <vector>
#include <cstdlib>
#define CHECK(x) do { auto err=(x); if(err!=cudaSuccess){std::fprintf(stderr,"error on line %d: %s\n",__LINE__,cudaGetErrorString(err));std::exit(1);} } while(0)
__global__ void fill_keys(unsigned long long *keys, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) keys[i]=(0xdeadbeef043dda57ull*unsigned(i)) ^ (0x463711928076b95bull*(unsigned(i)/17));
}
int main(){
 for(auto shape: std::vector<std::pair<int,int>>{{12816,1},{25625,2},{202112,8},{404224,5},{524288,1}}) {
  int width=shape.first, rows=shape.second;
  int keep=top;
  int n=width*rows;
  unsigned long long *in,*out,*inter; int *offsets; unsigned char *tmp;
  CHECK(cudaMalloc(&in,n*8));CHECK(cudaMalloc(&out,n*8));CHECK(cudaMalloc(&inter,n*4));CHECK(cudaMalloc(&offsets,(rows+1)*4));
  fill_keys<<<(n+255)/256,256>>>(in,n);
  size_t bytes=0; CHECK(sort_index_rows(nullptr,bytes,in,out,width,rows,offsets,keep));
  CHECK(cudaMalloc(&tmp,std::max(bytes,size_t(1))));
  cudaEvent_t a,z;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&z));
  for(int mode: {0,1,0}) {
   auto launch=[&](){
    if(mode) CHECK(tiled_selection(in,out,inter,width,rows,keep));
    else{
     if (!wide_index_sort(width, rows)) index_bounds<<<1,256>>>(offsets,width,rows);
     auto avail=bytes;CHECK(sort_index_rows(tmp,avail,in,out,width,rows,offsets,keep));
    }
   };
   for(int i=0;i<50;++i)launch();
   std::vector<float> times;
   for(int j=0;j<9;++j){
    CHECK(cudaEventRecord(a));for(int i=0;i<150;++i)launch();CHECK(cudaEventRecord(z));CHECK(cudaEventSynchronize(z));
    float ms;CHECK(cudaEventElapsedTime(&ms,a,z));times.push_back(ms*1000.f/150);
   }
   std::sort(times.begin(),times.end());
   std::printf("%d %d %d %d %.3f\n",width,rows,keep,mode,times[4]);
  }
  CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(z));CHECK(cudaFree(tmp));CHECK(cudaFree(offsets));CHECK(cudaFree(inter));CHECK(cudaFree(out));CHECK(cudaFree(in));
 }
}
