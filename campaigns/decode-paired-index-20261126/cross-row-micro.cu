#include "attention.cu"
#include <cstdio>
#include <vector>
#include <cstdlib>
#define CHECK(x) do { auto err=(x); if(err!=cudaSuccess){std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(err));std::exit(1);} } while(0)
__global__ void init_keys(unsigned char *k,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int j=0;j<128;++j)k[i*132+j]=((unsigned(i)*37+j*17)%121)|((i^j)&1?128:0);
 *reinterpret_cast<float*>(k+i*132+128)=0.01f*(1+i%11);
}
__global__ void init_queries(float *q,float *w,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 q[i]=(int(i*199u%509)-255)*.01f;if(i<n/128)w[i]=(int(i*223u%421)-210)*.07f;
}
int main(){
 for(int width:{65537,202112,404224}) for(int rows:{4,5,7,8}){
  unsigned char* keys;float *q,*w,*o;int *ids;
  CHECK(cudaMalloc(&keys,size_t(width)*132));CHECK(cudaMalloc(&q,rows*32*128*4));CHECK(cudaMalloc(&w,rows*32*4));CHECK(cudaMalloc(&ids,size_t(width)*rows*4));CHECK(cudaMalloc(&o,size_t(width)*rows*4));
  init_keys<<<(width+255)/256,256>>>(keys,width);init_queries<<<rows*16,256>>>(q,w,rows*32*128);
  index_ids<<<dim3((width+255)/256,rows),256>>>(nullptr,ids,width,width-rows,1);
  cudaEvent_t a,z;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&z));
  for(int mode:{0,8,16,0}){
   auto launch=[&](){
    dim3 orig((width+31)/32,rows),paired((width+15)/16,(rows+1)/2);
    if(mode==8)score_packed_keys<8,true><<<paired,128>>>(keys,q,w,ids,width,rows,o);
    else if(mode==16)score_packed_keys<16,true><<<paired,128>>>(keys,q,w,ids,width,rows,o);
    else if(int64_t(width)*rows>=524288)score_packed_keys<8><<<orig,128>>>(keys,q,w,ids,width,rows,o);
    else score_packed_keys<16><<<orig,128>>>(keys,q,w,ids,width,rows,o);
   };
   for(int i=0;i<8;++i)launch();
   std::vector<float> times;
   for(int j=0;j<7;++j){
    CHECK(cudaEventRecord(a));for(int i=0;i<20;++i)launch();CHECK(cudaEventRecord(z));CHECK(cudaEventSynchronize(z));
    float ms;CHECK(cudaEventElapsedTime(&ms,a,z));times.push_back(ms*1000.f/20);
   }
   std::sort(times.begin(),times.end());std::printf("%d %d %d %.2f\n",width,rows,mode,times[3]);std::fflush(stdout);
  }
  CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(z));CHECK(cudaFree(o));CHECK(cudaFree(ids));CHECK(cudaFree(w));CHECK(cudaFree(q));CHECK(cudaFree(keys));
 }
}
