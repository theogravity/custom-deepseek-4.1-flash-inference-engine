#include "attention.cu"
#include "score_groups.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#define C(f) do { cudaError_t e=(f); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#f,cudaGetErrorString(e));exit(1);} } while(0)
template<typename T>T* upload(std::vector<T> a){T *d=nullptr;C(cudaMalloc(&d,a.size()*sizeof(T)));C(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
int main(){
 C(cudaSetDevice(0));cudaEvent_t start,end;C(cudaEventCreate(&start));C(cudaEventCreate(&end));std::mt19937 rng(57338);
 for(auto shape:std::vector<std::pair<int,int>>{{127,3},{16384,1},{16384,7},{16384,24},{50500,7},{101000,7},{202000,4},{202001,8},{262081,8},{65536,16}}){
  int width=shape.first, rows=shape.second;
  int nkey=width+11;std::vector<uint8_t> kb(size_t(nkey)*132);
  for(int i=0;i<nkey;i++){float scale=0.125f*(1+(rng()%5));memcpy(kb.data()+i*132+128,&scale,4); for(int j=0;j<128;j++)kb[i*132+j]=(rng()%2?128:0) | (rng()%126);}
  std::vector<float> q(rows*ih*idim),wt(rows*ih),out(size_t(rows)*width);
  for(float &x:q)x=(int(rng()%1401)-700)/97.f;for(float &x:wt)x=(int(rng()%179)-89)/43.f;
  std::vector<int> map(out.size());for(auto &x:map)x=rng()%87?int(rng()%nkey):-1;
  auto *k=upload(kb);auto *query=upload(q),*w=upload(wt),*o=upload(out);auto *m=upload(map);
  for(bool sparse:{false,true}){
   auto run=[&](int mode){dim3 grid((width+31)/32,rows);int* ids=sparse?m:nullptr;
    if(mode==32)score_packed_keys<<<grid,128>>>(k,query,w,ids,width,o);else if(mode==16)score_grouped_keys<16><<<grid,128>>>(k,query,w,ids,width,o);else score_grouped_keys<8><<<grid,128>>>(k,query,w,ids,width,o);
   };
   std::vector<float> ref(out.size());run(32);C(cudaMemcpy(ref.data(),o,ref.size()*4,cudaMemcpyDeviceToHost));
   for(int mode:{8,16,32}){run(mode);C(cudaMemcpy(out.data(),o,out.size()*4,cudaMemcpyDeviceToHost));if(memcmp(out.data(),ref.data(),out.size()*4)){fprintf(stderr,"score mismatch %d %d %d\n",width,rows,mode);return 2;}}
   for(int rep=0;rep<6;rep++)for(int kx=0;kx<3;kx++){int mode=(int[]){32,16,8}[(kx+rep)%3];run(mode);C(cudaEventRecord(start));for(int n=0;n<5;n++)run(mode);C(cudaEventRecord(end));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,start,end));printf("%d %d %d %d %d %.6f\n",width,rows,int(sparse),mode,rep,ms/5.0); }
   fflush(stdout);
  }
  C(cudaFree(k));C(cudaFree(query));C(cudaFree(w));C(cudaFree(o));C(cudaFree(m));
 }
}
