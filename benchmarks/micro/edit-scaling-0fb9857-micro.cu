// Compare one routed gate/up projection at unchanged math and split boundaries.
#include "moe.cu"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#define OK(expr) do { auto err=(expr); if(err!=cudaSuccess){fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(err));exit(1);} } while(0)
template<class T> T *allocate(size_t n) { T *out; OK(cudaMalloc(&out,n*sizeof(T))); return out; }
__global__ void pattern(uint32_t *dst,size_t words){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<words){uint32_t z=uint32_t(i)^0x24988aa3;z^=z>>16;z*=0x045d9f3b;z^=z>>16;dst[i]=z;}
}
template<int warps>
void launch(const Job* jobs,const half* x,float* out,int nj,int rows){
    project<Gate,false,true,true,warps><<<dim3(I/(16*warps),nj,16),warps*32>>>(jobs,x,out,rows,8);
}
int main() {
    int dev=0; cudaDeviceProp prop{};OK(cudaGetDeviceProperties(&prop,dev));
    if(!strstr(prop.name,"RTX PRO 6000")||prop.major!=12)return 2;
    OK(cudaSetDevice(dev));
    char* weights=allocate<char>(48*EXPERT);
    pattern<<<(48*EXPERT/4+255)/256,256>>>((uint32_t*)weights,48*EXPERT/4);OK(cudaGetLastError());
    for(int nj:{5,4,16,30,32,34,48}) {
        const int boundary[]={1,8,9,15,16};
        std::vector<Job> jobs;int rows=0;
        for(int n=0;n<nj;n++){int count=nj==5?boundary[n]:1+(n*7)%9;jobs.push_back({weights+n*EXPERT,rows,count});rows+=count;}
        Job* j=allocate<Job>(nj);OK(cudaMemcpy(j,jobs.data(),nj*sizeof(Job),cudaMemcpyHostToDevice));
        std::vector<half> input(2*rows*H);for(size_t i=0;i<input.size();i++)input[i]=__float2half(float(int(i*47%251)-125)/97);
        half* x=allocate<half>(input.size());OK(cudaMemcpy(x,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        size_t elems=size_t(16)*rows*I;float* out=allocate<float>(elems);
        std::vector<float> gold(elems),got(elems);
        using Fn=void(*)(const Job*,const half*,float*,int,int);
        Fn fns[]={launch<2>,launch<4>,launch<8>};int ws[]={2,4,8};
        launch<4>(j,x,out,nj,rows);OK(cudaMemcpy(gold.data(),out,elems*4,cudaMemcpyDeviceToHost));
        if(std::none_of(gold.begin(),gold.end(),[](float v){return v!=0 && isfinite(v);}))return 4;
        cudaEvent_t a,b;OK(cudaEventCreate(&a));OK(cudaEventCreate(&b));
        for(int t=0;t<3;t++){
            OK(cudaMemset(out,0xa5,elems*4));fns[t](j,x,out,nj,rows);
            OK(cudaMemcpy(got.data(),out,elems*4,cudaMemcpyDeviceToHost));
            if(memcmp(got.data(),gold.data(),elems*4)){fprintf(stderr,"bit mismatch jobs=%d warp=%d\n",nj,ws[t]);return 3;}
        }
        for(int rep=0;nj!=5 && rep<9;rep++)for(int q=0;q<3;q++){
            int t=(rep+q)%3;OK(cudaEventRecord(a));
            for(int i=0;i<40;i++)fns[t](j,x,out,nj,rows);
            OK(cudaEventRecord(b));OK(cudaEventSynchronize(b));float ms;OK(cudaEventElapsedTime(&ms,a,b));
            printf("{\"jobs\":%d,\"rows\":%d,\"warps\":%d,\"rep\":%d,\"us\":%.5f}\n",nj,rows,ws[t],rep,ms*25);
        }
        OK(cudaEventDestroy(a));OK(cudaEventDestroy(b));OK(cudaFree(out));OK(cudaFree(x));OK(cudaFree(j));
    }
    OK(cudaFree(weights)); return 0;
}
