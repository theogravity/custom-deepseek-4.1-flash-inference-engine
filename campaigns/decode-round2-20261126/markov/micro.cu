#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "work/native/sample.cu"
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(1); }
}
__global__ void make_tables(__nv_bfloat16 *a, __nv_bfloat16 *b, size_t n) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ a[i]=__float2bfloat16((int(i%29)-14)*0.001f); b[i]=__float2bfloat16((int(i%47)-23)*0.001f); }
}
__global__ void make_scores(float *v, int vocab,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i < vocab*rows) v[i]=(i%vocab==(30021*(1+i/vocab))%vocab)?20.f:float(i%347)*0.003f;
}
int main() {
    const int vocab=129280, rows=7;
    size_t n=size_t(vocab)*256;
    __nv_bfloat16 *a,*b; float *scores,*parts,*ids;
    check(cudaMalloc(&a,n*sizeof(*a)));check(cudaMalloc(&b,n*sizeof(*b)));
    check(cudaMalloc(&scores,size_t(vocab)*rows*sizeof(float)));
    check(cudaMalloc(&parts,((vocab+63)/64)*sizeof(Winner)));check(cudaMalloc(&ids,rows*sizeof(float)));
    make_tables<<<(n+255)/256,256>>>(a,b,n);make_scores<<<(vocab*rows+255)/256,256>>>(scores,vocab,rows);
    cudaEvent_t t,z;check(cudaEventCreate(&t));check(cudaEventCreate(&z));
    for(int leg=0;leg<3;++leg){
        int mode=leg==1;
        std::vector<float> times;
        for(int rep=0;rep<13;++rep){
            check(cudaEventRecord(t));
            for(int j=0;j<10;++j) check(cudaError_t(ds4_markov(0,a,b,scores,parts,ids,47,rows,vocab,mode)));
            check(cudaEventRecord(z));check(cudaEventSynchronize(z));
            float ms;check(cudaEventElapsedTime(&ms,t,z));if(rep)times.push_back(ms*1000/10);
        }
        float out[rows];check(cudaMemcpy(out,ids,sizeof(out),cudaMemcpyDeviceToHost));
        for(int i=0;i<rows;++i)if(out[i]!=(30021*(i+1))%vocab)return 2;
        std::sort(times.begin(),times.end());
        std::printf("{\"mode\":%d,\"median_us_for_seven_rows\":%.3f,\"timed_samples\":12,\"matching_ids\":7}\n",mode,(times[5]+times[6])/2);
    }
}
