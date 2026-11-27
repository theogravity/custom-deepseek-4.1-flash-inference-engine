#include "mxfp8.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void check(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
__global__ void init(unsigned char *w,unsigned char *s,__nv_bfloat16 *x,size_t n,size_t nx) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) w[i]=((i*41+i/99)%96+1)|((i%7<3)?128:0);
    if(i<n/32) s[i]=119+((i*19+i/7)%12);
    if(i<nx) x[i]=__float2bfloat16((int((i*37+i/65)%91)-45)/16.f);
}
__global__ void mismatch(const unsigned *a,const unsigned *b,unsigned *count,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && a[i]!=b[i])atomicAdd(count,1);
}
int main() {
    constexpr int rows=129280,cols=5120,max_batch=512;
    size_t n=size_t(rows)*cols,nx=size_t(max_batch)*cols;
    uint2 *w; unsigned char *s; __nv_bfloat16 *x; float *ref,*out; unsigned *errors;
    check(cudaMalloc(&w,n));check(cudaMalloc(&s,n/32));check(cudaMalloc(&x,nx*sizeof(*x)));
    check(cudaMalloc(&ref,size_t(rows)*max_batch*4));check(cudaMalloc(&out,size_t(rows)*max_batch*4));check(cudaMalloc(&errors,4));
    init<<<(n+255)/256,256>>>((unsigned char*)w,s,x,n,nx);
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    for(int batch: {8,9,15,16,17,31,32,47,48,49,64,96,97,128,129,511,512}) {
        for(int leg=0;leg<3;++leg) {
            int tiles=leg==1?(batch<=8?1:(batch<=16?2:4)):1;
            dim3 grid(rows/64,(batch+8*tiles-1)/(8*tiles));
            auto launch=[&](float *dest){
                if(tiles==4)mx_project<true,4><<<grid,128>>>(w,s,x,dest,rows,cols,batch,1,1);
                else if(tiles==2)mx_project<true,2><<<grid,128>>>(w,s,x,dest,rows,cols,batch,1,1);
                else mx_project<true><<<grid,128>>>(w,s,x,dest,rows,cols,batch,1,1);
            };
            // Reference and candidate start with different sentinels at every shape.
            check(cudaMemset(leg==0?ref:out,leg==0?0x7f:0xff,size_t(batch)*rows*4));
            std::vector<float> times;
            for(int rep=0;rep<9;++rep){
                check(cudaEventRecord(a));
                for(int i=0;i<3;++i)launch(leg==0?ref:out);
                check(cudaEventRecord(b));check(cudaEventSynchronize(b));
                float ms;check(cudaEventElapsedTime(&ms,a,b));if(rep)times.push_back(ms*1000/3);
            }
            if(leg!=0){
                check(cudaMemset(errors,0,4));mismatch<<<(batch*rows+255)/256,256>>>((unsigned*)ref,(unsigned*)out,errors,batch*rows);
                unsigned count;check(cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost));
                if(count){std::fprintf(stderr,"%d rows: %u differing partials\n",batch,count);return 2;}
            }
            std::sort(times.begin(),times.end());
            std::printf("{\"batch\":%d,\"tiles\":%d,\"leg\":%d,\"median_us\":%.3f,\"compared_values\":%d}\n",batch,tiles,leg,(times[3]+times[4])/2,leg?batch*rows:0);
        }
    }
}
