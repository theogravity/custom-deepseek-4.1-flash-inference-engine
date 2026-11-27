// Compile this fixture with nvcc on the model workstation. It measures only
// rank selection and chronological ordering. The Rust index fixture also times
// key scoring, causal filtering and the candidate-block selection.
#include "../native/attention.cu"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using Rank = unsigned long long;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct DeviceArray {
    T *data = nullptr;
    explicit DeviceArray(size_t n) { check(cudaMalloc(&data,n*sizeof(T))); }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

Rank host_rank(float score, int id) {
    unsigned bits;
    std::memcpy(&bits,&score,sizeof(bits));
    bits ^= (bits>>31) ? 0xffffffffu : 0x80000000u;
    return (Rank(bits)<<32) | ~unsigned(id);
}

void run_case(int rows, int width, bool stress) {
    auto size = size_t(rows)*width;
    std::vector<Rank> ranks(size);
    std::vector<int> expected(rows*top);
    unsigned rng=1234567;
    for (int row=0;row<rows;++row) {
        auto *start = ranks.data()+size_t(row)*width;
        for (int id=0;id<width;++id) {
            rng = 1664525u*rng + 1013904223u;
            // Signed scores, numerous exact cutoff ties and invisible slots.
            float score = static_cast<int>(rng>>17)-16384;
            if (stress && row%5==4) score = id%2 ? 0.f : -0.f;
            const bool valid = stress && row%5==3 ? id<193 : id<width-(row%17);
            start[id] = valid ? host_rank(score,id) : 0;
        }
        if (stress && (row%5==1 || row%5==2)) {
            for (int i=0;i<sample_size;++i) {
                int id = (size_t(i)*width + width/2)/sample_size;
                start[id] = host_rank(row%5==1 ? INFINITY : -INFINITY,id);
            }
        }
        std::vector<Rank> tmp(start,start+width);
        std::nth_element(tmp.begin(),tmp.begin()+top,tmp.end(),std::greater<Rank>());
        auto *picked = expected.data()+size_t(row)*top;
        for (int i=0;i<top;++i) picked[i]=tmp[i] ? int(~unsigned(tmp[i])) : -1;
        std::sort(picked,picked+top,[](int a,int b){return unsigned(a)<unsigned(b);});
    }
    DeviceArray<Rank> input(size), output(size), scratch((size+1)/2), thresholds(rows);
    DeviceArray<int> offsets(rows+1), counters(rows), picks(size_t(rows)*top);
    check(cudaMemcpy(input.data,ranks.data(),size*sizeof(Rank),cudaMemcpyHostToDevice));
    index_bounds<<<(rows+block)/block,block>>>(offsets.data,width,rows);
    check(cudaGetLastError());
    size_t bytes=0;
    auto baseline = [&](void *temp, size_t& available) {
        return sort_index_rows(temp,available,input.data,output.data,width,rows,offsets.data,top);
    };
    check(baseline(nullptr,bytes));
    DeviceArray<uint8_t> sort_space(std::max<size_t>(1,bytes));
    for (bool sampled : {false,true}) {
        auto run = [&] {
            if (sampled) {
                check(sampled_selection(input.data,output.data,scratch.data,width,rows,thresholds.data,counters.data));
            } else if (use_tiled_selection(width,rows,top)) {
                check(tiled_selection(input.data,output.data,scratch.data,width,rows,top));
            } else {
                check(baseline(sort_space.data,bytes));
            }
            chronological_ids<top><<<rows,block>>>(output.data,width,picks.data);
            check(cudaGetLastError());
        };
        run();
        std::vector<int> got(size_t(rows)*top);
        check(cudaMemcpy(got.data(),picks.data,got.size()*sizeof(int),cudaMemcpyDeviceToHost));
        if (got!=expected) throw std::runtime_error("chronological winners differ from CPU");
        int rescued=0;
        if (sampled) {
            std::vector<int> counts(rows);
            check(cudaMemcpy(counts.data(),counters.data,rows*sizeof(int),cudaMemcpyDeviceToHost));
            for (int row=0;row<rows;++row) {
                if (counts[row]<top || counts[row]>sample_size) ++rescued;
                if (stress && ((row%5==1 && counts[row]>=top) ||
                               (row%5==2 && counts[row]<=sample_size))) {
                    throw std::runtime_error("adversarial fixture missed its rescue branch");
                }
            }
        }
        std::vector<double> trials;
        for (int i=0;i<5;++i) {
            auto start=std::chrono::steady_clock::now();
            for (int j=0;j<20;++j) run();
            check(cudaDeviceSynchronize());
            trials.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20.);
        }
        std::sort(trials.begin(),trials.end());
        std::printf("rows=%d width=%d stress=%d sampled=%d rescued=%d median_us=%.1f\n",
                    rows,width,stress,sampled,rescued,trials[2]);
    }
}
}  // namespace

int main() {
    for (auto shape : {std::pair{64,16391},{32,65536},{16,131072},{8,262144},{4,505000}}) {
        run_case(shape.first,shape.second,false);
    }
    run_case(16,131075,true);
}
