from pathlib import Path
root=Path(__file__).resolve().parent/'markov'
p=root/'native/sample.cu';s=p.read_text().replace('__global__ void markov_parts','template<int Tracks>\n__global__ void markov_parts',1)
i=s.index('    for(int t=0;t<8;++t) {',s.index('void markov_parts'));j=s.index('    block_best(w,out+',i)
s=s[:i]+'''    // Keep independent vocabulary dot products in flight. Every id retains
    // its original eight FMAs and lane reduction; a partial final tile is legal.
    for (int t = 0; t < 8; t += Tracks) {
        float sum[Tracks] = {};
        const int first = blockIdx.x * 64 + t * 8 + warp;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int k = 0; k < Tracks; ++k) {
                int id = first + 8 * k;
                if (id < vocab)
                    sum[k] = fmaf(x[j], __bfloat162float(head[size_t(id)*256 + lane + j*32]), sum[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < Tracks; ++k) {
            int id = first + 8 * k;
            if (id < vocab) {
                for (int d = 16; d; d /= 2)
                    sum[k] += __shfl_down_sync(0xffffffff, sum[k], d);
                if (!lane) {
                    float value = sum[k] + base[id];
                    w = best(w, {value, isfinite(value) ? id : -1});
                }
            }
        }
    }
'''+s[j:]
s=s.replace('float *scratch,float *ids,int seed,int rows,int vocab) {','float *scratch,float *ids,int seed,int rows,int vocab,int paired) {')
s=s.replace('        markov_parts<<<parts,256>>>','''        if (paired)
            markov_parts<2><<<parts,256>>>((const __nv_bfloat16*)emb,(const __nv_bfloat16*)head,
                base+size_t(i)*vocab,vocab,seed,i ? ids+i-1 : nullptr,(Winner*)scratch);
        else
            markov_parts<1><<<parts,256>>>''')
p.write_text(s)
p=root/'src/runtime.rs';s=p.read_text().replace('        vocab: i32,\n    ) -> i32;', '        vocab: i32,\n        paired: i32,\n    ) -> i32;',1).replace('    head_row_tiles: bool,','    head_row_tiles: bool,\n    markov_pairs: bool,').replace('            head_row_tiles: match','''            markov_pairs: match std::env::var("DS4_MARKOV_PAIRS").as_deref() {
                Ok("1") => true,
                Ok("0") | Err(std::env::VarError::NotPresent) => false,
                _ => anyhow::bail!("DS4_MARKOV_PAIRS must be 0 or 1"),
            },
            head_row_tiles: match''',1).replace('                VOCAB as i32,\n            )','                VOCAB as i32,\n                self.markov_pairs.into(),\n            )',1)
i=s.index('        assert_eq!(\n            unsafe {\n                ds4_markov(',s.index('fn markov_chain_and_greedy_reductions'));j=s.index('        let n = 4096;',i)
part=s[i:j].replace('                    cols as i32,','                    cols as i32,\n                    paired,')
k=part.index('        x[1] = f32::INFINITY;')
s=s[:i]+'        for paired in [0, 1] {\n'+part[:k]+'        }\n'+part[k:part.index('        assert_eq!(',k)]+'        for paired in [0, 1] {\n'+part[part.index('        assert_eq!(',k):]+'        }\n'+s[j:]
p.write_text(s)
