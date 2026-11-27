from pathlib import Path
import json,statistics,subprocess
p=Path(__file__).resolve().parent
r=[json.loads(l) for l in (p/'micro.jsonl').read_text().splitlines()]
assert len(r)==162
m={(j,w):statistics.median(v['us'] for v in r if v['jobs']==j and v['warps']==w) for j in [4,16,30,32,34,48] for w in [2,4,8]}
# Require a consistent isolated win on the measured common C6 grids before spending a serving leg.
jobs=[30,32,34]
w=min([2,4,8],key=lambda w:statistics.median(m[j,w]/m[j,4] for j in jobs))
if w==4 or max(m[j,w]/m[j,4] for j in jobs)>=.98:
 print('no common-grid candidate');(p/'no-tile').touch();raise SystemExit
lo=16 if m[16,w]<m[16,4] else 30
hi=48 if m[48,w]<m[48,4] else 34
f=p/'work/native/moe.cu';s=f.read_text()
old='    project<Gate, mcg, true, packed_decode><<<dim3(I/64,n_jobs,2*splits),128>>>(jobs, rotated, partial, rows, splits);'
assert s.count(old)==1
new=f'''    // The measured verification grids use smaller batches of distinct expert jobs.
    constexpr int verify_warps = {w};
    if (!mcg && packed_decode && splits == 8 && n_jobs >= {lo} && n_jobs <= {hi}) {{
        project<Gate, mcg, true, packed_decode, verify_warps>
            <<<dim3(I/(16*verify_warps),n_jobs,2*splits),32*verify_warps>>>(jobs, rotated, partial, rows, splits);
    }} else {{
        {old.strip()}
    }}'''
f.write_text(s.replace(old,new))
(p/'tile-selection.json').write_text(json.dumps(dict(warps=w,minimum_jobs=lo,maximum_jobs=hi)))
subprocess.run(['git','diff','--check'],cwd=p/'work',check=True)
subprocess.run(['git','add','native/moe.cu'],cwd=p/'work',check=True)
subprocess.run(['git','commit','-m',f'Screen {w}-warp gate/up blocks on batched target verification'],cwd=p/'work',check=True)
with (p/'tile-build.log').open('w') as f:subprocess.run(['cargo','build','--release','--locked','--offline'],cwd=p/'work',stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['docker','build','-t','ds4-flash-rs:edit-gap-tile'+str(w),'.'],cwd=p/'work',check=True)
