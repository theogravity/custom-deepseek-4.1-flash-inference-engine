from summarize import *
import subprocess
expdir=Path.cwd()
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def concordance(a,b):
 na,nb=get(a)[1:],get(b)[1:]
 assert len(na)==len(nb)==6
 rows=[(x,y) for wa,wb in zip(na,nb,strict=True) for x,y in zip(wa['rows'],wb['rows'],strict=True)]
 assert all(x['prompt_sha256']==y['prompt_sha256'] for x,y in rows)
 return {'matched_requests':len(rows),**{k+'_matches':sum(x[k+'_sha256']==y[k+'_sha256'] for x,y in rows) for k in ['content','reasoning']}}
def compact(n):
 s=summarize(n)
 for w in [s['warmup']]+s['measured_waves']:
  output=[]
  for r in w['rows']:
   assert r['first_function_present']
   output.append({k:r[k] for k in ['start_function','reasoning_sha256','content_sha256','content_bytes','reasoning_bytes','first_s','end_s']})
  w['rows']=output
  for k in ['image','env']: w.pop(k)
 s['raw_sha256']=digest(p/(n+'.jsonl'))
 s['resource_samples_sha256']=digest(p/(n+'.resources.json'))
 return s
def screen():
 rows=[json.loads(l) for l in (p/'micro.jsonl').read_text().splitlines()]
 result={}
 for j in sorted({r['jobs'] for r in rows}):
  group=[r for r in rows if r['jobs']==j]
  assert len(group)==27
  result[j]={w:med(r['us'] for r in group if r['warps']==w) for w in [2,4,8]}
 return dict(target='paired packed MUL1 gate/up, 8 split-K partials',timed_kernel_calls_per_sample=40,samples_per_width_shape=9,
  synthetic_gate_up_bit_check=True,isolated_boundary_jobs=[1,8,9,15,16],micro_source_sha256=digest(p/'micro.cu'),raw_sha256=digest(p/'micro.jsonl'),median_kernel_us=result)
def cpu_logs():
 raw=json.loads((p/'cpu-worker-counters.json').read_text())
 keys=('rank','jobs_since_start','jobs_since_last_log','mean_us','rows','rescues','errors','degraded')
 return {cid:[{k:(float(v) if k=='mean_us' else int(v)) for k,v in zip(keys,row,strict=True)} for row in rows] for cid,rows in raw.items()}
def make(labels):
 d=dict(control_source='3d8c541ebfd9abd2e3b9fb486fa073a811c7d096',release='0fb9857',watts_per_gpu=375,cpu_socket_watts=225,
  cpu_count_per_pool=10,vllm_cpu_expert_log_samples=cpu_logs(),service_smoke=json.loads((p/'service-smoke.json').read_text()),published_script_git='362838bb8e5ae683760c96a13c4389cb7770671f',bench_harness_sha256=digest(expdir/'tools/edit_scaling.py'),
  prompt_sha256=[r['prompt_sha256'] for r in get('rust-a')[0]['rows']],prompt_tokens_each=25625,output_tokens_each=1500,clients=6,
  legs={n:compact(n) for n in labels},host_expert_transfers=json.loads((p/'moe-telemetry.json').read_text()),cuda_tile_screen=screen(),native_short_cuda_profile=json.loads((p/'edit-profile.json').read_text()))
 pairs=[('rust-a','rust-b'),('rust-tune-a','rust-tune-b'),('rust-tune-a','k5'),('rust-a','vllm-on-a'),('vllm-on-a','vllm-off'),('vllm-on-a','vllm-on-b'),('vllm-off','vllm-on-b')]+[('rust-a',n) for n in labels if n.startswith(('k5','tile','rust-post'))]
 d['paired_wave_output_comparison']={a+' vs '+b:concordance(a,b) for a,b in pairs}
 return d
if __name__=='__main__':
 import sys
 obj=make(sys.argv[1:])
 (expdir/'results/edit-scaling-375w.json').write_text(json.dumps(obj,indent=2)+'\n')
