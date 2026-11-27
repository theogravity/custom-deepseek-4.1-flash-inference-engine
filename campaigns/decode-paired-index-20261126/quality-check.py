from pathlib import Path
import hashlib,json,statistics
p=Path(__file__).resolve().parent
d={}
for label in ['cpu-512-off-a','cpu-512-on','cpu-512-off-b','cpu-768-off-a','cpu-768-on','cpu-768-off-b','cpu-receipt','cpu-receipt-stress']:
 info=json.loads((p/(label+'.info')).read_text());assert not (p/(label+'.failed')).exists()
 raw=(p/(label+'.jsonl')).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==info['raw_sha256']
 ws=[json.loads(x) for x in raw.splitlines()]
 summary=ws.pop(); warm=ws.pop(0)
 assert len(ws)==(6 if info['workload']=='code' else 2)
 for w in ws:
  z=w['delta']; assert z['requests_started']==6 and z['rotations']==0 and z['six_target_passes']>0 and z['prefix_reused_tokens']==(2424704 if info['workload']=='code' else 2423808)
  for r in w['rows']:
   assert r['exact'] and r['output']==info['output'] and r['finish']=='length'
  j=z['cpu_expert_jobs'];rows=z['cpu_expert_rows'];assert j<=rows<=2*j
 if info['env']['DS4_CPU_EXPERTS']=='1' and label!='cpu-receipt':
  assert sum(info['per_wave_cpu_jobs'])>0
 else:
  assert sum(info['per_wave_cpu_jobs'])==0
  if info['workload']=='code':assert info['matches']==[6]*6
 info['warm_cpu_jobs']=warm['delta']['cpu_expert_jobs']
 info['receipt_or_code_checks']=sum(len(w['rows']) for w in ws)
 info['output_digests']=[[r['sha256'] for r in w['rows']] for w in ws]
 info['per_wave_whole_tok_s']=[w['wall_tok_s'] for w in ws]
 info['resident_tok_s']=statistics.median(w['resident_decode']['tok_s'] for w in ws)
 d[label]=info
assert len({x['image'] for x in d.values()})==1
(p/'cpu-result.json').write_text(json.dumps(d,indent=2)+'\n')
