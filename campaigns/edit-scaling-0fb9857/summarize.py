from pathlib import Path
import statistics as st,json,hashlib
p=Path(__file__).resolve().parent
med=st.median
def get(n):return [json.loads(l) for l in (p/(n+'.jsonl')).read_text().splitlines()]
def hashes(w):return [(r['content_sha256'],r['reasoning_sha256']) for r in w['rows']]
def summarize(n):
 allrows=get(n);rows=allrows[1:];assert len(rows)==6
 resource=json.loads((p/(n+'.resources.json')).read_text())
 assert len(resource)>20
 sums=[]
 for w in allrows:
  assert all(v['first_function_present'] and v['usage']['prompt_tokens']==25625 and v['usage']['completion_tokens']==1500 for v in w['rows'])
  a=w['delta']
  if w['spec']:
   rounds,proposed,accepted=map(sum,zip(*w['spec'].values()))
   assert 0<=9000-accepted-rounds<=6,(n,w['wave'],accepted,rounds)
  else:
   def cnt(k):
    base='spec_decode_num_'+k
    return a.get(base+'_total',a.get(base))
   rounds,proposed,accepted=map(cnt,['drafts','draft_tokens','accepted_tokens'])
   assert all(x is not None and x>=0 for x in (rounds,proposed,accepted)),a
   assert a['generation_tokens_total']==9000 and a['request_success_total']==6
  sums.append(dict(rounds=rounds,proposed=proposed,accepted=accepted,accepted_per_round=accepted/rounds,proposed_per_round=proposed/rounds,
   # This is service delivery time per sequence proposal round. Multiple sessions share model forwards.
   rounds_per_second=rounds/(9000/w['upstream_tok_s']),accepted_fraction=accepted/proposed))
 keys=('upstream_tok_s','shared_tok_s','whole_wave_tok_s','cpu_cores')
 return dict(image=rows[0]['image'],env=rows[0]['env'],medians={k:med(w[k] for w in rows) for k in keys}|{k:med(s[k] for s in sums[1:]) for k in sums[0]}, 
  minimum_free_mib=[min(r[1][i][0] for r in resource) for i in range(2)],minimum_available_host_mib=min(r[2] for r in resource),
  distinct_outputs=[len({hashes(w)[i] for w in rows}) for i in range(6)],measured_waves=rows,warmup=allrows[0],
  sampled_median_utilization_warmup_included=[med(r[1][i][1] for r in resource) for i in range(2)])
if __name__=='__main__':
 import sys
 for n in sys.argv[1:]:
  s=summarize(n)
  print(n, {k:round(v,3) for k,v in s['medians'].items()},s['distinct_outputs'])
