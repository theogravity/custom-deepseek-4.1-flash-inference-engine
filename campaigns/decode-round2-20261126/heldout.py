from pathlib import Path
import sys,fcntl,json,statistics,importlib.util,hashlib
s=importlib.util.spec_from_file_location('h','/tmp/ds4-cycle2-1126/run.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
p=h.p/sys.argv[1];b=h.b;q=h.q;q.p=p;result={};gold={}
reference=json.loads((h.root/'results/c1-lookup.json').read_text())['heldout-summary']['greedy_fixed_output']
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',h.flags,b.native)
 (p/'heldout-summary.json').unlink(missing_ok=True);images={}
 try:
  for label,img,env,_,_ in json.loads((p/'config.json').read_text()):
   images[label]=b.start('c1-held-'+p.name+'-'+label,'native',img,env);before=b.idle('native')
   tag='held-'+label
   raw=q.shell(tag,['tools/code_speed.py','--tokens','512','--rounds','7'])
   assert raw[-1]['samples']==18 and len(raw)==22
   groups={}
   for w in raw[:-1]:
    k=w['workload'];v=(w['prompt_tokens'],w['visible_tokens'],w['sha256'],w['finish'])
    assert gold.setdefault(k,v)==v
    assert list(v)==reference[k],(k,v,reference[k])
    if not w['warmup']:groups.setdefault(k,[]).append(w['decode_tok_s'])
   assert all(len(v)==6 for v in groups.values()) and groups.keys()=={'game','parser','cache'}
   z=b.idle('native');assert z['requests_started']-before['requests_started']==21 and z['rotations']==before['rotations']
   result[label]={k:dict(rates=v,median=statistics.median(v)) for k,v in groups.items()}
   result[label]['raw_sha256']=hashlib.sha256((p/(tag+'.jsonl')).read_bytes()).hexdigest()
   b.verify('native',h.flags|env,img)
   b.stamp(tag+' '+str({k:round(statistics.median(v),2) for k,v in groups.items()}))
  (p/'heldout-summary.json').write_text(json.dumps({'greedy_fixed_output':gold,'by_leg':result,'images':images},indent=2)+'\n')
 finally:b.restore();b.verify('native',h.flags,b.native)
