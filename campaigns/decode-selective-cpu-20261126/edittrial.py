from pathlib import Path
import fcntl,json,os,sys,subprocess as sp,statistics
os.chdir('/home/benchmark/llm/ds4-flash-rs')
sys.path.insert(0,'/tmp/ds4-edit-gap');import run as b
root=Path.cwd(); p=Path(sys.argv[1]);cfg=json.loads(p.read_text())
gold=[(r['content_sha256'],r['reasoning_sha256']) for r in json.loads(Path('/tmp/ds4-edit-gap/rust-a.jsonl').read_text().splitlines()[1])['rows']]
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',{'DS4_EXPERT_EVICTION':'lfu'},b.native)
 try:
  for label,img,env in cfg:
   b.leg(label,'native',img,env)
   raw=(b.p/(label+'.jsonl')).read_bytes()
   samples=[json.loads(l) for l in raw.splitlines()]
   assert len(samples)==7 and all([(v['content_sha256'],v['reasoning_sha256']) for v in w['rows']]==gold for w in samples)
   (p.parent/(label+'.jsonl')).write_bytes(raw)
   (p.parent/(label+'.resource.json')).write_bytes((b.p/(label+'.resources.json')).read_bytes())
   ws=samples[1:]
   data=dict(label=label,image=samples[0]['image'],env=env,median_tok_s=statistics.median(w['upstream_tok_s'] for w in ws),whole_wave=statistics.median(w['whole_wave_tok_s'] for w in ws),per_wave=[w['upstream_tok_s'] for w in ws],matches=[6]*6,raw_sha256=__import__('hashlib').sha256(raw).hexdigest())
   (p.parent/(label+'.info')).write_text(json.dumps(data,indent=2))
 finally:b.restore()
