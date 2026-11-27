from run import *
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
 try:
  lab='moe-stats'
  start(lab,'native',native,{'DS4_MOE_STATS':'1'})
  gold=[(z['content_sha256'],z['reasoning_sha256']) for z in json.loads((p/'rust-a.jsonl').read_text().splitlines()[1])['rows']]
  out=[];samples=[]
  for i in range(3):
   idle('native');since=datetime.now(timezone.utc).isoformat()
   w=checked_wave(6,samples)
   idle('native')
   assert gold==[(z['content_sha256'],z['reasoning_sha256']) for z in w['rows']]
   logs=sp.check_output(['docker','logs','--since',since,svc],stderr=sp.STDOUT,text=True)
   pattern=r'^moe pass=(\w+) rows=(\d+) routes=(\d+) unique_layer_experts=(\d+) expert_uploads=(\d+) expert_h2d_bytes=(\d+) capacity_queries=(\d+)$'
   vals={}
   for k,*nums in re.findall(pattern,logs,re.M):
    d=vals.setdefault(k,{x:0 for x in ['passes','rows','routes','unique_expert_accesses','uploads','bytes','queries']})
    d['passes']+=1
    for field,val in zip(['rows','routes','unique_expert_accesses','uploads','bytes','queries'],nums):d[field]+=int(val)
   assert vals and all(a['uploads']*13315584==a['bytes'] for a in vals.values())
   out.append(dict(warmup=not i,pass_counters=vals,seconds=9000/w['whole_wave_tok_s']))
  (p/'moe-telemetry.json').write_text(json.dumps(out,indent=2))
  stamp('native MoE counters complete')
 finally:restore()
