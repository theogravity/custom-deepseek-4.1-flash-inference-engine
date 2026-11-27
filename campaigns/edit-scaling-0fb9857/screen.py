from run import *
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
 idle('native')
 try:
  stamp('micro starting on the first inference card')
  sp.run(comp+base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  # Lock + idle service leaves both inference GPUs free. CUDA_DEVICE_ORDER inside Compose maps only host GPUs 1/2.
  # Give the benchmark the same explicitly selected inference GPU UUID without exposing the desktop card.
  uuid=read('nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader')
  with (p/'micro.jsonl').open('w') as out:
   sp.run([str(p/'micro')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=out)
  rows=[json.loads(l) for l in (p/'micro.jsonl').read_text().splitlines()]
  assert len(rows)==6*9*3
  for jobs in sorted({r['jobs'] for r in rows}):
   print(jobs,{k:round(statistics.median(r['us'] for r in rows if r['warps']==k and r['jobs']==jobs),2) for k in [2,4,8]},flush=True)
 finally:restore()
