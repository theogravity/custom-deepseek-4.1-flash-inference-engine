from run import *
deadline=time.monotonic()+2400
while not (p/'micro.jsonl').exists() or len((p/'micro.jsonl').read_text().splitlines())<162:
 if (p/'screen.log').exists() and 'Traceback' in (p/'screen.log').read_text():raise RuntimeError('microbenchmark failed')
 if time.monotonic()>deadline:raise TimeoutError('screen waiting')
 time.sleep(4)
sp.run([sys.executable,str(p/'pick.py')],check=True)
m=(p/'tile-selection.json')
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
 try:
  leg('rust-tune-a','native',native,{})
  leg('k5','native',native,{'DS4_SPEC_K':'5'})
  leg('rust-tune-b','native',native,{})
  if m.exists():
   label='tile'+str(json.loads(m.read_text())['warps'])
   leg(label,'native','ds4-flash-rs:edit-gap-'+label,{})
   leg('rust-post','native',native,{})
 finally:restore()
