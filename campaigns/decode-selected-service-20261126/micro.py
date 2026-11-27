from pathlib import Path
import os,fcntl,sys,subprocess as sp
os.chdir('/home/benchmark/llm/ds4-flash-rs')
sys.path.insert(0,'/tmp/ds4-edit-gap')
import run as b
p=Path(sys.argv[1]); exe=p/'micro'
assert exe.is_file()
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',{'DS4_EXPERT_EVICTION':'lfu'},b.native);b.idle('native')
 try:
  sp.run(b.comp+b.base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  uuid=b.read('nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader')
  with (p/'micro.txt').open('w') as out:
   sp.run([str(exe)],stdout=out,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid))
 finally:b.restore()
