from pathlib import Path
import os,sys,fcntl,subprocess as sp
import importlib.util
s=importlib.util.spec_from_file_location('h','/tmp/ds4-cycle2-1126/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=m.p/sys.argv[1];b=m.b
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',m.flags,b.native);b.idle('native')
 try:
  sp.run(b.comp+b.base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  uuid=b.read('nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader')
  with (p/'micro.txt').open('w') as out:sp.run([str(p/'micro')],stdout=out,cwd=p,env=os.environ|{'CUDA_VISIBLE_DEVICES':uuid},check=True)
 finally:
  b.restore();b.verify('native',m.flags,b.native)
