import os,sys,subprocess as sp,fcntl,json,re
from pathlib import Path
root=Path('/home/benchmark/llm/ds4-flash-rs');os.chdir(root);sys.path.insert(0,'/tmp/ds4-edit-gap');import run as b
p=Path('/tmp/ds4-cpu-eval')
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 b.verify('native',{'DS4_EXPERT_EVICTION':'lfu'},b.native);b.idle('native')
 args=sys.argv[1:]
 test,name=args[:2]; flags=dict(k.split('=',1) for k in args[2:])
 path=Path(test); assert path.is_file()
 ids=b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
 env=dict(os.environ,DS4_TEST_DEVICE='0',CUDA_VISIBLE_DEVICES=','.join(ids),CUDA_DEVICE_ORDER='PCI_BUS_ID',DS4_TEST_PACK='/mnt/models/dsv41-flash-exl3',DS4_TEST_CUDA_DEVICE='0',**flags)
 try:
  sp.run(b.comp+b.base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  outpath=p/('cpu-fixed-test.log' if name=='fixed_routes_match_reference_within_cpu_precision' else name+'.test.log')
  with outpath.open('w') as log:
   sp.run([str(path),name,'--ignored','--nocapture','--test-threads=1'],env=env,check=True,stdout=log,stderr=sp.STDOUT)
  assert '1 passed; 0 failed' in outpath.read_text()

 finally:b.restore()
