from pathlib import Path
import subprocess as sp,sys,os,fcntl,re
sys.path.insert(0,'/tmp/ds4-c1-1126')
import importlib.util
s=importlib.util.spec_from_file_location('h','/tmp/ds4-c1-1126/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=m.b;p=m.p/sys.argv[1];flag=sys.argv[2];name=sys.argv[3] if len(sys.argv)>3 else ''
elf=sp.check_output(['python3','/tmp/ds4-cpu-eval/testbin.py',p/'tests.json'],text=True).strip()
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',m.flags,b.native);b.idle('native')
 ids=b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
 env=os.environ|{'CUDA_VISIBLE_DEVICES':','.join(ids),'DS4_TEST_DEVICE':'0','DS4_TEST_CUDA_DEVICE':'0','DS4_TEST_PACK':'/mnt/models/dsv41-flash-exl3','DS4_BATCH_TOPK':'1',**__import__('json').loads((p/'config.json').read_text())[1][2]}
 try:
  sp.run(b.comp+b.base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  opts=[name,'--ignored'] if name else ['--include-ignored']
  with (p/'gpu-tests.log').open('w') as f:sp.run([elf,*opts,'--test-threads=1'],cwd=p/'work',env=env,stdout=f,stderr=sp.STDOUT,check=True)
  assert re.search(r'test result: ok\. [1-9][0-9]* passed; 0 failed', (p/'gpu-tests.log').read_text())
 finally:
  b.restore();b.verify('native',m.flags,b.native)
