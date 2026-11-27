from pathlib import Path
import sys,os,json,fcntl,subprocess as sp,re
p=Path(sys.argv[1]).resolve()
elf=sp.check_output([sys.executable,str(p.parent/'testbin.py'),str(p/'tests.json')],text=True).strip()
flags=json.loads((p/'long.json').read_text())[1]['env']
os.chdir('/home/benchmark/llm/ds4-flash-rs');sys.path.insert(0,'/tmp/ds4-edit-gap');import run as b
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',{'DS4_EXPERT_EVICTION':'lfu'},b.native);b.idle('native')
 ids=b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
 env=dict(os.environ,DS4_TEST_DEVICE='0',CUDA_VISIBLE_DEVICES=','.join(ids),CUDA_DEVICE_ORDER='PCI_BUS_ID',DS4_TEST_PACK='/mnt/models/dsv41-flash-exl3',DS4_TEST_CUDA_DEVICE='0',**flags)
 try:
  sp.run(b.comp+b.base+['stop','--timeout','-1'],check=True,stdout=sp.DEVNULL)
  with (p/'full-tests.log').open('w') as f:
   sp.run([elf,'--include-ignored','--test-threads=1'],cwd=p/'work',stdout=f,stderr=sp.STDOUT,env=env,check=True)
  txt=(p/'full-tests.log').read_text();assert re.search(r'test result: ok\. [1-9][0-9]+ passed; 0 failed; 0 ignored;',txt)
 finally:b.restore()
