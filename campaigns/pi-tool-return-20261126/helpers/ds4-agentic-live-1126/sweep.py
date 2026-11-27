import fcntl,json,os,sys,time,threading,signal
from pathlib import Path
from trial import one,p
os.chdir('/home/benchmark/llm/ds4-flash-rs');sys.path.insert(0,'/tmp/ds4-edit-gap')
import run as b
base=['-f','compose.yml'];deployed=base+['-f','compose.decode-round2.yml'];prior=base+['-f','compose.c1-optimized.yml']
images=['ds4-flash-rs:c1-consensus-20261126','ds4-flash-rs:c2-verified-20261126']
def flags(idx):return {'DS4_EXPERT_EVICTION':'lfu','DS4_C1_OUTPUT_CONSENSUS':'1','DS4_BATCH_TOPK':'1','DS4_LOOKUP_K':'128','DS4_OUTPUT_LOOKUP':'single','DS4_ADAPTIVE_LOOKUP':'1','DS4_PROMPT_LOOKUP':'1'}|({'DS4_EXTEND_REPETITION':'1','DS4_HEAD_ROW_TILES':'1'} if idx else {})
def sample_memory(stop):
 low=0
 import subprocess as sp
 while not stop.wait(.9):
  vals=sp.check_output(['nvidia-smi','-i','1,2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
  low=low+1 if min(map(int,vals))<128 else 0
  ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))//1024
  if low>=5 or ram<8192:os.kill(os.getpid(),signal.SIGINT);return
def watch(stop):
 try:sample_memory(stop)
 except Exception as error:
  print('Watchdog failed:', type(error).__name__,flush=True)
  os.kill(os.getpid(),signal.SIGINT)
def check_mount(expected):
 actual=b.read('docker','inspect','--format','{{range .Mounts}}{{if eq .Destination "/prefix"}}{{.Source}}{{end}}{{end}}',b.svc)
 assert Path(actual).resolve()==expected.resolve(),(actual,expected)
def serve(label,idx):
 b.idle('native')
 path=p/('kv-'+label);path.mkdir(exist_ok=False,mode=0o700)
 os.environ['DS4_HOST_PREFIX_DIR']=str(path)
 try:b.up('agentbench-'+label,deployed if idx else prior)
 finally:os.environ.pop('DS4_HOST_PREFIX_DIR')
 check_mount(path);return b.verify('native',flags(idx),images[idx])
def restore():
 # Drain any active caller before a replacement. Missing or unhealthy containers
 # use the same image-verified recovery path as the other workstation sweeps.
 b.base=deployed;b.native=images[1]
 while True:
  import subprocess
  state=subprocess.run(['docker','inspect','--format','{{.State.Health.Status}}',b.svc],capture_output=True,text=True)
  if state.returncode or state.stdout.strip()!='healthy':break
  try:b.idle('native');break
  except RuntimeError:continue
  except Exception:break
 b.restore();b.verify('native',flags(1),images[1]);check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
def run():
 if 'DS4_HOST_PREFIX_DIR' in os.environ:raise ValueError('unexpected prefix override')
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',flags(1),images[1])
  try:
   for label,idx in [('before',0),('selected',1),('after',0)]:
    image=serve(label,idx)
    with (p/(label+'.jsonl')).open('x') as out:
     for sample in range(7):
      a=b.idle('native');stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
      try:r=one(b.bench.key,sample)
      finally:stop.set();t.join()
      z=b.idle('native');b.verify('native',flags(idx),images[idx])
      r.update(image=image,leg=label,warmup=sample==0,settings=flags(idx),engine={k:z[k]-a[k] for k in ('requests_started','prefix_reused_tokens','rotations')})
      if r.get('turns') and r['engine']['requests_started']!=len(r['turns']):r['excluded']='unrelated engine requests'
      print(json.dumps(r),file=out,flush=True)
      print(label,sample,r['passed'],round(r['wall_s'],2) if r['wall_s'] is not None else None,len(r.get('turns',[])),r.get('workflow'),r.get('error',''),r.get('excluded',''),flush=True)
      if sample==0 and (r.get('error') or not r.get('complete_streams')):raise RuntimeError('warm-up instrumentation failed')
  finally:
   restore()
if __name__=='__main__':run()
