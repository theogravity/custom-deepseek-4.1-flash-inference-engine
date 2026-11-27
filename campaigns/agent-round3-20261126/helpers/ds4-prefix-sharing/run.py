from pathlib import Path
from statistics import median
from contextlib import contextmanager
from threading import Event, Thread
import sys, os, json, subprocess, re, time, fcntl
sys.path.insert(0,'/tmp/ds4-next-375w-2301')
import bench as b
p=Path(__file__).resolve().parent
svc=b.svc
def drained():
 for _ in range(180):
  m=b.native('/metrics')
  if all(m[k]==0 for k in ('logical_active','waiting','resident','swapped')):return
  time.sleep(1)
 raise RuntimeError('service in use')
b.drain=drained
b.docker+=['-f','compose.expert-cache-lfu.yml']
base='ds4-flash-rs:expert-cache-policy-3209b98'
cand='ds4-flash-rs:shared-prefix-test'
lfu={'DS4_EXPERT_EVICTION':'lfu'}
ref=json.loads((b.root/'results/long-context-tuning-375w.json').read_text())
def compose(label,flags):
 with (p/(label+'.compose-log')).open('a') as f:
  subprocess.run(b.docker+flags,stdout=f,stderr=f,check=True)
def inspect():
 snap=json.loads(b.read('docker','inspect',svc))[0]
 return snap['Image'],snap['State']['Health']['Status'],dict(x.split('=',1) for x in snap['Config']['Env'] if '=' in x)
def verify(image,override):
 expected=b.read('docker','image','inspect','-f','{{.Id}}',image)
 active,health,env=inspect()
 assert active==expected and health=='healthy'
 for k,v in b.pack['environment'].items():
  if k.startswith('DS4_'):assert env.get(k)==override.get(k,v),(k,env.get(k))
 for k,v in override.items():assert env[k]==v
 for k in ['DS4_SPEC_K','DS4_EXPERT_TRACE_DIR','DS4_EXPERT_EVICTION']:
  assert env.get(k)==override.get(k),(k,env.get(k))
 powers=b.read('nvidia-smi','-i','1,2','--query-gpu=name,enforced.power.limit','--format=csv,noheader,nounits').splitlines()
 assert len(powers)==2 and all('RTX PRO 6000' in x and x.endswith('375.00') for x in powers)
 hw=next(k for k in Path('/sys/class/hwmon').glob('hwmon*') if (k/'name').read_text().strip()=='amd_hsmp_hwmon')
 assert int((hw/'power1_cap').read_text())==225000000
 return active
def start(label,image,override,trace):
 b.drain()
 opts={'image':image,'environment':override}
 if trace:opts['volumes']=[str(p/label)+':/experiment-trace']
 cfg=p/(label+'-override.json');cfg.write_text(json.dumps({'services':{svc:opts}}))
 compose(label,['-f',str(cfg),'up','--no-build','--force-recreate','-d','--wait','--wait-timeout','150'])
 return verify(image,override)
def run(label,image,override,trace=False,clients=6,output=512):
 b.stamp(label+' start')
 ident=start(label,image,override,trace)
 shape=('qualified-six512' if clients==6 and output==512 else ('qualified-one' if clients==1 and output==512 else 'ram-six'))
 if trace:
  src=(b.root/'tools/long_decode.py').read_text().replace('        start = monotonic()','        start_wall_us=__import__("time").time_ns()//1000\n        start = monotonic()').replace('dict(wave=wave, warmup=not wave,','dict(wave=wave, start_wall_us=start_wall_us, end_wall_us=__import__("time").time_ns()//1000, warmup=not wave,')
  script=p/'long_decode_trace.py';script.write_text(src)
 else:script=b.root/'tools/long_decode.py'
 cmd=b.runner+[str(script),'--histories',b.frozen,'--clients',str(clients),'--output',str(output),'--sample-steady']
 pid=json.loads(b.read('docker','inspect',svc))[0]['State']['Pid']
 cgpath=Path('/proc',str(pid),'cgroup').read_text().split('::')[-1].strip().lstrip('/')
 stat=Path('/sys/fs/cgroup')/cgpath/'memory.stat'
 samples=[]
 with (p/(label+'.jsonl')).open('w') as out,(p/(label+'.err')).open('w') as err:
  pr=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,DSV41_API_KEY=b.key,PYTHONPATH=str(b.root/'tools')))
  while pr.poll() is None:
   gpu=[int(x) for x in b.read('nvidia-smi','-i','1,2','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()]
   ram=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])//1024
   anon=int(next(x.split()[1] for x in stat.read_text().splitlines() if x.startswith('anon ')))
   samples.append([*gpu,ram,anon])
   if ram<8192 or (len(samples)>=5 and all(min(s[:2])<128 for s in samples[-5:])):
    (p/(label+'-watchdog.json')).write_text(json.dumps(samples))
    pr.terminate();pr.wait(timeout=30);raise RuntimeError(f'memory watchdog: {samples[-5:]}')
   time.sleep(1)
  assert pr.returncode==0
 verify(image,override)
 raw=[json.loads(x) for x in (p/(label+'.jsonl')).read_text().splitlines()]
 assert raw[-1]['prompts']==ref['code_request_proofs'][:clients]
 hashes=ref['coding'][shape]['output_sha256']
 assert len(raw)==8 and all([v['sha256'] for v in x['rows']]==hashes and x['delta']['rotations']==0 for x in raw[:-1])
 waves=raw[1:-1]
 info=dict(image=ident,sha256=hashes,env=override,clients=clients,output=output,free_mib_min=[min(x[i] for x in samples) for i in range(3)],waves=[{k:w[k] for k in ['aggregate_decode_tok_s','wall_tok_s','first_response_spread_s','resident_decode','delta']} for w in waves],delivery=raw[-1]['median_decode_tok_s'],wall=raw[-1]['median_wall_tok_s'],resident=median(w['resident_decode']['tok_s'] for w in waves))
 info['peak_anon_mib']=max(s[3] for s in samples)/2**20
 info['restore_lines']= [l for l in subprocess.check_output(['docker','logs',svc],stderr=subprocess.STDOUT,text=True).splitlines() if '"kv_restore"' in l]
 assert all(w['delta']['requests_started']==clients and w['delta']['prefix_reused_tokens']==(2424704 if clients==6 else w['delta']['prefix_reused_tokens']) and (clients==1 or w['delta']['six_target_passes']>0) for w in waves)
 if trace:
  from trace_check import check
  logs=subprocess.check_output(['docker','logs',svc],stderr=subprocess.STDOUT,text=True)
  info['manifest']=check(p/label,logs,raw)
 (p/(label+'.json')).write_text(json.dumps(info,indent=2))
 b.stamp(label+' complete '+str(round(info['delivery'],1)))
@contextmanager
def reserve():
 with open('/tmp/ds4-native-bench.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  verify(base,lfu)
  try:yield
  finally:
   try:
    b.drain()
   except Exception:
    probe=subprocess.run(['docker','inspect',svc],capture_output=True,text=True)
    try:state=json.loads(probe.stdout)[0]['State'] if probe.returncode==0 else {}
    except (ValueError,KeyError,TypeError,IndexError):state={}
    if state.get('Running') and state.get('Health',{}).get('Status')=='healthy':
     # The HTTP health check is TCP-level. Allow a live session to finish before replacement.
     deadline=time.monotonic()+240
     while time.monotonic()<deadline:
      try:
       if not b.native('/metrics')['logical_active']:break
      except (OSError,ValueError,KeyError,TypeError):pass
      time.sleep(1)
     else:
      # Stop admission and let axum's graceful shutdown and worker join drain outstanding work.
      # A negative timeout tells Docker to wait without forcing SIGKILL.
      compose('graceful-stop',['stop','--timeout','-1'])
   compose('restore',['up','--no-build','-d','--wait','--wait-timeout','150'])
   verify(base,lfu)

if __name__=='__main__':
 with reserve():
  for name in sys.argv[1:]:
   run(name,base if name.startswith('control') else cand,dict(lfu,**({'DS4_TIMING_RESTORES':'1'} if 'phases' in name else {})), clients=1 if 'one' in name else 6, output=768 if '768' in name else 512)
