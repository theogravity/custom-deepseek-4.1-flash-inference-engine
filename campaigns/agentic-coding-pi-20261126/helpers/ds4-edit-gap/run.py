from pathlib import Path
from datetime import datetime,timezone
import sys,os,statistics,json,time,re,fcntl,subprocess as sp,threading
from urllib.request import Request,urlopen
p=Path(__file__).resolve().parent
root=Path.cwd()
sys.path[:0]=[str(root/'tools'),'/tmp/ds4-next-375w-2301']
import edit_scaling as edit,bench
svc=bench.svc; native='ds4-flash-rs:shared-prefix-3d8c541'
vimg='dsv41-flash-1017-exl3-3bpw-sm120-lmcache:latest'
comp=['docker','compose','--project-name','models','--env-file','../models/.env']
base=['-f','compose.yml','-f','compose.shared-prefix-lfu.yml']
ref=['-f','../models/dsv41-flash-exl3.yml']
url='http://127.0.0.1:8001'
def stamp(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)
def read(*cmd):return sp.check_output(cmd,text=True).strip()
def inspect():
 d=json.loads(read('docker','inspect',svc))[0]
 return d['Image'],d['State'],dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
def ident(image):return read('docker','image','inspect','-f','{{.Id}}',image)
def metric(kind):
 with urlopen(Request(url+'/metrics',headers={'Authorization':'Bearer '+bench.key}),timeout=10) as f:txt=f.read().decode()
 if kind=='native':return {k:v for k,v in json.loads(txt).items() if isinstance(v,(int,float))}
 vals={}
 for line in txt.splitlines():
  m=re.fullmatch(r'(vllm:[a-zA-Z0-9_]+)(?:\{.*?\})?\s+([\deE.+-]+)(?:\s+\d+)?',line)
  if m:
   k=m[1][5:]
   if k.startswith(('spec_decode_', 'generation_tokens_total','num_requests_','request_success_total','prompt_tokens_total')) and not k.endswith(('_created','_bucket')):
    vals[k]=vals.get(k,0)+float(m[2])
 return vals
def idle(kind):
 for _ in range(300):
  a=metric(kind)
  if kind=='native':active=sum(a[k] for k in ['logical_active','waiting','resident','swapped'])
  else:active=sum(a[k] for k in ['num_requests_running','num_requests_waiting'])
  if not active:return a
  time.sleep(.1)
 raise RuntimeError('service still in use')
def verify(kind,env,image):
 img,s,vals=inspect()
 assert img==ident(image) and s['Health']['Status']=='healthy'
 assert all(vals.get(k)==v for k,v in env.items())
 g=read('nvidia-smi','-i','1,2','--query-gpu=name,enforced.power.limit','--format=csv,noheader,nounits').splitlines()
 assert len(g)==2 and all(v.endswith('375.00') and 'RTX PRO 6000' in v for v in g)
 hw=next(k for k in Path('/sys/class/hwmon').glob('hwmon*') if (k/'name').read_text().strip()=='amd_hsmp_hwmon');assert int((hw/'power1_cap').read_text())==225000000
 return img
def current_kind():return 'vllm' if inspect()[0]==ident(vimg) else 'native'
def up(label,files):
 with (p/(label+'.compose.log')).open('a') as f:
  sp.run(comp+files+['up','--no-build','--force-recreate','-d','--wait','--wait-timeout','1200'],check=True,stdout=f,stderr=sp.STDOUT)
def start(label,kind,image,over):
 idle(current_kind())
 env=({'DS4_EXPERT_EVICTION':'lfu'} if kind=='native' else {'DSV41_LMCACHE_DIR':'/root/.cache/lmcache-edit-gap'})|over
 cfg=p/(label+'.override.json');cfg.write_text(json.dumps({'services':{svc:{'image':image,'environment':env}}}))
 up(label,(base if kind=='native' else ref)+['-f',str(cfg)])
 return verify(kind,env,image)
def restore():
 # A failed Compose replacement may leave no container or only a partial boot.
 info=sp.run(['docker','inspect',svc],capture_output=True,text=True)
 if info.returncode==0 and json.loads(info.stdout)[0]['State'].get('Running'):
  try:
   idle(current_kind())
  except Exception:
   sp.run(['docker','stop','--timeout','-1',svc],check=True,stdout=sp.DEVNULL)
 up('restore',base)
 verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
def stats(since):
 logs=sp.check_output(['docker','logs','--since',since,svc],stderr=sp.STDOUT,text=True)
 totals={}
 for name,r,pr,a in re.findall(r'speculative source=(learned|lookup) rounds=(\d+) proposed=(\d+) accepted=(\d+)',logs):
  row=totals.setdefault(name,[0]*3)
  for i,v in enumerate([r,pr,a]):row[i]+=int(v)
 return totals
def cputime(pid):
 cg=Path('/proc',str(pid),'cgroup').read_text().split('::')[-1].strip().lstrip('/')
 path=Path('/sys/fs/cgroup')/cg/'cpu.stat'
 return int(next(l.split()[1] for l in path.read_text().splitlines() if l.startswith('usage_usec ')))/1e6
def checked_wave(clients,samples):
 import tempfile
 # Independent HTTPS client lifetime lets the watchdog actually cancel its sockets.
 with tempfile.TemporaryFile(mode='w+') as out:
  cmd=[sys.executable,str(root/'tools/edit_scaling.py'),'--waves','1','--clients',str(clients)]
  child=sp.Popen(cmd,stdout=out,env=dict(os.environ,DSV41_API_KEY=bench.key))
  try:
   while child.poll() is None:
    gpu=read('nvidia-smi','-i','1,2','--query-gpu=memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits').splitlines()
    gpu=[[float(i) for i in r.split(',')] for r in gpu]
    assert len(gpu)==2
    ram=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])//1024
    samples.append([time.time(),gpu,ram])
    if ram<8192 or (len(samples)>=5 and all(min(a[0] for a in z[1])<128 for z in samples[-5:])):
     raise RuntimeError(f'memory watchdog {samples[-1:]}')
    try:child.wait(timeout=1)
    except sp.TimeoutExpired:pass
   if child.returncode: raise RuntimeError(f'wave client exited {child.returncode}')
   out.seek(0);return json.load(out)
  finally:
   if child.poll() is None:
    child.terminate();child.wait(timeout=20)
def leg(label,kind,image,env,clients=6):
 stamp(label+' starting')
 identifier=start(label,kind,image,env)
 pid=inspect()[1]['Pid'];samples=[]
 stamp(label+' serving')
 try:
  with (p/(label+'.jsonl')).open('w') as f:
   for idx in range(7):
    a=idle(kind);since=datetime.now(timezone.utc).isoformat();cpu=cputime(pid);before=time.monotonic()
    w=checked_wave(clients,samples)
    w.pop('wave');w.pop('warmup')
    elapsed=time.monotonic()-before;cpu=cputime(pid)-cpu
    z=idle(kind)
    if kind=='vllm':
     deadline=time.monotonic()+10
     while time.monotonic()<deadline and (z.get('request_success_total',0)-a.get('request_success_total',0)!=clients or z.get('generation_tokens_total',0)-a.get('generation_tokens_total',0)!=1500*clients):
      time.sleep(.1);z=idle(kind)
    delta={k:v-a.get(k,0) for k,v in z.items() if v!=a.get(k,0)}
    if kind=='native':
     assert delta['requests_started']==clients and delta.get('rotations',0)==0
     if clients==6:assert delta['six_target_passes']>0
    else:assert delta.get('request_success_total')==clients and delta.get('generation_tokens_total')==1500*clients,(delta.keys(),delta.get('request_success_total'),delta.get('generation_tokens_total'))
    assert all(v['usage']['prompt_tokens'] in range(25570,25650) for v in w['rows'])
    print(json.dumps(dict(wave=idx,warmup=not idx,image=identifier,env=env,cpu_cores=cpu/elapsed,delta=delta,spec=stats(since) if kind=='native' else {},**w)),file=f,flush=True)
  verify(kind,env|({'DS4_EXPERT_EVICTION':'lfu'} if kind=='native' else {}),image)
 finally:
  (p/(label+'.resources.json')).write_text(json.dumps(samples))
 rows=[json.loads(l) for l in (p/(label+'.jsonl')).read_text().splitlines()][1:]
 stamp(f'{label} completed {statistics.median(w["upstream_tok_s"] for w in rows):.2f} aggregate token/s')
if __name__=='__main__':
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
  try:
   for label in sys.argv[1:]:
    if label.startswith('vllm'):leg(label,'vllm',vimg,{'EXL3_CPU_EXPERTS':'0' if 'off' in label else '1'})
    elif label.startswith('tile'):leg(label,'native','ds4-flash-rs:edit-gap-'+label,{})
    elif label.startswith('k5'):leg(label,'native',native,{'DS4_SPEC_K':'5'})
    else:leg(label,'native',native,{})
  finally:restore()
