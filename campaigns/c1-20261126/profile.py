import sys
sys.path.insert(0,'/tmp/ds4-edit-gap')
from run import *
native='ds4-flash-rs:attention-topk-checkpoint'
base=['-f','compose.yml','-f','compose.decode-optimized.yml']
import subprocess,os,secrets
from urllib.request import Request,urlopen
p=Path('/tmp/ds4-c1-1126/timeline')
docker=comp+base
def profile():
 im=verify('native',{'DS4_EXPERT_EVICTION':'lfu'},native)
 return dict(image=im,env={k:v for k,v in inspect()[2].items() if k.startswith('DS4_')})
def drain():idle('native')
def command(label,args):
 with (p/(label+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=f,check=True)
from threading import Thread
import secrets
from contextlib import ExitStack
session='ds4-c1-1126'
box='ds4-c1-profile-1126'
nsys_bin='/opt/nvidia/nsight-systems/2026.5.2/target-linux-x64/nsys'
run_ns=['docker','exec',box,nsys_bin]
local='http://127.0.0.1:18005'
ephemeral=secrets.token_hex(24)
def metrics():
 with urlopen(Request(local+'/metrics',headers={'Authorization':'Bearer '+ephemeral}),timeout=2) as r:return json.load(r)
def nsys(args,log,timeout=30):
 return subprocess.run(run_ns+args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def capture(n,limit,seconds):
 name='profile-c'+str(n);stamp(name+' start')
 baseline=metrics()['requests_started']; assert not metrics()['logical_active']
 cmd=[sys.executable,'tools/edit_scaling.py','--clients',str(n),'--waves','2','--base',local]
 with ExitStack() as files:
  out=files.enter_context((p/(name+'.out')).open('w')); err=files.enter_context((p/(name+'.err')).open('w'));log=files.enter_context((p/(name+'-nsys.log')).open('w'))
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,DSV41_API_KEY=ephemeral))
  frames=[]
  def sample():
   while proc.poll() is None:
    try:
     t=time.monotonic();d=metrics();frames.append(dict(t=t,started=d['requests_started'],decoding=d['decoding'],active=d['logical_active'],processed=d['processed_tokens'],joint=d['six_target_passes']))
    except OSError:pass
    time.sleep(.075)
  def memory_guard():
   while proc.poll() is None:
    try:
     gpu=[float(v) for v in read('nvidia-smi','-i','1,2','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()]
     ram=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])//1024
     if min(gpu)<128 or ram<8192:raise RuntimeError('profiling memory pressure')
    except Exception:
     proc.terminate();return
    time.sleep(1)
  memory_worker=Thread(target=memory_guard,daemon=True);memory_worker.start()
  observer=Thread(target=sample,daemon=True);observer.start()
  try:
   until=time.monotonic()+240
   while time.monotonic()<until and proc.poll() is None:
    if frames and frames[-1]['decoding']==n and frames[-1]['started']==baseline+2*n:break
    time.sleep(.025)
   else:raise RuntimeError(name+' missed fully resident second wave')
   start=time.monotonic()
   nsys(['start','--session='+session,'--sample=none','--cpuctxsw=none','-o',str(p/name)],log)
   armed=time.monotonic();time.sleep(seconds);stop=time.monotonic()
   nsys(['stop','--session='+session],log,90)
   rc=proc.wait(timeout=240);assert rc==0
   inside=[x for x in frames if start<=x['t']<=stop+.15]
   assert len(inside)>=5 and all(x['started']==baseline+2*n and x['decoding']==x['active']==n for x in inside)
   assert inside[-1]['processed']>inside[0]['processed'] and (n==1 or inside[-1]['joint']>inside[0]['joint'])
   (p/(name+'-scope.json')).write_text(json.dumps(dict(first=inside[0],last=inside[-1],sample_n=len(inside),start=start,armed=armed,stop=stop,observations=inside),indent=2))
   stamp(name+' complete')
  finally:
   if proc.poll() is None:
    proc.terminate();proc.wait(timeout=30)
   observer.join(timeout=3);memory_worker.join(timeout=3)
def main():
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  config=profile();drain()
  command('pause-profile',docker+['stop','--timeout','-1'])
  try:
   start=docker+['run','--no-deps','-d','--name',box,'-v','/opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro','-v',f'{p}:{p}','-e','HOME=/tmp','-e','DSV41_API_KEY','-e','LITELLM_MASTER_KEY','--entrypoint','/bin/sleep',svc,'infinity']
   env=dict(os.environ,DSV41_API_KEY=ephemeral,LITELLM_MASTER_KEY=ephemeral)
   with (p/'profile-server.log').open('w') as log:
    subprocess.run(start,env=env,stdout=log,stderr=log,check=True)
    assert read('docker','inspect','-f','{{.Image}}',box)==config['image']
    snap=json.loads(read('docker','inspect',box))[0]
    vals=dict(v.split('=',1) for v in snap['Config']['Env'] if '=' in v)
    assert all(vals.get(k)==v for k,v in config['env'].items())
    assert vals['LITELLM_MASTER_KEY']==vals['DSV41_API_KEY']==ephemeral
    args=run_ns+['launch','--session-new='+session,'--trace=cuda','/usr/local/bin/ds4-flash-rs','serve','--bind','127.0.0.1:18005','--max-context','524288','--chunk-size','64']
    proc=subprocess.Popen(args,stdout=log,stderr=log)
    try:
     for i in range(40):
      try:
       if metrics()['max_seqs']==6:break
      except OSError:pass
      time.sleep(1)
     else:raise RuntimeError('exact image profiler failed readiness')
     capture(1,1500,1.2)
    finally:
     while proc.poll() is None:
      try:
       if not metrics()['logical_active']:break
      except OSError:break
      time.sleep(1)
     subprocess.run(run_ns+['shutdown','--session='+session,'--kill=sigterm'],stdout=log,stderr=log,timeout=45)
     proc.wait(timeout=40)
  finally:
   subprocess.run(['docker','rm','-f',box],capture_output=True,timeout=60)
   command('restore-profile',docker+['up','--no-build','-d','--wait','--wait-timeout','120'])
  assert profile()==config
  stamp('profile finished, tagged service healthy')
if __name__=='__main__':main()
