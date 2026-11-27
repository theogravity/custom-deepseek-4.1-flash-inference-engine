from run import *
import statistics
def shell(label,cmd):
 with (p/(label+'.jsonl')).open('w') as out,(p/(label+'.err')).open('w') as err:
  proc=subprocess.Popen(b.runner+cmd,stdout=out,stderr=err,env=dict(os.environ,DSV41_API_KEY=b.key,PYTHONPATH=str(b.root/'tools')))
  try:
   pid=json.loads(b.read('docker','inspect',svc))[0]['State']['Pid']
   cg=Path('/proc',str(pid),'cgroup').read_text().split('::')[-1].strip().lstrip('/')
   stat=Path('/sys/fs/cgroup')/cg/'memory.stat'
   samples=[]
   while proc.poll() is None:
    gpu=[int(v) for v in b.read('nvidia-smi','-i','1,2','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()]
    ram=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])//1024
    anon=int(next(l.split()[1] for l in stat.read_text().splitlines() if l.startswith('anon ')))
    samples.append([*gpu,ram,anon/2**20])
    if ram<8192 or (len(samples)>=5 and all(min(v[:2])<128 for v in samples[-5:])):raise RuntimeError(f'watchdog {samples[-5:]}')
    time.sleep(1)
   assert proc.returncode==0
   (p/(label+'-memory.json')).write_text(json.dumps(dict(min_free_mib=[min(x[i] for x in samples) for i in range(3)],peak_sampled_cgroup_anon_mib=max(x[3] for x in samples))))
  finally:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=45)
 return [json.loads(l) for l in (p/(label+'.jsonl')).read_text().splitlines()]

def full(label,image):
 start(label,image,lfu,False)
 # Verify the transition from recent coding experts into the full-window receipt workload.
 warm=shell(label+'-training',['tools/long_decode.py','--histories',b.frozen,'--clients','6','--output','512','--rounds','1'])
 assert all([r['sha256'] for r in w['rows']]==ref['coding']['qualified-six512']['output_sha256'] for w in warm[:-1])
 data=shell(label,['/tmp/ds4-eviction-1126/six_streams_idle.py','--input','524160','--output','128','--continue-histories',b.frozen,'--rounds','7'])
 gold=ref['c6_full_retrieval']['ram-six']
 assert len(data)==7
 for i,w in enumerate(data):
  a,z=w['metrics_before'],w['metrics_after']
  assert w['all_valid'] and w['concurrent'] and w['six_target_passes_delta']>0 and a['rotations']==z['rotations']
  assert [r['sha256'] for r in w['rows']]==gold['output_sha256']
  assert [r['extended_ids_sha256'] for r in w['rows']]==gold['prompt_ids_sha256']
  if i:assert z['prefix_reused_tokens']-a['prefix_reused_tokens']==6*524096
 b.stamp('six full windows qualified')
def eight(tier):
 label='candidate-eight-'+tier
 opts=dict(lfu)
 if tier=='disk': opts['DS4_SWAP_RAM_MIB']='0'
 start(label,cand,opts,False)
 data=shell(label,['tools/kv_rotation.py','--agents','8','--input','523776','--output','512','--shared','--expect-tier',tier])
 z=data[0]
 assert z['passed'] and z['peak_resident']==6 and z['peak_swapped']>=2
 assert [v['sha256'] for v in z['rows']]==[v['sha256'] for v in ref['eight_full_window']['rows']]
 b.stamp('eight full windows qualified '+tier)
def gpu():
 b.drain()
 compose('pre-tests',['stop','--timeout','-1'])
 meta=subprocess.check_output(['cargo','test','--release','--locked','--offline','--no-run','--message-format=json'],cwd=p/'work',text=True)
 elf=next(json.loads(l)['executable'] for l in meta.splitlines() if json.loads(l).get('executable'))
 assert Path(elf).is_file()
 cmd=b.docker+['run','--rm','--no-deps','-T','-v',elf+':/experiment-tests:ro','-e','DS4_TEST_PACK=/mnt/models/dsv41-flash-exl3','-e','DS4_TEST_CUDA_DEVICE=0','-e','DS4_TEST_DEVICE=0','--entrypoint','/experiment-tests',svc,'--include-ignored','--test-threads=1']
 with (p/'gpu-tests.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 compose('post-tests',['up','--no-build','-d','--wait','--wait-timeout','150'])
 b.stamp('GPU suite passed')
def phase(label,image):
 opts=dict(lfu,DS4_TIMING_RESTORES='1')
 start(label,image,opts,False)
 raw=shell(label,['tools/long_decode.py','--histories',b.frozen,'--clients','6','--output','512','--rounds','1'])
 for w in raw[:-1]:assert [r['sha256'] for r in w['rows']]==ref['coding']['qualified-six512']['output_sha256']
 lines=subprocess.check_output(['docker','logs',svc],stderr=subprocess.STDOUT,text=True).splitlines()
 rows=[json.loads(l) for l in lines if l.startswith('{') and '"kv_restore"' in l]
 rows=[r for r in rows if r['kind']=='prefix']
 assert sum(r['tier']=='ram' for r in rows)>=6
 (p/(label+'.json')).write_text(json.dumps(rows,indent=2))
 b.stamp(label+' complete')
def published(label,image):
 start(label,image,lfu,False)
 b.drain()
 with (p/(label+'.txt')).open('w') as f:
  subprocess.run([sys.executable,'-I',str(p/'decode_bench2.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,API_KEY=b.key),check=True)
 lines=(p/(label+'.txt')).read_text().splitlines()
 assert len(lines)==7 and lines[0].startswith('warm-up: prompt 25625 tokens')
 assert all(re.search(r'rep[01] [146] streams: per-stream .*aggregate [0-9]+ tok/s',l) for l in lines[1:])
 verify(image,lfu)
 b.stamp(label+' complete '+lines[-1])
if __name__=='__main__':
 with reserve():
  for stage in sys.argv[1:]:
   if stage.endswith('full'):full(stage,base if stage.startswith('control') else cand)
   elif stage=='gpu':gpu()
   elif stage=='phase':
    phase('control-phase',base);phase('candidate-phase',cand)
   else:eight(stage)

  if 'disk' in sys.argv[1:]:
   published('published-control',base)
   published('published-shared',cand)
