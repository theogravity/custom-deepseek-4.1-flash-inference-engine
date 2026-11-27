from datetime import datetime, timezone
import argparse,fcntl,hashlib,json,os,re,subprocess as sp,sys,time,threading
from pathlib import Path
from urllib.request import Request,urlopen
p=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/ds4-agentic-live-1126')
from sweep import b, flags, images, watch, restore, deployed, check_mount
candidate='ds4-flash-rs:tool-prefix-ram-1126'

def sha(v):
 return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def response(body):
 req=Request(b.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+b.bench.key})
 wire_sha=hashlib.sha256(req.data).hexdigest()
 a=b.idle('native');admit_since=datetime.now(timezone.utc).isoformat();start=time.monotonic();first=last=None;reason=[];visible=[];calls={};done=False;finish=usage=None
 with urlopen(req,timeout=240) as f:
  for line in f:
   if not line.startswith(b'data:'):continue
   if line[5:].strip()==b'[DONE]':done=True;continue
   msg=json.loads(line[5:]);usage=msg.get('usage') or usage
   for c in msg.get('choices',[]):
    delta=c['delta']
    if any(delta.get(k) for k in ('content','reasoning_content','tool_calls')):
     last=time.monotonic()
     if first is None:first=last
    reason.append(delta.get('reasoning_content') or '');visible.append(delta.get('content') or '')
    for tc in delta.get('tool_calls') or []:
     slot=calls.setdefault(tc['index'],['','']);fn=tc.get('function',{})
     slot[0]+=fn.get('name') or '';slot[1]+=fn.get('arguments') or ''
    finish=c.get('finish_reason') or finish
 end=time.monotonic();wall=end-start
 assert done and first is not None and usage and finish in {'stop','tool_calls','length'}
 z=b.idle('native');assert z['requests_started']-a['requests_started']==1 and z['rotations']==a['rotations']
 out=(''.join(reason),''.join(visible),[(v[0],json.loads(v[1])) for _,v in sorted(calls.items())])
 return dict(admit_since=admit_since,start_mono=start,end_mono=end,idle_tokens_before=a.get('idle_tool_prefill_tokens',0),idle_tokens_after=z.get('idle_tool_prefill_tokens',0),wire_sha=wire_sha,input_sha=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),output_sha=sha(out),ordered_output_sha=hashlib.sha256(json.dumps(out).encode()).hexdigest(),reason_sha=sha(out[0]),text_sha=sha(out[1]),call_sha=sha(out[2]),prompt=usage['prompt_tokens'],tokens=usage['completion_tokens'],reuse=z['prefix_reused_tokens']-a['prefix_reused_tokens'],finish=finish,ttft=first-start,wall=wall,stream=last-first),out

def compare(out,following):
 history=next(m for m in reversed(following['messages']) if m['role']=='assistant')
 expected=(history.get('reasoning_content') or '',history.get('content') or '',[(c['function']['name'],json.loads(c['function']['arguments'])) for c in history.get('tool_calls',[])])
 return [json.dumps(actual)==json.dumps(original) for actual,original in zip(out,expected)]

def run(label,enabled,wide=False,idle=False,delay=0.,bound=None):
 root=p/('kv-'+label);root.mkdir(mode=0o700)
 image=('ds4-flash-rs:canonical-idle-final-1126' if label.startswith('q') else 'ds4-flash-rs:canonical-idle-grace-1126') if idle else images[1] if wide else candidate
 if bound is not None:
  assert idle and not wide
  image='ds4-flash-rs:pi-bound-router-1126'
 opts=flags(1)|{'DS4_TOOL_PREFIX':str(int(enabled))}
 override=p/(label+'.override.json');override.write_text(json.dumps({'services':{b.svc:{'image':image,'environment':{'DS4_TOOL_PREFIX':opts['DS4_TOOL_PREFIX']}}}}))
 if idle:
  opts=flags(1)|{'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':'32768' if enabled else '0'}
  override.write_text(json.dumps({'services':{b.svc:{'image':image,'environment':{'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':opts['DS4_IDLE_TOOL_PREFILL_MAX_TOKENS']}}}}))
  if bound is not None:
   env={'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':opts['DS4_IDLE_TOOL_PREFILL_MAX_TOKENS'],'DS4_BOUNDED_PREFILL_ROUTER':str(int(bound))}
   opts=flags(1)|env
   override.write_text(json.dumps({'services':{b.svc:{'image':image,'environment':env}}}))
  seed=p/'kv-width-off-a'
  assert seed.is_dir() and any(seed.glob('*.kv'))
  import shutil
  for f in seed.glob('*.kv'):shutil.copyfile(f,root/f.name)
 if wide:
  opts=flags(1)
  override.write_text(json.dumps({'services':{b.svc:{'image':image,'command':['serve','--bind','0.0.0.0:8001','--max-context','524288','--chunk-size','128']}}}))
 b.idle('native');os.environ['DS4_HOST_PREFIX_DIR']=str(root)
 try:b.up('pi-return-'+label,deployed+['-f',str(override)])
 finally:os.environ.pop('DS4_HOST_PREFIX_DIR')
 check_mount(root);image_id=b.verify('native',opts,image)
 assert json.loads(b.read('docker','inspect','--format','{{json .Config.Cmd}}',b.svc))[-1]==('128' if wide else '64')
 folder=p/'capture-b';rows=[];original=json.loads((folder/'task.json').read_text())
 assert original['isolated'] and original['graded_correct']
 # Warm the same engine kernels and static input prefix without caching any generated content.
 warm=json.loads((folder/'request-04.json').read_text());warm['max_tokens']=8
 response(warm)
 for n in (4,5,6):
  body=json.loads((folder/f'request-{n:02}.json').read_text())
  if n==5 and delay:time.sleep(max(0,rows[-1]['end_mono']+delay-time.monotonic()))
  row,out=response(body)
  row['gap_since_previous_s']=row['start_mono']-rows[-1]['end_mono'] if rows else None
  next_body=json.loads((folder/f'request-{n+1:02}.json').read_text())
  row['matches_recorded']=compare(out,next_body)
  source=original['turns'][n-1]
  assert row['input_sha']==source['request_sha256'] and row['prompt']==source['usage']['prompt_tokens']
  # Negative parity results are recorded as experimental failures.
  row['verified']=all(row['matches_recorded']) and row['tokens']==source['usage']['completion_tokens'] and row['finish']==source['finish']
  logs=sp.check_output(['docker','logs','--since',row['admit_since'],b.svc],stderr=sp.STDOUT,text=True)
  marker=f"prefill tokens={row['prompt']} reused={row['reuse']} "
  assert logs.count(marker)==1
  logs=logs.split(marker)[0]
  hits=re.findall(r'prefix tier=(ram|disk|tool_ram) tokens=(\d+)',logs)
  row['restore_tier']=hits[-1][0] if hits else None
  assert not row['reuse'] or (hits and int(hits[-1][1])==row['reuse'])
  row['image']=image_id;row['turn']=n;row['mode']=enabled;rows.append(row)
  (p/(label+'.json')).write_text(json.dumps(rows,indent=2))
  print(label,n,'prefill',row['prompt'],row['reuse'],'generated',row['tokens'],'ttft',round(row['ttft'],3),'wall',round(row['wall'],2),'source match',row['matches_recorded'],flush=True)
 b.verify('native',opts,image)
 return all(r['verified'] for r in rows)

if __name__=='__main__':
 forarg=argparse.ArgumentParser();forarg.add_argument('labels',nargs='+');forarg.add_argument('--delay',type=float,default=0.);args=forarg.parse_args()
 with open('/tmp/ds4-native-bench.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b.verify('native',flags(1),images[1]);stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
  try:
   passed=[run(label,'on' in label,'wide' in label,'idle' in label,args.delay) for label in args.labels]
   if not all(passed):raise RuntimeError('captured output parity failed; keep rejected receipts')
  finally:
   stop.set();t.join();restore()
