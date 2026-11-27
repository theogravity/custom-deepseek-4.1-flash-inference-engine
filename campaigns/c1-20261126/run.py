from pathlib import Path
import os,sys,json,fcntl,subprocess as sp,statistics as st,hashlib,time,importlib.util
from datetime import datetime,timezone
p=Path(__file__).resolve().parent
root=Path('/home/benchmark/llm/ds4-flash-rs');os.chdir(root);sys.path.insert(0,'/tmp/ds4-edit-gap')
import run as b
b.native='ds4-flash-rs:attention-topk-checkpoint';b.base=['-f','compose.yml','-f','compose.decode-optimized.yml']
flags={'DS4_EXPERT_EVICTION':'lfu','DS4_BATCH_TOPK':'1'}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module('old_native','/tmp/ds4-prefix-sharing/run.py');sys.modules['run']=old
q=module('old_checks','/tmp/ds4-prefix-sharing/qualify.py');sys.modules['run']=b
gold=json.loads((root/'results/long-context-tuning-375w.json').read_text())
assert hashlib.sha256(Path(old.b.frozen).read_bytes()).hexdigest()=='1bde155d1d810e589a9cc5a7c298d6438693b7fa5416ffa1df6bef937d960990'
def leg(folder,label,img,env,shapes,rounds=6):
 q.p=folder
 ident=b.start('c1-'+folder.name+'-'+label,'native',img,env) if label!='gold' else b.verify('native',flags,img)
 info={'image':ident,'env':env}
 for shape in shapes:
  tag=label+'-'+shape;before=b.idle('native');since=datetime.now(timezone.utc).isoformat()
  clients=6 if shape.startswith('six') else 1
  if shape in ['edit','sixedit']:
   ws=q.shell(tag,[str(p/'edit.py'),str(clients),str(rounds)])
   assert len(ws)==rounds+1;ms=ws[1:];key='upstream_tok_s'
   def dig(w):return [[r['prompt_sha256'],r['reasoning_sha256'],r['content_sha256']] for r in w['rows']]
   expected=dig(json.loads(Path('/tmp/ds4-edit-gap/rust-a.jsonl').read_text().splitlines()[1]))[:clients]
   assert all(w['delta']['prefix_reused_tokens']==clients*25600 for w in ms)
   assert all(dig(w)==expected for w in ws),(tag,'output mismatch')
   wall='whole_wave_tok_s'
  else:
   ws=q.shell(tag,['tools/long_decode.py','--histories',old.b.frozen,'--clients',str(clients),'--output','512','--rounds',str(rounds)])
   assert len(ws)==rounds+2 and ws[-1]['prompts']==gold['code_request_proofs'][:clients]
   expected=gold['coding']['qualified-one' if clients==1 else 'qualified-six512']['output_sha256']
   for w in ws[:-1]:
    assert [r['sha256'] for r in w['rows']]==expected and w['delta']['rotations']==0
    if w['warmup'] is False:assert w['delta']['prefix_reused_tokens']==(404096 if clients==1 else 2424704)
   key='aggregate_decode_tok_s';wall='wall_tok_s';ms=ws[1:-1]
  z=b.idle('native'); assert z['requests_started']-before['requests_started']==(rounds+1)*clients and before['rotations']==z['rotations']
  if clients==6:assert z['six_target_passes']>before['six_target_passes']
  b.verify('native',flags|env,img)
  rates=[w[key] for w in ms];info[shape]=dict(rates=rates,median=st.median(rates),whole=st.median(w[wall] for w in ms),hashes=expected,raw_sha256=hashlib.sha256((folder/(tag+'.jsonl')).read_bytes()).hexdigest(),spec_including_warmup=b.stats(since))
  b.stamp(f'{folder.name} {label} {shape} {st.median(rates):.2f} tok/s')
 (folder/(label+'.json')).write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':
 folder=p/sys.argv[1];cfg=json.loads((folder/sys.argv[2]).read_text())
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',flags,b.native)
  for row in cfg:
   for suffix in ['.json']:(folder/(row[0]+suffix)).unlink(missing_ok=True)
  try:
   for args in cfg:leg(folder,*args)
  finally:
   if folder.name!='baseline':
    b.restore();b.verify('native',flags,b.native)
