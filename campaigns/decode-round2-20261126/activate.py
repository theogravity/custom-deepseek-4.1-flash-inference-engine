from pathlib import Path
import sys,json,fcntl,subprocess as sp,importlib.util,os
s=importlib.util.spec_from_file_location('h','/tmp/ds4-cycle2-1126/run.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
p=h.p/sys.argv[1];tag=sys.argv[2];cfg=json.loads((p/'config.json').read_text())[1];b=h.b
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',h.flags,b.native);b.idle('native')
 try:
  assert b.ident(tag)==b.ident(cfg[1])
  b.up('c2-selected-activate',['-f','compose.yml','-f','compose.decode-round2.yml'])
  img=b.verify('native',h.flags|cfg[2],tag)
  req=b.Request(b.url+'/v1/chat/completions',data=json.dumps({'model':'DeepSeek V4.1 Flash','messages':[{'role':'user','content':'Return only the result of 11 plus 13.'}],'temperature':0,'reasoning_effort':'none','max_tokens':24}).encode(),headers={'Authorization':'Bearer '+b.bench.key,'Content-Type':'application/json'})
  with b.urlopen(req,timeout=180) as resp:body=json.load(resp)
  assert body['choices'][0]['message']['content'].strip().strip('.')=='24'
  m=b.idle('native')
  assert m['max_live_seqs']==32 and m['max_seqs']==6
  auth={'Authorization':'Bearer '+b.bench.key}
  with b.urlopen(b.Request(b.url+'/metrics',headers=auth),timeout=10) as resp:swap=json.load(resp)['swap']
  tiers={k:swap[k] for k in ['ram_limit_bytes','disk_limit_bytes']}
  assert tiers=={'ram_limit_bytes':48*2**30,'disk_limit_bytes':128*2**30}
  (p/'activation.json').write_text(json.dumps(dict(image=img,tag=tag,flags=cfg[2]|h.flags,smoke='11 + 13 = 24',limits={'max_seqs':m['max_seqs'],'max_live_seqs':m['max_live_seqs'],**tiers}),indent=2)+'\n')
 except BaseException:
  b.restore();b.verify('native',h.flags,b.native);raise
