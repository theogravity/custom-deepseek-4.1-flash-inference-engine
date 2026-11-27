from pathlib import Path
import os,sys,json,fcntl
from urllib.request import Request,urlopen
os.chdir('/home/benchmark/llm/ds4-flash-rs');sys.path.insert(0,'/tmp/ds4-edit-gap');import run as b
p=Path('/tmp/ds4-cpu-eval/selected-service')
v=json.loads((p/'final-service-validation.json').read_text())
flags=v['flags']|{'DS4_EXPERT_EVICTION':'lfu'}
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  b.verify('native',flags,v['tag']);b.idle('native')
  b.up('activate-final-decode',['-f','compose.yml','-f','compose.decode-optimized.yml'])
  b.verify('native',flags,v['tag'])
  req=Request(b.url+'/v1/chat/completions',data=json.dumps({'model':'DeepSeek V4.1 Flash','messages':[{'role':'user','content':'Reply with exactly the numeral for two plus two.'}],'reasoning_effort':'none','max_tokens':24,'temperature':0}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+b.bench.key})
  with urlopen(req,timeout=180) as r:resp=json.load(r)
  reply=resp['choices'][0]['message']['content'];assert '4' in reply,repr(reply)
  m=b.idle('native'); assert m['max_seqs']==6 and m['max_live_seqs']==32
  print(json.dumps(dict(image=v['tag'],identity=v['image'],healthy=True,smoke_reply=reply,finish=resp['choices'][0]['finish_reason'],max_seqs=m['max_seqs'],max_live_seqs=m['max_live_seqs'],flags=flags),indent=2))
 except BaseException:
  b.restore();raise
