import os,sys,json
from urllib.request import Request,urlopen
sys.path.insert(0,'/home/benchmark/llm/ds4-flash-rs/tools');from edit_scaling import wave
key=os.environ['DSV41_API_KEY'];url='http://127.0.0.1:8001';clients,rounds=map(int,sys.argv[1:])
def metrics():
 with urlopen(Request(url+'/metrics',headers={'Authorization':'Bearer '+key}),timeout=10) as r:return json.load(r)
for i in range(rounds+1):
 a=metrics();w=wave(url+'/v1/chat/completions',key,clients);z=metrics()
 d={k:z[k]-a[k] for k in ('requests_started','prefix_reused_tokens','rotations','six_target_passes')}
 assert d['requests_started']==clients and d['rotations']==0 and (not i or d['prefix_reused_tokens']==clients*25600),(i,d)
 if clients==6:assert d['six_target_passes']>0
 print(json.dumps(dict(wave=i,warmup=i==0,delta=d,**w)),flush=True)
