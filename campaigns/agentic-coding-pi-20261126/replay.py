import fcntl,hashlib,json,threading,time
from urllib.request import Request,urlopen
from sweep import p,b,images,flags,serve,restore,watch
def measure(body):
 req=Request(b.url+'/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+b.bench.key,'Content-Type':'application/json'})
 first=last=usage=finish=None;done=False;events=0;tools=False
 content={'reasoning_content':[],'content':[]}
 start=time.perf_counter()
 with urlopen(req,timeout=180) as resp:
  for line in resp:
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':done=True;continue
   packet=json.loads(data)
   if packet.get('usage'):usage=packet['usage']
   for item in packet.get('choices',[]):
    delta=item.get('delta',{})
    if delta.get('tool_calls'):tools=True
    changed=False
    for k in content:
     if delta.get(k):content[k].append(delta[k]);changed=True
    if changed:
     last=time.perf_counter();events+=1
     if first is None:first=last
    finish=item.get('finish_reason') or finish
 wall=time.perf_counter()-start
 assert done and usage and finish=='length' and usage['completion_tokens']==256 and not tools and events>1 and last>first,(done,finish,events,tools,usage)
 h={k:hashlib.sha256(''.join(v).encode()).hexdigest() for k,v in content.items()}
 return {'hashes':h,'prompt_tokens':usage['prompt_tokens'],'output_tokens':usage['completion_tokens'],'events':events,'wall_s':wall,'ttft_s':first-start,'stream_s':last-first,'decode_tok_s':255/(last-first),'api_tok_s':256/wall}
def main():
 seed=json.loads((p/'frozen-red-turn.json').read_text());assert seed['messages'][-1]['role']=='tool' and seed['chat_template_kwargs']['reasoning_effort']=='max' and seed['temperature']==0
 original_hash=hashlib.sha256(json.dumps(seed,sort_keys=True).encode()).hexdigest()
 assert seed.get('max_tokens')==64000 and '--- FAIL: TestFlagValues' in json.dumps(seed['messages'][-1]['content'])
 assert any(original_hash==t['request_sha256'] for label in ('before','selected','after') for line in (p/(label+'.jsonl')).read_text().splitlines() for t in json.loads(line).get('turns',[]))
 # Preserve nested tool-schema order as sent by Pi. The Rust prompt formatter
 # retains JSON insertion order, which can change the model's token sequence.
 seed['max_tokens']=256;body=json.dumps(seed).encode()
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',flags(1),images[1]);expected=None;cache=None
  try:
   for label,idx in [('slice-before',0),('slice-selected',1),('slice-after',0)]:
    image=serve(label,idx)
    with (p/(label+'.jsonl')).open('x') as out:
     for i in range(7):
      a=b.idle('native');stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
      try:r=measure(body)
      finally:stop.set();t.join()
      z=b.idle('native');b.verify('native',flags(idx),images[idx]);hits=z['prefix_reused_tokens']-a['prefix_reused_tokens']
      assert z['requests_started']-a['requests_started']==1 and z['rotations']==a['rotations']
      if expected is None:expected=r['hashes']
      assert r['hashes']==expected
      if i:
       if cache is None:cache=hits
       assert hits==cache and hits>0
      r.update(i=i,image=image,settings=flags(idx),prefix_reused_tokens=hits,request_sha256=hashlib.sha256(body).hexdigest())
      print(json.dumps(r),file=out,flush=True)
    print(label,'finished',flush=True)
  finally:restore()
if __name__=='__main__':main()
