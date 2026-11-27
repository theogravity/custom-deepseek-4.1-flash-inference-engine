import fcntl,json,threading,sys
from sweep import b,images,serve,restore,watch,flags
from trial import one,p
def main():
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  b.verify('native',flags(1),images[1])
  assert len((p/'before.jsonl').read_text().splitlines())==3
  try:
   for label,idx in [('selected',1),('after',0)]:
    image=serve(label,idx)
    with (p/(label+'.jsonl')).open('x') as out:
     for sample in range(3):
      a=b.idle('native');stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
      try:r=one(b.bench.key,sample)
      finally:stop.set();t.join()
      z=b.idle('native');b.verify('native',flags(idx),images[idx])
      r.update(image=image,leg=label,warmup=sample==0,settings=flags(idx),engine={k:z[k]-a[k] for k in ('requests_started','prefix_reused_tokens','rotations')})
      if r['engine']['requests_started']!=len(r.get('turns',[])):r['excluded']='unrelated engine requests'
      print(json.dumps(r),file=out,flush=True)
      print(label,sample,r['passed'],r.get('workflow'),r['wall_s'],len(r.get('turns',[])),r.get('error',''),flush=True)
  finally:restore()
if __name__=='__main__':main()
