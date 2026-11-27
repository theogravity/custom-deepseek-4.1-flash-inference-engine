"""A single prespecified Pi effort change on the selected profile, after A/B/A."""
import fcntl,json,threading
from summarize import records,wire_settings
from sweep import b,images,serve,restore,watch,flags
from trial import one,p

def main():
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert len(records('slice-after'))==7
  b.verify('native',flags(1),images[1])
  try:
   label='selected-high'
   image=serve(label,1)
   with (p/(label+'.jsonl')).open('x') as out:
    for sample in range(3):
     a=b.idle('native');stop=threading.Event()
     t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
     try:r=one(b.bench.key,sample,thinking_override='high')
     finally:stop.set();t.join()
     z=b.idle('native');b.verify('native',flags(1),images[1])
     r.update(image=image,leg=label,warmup=sample==0,settings=flags(1),engine={k:z[k]-a[k] for k in ('requests_started','prefix_reused_tokens','rotations')})
     if r['engine']['requests_started']!=len(r.get('turns',[])):r['excluded']='unrelated engine requests'
     print(json.dumps(r),file=out,flush=True)
     print(label,sample,r['passed'],r.get('graded_correct'),r['wall_s'],r.get('client_finish'),r.get('error'),flush=True)
     wire_settings([r],effort='high')
  finally:restore()
if __name__=='__main__':main()
