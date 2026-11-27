from run_pair import *
def main():
 with open('/tmp/ds4-native-bench.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',flags(1),images[1])
  stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,),daemon=True);watcher.start()
  try:
   # Each observation replays the full tool-producing response on a fresh process.
   # All three arms see the same six tool durations in the same order.
   gaps=(0,2,6,0,2,6)
   for arm,enabled in [('a',False),('on',True),('b',False)]:
    for i,gap in enumerate(gaps):
     lab=f'q-{arm}-{i}'
     if not run(lab,enabled,idle=True,delay=gap):raise RuntimeError('full recorded agent parity failed '+lab)
     rows=json.loads((p/(lab+'.json')).read_text());producer,consumer,following=rows
     assert producer['reuse']==2752 and following['reuse']==6848
     assert gap<=consumer['gap_since_previous_s']<=gap+0.18
     actual=consumer['reuse']
     assert consumer['restore_tier']=='ram'
     if not enabled or gap==0:assert actual==2752 and consumer['idle_tokens_before']==0
     else:assert actual>2752 and consumer['idle_tokens_before']>0
  finally:stop.set();watcher.join();restore()
if __name__=='__main__':main()
