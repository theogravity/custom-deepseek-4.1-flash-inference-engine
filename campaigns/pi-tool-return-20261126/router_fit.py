"""Validate a promising isolated immediate-return result with full Pi producer and consumer."""
from run_pair import *
from statistics import median
if __name__=='__main__':
    decision=json.loads((p/'router-decision.json').read_text())
    if not decision['repeat_on_full_exchange']:raise SystemExit(0)
    with open('/tmp/ds4-native-bench.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b.verify('native',flags(1),images[1])
        shutdown=threading.Event();watcher=threading.Thread(target=watch,args=(shutdown,),daemon=True);watcher.start()
        try:
            results={}
            for label,bound in [('a',False),('on',True),('b',False)]:
                ttft=[]
                for n,delay in enumerate([0.,0.,6.]):
                    lab=f'confirm-router-{label}-{n}'
                    passed=run(lab,True,idle=True,delay=delay,bound=bound)
                    rows=json.loads((p/(lab+'.json')).read_text())
                    assert passed and rows[1]['reuse']==(2752 if delay==0. else 6848)
                    assert rows[1]['restore_tier']=='ram' and all(r['verified'] for r in rows)
                    assert delay<=rows[1]['gap_since_previous_s']<=delay+.18
                    ttft.append(rows[1]['ttft'])
                results[label]=dict(immediate=ttft[:2],six=ttft[2],median=median(ttft[:2]))
            gains=[100*(1-results['on']['median']/results[c]['median']) for c in ['a','b']]
            decision.update(confirmed=results,confirmed_improvement_pct=gains,promote_router=min(gains)>=3)
            (p/'router-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        finally:shutdown.set();watcher.join();restore()
