from roundbench import *
from qualification import coding_workloads
out=p/'mixed-receipts';out.mkdir(exist_ok=True)
with open('/tmp/ds4-native-bench.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert (p/'qualification/passed.json').is_file()
    b.verify('native',common,release)
    (out/'summary.json').unlink(missing_ok=True)
    docs={}
    stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
    try:
        for tag,(image,env) in zip(['a','on','b'],json.loads((p/'release.json').read_text()),strict=True):
            sub=out/tag;sub.mkdir(exist_ok=True)
            h.q.p=sub
            w={'image':serve('mixed-'+tag,image,env)}
            def record(k,v):
                w[k]=v
                (sub/'results.json').write_text(json.dumps(w,indent=2)+'\n')
            coding_workloads(record);b.verify('native',common|env,image)
            docs[tag]=w
            print(tag,w['short-edit']['median'],w['long-c1']['median'],flush=True)
        results={k:[docs[tag][k]['median'] for tag in ['a','on','b']] for k in ['short-edit','long-c1']}
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        for k,(a,on,bval) in results.items():
            assert on>=0.97*max(a,bval),(k,a,on,bval)
    finally:stop.set();t.join();restore()
