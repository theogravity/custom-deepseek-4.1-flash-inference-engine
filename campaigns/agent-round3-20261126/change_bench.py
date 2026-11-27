from roundbench import *
from changing import bodies,sha
from run_pair import response
def main():
    folder=p/'changing-receipts';folder.mkdir(exist_ok=True)
    cfg=['ds4-flash-rs:agent3-text',{'DS4_AGENT_TEXT_CACHE':'0'}]
    ref=json.loads((p/'changing-reference.json').read_text());payloads=bodies()
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',common,release)
        for k in ('a','on','b','summary'):(folder/(k+'.json')).unlink(missing_ok=True)
        stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
        try:
            for mode in ['a','on','b']:
                env={'DS4_AGENT_TEXT_CACHE':'1' if mode=='on' else '0'}
                image=serve('changed-'+mode,cfg[0],env)
                counts=[];generations=[]
                for q,r in zip(payloads,ref,strict=True):
                    assert sha(q)==r['wire']
                    a=b.idle('native');result,dt=post('/v1/messages/count_tokens',json.dumps(q).encode())
                    assert result['input_tokens']==r['tokens'] and b.idle('native')['requests_started']==a['requests_started']
                    counts.append(dt)
                    row,_=response(q)
                    assert row['wire_sha']==r['wire'] and row['prompt']==r['tokens'] and row['tokens']==16 and row['finish']=='length'
                    generations.append({k:row[k] for k in ['prompt','tokens','wire_sha','ordered_output_sha','ttft','reuse','finish']})
                b.verify('native',common|env,cfg[0])
                receipt=dict(image=image,settings=env,count_s=counts,generations=generations)
                (folder/(mode+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                print(mode,'count',median(counts[1:]),'TTFT',median(r['ttft'] for r in generations[1:]),flush=True)
            alllegs=[json.loads((folder/(k+'.json')).read_text()) for k in ['a','on','b']]
            assert len({l['image'] for l in alllegs})==1
            for i in range(7):
                for k in ['ordered_output_sha','prompt','tokens','wire_sha','finish']:
                    assert len({l['generations'][i][k] for l in alllegs})==1
                if i:assert len({l['generations'][i]['reuse'] for l in alllegs})==1 and all(l['generations'][i]['reuse']>=403900 for l in alllegs)
            (folder/'summary.json').write_text(json.dumps({'exact_responses':21,'count_s':[median(l['count_s'][1:]) for l in alllegs],'ttft_s':[median(r['ttft'] for r in l['generations'][1:]) for l in alllegs],'reference':ref},indent=2)+'\n')
        finally:stop.set();t.join();restore()
if __name__=='__main__':main()
