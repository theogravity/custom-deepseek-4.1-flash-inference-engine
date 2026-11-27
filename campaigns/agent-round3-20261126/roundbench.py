"""Run one change against fresh controls on both sides, restoring the active release."""
import sys,os,json,time,fcntl,threading,hashlib,subprocess as sp,importlib.util
from pathlib import Path
from statistics import median
from datetime import datetime,timezone
from urllib.request import Request,urlopen
p=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/ds4-tool-return-1126')
from run_pair import b,flags,check_mount,watch
from replay import measure
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=module('oldlong','/tmp/ds4-cycle2-1126/run.py')
release='ds4-flash-rs:agentic-idle-20261126'
files=['-f','compose.yml','-f','compose.decode-round2.yml','-f','compose.agentic.yml']
common=flags(1)|{'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':'32768'}
def serve(label,img,over):
    b.idle('native')
    cfg=p/(label+'.compose.json')
    cfg.write_text(json.dumps({'services':{b.svc:{'image':img,'environment':over}}}))
    b.up('r3-'+label,files+['-f',str(cfg)])
    check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
    return b.verify('native',common|over,img)
def restore():
    b.base=files;b.native=release;b.restore();b.verify('native',common,release)
    check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
def post(path,body):
    req=Request(b.url+path,data=body,headers={'Authorization':'Bearer '+b.bench.key,'Content-Type':'application/json'})
    start=time.monotonic()
    with urlopen(req,timeout=120) as f: value=json.load(f)
    return value,time.monotonic()-start
def consume(name,label,folder):
    leg={};a=b.idle('native')
    if name=='text':
        frozen=json.loads(Path(h.old.b.frozen).read_text())['sessions'][0]['messages']
        a_body={'model':'DeepSeek V4.1 Flash','reasoning_effort':'none','messages':frozen}
        counts=[];times=[]
        for n in range(7):
            body=a_body|{'messages':frozen+[{'role':'user','content':f'Apply a fix to the parser and verify revision {n}.'}]}
            value,t=post('/v1/messages/count_tokens',json.dumps(body).encode())
            counts.append(value['input_tokens']);times.append(t)
        assert b.idle('native')['requests_started']==a['requests_started'] and min(counts)>404000
        leg['count_tokens']=dict(input_tokens=counts,seconds=times,median=median(times[1:]))
    if name in ('text','hash','markov','release'):
        seed=json.loads(Path('/tmp/ds4-agentic-live-1126/frozen-red-turn.json').read_text());seed['max_tokens']=256;wire=json.dumps(seed).encode()
        golden=json.loads(Path('/tmp/ds4-agentic-live-1126/slice-selected.jsonl').read_text().splitlines()[1])
        assert hashlib.sha256(wire).hexdigest()==golden['request_sha256']
        rows=[]
        for i in range(7):
            before=b.idle('native');w=measure(wire);after=b.idle('native')
            assert w['hashes']==golden['hashes'] and w['events']==w['output_tokens']==256 and w['prompt_tokens']==2786
            assert after['requests_started']-before['requests_started']==1 and before['rotations']==after['rotations']
            if i:assert after['prefix_reused_tokens']-before['prefix_reused_tokens']==2752
            rows.append(w)
        leg['pi']=dict(median=median(w['decode_tok_s'] for w in rows[1:]),ttft=median(w['ttft_s'] for w in rows[1:]),rows=rows)
    h.q.p=folder
    if name in ('text','hash','markov','release'):
        before=b.idle('native')
        since=datetime.now(timezone.utc).isoformat()
        args=['tools/long_decode.py','--histories',h.old.b.frozen,'--clients','1','--output','512','--rounds','6']
        rows=h.q.shell(label+'-long',args)
        after=b.idle('native')
        assert after['requests_started']-before['requests_started']==7 and before['rotations']==after['rotations']
        assert len(rows)==8 and rows[-1]['prompts']==h.gold['code_request_proofs'][:1]
        assert all([r['sha256'] for r in w['rows']]==h.gold['coding']['qualified-one']['output_sha256'] for w in rows[:-1])
        assert all(w['delta']['prefix_reused_tokens']==404096 for w in rows[1:-1])
        if name=='hash':
            logs=sp.check_output(['docker','logs','--since',since,b.svc],stderr=sp.STDOUT,text=True)
            reports=[json.loads(l) for l in logs.splitlines() if l.startswith('{') and '"kv_restore"' in l]
            reports=[r for r in reports if r['kind']=='prefix' and r['position']==404096]
            assert len(reports)==7 and all(r['tier']=='ram' and r['draft_included'] for r in reports[1:])
            leg['lookup_ms']={k:[r[k] for r in reports[1:]] for k in ['target_lookup_read_ms','draft_lookup_read_ms']}
        ms=rows[1:-1]
        leg['long']=dict(decode=median(w['aggregate_decode_tok_s'] for w in ms),ttft=median(w['rows'][0]['ttft_s'] for w in ms),wall=median(w['wall_tok_s'] for w in ms),samples=[w['rows'][0]['ttft_s'] for w in ms])
    if name=='release':
        start=b.idle('native')
        rows=h.q.shell(label+'-six',['tools/long_decode.py','--histories',h.old.b.frozen,'--clients','6','--output','512','--rounds','6'])
        end=b.idle('native')
        assert end['requests_started']-start['requests_started']==42 and end['rotations']==start['rotations']
        assert len(rows)==8 and rows[-1]['prompts']==h.gold['code_request_proofs']
        assert all([r['sha256'] for r in w['rows']]==h.gold['coding']['qualified-six512']['output_sha256'] for w in rows[:-1])
        assert all(w['delta']['prefix_reused_tokens']==2424704 and w['delta']['six_target_passes']>0 for w in rows[1:-1])
        leg['six']=dict(decode=median(w['aggregate_decode_tok_s'] for w in rows[1:-1]),wall=median(w['wall_tok_s'] for w in rows[1:-1]),samples=[w['aggregate_decode_tok_s'] for w in rows[1:-1]])
    return leg
def run(name):
    folder=p/(name+'-receipts');folder.mkdir(exist_ok=True)
    config=json.loads((p/(name+'.json')).read_text())
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',common,release)
        for key in ('a','on','b','summary'):(folder/(key+'.json')).unlink(missing_ok=True)
        stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
        try:
            for key,(img,env) in zip(('a','on','b'),config,strict=True):
                label=name+'-'+key
                identity=serve(label,img,env);data=consume(name,label,folder);b.verify('native',common|env,img)
                data.update(image=identity,flags=env)
                (folder/(key+'.json')).write_text(json.dumps(data,indent=2)+'\n')
                print(key,[(k,{a:v for a,v in data[k].items() if isinstance(v,(int,float))}) for k in ('count_tokens','pi','long') if k in data],flush=True)
        finally:stop.set();t.join();restore()
if __name__=='__main__':run(sys.argv[1])
