"""Temporary workstation runner for controlled native engine trials."""
import sys,os,json,time,subprocess as sp,re,fcntl,statistics,hashlib,traceback
from pathlib import Path
from urllib.request import Request,urlopen
root=Path('/home/benchmark/llm/ds4-flash-rs')
p=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/ds4-edit-gap')
os.chdir(root)
import run as b
hist=Path('/tmp/ds4-six.pFsc06/404k-histories.json')
assert hashlib.sha256(hist.read_bytes()).hexdigest()=='1bde155d1d810e589a9cc5a7c298d6438693b7fa5416ffa1df6bef937d960990'
restore_image=b.native
lfu={'DS4_EXPERT_EVICTION':'lfu'}
toolsroot=Path(os.environ.get('TRIAL_TOOLS',str(p/'work/tools')))
def leg(folder,label,img,env,clients,budget,rounds,kind,variation):
    for ext in ['.failed','.info']:
        (folder/(label+ext)).unlink(missing_ok=True)
    ident=b.start(label,'native',img,env);b.idle('native')
    cmd=b.bench.runner+[str(toolsroot/'long_decode.py'),'--histories',str(hist),'--clients',str(clients),'--output',str(budget),'--rounds',str(rounds),'--workload',kind,'--sample-steady']
    if variation:cmd.append('--allow-output-variation')
    guard=[];b.stamp(f'{label} serving')
    with (folder/(label+'.jsonl')).open('w') as out,(folder/(label+'.err')).open('w') as err:
        pr=sp.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,DSV41_API_KEY=b.bench.key))
        try:
            until=time.monotonic()+700
            while pr.poll() is None:
                gpu=[int(x) for x in b.read('nvidia-smi','-i','1,2','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()]
                ram=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])//1024
                guard.append([*gpu,ram])
                if ram<8192 or (len(guard)>=5 and all(min(x[:2])<128 for x in guard[-5:])):raise RuntimeError('memory watchdog')
                if time.monotonic()>until:raise TimeoutError('wave suite deadline')
                try:pr.wait(timeout=.8)
                except sp.TimeoutExpired:pass
            if pr.returncode:raise RuntimeError((folder/(label+'.err')).read_text()[-1400:])
        finally:
            if pr.poll() is None:pr.terminate();pr.wait(timeout=20)
            (folder/(label+'.resource.json')).write_text(json.dumps(guard))
    b.verify('native',lfu|env,img)
    raw=[json.loads(s) for s in (folder/(label+'.jsonl')).read_text().splitlines()]
    assert len(raw)==rounds+2
    waves=raw[1:-1]
    if kind=='code' and clients==6 and budget in [512,768]:
        ref=json.loads((root/'results/long-context-tuning-375w.json').read_text())
        gold=ref['coding']['qualified-six512' if budget==512 else 'ram-six']['output_sha256']
        assert raw[-1]['prompts']==ref['code_request_proofs']
        matches=[sum(r['sha256']==h for r,h in zip(w['rows'],gold,strict=True)) for w in waves]
        if not variation:assert matches==[clients]*rounds,(label,matches)
    else:matches=None
    c=[w['delta'].get('cpu_expert_jobs',0) for w in waves]
    rc=[w['delta'].get('cpu_expert_rows',0) for w in waves]
    assert all(j<=r<=2*j for j,r in zip(c,rc))
    if env.get('DS4_CPU_EXPERTS')=='1' and kind=='code':assert sum(c)>0,'ON did not execute any warm CPU routes'
    else:
        if env.get('DS4_CPU_EXPERTS')!='1':assert sum(c)==0

    info=dict(label=label,image=ident,env=env,clients=clients,output=budget,workload=kind,median_tok_s=statistics.median(w['aggregate_decode_tok_s'] for w in waves),whole_wave=statistics.median(w['wall_tok_s'] for w in waves),per_wave=[w['aggregate_decode_tok_s'] for w in waves],matches=matches,per_wave_cpu_jobs=c,per_wave_cpu_rows=[w['delta'].get('cpu_expert_rows',0) for w in waves],prefix_reuse=[w['delta']['prefix_reused_tokens'] for w in waves],free_mib_min=[min(z[i] for z in guard) for i in range(3)],raw_sha256=hashlib.sha256((folder/(label+'.jsonl')).read_bytes()).hexdigest())
    (folder/(label+'.info')).write_text(json.dumps(info,indent=2));b.stamp(f'{label} {info["median_tok_s"]:.1f} tok/s, jobs {c}, matched {matches}')
def main():
    config=Path(sys.argv[1]);folder=config.parent
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',lfu,restore_image)
        rows=json.loads(config.read_text())
        for row in rows:
            for ext in ['.failed','.info','.jsonl','.resource.json','.err']:
                (folder/(row['label']+ext)).unlink(missing_ok=True)
        failures=[]
        try:
            for row in rows:
                try:leg(folder,**row)
                except Exception as exc:
                    b.stamp(row['label']+' FAILED '+str(exc));(folder/(row['label']+'.failed')).write_text(traceback.format_exc())
                    failures.append(row['label'])
                    if row['label'].endswith('off-a'):raise
        finally:
            b.restore();b.stamp('qualified shared-prefix service restored')
        if failures:raise RuntimeError('failed legs '+str(failures))
if __name__=='__main__':main()
