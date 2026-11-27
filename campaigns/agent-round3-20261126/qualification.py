"""Full-window, tier-rotation and C1 qualification on the candidate image."""
import fcntl, importlib.util, json, os, subprocess, sys
from pathlib import Path
from statistics import median
from roundbench import p,b,h,common,release,files,serve,restore
q=h.q
folder=p/'qualification'
folder.mkdir(exist_ok=True)
q.p=folder
chosen=json.loads((p/'chosen.json').read_text())
image=chosen['image']
enabled=chosen['flags']
reference=h.gold
root=h.root

def start(tier):
    options=enabled|({'DS4_SWAP_RAM_MIB':'0'} if tier=='disk' else {})
    return serve('final-qualification-'+tier,image,options)

def coding_workloads(record):
    from run_pair import response,compare
    import time
    fixture=Path('/tmp/ds4-tool-return-1126/capture-b')
    original=json.loads((fixture/'task.json').read_text())
    assert original['graded_correct'] and original['isolated']
    golden=json.loads((root/'results/pi-tool-return.json').read_text())['original_captured_turns']
    recorded=[]
    for n in (4,5,6):
        if n==5:time.sleep(6)
        qbody=json.loads((fixture/f'request-{n:02}.json').read_text())
        row,output=response(qbody)
        future=json.loads((fixture/f'request-{n+1:02}.json').read_text())
        ref=original['turns'][n-1]
        assert row['input_sha']==ref['request_sha256'] and all(row[k]==golden[str(n)][k] for k in ('wire_sha','ordered_output_sha','prompt','tokens','finish'))
        assert all(compare(output,future)) and row['tokens']==ref['usage']['completion_tokens'] and row['prompt']==ref['usage']['prompt_tokens'] and row['finish']==ref['finish']
        recorded.append({k:row[k] for k in ['wire_sha','ordered_output_sha','prompt','tokens','finish']})
    record('pi-tool-exchange',recorded)
    expected = [r for l in (root/'results/hybrid-context.jsonl').read_text().splitlines() if 'context' in (r:=json.loads(l))]
    before=b.idle('native')
    rows = q.shell('context-needles',['tools/context_gate.py','--seed',str(expected[0]['seed'])])
    after=b.idle('native')
    assert after['requests_started']-before['requests_started']==2 and after['rotations']==before['rotations']
    receipts = [w for w in rows if 'context' in w]
    assert len(receipts)==len(expected)==2
    for a,z in zip(receipts,expected,strict=True):
        assert a['exact'] and all(a[k]==z[k] for k in ['context','expected','received','prompt_tokens'])
    record('needles',[r['prompt_tokens'] for r in receipts])
    for tag,cmd in [('short-edit',['/tmp/ds4-cycle2-1126/edit.py','1','6']),('long-c1',['tools/long_decode.py','--histories',h.old.b.frozen,'--clients','1','--output','512','--rounds','6'])]:
        a = b.idle('native')
        rows = q.shell(tag,cmd)
        z = b.idle('native')
        assert z['requests_started']-a['requests_started']==7 and z['rotations']==a['rotations']
        if tag=='short-edit':
            keys = ['prompt_sha256','reasoning_sha256','content_sha256']
            ref=json.loads(Path('/tmp/ds4-edit-gap/rust-a.jsonl').read_text().splitlines()[1])['rows'][0]
            assert len(rows)==7 and all(all(w['rows'][0][k]==ref[k] for k in keys) for w in rows)
            assert all(w['delta']['prefix_reused_tokens']==25600 for w in rows[1:])
            rates=[w['upstream_tok_s'] for w in rows[1:]]
        else:
            assert len(rows)==8 and rows[-1]['prompts']==reference['code_request_proofs'][:1]
            assert all([r['sha256'] for r in w['rows']]==reference['coding']['qualified-one']['output_sha256'] for w in rows[:-1])
            assert all(w['delta']['prefix_reused_tokens']==404096 for w in rows[1:-1])
            rates=[w['aggregate_decode_tok_s'] for w in rows[1:-1]]
        record(tag,{'samples':rates,'median':median(rates)})

def run():
    from attest import attest
    proof=attest()
    result = {'image':start('default')}
    assert result['image']==proof['image']
    def record(key, val):
        result[key] = val
        (folder/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        print(key, val,flush=True)
    coding_workloads(record)
    rows=q.shell('six-full',['/tmp/ds4-eviction-1126/six_streams_idle.py','--input','524160','--output','128','--continue-histories',h.old.b.frozen,'--rounds','3'])
    gold=reference['c6_full_retrieval']['ram-six']
    assert len(rows)==3
    for i,w in enumerate(rows):
        a,z=w['metrics_before'],w['metrics_after']
        assert w['all_valid'] and w['concurrent'] and w['six_target_passes_delta']>0 and a['rotations']==z['rotations']
        assert [r['sha256'] for r in w['rows']]==gold['output_sha256'] and [r['extended_ids_sha256'] for r in w['rows']]==gold['prompt_ids_sha256']
        if i: assert z['prefix_reused_tokens']-a['prefix_reused_tokens']==6*524096
    record('six-full-waves',3)
    for tier in ['ram','disk']:
        assert start(tier)==result['image']
        before=b.idle('native')
        w,=q.shell('eight-'+tier,['tools/kv_rotation.py','--agents','8','--input','523776','--output','512','--shared','--expect-tier',tier])
        after=b.idle('native')
        assert after['requests_started']-before['requests_started']==16
        admitted=w['metrics_after']['requests_started']-w['metrics_before']['requests_started']
        assert admitted==8 and w['passed'] and w['peak_resident']==6 and w['peak_swapped']>=2
        io=w['swap_delta']
        other='disk' if tier=='ram' else 'ram'
        assert io[tier+'_reads']>0 and io[tier+'_writes']>0 and io[other+'_reads']==io[other+'_writes']==0
        assert io['read_failures']==io['write_failures']==0
        assert [r['sha256'] for r in w['rows']]==[r['sha256'] for r in reference['eight_full_window']['rows']]
        record('eight-'+tier,{k:w[k] for k in ['peak_resident','peak_swapped','passed']}|dict(concurrent=admitted,total_including_controls=16,io=io))
    b.idle('native')
    subprocess.run(b.comp+files+['stop','--timeout','-1'],check=True)
    ids = b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
    env=os.environ|common|enabled|{'CUDA_VISIBLE_DEVICES':','.join(ids),'DS4_TEST_DEVICE':'0','DS4_TEST_CUDA_DEVICE':'0','DS4_TEST_PACK':'/mnt/models/dsv41-flash-exl3'}
    binary=subprocess.check_output(['python3','/tmp/ds4-cpu-eval/testbin.py',p/'qualified-tests.json'],text=True).strip()
    import hashlib
    with open(binary,'rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==proof['test_binary_sha256']
    with (folder/'gpu-tests.log').open('w') as f:
        subprocess.run([binary,'--include-ignored','--test-threads=1'],cwd=p/chosen['worktree'],env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
    import re
    stats=re.search(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored', (folder/'gpu-tests.log').read_text())
    assert stats and int(stats[1])==95
    record('gpu-tests',int(stats[1]))
    result['memory']={k:json.loads((folder/(k+'-memory.json')).read_text()) for k in ['context-needles','short-edit','long-c1','six-full','eight-ram','eight-disk']}
    (folder/'passed.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        b.verify('native',common,release)
        for name in ('passed.json','progress.json'):
            (folder/name).unlink(missing_ok=True)
        try: run()
        finally: restore()
