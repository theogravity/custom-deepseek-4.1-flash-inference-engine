"""Full-window, tier-rotation and C1 qualification on the candidate image."""
import fcntl, importlib.util, json, os, subprocess, sys
from pathlib import Path
from statistics import median
from run_pair import b, flags, images, deployed, check_mount, restore, p
spec = importlib.util.spec_from_file_location('historical', '/tmp/ds4-cycle2-1126/run.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
q = h.q
folder = p/'qualification'
folder.mkdir(exist_ok=True)
q.p = folder
image = 'ds4-flash-rs:canonical-idle-final-1126'
enabled = {'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':'32768'}
reference = h.gold
root = h.root

def start(tier):
    options = enabled | ({'DS4_SWAP_RAM_MIB':'0'} if tier == 'disk' else {})
    f = folder/(tier+'-override.json')
    f.write_text(json.dumps({'services':{b.svc:{'image':image,'environment':options}}}))
    b.idle('native')
    b.up('idle-qualification-'+tier, deployed+['-f',str(f)])
    check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
    return b.verify('native',flags(1)|options,image)

def run():
    result = {'image':start('default')}
    def record(key, val):
        result[key] = val
        (folder/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        print(key, val,flush=True)
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
    subprocess.run(b.comp+deployed+['stop','--timeout','-1'],check=True)
    ids = b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
    env=os.environ|flags(1)|enabled|{'CUDA_VISIBLE_DEVICES':','.join(ids),'DS4_TEST_DEVICE':'0','DS4_TEST_CUDA_DEVICE':'0','DS4_TEST_PACK':'/mnt/models/dsv41-flash-exl3'}
    binary=subprocess.check_output(['python3','/tmp/ds4-cpu-eval/testbin.py',p/'tests-meta.json'],text=True).strip()
    with (folder/'gpu-tests.log').open('w') as f:
        subprocess.run([binary,'--include-ignored','--test-threads=1'],cwd=p/'idle',env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
    assert '92 passed; 0 failed; 0 ignored' in (folder/'gpu-tests.log').read_text()
    record('gpu-tests',92)
    result['memory']={k:json.loads((folder/(k+'-memory.json')).read_text()) for k in ['context-needles','short-edit','long-c1','six-full','eight-ram','eight-disk']}
    (folder/'passed.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert all(len(json.loads((p/f'q-{arm}-{i}.json').read_text()))==3 for arm in ('a','on','b') for i in range(6))
        b.verify('native',flags(1),images[1])
        for name in ('passed.json','progress.json'):
            (folder/name).unlink(missing_ok=True)
        try: run()
        finally: restore()
