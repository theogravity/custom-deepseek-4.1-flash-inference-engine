"""Check the native observations against the three messages Pi subsequently sent."""
import hashlib, json
from pathlib import Path
from statistics import median
from post_canonical import analyze as canonical
p=Path(__file__).resolve().parent

def load(f):
    return json.loads(f.read_text())

def digest(v):
    return hashlib.sha256(json.dumps(v).encode()).hexdigest()

def summarize():
    task=load(p/'capture-b/task.json')
    assert task['graded_correct'] and task['isolated'] and task['complete_streams']
    fixture={}
    for n in (4,5,6):
        body=load(p/f'capture-b/request-{n:02}.json')
        next_body=load(p/f'capture-b/request-{n+1:02}.json')
        a=next(m for m in reversed(next_body['messages']) if m['role']=='assistant')
        out=(a.get('reasoning_content') or '',a.get('content') or '',
             [(c['function']['name'],json.loads(c['function']['arguments'])) for c in a['tool_calls']])
        turn=task['turns'][n-1]
        fixture[n]=dict(wire_sha=digest(body),ordered_output_sha=digest(out),tokens=turn['usage']['completion_tokens'],
                        prompt=turn['usage']['prompt_tokens'],finish=turn['finish'])
    by_arm={}
    for arm in ['a','on','b']:
        groups={}
        for i,gap in enumerate([0,2,6,0,2,6]):
            rows=load(p/f'q-{arm}-{i}.json')
            assert [r['turn'] for r in rows]==[4,5,6]
            for row in rows:
                assert row['verified'] and row['mode']==(arm=='on') and all(row[k]==v for k,v in fixture[row['turn']].items())
            prod,r,after=rows
            assert r['restore_tier']=='ram' and prod['reuse']==2752 and after['reuse']==6848
            assert gap<=r['gap_since_previous_s']<=gap+.18
            if gap==0 or arm!='on':
                assert r['reuse']==2752 and r['idle_tokens_before']==0
            else:
                assert r['reuse']>2752 and r['idle_tokens_before']>0
            samples=groups.setdefault(str(gap),[])
            samples.append({k:r[k] for k in ['ttft','wall','reuse','gap_since_previous_s','idle_tokens_before']})
        by_arm[arm]={k:{'samples':v,'median_ttft':median(r['ttft'] for r in v)} for k,v in groups.items()}
    pilot=load(p/'pilot-gate.json')
    assert not pilot['eligible_for_deployment']
    comparison={}
    for label,rows in pilot['legs'].items():
        comparison[label]=[{k:r[k] for k in ['turn','tokens','prompt','reuse','ttft','matches_recorded','exact_recorded_continuation']} for r in rows]
    isolated=canonical('width-off-a width-on width-off-b gather-off-a gather-on gather-off-b'.split())
    if (p/'bound-off-b.json').exists():
        isolated|=canonical('bound-off-a bound-on bound-off-b'.split())
    assert all(r['matches']==7 for r in isolated.values())
    wider=load(p/'real-wide.json')
    assert [r['verified'] for r in wider]==[False,True,False]
    result={'power_caps_w':{'gpu_1':375,'gpu_2':375,'cpu':225},'original_captured_turns':fixture,
        'original_live_return_ttft_s':task['turns'][4]['first_event_s'],
        'coding_fixture':{'source':'Pi Go flag parser repair','effort':'high','completion_limit':64000,
                'graded_correct':task['graded_correct'],'patch_sha256':task['patch_sha256']},
        'idle_ab_a':by_arm,'output_parity':{'replays':18,'checked_responses':54},
        'idle_image':rows_image(),'idle_source_commit':'a796721','transient_kv_checkpoint':'df76fad',
        'generated_kv_rejected':comparison,'canonical_only':isolated,
        'wide_chunk_replay':[{k:r[k] for k in ['turn','tokens','verified','matches_recorded','ttft']} for r in wider]}
    return result

def rows_image():
    images={r['image'] for f in p.glob('q-*.json') if 'override' not in f.name for r in load(f)}
    assert len(images)==1
    return images.pop()

if __name__=='__main__':
    out=summarize()
    if (p/'qualification/passed.json').exists():
        qualification=load(p/'qualification/passed.json')
        assert qualification['image']==out['idle_image']
        out['qualification']=qualification
    print(json.dumps(out,indent=2))
