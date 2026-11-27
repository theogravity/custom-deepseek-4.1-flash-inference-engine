import math
import fcntl,json,os,subprocess as sp
from pathlib import Path
from roundbench import p,b,common,release,files,restore,check_mount
from run_pair import response
from verify import analyze
def main():
    chosen=json.loads((p/'chosen.json').read_text())
    qualified=json.loads((p/'qualification/passed.json').read_text())
    validation=analyze('release')
    root=Path('/home/benchmark/llm/ds4-flash-rs');os.chdir(root)
    assert qualified['image']==b.ident(chosen['image']) and qualified['gpu-tests']==95
    mix=json.loads((p/'mixed-receipts/summary.json').read_text())
    assert set(mix)=={'short-edit','long-c1'}
    assert all(isinstance(v,list) and len(v)==3 and all(type(n) in (int,float) and math.isfinite(n) and n>0 for n in v) for v in mix.values())
    assert all(v[1]>=0.97*max(v[0],v[2]) for v in mix.values())
    assert qualified['six-full-waves']==3 and len(qualified['pi-tool-exchange'])==3
    assert all(qualified['eight-'+tier]['total_including_controls']==16 for tier in ['ram','disk'])
    assert not sp.check_output(['git','status','--porcelain'],text=True)
    assert b.read('git','rev-parse','HEAD')==b.read('git','rev-parse','checkpoint-2026-11-26-agent3-release^{}')
    with open('/tmp/ds4-native-bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',common,release);b.idle('native')
        try:
            b.up('release-agent-round3',files[:-2]+['-f','compose.coding.yml'])
            env=common|chosen['flags']
            image=b.verify('native',env,chosen['image'])
            assert image==qualified['image']
            check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
            q=json.loads(Path('/tmp/ds4-tool-return-1126/capture-b/request-04.json').read_text());q['max_tokens']=8
            row,_=response(q)
            a=b.idle('native')
            assert row['prompt']==2780 and row['tokens']==8 and row['finish']=='length'
            assert a['max_seqs']==6 and a['max_live_seqs']==32 and a['logical_active']==0
            with (p/'deployed.json').open('w') as f:json.dump(dict(image=image,source=chosen['commit'],release=b.read('git','rev-parse','HEAD'),max_seqs=6,max_live_seqs=32,smoke=dict(prompt=row['prompt'],output=row['tokens']),flags=env),f,indent=2)
        except BaseException:
            restore()
            raise
if __name__=='__main__':main()
