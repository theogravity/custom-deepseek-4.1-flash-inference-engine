"""Promote the exact qualified image using the checked-in compose files."""
import fcntl,json,subprocess as sp
from pathlib import Path
from run_pair import b,flags,images,deployed,check_mount,restore,p,response
from analyze import summarize
new='ds4-flash-rs:agentic-idle-20261126'
with open('/tmp/ds4-native-bench.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    measured=summarize()
    qualified=json.loads((p/'qualification/passed.json').read_text())
    assert measured['idle_image']==qualified['image'] and qualified['gpu-tests']==92
    assert all(qualified['eight-'+k]['concurrent']==8 for k in ['ram','disk'])
    assert qualified['six-full-waves']==3
    assert Path('compose.agentic.yml').resolve().is_file()
    assert sp.check_output(['git','status','--porcelain'],text=True)==''
    old=b.verify('native',flags(1),images[1])
    b.idle('native');sp.run(['docker','tag',qualified['image'],new],check=True)
    try:
        b.up('agentic-idle-release',deployed+['-f','compose.agentic.yml'])
        enabled=flags(1)|{'DS4_IDLE_TOOL_PREFILL_MAX_TOKENS':'32768'}
        running=b.verify('native',enabled,new)
        assert running==qualified['image']
        check_mount(Path.home()/'.cache/ds4-flash-rs/prefix')
        body=json.loads((p/'capture-b/request-04.json').read_text());body['max_tokens']=8
        smoke,_=response(body)
        assert smoke['tokens']==8 and smoke['finish']=='length' and smoke['prompt']==2780
        metrics=b.idle('native')
        assert metrics['max_seqs']==6 and metrics['max_live_seqs']==32 and metrics['logical_active']==0
        (p/'deployed.json').write_text(json.dumps(dict(previous_image=old,running_image=running,flags=enabled,
            resident_slots=metrics['max_seqs'],logical_limit=metrics['max_live_seqs'],test_prompt=smoke['prompt'],test_output=smoke['tokens'],
            authenticated=True),indent=2)+'\n')
    except BaseException:
        restore();raise
