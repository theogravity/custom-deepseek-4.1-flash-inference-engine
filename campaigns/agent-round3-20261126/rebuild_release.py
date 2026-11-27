from pathlib import Path
import os,json,subprocess as sp,shutil,fcntl
from roundbench import p,b,common,release
with open('/tmp/ds4-native-bench.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',common,release)
    bad=p/'reused-build';bad.mkdir()
    for name in ['release-receipts','qualification','mixed-receipts','release-provenance.json','qualified-tests.json','release-summary.log','release-bench.log','qualification.log','mixed.log','release-build.log','release-image.log']:
        f=p/name
        if f.exists():f.rename(bad/name)
    choice=json.loads((p/'chosen.json').read_text())
    orig=p/choice['worktree'];commit=choice['commit']
    tree=p/'clean-src';target=p/'release-target';assert not target.exists() and not tree.exists()
    sp.run(['git','worktree','add','--detach',tree,commit],cwd=orig,check=True)
    choice['worktree']='clean-src'
    (p/'chosen.json').write_text(json.dumps(choice,indent=2)+'\n')
    env=os.environ|{'CARGO_TARGET_DIR':str(target)}
    with (p/'release-build.log').open('w') as f:
        sp.run(['cargo','build','--release','--locked','--offline'],cwd=tree,env=env,stdout=f,stderr=sp.STDOUT,check=True)
    with (p/'qualified-tests.json').open('w') as f:
        sp.run(['cargo','test','--release','--locked','--offline','--no-run','--message-format=json'],cwd=tree,env=env,stdout=f,check=True)
    binary=target/'release/ds4-flash-rs'
    context=p/'release-image';(context/'target/release').mkdir(parents=True)
    shutil.copy2(binary,context/'target/release/ds4-flash-rs');shutil.copy2(tree/'Dockerfile',context/'Dockerfile')
    with (p/'release-image.log').open('w') as f:
        sp.run(['docker','build','-t',choice['image'],str(context)],stdout=f,stderr=sp.STDOUT,check=True)
