from pathlib import Path
import hashlib,json,subprocess as sp
p=Path(__file__).resolve().parent
def verify_source(worktree):
    choice=json.loads((p/'chosen.json').read_text())
    tree=p/choice['worktree']
    assert Path(worktree).resolve()==tree.resolve()
    assert sp.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==choice['commit']
    assert not sp.check_output(['git','status','--porcelain'],cwd=tree,text=True)
    return choice,tree
def attest():
    choice,tree=verify_source(p/json.loads((p/'chosen.json').read_text())['worktree'])
    meta=[json.loads(l) for l in (p/'qualified-tests.json').read_text().splitlines()]
    tests=[r for r in meta if r.get('executable') and r.get('profile',{}).get('test')]
    assert len(tests)==1 and Path(tests[0]['manifest_path']).resolve()==(tree/'Cargo.toml').resolve()
    target=p/'release-target'
    binary=target/'release/ds4-flash-rs'
    test=Path(tests[0]['executable'])
    assert test.resolve().is_relative_to(target.resolve())
    names=sp.check_output([test,'--list'],text=True)
    assert '95 tests, 0 benchmarks' in names and 'bulk_keys_preserve_existing' not in names and 'concurrent_inserts_share' in names
    packed=binary.read_bytes()
    assert b'DS4_AGENT_TEXT_CACHE' in packed and b'DS4_BULK_PREFIX_KEYS' not in packed and b'DS4_MARKOV_PAIRS' not in packed
    digest=lambda f:hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
    exe=digest(binary)
    in_image=sp.check_output(['docker','run','--rm','--entrypoint','/usr/bin/sha256sum',choice['image'],'/usr/local/bin/ds4-flash-rs'],text=True).split()[0]
    assert exe==in_image
    ident=sp.check_output(['docker','image','inspect','-f','{{.Id}}',choice['image']],text=True).strip()
    r={'commit':choice['commit'],'image':ident,'binary_sha256':exe,'test_binary_sha256':digest(Path(tests[0]['executable']))}
    receipt=p/'release-provenance.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())==r,'release provenance differs from the clean build'
    else:
        with receipt.open('x') as target:target.write(json.dumps(r,indent=2)+'\n')
    return r
if __name__=='__main__':
    import sys
    if len(sys.argv)==2:verify_source(sys.argv[1])
    else:attest()
