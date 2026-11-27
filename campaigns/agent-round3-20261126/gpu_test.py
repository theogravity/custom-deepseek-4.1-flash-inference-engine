"""Serialize the dedicated Markov test with a drained service."""
from roundbench import *
with open('/tmp/ds4-native-bench.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',common,release);b.idle('native')
    meta=sp.check_output(['cargo','test','--release','--locked','--offline','--no-run','--message-format=json'],cwd=p/'markov',env=os.environ|{'CARGO_TARGET_DIR':'/tmp/ds4-cpu-eval/target'},text=True)
    binary=next(json.loads(l)['executable'] for l in meta.splitlines() if json.loads(l).get('executable') and json.loads(l).get('profile',{}).get('test'))
    ids=b.read('nvidia-smi','-i','1,2','--query-gpu=uuid','--format=csv,noheader').splitlines()
    try:
        sp.run(b.comp+files+['stop','--timeout','-1'],check=True)
        out=sp.check_output([binary,'runtime::tests::markov_chain_and_greedy_reductions','--ignored','--test-threads=1'],env=os.environ|{'CUDA_VISIBLE_DEVICES':','.join(ids),'DS4_TEST_DEVICE':'0'},text=True)
        assert '1 passed; 0 failed' in out
        (p/'markov-gpu-test.log').write_text(out)
    finally:restore()
