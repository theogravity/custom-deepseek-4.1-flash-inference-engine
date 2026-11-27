set -eu
p=/tmp/ds4-agent-round3
cd "$p"
test -s markov-gpu-test.log
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('markov-receipts/summary.json').read_text())
assert min(r['pi_median']['gain_vs_controls_pct'])<3
PY
git -C markov add src/runtime.rs native/sample.cu
git -C markov diff --cached --check
git -C markov commit -m 'Evaluate two-score interleaving in the exact DSpark Markov head'
git -C markov tag -a checkpoint-2026-11-26-agent3-02-markov -m 'Paired Markov scores, Pi and long-coding decode benchmark'
git -C /home/benchmark/llm/ds4-flash-rs worktree add -b experiment/agent-prefix-keys "$p/hash" experiment/agent-text-cache
python3 make_hash.py
env -u SHELLOPTS bash build.sh hash
python3 - <<'PY'
import json
from pathlib import Path
a=['ds4-flash-rs:agent3-hash',{'DS4_AGENT_TEXT_CACHE':'1','DS4_BULK_PREFIX_KEYS':'0','DS4_TIMING_RESTORES':'1'}]
c=[a[0],dict(a[1],DS4_BULK_PREFIX_KEYS='1')]
Path('hash.json').write_text(json.dumps([a,c,a]))
PY
python3 -u roundbench.py hash > hash-bench.log 2>&1
python3 verify.py hash > hash-summary.log
