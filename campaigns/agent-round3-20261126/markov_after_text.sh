set -eu
p=/tmp/ds4-agent-round3
cd "$p"
# Launch the second experiment only once the final first-experiment gate is recorded.
test -f changing-receipts/summary.json
git -C text tag -a checkpoint-2026-11-26-agent3-01-text -m 'Agent text-token cache, 404k changing-return replay and frozen coding gates passed'
git -C /home/benchmark/llm/ds4-flash-rs worktree add -b experiment/agent-markov-pairs "$p/markov" experiment/agent-text-cache
python3 make_markov.py
env -u SHELLOPTS bash build.sh markov
python3 - <<'PY'
import json
from pathlib import Path
a=['ds4-flash-rs:agent3-markov',{'DS4_AGENT_TEXT_CACHE':'1','DS4_MARKOV_PAIRS':'0'}]
c=[a[0],dict(a[1],DS4_MARKOV_PAIRS='1')]
Path('markov.json').write_text(json.dumps([a,c,a]))
PY
python3 -u roundbench.py markov > markov-bench.log 2>&1
python3 verify.py markov > markov-summary.log
python3 gpu_test.py
