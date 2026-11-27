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
