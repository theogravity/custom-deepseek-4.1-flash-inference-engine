import fcntl,subprocess as sp,time
from pathlib import Path
p=Path(__file__).resolve().parent
rows=p/'selected-high.jsonl'
while not rows.exists() or len(rows.read_text().splitlines())<3:
    f=p/'high.log'
    if f.exists() and 'Traceback' in f.read_text() and 'File "'+str(p/'high.py')+'"' in f.read_text():
        if 'AssertionError' in f.read_text() or 'ValueError' in f.read_text():raise SystemExit('high trial failed')
    time.sleep(2)
with open('/tmp/ds4-native-bench.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    sp.run(['python3',str(p/'prompt-audit.py')],check=True)
    sp.run(['python3',str(p/'evidence.py')],check=True)
    sp.run(['python3',str(p/'archive.py')],check=True)
