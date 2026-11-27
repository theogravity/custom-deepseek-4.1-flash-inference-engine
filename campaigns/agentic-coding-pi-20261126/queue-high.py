import subprocess as sp,time
from pathlib import Path
p=Path(__file__).resolve().parent
while True:
 log=p/'replay.log'
 done=p/'slice-after.jsonl'
 if done.exists() and len(done.read_text().splitlines())==7:break
 if log.exists() and 'Traceback (most recent call last):' in log.read_text():raise SystemExit('Replay did not complete; investigate before high effort trial')
 time.sleep(2)
with (p/'high.log').open('w') as f:
 sp.run(['python3','-u',str(p/'high.py')],check=True,stdout=f,stderr=sp.STDOUT)
