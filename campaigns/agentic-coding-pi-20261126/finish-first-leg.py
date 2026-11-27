import os,time,signal
from pathlib import Path
from setup import p
for d in Path('/proc').glob('[0-9]*'):
 try:
  if (d/'cmdline').read_bytes().split(b'\0')[:2]==[b'python3',b'sweep.py'] and (d/'fd/1').resolve()==p/'sweep.log':
   pid=int(d.name);break
 except FileNotFoundError:pass
else:raise SystemExit('sweep not running')
while len((p/'before.jsonl').read_text().splitlines())<3:time.sleep(.04)
os.kill(pid,signal.SIGINT)
