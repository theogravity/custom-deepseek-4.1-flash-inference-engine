from pathlib import Path
import re,json,subprocess as sp,time
from setup import p
seen=set()
while True:
 ctl=sp.run(['docker','inspect','--format','{{.Id}} {{range .Mounts}}{{if eq .Destination "/prefix"}}{{.Source}}{{end}}{{end}}','dsv41-flash-exl3'],capture_output=True,text=True)
 if not ctl.returncode:
  cid,mount=ctl.stdout.strip().split()
  if mount in [str(p/('kv-'+k)) for k in ('before','selected','after')] and cid not in seen:
   seen.add(cid)
   label=Path(mount).name.removeprefix('kv-')
   with sp.Popen(['docker','logs','--timestamps','--follow',cid],stdout=sp.PIPE,stderr=sp.STDOUT,text=True) as proc,(p/('spec-'+label+'.jsonl')).open('w') as f:
    for l in proc.stdout:
     m=re.search(r'speculative source=(learned|lookup) rounds=(\d+) proposed=(\d+) accepted=(\d+)',l)
     if m:print(json.dumps([l.split()[0],m[1],*map(int,m.groups()[1:])]),file=f,flush=True)
   if label=='after':break
 time.sleep(1)
