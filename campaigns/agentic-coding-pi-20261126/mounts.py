import json,subprocess as sp,time
from setup import p
seen=set();last=None
while True:
 v=sp.run(['docker','inspect','--format','{{.Image}} {{.State.Health.Status}} {{range .Mounts}}{{if eq .Destination "/prefix"}}{{.Source}}{{end}}{{end}}','dsv41-flash-exl3'],capture_output=True,text=True)
 if v.returncode==0:
  fields=v.stdout.strip().split();mount=fields[-1]
  if fields[1]=='healthy' and mount!=last:
   seen.add(mount);last=mount
   with (p/'mounts.jsonl').open('a') as out:print(json.dumps({'observed':time.time(),'image':fields[0],'mount':mount}),file=out)
   if str(p/'kv-after') in seen and not mount.startswith(str(p)):break
 time.sleep(1)
