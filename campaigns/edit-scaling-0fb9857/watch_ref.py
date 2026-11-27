from pathlib import Path
from datetime import datetime
import json,re,subprocess as sp,time
p=Path(__file__).resolve().parent
# Save only structured CPU worker counters; container startup logs can contain configuration credentials.
pattern=r'\[cpuexp r(\d+)\]\s*(\d+) jobs\s*\(\+(\d+), mean ([\d.]+) us\), rows (\d+), rescues (\d+), errors (\d+), degraded (\d+)'
seen={}
while not (p/'vllm-on-b.jsonl').exists() or len((p/'vllm-on-b.jsonl').read_text().splitlines())<7:
 info=sp.run(['docker','inspect','-f','{{.Id}} {{.Config.Image}} {{.State.Status}}','dsv41-flash-exl3'],capture_output=True,text=True)
 if 'sm120-lmcache' in info.stdout and 'running' in info.stdout:
  cid=info.stdout.split()[0]
  log=sp.run(['docker','logs',cid],capture_output=True,text=True)
  rows=re.findall(pattern,log.stdout+'\n'+log.stderr)
  if rows:
   seen[cid]=rows
   (p/'cpu-worker-counters.json').write_text(json.dumps(seen))
 if (p/'reference.log').exists() and 'Traceback' in (p/'reference.log').read_text():break
 time.sleep(12)
