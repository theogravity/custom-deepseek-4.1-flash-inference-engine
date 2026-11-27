from pathlib import Path
import json,subprocess as sp,sys,time
p=Path(__file__).resolve().parent;d=p/'lookup'
until=time.monotonic()+1800
while not ((d/'heldout-summary.json').exists() and (d/'repeat-summary.json').exists()):
 if time.monotonic()>until:raise RuntimeError('waiting for held-out/confirmation exceeded 30 minutes')
 time.sleep(3)
r=json.loads((d/'repeat-summary.json').read_text())['legs']
for sh in ('edit','long'):
 a,on,b=[r['repeat-'+label][sh]['median'] for label in ('off-a','on','off-b')]
 assert on>max(a,b),(sh,a,on,b)
held=json.loads((d/'heldout-summary.json').read_text())['by_leg']
for sh in ('game','parser','cache'):
 a,on,b=[held[label][sh]['median'] for label in ('off-a','on','off-b')]
 print(sh,a,on,b,flush=True)
 assert on>=max(a,b)*0.97,(sh,a,on,b)
sp.run([sys.executable,p/'gates.py','lookup'],check=True)
