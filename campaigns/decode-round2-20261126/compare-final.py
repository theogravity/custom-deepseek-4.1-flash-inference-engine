from pathlib import Path
import json,sys,subprocess as sp
folder=Path(__file__).resolve().parent/sys.argv[1]
cfg=json.loads((folder/'config.json').read_text())
base='ds4-flash-rs:c1-consensus-20261126'
for row in cfg:
 row[3]=sys.argv[2:] or ['edit','long','sixedit','sixlong']
 if row[0]!='on':row[1]=base;row[2]={}
 row[0]='original-'+row[0]
(folder/'original-config.json').write_text(json.dumps(cfg,indent=2))
sp.run([sys.executable,'run.py',folder.name,'original-config.json'],check=True)
sp.run([sys.executable,'extra.py',folder.name,'original'],check=True)
d=json.loads((folder/'original-summary.json').read_text())
for k in cfg[0][3]:
 a,n,b=[d['legs'][r[0]][k]['median'] for r in cfg]
 print(k,a,n,b,flush=True)
 assert n>=.97*(a+b)/2,(k,'regression',a,n,b)
