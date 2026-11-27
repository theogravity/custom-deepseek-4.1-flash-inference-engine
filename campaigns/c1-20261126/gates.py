from pathlib import Path
import json,subprocess as sp,sys,statistics
p=Path(__file__).resolve().parent;name=sys.argv[1];d=p/name
sp.run([sys.executable,p/'run.py',name,'gate-config.json'],check=True)
sp.run([sys.executable,p/'extra.py',name,'gate'],check=True)
data=json.loads((d/'gate-summary.json').read_text())
for shape in ('sixedit','sixlong'):
 a,c,b=[data['legs']['gate-'+label][shape]['median'] for label in ('off-a','on','off-b')]
 print(shape,a,c,b,flush=True)
 if c < max(a,b)*0.97:raise RuntimeError('six-client regression in '+shape)
sp.run([sys.executable,p/'qualify.py',name],check=True)
