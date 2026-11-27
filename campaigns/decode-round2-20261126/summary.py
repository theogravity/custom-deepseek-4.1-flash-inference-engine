from pathlib import Path
from statistics import mean
import json,sys
p=Path(__file__).resolve().parent/sys.argv[1];cfg=json.loads((p/(sys.argv[2] if len(sys.argv)>2 else 'config.json')).read_text())
rows=[json.loads((p/(r[0]+'.json')).read_text()) for r in cfg]
for shape in cfg[0][3]:
 a,on,b=[v[shape]['median'] for v in rows]
 print(f'{shape}: {a:.2f} / {on:.2f} / {b:.2f} tok/s. vs control midpoint {(on/mean((a,b))-1)*100:+.2f}%, vs each {(on/a-1)*100:+.2f}% and {(on/b-1)*100:+.2f}%')
 for k in ('learned','lookup'):
  print(k,[v[shape]['spec_including_warmup'].get(k) for v in rows])
