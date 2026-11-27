from pathlib import Path
import json,sys
p=Path(__file__).resolve().parent
for name in ['cycle','markov','head']:
 for kind in ['', 'repeat-', 'gate-', 'original-']:
  files=[p/name/(kind+leg+'.json') for leg in ['off-a','on','off-b']]
  if not all(f.exists() for f in files): continue
  ds=[json.loads(f.read_text()) for f in files]
  for key in ds[0]:
   if key in ['env','image']:continue
   a,c,b=(d[key]['median'] for d in ds)
   print(f'{name} {kind}{key} {a:.2f} / {c:.2f} / {b:.2f} ({(c/((a+b)/2)-1)*100:+.2f}%)')
