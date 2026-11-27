import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent/__import__('sys').argv[1];f=p/'full-validation.json';v=json.loads(f.read_text());raw={}
for k in ['context-needles','six-full','eight-ram','eight-disk']:
 data=(p/(k+'.jsonl')).read_bytes();ws=[json.loads(l) for l in data.splitlines()]
 raw[k]=hashlib.sha256(data).hexdigest()
 if k.startswith('eight'):
  w=ws[0];s=v[k.replace('-','_')]
  assert s['passed'] and w['passed'] and w['peak_logical_active']==8 and len(w['rows'])==8
  assert all(r['serial_sha256']==r['sha256'] for r in w['rows'])
  s.update(shared_register=True,peak_logical=w['peak_logical_active'],matched_serial_responses=8,swap_delta=w['swap_delta'],concurrent_seconds=w['wave_seconds'])
v['raw_sha256']=raw
f.write_text(json.dumps(v,indent=2)+'\n')
