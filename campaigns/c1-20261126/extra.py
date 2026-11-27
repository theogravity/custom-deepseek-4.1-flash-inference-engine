from pathlib import Path
import sys,json,statistics
p=Path('/tmp/ds4-c1-1126')/sys.argv[1];kind=sys.argv[2]
cfg=json.loads((p/'config.json').read_text())
shapes=sys.argv[3:]
if shapes:
 for r in cfg:r[0]=kind+'-'+r[0];r[3]=shapes
 (p/(kind+'-config.json')).write_text(json.dumps(cfg,indent=2))
else:
 cfg=json.loads((p/(kind+'-config.json')).read_text())
 data={r[0]:json.loads((p/(r[0]+'.json')).read_text()) for r in cfg}
 (p/(kind+'-summary.json')).write_text(json.dumps(dict(config=cfg,legs=data),indent=2))
