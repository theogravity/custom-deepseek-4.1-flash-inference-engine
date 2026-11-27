from pathlib import Path
from shutil import copyfile
import hashlib,json
p=Path(__file__).resolve().parent
dest=Path.home()/'.cache/ds4-flash-rs/profiles/decode-round2-20261126'
dest.mkdir(parents=True,exist_ok=True);dest.chmod(0o700)
files=[x for x in p.iterdir() if x.is_file() and (x.suffix in ('.py','.sh') or x.name=='report-base.md')]
for name in ('baseline','cycle','markov','head','selected'):
 if (p/name).exists():
  files.extend(x for x in (p/name).iterdir() if x.is_file() and x.name!='tests.json' and (x.suffix in {'.json','.jsonl','.cu','.patch'} or x.name in {'run.log','micro.txt','gpu-tests.log','build.log','clippy.log'}))
manifest={}
for x in files:
 where=dest/x.relative_to(p);where.parent.mkdir(parents=True,exist_ok=True);copyfile(x,where)
 manifest[str(x.relative_to(p))]=hashlib.sha256(where.read_bytes()).hexdigest()
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest),dest)
