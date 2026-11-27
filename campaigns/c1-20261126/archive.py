from pathlib import Path
from shutil import copyfile
import hashlib,json
p=Path(__file__).resolve().parent;dest=Path.home()/'.cache/ds4-flash-rs/profiles/c1-20261126'
dest.mkdir(parents=True,exist_ok=True);dest.chmod(0o700)
include={'run.py','edit.py','record.py','gpu.py','microrun.py','build.sh','config.py','extra.py','qualify.py','heldout.py','tag.sh','profile.py','archive.py','sass.py','micro-result.py','summary.py','gates.py','activate.py','goldcheck.py','verify-live.py','add-full-evidence.py','inspect-record.py','continue-lookup.py'}
keep=[x for x in p.iterdir() if x.name in include]
for name in ('wide','expert','expert-r2','lookup','baseline','timeline','selected'):
 keep.extend(x for x in (p/name).iterdir() if x.is_file() and (x.suffix in {'.json','.jsonl','.cu','.nsys-rep','.patch'} or x.name in {'run.log','micro.txt','gpu-tests.log','build.log','clippy.log'}))
manifest={}
for x in keep:
 where=dest/x.relative_to(p);where.parent.mkdir(parents=True,exist_ok=True);copyfile(x,where)
 manifest[str(x.relative_to(p))]=hashlib.sha256(where.read_bytes()).hexdigest()
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest),dest)
