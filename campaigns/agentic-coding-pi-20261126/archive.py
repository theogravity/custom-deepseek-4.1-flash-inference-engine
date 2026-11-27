from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent
dst=Path.home()/'.cache/ds4-flash-rs/profiles/agentic-coding-pi-20261126';dst.mkdir(exist_ok=True,mode=0o700,parents=True)
files=[*p.glob('*.py'),*p.glob('*.jsonl'),p/'frozen-red-turn.json',p/'prompt-audit.json',p/'original-functions.txt',p/'hidden.go.txt',*sorted((p/'fixture').iterdir())]
manifest={}
for f in files:
 target=dst/f.relative_to(p);target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(f,target)
 manifest[str(f.relative_to(p))]=hashlib.sha256(target.read_bytes()).hexdigest()
# Keep source dependencies from the earlier same-host comparison with this run.
for folder,name in [('ds4-edit-gap','run.py'),('ds4-next-375w-2301','bench.py')]:
 source=Path('/tmp')/folder/name
 target=dst/'helpers'/folder/name
 target.parent.mkdir(exist_ok=True,parents=True)
 shutil.copyfile(source,target)
 manifest[str(target.relative_to(dst))]=hashlib.sha256(target.read_bytes()).hexdigest()
(dst/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest),dst)
