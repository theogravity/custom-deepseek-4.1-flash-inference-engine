"""Archive benchmark receipts and reproducible analysis, excluding caches and credentials."""
from pathlib import Path
import hashlib, json, os, shutil
p=Path(__file__).resolve().parent
dest=Path.home()/'.cache/ds4-flash-rs/profiles/pi-tool-return-20261126'
dest.mkdir(mode=0o700,parents=True,exist_ok=True)
os.chmod(dest,0o700)
files=set()
for glob in ['*off-?.json','*on.json','q-*-?.json','real-wide.json','pilot-gate.json','*.py','router_run.sh',
             '*tests*.log','*clippy*.log','*image*.log','final-pairs.log','qualification.log','router-run.log','provenance.json','deployed.json','router-decision.json',
             'capture-b/request-*.json','capture-b/task.json','capture-b/*.py','capture-b/hidden.go.txt','capture-b/fixture/*',
             'qualification/*.json','qualification/*.jsonl','qualification/gpu-tests.log']:
    files.update(f for f in p.glob(glob) if f.is_file() and '.override.' not in f.name and '-override.' not in f.name)
deps=[('ds4-agentic-live-1126','sweep.py'),('ds4-agentic-live-1126','trial.py'),('ds4-agentic-live-1126','setup.py'),
      ('ds4-edit-gap','run.py'),('ds4-next-375w-2301','bench.py'),('ds4-cycle2-1126','run.py'),
      ('ds4-cycle2-1126','edit.py'),('ds4-prefix-sharing','run.py'),('ds4-prefix-sharing','qualify.py'),
      ('ds4-eviction-1126','six_streams_idle.py'),('ds4-cpu-eval','image.py'),('ds4-cpu-eval','testbin.py')]
items=[(f,f.relative_to(p)) for f in files]+[(Path('/tmp')/d/f,Path('helpers')/d/f) for d,f in deps]
manifest={}
secrets=[l.partition('=')[2].strip().strip("'").strip('"').encode() for l in (Path.home()/'llm/models/.env').read_text().splitlines() if l.startswith('LITELLM_MASTER_KEY=')]
assert len(secrets)==1 and len(secrets[0])>=12
for f,name in sorted(items):
    target=dest/name
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if secrets[0] in f.read_bytes(): raise RuntimeError("credential in archive input "+str(name))
    shutil.copyfile(f,target)
    os.chmod(target,0o600)
    manifest[str(name)]=hashlib.sha256(target.read_bytes()).hexdigest()
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(len(manifest),'files in',dest)
