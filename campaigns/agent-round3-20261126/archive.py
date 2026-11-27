import hashlib,json,os,shutil
from pathlib import Path
p=Path(__file__).resolve().parent
dest=Path.home()/'.cache/ds4-flash-rs/profiles/agent-round3-20261126'
dest.mkdir(exist_ok=True,parents=True,mode=0o700)
sources=set()
for pattern in ['*.py','*.sh','*.log','*.json','*-receipts/*.json','*-receipts/*.jsonl','mixed-receipts/*/*.jsonl','mixed-receipts/*/*.json','qualification/*.log','qualification/*.json','qualification/*.jsonl']:
    sources.update(f for f in p.glob(pattern) if f.is_file() and f.name not in ('qualified-tests.json','qualify_release.sh',) and 'compose' not in f.name and 'override' not in f.name)
manifest={}
entries=[(f,f.relative_to(p)) for f in sources]
for folder,name in [('ds4-tool-return-1126','run_pair.py'),('ds4-agentic-live-1126','replay.py'),
                    ('ds4-agentic-live-1126','sweep.py'),('ds4-agentic-live-1126','trial.py'),
                    ('ds4-agentic-live-1126','setup.py'),('ds4-edit-gap','run.py'),
                    ('ds4-next-375w-2301','bench.py'),('ds4-cycle2-1126','run.py'),
                    ('ds4-prefix-sharing','qualify.py'),('ds4-prefix-sharing','run.py'),
                    ('ds4-eviction-1126','six_streams_idle.py'),('ds4-cycle2-1126','edit.py')]:
    entries.append((Path('/tmp')/folder/name,Path('helpers')/folder/name))
for f in [Path('/tmp/ds4-six.pFsc06/404k-histories.json'),Path('/tmp/ds4-agentic-live-1126/frozen-red-turn.json')]:
    entries.append((f,Path('fixtures')/f.name))
for rel in ['release-provenance.json','release-receipts/summary.json','qualification/passed.json']:
    f=p/'reused-build'/rel
    if f.is_file(): entries.append((f,Path('excluded-shared-build')/rel))
for f,name in sorted(entries):
    target=dest/name;target.parent.mkdir(exist_ok=True,parents=True,mode=0o700)
    data=f.read_bytes()
    key=next(l.partition('=')[2].strip().strip("'").strip('"') for l in (Path.home()/'llm/models/.env').read_text().splitlines() if l.startswith('LITELLM_MASTER_KEY='))
    assert key.encode() not in data
    target.write_bytes(data);target.chmod(0o600)
    manifest[str(target.relative_to(dest))]=hashlib.sha256(data).hexdigest()
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest),dest)
