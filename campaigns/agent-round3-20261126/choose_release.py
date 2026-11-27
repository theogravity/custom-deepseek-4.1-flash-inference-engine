from pathlib import Path
import json,subprocess as sp
p=Path(__file__).resolve().parent
def get(exp,head):
    return json.loads((p/(exp+'-receipts')/(head+'.json')).read_text())
stats=get('hash','summary')
source=p/'hash'
with (source/'src/prefix.rs').open() as f:assert '!self.ram.is_empty()' in f.read()
sp.run(['git','add','src/prefix.rs'],cwd=source,check=True)
sp.run(['git','diff','--cached','--check'],cwd=source,check=True)
sp.run(['git','commit','-m','Hash bulk token bytes and reuse terminal KV keys'],cwd=source,check=True)
sp.run(['git','tag','-a','checkpoint-2026-11-26-agent3-03-prefix','-m','Bulk prefix lookup, cross-version disk and RAM names verified'],cwd=source,check=True)
# Set the final decision from the whole-request and internal measurements.
# This script expects an explicit decision after the measured arms have been read.
import sys
exp=sys.argv[1]
assert exp in ('text','hash')
selected=p/exp
assert not sp.check_output(['git','status','--porcelain'],cwd=selected,text=True)
flags={'DS4_AGENT_TEXT_CACHE':'1'}
if exp=='hash':flags['DS4_BULK_PREFIX_KEYS']='1'
choice={'image':'ds4-flash-rs:agent-coding-20261126','worktree':exp,'flags':flags,
    'commit':sp.check_output(['git','rev-parse','HEAD'],cwd=selected,text=True).strip()}
(p/'chosen.json').write_text(json.dumps(choice,indent=2)+'\n')
base=['ds4-flash-rs:agentic-idle-20261126',{}]
(p/'release.json').write_text(json.dumps([base,[choice['image'],flags],base]))
