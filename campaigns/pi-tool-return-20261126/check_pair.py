"""Gate pilot claimed parity independently against the captured SSE outputs and admissions."""
import hashlib,json
from pathlib import Path
p=Path(__file__).parent
source=p/'capture-b'
task=json.loads((source/'task.json').read_text());assert task['isolated'] and task['graded_correct']
def digest(value,*,ordered=False):
 return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=not ordered).encode()).hexdigest()
modes={}
for label in ['p1-off-a','p1-on','p1-off-b']:
 rows=json.loads((p/(label+'.json')).read_text());assert [r['turn'] for r in rows]==[4,5,6]
 for r in rows:
  n=r['turn'];req=json.loads((source/f'request-{n:02}.json').read_text());future=json.loads((source/f'request-{n+1:02}.json').read_text());raw=task['turns'][n-1]
  expected=next(m for m in reversed(future['messages']) if m['role']=='assistant')
  calls=[(c['function']['name'],json.loads(c['function']['arguments'])) for c in expected['tool_calls']]
  expected_out=[expected.get('reasoning_content') or '',expected.get('content') or '',calls]
  assert r['input_sha']==digest(req) and raw['request_sha256']==hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest() and r['prompt']==raw['usage']['prompt_tokens']
  r['reconstructed_ordered_wire_sha256']=hashlib.sha256(json.dumps(req).encode()).hexdigest()
  r['exact_recorded_continuation']=(r['output_sha']==digest(expected_out) and r['finish']==raw['finish'] and r['tokens']==raw['usage']['completion_tokens'])
 modes[label]=rows
for n in range(3):
 a,c,b=[modes[k][n] for k in modes];assert a['reconstructed_ordered_wire_sha256']==c['reconstructed_ordered_wire_sha256']==b['reconstructed_ordered_wire_sha256']
 assert a['exact_recorded_continuation'] and b['exact_recorded_continuation']
assert modes['p1-on'][0]['exact_recorded_continuation']
candidate_ok=all(r['exact_recorded_continuation'] for r in modes['p1-on'])
report={'eligible_for_deployment':candidate_ok,'legs':modes}
(p/'pilot-gate.json').write_text(json.dumps(report,indent=2))
print('quality gate',candidate_ok)
if not candidate_ok:raise SystemExit(1)
