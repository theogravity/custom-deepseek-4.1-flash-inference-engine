from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent/'cycle'
f=p/'initial-eight-disk.jsonl'
w=json.loads(f.read_text())
assert not w['passed'] and w['metrics_after']['requests_started']-w['metrics_before']['requests_started']==9
r={'reason':'Nine requests reached the server during an eight-client trial. Excluded from validation.','wave_passed':w['passed'],'requests_started':9,'peak_logical_active':w['peak_logical_active'],'matching_serial_responses':sum(x['sha256']==x['serial_sha256'] for x in w['rows']),'swap_delta':w['swap_delta'],'raw_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
d=json.loads((p/'detail.json').read_text());d['discarded_capacity_trial']=r;(p/'detail.json').write_text(json.dumps(d,indent=2)+'\n')
