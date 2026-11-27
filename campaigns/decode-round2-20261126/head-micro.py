from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent/'head';f=p/'micro.txt'
raw=[json.loads(l) for l in f.read_text().splitlines()]
by={}
for r in raw:by.setdefault(r['batch'],[]).append(r)
assert sorted(by)==[8,9,15,16,17,31,32,47,48,49,64,96,97,128,129,511,512]
for b,group in by.items():
 assert [r['leg'] for r in group]==[0,1,2] and all(r['compared_values']==b*129280 for r in group[1:])
(p/'micro-summary.json').write_text(json.dumps({'shape':'129280 vocabulary x 5120 hidden, rank-local projection, one split, synthetic nonzero signed FP8 weights and per-row scales, 3 dispatches per event sample, 8 timed samples per leg','by_verification_rows':by,'raw_sha256':hashlib.sha256(f.read_bytes()).hexdigest()},indent=2)+'\n')
for b,g in by.items():print(b,*[round(r['median_us']) for r in g])
