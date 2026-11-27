from pathlib import Path
import hashlib,json,statistics
p=Path(__file__).resolve().parent;d=json.loads((p/'baseline/gold.json').read_text())
assert d['image']=='sha256:888d1ec86cd7e4fb55cd80add72f6338c6afead29619c8bd4cd86027c4947182'
for k in ('edit','long'):
 b=(p/'baseline'/('gold-'+k+'.jsonl')).read_bytes();assert hashlib.sha256(b).hexdigest()==d[k]['raw_sha256']
 waves=[json.loads(l) for l in b.splitlines()]
 for w in waves[:7]:
  fields=['prompt_sha256','reasoning_sha256','content_sha256'] if k=='edit' else ['sha256']
  sig=[r[f] for r in w['rows'] for f in fields]
  assert sig==(d[k]['hashes'][0] if k=='edit' else d[k]['hashes'])
 rates=[w['upstream_tok_s' if k=='edit' else 'aggregate_decode_tok_s'] for w in waves[1:7]]
 assert rates==d[k]['rates'] and statistics.median(rates)==d[k]['median']
print('Verified original image, 14 response hashes and six measured rates per workload.')
