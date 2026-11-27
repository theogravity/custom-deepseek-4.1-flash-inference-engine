from pathlib import Path
import json,hashlib,sys,re,statistics as st
folder=Path('/tmp/ds4-c1-1126')/sys.argv[1]; repo=folder/'work'
cfg=json.loads((folder/'config.json').read_text()); raw={}
for label,img,flags,shapes,rounds in cfg:
 d=json.loads((folder/(label+'.json')).read_text())
 assert d['env']==flags
 for shape in shapes:
  source=(folder/(label+'-'+shape+'.jsonl')).read_bytes()
  assert hashlib.sha256(source).hexdigest()==d[shape]['raw_sha256']
  waves=[json.loads(l) for l in source.splitlines()]
  assert len(waves)==rounds+(1 if shape.endswith('edit') else 2)
  ws=waves[1:] if shape.endswith('edit') else waves[1:-1]
  assert len(ws)==rounds and all(w['delta']['requests_started']==1 and w['delta']['rotations']==0 and w['delta']['prefix_reused_tokens']==(25600 if shape=='edit' else 404096) for w in ws)
  assert d[shape]['rates']==[w['upstream_tok_s' if shape=='edit' else 'aggregate_decode_tok_s'] for w in ws]
 raw[label]=d
assert len({x['image'] for x in raw.values()})==1
log=(folder/'gpu-tests.log').read_text()
m=re.search(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored',log);assert m
data={'checkpoint':sys.argv[2],'decision':sys.argv[3],'suite':'One warm-up and six warm waves per workload on a freshly started process for each leg. Edit first, then coding.','devices':'host GPUs 1, 2; RTX PRO 6000 Blackwell, 375 W per GPU, 225 W CPU socket','edit':'25,625 input, 1,500 output, published first-token-to-end metric including reasoning; deterministic edit of functions 50–90.','long':'404,151 input, 512 output, first-to-last visible text excluding first token; frozen coding history.','gpu_tests_passed':int(m[1]),'base_profile':'compose.yml + compose.decode-optimized.yml; DS4_BATCH_TOPK=1, DS4_EXPERT_EVICTION=lfu. Each leg additionally sets the flags in config.','config':cfg,'legs':raw}
for f in ['micro-summary.json','repeat-summary.json','direct-summary.json','gate-summary.json','heldout-summary.json','full-validation.json','detail.json']:
 if (folder/f).exists():
  entry=json.loads((folder/f).read_text())
  if f=='heldout-summary.json':assert entry['images']=={k:v['image'] for k,v in raw.items()}
  if f=='full-validation.json':assert entry['image']==raw['on']['image'] and entry['flags']==raw['on']['env']
  data[f.split('.')[0]]=entry
(repo/'results'/('c1-'+folder.name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
