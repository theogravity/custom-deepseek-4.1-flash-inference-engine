from pathlib import Path
import json,hashlib,statistics
root=Path('/home/benchmark/llm/ds4-flash-rs');p=Path('/tmp/ds4-cpu-eval/full-window-confirm')
dest=root/'results/decode-selected-service.json';record=json.loads(dest.read_text())
legs={}
for name in ['off-a','on','off-b']:
 v=json.loads((p/(name+'-validation.json')).read_text())
 assert v['image']==record['image'] and v['six_full_matched']==42 and v['logical_limit']==32
 for shape in ['code','full']:
  raw=(p/(name+'-'+shape+'.jsonl')).read_bytes();v[shape+'_raw_sha256']=hashlib.sha256(raw).hexdigest()
  w=[json.loads(l) for l in raw.splitlines()]
  warm=w[1:] if shape=='full' else w[1:-1]
  assert len(warm)==(6 if shape=='full' else 3)
  measured=[r['aggregate_decode_tok_s'] for r in warm]
  assert measured==v['c6_524160_receipt_128' if shape=='full' else 'c6_404k_512']
  if shape=='full':
   assert all(a['metrics_after']['prefix_reused_tokens']-a['metrics_before']['prefix_reused_tokens']==6*524096 and a['metrics_after']['rotations']==a['metrics_before']['rotations'] and a['all_valid'] and a['concurrent'] for a in warm)
  v[shape+'_median_tok_s']=statistics.median(measured)
 legs[name]=v
record['bracketed_coding_to_full_window']=legs
record['activation']=json.loads(Path('/tmp/ds4-cpu-eval/selected-service/activation.json').read_text())
dest.write_text(json.dumps(record,indent=2)+'\n')
doc=root/'DECODE-OPTIMIZATIONS.md';s=doc.read_text()
vals=[legs[k]['full_median_tok_s'] for k in ['off-a','on','off-b']]
code=[legs[k]['code_median_tok_s'] for k in ['off-a','on','off-b']]
text=(f'A separate full-window timing repeated the transition from six coding sessions before each leg. With tiled ranking disabled, enabled and disabled again, the six warm full-window receipt waves in each leg had medians of **{vals[0]:.1f} / {vals[1]:.1f} / {vals[2]:.1f} aggregate tok/s**. The inputs were 524,160 tokens with 128 output tokens each. All 126 full-window responses matched the references, with six residents and zero rotations during the waves. The preceding 404k coding medians at 512 output tokens were {code[0]:.1f} / {code[1]:.1f} / {code[2]:.1f} aggregate tok/s.\n')
minimum=min(record['full_min']['min_free_mib'][:2]+legs['on']['full_min']['min_free_mib'][:2])
text+=f'\nAcross the two selected-profile coding-to-full-window checks, the lowest sampled free VRAM on either inference GPU was {minimum} MiB. Eight full-window clients were verified separately with six residents and two rotated through RAM or disk.\n'
doc.write_text(s+'\n'+text)
print(text)
