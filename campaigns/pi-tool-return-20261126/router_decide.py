from post_canonical import analyze
from pathlib import Path
import json
p=Path(__file__).resolve().parent
legs=analyze(['bound-off-a','bound-on','bound-off-b'])
a,n,b=[legs[l]['ttft'] for l in legs]
wins=[100*(1-n/x) for x in (a,b)]
print('router improvement over each control',wins)
res={'improvement_pct_vs_each_control':wins,'canonical_medians':legs,
     'repeat_on_full_exchange':min(wins)>=3,
     'description':'The gain was below 3% against at least one control, so the selected image retains the original router.'}
if res['repeat_on_full_exchange']:res['description']='The router improved both isolated controls by at least 3%. An independent complete-exchange replay is required before promoting this candidate.'
(p/'router-decision.json').write_text(json.dumps(res,indent=2)+'\n')
