from pathlib import Path
import hashlib,json,sys,statistics as st
root=Path(__file__).resolve().parent;folder=root/sys.argv[1];path=folder/'work/results'/('c1-'+folder.name+'.json');r=json.loads(path.read_text())
for k,g in [('legs',r['legs']),*((k,v['legs']) for k,v in r.items() if k.endswith('-summary') and isinstance(v,dict) and isinstance(v.get('legs'),dict))]:
 for label,leg in g.items():
  for sh,res in leg.items():
   if sh in ('env','image'):continue
   file=folder/(label+'-'+sh+'.jsonl');b=file.read_bytes()
   assert hashlib.sha256(b).hexdigest()==res['raw_sha256']
   rows=list(map(json.loads,b.splitlines()))[1:];rows=rows if sh.endswith('edit') else rows[:-1]
   assert len(rows)==6 and res['rates']==[w['upstream_tok_s' if sh.endswith('edit') else 'aggregate_decode_tok_s'] for w in rows] and res['median']==st.median(res['rates'])
print(path,'OK')
