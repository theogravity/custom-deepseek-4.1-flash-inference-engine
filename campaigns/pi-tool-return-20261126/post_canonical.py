import hashlib,json,sys,statistics as st
from pathlib import Path
p=Path(__file__).resolve().parent
def analyze(labels):
 f=p/'capture-b';body=json.loads((f/'request-05.json').read_text());target=next(m for m in reversed(json.loads((f/'request-06.json').read_text())['messages']) if m['role']=='assistant')
 out=(target.get('reasoning_content') or '',target.get('content') or '',[(c['function']['name'],json.loads(c['function']['arguments'])) for c in target['tool_calls']])
 usage=json.loads((f/'task.json').read_text())['turns'][4]
 wire=hashlib.sha256(json.dumps(body).encode()).hexdigest();reference=hashlib.sha256(json.dumps(out).encode()).hexdigest();result={}
 for lab in labels:
  rows=json.loads((p/(lab+'.json')).read_text());assert [r['i'] for r in rows]==list(range(7))
  for r in rows:
   assert r['wire_sha']==wire and r['prompt']==usage['usage']['prompt_tokens'] and r['input_sha']==usage['request_sha256']
   assert r['parity']==(r['ordered_output_sha']==reference and r['finish']==usage['finish'] and r['tokens']==usage['usage']['completion_tokens'])
  result[lab]={'matches':sum(r['parity'] for r in rows),'samples':len(rows),'ttft':st.median(r['ttft'] for r in rows[1:]),'wall':st.median(r['wall'] for r in rows[1:]),'reuse':rows[0]['reuse'],'chunk':rows[0]['chunk']}
 return result
if __name__=='__main__':
 result=analyze(sys.argv[1:]);print(json.dumps(result,indent=2))
 if any(v['matches']!=v['samples'] for v in result.values()):sys.exit(1)
