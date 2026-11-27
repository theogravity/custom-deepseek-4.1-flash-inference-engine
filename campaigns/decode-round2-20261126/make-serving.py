from pathlib import Path
import json,sys,subprocess as sp
p=Path(__file__).resolve().parent/sys.argv[1]
root=Path('/home/benchmark/llm/ds4-flash-rs')
tag=sys.argv[2]; cfg=json.loads((p/'config.json').read_text())[1]
identity=json.loads((p/'full-validation.json').read_text())['image']
assert sp.check_output(['docker','image','inspect',cfg[1],'-f','{{.Id}}'],text=True).strip()==identity
current=sp.run(['docker','image','inspect','-f','{{.Id}}',tag],capture_output=True,text=True)
if current.returncode==0:assert current.stdout.strip()==identity
else:sp.run(['docker','image','tag',identity,tag],check=True)
contents=(root/'compose.c1-optimized.yml').read_text().replace('ds4-flash-rs:c1-consensus-20261126',tag)
for k,v in cfg[2].items():
 assert k in ('DS4_HEAD_ROW_TILES','DS4_EXTEND_REPETITION') and v=='1'
 contents+='      '+k+': "${'+k+':-1}"\n'
(root/'compose.decode-round2.yml').write_text(contents)
