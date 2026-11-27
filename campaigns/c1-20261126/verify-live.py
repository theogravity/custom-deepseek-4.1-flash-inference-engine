from pathlib import Path
import sys,fcntl,json,importlib.util
s=importlib.util.spec_from_file_location('h','/tmp/ds4-c1-1126/run.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
p=h.p/'selected';p.mkdir(exist_ok=True);b=h.b
a=json.loads((h.p/'lookup/activation.json').read_text())
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 b.verify('native',a['flags'],a['tag']);h.leg(p,'gold',a['tag'],{'DS4_C1_OUTPUT_CONSENSUS':'1'},['edit','long'],6)
 d=json.loads((p/'gold.json').read_text());assert d['image']==a['image']
 b.verify('native',a['flags'],a['tag']);b.idle('native')
 (h.root/'results/c1-selected-service.json').write_text(json.dumps(a|{'c1_serving':d},indent=2)+'\n')
