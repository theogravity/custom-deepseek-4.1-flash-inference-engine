import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('h',str(Path(__file__).with_name('run.py')))
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
_,_,env=h.b.inspect()
ident=h.b.verify('native',h.flags,h.b.native)
out={k:env[k] for k in h.flags}
assert out==h.flags
path=h.p/'baseline/serving-config.json'
path.write_text(json.dumps({'image':ident,'serving':out,'timing':'read back after baseline on the same configured release'},indent=2)+'\n')
