from pathlib import Path
import subprocess as sp,sys,json,tempfile,hashlib,re
p=Path(__file__).resolve().parent/sys.argv[1]
config=json.loads((p/'config.json').read_text())[1]
image=sp.check_output(['docker','image','inspect',config[1],'-f','{{.Id}}'],text=True).strip()
assert image==json.loads((p/'on.json').read_text())['image']
def hashfile(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with tempfile.TemporaryDirectory() as d:
 target=Path(d)/'ds4-flash-rs'
 c=sp.check_output(['docker','create',image],text=True).strip()
 try:sp.run(['docker','cp',c+':/usr/local/bin/ds4-flash-rs',target],check=True)
 finally:sp.run(['docker','rm',c],stdout=sp.DEVNULL,check=True)
 exe=hashfile(target)
assert exe==hashfile(Path('/tmp/ds4-cpu-eval/target/release/ds4-flash-rs'))
log=(p/'gpu-tests.log').read_text()
tests=int(re.search(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored',log)[1])
base=json.loads((p.parent/'baseline/serving-config.json').read_text())
data={'image':image,'tested_release_sha256':exe,'gpu_suite':{'passed':tests,'log_sha256':hashfile(p/'gpu-tests.log'),'host_devices':[1,2],'effective_flags':base['serving']|config[2]},'starting_service':base}
(p/'detail.json').write_text(json.dumps(data,indent=2)+'\n')
