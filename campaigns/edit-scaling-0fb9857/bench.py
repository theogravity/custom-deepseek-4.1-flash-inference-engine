from pathlib import Path
from datetime import datetime, timezone
from statistics import median
import sys, os, json, subprocess, re, time, hashlib, fcntl
from urllib.request import Request, urlopen
p=Path(__file__).resolve().parent
root=Path.cwd()
svc='dsv41-flash-exl3'
docker=['docker','compose','--env-file','../models/.env','-f','compose.yml']
pack=json.loads((root/'results/serving-profile.json').read_text())
frozen='/tmp/ds4-six.pFsc06/404k-histories.json'
key=next(l.split('=',1)[1].strip().strip("'\042") for l in (root/'../models/.env').read_text().splitlines() if l.startswith('LITELLM_MASTER_KEY='))
runner=['uv','run','--offline','--no-project','--with','tokenizers==0.22.2','python']
def native(path):
 with urlopen(Request('http://127.0.0.1:8001'+path,headers={'Authorization':'Bearer '+key}),timeout=10) as r:return json.load(r)
def stamp(label):
 print(datetime.now().isoformat(timespec='seconds'),label,flush=True)
def read(*a):return subprocess.check_output(a,text=True).strip()
def command(label,args):
 with (p/(label+'.out')).open('w') as stdout,(p/(label+'.err')).open('w') as stderr:
  subprocess.run(args,stdout=stdout,stderr=stderr,check=True,env=dict(os.environ,DSV41_API_KEY=key))
def drain():
 for i in range(180):
  if not native('/metrics')['logical_active']:return
  time.sleep(1)
 raise RuntimeError('service still in use')
def profile():
 g=read('nvidia-smi','-i','1,2','--query-gpu=uuid,name,enforced.power.limit,memory.total','--format=csv,noheader,nounits').splitlines()
 assert len(g)==2 and all('RTX PRO 6000' in x and x.split(', ')[2]=='375.00' for x in g)
 h=next(d for d in Path('/sys/class/hwmon').glob('hwmon*') if (d/'name').read_text().strip()=='amd_hsmp_hwmon')
 cap=int((h/'power1_cap').read_text());assert cap==225000000
 c=json.loads(read('docker','inspect',svc))[0]
 assert c['State']['Health']['Status']=='healthy' and c['Image']==pack['image_id']
 vals=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
 return dict(gpus=g,cpu_cap_w=cap/1e6,image=c['Image'],env={k:v for k,v in vals.items() if k.startswith('DS4_')})
def start(label,override):
 drain()
 path=p/'override.json';path.write_text(json.dumps({'services':{svc:{'environment':override}}}))
 extra=['-f',str(path)] if override else []
 command(label+'-start',docker+extra+['up','--no-build','--force-recreate','-d','--wait','--wait-timeout','120'])
def run(label,override):
 stamp(label+' start');start(label,override)
 before=profile()
 for k,v in pack['environment'].items():
  if k.startswith('DS4_'):assert before['env'][k]==override.get(k,v)
 assert before['env'].get('DS4_SPEC_K')==override.get('DS4_SPEC_K')
 since=datetime.now(timezone.utc).isoformat()
 command(label,runner+['tools/long_decode.py','--histories',frozen,'--clients','6','--output','512','--sample-steady'])
 after=profile();assert before==after
 lines=[json.loads(l) for l in (p/(label+'.out')).read_text().splitlines()]
 rows=lines[:-1];assert len(rows)==7 and all(r['resident_decode'] for r in rows)
 logs=subprocess.check_output(['docker','logs','--since',since,svc],stderr=subprocess.STDOUT,text=True)
 counts=[(kind,int(r),int(pr),int(ac)) for kind,r,pr,ac in re.findall(r'speculative source=(learned|lookup) rounds=(\d+) proposed=(\d+) accepted=(\d+)',logs)]
 totals={kind:[sum(v[i] for v in counts if v[0]==kind) for i in [1,2,3]] for kind in ['learned','lookup']}
 m=rows[1:]
 summary=dict(profile=before,override=override,aggregate=lines[-1]['median_decode_tok_s'],wall=lines[-1]['median_wall_tok_s'],steady=median(x['resident_decode']['tok_s'] for x in m),first_spread=median(x['first_response_spread_s'] for x in m),output_hashes=[v['sha256'] for v in rows[0]['rows']],draft_totals_including_warmup=totals)
 (p/(label+'.json')).write_text(json.dumps(summary,indent=2));stamp(label+' complete '+str(round(summary['aggregate'],1)))
if __name__=='__main__':
 lock=open('/tmp/ds4-native-bench.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for name,settings in [('control-a',{}),('k5',{'DS4_SPEC_K':'5'}),('control-b',{}),('gate65',{'DS4_CONFIDENCE_THRESHOLD':'0.65'}),('control-c',{})]:
   run(name,settings)
 finally:
  drain();command('restore',docker+['up','--no-build','-d','--wait','--wait-timeout','120']);stamp('normal config restored')
