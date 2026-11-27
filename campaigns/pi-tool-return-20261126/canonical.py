"""Repeat an actual tool-result prompt from the same immutable on-disk prefill seed."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess as sp,threading
from pathlib import Path
from run_pair import b,flags,images,watch,restore,deployed,check_mount,p,response,compare
fixture=p/'capture-b'
def main(kind,labels,probes):
 donor=json.loads((fixture/'request-04.json').read_text())
 consumer=json.loads((fixture/'request-05.json').read_text())
 expected=json.loads((fixture/'request-06.json').read_text()); source=json.loads((fixture/'task.json').read_text())
 assert source['graded_correct'] and source['isolated']
 with open('/tmp/ds4-native-bench.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX); b.verify('native',flags(1),images[1]);stop=threading.Event();t=threading.Thread(target=watch,args=(stop,),daemon=True);t.start()
  try:
   for label in labels:
    on='on' in label
    root=p/('kv-'+label);root.mkdir(mode=0o700)
    opt={'DS4_PREFIX_RAM_MIB':'0'};service={'environment':opt}
    if kind=='width' and on:
     service['command']=['serve','--bind','0.0.0.0:8001','--max-context','524288','--chunk-size','128']
    if kind=='engram':opt['DS4_OVERLAP_PREFILL_ENGRAM']=str(int(on))
    if kind=='router':
     service['image']='ds4-flash-rs:pi-bound-router-1126'
     opt['DS4_BOUNDED_PREFILL_ROUTER']=str(int(on))
     opt['DS4_IDLE_TOOL_PREFILL_MAX_TOKENS']='0'

    override=p/(label+'.override.json');override.write_text(json.dumps({'services':{b.svc:service}}))
    b.idle('native');os.environ['DS4_HOST_PREFIX_DIR']=str(root)
    try:b.up('pi-canonical-'+label,deployed+['-f',str(override)])
    finally:os.environ.pop('DS4_HOST_PREFIX_DIR')
    check_mount(root);image=b.verify('native',flags(1)|opt,service.get('image',images[1]))
    chunk=128 if kind=='width' and on else 64
    assert json.loads(b.read('docker','inspect','--format','{{json .Config.Cmd}}',b.svc))[-1]==str(chunk)
    short=donor|{'max_tokens':8};response(short)
    seed=set(root.glob('*.kv'));assert seed;rows=[]
    for i in range(probes+1):
     since=b.read('date','-u','+%Y-%m-%dT%H:%M:%S.%NZ')
     r,out=response(consumer)
     assert r['input_sha']==source['turns'][4]['request_sha256'] and r['prompt']==source['turns'][4]['usage']['prompt_tokens']
     r['source_matches']=compare(out,expected);r['parity']=all(r['source_matches']) and r['finish']==source['turns'][4]['finish'] and r['tokens']==source['turns'][4]['usage']['completion_tokens']
     r['chunk']=chunk;r['i']=i;r['image']=image;r['label']=label
     logs=sp.check_output(['docker','logs','--since',since,b.svc],stderr=sp.STDOUT,text=True)
     assert f"prefix tier=disk tokens={r['reuse']}" in logs and r['reuse']==(source['turns'][3]['usage']['prompt_tokens']-1)//chunk*chunk
     rows.append(r);(p/(label+'.json')).write_text(json.dumps(rows,indent=2))
     print(label,i,r['reuse'],round(r['ttft'],3),r['source_matches'],flush=True)
     # Only this benchmark's private disk cache is mounted; its RAM tier is off.
     # Keep all producer checkpoints, remove every checkpoint the consumer added.
     b.idle('native');check_mount(root)
     for path in set(root.glob('*.kv'))-seed:path.unlink()
     assert set(root.glob('*.kv'))==seed
    b.verify('native',flags(1)|opt,service.get('image',images[1]))
   if any(not all(r['parity'] for r in json.loads((p/(label+'.json')).read_text())) for label in labels):
    raise RuntimeError('captured output parity failed; reject this configuration')
  finally:stop.set();t.join();restore()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('kind',choices=['width','engram','router']);a.add_argument('labels',nargs='+');a.add_argument('--probes',type=int,default=6);args=a.parse_args();main(args.kind,args.labels,args.probes)
