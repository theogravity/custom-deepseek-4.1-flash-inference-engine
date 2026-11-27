import json,fcntl,sys,importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('h','/tmp/ds4-c1-1126/run.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
p=h.p/sys.argv[1];cfg=json.loads((p/'config.json').read_text())[1]
q=h.q;q.p=p;b=h.b;ref=h.gold
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',h.flags,b.native)
 (p/'full-validation.json').unlink(missing_ok=True)
 try:
  img=b.start('c1-qual-'+p.name,'native',cfg[1],cfg[2]);r={'image':img,'flags':cfg[2]}
  receipt_reference=[w for line in (h.root/'results/hybrid-context.jsonl').read_text().splitlines() if 'context' in (w:=json.loads(line))]
  z=q.shell('context-needles',['tools/context_gate.py','--seed',str(receipt_reference[0]['seed'])])
  receipts=[w for w in z if 'context' in w]
  assert len(receipts)==len(receipt_reference)==2
  assert all((v['context'],v['expected'],v['received'])==(ref['context'],ref['expected'],ref['received']) for v,ref in zip(receipts,receipt_reference,strict=True))
  r['retrieval']=[dict(context=w['context'],length=w['prompt_tokens'],exact=w['exact']) for w in z if 'exact' in w]
  assert len(r['retrieval'])==2 and all(w['exact'] for w in r['retrieval'])
  data=q.shell('six-full',['/tmp/ds4-eviction-1126/six_streams_idle.py','--input','524160','--output','128','--continue-histories',h.old.b.frozen,'--rounds','3'])
  gold=ref['c6_full_retrieval']['ram-six'];assert len(data)==3
  for i,w in enumerate(data):
   a,z=w['metrics_before'],w['metrics_after']
   assert w['all_valid'] and w['concurrent'] and w['six_target_passes_delta']>0 and a['rotations']==z['rotations']
   assert [x['sha256'] for x in w['rows']]==gold['output_sha256'] and [x['extended_ids_sha256'] for x in w['rows']]==gold['prompt_ids_sha256']
   if i:assert z['prefix_reused_tokens']-a['prefix_reused_tokens']==6*524096
  r['six_full_waves_matched']=len(data)
  for tier in ['ram','disk']:
   b.start('c1-qual-'+p.name+'-'+tier,'native',cfg[1],cfg[2]|({'DS4_SWAP_RAM_MIB':'0'} if tier=='disk' else {}))
   w=q.shell('eight-'+tier,['tools/kv_rotation.py','--agents','8','--input','523776','--output','512','--shared','--expect-tier',tier])[0]
   assert w['passed'] and w['peak_resident']==6 and w['peak_swapped']>=2
   assert [x['sha256'] for x in w['rows']]==[x['sha256'] for x in ref['eight_full_window']['rows']]
   r['eight_'+tier]={k:w[k] for k in ['passed','peak_resident','peak_swapped']}
  r['memory']={tag:json.loads((p/(tag+'-memory.json')).read_text()) for tag in ('context-needles','six-full','eight-ram','eight-disk')}
  (p/'full-validation.json').write_text(json.dumps(r,indent=2)+'\n')
 finally:b.restore();b.verify('native',h.flags,b.native)
