from pathlib import Path
import sys,os,json,fcntl,subprocess as sp,statistics,hashlib
root=Path('/home/benchmark/llm/ds4-flash-rs')
os.chdir(root)
sys.path.insert(0,'/tmp/ds4-edit-gap')
import run as b
import importlib.util
# Reuse the independent full-window workload and its established RAM/GPU watchdog.
folder=Path('/tmp/ds4-prefix-sharing')
spec=importlib.util.spec_from_file_location('old_native_run',folder/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
editrun=sys.modules['run'];sys.modules['run']=old
s=importlib.util.spec_from_file_location('full_checks',folder/'qualify.py')
qual=importlib.util.module_from_spec(s);s.loader.exec_module(qual);sys.modules['run']=editrun
p=Path(sys.argv[1]); name=p.name;cfg=json.loads((p/'long.json').read_text())[1]
qual.p=p
ref=json.loads((root/'results/long-context-tuning-375w.json').read_text())
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);b.verify('native',{'DS4_EXPERT_EVICTION':'lfu'},b.native)
 try:
  configs=json.loads((p/'long.json').read_text())
  result={'image':b.ident(cfg['img']),'flags':cfg['env'],'c1_404k_code':{}}
  for suffix,item in [('before',configs[0]),('enabled',configs[1])]:
   b.start('q-'+name+'-'+suffix,'native',cfg['img'],item['env'])
   one=qual.shell('one-'+suffix,['tools/long_decode.py','--histories',str(old.b.frozen),'--clients','1','--output','512','--rounds','3'])
   assert len(one)==5 and one[-1]['prompts']==ref['code_request_proofs'][:1]
   assert all([r['sha256'] for r in w['rows']]==ref['coding']['qualified-one']['output_sha256'] for w in one[:-1])
   assert all(w['delta']['requests_started']==1 and w['delta']['rotations']==0 and w['delta']['prefix_reused_tokens']>404000 for w in one[1:-1])
   result['c1_404k_code'][suffix]=[w['aggregate_decode_tok_s'] for w in one[1:-1]]
  full=qual.shell('full-window',['/tmp/ds4-eviction-1126/six_streams_idle.py','--input','524160','--output','128','--continue-histories',str(old.b.frozen),'--rounds','3'])
  gold=ref['c6_full_retrieval']['ram-six'];assert len(full)==3
  for i,w in enumerate(full):
   a,z=w['metrics_before'],w['metrics_after']
   assert w['all_valid'] and w['concurrent'] and w['six_target_passes_delta']>0 and a['rotations']==z['rotations']
   assert [r['sha256'] for r in w['rows']]==gold['output_sha256']
   assert [r['extended_ids_sha256'] for r in w['rows']]==gold['prompt_ids_sha256']
   if i:assert z['prefix_reused_tokens']-a['prefix_reused_tokens']==6*524096
  result['six_full_qualified_waves']=len(full)
  for tier in ['ram','disk']:
   env=cfg['env']|({'DS4_SWAP_RAM_MIB':'0'} if tier=='disk' else {})
   b.start('q-'+name+'-'+tier,'native',cfg['img'],env)
   w=qual.shell('eight-'+tier,['tools/kv_rotation.py','--agents','8','--input','523776','--output','512','--shared','--expect-tier',tier])[0]
   assert w['passed'] and w['peak_resident']==6 and w['peak_swapped']>=2
   assert [r['sha256'] for r in w['rows']]==[r['sha256'] for r in ref['eight_full_window']['rows']]
   result['eight_'+tier]=dict(peak_resident=w['peak_resident'],peak_swapped=w['peak_swapped'],matched=8)
  (p/'full-validation.json').write_text(json.dumps(result,indent=2))
 finally:b.restore()
