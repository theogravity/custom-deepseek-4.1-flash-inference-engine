"""Run a coding-to-full-window transition on the selected image, then switch the normal profile."""
import sys,os,json,fcntl,subprocess as sp,statistics
from pathlib import Path
root=Path('/home/benchmark/llm/ds4-flash-rs');os.chdir(root);sys.path.insert(0,'/tmp/ds4-edit-gap')
import run as b
import importlib.util
oldpath=Path('/tmp/ds4-prefix-sharing')
spec=importlib.util.spec_from_file_location('old_native',oldpath/'run.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
editmod=sys.modules['run'];sys.modules['run']=o
spec=importlib.util.spec_from_file_location('old_checks',oldpath/'qualify.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);sys.modules['run']=editmod
p=Path('/tmp/ds4-cpu-eval/full-window-confirm');p.mkdir(exist_ok=True);q.p=p
ref=json.loads((root/'results/long-context-tuning-375w.json').read_text());image='ds4-flash-rs:attention-topk-checkpoint'
with open('/tmp/ds4-native-bench.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 b.verify('native',{'DS4_EXPERT_EVICTION':'lfu','DS4_BATCH_TOPK':'1'},image)
 try:
  for suffix,option in [('off-a','0'),('on','1'),('off-b','0')]:
   flags={'DS4_BATCH_TOPK':option}
   ident=b.start('full-recheck-'+suffix,'native',image,flags)
   code=q.shell(suffix+'-code',['tools/long_decode.py','--histories',o.b.frozen,'--clients','6','--output','512','--rounds','3','--sample-steady'])
   assert code[-1]['prompts']==ref['code_request_proofs']
   assert len(code)==5 and all([r['sha256'] for r in w['rows']]==ref['coding']['qualified-six512']['output_sha256'] for w in code[:-1])
   full=q.shell(suffix+'-full',['/tmp/ds4-eviction-1126/six_streams_idle.py','--input','524160','--output','128','--continue-histories',o.b.frozen,'--rounds','7'])
   assert len(full)==7
   for i,w in enumerate(full):
    a,z=w['metrics_before'],w['metrics_after']
    assert w['all_valid'] and w['concurrent'] and w['six_target_passes_delta']>0 and a['rotations']==z['rotations']
    assert [r['sha256'] for r in w['rows']]==ref['c6_full_retrieval']['ram-six']['output_sha256']
    assert [r['extended_ids_sha256'] for r in w['rows']]==ref['c6_full_retrieval']['ram-six']['prompt_ids_sha256']
    if i:assert z['prefix_reused_tokens']-a['prefix_reused_tokens']==6*524096
   active=b.verify('native',flags|{'DS4_EXPERT_EVICTION':'lfu'},image);assert active==ident
   out=dict(image=ident,tag=image,flags=flags,c6_404k_512=[w['aggregate_decode_tok_s'] for w in code[1:-1]],six_full_matched=42,c6_524160_receipt_128=[w['aggregate_decode_tok_s'] for w in full[1:]],logical_limit=b.metric('native').get('max_live_seqs'),full_min=json.loads((p/(suffix+'-full-memory.json')).read_text()),code_min=json.loads((p/(suffix+'-code-memory.json')).read_text()))
   (p/(suffix+'-validation.json')).write_text(json.dumps(out,indent=2)+'\n')
 finally:
  b.idle('native');b.up('restore-selected-full-recheck',['-f','compose.yml','-f','compose.decode-optimized.yml'])
  b.verify('native',{'DS4_EXPERT_EVICTION':'lfu','DS4_BATCH_TOPK':'1'},image)
