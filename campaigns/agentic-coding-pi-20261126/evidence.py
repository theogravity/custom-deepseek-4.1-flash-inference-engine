from pathlib import Path
from collections import Counter
import json, hashlib, statistics as st, subprocess as sp
from summarize import p, records, brief, summarize, wire_settings
from setup import launcher
c,thinking,extension=launcher()
labels=['before','selected','after'];results={};all_rows=[]
for label in [*labels,'selected-high']:
 rows=records(label)
 assert len(rows)==3 and [r['instance'] for r in rows]==list(range(3)) and [r['warmup'] for r in rows]==[True,False,False]
 assert all(not r.get('excluded') and r['engine']['requests_started']==len(r.get('turns',[])) and r['engine']['rotations']==0 for r in rows)
 assert all(r['image']==rows[0]['image'] and r['settings']==rows[0]['settings'] for r in rows)
 data=summarize(label);timed=rows[1:]
 # Keep the durations behind the public-test bounds auditable without saving conversation text.
 data['turns_by_instance']=[[{'number':t['turn'],'finish':t.get('finish'),'stream_completed':bool(t.get('done') and t.get('wall_s')),'prompt_tokens':t.get('usage',{}).get('prompt_tokens'),'output_tokens':t.get('usage',{}).get('completion_tokens'),'wall_s':t.get('wall_s'),'first_event_s':t.get('first_event_s'),'last_event_s':t.get('last_event_s'),'request_sha256':t['request_sha256']} for t in r['turns']] for r in rows]
 data['public_test_observations_by_instance']=[[{'result':t['test_result'],'next_request':t['after_batch'][0],'parser_sha256':t['after_batch'][1]} for t in r.get('tools',[]) if t.get('test_result') and t.get('after_batch')] if 'tools' in r else None for r in rows]
 ok=[r for r in timed if r['passed']]
 data.update(image=rows[0]['image'],flags=rows[0]['settings'],raw_sha256=hashlib.sha256((p/(label+'.jsonl')).read_bytes()).hexdigest(),wire=wire_settings(rows,effort='high' if label=='selected-high' else 'max'),warmup=brief(rows[0]))
 if ok:
  data['median_success_api_s']=st.median(sum(t['wall_s'] for t in r['turns']) for r in ok)
  data['median_success_api_fraction']=st.median(sum(t['wall_s'] for t in r['turns'])/r['wall_s'] for r in ok)
  data['median_success_reused_prompt_fraction']=st.median(r['engine']['prefix_reused_tokens']/sum(t['usage']['prompt_tokens'] for t in r['turns']) for r in ok)
  data['success_prompt_tokens_range']=[min(t['usage']['prompt_tokens'] for r in ok for t in r['turns']),max(t['usage']['prompt_tokens'] for r in ok for t in r['turns'])]
 data['all_three_launches']={'started':len(rows),'passed':sum(r['passed'] for r in rows),'graded_passes':sum(r.get('graded_correct') is True for r in rows),'graded_failures':sum(r.get('graded_correct') is False for r in rows),'ungraded':sum(r.get('graded_correct') is None for r in rows),'timed_out':sum(r.get('error')=='TimeoutExpired' for r in rows),'wall_s':[r['wall_s'] for r in rows],'tasks_with_confirmed_workflow':sum(bool(r.get('workflow')) for r in rows)}
 data['failure_types']=dict(Counter(r.get('error') or ('grader' if r.get('client_finish')=='stop' else r.get('client_finish')) for r in timed if not r['passed']))
 results[label]=data
 if label!='selected-high':all_rows.extend(rows)
assert results['before']['image']==results['after']['image']!=results['selected']['image']
pairs=[]
for i in range(3):
 group=[records(label)[i] for label in labels]
 pairs.append({'instance':i,'all_pass':all(r['passed'] for r in group),'matching_initial_request':len({r['turns'][0]['request_sha256'] for r in group if r.get('turns')})==1 and all(r.get('turns') for r in group),'same_final_tree':all(r.get('tree_sha256') for r in group) and len({r.get('tree_sha256') for r in group})==1,'wall_s':[r.get('wall_s') for r in group],'generated_tokens':[brief(r)['generated_tokens'] for r in group]})
doc={'hardware':'2 x RTX PRO 6000 Blackwell at 375 W, CPU socket at 225 W','task':'Single-client actual Pi session debugging an isolated extract of model-cli Go compose flag parsing. One historical task repeated with three paired session IDs. Public failing test and independent held-out tests.','source':{'repository':'local LLM setup repository','revision':'dfcefbfa3b279682038810d7563b23dc280d345a','source_path':'model-cli/internal/compose/compose.go','fixture_files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted([*(p/'fixture').iterdir(),p/'hidden.go.txt'])}},'client':{'name':'Pi','version':'1.4.0','runtime':sp.check_output([p/'node','--version'],text=True).strip(),'provider':'model-cli pi dsv41-flash-exl3 --direct','thinking_selection':thinking,'advertised_context':c['connection']['models'][0]['contextWindow'],'advertised_output':c['connection']['models'][0]['maxTokens'],'provider_extension_sha256':hashlib.sha256(extension).hexdigest(),'temperature':0,'pi_wall_limit_s':240,'maximum_model_turns':16,'schema_tools':['read','bash','edit','write'],'wire':wire_settings(all_rows)},'method':'Fresh server, private prefix cache and one fresh launch followed by two warm launches per leg. The full task samples are descriptive. Go compilation cache prewarmed and privately copied for every agent and grader. Pi wall includes model requests and local tools and excludes independent grading. API output rate includes prefill and delivery. Engine request counts must exclude other callers.','legs':results,'paired_sessions':pairs}
frozen=json.loads((p/'frozen-red-turn.json').read_text())
assert frozen['max_tokens']==64000 and frozen['messages'][-1]['role']=='tool' and '--- FAIL: TestFlagValues' in json.dumps(frozen['messages'][-1]['content'])
original_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
sources=[{'leg':r['leg'],'instance':r['instance'],'turn':t['turn']} for r in all_rows for t in r['turns'] if t['request_sha256']==original_hash]
assert sources
original_wire_hash=hashlib.sha256(json.dumps(frozen).encode()).hexdigest()
source_turns=[t for r in all_rows for t in r['turns'] if t['request_sha256']==original_hash]
original_prompt_counts={t['usage']['prompt_tokens'] for t in source_turns if t.get('usage')}
assert len(original_prompt_counts)==1
frozen['max_tokens']=256
replay_hash=hashlib.sha256(json.dumps(frozen).encode()).hexdigest()
slices={}
for lab in ('slice-before','slice-selected','slice-after'):
 raw=records(lab);assert len(raw)==7 and [r['i'] for r in raw]==list(range(7))
 rr=raw[1:];slices[lab]={'image':raw[0]['image'],'median_decode_tok_s':st.median(r['decode_tok_s'] for r in rr),'median_api_tok_s':st.median(r['api_tok_s'] for r in rr),'median_ttft_s':st.median(r['ttft_s'] for r in rr),'raw_sha256':hashlib.sha256((p/(lab+'.jsonl')).read_bytes()).hexdigest(),'measurements':raw}
for lab,idx in zip(('slice-before','slice-selected','slice-after'),labels):
 assert slices[lab]['image']==results[idx]['image']
 assert all(r['image']==results[idx]['image'] and r['settings']==results[idx]['flags'] for r in slices[lab]['measurements'])
assert len({r['prefix_reused_tokens'] for d in slices.values() for r in d['measurements'][1:]})==1
assert all(r['prefix_reused_tokens']>0 for d in slices.values() for r in d['measurements'][1:])
flat=[v for d in slices.values() for v in d['measurements']]
assert all(v['request_sha256']==replay_hash and v['hashes']==flat[0]['hashes'] and v['prompt_tokens']==flat[0]['prompt_tokens'] and v['output_tokens']==256 for v in flat)
assert flat[0]['prompt_tokens'] in original_prompt_counts
doc['frozen_turn_replay']={'original_canonical_sha256':original_hash,'original_wire_sha256':original_wire_hash,'captured_from':sources,'method':'Identical request captured after a real failing test, capped at 256 tokens. Each leg used one warm-up and six warm responses. All output hashes, prompt length and measured prefix reuse matched across legs. Streaming decode uses 255 first-to-last intervals with no bundled tool calls.','request_sha256':flat[0]['request_sha256'],'prompt_tokens':flat[0]['prompt_tokens'],'legs':slices}
high=results.pop('selected-high')
assert high['image']==results['selected']['image'] and high['flags']==results['selected']['flags']
initial_pairs=[r['turns'][0]['same_request_at_max_sha256']==s['turns'][0]['request_sha256'] for r,s in zip(records('selected-high'),records('selected'))]
assert initial_pairs==[True,True,True]
doc['pi_high_effort_trial']={'paired_initial_requests_match_after_effort_normalization':initial_pairs,'method':'After the primary sweep, repeat the same three session identifiers, cold first then two warm launches, on the same selected engine image. Pi selects high, keeping the original 64k output limit, prompt, temperature, fixture, deadlines and grader. Only the reasoning setting was changed in the launcher invocation. Secondary task-level observations on one fixture.','selected_high':high}
audit=json.loads((p/'prompt-audit.json').read_text())
assert audit['original_wire_sha256']==original_wire_hash and audit['native_token_count']==flat[0]['prompt_tokens']
doc['frozen_turn_replay']['prompt_encoding_audit']=audit
target=Path('/home/benchmark/llm/ds4-flash-rs/results/agentic-coding-pi.json')
target.write_text(json.dumps(doc,indent=2)+'\n')
