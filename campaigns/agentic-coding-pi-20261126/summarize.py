import hashlib,json,statistics as st
from pathlib import Path
p=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
def records(label):
 path=p/(label+'.jsonl')
 if not path.exists():return []
 return [json.loads(l) for l in path.read_text().splitlines()]
def rate_task(r):
 return sum(t['usage']['completion_tokens'] for t in r['turns'])/sum(t['wall_s'] for t in r['turns'])
def first_green_bounds(r):
 # Pi submits requests sequentially. The time spent in all other work includes
 # process startup, tools and delivery. Place all of it before or after green
 # to bound the first observed passing public test on the final parser.
 if not r.get('complete_streams') or r.get('graded_correct') is not True:
  return None
 final=r['patch_sha256']
 green=[t['after_batch'][0] for t in r['tools'] if t.get('test_result')=='pass' and t.get('after_batch') and t['after_batch'][1]==final]
 if not green:return None
 j=min(green)-1; turns=r['turns']
 return [sum(t['wall_s'] for t in turns[:j]),r['wall_s']-sum(t['wall_s'] for t in turns[j:])]
def brief(r):
 turns=r.get('turns',[]);api=[t for t in turns if t.get('wall_s') and t.get('usage')]
 return {'instance':r['instance'],'attempt_wall_s':r.get('wall_s'),'passed':r['passed'],'graded_correct':r.get('graded_correct'),'first_observed_public_green_wall_bounds_s':first_green_bounds(r),'workflow_observed':r.get('workflow'),'finish':r.get('client_finish',r.get('phase')),'error':r.get('error'),'excluded':r.get('excluded'),'requests':len(turns),'tools':len(r['tools']) if 'tools' in r else None,'patch_sha256':r.get('patch_sha256'),'engine':r['engine'],'first_request_sha256':turns[0]['request_sha256'] if turns else None,'turns_with_final_usage_and_time':len(api),'partial_or_missing_turns':len(turns)-len(api),'completed_turn_output_tokens':sum(t['usage']['completion_tokens'] for t in api),'generated_tokens':sum(t['usage']['completion_tokens'] for t in api) if r.get('complete_streams') else None,'prompt_range':([min(t['usage']['prompt_tokens'] for t in api),max(t['usage']['prompt_tokens'] for t in api)] if api else []),'completed_turn_upstream_s':sum(t['wall_s'] for t in api),'upstream_s':sum(t['wall_s'] for t in api) if r.get('complete_streams') else None,'api_tokens_per_s':rate_task(r) if r.get('complete_streams') else None,'median_turn_first_event_s':st.median(t['first_event_s'] for t in api if 'first_event_s' in t) if any('first_event_s' in t for t in api) else None}
def summarize(label):
 raw=records(label);rows=list(map(brief,raw[1:]));eligible=[v for v in rows if not v['excluded']];passes=[r for r in eligible if r['passed']]
 def med(rs,k):return st.median([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else None
 return {'attempted':len(rows),'eligible':len(eligible),'passes':len(passes),'confirmed_workflows':sum(bool(v['workflow_observed']) for v in eligible),'median_attempt_wall_s':med(eligible,'attempt_wall_s'),'median_success_wall_s':med(passes,'attempt_wall_s'),'median_success_tokens':med(passes,'generated_tokens'),'median_success_requests':med(passes,'requests'),'median_success_tools':med(passes,'tools'),'median_success_api_tokens_per_s':med(passes,'api_tokens_per_s'),'samples':rows}
if __name__=='__main__':
 for label in ('before','selected','after'):
  if records(label):print(label,{k:v for k,v in summarize(label).items() if k!='samples'})
def wire_settings(runs,effort='max'):
 # A configured maximum and an encoder budget are separate observations.
 turns=[t for r in runs for t in r.get('turns',[])]
 observed={(t['max_tokens'],json.dumps(t.get('template'),sort_keys=True)) for t in turns}
 if not turns or any(cap is None or not isinstance(cap,int) or cap<=0 for cap,template in observed):
  raise ValueError('missing observed wire budget')
 if len(observed)!=1:raise ValueError('provider settings changed across turns: '+str(observed))
 cap,template=observed.pop();template=json.loads(template)
 if not isinstance(template,dict) or template.get('thinking') is not True or not template.get('reasoning_effort'):
  raise ValueError('missing observed reasoning configuration')
 if cap!=64000 or template['reasoning_effort']!=effort: raise ValueError('unexpected wire cap or effort for the frozen launcher')
 return {'turns_checked':len(turns),'wire_max_tokens':cap,'wire_reasoning_effort':template['reasoning_effort'],'thinking':template['thinking']}
