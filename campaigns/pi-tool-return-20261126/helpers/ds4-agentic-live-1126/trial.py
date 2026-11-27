import re, stat, hashlib, http.server, json, os, shutil, socketserver, subprocess as sp, sys, tempfile, threading, time
from pathlib import Path
from urllib.request import urlopen, Request
from setup import box, p, release, launcher

model='DeepSeek V4.1 Flash'
prompt='Fix the Go compose flag parser in this checkout. It should accept --flag=value as well as --flag value, quoted values with spaces and environment expansion. Match complete option tokens only, and ignore flag-like text inside another quoted value. Preserve the integer helper and the public behavior covered by the existing tests. Inspect the source, run the failing tests, edit the implementation, run the tests again and report the result. Leave the supplied tests intact.'
def digest(b):return hashlib.sha256(b).hexdigest()
class UnixServer(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
 daemon_threads=True
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):
  srv=self.server
  if self.path!='/v1/chat/completions' or len(srv.rows)>=16:
   self.send_error(400);return
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if body['model']!=model or not body.get('stream'):self.send_error(400);return
  body.update(temperature=0,stream_options={'include_usage':True})
  # Give each independent session a distinct prefix. Paired runs use the same ID.
  first_message=body['messages'][0]
  prefix='Session '+srv.session_id+'\n'
  if isinstance(first_message['content'],str):first_message['content']=prefix+first_message['content']
  else:
   block=next(x for x in first_message['content'] if x.get('type')=='text')
   block['text']=prefix+block['text']
  def textcontent(m):
   content=m.get('content')
   return content if isinstance(content,str) else '\n'.join(x.get('text','') for x in (content or []) if isinstance(x,dict))
  if body['messages'][-1]['role']=='tool' and '--- FAIL: TestFlagValues' in textcontent(body['messages'][-1]):
   try:
    with (p/'frozen-red-turn.json').open('x') as snapshot:json.dump(body,snapshot,ensure_ascii=False)
   except FileExistsError:pass
  f=srv.work/'flags.go' 
  implementation_sha=digest(f.read_bytes()) if not f.is_symlink() and f.is_file() and f.stat().st_size<1000000 else None
  row={'max_tokens':body.get('max_tokens'),'template':body.get('chat_template_kwargs'),'reasoning_history_messages':sum(bool(m.get('reasoning_content')) for m in body['messages'] if m['role']=='assistant'),'implementation_sha':implementation_sha,'done':False,'request_sha256' :digest(json.dumps(body,sort_keys=True).encode()),'turn':len(srv.rows)+1}
  if row['turn']==1 and body.get('chat_template_kwargs',{}).get('reasoning_effort')=='high':
   normalized=body|{'chat_template_kwargs':body['chat_template_kwargs']|{'reasoning_effort':'max'}}
   row['same_request_at_max_sha256']=digest(json.dumps(normalized,sort_keys=True).encode())
  for message in reversed(body['messages']):
   if message['role']!='tool':break
   srv.observed[message['tool_call_id']]=(row['turn'],implementation_sha)
  srv.rows.append(row); start=time.perf_counter();first=None;last=None
  req=Request('http://127.0.0.1:8001/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+srv.key,'Content-Type':'application/json'})
  with urlopen(req,timeout=180) as upstream:
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   for line in upstream:
    if line.startswith(b'data:') and line[5:].strip()==b'[DONE]':row['done']=True
    if line.startswith(b'data:') and line[5:].strip()!=b'[DONE]':
     packet=json.loads(line[5:])
     if packet.get('usage'):row['usage']={k:packet['usage'].get(k,0) for k in ('completion_tokens','prompt_tokens')}
     for c in packet.get('choices',[]):
      delta=c.get('delta',{})
      if any(delta.get(k) for k in ('content','reasoning_content','tool_calls')):
       last=time.perf_counter()
       if first is None:first=last
      if c.get('finish_reason'):row['finish']=c['finish_reason']
    self.wfile.write(line);self.wfile.flush()
  row['wall_s']=time.perf_counter()-start
  if first is not None:row.update(first_event_s=first-start,last_event_s=last-start)
def copycache(dest):
 sp.run(['cp','-a','--reflink=auto',str(p/'cache'),str(dest)],check=True)
def task(key,instance,state,thinking_override=None):
 with tempfile.TemporaryDirectory(prefix='pi-task-',dir=p) as tmp:
  stage=Path(tmp);work=stage/'work';shutil.copytree(p/'fixture',work)
  copycache(stage/'cache');settings=stage/'pi';settings.mkdir()
  cfg,thinking,extension=launcher()
  if thinking_override is not None:thinking=thinking_override
  (stage/'launcher.mjs').write_bytes(extension)
  (settings/'settings.json').write_text('{"enableAutoUpdate":false}')
  socket=stage/'engine.sock'
  with UnixServer(str(socket),Handler) as server:
   server.key=key;server.rows=state['turns'];server.work=work;server.observed={};server.session_id=digest(str(instance).encode())[:16]
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   args=box()+['--bind',str(settings),'/pi','--setenv','PI_CODING_AGENT_DIR','/pi','--setenv','PI_TELEMETRY','0','--setenv','MODEL_CLI_PI_API_KEY','placeholder','--setenv','MODEL_CLI_PI_CONFIG',json.dumps(cfg),'--ro-bind',str(stage/'launcher.mjs'),'/launcher.mjs','--bind',str(work),'/work','--bind',str(stage/'cache'),'/cache','--ro-bind',str(p/'node'),'/node','--ro-bind',str(release),'/release','--ro-bind',str(socket),'/upstream.sock','--','/bin/sh','-c','socat TCP4-LISTEN:8180,bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:/upstream.sock & exec /node /release/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js "$@"','pi','--extension','/launcher.mjs','--provider',cfg['provider'],'--model',model,'--offline','--no-session','--no-mcp','--no-skills','--no-extensions','--no-context-files','--thinking',thinking,'--tools','read,bash,edit,write','--mode','json','--print',prompt]
   state['phase']='client'; started=time.perf_counter()
   try:r=sp.run(args,stdin=sp.DEVNULL,capture_output=True,timeout=240)
   finally:wall=time.perf_counter()-started;state['wall_s']=wall;server.shutdown();thread.join()
  state['phase']='transcript';events=[json.loads(l) for l in r.stdout.splitlines() if l.startswith(b'{')]
  tools=[];by_id={};final_finish=None
  for e in events:
   if e.get('type')=='tool_execution_start':
    a=e.get('args',{}); t={'name':e['toolName'],'arg_sha256':digest(json.dumps(a,sort_keys=True).encode()),'command':a.get('command','')[:500],'path':a.get('path','')}
    by_id[e['toolCallId']]=t;tools.append(t)
   if e.get('type')=='tool_execution_end':
    t=by_id[e['toolCallId']]; t['after_batch']=server.observed.get(e['toolCallId']); t['error']=e.get('isError',False)
    output='\n'.join(i.get('text','') for i in e.get('result',{}).get('content',[]) if isinstance(i,dict))
    t['result_sha256']=digest(output.encode())
    t['test_result']=('fail' if re.search(r'(?m)^--- FAIL: TestFlagValues',output)
                      else 'pass' if re.search(r'(?m)^ok\s+composeflags\s',output) else None) if re.search(r'\bgo test\b',t['command']) else None
   if e.get('type')=='message_end' and e.get('message',{}).get('role')=='assistant':final_finish=e['message'].get('stopReason')
  state['phase']='grader';grade=stage/'grade';grade.mkdir();size=0;files={}
  for f in work.rglob('*'):
   rel=f.relative_to(work);mode=f.lstat().st_mode
   if stat.S_ISDIR(mode):continue
   if not stat.S_ISREG(mode) or f.stat().st_size>1000000 or len(files)>=100:raise ValueError('unexpected fixture tree')
   data=f.read_bytes();size+=len(data)
   if size>4000000:raise ValueError('oversize fixture tree')
   dest=grade/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);files[str(rel)]=digest(data)
  unchanged=all(files.get(f)==digest((p/'fixture'/f).read_bytes()) for f in ('go.mod','flags_test.go'))
  if (grade/'zz_benchmark_hidden_test.go').exists():raise ValueError('grading filename conflict')
  shutil.copyfile(p/'fixture/flags_test.go',grade/'flags_test.go');shutil.copyfile(p/'fixture/go.mod',grade/'go.mod')
  shutil.copyfile(p/'hidden.go.txt',grade/'zz_benchmark_hidden_test.go');copycache(stage/'gradecache')
  g=sp.run(box()+['--ro-bind',str(grade),'/work','--bind',str(stage/'gradecache'),'/cache','--','go','test','-json','-count=1','./...'],capture_output=True,timeout=30)
  graded={x.get('Test') for l in g.stdout.splitlines() if l.startswith(b'{') and (x:=json.loads(l)).get('Action')=='pass'}
  correct=unchanged and g.returncode==0 and {'TestAdditionalCases','TestFlagValues'}<=graded
  original=digest((p/'fixture/flags.go').read_bytes());edited=files.get('flags.go')!=original
  inspected=lambda t: (t['name']=='read' and t['path'].endswith('flags.go') or re.search(r'(?:cat|sed|head|tail|rg)\b[^;&]*flags\.go',t['command'])  )
  # The messages submitted on each turn provide the actual file state after that turn's tool batch.
  failed_at=[t['after_batch'][0] for t in tools if t.get('test_result')=='fail' and t.get('after_batch') and t['after_batch'][1]==original]
  passed_at=[t['after_batch'][0] for t in tools if t.get('test_result')=='pass' and t.get('after_batch') and t['after_batch'][1]!=original]
  workflow=bool(failed_at and passed_at and min(failed_at)<max(passed_at) and any(inspected(t) and not t.get('error',True) and t.get('after_batch') and t['after_batch'][0]<=min(failed_at) for t in tools))
  complete=bool(server.rows) and all(t.get('done') and t.get('wall_s',0)>0 and t.get('finish') in ('stop','tool_calls','length') and min(t.get('usage',{}).get(k,0) for k in ('prompt_tokens','completion_tokens'))>0 for t in server.rows)
  return {'instance':instance,'wall_s':wall,'passed':correct and complete and r.returncode==0 and final_finish=='stop','graded_correct':correct,'workflow':workflow,'complete_streams':complete,'grader_passed_tests':sorted(str(k) for k in graded if k),'tree_sha256':digest(json.dumps(files,sort_keys=True).encode()),'patch_sha256':files.get('flags.go'),'tools':tools,'turns':server.rows,'client_finish':final_finish}
def one(key,instance,thinking_override=None):
 state={'instance':instance,'wall_s':None,'phase':'setup','turns':[]}
 try:return task(key,instance,state,thinking_override)
 except (Exception,TimeoutError) as e:
  return state|{'passed':False,'error':type(e).__name__,'reason':str(e)[:400] if not isinstance(e,sp.TimeoutExpired) else 'process deadline exceeded'}
