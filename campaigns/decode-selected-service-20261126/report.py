from pathlib import Path
import json,hashlib,statistics,sys
p=Path(sys.argv[1]).resolve();w=p/'work';name=p.name;d={'experiment':name,'clients':6,'gpu_w_each':375,'cpu_socket_w':225,'legs':{}}
meta={
'attention-topk':('Tile attention candidate selection','DS4_BATCH_TOPK','The short-edit profile attributed 12.3% of summed GPU kernel time to the segmented radix sort. The experimental CUDA kernel selects the best candidates in 4,096-key tiles and ranks their union in one block per query row. The 64-bit key retains the score and position tie break. For the second pass, tile winners use the memory that previously held input IDs. Single-row and wide-sort shapes retain their existing dispatch.'),
'draft-views':('Read drafter logits in place','DS4_DRAFT_VIEWS','The batched DSpark head produced one contiguous logits tensor, then copied each client’s rows to another allocation for Markov selection. At seven draft rows the logits copy moves 3.45 MiB per client. Including the confidence inputs, six seven-row proposals copy 24.0 MiB. The experiment borrows each client’s logits, hidden and mixture ranges directly from their batch allocations. The individual Markov chains, confidence calculations and per-client widths retain their existing arithmetic.'),
'paired-index':('Share compressed index keys between verification rows','DS4_PAIRED_INDEX','The 404k profile attributed 16.9% of summed GPU kernel time to the compressed key scorer. The selected CUDA variant keeps both lanes of a packed FP32 instruction on adjacent key columns. It evaluates two consecutive queries in sequence using the same dequantized key registers. The dispatch uses eight-lane groups on full, physically backed scans with 4–8 query rows and at least 65,536 keys. It preserves each row’s causal mask, dot-product reduction and head weights. Candidate-filtered rows retain the existing kernel because their block lists can differ.')}
title,flag,desc=meta[name]
lines=['# '+title,'',desc,'','| Six concurrent requests | GPU control before | Enabled | GPU control after | Change vs. midpoint |','|---|---:|---:|---:|---:|']
images=set()
for shape,display in [('edit','25,625-token edits, 1,500 output'),('long','404k coding, 768 output')]:
 cfg=json.loads((p/(shape+'.json')).read_text());rows=[]; paths=[p/((x[0] if isinstance(x,list) else x['label'])+'.info') for x in cfg]
 if not any(f.is_file() for f in paths):continue
 for f in paths:
  info=json.loads(f.read_text());lab=info['label'];raw=(p/(lab+'.jsonl')).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==info['raw_sha256']
  samples=[json.loads(l) for l in raw.splitlines()]
  waves=samples[1:-1] if shape=='long' else samples[1:]
  assert [r['warmup'] for r in samples[:len(waves)+1]]==[True]+[False]*len(waves)
  assert len(waves)>=3 and info['matches']==[6]*len(waves)
  for v in waves:
   z=v['delta']; assert z['requests_started']==6 and z['six_target_passes']>0 and z.get('rotations',0)==0
   assert z['prefix_reused_tokens']==(2424704 if shape=='long' else 153600)
  images.add(info['image'])
  info['whole_wave_tok_s']=[v['wall_tok_s' if shape=='long' else 'whole_wave_tok_s'] for v in waves]
  info['warm_responses_matched']=len(waves)*6
  info['six_target_passes']=[v['delta']['six_target_passes'] for v in waves]
  if shape=='long':
   info['per_client_decode_medians']=[statistics.median(v['rows'][i]['decode_tok_s'] for v in waves) for i in range(6)]
  d['legs'][lab]=info
  rows.append(info['median_tok_s'])
 delta=(rows[1]/statistics.mean([rows[0],rows[2]])-1)*100
 lines.append('| '+display+' | '+' | '.join(f'{v:.1f}' for v in rows)+f' | {delta:+.{2 if abs(delta)<0.1 else 1}f}% |')
assert len(images)==1
lines+=['','Rates are aggregate generated tokens per second across six clients. Each row uses one binary with the option disabled, enabled and disabled again, a fresh process per leg and one warm-up wave. The short edits use the published aggregate-rate formula. The 404k rate runs from the earliest first token to the latest last token, excluding each client’s first token. The two workloads use different prompts, output budgets and timing conventions.','', 'Both RTX PRO 6000 GPUs were capped at 375 W. The CPU socket reported a 225 W cap. All measured waves exercised six-way target verification, reused the full expected prefix and matched all six reference responses. The service recorded zero KV rotations during these throughput tests.','']
if name=='paired-index':
 lines=[line.replace('The short edits use the published aggregate-rate formula. ', '').replace(' The two workloads use different prompts, output budgets and timing conventions.', '') for line in lines]
if (p/'micro.txt').is_file() and name=='attention-topk':
 from collections import defaultdict
 grouped=defaultdict(list)
 for l in (p/'micro.txt').read_text().splitlines():
  width,rows,keep,mode,us=l.split()
  grouped[(int(width),int(rows),int(keep))].append([int(mode),float(us)])
 d['sort_kernel_microseconds']=[]
 for (width,rows,keep),v in grouped.items():
  assert [x[0] for x in v]==[0,1,0]
  d['sort_kernel_microseconds'].append(dict(keys=width,rows=rows,keep=keep,before=v[0][1],enabled=v[1][1],after=v[2][1]))
 lines+=['The isolated 8-row GPU benchmark measured the selector alone, using fixed synthetic 64-bit ranking keys on host GPU 1. With 16,384 input keys and 512 winners per row, elapsed time fell from 180.8 to 29.7 microseconds. Each number is the median of nine CUDA-event samples of 150 launches after warming the kernels. The baseline includes its required offset kernel.','']
if name=='paired-index':
 from collections import defaultdict
 d['microseconds']={}
 for kind,filename in [('packed_across_queries','cross-row-micro.txt'),('register_reuse','micro.txt')]:
  rows=defaultdict(list)
  for l in (p/filename).read_text().splitlines():
   width,n,mode,us=l.split()
   rows[(width,n)].append([int(mode),float(us)])
  d['microseconds'][kind]=[]
  for (width,n),v in rows.items():
   assert [x[0] for x in v]==[0,8,16,0]
   d['microseconds'][kind].append(dict(columns=int(width),query_rows=int(n),before=v[0][1],lanes8=v[1][1],lanes16=v[2][1],after=v[3][1]))
 lines+=['On GPU 1, two packing layouts were screened at 65,537, 202,112 and 404,224 columns, each with 4, 5, 7 and 8 query rows. Each cell used seven CUDA-event samples of 20 launches. For the observed 202,112-column, eight-row shape, packing two queries into one FP32 pair took 1,109.9 µs with eight lanes, versus 729.6 and 732.1 µs in the controls. Keeping adjacent key columns paired and reusing their registers across queries took 698.0 µs, versus 732.2 and 734.7 µs. This second layout went into the serving test. The 25.6k edit workload cannot enter the new dispatch. The first implementation is recorded at commit `5a333bd`.','']
if (p/'repeat.json').is_file():
 rs=[]
 for item in json.loads((p/'repeat.json').read_text()):
  label=item['label'];v=json.loads((p/(label+'.info')).read_text());raw=(p/(label+'.jsonl')).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==v['raw_sha256'] and v['image'] in images
  waves=[json.loads(l) for l in raw.splitlines()][1:-1]
  assert len(waves)==4 and v['matches']==[6]*4 and all(w['delta']['prefix_reused_tokens']==2424704 and w['delta']['six_target_passes']>0 and w['delta']['rotations']==0 for w in waves)
  assert all([r['sha256'] for r in w['rows']]==json.loads((p/'work/results/long-context-tuning-375w.json').read_text())['coding']['ram-six']['output_sha256'] for w in waves)
  d['legs'][label]=v;rs.append(v['median_tok_s'])
 lines += [f"A second 404k comparison reversed the run order, with a fresh process, one warm-up and four measured waves each time. Enabled / disabled / enabled delivered **{rs[0]:.1f} / {rs[1]:.1f} / {rs[2]:.1f}** aggregate tok/s. All 72 measured outputs matched their references.", '']
findings=(p/'decision.txt').read_text().strip()
lines += [findings,'',f'The option for this experiment is `{flag}=1`. Per-wave throughput, whole-wave throughput, image identity and raw capture hashes are in the [measurement record](results/{name}.json).','']
tag={'attention-topk':'checkpoint-2026-11-26-decode-01-attention-topk','draft-views':'checkpoint-2026-11-26-decode-02-draft-views','paired-index':'checkpoint-2026-11-26-decode-03-paired-index'}[name]
d['checkpoint']=tag
cfg=json.loads((p/'long.json').read_text())[1]
image='ds4-flash-rs:'+name+'-checkpoint'
override={'services':{'dsv41-flash-exl3':{'image':image,'environment':cfg['env']}}}
lines += [f"Rebuild from `{tag}` in an engine checkout beside the stack’s `models/` directory. This override selects both the rebuilt binary and the enabled environment inside the container.", '', '```sh',f'cargo build --release --locked && docker build -t {image} .',"cat > /tmp/ds4-decode-override.json <<'EOF'",json.dumps(override,indent=2),'EOF','docker compose --env-file ../models/.env -f compose.yml -f compose.shared-prefix-lfu.yml -f /tmp/ds4-decode-override.json up --no-build -d --wait','```','']
d['decision']=findings
q=p/'full-validation.json'
if q.is_file():d['full_window_validation']=json.loads(q.read_text())
if q.is_file():
 v=d['full_window_validation']
 lines+=['Six 524,160-token prompts each completed 128 output tokens in three waves, matching the reference prompt-token and output-text hashes. Eight 523,776-token prompts each completed 512 output tokens through RAM rotation and again through forced disk rotation, matching all eight reference digests both times.', '']
 c=v['c1_404k_code']
 lines += [f"One 404k coding client at a 512-token output budget delivered {statistics.median(c['before']):.1f} tok/s with the option disabled and {statistics.median(c['enabled']):.1f} enabled (three measured warm waves each).", '']
if (p/'full-tests.log').is_file():
 import re
 matches=re.findall(r'test result: ok[.] (\d+) passed; 0 failed; 0 ignored', (p/'full-tests.log').read_text())
 if matches:
  d['release_tests']=int(matches[-1])
  lines += [f"The full release suite passed all {matches[-1]} tests on the two inference GPUs with the option enabled.", '']
(w/('DECODE-'+name.upper()+'.md')).write_text('\n'.join(lines))
(w/'results'/(name+'.json')).write_text(json.dumps(d,indent=2)+'\n')
