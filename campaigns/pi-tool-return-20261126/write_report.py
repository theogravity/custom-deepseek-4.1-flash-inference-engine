from pathlib import Path
import json, statistics
from analyze import summarize
p=Path(__file__).resolve().parent
dest=p/'idle'
result=summarize()
qual=json.loads((p/'qualification/passed.json').read_text())
assert result['idle_image']==qual['image']
result['qualification']=qual
# Audit counts independently of the quality gates inside the existing clients.
for tier in ['ram','disk']:
    w=json.loads((p/f'qualification/eight-{tier}.jsonl').read_text())
    a,z=w['metrics_before'],w['metrics_after']
    delta=w['swap_delta'];other='disk' if tier=='ram' else 'ram'
    assert z['requests_started']-a['requests_started']==8
    assert delta[tier+'_reads']>0 and delta[tier+'_writes']>0 and delta[other+'_reads']==delta[other+'_writes']==0
    assert delta['read_failures']==delta['write_failures']==0
    result['qualification']['eight-'+tier]['io']=delta
for line in (p/'qualification/six-full.jsonl').read_text().splitlines():
    w=json.loads(line);assert w['metrics_after']['requests_started']-w['metrics_before']['requests_started']==6
result['build']=json.loads((p/'provenance.json').read_text())
if (p/'router-decision.json').exists():
    result['bounded_router']=json.loads((p/'router-decision.json').read_text())
(dest/'results/pi-tool-return.json').write_text(json.dumps(result,indent=2)+'\n')
metric=result['idle_ab_a']
rows=[]
for g in ('0','2','6'):
    a,n,b=[metric[k][g] for k in ('a','on','b')]
    times=' | '.join(f"{v['median_ttft']:.3f} s" for v in [a,n,b])
    reuse=', '.join(str(r['reuse']) for r in n['samples'])
    rows.append(f"| {'Immediate' if g=='0' else g+' seconds'} | {times} | {reuse} |")
iso=result['canonical_only']
def triples(labels):
    return ' / '.join(f"{iso[k]['ttft']:.3f}" for k in labels)
qual= result['qualification']
gaps=[r['gap_since_previous_s'] for arm in metric.values() for r in arm['0']['samples']]
md=f'''# Time to first output after a Pi tool call

A native DeepSeek V4.1 Flash server can prepare the previous assistant turn while a coding tool runs. The benchmark used a real Pi edit followed by its real tool result. With a simulated six-second interval for the tool, the next reasoning, text or tool event arrived in **{metric['on']['6']['median_ttft']:.2f} s**. With an immediate return it took **{metric['on']['0']['median_ttft']:.2f} s**. The captured Pi session fixed a Go flag parser and passed the public and held-out tests. The replay covers three consecutive model calls from that fix. [Measurements](results/pi-tool-return.json)

## First-event latency at one client

| Tool duration | Disabled, before | Enabled | Disabled, after | Enabled RAM prefix, tokens in each run |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Each cell is the median of two independent full-exchange replays. All 18 started with a fresh server, a private copy of the same producer-only cache and the same eight-token warm-up. Each generated the complete 4,103-token assistant turn from 2,780 input tokens before submitting the 6,908-token tool-result prompt. The next reply issued the recorded shell call in 62 tokens. The third prompt had 6,993 tokens and its reply had 542. All 54 completed responses matched the captured reasoning, visible text, ordered tool names and arguments, output length and finish reason. The baseline reused 2,752 tokens for the tool result in every replay.

Timing starts at the tool-result POST and includes the first reasoning, text or tool-call delta. Pi's real edit produced a **4.49 s** wait on the previous service. The controlled tool intervals begin at the end of the producer response. The immediate client took {min(gaps)*1000:.1f}–{max(gaps)*1000:.1f} ms to submit. The two- and six-second intervals simulate tools such as running tests. Their own runtime is additional to the first-event latency in the table. This workload used Pi 1.4.0 at `high`, greedy decoding, a 64,000-token output limit, two RTX PRO 6000s at 375 W each and a 225 W CPU socket cap. The earlier 12.0 s observation followed an 11,421-token turn whose exact request body was unavailable. [Original Pi experiment](AGENTIC-CODING.md)

## How serving works

When a completed model response emits a tool call after at least 512 generated tokens, the worker retains the token IDs and releases the decode state. This happens before the external tool reports success or failure. If the next request has yet to arrive after 200 ms, it restores the ordinary prompt prefix and runs the same 64-token prefill chunks used by incoming requests. It checks for queued work between chunks. The engine stores up to eight checkpoints in the existing RAM prefix cache and matches the actual next prompt by token prefix. The target KV takes priority when there is too little RAM for its drafter sidecar.

The serving profile enables this work for completed histories of at most 32,768 tokens. Larger histories follow ordinary foreground prefill. Idle prefill can consume GPU cycles even if the next request never reuses the result, and inserting checkpoints can evict older RAM prefixes. The metric `idle_tool_prefill_tokens` counts tokens processed in the background. Foreground reuse is recorded separately in `prefix_reused_tokens`. [Scheduler](src/api/scheduler.rs) · [Model](src/model.rs)

## Other experiments

Reusing generated KV directly brought the first event down from 4.60 s to 0.34 s (return control 4.28 s). The second response generated 59 tokens instead of 62 and the following response issued a different shell call. This experiment is preserved at `checkpoint-2026-11-26-pi-decode-kv`. The serving profile uses canonical prefill.

Using 128-token prefill chunks improved isolated disk-seeded median latency from {iso['width-off-a']['ttft']:.3f} to {iso['width-on']['ttft']:.3f} s (return control {iso['width-off-b']['ttft']:.3f} s). When the whole exchange was regenerated, the producer made a different edit. Overlapping Engram gathering measured {triples(['gather-off-a','gather-on','gather-off-b'])} s in the same disabled/enabled/disabled design. Each isolated leg used one warm-up and six timed requests. [Output and timing checks](results/pi-tool-return.json)
'''
if 'bound-on' in iso:
    md+=f"""
The router experiment used a cheap upper bound to skip expert scores that could never enter the top six. Disabled/enabled/disabled medians on the isolated return were {triples(['bound-off-a','bound-on','bound-off-b'])} s. The full request, reasoning, text and ordered tool arguments matched in all 21 probes. {result['bounded_router']['description']} The code is preserved at `checkpoint-2026-11-26-pi-router-bound`.
"""
md+=f'''
## Capacity and deployment

The selected image passed all {qual['gpu-tests']} release tests, including the CUDA and real-checkpoint cases. It recovered the reference needles at {qual['needles'][0]:,} and {qual['needles'][1]:,} input tokens and completed three six-client waves at 524,160 input + 128 output tokens per client. Eight clients at 523,776 input + 512 output matched serial controls on both RAM and forced-disk rotation with six resident and two swapped. The one-client spot checks measured {qual['short-edit']['median']:.1f} tok/s on the 25,625-token structured edit and {qual['long-c1']['median']:.1f} tok/s on 404,151-token coding. Each decode rate is the median of six warm requests. [Full-window and decode checks](results/pi-tool-return.json)

To start the selected profile from the engine checkout, run

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-round2.yml -f compose.agentic.yml up --no-build -d --wait
```

To roll back, use the same command with just `compose.yml` and `compose.decode-round2.yml`. To test the same binary with idle prefill disabled, export `DS4_IDLE_TOOL_PREFILL_MAX_TOKENS=0` before starting the three-file profile. To rebuild the local image in a separate worktree, run the following commands from the engine checkout.

```sh
git worktree add --detach ../ds4-agentic-build checkpoint-2026-11-26-pi-idle-prefill
(cd ../ds4-agentic-build && CARGO_TARGET_DIR=target cargo build --release --locked && docker build -t ds4-flash-rs:agentic-idle-20261126 .)
```

Use the release checkout for the three-file Compose command. The published runtime image and a rebuild from the source checkpoint agreed on every loadable section except any generated build ID. The local raw replay receipts, captured benchmark request bodies and validation scripts are under `~/.cache/ds4-flash-rs/profiles/pi-tool-return-20261126/`.
'''
(dest/'TOOL-RETURN.md').write_text(md)
