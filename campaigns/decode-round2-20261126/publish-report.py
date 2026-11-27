from pathlib import Path
import json
p=Path(__file__).resolve().parent;repo=Path('/home/benchmark/llm/ds4-flash-rs')
d=p/'head'
read=lambda path:json.loads(path.read_text())
def cols(name,kind,shape):
 legs=[read(p/name/(kind+l+'.json'))[shape]['median'] for l in ['off-a','on','off-b']]
 a,v,b=legs
 return f'{a:.1f} / **{v:.1f}** / {b:.1f} ({(2*v/(a+b)-1)*100:+.1f}%)'
def live(sh):return selected['c1_serving'][sh]['median']
selected=read(repo/'results/decode-round2-selected.json')
tag=selected['tag'];source='checkpoint-2026-11-26-c2-03-head'
r=(p/'report-base.md').read_text()
r+='''3. **Share vocabulary-head weights across verification rows.** The packed CUDA head uses two eight-row tiles for 9–16 verification rows and four for wider batches. The target keeps each row’s original tensor-core accumulation and reduction order. The full-size synthetic GPU comparison checked 233,867,520 values across 17 shapes up through 512 rows. Every candidate output matched the single-tile reference bit for bit. At 48 rows the projection took 1.26 ms with tiling and 2.63–2.67 ms in the controls. The 404k single-client continuation used the learned drafter and verified at most eight rows. Six concurrent continuations could share the head weights. [Experiment, serving comparisons and capacity](results/c1-head.json)

Each cell lists disabled / enabled / disabled medians in tok/s. The percentage uses the average of the two control medians. The shortlist and head experiments both started with longer repetitions enabled. The shortlist stayed off in the head experiment.

| Experiment | C1 edit at 25.6k | C1 coding at 404k |
|---|---:|---:|
'''
for name,label in [('cycle','Longer repetitions'),('markov','Markov shortlist'),('head','Head row tiling'),('head','Head row tiling, new starts')]:
 kind='repeat-' if label.endswith('starts') else ''
 r+=f"| {label} | {cols(name,kind,'edit')} | {cols(name,kind,'long')} |\n"
r+='''
## Selected service

The cumulative comparisons use the original deployed image on either side of the final candidate. The six-client comparisons change only the vocabulary head while keeping longer repetitions enabled.

| Workload | Control / selected / control, tok/s |
|---|---:|
'''
for desc,kind,k in [('C1 edit at 25.6k','original-','edit'),('C1 coding at 404k after the edit','original-','long'),('C1 coding at 404k, fresh long-only processes','fresh-','long'),('Six edits at 25.6k, aggregate','gate-','sixedit'),('Six coding sessions at 404k, aggregate','gate-','sixlong')]:
 r+=f'| {desc} | {cols("head",kind,k)} |\n'
rates=read(d/'heldout-summary.json')['by_leg']
r+='\nThe final head change was also tested on three short coding prompts with 512 output tokens. Their control / enabled / control medians in tok/s were '+', '.join(name+' '+' / '.join(f"{rates[leg][key]['median']:.1f}" for leg in ['off-a','on','off-b']) for key,name in [('game','Python Tetris'),('parser','C JSON parser'),('cache','Rust TTL cache')])+'. All matched their frozen hashes.\n'
r+=f'''
On the same workstation, the earlier six-client vLLM edit benchmark measured 485.0 aggregate tok/s with CPU experts disabled and 537.2–561.0 with them enabled. The selected Rust engine’s 506.4 tok/s is 5.7–9.7% below those CPU-enabled runs. Both stacks used the same edit requests and power limits. They used different drafters and could generate different continuations. [Earlier same-host comparison](EDIT-SCALING-EXPERIMENT.md#same-host-comparison)

Both retained candidates recovered every receipt at 266,174 and 520,362 prompt tokens and served three six-client waves filling six independent 524,288-token windows. Eight 523,776-input + 512-output requests returned the same outputs as serial controls with six resident and two swapped sessions. RAM and forced-disk tests each recorded eight reads and eight writes with zero I/O failures. During the first experiment, one disk trial counted nine requests. Its eight test-client outputs matched, and a clean eight-request rerun passed. The final head build passed all 90 release tests with both GPUs and the checkpoint.

On the selected service, six warm single-client requests per workload measured **{live('edit'):.1f} tok/s on edits** and **{live('long'):.1f} tok/s at 404k after edits**. The service reported six resident slots, 32 logical sessions, 48 GiB RAM swap and 128 GiB disk swap. [Live image, flags and throughput](results/decode-round2-selected.json)

## Build and rollback

The stable release tag is `stable-2026-11-26-c2-375w`. The local selected image is `{tag}`. To rebuild, check out `{source}` in a separate worktree, run `cargo build --release --locked`, then `docker build -t {tag} .` there. The rebuild with expanded test coverage matched every loadable section of the tested executable except its build ID. The ELF symbol string table also differed. Use the engine's main checkout for the serving commands.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-round2.yml up --no-build -d --wait
```

To restore the original 366 tok/s single-client profile, run the following command.

```sh
docker compose --env-file ../models/.env -f compose.yml -f compose.c1-optimized.yml up --no-build -d --wait
```

The experiment checkpoints are `checkpoint-2026-11-26-c2-01-repetition`, `checkpoint-2026-11-26-c2-02-markov` and `{source}`. The second tag preserves the rejected implementation. Raw waves, checksums, GPU microbenchmarks and measured settings are archived on this machine in `~/.cache/ds4-flash-rs/profiles/decode-round2-20261126/`. [Edit benchmark](tools/edit_scaling.py) · [Frozen coding-history reproduction](LONG-CONTEXT-BENCHMARK.md#reproduce-the-coding-workload)
'''
(repo/'DECODE-ROUND2.md').write_text(r)
f=repo/'README.md';s=f.read_text().replace('compose.c1-optimized.yml','compose.decode-round2.yml').replace('It uses verified output repetition for single-client drafting, tiled attention ranking, shared prefix restoration and LFU expert eviction. [Single-client performance and checkpoints](C1-DECODE-EXPERIMENTS.md)','It uses longer verified output repetition for single-client drafting and reuses vocabulary-head weights across verification rows. It retains tiled attention ranking, shared prefix restoration and LFU expert eviction. [Performance and checkpoints](DECODE-ROUND2.md)').replace('C1-DECODE-EXPERIMENTS.md#run-reproduce-or-roll-back','DECODE-ROUND2.md#build-and-rollback')
a=s.index('\n',s.index('| `DS4_C1_OUTPUT_CONSENSUS`'))
s=s[:a]+'''
| `DS4_EXTEND_REPETITION` | 1 in the selected overlay | Extend strongly supported single-client output proposals to two periods and up to 96 tokens. |
| `DS4_HEAD_ROW_TILES` | 1 in the selected overlay | Reuse vocabulary-head weights across multiple eight-row tiles during target verification. |'''+s[a:]
s=s.replace('DS4_BATCH_TOPK=1 DS4_TEST_DEVICE','DS4_BATCH_TOPK=1 DS4_EXTEND_REPETITION=1 DS4_HEAD_ROW_TILES=1 DS4_EXPERT_EVICTION=lfu DS4_TEST_DEVICE')
f.write_text(s.replace('1 in the single-client overlay | Use repeated output','1 in the selected overlay | Use repeated output'))
f=repo/'PLAN.md'
with f.open('a') as out:out.write(f'\n## Second decode round at 375 W, 2026-11-26\n\nLonger repetition proposals and vocabulary-head row tiling reached {live("edit"):.1f} tok/s on the deployed single-client edit. The Markov shortlist made the draft scan faster but reduced accepted tokens per verification round. Six concurrent 404k coding sessions reached {read(d/"gate-on.json")["sixlong"]["median"]:.1f} tok/s aggregate with the wider head. [Measured controls, qualifications and three checkpoints](DECODE-ROUND2.md)\n')
