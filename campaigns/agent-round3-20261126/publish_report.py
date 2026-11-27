import math
from pathlib import Path
from statistics import median
import json,subprocess
p=Path(__file__).resolve().parent
out=p/'publication'
def load(f):return json.loads((p/f).read_text())
def tag(s):return subprocess.check_output(['git','-C',p/'text','rev-parse',s+'^{}'],text=True).strip()
def fmt(numbers,fn):return ' / '.join(fn(v) for v in numbers)
def sec(numbers):return fmt(numbers,lambda t:f'{1000*t:.1f}')
def rates(numbers):return fmt(numbers,lambda t:f'{t:.1f}')
def short_time(numbers):return fmt(numbers,lambda t:f'{t:.3f}')
e={name:load(name+'-receipts/summary.json') for name in ['text','markov','hash','release']}
changing=load('changing-receipts/summary.json')
qual=load('qualification/passed.json')
chosen=load('chosen.json')
phases={k:[median(load('hash-receipts/'+arm+'.json')['lookup_ms'][k]) for arm in ('a','on','b')] for k in ['target_lookup_read_ms','draft_lookup_read_ms']}
images={k:[load(k+'-receipts/'+arm+'.json')['image'] for arm in ('a','on','b')] for k in e}
assert qual['image']==images['release'][1] and qual['gpu-tests']==95
for k in e: assert images[k][0]==images[k][2]
experiments=['checkpoint-2026-11-26-agent3-01-text','checkpoint-2026-11-26-agent3-02-markov','checkpoint-2026-11-26-agent3-03-prefix']
mixed=load('mixed-receipts/summary.json')
assert set(mixed)=={'short-edit','long-c1'}
assert all(isinstance(v,list) and len(v)==3 and all(type(n) in (int,float) and math.isfinite(n) and n>0 for n in v) for v in mixed.values())
assert all(v[1]>=0.97*max(v[0],v[2]) for v in mixed.values())
rate=e['release'];m=e['markov'];hh=e['hash']
receipt={'date':'2026-11-26','power_caps_w':{'gpu1':375,'gpu2':375,'cpu':225},
  'build':load('release-provenance.json'),'variant_identity':load('variant-identity.json'),'checkpoints':{s:tag(s) for s in experiments},'tested_images':images,'release':chosen,'same_host_medians':e,
  'mixed_workload_control_candidate_control':mixed,'changing_history':changing,'hash_lookup_median_ms':phases,'full_validation':qual,
  'excluded_release_reason':'Earlier release results came from an unintended shared-target binary and were excluded.',
  'raw_archive':'~/.cache/ds4-flash-rs/profiles/agent-round3-20261126/'}
# The independent length-checked experiment is the source for all count claims.
receipt['same_host_medians']['text'].pop('count_tokens_median',None)
(out/'agent-round3.json').write_text(json.dumps(receipt,indent=2)+'\n')
kept=chosen['worktree']=='hash'
report=f"""# Coding-agent latency and decode on two RTX PRO 6000s

The native server reuses token IDs for ordinary text that appears unchanged in a later message. On the six timed follow-ups to a 404k-token conversation, the median wait for the first output fell to **{changing['ttft_s'][1]:.3f} s**. The two controls measured **{changing['ttft_s'][0]:.3f} s** and **{changing['ttft_s'][2]:.3f} s**. Each of the seven prompts, including warm-up, had a different final instruction. All 21 generated replies matched across runs. [Machine-readable results](results/agent-round3.json)

## Three experiments

| Change | Measurement | Control / candidate / return control | Decision |
|---|---|---:|---|
| Cache tokenized message text | First output on changing 404k history, seconds | {short_time(changing['ttft_s'])} | Enable |
| Interleave independent DSpark Markov scores | Captured Pi reasoning, decode tok/s | {rates(m['pi_median']['aba'])} | Keep original scorer |
| Compute KV keys from bulk token bytes | First output on repeated 404k coding request, ms | {sec(hh['long_ttft']['aba'])} | {'Enable' if kept else 'Keep original lookup'} |

**Tokenized text.** The server caches whole ordinary-text spans separately from chat control tokens. It uses an exact text match within the serving tokenizer. It stores up to 64 entries and 32 MiB of text and token-ID payload in host RAM. Text shorter than 512 bytes bypasses the cache. New or edited text still uses the original tokenizer. The cache is process-local and starts empty after a restart.

The 404k replay extended a frozen two-message history with seven different user instructions. Every native token count matched the independent checkpoint tokenizer, from 404,137 to 404,172 tokens. Each count request preceded generation on that body. Median count endpoint latency in milliseconds was {sec(changing['count_s'])}. The metric includes local HTTP, rendering and tokenization. Generation timing starts at its own POST. All measured arms restored the same number of KV tokens for each prompt and completed the same 16-token replies. An initial run was excluded because its first control restored 64 fewer KV tokens on two prompts. The reported run started after those disk entries existed for all arms.

**DSpark Markov head.** The CUDA kernel interleaved two independent dot products in one warp, keeping the original eight FMAs and reduction order for each vocabulary ID. Both paths passed a dedicated GPU check for chained predictions, ties across thread blocks, a partial final tile and nonfinite scores. The 404k decode rates were {rates(m['long_decode']['aba'])} tok/s. The Pi rate was within the variation between runs. Its code is preserved on the second experiment tag.

**KV prefix keys.** The Rust implementation updates SHA-256 in blocks and computes only the terminal digest for a saved target or drafter checkpoint. It tries the final RAM key first and falls back to the existing longest-prefix search. Target lookup and read measured {fmt(phases['target_lookup_read_ms'], lambda v:f'{v:.2f}')} ms. Drafter lookup and read measured {fmt(phases['draft_lookup_read_ms'], lambda v:f'{v:.2f}')} ms. These phase measurements used six warm RAM restores of 404,096 tokens. Tests compare both target keys and drafter keys with the original byte encoding and include existing disk entries.

## Selected release

| One client | Previous release | Selected release | Previous release repeated |
|---|---:|---:|---:|
| Captured Pi turn, decode tok/s | {' | '.join(f'{v:.1f}' for v in rate['pi_median']['aba'])} |
| 404,151-token coding, first output (ms) | {' | '.join(f'{1000*v:.1f}' for v in rate['long_ttft']['aba'])} |
| 404,151-token coding, decode tok/s | {' | '.join(f'{v:.1f}' for v in rate['long_decode']['aba'])} |

Six concurrent 404k coding histories measured {rates(rate['six_decode']['aba'])} aggregate decode tok/s (previous / selected / previous) and {rates(rate['six_wall']['aba'])} aggregate output tok/s including the whole request wave. Each client generated 512 tokens. Every output matched the saved per-agent reference.

A separate mixed sequence replayed the full Pi edit, two long retrievals and the structured edit before switching to 404k coding. The initial qualification measured {qual['long-c1']['median']:.1f} tok/s on that transition. Repeating the sequence on fresh processes gave {rates(mixed['long-c1'])} tok/s on 404k coding and {rates(mixed['short-edit'])} tok/s on the structured edit. The earlier throughput table uses the shorter Pi turn immediately before the 404k workload.

All throughput figures are medians of six warm native requests or waves per fresh engine. The captured Pi request has 2,786 input tokens and a 256-token reasoning continuation. The long request generates 512 tokens. Decode measures first-to-last generated output. Counting the entire single-client 404k request, throughput was {rates(rate['long_wall']['aba'])} output tok/s. The 404k decode rate uses 511 intervals. The Pi rate uses 255. The cards ran at 375 W each and the CPU socket at 225 W. The existing 154 tok/s sequential weight-read ceiling applies to one target token per pass. The speculative rates above depend on accepted proposals. [Ceiling and assumptions](THEORY.md)

An earlier release run was discarded when its test list exposed a stale executable in a shared Cargo build directory. The reported release was rebuilt from the tagged source in an empty target directory. Its server and test executable hashes were checked before qualification. The clean release build reproduced the complete 4,103-token Pi edit, its 62-token tool-return reply and the 542-token following reply. It passed {qual['gpu-tests']} tests with the model and both GPUs. It recovered the exact receipts at {qual['needles'][0]:,} and {qual['needles'][1]:,} tokens. It served three waves of six independent full windows, then eight agents with six resident and two swapped on each of the RAM and forced-disk tiers. Both tiers returned the reference replies with zero read or write failures. The 25,625-token edit measured {qual['short-edit']['median']:.1f} tok/s. Previous tool-return idle prefill remains enabled. [Prior tool-return measurements](TOOL-RETURN.md)

The three source checkpoints are `{experiments[0]}`, `{experiments[1]}`, and `{experiments[2]}`. Raw samples and per-run settings are archived on this workstation under `~/.cache/ds4-flash-rs/profiles/agent-round3-20261126/`.

## Build and rollback

From the engine checkout, build the release in a separate worktree and start it with the selected profile.

```sh
git worktree add --detach ../ds4-agent-round3-build checkpoint-2026-11-26-agent3-release
(cd ../ds4-agent-round3-build && CARGO_TARGET_DIR=target cargo build --release --locked && docker build -t ds4-flash-rs:agent-coding-20261126 .)
docker compose --env-file ../models/.env -f compose.yml -f compose.decode-round2.yml -f compose.coding.yml up --no-build -d --wait
```

To restore the preceding release, replace `compose.coding.yml` with `compose.agentic.yml` in the launch command.
"""
(out/'AGENT-ROUND3.md').write_text(report)
