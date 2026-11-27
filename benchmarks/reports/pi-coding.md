# Pi coding a Go fix on the native engine

The selected Rust/CUDA engine brought the final parser to passing public tests in **119.6–120.4 seconds** on its one clean `xhigh` completion. Pi finished that session at **155.0 seconds** after eight model calls, nine tools and 25,750 generated tokens. The other two attempts ended at the token limit and the wall-time limit. The 386.7 tok/s recorded on a fixed, repetitive edit measures a different workload. [Structured-edit benchmark](decode-round-2.md)

## Workload

Pi 1.4.0 ran one client against DeepSeek V4.1 Flash. Each launch started with a broken copy of the Go flag parser from `model-cli/internal/compose/compose.go` at parent-repository commit `dfcefbfa3b27`. Pi could read the files, run shell commands, edit code and write files. The request asked it to fix flag boundaries, quoted values and environment expansion, run the tests and report the result. The test suite and starting source were identical for all launches. A second sandbox graded submitted files against the original public tests and held-out cases. A 240-second client deadline ended long attempts before grading.

The provider came from `model-cli pi dsv41-flash-exl3 --direct`. Pi selected `xhigh`, which sent `thinking=true`, `reasoning_effort=max` and `max_tokens=64000`. Temperature was zero. Each attempt used a fresh working tree, private Go build cache and a session identifier paired across profiles. The engine started with a private empty prefix-cache directory for each leg. Both RTX PRO 6000s were capped at 375 W and the CPU socket at 225 W.

## Measurements

A clean completion requires an independently correct patch and Pi ending normally. A generated patch can pass both suites even if Pi reaches its output limit on the final response. Timed-out attempts have an unknown grade. The wall clock includes local tools and model calls and ends when Pi exits or reaches the deadline. The independent grader runs afterward.

For completed streams, the public-test timing bounds use the first passing test on the final parser that Pi included in its next request. The lower bound sums all preceding model calls. The upper bound subtracts the duration of that next request and all later requests from Pi's wall time. The grader confirms the final submitted tree afterward.

## Pi task outcomes

| Profile and Pi effort | Clean finishes | Correct patches | Incorrect patches | Ungraded | Pi wall, fresh / warm / warm |
|---|---:|---:|---:|---:|---|
| Previous engine, `xhigh`, first control | 1/3 | 2 | 0 | 1 | 180.0 / 240.1 / 196.8 s |
| Deployed engine, `xhigh` | 1/3 | 1 | 1 | 1 | 220.3 / 240.0 / 155.0 s |
| Previous engine, `xhigh`, return control | 2/3 | 2 | 0 | 1 | 240.0 / 173.6 / 97.6 s |

In the first control's third launch, the final parser passed the public tests at 85.4–86.7 seconds. Pi then issued two more model calls before its final request. That last request generated the full 64,000 tokens in 104.2 seconds. The final tree passed the held-out tests, and Pi reported the output limit.

On the deployed engine's clean run, one turn generated 11,421 tokens before the next tool result came back. That next model call took 12.0 seconds to deliver its first event. Across eight calls the engine reused 80,128 of 104,164 input tokens. That profile stored prefix checkpoints during input processing. A follow-up captured another complete Pi tool exchange and measured the effect of preparing its generated history while the tool was running. [Tool-return latency](tool-return.md)

## Pi reasoning effort

A secondary trial kept the deployed engine, 64,000-token output limit, fixture, tools, deadlines and three session identifiers. Pi selected `high` in place of `xhigh`. The initial request hash in each pair matched after normalizing just this field. Two runs graded correct and finished normally in **141.4** and **128.5** seconds. Their final parsers first passed the public tests at **107.1–108.8** and **124.2–125.4** seconds respectively. The third reached 240.0 seconds and was ungraded. That third session identifier was the one clean completion at `xhigh`.

These observations support trying `high` on this kind of edit. A wider set of bugs and repositories would be needed to pick a default for coding in general. To use it for one Pi session through the normal proxy, run `eval "$(model-cli pi dsv41-flash-exl3 --thinking high)"`. The engine trial used the same setting on the direct route.

## Decode on the same tool-result turn

The first deployed-engine session supplied a real request immediately after the failing Go test. Its prompt contained 2,786 tokens, including the tool schemas, previous reasoning, tool calls and the actual test failure. The replay preserved the request and its JSON ordering, changing only its output limit to 256. Each of the three fresh engines warmed the request once and served it six more times. The native renderer produced the same prompt text and all 2,786 token IDs as the checkpoint reference encoder. All 21 capped responses produced the same reasoning and text hashes. All 18 warm requests reused 2,752 prompt tokens and delivered 256 incremental output events.

| Warm median | Previous engine, before | Deployed engine | Previous engine, after |
|---|---:|---:|---:|
| Streaming decode, tok/s | 178.3 | **178.0** | 178.7 |
| Entire API call, output tok/s | 173.4 | **173.2** | 174.0 |
| Time to first output event | 46.7 ms | **45.4 ms** | 43.6 ms |

Streaming decode divides 255 token intervals by the time from the first to last output event. The entire-call rate includes prompt processing, first-token latency and stream completion. Both throughput rates on the deployed profile were within 0.5% of their respective controls on this turn. The full coding sessions also had longer requests and different tool outcomes.

## Scope

This is one extracted historical bug, repeated with three session identifiers per profile. Tool output and later model requests can differ between launches. Full-task wall times describe observed behavior on this fixture. Earlier 25.6k edit streams, 404k code continuations and six-client waves measure other parts of the serving workload. [Decode and concurrency measurements](decode-round-2.md)

The fixture, held-out tests, Pi driver, captured benchmark-only request and per-turn raw results are archived on this workstation in `~/.cache/ds4-flash-rs/profiles/agentic-coding-pi-20261126/`. The repository receipt includes request hashes, test observations, timing bounds and every capped replay. [Machine-readable measurements](../results/agents/agentic-coding-pi.json)
