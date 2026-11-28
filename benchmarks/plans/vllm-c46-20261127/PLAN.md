# vLLM C4/C6 decode gap experiment

## Objective
Identify a vLLM decode optimization that can close the native engine's four- and six-client gap without regressing the current single-client default or full-context safety.

## Fixed workload
- Dual RTX PRO 6000 Blackwell, 375 W per GPU
- CPU cap recorded as 225 W in the prior benchmark; re-read before this experiment
- 25,625-token shared coding prompt, 1,500 output tokens, temperature 0
- 1 warm-up + 6 measured waves; identical model pack and request payloads

## Current reference
Current batch-adaptive native: C1/C4/C6 = 410.0 / 497.8 / 503.8 tok/s. Same-host prior vLLM at 375 W: 537.2 and 561.0 tok/s with CPU experts enabled; 485.0 tok/s with them disabled. vLLM's published C4/C6 are 518/557 tok/s at 600 W.

## vLLM features to map
1. Full decode CUDA graphs, captured at all useful batch sizes through max-num-seqs x (1 + speculative width).
2. DSpark speculative execution with scheduler-side acceptance-aware allocation.
3. Optional CPU host-expert pools overlapping uncached target expert work with GPU execution.

## Candidate experiment
Re-test the already implemented selective CPU-expert path on the exact C4/C6 short coding workload. Keep it opt-in. Do not promote unless repeatable throughput improves over the current native baseline with stable outputs and no incomplete streams.

## Completion checks
Record the actual live server profile, GPU/CPU power caps, C4/C6 wave distributions, output lengths/hashes, CPU expert jobs, and resource headroom. Independent review is required before merging any code. Keep or reject based on paired evidence.
