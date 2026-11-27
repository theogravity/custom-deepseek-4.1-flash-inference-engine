# Four- and six-client code-edit decoding

The standalone engine produced **584 aggregate output tok/s at four clients** and **583 at six** on the upstream Python edit. Its previous coding profile reached 457.5 and 502.5. The one-client median held at 383 tok/s. Each inference GPU ran at 375 W and the measured CPU socket cap was 225 W. [Per-wave measurements, output hashes and original client results](../results/upstream/concurrent-repetition.json)

| Clients | Published vLLM at 600 W | Previous native at 375 W | Current native at 375 W |
|---:|---:|---:|---:|
| 1 | 282 | 382.5 | 383 |
| 4 | 518 | 457.5 | **584** |
| 6 | 557 | 502.5 | **583** |

Each native headline is the median of six second passes of the unmodified upstream benchmark. Each client receives a 25,625-token Python module and generates 1,500 tokens at temperature zero. Aggregate rate divides all completion tokens, including reasoning, by the longest stream’s first-output-to-close interval. The vLLM values are one published second pass on a different date, with different GPU power limits. The historical default thinking effort is unverified. The code edit contains repeated, numbered functions. Its gain depends on how often an agent produces predictable continuations. [Pinned upstream client](https://github.com/theogravity/deepseek-flash-v4.1-dual-rtx-6000/blob/f020f98034679d9ec2d132e55603ba6a35a51a5a/bench/decode_bench2.py) · [Original serving conditions](headline-reference.md#serving-conditions)

## What changed

The previous profile restricted verified output repetition to a single decoding client. The new profile lets a client propose a repeated sequence from its own committed output while peers decode. It also extrapolates supported numeric sequences in repeated code. The target model verifies every proposal and DSpark supplies drafts when the available match is weak. Target verification passes involving all six clients fell from 1,494 before to 905 over six measured six-client waves. [Implementation](../../src/prompt_lookup.rs)

## Matched controls and first output

| Clients | Control before, tok/s | Repetition enabled, tok/s | Control after, tok/s | Median first output in seconds, before / enabled / after |
|---:|---:|---:|---:|---|
| 4 | 463.7 | **590.2** | 462.5 | 0.182 / 0.181 / 0.182 |
| 6 | 505.2 | **582.9** | 505.2 | 0.266 / 0.265 / 0.266 |

Each arm started a fresh process from the same immutable image. The only changed setting was `DS4_OUTPUT_LOOKUP` (single, repeated, single). Each concurrency had one warm-up and six measured waves per arm. The first-output figures are medians across all individual timed streams, from starting the HTTP request to the first visible or reasoning delta. Every stream in the matched waves restored 25,600 prefix tokens. We compared each stream’s prompt, reasoning and visible output digests across the three arms. All matched, all returned the requested starting function, and every client emitted 1,500 tokens. The reference publication does not give comparable four- and six-client warm first-output distributions. [Timing and identity client](../../tools/edit_scaling.py)

For reproduction, start the profile in the [serving guide](../../docs/serving.md). Export the engine API key as `API_KEY`, then run the following commands for each policy, restarting the serving process between arms. The upstream client uses port 8001. A populated prefix cache makes its 25,600 shared tokens available from the start.

```sh
python3 tools/edit_scaling.py --clients 4 --waves 7
python3 tools/edit_scaling.py --clients 6 --waves 7
# From the linked upstream checkout, run the official headline six times.
for i in 1 2 3 4 5 6; do python3 -I bench/decode_bench2.py; done
```

The previous profile uses `DS4_OUTPUT_LOOKUP=single`. Drop the concurrent coding overlay from the launch command to restore the previous image and policy. The runtime still provides six resident slots, a 524,288-token window per session, and RAM and disk KV offload. The canceled six-client long-context run was excluded from this qualification.
