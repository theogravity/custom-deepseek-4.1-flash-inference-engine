#!/usr/bin/env python3
"""Qualify live KV rotation with more clients than GPU slots, then replay each request alone."""

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import threading
import time

from six_streams import (DEFAULT_SEED, STREAMS, api_keys, continued_documents, document,
                         load_tokenizer, query, stream)


def inputs(args, key):
    if not args.histories:
        origins = [{'first_agent': 0, 'count': args.agents, 'base_seed': args.seed,
                    'register': 'shared' if args.shared else 'independent'}]
        return [document(args, key, i) for i in range(args.agents)], [], origins
    frozen = json.loads(args.histories.read_text())
    previous = copy.copy(args)
    previous.seed = frozen['seed']
    requests, proofs = continued_documents(previous, key, load_tokenizer(args.tokenizer), frozen)
    # The first six contain the actual replies from six independent prior
    # conversations. Extra clients get new end receipts on a known long register.
    extra = copy.copy(args)
    extra.shared = True
    requests.extend(document(extra, key, i) for i in range(STREAMS, args.agents))
    origins = [{'first_agent': 0, 'count': STREAMS, 'base_seed': frozen['seed'], 'register': 'prior_conversations'},
               {'first_agent': STREAMS, 'count': args.agents-STREAMS, 'base_seed': extra.seed, 'register': 'shared'}]
    return requests, proofs, origins


def run(args):
    key, completion_key = api_keys(args.litellm)
    requests, proofs, origins = inputs(args, key)
    metrics = lambda: query(args.base, '/metrics', key)
    before = metrics()
    if before['logical_active'] or not before['max_seqs'] < args.agents <= before['max_live_seqs']:
        raise ValueError('qualification requires an idle service and more clients than resident slots within the live admission limit')
    peak_logical = peak_resident = peak_swapped = 0
    started = time.monotonic()
    gate = threading.Barrier(args.agents)
    with concurrent.futures.ThreadPoolExecutor(args.agents) as pool:
        futures = [pool.submit(stream, args, completion_key, text, expected, gate, i)
                   for i, (text, expected) in enumerate(requests)]
        pending = set(futures)
        while pending:
            _, pending = concurrent.futures.wait(pending, timeout=0.12)
            observed = metrics()
            peak_logical = max(peak_logical, observed['logical_active'])
            peak_resident = max(peak_resident, observed['resident'])
            peak_swapped = max(peak_swapped, observed['swapped'])
        rows = [f.result() for f in futures]
    seconds = max(row['end'] for row in rows) - started
    # A response can reach its client just before the worker publishes its
    # final idle sample. Wait briefly so a retained ticket cannot go unnoticed.
    deadline = time.monotonic() + 10
    after = metrics()
    while after['logical_active'] and time.monotonic() < deadline:
        time.sleep(0.05)
        after = metrics()
    overlap = min(r['last'] for r in rows) - max(r['first'] for r in rows)
    delta = {k: after['swap'][k] - before['swap'][k] for k in (
        'ram_reads', 'ram_writes', 'disk_reads', 'disk_writes', 'read_failures', 'write_failures')}
    tiers = ['ram','disk'] if args.expect_tier == 'both' else [args.expect_tier]
    passed = (peak_logical == args.agents and 0 < peak_resident <= before['max_seqs']
              and peak_swapped > 0 and after['logical_active'] == 0
              and after['swap']['ram_bytes'] == after['swap']['disk_bytes'] == 0
              and after['rotations'] > before['rotations'] and after['resumes'] > before['resumes']
              and after['suspend_failures'] == before['suspend_failures']
              and after['resume_failures'] == before['resume_failures']
              and delta['read_failures'] == delta['write_failures'] == 0
              and all(delta[t+'_reads'] > 0 and delta[t+'_writes'] > 0 for t in tiers)
              and len({r['stream_id'] for r in rows}) == args.agents
              and (args.allow_staggered_output or overlap > 0))
    for i, (row, (text, expected)) in enumerate(zip(rows, requests, strict=True)):
        control = stream(args, completion_key, text, expected, threading.Barrier(1), i)
        row['serial_sha256'] = control['sha256']
        passed &= all(r['exact'] and r['prompt'] == args.input and r['output'] == args.output
                      and r['finish'] == 'length' for r in [row,control])
        passed &= row['sha256'] == control['sha256']
        for field in ('first','last','end'):
            row[field+'_s'] = row.pop(field) - started
    print(json.dumps({'passed':passed,'requested_tier':args.expect_tier,'agents':args.agents,
                      'input_tokens_each':args.input,'output_tokens_each':args.output,
                      'cohorts':origins,'continuations':proofs,
                      'overlap_required':not args.allow_staggered_output,
                      'all_stream_overlap_s':overlap,'wave_seconds':seconds,
                      'peak_logical_active':peak_logical,'peak_resident':peak_resident,
                      'peak_swapped':peak_swapped,'swap_delta':delta,'metrics_before':before,
                      'metrics_after':after,'rows':rows}),flush=True)
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='http://127.0.0.1:8001')
    ap.add_argument('--litellm', nargs='?', const='http://127.0.0.1:4000', metavar='ROOT')
    ap.add_argument('--agents', type=int, default=8)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--input', type=int, default=12000)
    ap.add_argument('--output', type=int, default=512)
    ap.add_argument('--shared', action='store_true', help='use the same register with distinct request-specific receipts')
    ap.add_argument('--histories', type=Path, help='extend the first six actual prior conversations from six_streams.py')
    ap.add_argument('--tokenizer', type=Path, default=Path('/mnt/models/dsv41-flash-exl3/tokenizer.json'))
    ap.add_argument('--expect-tier', choices=['ram','disk','both'], default='ram')
    ap.add_argument('--allow-staggered-output', action='store_true', help='allow unequal cold prefill lengths to stagger output; warm fairness gates require overlap by default')
    ap.set_defaults(save_histories=None)
    args = ap.parse_args()
    if not 2 <= args.agents <= 64 or not 2 <= args.output or not 12000 <= args.input <= 524288-args.output:
        ap.error('agent count or context budget out of range')
    if args.histories and (args.shared or args.agents <= STREAMS):
        ap.error('staged histories need more than six agents and supply their own distinct registers')
    return 0 if run(args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
