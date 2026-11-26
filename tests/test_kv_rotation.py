import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import kv_rotation as bench


class RotationEvidence(unittest.TestCase):
    def test_extended_history_records_both_actual_seed_sources(self):
        with tempfile.TemporaryDirectory() as d:
            history = Path(d) / 'prior.json'
            history.write_text('{"seed":710}')
            args = argparse.Namespace(histories=history, agents=8, seed=99, tokenizer=Path(d))
            with patch.object(bench, 'load_tokenizer'), patch.object(bench, 'continued_documents', return_value=([('prior', [])]*6, [])) as old, patch.object(bench, 'document', return_value=('new', [])) as fresh:
                requests, _, cohorts = bench.inputs(args, 'native')
            self.assertEqual(len(requests), 8)
            self.assertEqual(old.call_args.args[0].seed, 710)
            self.assertEqual(fresh.call_args.args[0].seed, 99)
            self.assertTrue(fresh.call_args.args[0].shared)
            self.assertEqual([(c['count'], c['base_seed']) for c in cohorts], [(6, 710), (2, 99)])

    def test_generation_keys_and_required_overlap(self):
        tier = dict.fromkeys(['ram_reads','ram_writes','disk_reads','disk_writes','read_failures','write_failures','ram_bytes','disk_bytes'], 0)
        before = dict(logical_active=0,max_seqs=1,max_live_seqs=2,resident=0,swapped=0,rotations=0,resumes=0,suspend_failures=0,resume_failures=0,swap=tier)
        mid = dict(before,logical_active=2,resident=1,swapped=1)
        after = dict(before,rotations=1,resumes=1,swap=dict(tier,ram_reads=1,ram_writes=1))
        for stagger, expected in [(False, True),(True, False)]:
            args = argparse.Namespace(agents=2,base='native',litellm='proxy',expect_tier='ram',allow_staggered_output=False,input=12000,output=128)
            start = time.monotonic()
            def completion(args,key,text,need,gate,i):
                self.assertEqual(key, 'proxy-key')
                gate.wait(timeout=2)
                first = start + (i if stagger else 0)
                return dict(stream_id=str(i),first=first,last=first+.8,end=first+.9,sha256=str(i),exact=True,prompt=args.input,output=args.output,finish='length')
            snapshots = iter((before,mid,after))
            def metrics(url,path,key):
                self.assertEqual((url,path,key),('native','/metrics','native-key'))
                return next(snapshots, after)
            with patch.dict(os.environ,{'DSV41_API_KEY':'native-key','LITELLM_MASTER_KEY':'proxy-key'}), patch.object(bench,'inputs',return_value=([('one',[]),('two',[])],[],[])), patch.object(bench,'query',side_effect=metrics), patch.object(bench,'stream',side_effect=completion), contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(bench.run(args), expected)
            self.assertEqual(json.loads(captured.getvalue())['passed'], expected)


if __name__ == '__main__':
    unittest.main()
