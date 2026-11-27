"""Offline proof and single-pass checks for the 404k capacity qualification."""
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prefill_overlap
import six_capacity as cap


def counters(started):
    return {k: 0 for k in cap.COUNTERS} | {"requests_started": started}


class SixCapacity(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(agent=i, prompt_sha256=str(i), done_s=5,
                          usage=dict(prompt_tokens=404000, completion_tokens=64, total_tokens=404064),
                          finish="stop") for i in range(6)]
        self.snap = dict(logical_active=6, resident=6, decoding=6, prefilling=0, waiting=0, swapped=0,
                         processed_tokens=2424000, elapsed_s=4)

    def test_live_proof_and_counter_rules(self):
        verify = lambda rows, snaps, after: cap.validate(rows, snaps, counters(10), after)[0]
        self.assertFalse(verify(self.rows, [self.snap], counters(16)))
        for change in [dict(resident=5), dict(decoding=5), dict(swapped=1), dict(elapsed_s=5),
                       dict(processed_tokens=2423999)]:
            with self.subTest(change=change):
                self.assertTrue(verify(self.rows, [self.snap | change], counters(16)))
        for k in cap.COUNTERS:
            after = counters(16)
            after[k] += 1
            self.assertTrue(verify(self.rows, [self.snap], after), k)
        for bad in [dict(done_s=3), dict(prompt_sha256="1"), dict(usage={}), dict(finish="length")]:
            self.assertTrue(verify([self.rows[0] | bad] + self.rows[1:], [self.snap], counters(16)))
        self.assertTrue(verify([dict(agent=0, prompt_sha256="0", done_s=9, error="OOM")] + self.rows[1:],
                               [self.snap], counters(16)))

    def test_exactly_six_streams_and_one_live_observation(self):
        requests = []
        arrived = threading.Condition()
        observed = threading.Event()
        def fake_http(req, timeout):
            self.assertEqual(req.get_header("Authorization"), "Bearer synthetic")
            self.assertEqual(req.full_url, "http://fake/v1/chat/completions")
            body = json.loads(req.data)
            with arrived:
                requests.append(body)
                arrived.notify_all()
            self.assertTrue(observed.wait(3))
            return io.BytesIO(b'data: {"choices":[{"delta":{"reasoning_content":"work"}}]}\n'
                              b'data: {"usage":{"prompt_tokens":404000,"completion_tokens":128,"total_tokens":404128},'
                              b'"choices":[{"delta":{},"finish_reason":"length"}]}\n'
                              b'data: [DONE]\n')

        def metrics(*args):
            with arrived:
                self.assertTrue(arrived.wait_for(lambda: len(requests) == 6, 3))
            return self.snap

        # Release the simulated SSE streams after the real polling loop records its observation.
        def release(pending, timeout):
            observed.set()
            return [], set()

        with (patch.object(prefill_overlap, "urlopen", side_effect=fake_http),
              patch.object(cap, "idle", side_effect=[counters(10), counters(16)]),
              patch.object(cap, "query", side_effect=metrics), patch.object(cap, "wait", side_effect=release),
              patch.object(cap, "review", side_effect=lambda seed, agent: f"Review module {seed}.{agent}")):
            result = cap.run("http://fake", "synthetic", 144)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(requests), 6)
        self.assertEqual(len({r["messages"][0]["content"] for r in requests}), 6)
        self.assertTrue(all(r["max_tokens"] == 128 for r in requests))

    def test_documents_have_distinct_sources(self):
        first, second = cap.review(12, 0), cap.review(12, 1)
        self.assertEqual(first.count("def rule_"), 17606)
        self.assertNotEqual(first.splitlines()[1:], second.splitlines()[1:])
        self.assertEqual(first, cap.review(12, 0))


if __name__ == "__main__":
    unittest.main()
