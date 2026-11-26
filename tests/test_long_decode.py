"""Verify that the steady decode estimate only counts the same resident cohort."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import long_decode


class ResidentDecodeTests(unittest.TestCase):
    def measure(self, records):
        clients = {object(), object()}
        polls = [(set(), clients) for _ in records] + [(clients, set())]
        times = [t + offset for t, _, _ in records for offset in (0, 0.002)]
        states = [dict(requests_started=2, active=2, decoding=2, logical_active=2,
                       prefilling=0, swapped=0, processed_tokens=count) | extra
                  for _, count, extra in records]
        with (patch.object(long_decode, "wait", side_effect=polls),
              patch.object(long_decode, "query", side_effect=states),
              patch.object(long_decode, "monotonic", side_effect=times)):
            return long_decode.sample_resident_decode(clients, "", "", 2, 0)

    def test_uses_longest_continuous_interval_and_excludes_prefill(self):
        r = self.measure([(0, 1000, {}), (0.2, 1070, {}), (0.4, 5000, {"prefilling": 1}),
                          (0.6, 6000, {}), (1.2, 6100, {}), (1.8, 6240, {}),
                          (2.0, 6800, {"swapped": 1}), (2.2, 7000, {}), (3.25, 7030, {})])
        self.assertEqual((r["committed_tokens"], r["samples"]), (240, 3))
        self.assertAlmostEqual(r["tok_s"], 200)
        self.assertAlmostEqual(r["max_metrics_rtt_ms"], 2)

    def test_one_second_with_the_same_cohort_is_required(self):
        records = [(0, 500, {}), (0.8, 560, {}), (1.5, 700, {"requests_started": 3})]
        self.assertIsNone(self.measure(records))
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            self.measure([(0, 500, {}), (1.5, 499, {})])
