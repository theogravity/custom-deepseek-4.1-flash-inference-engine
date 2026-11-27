"""Offline checks for the completed, isolated two-client workload and timing bounds."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cold_prefill
import prefill_overlap as overlap


def event(**fields):
    return b"data: " + json.dumps(fields).encode() + b"\n"


def response(usage, finish="length", done=True):
    fields = {"choices": [{"delta": {"content": "sample"}, "finish_reason": finish}], "usage": usage}
    return io.BytesIO(event(**fields) + (b"data: [DONE]\n" if done else b""))


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(requests_started=11, prefix_reused_tokens=8, rotations=2, resumes=2,
                           suspend_failures=0, resume_failures=0)
        self.after = self.before | {"requests_started": 13}
        self.first = overlap.Stream(0, [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 25], 30,
                                    dict(prompt_tokens=70, completion_tokens=1500, total_tokens=1570))
        self.second = overlap.Stream(10, [22, 23], 24,
                                     dict(prompt_tokens=68049, completion_tokens=64, total_tokens=68113))

    def measure(self):
        return overlap.summarize(self.first, self.second, self.before, self.after)

    def test_gap_is_whole_interval_and_done_is_distinct(self):
        row = self.measure()
        self.assertEqual((row["peer_events_during_second_ttft"], row["second_ttft_s"]), (2, 12))
        self.assertEqual((row["max_gap_start_from_second_start_s"], row["max_gap_end_from_second_start_s"],
                          row["max_peer_interevent_s"]), (-2, 10, 12))
        self.assertEqual((row["first_client_to_last_event_s"], row["first_client_to_done_s"]), (25, 30))

    def test_third_admission_cache_rotation_or_failure_rejected(self):
        for name in self.after:
            with self.subTest(name=name), patch.dict(self.after, {name: self.after[name] + 1}):
                with self.assertRaisesRegex(RuntimeError, "workload counters"):
                    self.measure()

    def test_wrong_length_or_missing_usage_rejected(self):
        for stream in (self.first, self.second):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                with self.subTest(k=k), patch.dict(stream.usage):
                    stream.usage.pop(k)
                    with self.assertRaisesRegex(RuntimeError, "token counts"):
                        self.measure()
            with patch.dict(stream.usage, {"completion_tokens": 10}):
                with self.assertRaisesRegex(RuntimeError, "token counts"):
                    self.measure()

    def test_no_interleave_or_late_request_rejected(self):
        for events in ([1, 2, 3, 4, 5, 6, 7, 11, 16, 25],  # review started before eight peer events
                       list(range(1, 10)),                 # peer completed before HTTP worker started
                       list(range(1, 9)) + [16, 21],       # peer completed during the review's wait
                       list(range(1, 9)) + [10, 22, 25]):  # events at boundaries only
            with self.subTest(events=events):
                self.first.events = events
                with self.assertRaises(RuntimeError):
                    self.measure()

    def test_incomplete_stream_is_rejected_even_with_usage_and_finish(self):
        with patch.object(overlap, "urlopen", return_value=response(self.first.usage, done=False)):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                overlap.stream("http://example.test", "example", "input", 1500)

    def test_drain_requires_zero_waiting_and_resident(self):
        idle = dict(logical_active=0, resident=0, waiting=0, swapped=0)
        with (patch.object(overlap, "query", side_effect=[idle | {"waiting": 1}, idle]),
              patch.object(overlap.time, "sleep")):
            self.assertEqual(overlap.idle("", ""), idle)
        with (patch.object(overlap, "query", return_value=idle | {"resident": 1}),
              patch.object(overlap.time, "monotonic", side_effect=[5, 26])):
            with self.assertRaisesRegex(RuntimeError, "drain"):
                overlap.idle("", "")

    def test_parent_deadline_creates_no_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.json"
            with (patch.object(overlap.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 600)) as run,
                  self.assertRaises(subprocess.TimeoutExpired)):
                overlap.measure("", path)
            self.assertEqual(run.call_args.kwargs["timeout"], 600)
            self.assertFalse(path.exists())
            path.write_text("older result")
            with patch.object(overlap.subprocess, "run") as run, self.assertRaises(FileExistsError):
                overlap.measure("", path)
            run.assert_not_called()
            self.assertEqual(path.read_text(), "older result")

    def test_cold_timing_only_records_completion_without_grading_answer(self):
        state = self.before | {"logical_active": 0, "multi_sequence_steps": 4}
        def run(timing, count=1):
            output = io.StringIO()
            with (patch.object(cold_prefill, "query", side_effect=[state, state | {"requests_started": 11+count}]),
                  patch.object(cold_prefill, "urlopen", return_value=response(self.second.usage, finish="stop")),
                  contextlib.redirect_stdout(output)):
                cold_prefill.probe("http://example.test", "example", 20, 99, timing_only=timing)
            return json.loads(output.getvalue())
        row = run(True)
        self.assertTrue(row["passed"])
        self.assertFalse(row["answer_graded"])
        self.assertNotIn("answer", row)
        with self.assertRaises(RuntimeError):
            run(False)
        with self.assertRaises(RuntimeError):
            run(True, 2)
