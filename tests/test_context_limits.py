import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import context_limits as gate


class ContextLimitsTests(unittest.TestCase):
    def test_distinct_native_templates_are_counted_at_the_exact_boundary(self):
        def count(_base, _path, _key, body):
            controls = 4 if body.get("reasoning_effort") == "none" else 30
            return {"input_tokens": controls + len(body["messages"][-1]["content"].split())}
        with patch.object(gate, "query", side_effect=count):
            prompts = gate.sized_messages(SimpleNamespace(base="unused", histories=None, window=256, litellm=True), "key")
        for (codex, size), messages in prompts.items():
            controls = 30 if codex else 4
            self.assertEqual(size, controls + len(messages[-1]["content"].split()))
        self.assertEqual(set(prompts), {(False, 255), (False, 256), (False, 257), (True, 257)})

    def test_rejection_distinguishes_http_errors_from_in_stream_proxy_errors(self):
        delta = dict.fromkeys(("requests_started", "batch_steps", "prefix_reused_tokens", "rotations", "resumes",
                               "ram_writes", "ram_reads", "disk_writes", "disk_reads"), 0)
        message = ("prompt is too long: 257 tokens > 255 maximum for generation. "
                   "This model's maximum context length is 256 tokens (DeepSeek V4.1 Flash).")
        row = dict(route="native", wire="openai", requested_max=1, prompt_tokens=257, backend_delta=delta,
                   http_status=400, content_type="application/json",
                   error=dict(message=message, type="invalid_request_error", param="messages", code="context_length_exceeded"))
        gate.validate(row, 256)
        stream = dict(row, http_status=200, content_type="text/event-stream", sse_event="response.failed")
        with self.assertRaises(ValueError):
            gate.validate(stream, 256)
        for requested_stream in (False, True):
            with self.assertRaisesRegex(ValueError, "before streaming"):
                gate.validate(dict(stream, route="litellm", wire="responses", stream=requested_stream), 256)
        with self.assertRaisesRegex(ValueError, "Codex requires"):
            gate.validate(dict(row, route="litellm", wire="responses", error=dict(row["error"], code="400")), 256)
        for change in [dict(backend_delta=dict(delta, requests_started=1)),
                       dict(backend_delta=dict(delta, disk_reads=1)),
                       dict(error=dict(row["error"], code="400")),
                       dict(error=dict(row["error"], message=message.replace("257", "260")))]:
            with self.assertRaises(ValueError):
                gate.validate(dict(row, **change), 256)
