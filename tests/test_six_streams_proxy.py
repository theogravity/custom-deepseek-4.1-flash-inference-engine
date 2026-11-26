"""Exercise the actual six-client driver against isolated native and proxy endpoints."""
import contextlib
import io
import itertools
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import six_streams as bench


class ProxyRouteTests(unittest.TestCase):
    native = "http://native.invalid"
    proxy = "http://proxy.invalid"

    def run_round(self, use_proxy, fault=None, default_proxy=False):
        clients = threading.Barrier(6)
        ids = itertools.count()
        sizes, counters = [], []
        proxy_url = "http://127.0.0.1:4000" if default_proxy else self.proxy
        route = proxy_url if use_proxy else self.native

        def native(base, path, key, body=None):
            self.assertEqual((base, key), (self.native, "native-key"))
            if path == "/metrics":
                counters.append(True)
                return {"active": 0, "processed_tokens": 0,
                        "six_target_passes": int(len(counters) > 1 and fault != "no_batch")}
            self.assertEqual(path, "/v1/messages/count_tokens")
            self.assertEqual(body["model"], bench.MODEL)
            self.assertNotIn("cache", body)
            count = len(body["messages"][0]["content"].split())
            sizes.append(count)
            return {"input_tokens": count}

        @contextlib.contextmanager
        def complete(base, path, key, body):
            self.assertEqual((base, path), (route, "/v1/chat/completions"))
            self.assertEqual(key, "proxy-key" if use_proxy else "native-key")
            self.assertEqual(body["model"], bench.LITELLM_MODEL if use_proxy else bench.MODEL)
            if use_proxy:
                self.assertEqual(body["cache"], {"no-cache": True, "no-store": True})
            else:
                self.assertNotIn("cache", body)
            self.assertEqual(body["stream_options"], {"include_usage": True})
            prompt = body["messages"][0]["content"]
            text = ("import pygame" if "pygame" in prompt else
                    "#include <stdio.h> // json" if "JSON parser" in prompt else "use lru_cache;")
            sid = str(next(ids)) if fault != "same_id" else "one"

            def data(obj):
                return b"data: " + json.dumps(obj).encode() + b"\n"

            def events():
                yield data({"id":sid, "choices":[{"delta":{"content":text}}]})
                clients.wait(timeout=5)
                delta = {"content":"\n"}
                if fault in bench.NON_TEXT_FIELDS:
                    delta[fault] = "unexpected output"
                yield data({"id":sid, "choices":[{"delta":delta}]})
                yield data({"id":sid, "choices":[{"delta":{}, "finish_reason":"length"}],
                            "usage":{"prompt_tokens":len(prompt.split()) + int(fault == "input"),
                                     "completion_tokens":16 + int(fault == "output")}})
                if fault != "truncated":
                    yield b"data: [DONE]\n"
            yield events()

        args = ["test", "--base", self.native, "--output", "16"]
        if use_proxy:
            args.append("--litellm")
            if not default_proxy:
                args.append(self.proxy)
        output = io.StringIO()
        with patch.object(sys, "argv", args), patch.dict(os.environ, {
                "DSV41_API_KEY": "native-key", "LITELLM_MASTER_KEY": "proxy-key"}), \
             patch.object(bench, "query", native), patch.object(bench, "http", complete), \
             contextlib.redirect_stdout(output):
            status = bench.main()
        self.assertGreaterEqual(len(counters), 2)
        self.assertEqual(len(sizes), 6 if use_proxy else 0)
        return status, json.loads(output.getvalue())

    def test_native_and_proxy(self):
        for use_proxy, default in [(False, False), (True, False), (True, True)]:
            with self.subTest(proxy=use_proxy, default=default):
                status, result = self.run_round(use_proxy, default_proxy=default)
                self.assertEqual(status, 0)
                self.assertTrue(result["all_valid"] and result["concurrent"])
                self.assertEqual(result["completion_route"], "litellm" if use_proxy else "native")

    def test_proxy_still_requires_native_work_and_valid_streams(self):
        for fault in ["no_batch", "input", "output", "same_id"]:
            with self.subTest(fault=fault):
                status, result = self.run_round(True, fault)
                self.assertEqual(status, 1)
                self.assertFalse(result["concurrent"] if fault == "no_batch" else result["all_valid"])
        with self.assertRaisesRegex(ValueError, "incomplete stream"):
            self.run_round(True, "truncated")
        for field in bench.NON_TEXT_FIELDS:
            with self.subTest(nontext=field), self.assertRaisesRegex(ValueError, "plain visible output"):
                self.run_round(True, field)

    def test_proxy_disallows_serial_baseline(self):
        with patch.object(sys, "argv", ["test", "--litellm", "--baseline"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            bench.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
