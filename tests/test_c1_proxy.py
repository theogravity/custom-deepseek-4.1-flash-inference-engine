import argparse
import contextlib
import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import c1_proxy as bench

RECEIPTS = ["R123456789abc", "Rabcdef987654"]
MESSAGES = [{"role":"user", "content":"Vault register"}]
ANSWER = " ".join(RECEIPTS)


@contextlib.contextmanager
def endpoint(coalesced):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, dict(self.headers), body))
            chunks = [ANSWER] if coalesced else [RECEIPTS[0]+" ", RECEIPTS[1]]
            if self.path == "/v1/chat/completions":
                events = [{"id":"abc","choices":[{"delta":{"content":text}}]} for text in chunks]
                events += [{"id":"abc","choices":[{"delta":{},"finish_reason":"length"}]},
                           {"id":"abc","choices":[],"usage":{"prompt_tokens":256,"completion_tokens":16}}]
                events = [json.dumps(e) for e in events] + ["[DONE]"]
            else:
                events = [{"type":"message_start","message":{"id":"abc","content":[],"usage":{"input_tokens":256}}},
                          {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}]
                events += [{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":t}} for t in chunks]
                events += [{"type":"content_block_stop","index":0},
                           {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":16}},
                           {"type":"message_stop"}]
                events = [json.dumps(e) for e in events]
            data = "".join("data: "+e+"\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self,*args):
            pass
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", seen
        finally:
            server.shutdown()
            worker.join()


class SingleClientTests(unittest.TestCase):
    def test_loopback_protocols_routes_and_one_visible_frame(self):
        with endpoint(False) as (native, direct), endpoint(True) as (proxy, proxied):
            args = argparse.Namespace(output=16,native=native,proxy=proxy,timeout=4)
            for wire in ("openai","anthropic"):
                a = bench.stream_request(args, wire, "native", "direct-key", MESSAGES, RECEIPTS, 256)
                b = bench.stream_request(args, wire, "proxy", "proxy-key", MESSAGES, RECEIPTS, 256)
                self.assertEqual(a["sha256"], b["sha256"])
                self.assertGreater(a["decode_tok_s"], 0)
                self.assertEqual((b["text_chunks"], b["decode_ms"], b["decode_tok_s"]), (1, 0, None))
                self.assertGreaterEqual(b["total_ms"], b["ttft_ms"])
            for i, wire in enumerate(("openai","anthropic")):
                direct_path, direct_headers, direct_body = direct[i]
                proxy_path, proxy_headers, proxy_body = proxied[i]
                self.assertEqual(direct_path, "/v1/chat/completions" if wire == "openai" else "/v1/messages")
                self.assertEqual(direct_path, proxy_path)
                self.assertEqual({k:v for k,v in proxy_body.items() if k not in ("model","cache")},
                                 {k:v for k,v in direct_body.items() if k!="model"})
                self.assertEqual(proxy_body["cache"], {"no-cache":True,"no-store":True})
                self.assertEqual(proxy_body["model"], bench.LITELLM_MODEL if wire=="openai" else bench.MODEL)
                self.assertNotIn("cache", direct_body)
                if wire=="openai":
                    self.assertEqual(direct_headers["Authorization"],"Bearer direct-key")
                    self.assertEqual(proxy_headers["Authorization"],"Bearer proxy-key")
                else:
                    self.assertEqual(direct_headers["X-Api-Key"],"direct-key")
                    self.assertEqual(proxy_headers["X-Api-Key"],"proxy-key")

    def test_cache_state_and_backend_exclusivity(self):
        baseline = dict(logical_active=0, requests_started=10, batch_steps=55, rotations=2,
                        multi_sequence_steps=0, multi_target_passes=10, prefix_reused_tokens=80)
        complete = dict(baseline, requests_started=11, batch_steps=63, prefix_reused_tokens=272)
        cases = [("warm",complete,True),("cold",complete,False),
                 ("cold",dict(complete, prefix_reused_tokens=80), True),
                 ("warm",dict(complete, requests_started=12), False),
                 ("warm",dict(complete, requests_started=10), False),
                 ("warm",dict(complete, multi_sequence_steps=1), False),
                 ("warm",dict(complete, prefix_reused_tokens=208), False)]
        for mode, after, valid in cases:
            args = argparse.Namespace(native="native",mode=mode,chunk_size=64)
            with self.subTest(mode=mode,counters=after), patch.object(bench,"query",side_effect=[baseline,after]), \
                 patch.object(bench,"stream_request",return_value={"output_tokens":16}):
                def call():
                    return bench.checked_call(args,"openai","proxy",["native-key","proxy-key"],MESSAGES,RECEIPTS,256,False)
                if valid:
                    self.assertEqual(call()["backend"]["batch_steps"], 8)
                else:
                    with self.assertRaises(ValueError):
                        call()

    def test_counterbalanced_paired_differences_and_missing_rates(self):
        args = argparse.Namespace(contexts=[256],wire="openai",mode="warm",seed=5,output=16,pairs=6,histories=None)
        def complete(args, wire, route, keys, messages, receipts, size, warmup):
            row = dict.fromkeys(bench.MEASUREMENTS, 17 if route=="proxy" else 10)
            row.update(sha256="fixed",route=route)
            if route=="proxy":
                row["decode_tok_s"]=None
            return row
        out=[]
        with patch.object(bench,"workload",return_value=(MESSAGES,RECEIPTS,"fixture")), \
             patch.object(bench,"api_keys",return_value=("k","v")), patch.object(bench,"checked_call",side_effect=complete):
            bench.run(args,out.append)
        pairs=[r for r in out if r["kind"]=="pair"]
        self.assertEqual([p["order"] for p in pairs], [["native","proxy"],["proxy","native"]]*3)
        self.assertTrue(all(p["proxy_minus_native"]["ttft_ms"]==7 for p in pairs))
        self.assertEqual([r["warmup"] for r in out if r["kind"]=="sample"].count(True),2)
        self.assertEqual(out[-1]["proxy_minus_native"]["ttft_ms"]["median"],7)
        self.assertEqual(out[-1]["proxy_minus_native"]["decode_tok_s"]["missing"],6)

    def test_rejects_hidden_outputs_and_changed_ids(self):
        event=bench.Completion()
        event.openai(b'{"id":"a","choices":[{"delta":{"content":"yes"}}]}')
        for raw in [b'{"id":"b","choices":[]}',b'{"id":"a","choices":[{"delta":{"reasoning":"hidden"}}]}']:
            with self.assertRaises(ValueError):
                event.openai(raw)
        event=bench.Completion()
        event.anthropic(b'{"type":"message_start","message":{"id":"a","usage":{"input_tokens":256}}}')
        with self.assertRaises(ValueError):
            event.anthropic(b'{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"hidden"}}')


if __name__ == "__main__":
    unittest.main()
