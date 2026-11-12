import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.join(HERE, "..", "client", "r97s_requests.py")
sys.path.insert(0, os.path.join(HERE, "..", "client"))
import r97s_requests as rq  # noqa: E402


class Fake(BaseHTTPRequestHandler):
    cache, log, drop_cache = [], [], False

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        if self.path == "/flush_cache":
            Fake.cache = []
            Fake.log.append("flush")
            out = b"Cache flushed.\n"
        else:
            ids = body["input_ids"]
            cached = 0
            while cached < min(len(ids) - 1, len(Fake.cache)) and ids[cached] == Fake.cache[cached]:
                cached += 1
            if Fake.drop_cache:
                cached = max(0, cached - 1)
            Fake.cache = list(ids)
            Fake.log.append((body["rid"], len(ids), body["sampling_params"]))
            out = json.dumps({"text": "x", "output_ids": [5], "meta_info": {
                "id": body["rid"], "prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class RequestsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Fake.cache, Fake.log, Fake.drop_cache = [], [], False
        self.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.prompts = os.path.join(self.tmp.name, "prompts.json")
        pr = {"tail": 4, "commit": "abc", "prompts": {
            "10": {"prefix": 10, "ids": list(range(14)), "sha256": "s10"},
            "20": {"prefix": 20, "ids": list(range(24)), "sha256": "s20"}}}
        json.dump(pr, open(self.prompts, "w"))

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def run_client(self, *extra):
        out = os.path.join(self.tmp.name, "r.json")
        p = subprocess.run([sys.executable, CLIENT, "--port", str(self.srv.server_address[1]), "--prompts", self.prompts,
                            "--out", out, *extra], capture_output=True, text=True)
        return p, json.load(open(out))

    def test_sequence_and_counts(self):
        p, r = self.run_client()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("R97S REQUESTS DONE ok", p.stdout)
        greedy = {"temperature": 0, "max_new_tokens": 1}
        self.assertEqual(Fake.log, ["flush", ("r97s-10-cold", 10, greedy), ("r97s-10-turn", 14, greedy), "flush",
                                    ("r97s-10-flushed", 14, greedy), "flush", ("r97s-20-cold", 20, greedy),
                                    ("r97s-20-turn", 24, greedy), "flush", ("r97s-20-flushed", 24, greedy)])
        self.assertEqual([(x["depth"], x["state"], x["prompt_tokens"], x["cached_tokens"]) for x in r["requests"]],
                         [(10, "cold", 10, 0), (10, "turn", 14, 10), (10, "flushed", 14, 0),
                          (20, "cold", 20, 0), (20, "turn", 24, 20), (20, "flushed", 24, 0)])
        self.assertEqual(r["prompts_sha"], {"10": "s10", "20": "s20"})

    def test_depth_selection(self):
        p, r = self.run_client("--depths", "20")
        self.assertEqual(p.returncode, 0)
        self.assertEqual([x["depth"] for x in r["requests"]], [20, 20, 20])

    def test_an_unexpected_cached_count_fails_but_finishes(self):
        Fake.drop_cache = True
        p, r = self.run_client()
        self.assertEqual(p.returncode, 1)
        self.assertIn("R97S REQUESTS DONE FAIL 10-turn: cached_tokens 9, want 10", p.stdout)
        self.assertEqual(len(r["requests"]), 6)

    def test_plan_rejects_a_wrong_id_count(self):
        with self.assertRaises(ValueError):
            rq.plan({"tail": 4, "prompts": {"10": {"prefix": 10, "ids": list(range(13))}}}, [10])


if __name__ == "__main__":
    unittest.main()
