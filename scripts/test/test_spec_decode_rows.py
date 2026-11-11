#!/usr/bin/env python3
"""Unit tests for the canonical-row probe that pairs a streamed rate with the server's verify count."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

import spec_decode_rows as sdr  # noqa: E402


def _endpoint(tokens, verifies, unstreamed_tokens=None):
    """A chat endpoint that answers `tokens` greedy tokens; `verifies` None = a server without a speculative worker."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json({"data": [{"id": "mock"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stream = bool(req.get("stream"))
            n = tokens if stream or unstreamed_tokens is None else unstreamed_tokens
            usage = {"prompt_tokens": 17, "completion_tokens": n}
            if verifies and req.get("return_spec_tokens_details"):
                usage["spec_tokens_details"] = {"spec_verify_ct": verifies}
            if not stream:
                self._json({"choices": [{"message": {"content": "The " + "word " * (n - 1)}, "finish_reason": "stop"}],
                            "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(obj):
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            event({"choices": [{"delta": {"role": "assistant", "content": "The "}, "finish_reason": None}]})
            for _ in range(n - 1):
                time.sleep(0.002)
                event({"choices": [{"delta": {"content": "word "}, "finish_reason": None}]})
            event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            event({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _rows(tokens, verifies, unstreamed_tokens=None):
    server = _endpoint(tokens, verifies, unstreamed_tokens)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "rows.json"
            argv = ["spec_decode_rows.py", "--port", str(server.server_address[1]), "--contexts", "128,8192",
                    "--runs", "2", "--maxtok", "80", "--think-off", "--label", "mock", "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                rc = sdr.main()
            return rc, json.loads(out.read_text())
    finally:
        server.shutdown()
        server.server_close()


class SpecDecodeRowsTest(unittest.TestCase):
    def test_speculative_rows_split_into_accept_and_cycle(self):
        rc, result = _rows(37, 11)
        self.assertEqual(rc, 0)
        self.assertEqual(len(result["points"]), 2)
        for point in result["points"]:
            self.assertEqual(point["runs_completion_tokens"], [37, 37])
            self.assertEqual(point["runs_verify_ct"], [11, 11])
            self.assertEqual(point["paired_runs"], 2)
            self.assertAlmostEqual(point["median_accept_len"], 37 / 11, places=3)
            for run in point["runs"]:
                self.assertTrue(run["same_head"])
                # cycle = 1000 * accept length / rate, from the unrounded rate within rounding of the stored one
                self.assertAlmostEqual(run["cycle_ms"], 1000.0 * run["accept_len"] / run["tok_s"], delta=0.05)

    def test_a_server_without_a_drafter_keeps_the_rate_alone(self):
        rc, result = _rows(37, None)
        self.assertEqual(rc, 0)
        for point in result["points"]:
            self.assertEqual(point["runs_verify_ct"], [None, None])
            self.assertIsNone(point["median_accept_len"])
            self.assertIsNone(point["median_cycle_ms"])
            self.assertGreater(point["median_tps"], 0)

    def test_unequal_completion_tokens_give_no_cycle(self):
        # The unstreamed request took another continuation: its verify count does not belong to the streamed rate.
        rc, result = _rows(37, 11, unstreamed_tokens=41)
        self.assertEqual(rc, 0)
        for point in result["points"]:
            self.assertEqual(point["paired_runs"], 0)
            self.assertIsNone(point["median_cycle_ms"])
            self.assertAlmostEqual(point["median_accept_len"], 41 / 11, places=3)


if __name__ == "__main__":
    unittest.main()
