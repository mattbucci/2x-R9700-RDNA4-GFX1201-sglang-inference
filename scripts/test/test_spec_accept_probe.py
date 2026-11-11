#!/usr/bin/env python3
"""spec_accept_probe.py without a server: what a case's hash covers, and the compare's blank-text count.

A fake `requests` answers the probe; nothing here opens a GPU or a socket.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import spec_accept_probe as sap  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeServer:
    """Answers /v1/models and /v1/chat/completions; `message_of(prompt)` builds the assistant message."""

    def __init__(self, message_of, verify=7, tokens=21):
        self.message_of, self.verify, self.tokens, self.bodies = message_of, verify, tokens, []

    def get(self, url, timeout=None):
        return Reply({"data": [{"id": "fake-model"}]})

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        meta = {"spec_verify_ct": self.verify} if self.verify else {}
        return Reply({"choices": [{"message": self.message_of(json["messages"][0]["content"]),
                                   "finish_reason": "length"}],
                      "usage": {"prompt_tokens": 11, "completion_tokens": self.tokens}, "meta_info": meta})


def args(**kw):
    base = dict(port=1, haystack="", doc_chars="", doc_tokens="", tokenizer="", no_short=False, docs_per_size=4,
                max_tokens=300, label="arm", note="", out="", compare=None, floor=0.95)
    base.update(kw)
    return types.SimpleNamespace(**base)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def run_probe(server, **kw):
    old, sap.requests = sap.requests, server
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "arm.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sap.run(args(out=out, **kw))
            return rc, (load(out) if os.path.exists(out) else None), buf.getvalue()
    finally:
        sap.requests = old


class IdentityText(unittest.TestCase):
    def test_content_only_hashes_as_before(self):
        for message in ({"content": "abc"}, {"content": "abc", "reasoning_content": None},
                        {"content": "abc", "reasoning_content": ""}, {"content": "abc", "reasoning": ""}):
            self.assertEqual(sap.identity_text(message), "abc")

    def test_reasoning_is_part_of_the_text(self):
        self.assertEqual(sap.identity_text({"content": "", "reasoning_content": "think"}), "think\x00")
        self.assertEqual(sap.identity_text({"content": None, "reasoning": "think"}), "think\x00")
        self.assertEqual(sap.identity_text({"content": "abc", "reasoning_content": "think"}), "think\x00abc")

    def test_reasoning_and_content_do_not_alias(self):
        self.assertNotEqual(sap.identity_text({"content": "b", "reasoning_content": "a"}),
                            sap.identity_text({"content": "", "reasoning_content": "ab"}))

    def test_empty_reply(self):
        self.assertEqual(sap.identity_text({}), "")
        self.assertEqual(sha(sap.identity_text({"content": None})), sap.EMPTY_SHA)


class Run(unittest.TestCase):
    def test_thinking_off_model(self):
        rc, got, _ = run_probe(FakeServer(lambda prompt: {"content": "answer to " + prompt[:9],
                                                          "reasoning_content": None}))
        self.assertEqual(rc, 0)
        self.assertEqual(len(got["cases"]), len(sap.SHORT))
        row = got["cases"][0]
        self.assertEqual(row["sha"], sha("answer to " + sap.SHORT[0][:9]))
        self.assertEqual((row["verify_ct"], row["completion_tokens"], row["accept_len"]), (7, 21, 3.0))
        self.assertEqual(row["content_chars"], row["text_chars"])
        self.assertEqual(got["accept_len"], 3.0)

    def test_always_thinking_model_is_not_blank(self):
        # no thinking switch in the template: the whole budget is reasoning and the content is empty
        rc, got, _ = run_probe(FakeServer(lambda prompt: {"content": "", "reasoning_content": "plan " + prompt[:9]}))
        self.assertEqual(rc, 0)
        shas = {row["sha"] for row in got["cases"]}
        self.assertNotIn(sap.EMPTY_SHA, shas)
        self.assertGreater(len(shas), 1)
        self.assertTrue(all(row["content_chars"] == 0 and row["text_chars"] > 0 for row in got["cases"]))
        self.assertEqual(got["cases"][0]["sha"], sha("plan " + sap.SHORT[0][:9] + "\x00"))

    def test_request_body(self):
        server = FakeServer(lambda prompt: {"content": "x"})
        run_probe(server, max_tokens=123)
        body = server.bodies[0]
        self.assertEqual((body["max_tokens"], body["temperature"], body["return_spec_tokens_details"]), (123, 0, True))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["model"], "fake-model")

    def test_no_verify_count_stops(self):
        rc, got, text = run_probe(FakeServer(lambda prompt: {"content": "x"}, verify=0))
        self.assertEqual(rc, 2)
        self.assertIsNone(got)
        self.assertIn("no spec_verify_ct", text)


def receipt(d, name, rows):
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        json.dump({"cases": rows}, fh)
    return path


def row(i, text, verify=10, tokens=30):
    return {"id": f"doc-12000-{i}", "completion_tokens": tokens, "verify_ct": verify,
            "accept_len": round(tokens / verify, 4), "sha": sha(text)}


class Compare(unittest.TestCase):
    def compare(self, ctl, trt):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "cmp.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sap.compare(args(compare=[receipt(d, "c.json", ctl), receipt(d, "t.json", trt)], out=out))
            return rc, load(out), buf.getvalue()

    def test_blank_pairs_are_counted(self):
        ctl = [row(0, ""), row(1, ""), row(2, "same"), row(3, "left")]
        trt = [row(0, ""), row(1, ""), row(2, "same"), row(3, "right", verify=12)]
        rc, got, text = self.compare(ctl, trt)
        self.assertEqual(rc, 0)
        g = got["groups"]["all"]
        self.assertEqual((g["cases"], g["same_text"], g["blank_text"], g["same_verify_count"]), (4, 3, 2, 3))
        self.assertIn("blank-text=2", text)

    def test_no_blank_keeps_the_line(self):
        rows = [row(i, f"text {i}") for i in range(4)]
        rc, got, text = self.compare(rows, rows)
        self.assertEqual(got["groups"]["all"]["blank_text"], 0)
        self.assertNotIn("blank-text", text)
        self.assertIn("ACCEPT VERDICT pass", text)
        line = [ln for ln in text.splitlines() if ln.startswith("ACCEPT all")][0]
        self.assertTrue(line.endswith("lower/higher=0/0"))

    def test_one_side_blank_is_a_difference(self):
        rc, got, _ = self.compare([row(0, "")], [row(0, "text")])
        g = got["groups"]["all"]
        self.assertEqual((g["same_text"], g["blank_text"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
