#!/usr/bin/env python3
"""greedy_route_probe.py without a server: the request budgets, --extra-tokens, and the compare's blank cases.

A fake `requests` plays two models: one that honours `enable_thinking: false`, and one whose template has no
thinking switch and reasons for a fixed number of tokens before every answer.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import greedy_route_probe as grp  # noqa: E402

BUDGETS = {"plain": [16], "code": [220], "tool-call": [128], "think-tool": [1024], "json-schema": [96],
           "logprobs": [6], "logit-bias": [16, 16], "penalty": [160], "concurrent": [16, 16, 16], "sampled": [16]}
CAPITALS = {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Egypt": "Cairo", "Peru": "Lima", "Canada": "Ottawa"}


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeServer:
    """`think_tokens` > 0: the model reasons that many tokens first, and a smaller budget leaves the content empty."""

    def __init__(self, think_tokens=0):
        self.think_tokens, self.budgets = think_tokens, []

    def get(self, url, timeout=None):
        return Reply({"data": [{"id": "fake-model"}]})

    def answer(self, body):
        prompt = body["messages"][0]["content"]
        if "tools" in body:
            city = "Osaka" if "Osaka" in prompt else "Lima"
            return {"content": None, "reasoning_content": "the user wants the weather",
                    "tool_calls": [{"function": {"name": "get_weather",
                                                 "arguments": json.dumps({"city": city, "unit": "celsius"})}}]}
        if "response_format" in body:
            text = json.dumps({"capital": "Nairobi", "continent": "Africa"})
        elif prompt == grp.CODE:
            text = "def merge(a, b):\n    return sorted(a + b)"
        else:
            text = next(v for k, v in CAPITALS.items() if k in prompt)
            if body.get("logit_bias"):
                text = "Lutetia"
        if body["max_tokens"] <= self.think_tokens:
            return {"content": "", "reasoning_content": "1.  **Analyze the request"}
        return {"content": text, "reasoning_content": "thinking" if self.think_tokens else None}

    def post(self, url, json=None, timeout=None):
        if url.endswith("/tokenize"):
            return Reply({"tokens": [len(json["prompt"])]})
        self.budgets.append(json["max_tokens"])
        choice = {"message": self.answer(json), "finish_reason": "stop"}
        if json.get("logprobs") and choice["message"]["content"]:
            tops = [{"token": f"t{i}", "logprob": -float(i)} for i in range(5)]
            choice["logprobs"] = {"content": [{"token": "t0", "logprob": 0.0, "top_logprobs": tops}]}
        return Reply({"choices": [choice]})


def run_probe(server, extra_tokens):
    old, grp.requests = grp.requests, server
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "arm.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = grp.run(types.SimpleNamespace(port=1, out=out, extra_tokens=extra_tokens))
            with open(out, encoding="utf-8") as fh:
                return rc, json.load(fh), buf.getvalue()
    finally:
        grp.requests = old


def passed(receipt):
    return sorted(c["name"] for c in receipt["cases"] if c["ok"])


class Budgets(unittest.TestCase):
    def test_default_budgets_are_unchanged(self):
        server = FakeServer()
        rc, got, text = run_probe(server, 0)
        self.assertEqual((rc, got["passed"], got["extra_tokens"]), (0, 10, 0))
        self.assertEqual(sorted(server.budgets), sorted(b for v in BUDGETS.values() for b in v))
        self.assertIn("ROUTE PROBE DONE passed=10/10", text)

    def test_extra_tokens_reach_every_request(self):
        server = FakeServer()
        rc, got, _ = run_probe(server, 1024)
        self.assertEqual((rc, got["extra_tokens"]), (0, 1024))
        self.assertEqual(sorted(server.budgets), sorted(b + 1024 for v in BUDGETS.values() for b in v))

    def test_case_list_matches_the_budget_table(self):
        self.assertEqual([name for name, _ in grp.CASES], list(BUDGETS))


class AlwaysThinking(unittest.TestCase):
    def test_default_budgets_read_empty_content(self):
        # 242 tokens of reasoning before every answer: only the tool-call cases survive the default budgets
        rc, got, text = run_probe(FakeServer(think_tokens=242), 0)
        self.assertEqual(rc, 1)
        self.assertEqual(passed(got), ["think-tool", "tool-call"])
        self.assertIn("[route] concurrent: FAIL", text)
        self.assertIn("[route] plain: FAIL", text)

    def test_extra_tokens_reach_the_answers(self):
        rc, got, text = run_probe(FakeServer(think_tokens=242), 1024)
        self.assertEqual((rc, got["passed"]), (0, 10))
        self.assertIn("[route] concurrent: PASS", text)


class LoopingServer(FakeServer):
    """Reasons before every answer; under a logit bias the reasoning never reaches one."""

    def __init__(self, reasoning):
        super().__init__()
        self.reasoning = reasoning

    def answer(self, body):
        if body.get("logit_bias"):
            return {"content": "", "reasoning_content": self.reasoning}
        return super().answer(body)


class BiasCase(unittest.TestCase):
    def bias_row(self, server):
        rc, got, _ = run_probe(server, 1024)
        return rc, got, next(c for c in got["cases"] if c["name"] == "logit-bias")

    def test_a_reasoning_loop_under_the_ban_passes(self):
        rc, got, row = self.bias_row(LoopingServer('I cannot write it. "Par..." I think I am stuck.'))
        self.assertEqual((rc, got["passed"]), (0, 10))
        self.assertIn("stuck", row["identity"])
        self.assertIn("finish=stop", row["detail"])

    def test_the_banned_word_in_the_reasoning_fails(self):
        rc, got, row = self.bias_row(LoopingServer("The answer is Paris."))
        self.assertEqual((rc, row["ok"]), (1, False))

    def test_an_empty_generation_fails(self):
        rc, got, row = self.bias_row(LoopingServer(""))
        self.assertEqual((rc, row["ok"]), (1, False))

    def test_another_answer_still_passes(self):
        rc, got, row = self.bias_row(FakeServer())
        self.assertEqual((row["ok"], row["identity"]), (True, "Lutetia"))


class Compare(unittest.TestCase):
    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            for path, receipt in ((pa, a), (pb, b)):
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(receipt, fh)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = grp.compare(pa, pb)
            return rc, buf.getvalue()

    def test_two_empty_arms_are_blank_not_identical(self):
        _, blind, _ = run_probe(FakeServer(think_tokens=242), 0)
        rc, text = self.compare(blind, blind)
        self.assertIn("plain: BLANK on both", text)
        self.assertIn("ROUTE COMPARE identical=2/2 blank=6", text)
        self.assertEqual(rc, 0)

    def test_all_blank_is_not_a_pass(self):
        rows = [{"name": "plain", "ok": False, "detail": "", "identity": ""}]
        rc, text = self.compare({"cases": rows}, {"cases": rows})
        self.assertEqual(rc, 1)
        self.assertIn("ROUTE COMPARE identical=0/0 blank=1", text)

    def test_identical_arms(self):
        _, got, _ = run_probe(FakeServer(), 0)
        rc, text = self.compare(got, got)
        self.assertEqual(rc, 0)
        self.assertTrue(text.strip().endswith("ROUTE COMPARE identical=8/8"))

    def test_a_divergence_is_reported(self):
        _, got, _ = run_probe(FakeServer(), 0)
        other = json.loads(json.dumps(got))
        next(c for c in other["cases"] if c["name"] == "code")["identity"] += " # tail"
        rc, text = self.compare(got, other)
        self.assertEqual(rc, 1)
        self.assertIn("code: DIVERGE at char", text)
        self.assertIn("ROUTE COMPARE identical=7/8", text)

    def test_different_budgets_do_not_compare(self):
        _, a, _ = run_probe(FakeServer(), 0)
        _, b, _ = run_probe(FakeServer(), 1024)
        rc, text = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn("budgets differ", text)

    def test_receipt_without_the_field_reads_as_zero(self):
        _, a, _ = run_probe(FakeServer(), 0)
        old = {k: v for k, v in a.items() if k != "extra_tokens"}
        rc, _ = self.compare(old, a)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
