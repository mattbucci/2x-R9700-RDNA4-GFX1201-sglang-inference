#!/usr/bin/env python3
"""consol_session.py without a server: boot order, the identity rules of a boot, and the summary's verdicts on
synthetic receipts. Nothing here opens a GPU or a socket."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import consol_session as cs  # noqa: E402

MARK = "DeltaNet projections: TP-split over 2 ranks"
LATE = "extend split-KV route active"


def session(arms=None, spec=False, block=5):
    s = {"session": "t", "preset": "qwen38", "env": {"QWEN38_SPEC": "1" if spec else "0"}, "spec": spec, "seed": 7,
         "caps": 5, "block": block, "decode": {"contexts": "128,8192", "args": ["--think-off"]},
         "prefill": {"contexts": "8192", "runs": 2}, "server_env": {"SGLANG_A": "1"},
         "markers": {"layout": MARK, "route": {"re": LATE, "late": True}},
         "arms": arms if arms is not None else [
             {"name": "p121", "patches": ["121"], "env": {"SGLANG_A": "0"}, "server_env": {"SGLANG_A": "0"}, "off": ["layout"]},
             {"name": "p081", "patches": ["081"], "env": {"SGLANG_B": "0"}, "server_env": {"SGLANG_B": "0"},
              "on": {"fallback": {"re": "torch RMSNorm fallback", "late": True}}}]}
    if spec:
        s["spec_probe"] = {"sampling": "temperature=1.0", "doc_chars": "12000", "docs_per_size": 2, "offsets": [0, 1190000]}
    return s


def write(directory, name, doc):
    with open(os.path.join(directory, name), "w") as f:
        json.dump(doc, f)


def receipts(directory, boot, tps, prefill_s, ttft=100.0, cycle=None, accept=None, capacity=None):
    write(directory, f"decode-{boot}.json", {"points": [{"context": c, "input_len": c, "median_tps": tps, "median_ttft_ms": ttft}
                                                         for c in (128, 8192)]})
    write(directory, f"prefill-{boot}.json", {"rows": [{"context": 8192, "input_len": 7331, "median_s": prefill_s}]})
    if cycle is not None:
        write(directory, f"cycle-{boot}-sampled.json", {"depths": {"24": {"decode_batches": {"cycle_ms": cycle, "accept_len": 3.5}}}})
    if accept is not None:
        ids, per_verify = accept
        write(directory, f"accept-{boot}.json", {"cases": [{"id": i, "completion_tokens": 300, "verify_ct": per_verify, "sha": "ab" * 8}
                                                            for i in ids]})
    if capacity is not None:
        with open(os.path.join(directory, "capacity.txt"), "a") as f:
            f.write(f"CAP {boot} {capacity}\n")


def summarize(s, directory, keep=0.03, drop=0.01):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        cs.cmd_summary(s, directory, directory, keep, drop)
    with open(os.path.join(directory, "summary.json")) as f:
        return json.load(f), out.getvalue()


class Plan(unittest.TestCase):
    def test_blocks_run_both_orders_between_controls(self):
        arms = [{"name": f"a{i}", "env": {"K": "0"}, "server_env": {"K": "0"}, "reads": {"file": "x", "pattern": "K"}} for i in range(7)]
        boots = [r[0] for r in cs.plan(session(arms))]
        self.assertEqual(len(boots), 1 + (2 * 5 + 2) + (2 * 2 + 2))
        self.assertEqual(boots[:13], ["warm", "b1-ctl1", "b1-a0a", "b1-a1a", "b1-a2a", "b1-a3a", "b1-a4a",
                                      "b1-a4b", "b1-a3b", "b1-a2b", "b1-a1b", "b1-a0b", "b1-ctl2"])
        self.assertEqual(boots[13:], ["b2-ctl1", "b2-a5a", "b2-a6a", "b2-a6b", "b2-a5b", "b2-ctl2"])

    def test_arm_environment_extends_the_session_environment(self):
        rows = {r[0]: r for r in cs.plan(session())}
        self.assertEqual(rows["b1-ctl1"][4], {"QWEN38_SPEC": "0"})
        self.assertEqual(rows["b1-p121a"][4], {"QWEN38_SPEC": "0", "SGLANG_A": "0"})

    def test_an_environment_arm_without_a_marker_or_source_check_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for arm in ({"name": "x", "env": {"K": "0"}, "server_env": {"K": "0"}},
                        {"name": "x", "env": {"K": "0"}, "off": ["layout"]},
                        {"name": "x", "tree": "/nowhere"},                                      # no patches named
                        {"name": "x", "tree": "/nowhere", "patches": ["087"], "tree_check": {"file": "a.py"}},
                        {"name": "ctl", "env": {"K": "0"}, "server_env": {"K": "0"}, "off": ["layout"]}):
                write(d, "s.json", session([arm]))
                with self.assertRaises(SystemExit):
                    cs.session(os.path.join(d, "s.json"))


class Identity(unittest.TestCase):
    def setUp(self):
        self.s = session()

    def test_control_shows_every_marker_and_no_arm_line(self):
        self.assertEqual(cs.identity(self.s, "ctl", MARK, {"SGLANG_A": "1"}, "boot"), [])
        self.assertEqual(cs.identity(self.s, "ctl", MARK + LATE, {"SGLANG_A": "1"}, "end"), [])
        self.assertTrue(cs.identity(self.s, "ctl", MARK, {"SGLANG_A": "1"}, "end"))  # the late line never came
        self.assertTrue(cs.identity(self.s, "ctl", "", {"SGLANG_A": "1"}, "boot"))
        self.assertTrue(cs.identity(self.s, "ctl", MARK + "torch RMSNorm fallback", {"SGLANG_A": "1"}, "boot"))

    def test_a_launcher_variable_that_did_not_reach_the_server_fails(self):
        self.assertEqual(cs.identity(self.s, "p121", "", {"SGLANG_A": "0"}, "boot"), [])
        self.assertTrue(cs.identity(self.s, "p121", "", {"SGLANG_A": "1"}, "boot"))  # off-vs-off pair
        self.assertTrue(cs.identity(self.s, "p121", MARK, {"SGLANG_A": "0"}, "boot"))  # the mechanism still logs

    def test_an_arm_line_is_required_by_the_end(self):
        env = {"SGLANG_A": "1", "SGLANG_B": "0"}
        self.assertEqual(cs.identity(self.s, "p081", MARK, env, "boot"), [])
        self.assertTrue(cs.identity(self.s, "p081", MARK + LATE, env, "end"))
        self.assertEqual(cs.identity(self.s, "p081", MARK + LATE + "torch RMSNorm fallback", env, "end"), [])
        self.assertTrue(cs.identity(self.s, "p081", MARK, {"SGLANG_A": "1"}, "boot"))  # SGLANG_B unset

    def test_a_tree_arm_needs_its_tree_and_a_live_boot_none(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "dev", "python", "sglang"))
            os.makedirs(os.path.join(d, "ext"))
            tree, ext = os.path.join(d, "dev"), os.path.join(d, "ext")
            s = session([{"name": "p087", "tree": tree, "tree_check": {"file": "sglang/x.py", "absent": "PV_BF16"}}])
            env = {"SGLANG_A": "1", "PYTHONPATH": f"{ext}:{tree}/python"}
            self.assertEqual(cs.identity(s, "p087", MARK, env, "boot"), [])
            self.assertTrue(cs.identity(s, "p087", MARK, {"SGLANG_A": "1", "PYTHONPATH": ext}, "boot"))
            self.assertTrue(cs.identity(s, "ctl", MARK, env, "boot"))
            self.assertEqual(cs.identity(s, "ctl", MARK, {"SGLANG_A": "1", "PYTHONPATH": ext + ":"}, "boot"), [])


class LaunchEnv(unittest.TestCase):
    TRACE = "+ export SGLANG_A=1\n++ export 'SGLANG_C=a b'\n+ SGLANG_A=1\n+ echo done\n"

    def test_a_server_name_the_launcher_derives_is_overwritten(self):
        s = session([{"name": "bad", "env": {"SGLANG_A": "0"}, "server_env": {"SGLANG_A": "0"}, "off": ["layout"]},
                     {"name": "good", "env": {"A_KNOB": "0"}, "server_env": {"SGLANG_A": "0"}, "off": ["layout"]},
                     {"name": "pass", "env": {"SGLANG_B": "0"}, "server_env": {"SGLANG_B": "0"}, "off": ["layout"]}])
        self.assertEqual(cs.launch_env(s, "ctl", self.TRACE), [])
        self.assertEqual(cs.launch_env(s, "bad", self.TRACE), ["SGLANG_A=1 after the launcher, want 0"])
        self.assertEqual(cs.launch_env(s, "good", self.TRACE.replace("SGLANG_A=1", "SGLANG_A=0")), [])
        self.assertEqual(cs.launch_env(s, "pass", self.TRACE), [])        # not touched by the launcher: passed through
        self.assertTrue(cs.launch_env(s, "ctl", "+ echo nothing exported\n"))
        self.assertTrue(cs.launch_env(s, "nosuch", self.TRACE))


class Summary(unittest.TestCase):
    def test_verdicts_follow_the_best_row_beyond_the_control_spread(self):
        arms = [{"name": n, "patches": [n[1:]], "env": {"K": "0"}, "server_env": {"K": "0"}, "reads": {"file": "x", "pattern": "K"}}
                for n in ("p1", "p2", "p3", "p4")]
        s = session(arms)
        with tempfile.TemporaryDirectory() as d:
            receipts(d, "b1-ctl1", 30.0, 3.00, capacity=400000)
            receipts(d, "b1-ctl2", 30.06, 3.00, capacity=400000)
            for suffix in "ab":
                receipts(d, f"b1-p1{suffix}", 27.0, 3.00, capacity=400000)                 # decode: the mechanism buys 11 %
                receipts(d, f"b1-p2{suffix}", 30.03, 3.01, ttft=115.0, capacity=400000)    # nothing; ttft +15 ms is below the floor
                receipts(d, f"b1-p3{suffix}", 30.03, 3.06, capacity=400000)                # prefill 2 %: between
                receipts(d, f"b1-p4{suffix}", 30.03, 3.00, capacity=300000)                # capacity: +33 %
            summary, out = summarize(s, d)
        v = {n: a["verdict"] for n, a in summary["arms"].items()}
        self.assertEqual(v, {"p1": "keep", "p2": "drop-candidate", "p3": "between", "p4": "keep"})
        self.assertEqual(summary["arms"]["p1"]["best_row"], "decode@128")
        self.assertAlmostEqual(summary["arms"]["p1"]["best_gain"], 30.03 / 27 - 1, places=3)
        self.assertEqual(summary["arms"]["p4"]["best_row"], "capacity")
        self.assertEqual(summary["arms"]["p2"]["rows"]["ttft@128"]["gain"], 0.0)
        self.assertEqual(summary["gate"], "measured")
        self.assertIn("CONSOL GATE measured", out)

    def test_a_mechanism_that_costs_is_named(self):
        arms = [{"name": "p1", "patches": ["1"], "env": {"K": "0"}, "server_env": {"K": "0"}, "reads": {"file": "x", "pattern": "K"}}]
        with tempfile.TemporaryDirectory() as d:
            for boot, tps in (("b1-ctl1", 30.0), ("b1-ctl2", 30.0), ("b1-p1a", 31.5), ("b1-p1b", 31.5)):
                receipts(d, boot, tps, 3.0)
            summary, _ = summarize(session(arms), d)
        arm = summary["arms"]["p1"]
        self.assertEqual((arm["verdict"], arm["cost_row"]), ("between", "decode@128"))
        self.assertLess(arm["cost"], -0.04)

    def test_a_missing_receipt_is_incomplete_not_a_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            for boot in ("b1-ctl1", "b1-ctl2", "b1-p121a", "b1-p121b", "b1-p081a"):
                receipts(d, boot, 30.0, 3.0)
            summary, _ = summarize(session(), d)
            self.assertEqual(summary["gate"], "incomplete:p081")
            self.assertNotIn("p081", summary["arms"])
            os.remove(os.path.join(d, "prefill-b1-ctl2.json"))
            summary, _ = summarize(session(), d)
            self.assertEqual(summary["gate"], "incomplete:p121,p081")

    def test_speculative_rows_pair_documents_by_order(self):
        s = session(spec=True)
        first, second = ["doc-12000-0", "doc-12000-1"], ["doc-12000-o1190000-0", "doc-12000-o1190000-1"]
        with tempfile.TemporaryDirectory() as d:
            receipts(d, "b1-ctl1", 60.0, 3.0, cycle=50.0, accept=(first, 100))
            receipts(d, "b1-ctl2", 60.0, 3.0, cycle=50.0, accept=(second, 100))
            receipts(d, "b1-p121a", 45.0, 3.0, cycle=55.0, accept=(first, 100))     # tok/s: another continuation
            receipts(d, "b1-p121b", 45.0, 3.0, cycle=55.0, accept=(second, 100))
            receipts(d, "b1-p081a", 60.0, 3.0, cycle=50.0, accept=(first, 120))    # accept length 3.0 -> 2.5
            receipts(d, "b1-p081b", 60.0, 3.0, cycle=50.0, accept=(first, 120))    # wrong document set on the reverse pass
            summary, _ = summarize(s, d)
        self.assertEqual(summary["arms"]["p121"]["verdict"], "keep")
        self.assertEqual(summary["arms"]["p121"]["best_row"], "cycle@24")
        self.assertTrue(summary["arms"]["p121"]["rows"]["decode@128"]["informational"])
        self.assertEqual(summary["arms"]["p121"]["rows"]["accept"]["documents"], 4)
        self.assertEqual(summary["gate"], "incomplete:p081")

    def test_an_empty_accept_text_makes_the_arm_unreadable(self):
        s = session(spec=True)
        first, second = ["doc-12000-0"], ["doc-12000-o1190000-0"]
        with tempfile.TemporaryDirectory() as d:
            for boot, ids in (("b1-ctl1", first), ("b1-ctl2", second), ("b1-p121a", first), ("b1-p121b", second),
                              ("b1-p081a", first), ("b1-p081b", second)):
                receipts(d, boot, 40.0 if "p121" in boot else 60.0, 3.0, cycle=50.0, accept=(ids, 100))
            path = os.path.join(d, "accept-b1-p081b.json")
            with open(path) as f:
                doc = json.load(f)
            doc["cases"][0]["sha"] = cs.EMPTY_SHA
            write(d, "accept-b1-p081b.json", doc)
            summary, out = summarize(s, d)
        self.assertEqual(summary["gate"], "incomplete:p081")
        self.assertEqual(summary["arms"]["p121"]["verdict"], "drop-candidate")
        self.assertIn("empty text", out)


class Committed(unittest.TestCase):
    def test_every_committed_session_file_loads_and_plans(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments",
                            "20-patch-consolidation", "sessions")
        files = sorted(f for f in os.listdir(root) if f.endswith(".json"))
        self.assertGreaterEqual(len(files), 3)
        for f in files:
            with self.subTest(session=f):
                s = cs.session(os.path.join(root, f))  # exits on a malformed arm
                self.assertEqual(s["session"], f[:-len(".json")])
                arms, size = len(s["arms"]), int(s.get("block", 5))
                blocks = -(-arms // size)
                self.assertEqual(len(cs.plan(s)), 1 + 2 * blocks + 2 * arms)
                specs = list(s["markers"].values()) + [m for a in s["arms"] for m in a.get("on", {}).values()]
                for spec in specs:
                    cs.re.compile(cs.marker(spec)[0])
                for a in s["arms"]:
                    for k, v in a.get("server_env", {}).items():
                        self.assertNotEqual(s["server_env"].get(k, "unset"), v, f"{a['name']}: {k} equals the control")


if __name__ == "__main__":
    unittest.main()
