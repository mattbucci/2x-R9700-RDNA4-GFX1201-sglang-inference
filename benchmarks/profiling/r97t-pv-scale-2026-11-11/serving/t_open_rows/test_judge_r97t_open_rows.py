#!/usr/bin/env python3
"""Tests of judge_r97t_open_rows.py on synthetic receipts (the stand-ins of fake_probe.py)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_probe as F  # noqa: E402
import judge_r97t_open_rows as J  # noqa: E402

JUDGE = os.path.join(os.path.dirname(HERE), "judge_r97t_open_rows.py")


def write(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def build(d, scen="equal", cells=20, boots=J.BOOTS, cells_b=None):
    """The receipts of a whole run; the first-token receipts go through the judge's own --combine."""
    for b in boots:
        arm = "spec-" + b
        for ph in ("A", "B"):
            files = []
            for i in range(1, (cells if ph == "A" or cells_b is None else cells_b) + 1):
                files.append(os.path.join(d, f"cell-{ph}-{arm}-{i}.json"))
                write(d, os.path.basename(files[-1]), F.decode(arm, b, [128], 5, 80, i, scen))
            with open(os.devnull, "w") as null:
                old, sys.stdout = sys.stdout, null
                try:
                    J.combine(os.path.join(d, f"ttft-{ph}-{arm}.json"), files)
                finally:
                    sys.stdout = old
        write(d, f"prefill-{arm}.json", F.prefill(arm, b, 2, scen))
        if b in J.DEEP_BOOTS:
            for name, n, per, off in J.DEPTHS:
                write(d, f"accept-{name}-{arm}.json", F.accept(arm, b, (n,), per, off if b.endswith("2") else 0, scen))


def run(d, **kw):
    return J.judge("flint", d, say=lambda s: None, **kw)


def edit(d, name, fn):
    p = os.path.join(d, name)
    j = json.load(open(p, encoding="utf-8"))
    fn(j)
    write(d, name, j)


class Rows(unittest.TestCase):
    def test_permutation_probability_is_exact(self):
        self.assertAlmostEqual(J.perm_p([1, 2, 3, 4], [5, 6, 7, 8]), 1 / 70)
        self.assertAlmostEqual(J.perm_p([5, 6, 7, 8], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(J.perm_p([1, 2, 3, 5], [4, 6, 7, 8]), 2 / 70)
        self.assertGreater(J.perm_p([1, 4, 5, 8], [2, 3, 6, 7]), 0.4)

    def test_time_row_classes(self):
        # separated but inside the floor: ok, whatever the probability says
        r = J.time_row([100.0, 100.1, 100.2, 100.3], [100.5, 100.6, 100.7, 100.8])
        self.assertEqual((r["verdict"], r["p"]), ("ok", round(1 / 70, 4)))
        # every treatment boot 2 % slower: loss
        r = J.time_row([74.6, 74.3, 74.5, 74.4], [76.0, 75.9, 76.1, 75.8])
        self.assertEqual(r["verdict"], "loss")
        self.assertLess(r["x"], J.FLOOR)
        # one slow treatment boot moves the mean below the floor, the split is not unusual: open
        r = J.time_row([75.0, 75.1, 74.9, 75.0], [75.0, 75.1, 82.5, 74.9])
        self.assertEqual(r["verdict"], "open")
        self.assertGreater(r["p"], J.P_MAX)
        # a faster treatment is ok
        self.assertEqual(J.time_row([76.0, 75.9, 76.1, 75.8], [74.6, 74.3, 74.5, 74.4])["verdict"], "ok")

    def test_time_row_needs_four_boots_per_side(self):
        self.assertEqual(J.time_row([74.6, 74.3], [76.0, 75.9])["verdict"], "unreadable")
        self.assertEqual(J.time_row([74.6, 74.3, None, 74.4], [76.0, 75.9, 76.1, 75.8])["verdict"], "unreadable")
        self.assertEqual(J.time_row([74.6, 74.3, 0, 74.4], [76.0, 75.9, 76.1, 75.8])["verdict"], "unreadable")

    def test_verdict_precedence(self):
        rows = {n: {"verdict": "ok"} for n in J.DECISION}
        self.assertEqual(J.verdict(rows), "resolved-ok")
        rows["prefill:58K"]["verdict"] = "open"
        rows["accept:deep"]["verdict"] = "inconclusive"
        self.assertEqual(J.verdict(rows), "open:prefill:58K,accept:deep")
        rows["ttft:24"]["verdict"] = "loss"
        self.assertEqual(J.verdict(rows), "kill:ttft:24")
        rows["accept:deep"]["verdict"] = "drop"
        self.assertEqual(J.verdict(rows), "kill:ttft:24,accept:deep")


class Whole(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_equal_resolves(self):
        build(self.d)
        res = run(self.d)
        self.assertEqual(res["verdict"], "resolved-ok")
        self.assertEqual(res["rows"]["accept:deep"]["documents"], 2 * sum(d[2] for d in J.DEPTHS))
        self.assertEqual(res["rows"]["accept:deep"]["same_text"], 0)
        self.assertEqual(res["rows"]["ttft:24"]["cells"]["off1"], 40)
        self.assertEqual(set(res["decision_rows"]), set(J.DECISION))

    def test_first_token_loss_kills(self):
        build(self.d, "ttft-loss")
        res = run(self.d)
        self.assertEqual(res["verdict"], "kill:ttft:24")
        self.assertEqual(res["rows"]["ttft:24:A"]["verdict"], "loss")  # a record row, not named in the verdict

    def test_one_slow_boot_is_open(self):
        build(self.d, "ttft-one-boot")
        res = run(self.d)
        self.assertEqual(res["verdict"], "open:ttft:24")
        self.assertGreater(res["rows"]["ttft:24"]["p"], J.P_MAX)

    def test_prefill_loss_kills(self):
        build(self.d, "prefill-loss")
        self.assertEqual(run(self.d)["verdict"], "kill:prefill:7.3K,prefill:58K,prefill:197K")

    def test_accept_drop_kills(self):
        build(self.d, "accept-drop")
        res = run(self.d)
        self.assertEqual(res["verdict"], "kill:accept:deep")
        self.assertEqual([res["rows"][f"accept:{n}"]["verdict"] for n in J.DEPTH_NAMES], ["drop", "drop"])

    def test_a_drop_at_one_depth_kills_under_an_ok_pool(self):
        build(self.d)
        for b in ("on1", "on2"):  # more verifies at 197K, fewer at 58K: the pool stays above the floor
            edit(self.d, f"accept-197K-spec-{b}.json", lambda j: [c.update(verify_ct=round(c["verify_ct"] * 1.08)) for c in j["cases"]])
            edit(self.d, f"accept-58K-spec-{b}.json", lambda j: [c.update(verify_ct=round(c["verify_ct"] * 0.96)) for c in j["cases"]])
        res = run(self.d)
        deep = res["rows"]["accept:deep"]
        self.assertGreaterEqual(deep["ratio"], J.FLOOR)
        self.assertEqual(res["rows"]["accept:197K"]["verdict"], "drop")
        self.assertEqual((deep["verdict"], deep["dropped_depths"]), ("drop", ["197K"]))
        self.assertEqual(res["verdict"], "kill:accept:deep")

    def test_scattered_accept_is_inconclusive(self):
        build(self.d)
        for b in ("on1", "on2"):
            for name in J.DEPTH_NAMES:
                edit(self.d, f"accept-{name}-spec-{b}.json",
                     lambda j: [c.update(verify_ct=round(c["verify_ct"] * (1.45 if i % 2 else 0.65))) for i, c in enumerate(j["cases"])])
        res = run(self.d)
        deep = res["rows"]["accept:deep"]
        self.assertLess(deep["ratio"], J.FLOOR)
        self.assertGreaterEqual(deep["ratio_ci95"][1], 1.0)
        self.assertEqual(deep["verdict"], "inconclusive")
        self.assertEqual(res["verdict"], "open:accept:deep")

    def test_accept_receipts_that_cannot_be_read(self):
        build(self.d)
        os.remove(os.path.join(self.d, "accept-197K-spec-on2.json"))
        res = run(self.d)
        self.assertEqual(res["rows"]["accept:197K"]["verdict"], "unreadable")
        self.assertEqual(res["rows"]["accept:58K"]["verdict"], "ok")
        self.assertEqual(res["verdict"], "open:accept:deep")
        for change, why in ((lambda j: j["cases"].pop(), "document sets differ"),
                            (lambda j: j["cases"][0].update(verify_ct=0), "no verify count"),
                            (lambda j: j["cases"][0].update(sha=J.EMPTY_SHA), "empty text")):
            build(self.d)
            edit(self.d, "accept-58K-spec-on1.json", change)
            res = run(self.d)
            self.assertIn(why, res["rows"]["accept:58K"]["why"])
            self.assertEqual(res["verdict"], "open:accept:deep")

    def test_the_second_pair_must_read_other_documents(self):
        build(self.d)
        for b in ("off2", "on2"):  # the second pair without the offset: the ids of the first pair again
            write(self.d, f"accept-58K-spec-{b}.json", F.accept("spec-" + b, b, (58000,), J.DEPTHS[0][2], 0, "equal"))
        res = run(self.d)
        self.assertIn("share document ids", res["rows"]["accept:58K"]["why"])
        self.assertEqual(res["verdict"], "open:accept:deep")

    def test_a_boot_with_too_few_cells_is_not_read(self):
        build(self.d, cells_b=6)
        res = run(self.d)
        self.assertEqual(res["rows"]["ttft:24"]["verdict"], "unreadable")
        self.assertEqual(res["rows"]["ttft:24:A"]["verdict"], "ok")
        self.assertEqual(res["verdict"], "open:ttft:24")
        self.assertEqual(run(self.d, min_cells=3)["verdict"], "resolved-ok")

    def test_missing_boots_leave_it_open_and_a_kill_stands(self):
        build(self.d, boots=J.BOOTS[:4])
        self.assertEqual(run(self.d)["verdict"], "open:ttft:24,prefill:7.3K,prefill:58K,prefill:197K")
        build(self.d, "accept-drop", boots=J.BOOTS[:4])
        self.assertEqual(run(self.d)["verdict"], "kill:accept:deep")

    def test_combine_counts_unreadable_cells(self):
        good = os.path.join(self.d, "c1.json")
        write(self.d, "c1.json", F.decode("spec-off1", "off1", [128], 5, 80, 1, "equal"))
        out = os.path.join(self.d, "ttft-A-spec-off1.json")
        r = subprocess.run([sys.executable, JUDGE, "--combine", out, good, os.path.join(self.d, "absent.json")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("1 of 2 cells", r.stdout)
        j = json.load(open(out, encoding="utf-8"))
        self.assertEqual((len(j["cells"]), j["cells"][1]["ttft_ms"]), (2, None))
        self.assertEqual(J.cells_of(self.d, "A", "off1"), [j["cells"][0]["ttft_ms"]])

    def test_cli_tables(self):
        def cli(*args):
            r = subprocess.run([sys.executable, JUDGE, *args], capture_output=True, text=True)
            return r.returncode, r.stdout.split()
        self.assertEqual(cli("--boots", "flint"), (0, list(J.BOOTS)))
        self.assertEqual(cli("--configs", "qwen38"), (0, ["spec"]))
        self.assertEqual(cli("--phases", "flint", "off2"), (0, ["ttftA", "prefill2", "ttftB", "deep", "touch"]))
        self.assertEqual(cli("--phases", "flint", "on4"), (0, ["ttftA", "prefill2", "ttftB", "touch"]))
        self.assertEqual(cli("--boots", "laguna")[0], 2)
        self.assertEqual(cli("--phases", "flint", "on9")[0], 2)
        rc, words = cli("--depths")
        self.assertEqual((rc, words), (0, [str(x) for d in J.DEPTHS for x in d]))
        # the order balances the positions: both sides sum to the same position total
        pos = {b: i for i, b in enumerate(J.BOOTS)}
        self.assertEqual(sum(pos[b] for b in J.BOOTS if b.startswith("on")), sum(pos[b] for b in J.BOOTS if b.startswith("off")))
        self.assertEqual([p for p in J.PAIRS], [("off1", "on1"), ("off2", "on2")])


if __name__ == "__main__":
    unittest.main()
