import copy
import glob
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R97S = os.path.join(HERE, "..", "..", "..", "r97s-dump-2026-11-11")
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(R97S, "t_emu"))
import judge_r97t as jr  # noqa: E402
import test_analyze_dump as base  # noqa: E402  (the fake-dump writer of the R97-S analysis tests)

ad = jr.load_analysis(R97S)
ARMS = (("scaled", "fp8", "x256"), ("off", "fp8", "kernel"), ("bf16-kv-scaled", "bf16", "bf16"))
N = len(base.DEPTHS) * len(base.LAYERS)


def fake_cell(x256=1e-5, kernel=3e-3, exact=1e-3, vs=((1e-3, 4e-3), (1e-3, 4e-3)), bf16=4e-5, off=(1e-5, 4e-3)):
    """one depth and layer of the three arms as the frozen analysis reports them (two ranks, den2 = 1 each)"""
    def obs(dists, gaps=None):
        ranks = []
        for r in range(2):
            v = {k: {"num2": d * d, "vs_exact": 0.0} for k, d in dists.items()}
            if gaps:
                v["x256"]["vs_exact"], v["kernel"]["vs_exact"] = gaps[r]
            ranks.append({"walk": "splitkv", "rows": 64, "prefix": 6000, "den2": 1.0, "variants": v})
        return ranks
    return {"scaled": obs({"x256": x256, "kernel": kernel, "exact": exact}, vs),
            "off": obs({"x256": 3e-3, "kernel": off[0], "exact": off[1]}),
            "bf16": obs({"bf16": bf16, "exact": 1e-3})}


def fake_report(cell):
    rep = {"arms": {}}
    for name, pool in (("scaled", "fp8"), ("off", "fp8"), ("bf16", "bf16")):
        rep["arms"][name] = {"pool": pool, "ranks": {
            "rank%d" % r: {"depths": {"6000": {"layers": {"0": {"turn": cell[name][r]}}}}} for r in range(2)}}
    return rep


def fake_r97s(floor=1e-4, measured=1e-4):
    return {"verdict": "confirmed", "cells": [{"depth": 6000, "layer": 0, "floor": floor, "floor_measured": measured,
                                              "floor_from": "bf16-kv", "kernel": 2e-5, "exact": 4e-3, "x256": 3e-3,
                                              "state": "confirmed"}]}


def one(cell, r97s=None):
    return jr.judge(ad, fake_report(cell), r97s or fake_r97s(), "scaled", "off", "bf16", [6000], [0])


class JudgeRuleTest(unittest.TestCase):
    def test_a_cell_on_the_scaled_reference_is_valid(self):
        verdict, cells = one(fake_cell())
        self.assertEqual(verdict, "fix-valid")
        c = cells[0]
        self.assertEqual((c["valid"], c["off"]["state"], c["bf16"]["ok"], c["floor"]), (True, "confirmed", True, 1e-4))
        self.assertAlmostEqual(c["scaled"]["x256"], 1e-5, places=12)

    def test_the_match_bar_is_twice_the_r97s_floor(self):
        self.assertEqual(one(fake_cell(x256=1.99e-4))[0], "fix-valid")
        verdict, cells = one(fake_cell(x256=2.01e-4))
        self.assertEqual((verdict, cells[0]["scaled"]["match"], cells[0]["scaled"]["closer"]), ("not-valid:1", False, True))
        # the floor is the R97-S value of the cell, not this job's bf16 arm
        self.assertEqual(one(fake_cell(x256=2.01e-4, bf16=1.9e-4))[0], "not-valid:1")
        self.assertEqual(one(fake_cell(x256=2.01e-4), fake_r97s(floor=1.1e-4, measured=1.1e-4))[0], "fix-valid")

    def test_the_scaled_reference_must_be_closer_to_exact_weights_on_both_ranks(self):
        verdict, cells = one(fake_cell(vs=((1e-3, 4e-3), (4e-3, 4e-3))))
        self.assertEqual((verdict, cells[0]["scaled"]["match"], cells[0]["scaled"]["closer"]), ("not-valid:1", True, False))
        self.assertEqual(cells[0]["scaled"]["ref_vs_exact"], {"x256": 4e-3, "kernel": 4e-3})

    def test_an_off_arm_that_does_not_repeat_r97s_decides_nothing(self):
        for off, state in (((2.1e-4, 4e-3), "unexplained"), ((1e-5, 4.9e-4), "not-present")):
            verdict, cells = one(fake_cell(off=off))
            self.assertEqual((verdict, cells[0]["off"]["state"]), ("invalid:off-1", state))
        self.assertEqual(one(fake_cell(off=(2e-4, 5e-4)))[0], "fix-valid")

    def test_the_bf16_bound_is_twice_its_r97s_distance_and_never_under_the_minimum(self):
        self.assertEqual(one(fake_cell(bf16=1.99e-4))[0], "fix-valid")
        verdict, cells = one(fake_cell(bf16=2.01e-4))
        self.assertEqual((verdict, cells[0]["bf16"]["ok"], cells[0]["bf16"]["bound"]), ("invalid:bf16-1", False, 2e-4))
        low = fake_r97s(floor=6.1e-5, measured=2e-5)
        self.assertEqual(one(fake_cell(bf16=6.0e-5), low)[0], "fix-valid")
        verdict, cells = one(fake_cell(bf16=6.2e-5), low)
        self.assertEqual((verdict, cells[0]["bf16"]["bound"]), ("invalid:bf16-1", 6.1e-5))

    def test_both_identity_lines_are_named_and_come_before_the_fix(self):
        self.assertEqual(one(fake_cell(x256=1.0, bf16=1.0, off=(1.0, 1.0)))[0], "invalid:off-1+bf16-1")

    def test_an_r97s_summary_that_is_not_this_grid_is_unreadable(self):
        for change in ({"verdict": "unexplained"}, {"cells": []}):
            self.assertEqual(one(fake_cell(), {**fake_r97s(), **change})[0], "unreadable:r97s-summary")
        for key, value in (("state", "unexplained"), ("floor_from", "no-pool"), ("layer", 1)):
            r = fake_r97s()
            r["cells"][0][key] = value
            self.assertEqual(one(fake_cell(), r)[0], "unreadable:r97s-summary")

    def test_arm_roles_and_missing_cells(self):
        rep = fake_report(fake_cell())
        rep["arms"]["bf16"]["pool"] = "fp8"
        self.assertEqual(jr.judge(ad, rep, fake_r97s(), "scaled", "off", "bf16", [6000], [0])[0], "unreadable:pools-fp8-fp8-fp8")
        rep = fake_report(fake_cell())
        del rep["arms"]["off"]["ranks"]["rank1"]["depths"]["6000"]["layers"]["0"]["turn"]
        self.assertEqual(jr.judge(ad, rep, fake_r97s(), "scaled", "off", "bf16", [6000], [0])[0], "unreadable:off-turn-6000-L00")
        rep = fake_report(fake_cell())
        rep["arms"]["scaled"]["ranks"]["rank0"] = {"unreadable": "no-forward", "forwards": 0, "foreign": 0}
        self.assertEqual(jr.judge(ad, rep, fake_r97s(), "scaled", "off", "bf16", [6000], [0])[0],
                         "unreadable:scaled-rank0-no-forward")


class ThreeArmDumpTest(unittest.TestCase):
    """fake dumps through the frozen analysis: what a served arm leaves, at small dimensions"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pr, cls.sk = base.prompts(), base.sinks()
        cls.roots = {}
        for key, pool, weights in (("fp8-x256", "fp8", "x256"), ("fp8-kernel", "fp8", "kernel"),
                                   ("bf16-bf16", "bf16", "bf16"), ("bf16-exact", "bf16", "exact")):
            cls.roots[key] = os.path.join(cls.tmp.name, key)
            base.serve(cls.roots[key], pool, weights, cls.pr, cls.sk)
        kw = dict(tp=base.TP, head_dim=base.D, tail=base.TAIL, min_prefix=base.MIN_PREFIX)
        cls.kw = kw
        cls.r97s = ad.analyze([("preset", cls.roots["fp8-kernel"], base.client(cls.pr), "fp8"),
                               ("bf16-kv", cls.roots["bf16-bf16"], base.client(cls.pr), "bf16")],
                              cls.sk, cls.pr, list(base.LAYERS), **kw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_job(self, scaled="fp8-x256", off="fp8-kernel", bf16="bf16-bf16", r97s=None):
        arms = [(name, self.roots[key], base.client(self.pr), pool)
                for (name, pool, _), key in zip(ARMS, (scaled, off, bf16))]
        return jr.analyze(ad, arms, self.sk, self.pr, list(base.LAYERS), r97s or self.r97s, **self.kw)

    def test_the_reference_job_is_confirmed(self):
        self.assertEqual((self.r97s["verdict"], len(self.r97s["cells"])), ("confirmed", N))

    def test_a_scaled_dump_matches_and_the_identity_arms_hold(self):
        rep = self.run_job()
        self.assertEqual(rep["verdict"].split(":")[0] in ("fix-valid", "not-valid"), True)
        self.assertEqual(rep["roles"], {"scaled": "scaled", "off": "off", "bf16": "bf16-kv-scaled"})
        self.assertEqual(len(rep["cells"]), N)
        for c in rep["cells"]:
            self.assertEqual((c["off"]["state"], c["off"]["kernel"], c["bf16"]["ok"], c["bf16"]["dist"]),
                             ("confirmed", 0.0, True, 0.0))
            self.assertEqual((c["scaled"]["match"], c["scaled"]["x256"], c["walk"]), (True, 0.0, "splitkv"))
            self.assertGreater(c["scaled"]["kernel"], 2 * c["floor"])
            self.assertEqual(c["valid"], c["scaled"]["closer"])
            self.assertEqual(sorted(c["scaled"]["tails"]), ["cold-tail", "flushed-tail"])
            self.assertEqual([t["x256"] for t in c["scaled"]["tails"].values()], [0.0, 0.0])
        bad = sum(not c["valid"] for c in rep["cells"])
        self.assertEqual(rep["verdict"], "fix-valid" if bad == 0 else "not-valid:%d" % bad)
        self.assertEqual(rep["r97s_rule_same_job"]["verdict"], "confirmed")
        text = "\n".join(jr.lines(ad, rep))
        self.assertIn("R97T VERDICT " + rep["verdict"], text)
        self.assertIn("R97T ARM scaled depth 200 L00 | no-pool[extend 0+8]", text)
        self.assertIn("R97T CELL depth 200 L00 F 6.10e-05 [splitkv] | off kernel 0.00e+00", text)
        self.assertIn("R97T TAIL depth 330 L03 flushed-tail x256 0.00e+00 (0.00 F)", text)
        self.assertIn("R97T IDENTITY off confirmed %d of %d, bf16 within its bound %d of %d" % (N, N, N, N), text)
        self.assertNotIn("R97S", text)

    def test_a_switch_without_effect_fails_the_match_in_every_cell(self):
        rep = self.run_job(scaled="fp8-kernel")
        self.assertEqual(rep["verdict"], "not-valid:%d" % N)
        self.assertEqual([c["scaled"]["match"] for c in rep["cells"]], [False] * N)

    def test_a_switch_that_reaches_the_off_arm_decides_nothing(self):
        rep = self.run_job(off="fp8-x256")
        self.assertEqual(rep["verdict"], "invalid:off-%d" % N)

    def test_a_bf16_arm_off_its_emulation_decides_nothing(self):
        rep = self.run_job(bf16="bf16-exact")
        self.assertEqual(rep["verdict"].split("-")[0], "invalid:bf16")

    def test_a_missing_forward_is_unreadable(self):
        root = os.path.join(self.tmp.name, "short")
        base.serve(root, "fp8", "x256", self.pr, self.sk)
        os.remove(sorted(glob.glob(os.path.join(root, "rank1", "fwd*.pt")))[2])
        arms = [("scaled", root, base.client(self.pr), "fp8"), ("off", self.roots["fp8-kernel"], base.client(self.pr), "fp8"),
                ("bf16-kv-scaled", self.roots["bf16-bf16"], base.client(self.pr), "bf16")]
        rep = jr.analyze(ad, arms, self.sk, self.pr, list(base.LAYERS), self.r97s, **self.kw)
        self.assertEqual(rep["verdict"].startswith("unreadable:scaled-"), True)

    def test_a_reference_job_on_another_grid_is_unreadable(self):
        r = copy.deepcopy(self.r97s)
        r["cells"] = r["cells"][1:]
        self.assertEqual(self.run_job(r97s=r)["verdict"], "unreadable:r97s-summary")


if __name__ == "__main__":
    unittest.main()
