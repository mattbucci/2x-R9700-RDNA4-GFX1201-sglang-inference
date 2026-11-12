"""Unit tests of the R97-T serving judge on synthetic receipts: python3 -m unittest -v test_judge_r97t_serving (from this directory)."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import judge_r97t_serving as J  # noqa: E402
import synth  # noqa: E402

BOOTS = ("off1", "on1", "on2", "off2")
NEEDLE = {"flint": (1024, 32768, 131072, 197000), "qwen38": (32768, 131072, 197000)}


def build(d, preset, edit=None):
    """A complete receipt set; edit(kind, cfg, boot) -> kwargs for the synth builder, or 'skip'."""
    for cfg, phases in J.CONFIGS[preset]:
        spec = cfg == "spec"
        for b in BOOTS:
            arm = f"{cfg}-{b}"

            def kw(kind):
                return (edit(kind, cfg, b) if edit else None) or {}

            def put(kind, name, fn, *a, **default):
                k = kw(kind)
                if k == "skip":
                    return
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    json.dump(fn(arm, *a, **{**default, **k}), f)
            for ph in phases:
                if ph == "caps":
                    put("caps", f"capabilities-{arm}.json", synth.caps)
                elif ph.startswith("prefill"):
                    put("prefill", f"prefill-{arm}.json", synth.prefill)
                elif ph == "sweep":
                    put("decode", f"decode-{arm}.json", synth.decode, spec)
                elif ph == "forced":
                    put("forced", f"decode-forced-{arm}.json", synth.decode, spec, 2)
                elif ph == "sampled":
                    put("cycle", f"cycle-{arm}-sampled.json", synth.cycle)
                elif ph == "accept":
                    put("accept", f"accept-{arm}.json", synth.accept, offset=0 if b in ("off1", "on1") else 1190000)
                elif ph == "route":
                    put("route", f"route-{arm}.json", synth.route)
                elif ph == "cachehit":
                    put("cachehit", f"cachehit-{arm}.json", synth.cachehit)
                elif ph == "needle":
                    put("needle", f"needle-{arm}.json", synth.needle, NEEDLE[preset])


def run(preset, edit=None):
    lines = []
    with tempfile.TemporaryDirectory() as d:
        build(d, preset, edit)
        out = J.judge(preset, d, lines.append)
    return out, lines


def on(b):
    return b.startswith("on")


class Rows(unittest.TestCase):
    def v(self, c, t, hib=True):
        return J.classify(c, t, hib)["verdict"]

    def test_classes(self):
        self.assertEqual(self.v((100, 100), (99.5, 100.2)), "ok")
        self.assertEqual(self.v((100, 100), (97, 97)), "loss")
        self.assertEqual(self.v((100, 90), (98, 88)), "open")          # both below, inside the boot spread
        self.assertEqual(self.v((100, 100), (98.5, 101.6)), "noise")   # one order below, means level, inside the spread
        self.assertEqual(self.v((100, 100), (95, 100.1)), "open")      # one order far below
        self.assertEqual(self.v((100, 100), (98.5, 100.0)), "noise")   # the shortfall 0.005 is inside the treatment's own spread 0.015
        self.assertEqual(self.v((100, 100), (98.0, 98.95)), "loss")    # both below, the mean loss 0.015 exceeds the spread 0.0096
        self.assertEqual(self.v((10, 10), (10.3, 10.3), False), "loss")  # seconds are inverted
        self.assertEqual(self.v((10, 10), (9.7, 9.7), False), "ok")
        self.assertEqual(self.v((100, None), (100, 100)), "unreadable")
        self.assertEqual(self.v((100, 0), (100, 100)), "unreadable")

    def test_floor_is_the_rule(self):
        self.assertEqual(self.v((100, 100), (98.9, 98.9)), "loss")
        self.assertEqual(self.v((100, 100), (99.0, 99.0)), "ok")
        old = J.FLOOR
        try:
            J.FLOOR = 0.95
            self.assertEqual(self.v((100, 100), (97, 97)), "ok")
        finally:
            J.FLOOR = old

    def test_four(self):
        want = {(2, 2): "fine", (1, 2): "fine", (0, 2): "fine", (2, 0): "lower", (2, 1): "open", (1, 0): "open",
                (1, 1): "shared", (0, 0): "shared", (0, 1): "shared"}
        for (c, t), w in want.items():
            self.assertEqual(J.four(c, t), w, (c, t))

    def test_item_gate(self):
        self.assertEqual(J.item_gate({"a": (2, 2), "b": (2, 2)}, "full"), "full")
        self.assertEqual(J.item_gate({"a": (2, 0), "b": (2, 1), "c": (1, 1)}, "full"), "lower:a")
        self.assertEqual(J.item_gate({"b": (2, 1), "c": (1, 1)}, "full"), "open:b")
        self.assertEqual(J.item_gate({"c": (1, 1), "a": (0, 0)}, "full"), "shared:a+c")
        self.assertEqual(J.item_gate({"c": (2, 0)}, "same", "lost"), "lost:c")
        self.assertEqual(J.item_gate({"c": (1, 1)}, "same", "lost"), "shared-miss:c")

    def test_fisher(self):
        self.assertAlmostEqual(J.fisher_greater(5, 10, 0, 10), 3003 / 184756, places=9)
        self.assertAlmostEqual(J.fisher_greater(0, 10, 0, 10), 1.0)
        self.assertAlmostEqual(J.fisher_greater(1, 12, 1, 12), 1 - (12 * 11 / 2) / (24 * 23 / 2) * 1.0, places=6)

    def test_verdict_precedence(self):
        self.assertEqual(J.verdict({"speed": "ok", "caps-spec": "full"}), "ship-ok")
        self.assertEqual(J.verdict({"speed": "ok (x)", "caps-spec": "shared:thinking", "needle-spec": "shared-miss:a"}), "ship-ok")
        self.assertEqual(J.verdict({"speed": "open:a", "needle-spec": "lost:b"}), "kill:needle-spec")
        self.assertEqual(J.verdict({"speed": "loss:a", "accept-spec": "drop"}), "kill:speed+accept-spec")
        self.assertEqual(J.verdict({"speed": "ok", "accept-spec": "inconclusive", "route-spec": "unreadable"}),
                         "open:accept-spec+route-spec")
        self.assertEqual(J.verdict({"speed": "ok", "cachehit-nospec": "worse"}), "kill:cachehit-nospec")


class Whole(unittest.TestCase):
    def test_equal_ships(self):
        for preset in ("flint", "qwen38"):
            out, lines = run(preset)
            self.assertEqual(out["verdict"], "ship-ok", (preset, out["gates"]))
            self.assertEqual(lines[-1], "R97T SERVING VERDICT ship-ok")
            self.assertEqual(out["gates"]["speed"], "ok")
        out, _ = run("qwen38")
        self.assertEqual(sorted(out["gates"]), ["accept-spec", "cachehit-spec", "caps-spec", "needle-spec", "route-spec", "speed"])
        out, _ = run("flint")
        self.assertEqual(sorted(out["gates"]), ["accept-spec", "cachehit-nospec", "caps-nospec", "caps-spec", "needle-nospec",
                                                "route-nospec", "speed"])
        kinds = {n.split(":")[0]: r["kind"] for n, r in out["configs"]["nospec"]["speed_rows"].items()}
        self.assertEqual(kinds, {"prefill": "gate", "canon": "gate", "ttft": "soft", "forced": "gate"})
        kinds = {n.split(":")[0]: r["kind"] for n, r in out["configs"]["spec"]["speed_rows"].items()}
        self.assertEqual(kinds, {"prefill": "gate", "canon": "record", "ttft": "soft", "cycle": "gate"})

    def test_prefill_loss_kills(self):
        out, _ = run("qwen38", lambda k, c, b: {"slow": 1.03} if (k == "prefill" and on(b) and c == "nospec") else None)
        self.assertEqual(out["verdict"], "kill:speed")
        self.assertEqual(out["gates"]["speed"], "loss:nospec-prefill:7.3K+nospec-prefill:58K+nospec-prefill:197K")

    def test_one_order_slow_is_open(self):
        out, _ = run("flint", lambda k, c, b: {"slow": 1.05} if (k == "forced" and b == "on1") else None)
        self.assertTrue(out["verdict"].startswith("open:speed"), out["verdict"])

    def test_cycle_loss_kills_canonical_spec_row_does_not(self):
        out, _ = run("qwen38", lambda k, c, b: {"slow": 1.04} if (k == "cycle" and on(b)) else None)
        self.assertEqual(out["verdict"], "kill:speed")
        out, _ = run("qwen38", lambda k, c, b: {"slow": 1.2} if (k == "decode" and on(b) and c == "spec") else None)
        self.assertEqual(out["verdict"], "ship-ok")
        self.assertEqual(out["configs"]["spec"]["speed_rows"]["canon:58K"]["verdict"], "loss")

    def test_nospec_canonical_gates_only_on_equal_tokens(self):
        out, _ = run("qwen38", lambda k, c, b: {"slow": 1.2} if (k == "decode" and on(b) and c == "nospec") else None)
        self.assertEqual(out["verdict"], "kill:speed")
        out, _ = run("qwen38", lambda k, c, b: {"slow": 1.2, "tokens": 61} if (k == "decode" and on(b) and c == "nospec") else None)
        self.assertEqual(out["verdict"], "ship-ok")
        self.assertEqual(out["configs"]["nospec"]["speed_rows"]["canon:24"]["kind"], "record")

    def test_soft_ttft_loss_is_open(self):
        out, _ = run("flint", lambda k, c, b: {"ttft_slow": 1.1} if (k == "decode" and on(b) and c == "nospec") else None)
        self.assertEqual(out["verdict"], "open:speed")
        self.assertTrue(out["gates"]["speed"].startswith("open:nospec-ttft:24"), out["gates"]["speed"])

    def test_needle(self):
        cell = "131072@0.5"
        out, _ = run("qwen38", lambda k, c, b: {"miss": (cell,)} if (k == "needle" and on(b)) else None)
        self.assertEqual((out["verdict"], out["gates"]["needle-spec"]), ("kill:needle-spec", f"lost:{cell}"))
        out, _ = run("qwen38", lambda k, c, b: {"miss": (cell,)} if (k == "needle" and b == "on2") else None)
        self.assertEqual((out["verdict"], out["gates"]["needle-spec"]), ("open:needle-spec", f"open:{cell}"))
        out, _ = run("qwen38", lambda k, c, b: {"miss": (cell,)} if (k == "needle" and b in ("on2", "off1")) else None)
        self.assertEqual((out["verdict"], out["gates"]["needle-spec"]), ("ship-ok", f"shared-miss:{cell}"))
        out, _ = run("qwen38", lambda k, c, b: {"miss": (cell,)} if (k == "needle" and not on(b)) else None)
        self.assertEqual((out["verdict"], out["gates"]["needle-spec"]), ("ship-ok", "more"))

    def test_caps_and_route(self):
        out, _ = run("flint", lambda k, c, b: {"fail": ("thinking",)} if (k == "caps" and on(b) and c == "spec") else None)
        self.assertEqual((out["verdict"], out["gates"]["caps-spec"]), ("kill:caps-spec", "lower:thinking"))
        out, _ = run("flint", lambda k, c, b: {"fail": ("tool-call",)} if (k == "route" and b == "on1") else None)
        self.assertEqual((out["verdict"], out["gates"]["route-nospec"]), ("open:route-nospec", "open:tool-call"))
        out, _ = run("flint", lambda k, c, b: {"fail": ("logit-bias",)} if k == "route" else None)
        self.assertEqual((out["verdict"], out["gates"]["route-nospec"]), ("ship-ok", "shared:logit-bias"))

    def test_accept(self):
        out, lines = run("qwen38", lambda k, c, b: {"more_verifies": 0.05} if (k == "accept" and on(b)) else None)
        self.assertEqual((out["verdict"], out["gates"]["accept-spec"]), ("kill:accept-spec", "drop"))
        self.assertEqual(out["configs"]["spec"]["accept"]["documents"], 96)
        out, _ = run("qwen38", lambda k, c, b: {"more_verifies": -0.05} if (k == "accept" and on(b)) else None)
        self.assertEqual(out["gates"]["accept-spec"], "ok")
        # both pairs on one document set: the pooled reading needs two disjoint sets
        out, _ = run("qwen38", lambda k, c, b: {"offset": 0} if k == "accept" else None)
        self.assertEqual(out["gates"]["accept-spec"], "unreadable")

    def test_accept_blank_text_is_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            build(d, "qwen38")
            p = os.path.join(d, "accept-spec-on1.json")
            j = json.load(open(p))
            j["cases"][0]["sha"] = J.EMPTY_SHA
            json.dump(j, open(p, "w"))
            out = J.judge("qwen38", d, lambda s: None)
        self.assertEqual((out["verdict"], out["gates"]["accept-spec"]), ("open:accept-spec", "unreadable"))

    def test_cachehit(self):
        def bad(n_on1, n_on2, n_off1=1, n_off2=2):
            n = {"on1": n_on1, "on2": n_on2, "off1": n_off1, "off2": n_off2}
            return run("qwen38", lambda k, c, b: {"bad": n[b]} if k == "cachehit" else None)[0]
        out = bad(8, 9)
        self.assertEqual((out["verdict"], out["gates"]["cachehit-spec"]), ("kill:cachehit-spec", "worse"))
        self.assertEqual(out["configs"]["spec"]["cachehit"]["treatment"], [17, 24])
        self.assertEqual(bad(3, 2)["gates"]["cachehit-spec"], "open")       # pooled higher, one boot above the worse control
        self.assertEqual(bad(3, 3)["gates"]["cachehit-spec"], "open")       # both boots above, Fisher 0.23: not replicated enough
        self.assertEqual(bad(1, 2)["gates"]["cachehit-spec"], "not-worse")
        self.assertEqual(bad(2, 2)["gates"]["cachehit-spec"], "not-worse")  # pooled 4 against 3, no boot above the worse control
        self.assertEqual(bad(0, 0, 8, 8)["gates"]["cachehit-spec"], "not-worse")
        out, _ = run("qwen38", lambda k, c, b: {"blank": 6} if (k == "cachehit" and b == "on1") else None)
        self.assertEqual(out["gates"]["cachehit-spec"], "unreadable")

    def test_missing_receipts_leave_it_open(self):
        for kind, gate in (("needle", "needle-spec"), ("caps", "caps-spec"), ("route", "route-spec"), ("accept", "accept-spec"),
                           ("cachehit", "cachehit-spec")):
            out, _ = run("qwen38", lambda k, c, b: "skip" if (k == kind and b == "on2" and c == "spec") else None)
            self.assertEqual((out["verdict"], out["gates"][gate]), (f"open:{gate}", "unreadable"), kind)
        out, _ = run("qwen38", lambda k, c, b: "skip" if (k == "prefill" and b == "off2" and c == "nospec") else None)
        self.assertTrue(out["verdict"].startswith("open:speed"), out["verdict"])
        self.assertIn("nospec-prefill:58K", out["gates"]["speed"])

    def test_cli_tables(self):
        self.assertEqual([c for c, _ in J.CONFIGS["flint"]], ["nospec", "spec"])
        self.assertEqual([c for c, _ in J.CONFIGS["qwen38"]], ["spec", "nospec"])
        for preset in J.CONFIGS:
            for _, phases in J.CONFIGS[preset]:
                self.assertTrue(set(phases) <= {"caps", "prefill2", "prefill3", "sweep", "forced", "sampled", "accept", "route",
                                                "cachehit", "needle"})


if __name__ == "__main__":
    unittest.main()
