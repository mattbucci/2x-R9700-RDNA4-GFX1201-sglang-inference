#!/usr/bin/env python3
"""allreduce_wire_equality.py on the CPU: the enumeration, the exact rounding, the counters, both loops, the gate.

Needs torch (run it from the serving environment); nothing here opens a GPU or a second process.
"""
import contextlib
import io
import os
import random
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import allreduce_wire_equality as awe  # noqa: E402


def cell(pairs=2 * awe.NEED_PAIRS, **kw):
    return {**awe.new_counters(), "pairs": pairs, **kw}


def stages(**over):
    st = {"rccl-eager-8192": {"rank0": cell(), "rank1": cell()}, "car-graph-32": {"rank0": cell(), "rank1": cell()}}
    st.update(over)
    return st


class Enumeration(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch, self.cpu = torch, torch.device("cpu")

    def test_every_pair_once(self):
        self.assertEqual(awe.MULT % 2, 1)  # an odd multiplier is a bijection of the 2^32 indices
        a, b = awe.pair_bits(self.torch, awe.ALL_PAIRS - 1000, 1 << 18, self.cpu)  # across the wrap
        g = a * 65536 + b
        self.assertEqual(int(self.torch.unique(g).numel()), 1 << 18)
        self.assertEqual(int(g[1000]), 0)  # index 2^32 is index 0
        self.assertEqual(int(g[1001]), awe.MULT)
        self.assertTrue(0 <= int(a.min()) and int(a.max()) <= 65535 and 0 <= int(b.min()) and int(b.max()) <= 65535)

    def test_bit_views_round_trip(self):
        every = self.torch.arange(65536, dtype=self.torch.int64)
        f = awe.as_bf16(self.torch, every)
        self.assertTrue(bool(((awe.bits_of(self.torch, f).to(self.torch.int64) & 0xFFFF) == every).all()))
        self.assertEqual(float(f[0x3F80]), 1.0)
        self.assertEqual(float(f[0xC000]), -2.0)
        self.assertEqual(float(awe.bf16_fraction(0x4049)), float(f[0x4049]))
        self.assertIsNone(awe.bf16_fraction(0x7F80))

    def test_both_ranks_see_the_same_pair(self):
        w0, w1 = awe.dry_world(self.torch, 0), awe.dry_world(self.torch, 1)
        a0, b0, mine0, peer0, local0 = awe.operands(self.torch, w0, 77, 512)
        a1, b1, mine1, peer1, local1 = awe.operands(self.torch, w1, 77, 512)
        bits = lambda t: awe.bits_of(self.torch, t).tolist()  # noqa: E731
        self.assertEqual(bits(mine0), bits(peer1))
        self.assertEqual(bits(mine1), bits(peer0))
        self.assertEqual(bits(local0), bits(local1))
        self.assertEqual(a0.tolist(), a1.tolist())


class Rounding(unittest.TestCase):
    def test_ties_overflow_and_subnormals(self):
        F, half = awe.Fraction, awe.Fraction(1, 256)
        self.assertEqual(awe.round_bf16(1 + half), 0x3F80)  # tie to the even neighbour below
        self.assertEqual(awe.round_bf16(1 + 3 * half), 0x3F82)  # tie to the even neighbour above
        self.assertEqual(awe.round_bf16(-(1 + 3 * half)), 0xBF82)
        self.assertEqual(awe.round_bf16(awe.bf16_fraction(0x7F7F) + F(2) ** 119), 0x7F80)  # the top midpoint overflows
        self.assertEqual(awe.round_bf16(F(1, 1 << 134)), 0)  # half the smallest subnormal: tie to zero
        self.assertEqual(awe.round_bf16(F(3, 1 << 134)), 2)
        self.assertEqual(awe.exact_sum_bits(0x8000, 0x8000), 0x8000)
        self.assertEqual(awe.exact_sum_bits(0x8000, 0x0000), 0x0000)
        self.assertEqual(awe.exact_sum_bits(0x3F80, 0xBF80), 0x0000)

    def test_fp32_detour_is_the_correctly_rounded_sum(self):
        import torch

        pairs = awe.dry_pairs(random.Random(11))
        pa = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
        pb = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
        fa, fb = awe.as_bf16(torch, pa), awe.as_bf16(torch, pb)
        detour = (awe.bits_of(torch, (fa.float() + fb.float()).to(torch.bfloat16)).to(torch.int64) & 0xFFFF).tolist()
        self.assertEqual(detour, [awe.exact_sum_bits(x, y) for x, y in pairs])


class Counters(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def bf(self, *bits):
        return awe.as_bf16(self.torch, self.torch.tensor(list(bits)))

    def test_nan_zero_sign_and_examples(self):
        idx = self.torch.arange(5)
        c = awe.new_counters()
        w, d = self.bf(0x3F80, 0x7FC0, 0x7FC0, 0x3F80, 0x0000), self.bf(0x3F80, 0x7FC1, 0x3F80, 0x7FC0, 0x8000)
        awe.tally(self.torch, c, w, d, d.clone(), idx, idx)
        self.assertEqual((c["pairs"], c["wire_vs_detour"], c["wire_vs_local"], c["detour_vs_local"]), (5, 3, 3, 0))
        self.assertEqual(c["nan_payload_only"], 1)
        self.assertEqual([e["wire"] for e in c["examples"]], ["7fc0", "3f80", "0000"])

    def test_without_a_detour(self):
        idx = self.torch.arange(2)
        c = awe.new_counters()
        awe.tally(self.torch, c, self.bf(0x3F80, 0x4000), None, self.bf(0x3F80, 0x4001), idx, idx)
        self.assertEqual((c["wire_vs_local"], c["wire_vs_detour"], c["detour_vs_local"]), (1, 0, 0))

    def test_normal_operands_counted_apart(self):
        a = self.torch.tensor([0x0001, 0x3F80, 0x7F80, 0x3F80])
        b = self.torch.tensor([0x0002, 0x3F80, 0x3F80, 0x4000])
        c = awe.new_counters()
        awe.tally(self.torch, c, self.bf(0x0000, 0x4000, 0x7FC0, 0x4041), None, self.bf(0x0003, 0x4000, 0x7F80, 0x4040), a, b)
        self.assertEqual((c["wire_vs_local"], c["wire_vs_local_normal"]), (3, 1))


class Loops(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def both(self, wire, rank=0):
        w = awe.dry_world(self.torch, rank)
        return (awe.run_eager(self.torch, w, wire, 2048, 3, 41),
                awe.run_graph(self.torch, w, wire, 96, 2, 9, slots=5))

    def test_exact_wire(self):
        for rank in (0, 1):
            e, g = self.both(awe.FakeWire(self.torch), rank)
            for c, pairs in ((e, 3 * 2048), (g, 2 * 5 * 96)):
                self.assertEqual((c["pairs"], c["wire_vs_detour"], c["wire_vs_local"], c["detour_vs_local"]), (pairs, 0, 0, 0))

    def test_truncating_adder_is_caught(self):
        for c in self.both(awe.FakeWire(self.torch, "truncate")):
            self.assertGreater(c["wire_vs_detour"], 0)
            self.assertEqual(c["wire_vs_detour"], c["wire_vs_local"])
            self.assertEqual(c["detour_vs_local"], 0)

    def test_lost_peer_fails_the_control(self):
        for c in self.both(awe.FakeWire(self.torch, "lost-peer")):
            self.assertGreater(c["detour_vs_local"], 0)

    def test_declined_forms(self):
        e, g = self.both(awe.FakeWire(self.torch, decline_above=100))
        self.assertTrue(e["declined"] and g["declined"] and e["pairs"] == 0 and g["pairs"] == 0)
        e = awe.run_eager(self.torch, awe.dry_world(self.torch, 0), awe.FakeWire(self.torch, "truncate", decline_above=2048 * 2), 2048, 3, 41)
        g = awe.run_graph(self.torch, awe.dry_world(self.torch, 1), awe.FakeWire(self.torch, "truncate", decline_above=96 * 2), 96, 2, 9, slots=5)
        for c in (e, g):
            self.assertTrue(c["detour_declined"])
            self.assertGreater(c["wire_vs_local"], 0)
            self.assertEqual(c["wire_vs_detour"], 0)


class Gate(unittest.TestCase):
    def test_identical(self):
        g = awe.summarize(stages())
        self.assertEqual(g["gate"], "identical")
        self.assertEqual(g["pairs"], {"rccl": 2 * awe.NEED_PAIRS, "car": 2 * awe.NEED_PAIRS})

    def test_differs_and_special_only(self):
        g = awe.summarize(stages(**{"car-graph-9": {"rank0": cell(9, wire_vs_detour=4, wire_vs_local=4, wire_vs_local_normal=1), "rank1": cell(9)}}))
        self.assertEqual(g["gate"], "differs:4")
        g = awe.summarize(stages(**{"car-graph-9": {"rank0": cell(9, wire_vs_detour=4, wire_vs_local=4), "rank1": cell(9)}}))
        self.assertEqual(g["gate"], "differs:4:special-only")

    def test_unreadable(self):
        g = awe.summarize(stages(**{"rccl-graph-1": {"rank0": cell(9, detour_vs_local=2), "rank1": cell(9)}}))
        self.assertEqual(g["gate"], "unreadable:harness-rccl-graph-1")
        g = awe.summarize(stages(**{"car-graph-32": {"rank0": {"error": "x"}, "rank1": cell()}}))
        self.assertEqual(g["gate"], "unreadable:error-car-graph-32:short-car")
        g = awe.summarize({"rccl-eager-8192": {"rank0": cell(), "rank1": cell()}})
        self.assertEqual(g["gate"], "unreadable:short-car")
        g = awe.summarize(stages(**{"car-graph-32": {"rank0": cell(), "rank1": cell(10)}}))  # the smaller rank counts
        self.assertEqual(g["gate"], "unreadable:short-car")

    def test_local_sum_device_cell(self):
        import torch

        c = awe.local_device_check(torch, torch.device("cpu"), windows=2, n=1 << 10)
        self.assertEqual((c["checked"], c["pairs"], c["detour_vs_local"]), (2 << 10, 0, 0))
        g = awe.summarize(stages(**{"local-sum-device": {"rank0": c, "rank1": {**c, "detour_vs_local": 1}}}))
        self.assertEqual(g["gate"], "unreadable:harness-local-sum-device")
        self.assertEqual(awe.summarize(stages(**{"local-sum-device": {"rank0": c, "rank1": c}}))["gate"], "identical")

    def test_declined_and_local_only(self):
        g = awe.summarize(stages(**{"car-eager-2048": {"rank0": cell(0, declined=True), "rank1": cell(0, declined=True)},
                                    "car-eager-1024": {"rank0": cell(7, detour_declined=True), "rank1": cell(7, detour_declined=True)}}))
        self.assertEqual(g["gate"], "identical:declined-car-eager-2048:local-only-car-eager-1024")

    def test_dry_mode_passes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = awe.dry_cpu()
        self.assertEqual(rc, 0, out.getvalue())
        self.assertNotIn("FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
