"""CPU tests of the R97-T kernel check: arm outputs fabricated from the emulator's own variants."""
import copy
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "r97s-dump-2026-11-11", "emu"))
import build_inputs as bi  # noqa: E402
import check_kernels as ck  # noqa: E402
import emulate_prefix as ep  # noqa: E402

SERVING, PATCHED = "/x/serving", "/x/patched"


def meta(tree, switch):
    return {"tree": tree, "switch": {"extend_attention": switch, "verify_splitkv": switch}, "fp8_qk": True,
            "upcast_k": False}


def fabricate(cases):
    """base = the kernel-cast (fp8) or BF16 (bf16 pool) variant, on = the scaled variant, off = base; the fp32
    buffer holds the reference, the BF16 buffer its rounding"""
    arms = {"base": {"meta": meta(SERVING, None), "cells": {}}, "off": {"meta": meta(PATCHED, False), "cells": {}},
            "on": {"meta": meta(PATCHED, True), "cells": {}}}
    for c in cases:
        prefix = c["turn"]["pool"]["fp8"][0].shape[0]
        for pool in ("fp8", "bf16"):
            for kernel, info in (("extend", {}), ("splitkv", {"n_splits": ep.choose_n_splits(prefix)}),
                                 ("gqa8", {"n_splits": 4 * ep.choose_n_splits(prefix)})):
                refs, _ = ck.references(ep, c, pool, kernel, info)
                base = refs["kernel" if pool == "fp8" else "bf16"].to(torch.float32)
                on = refs["x256"].to(torch.float32) if pool == "fp8" else base.clone()
                key = "%s|%s|%s" % (c["name"], pool, kernel)
                for name, o in (("base", base), ("off", base.clone()), ("on", on)):
                    arms[name]["cells"][key] = dict(info, o=o.to(torch.bfloat16), o32=o.clone())
    return arms


class CheckKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [bi.synthetic("tiny", 4, 2, 32, 5000, 80, 16, 7, False),
                     bi.synthetic("tiny-sink", 4, 2, 32, 4100, 70, 16, 8, True)]
        cls.arms = fabricate(cls.cases)

    def run_check(self, arms):
        return ck.check(self.cases, arms, ep, SERVING, PATCHED)

    def test_pass(self):
        r = self.run_check(self.arms)
        self.assertEqual(r["verdict"], "pass", ck.lines(r))
        self.assertEqual(len(r["cells"]), 12)
        fp8 = [c for c in r["cells"] if c["pool"] == "fp8"]
        # the synthetic spread puts real mass under 2^-6 and the scaled reference is closer to exact weights (the
        # three-bit rounding of the weights above 2^-6 stays in both)
        self.assertTrue(all(c["subnormal_share"] > 0.05 for c in fp8))
        self.assertTrue(all(c["ref_vs_exact"]["x256"] < c["ref_vs_exact"]["kernel"] for c in fp8))
        # fp32 buffers sit at fp32 precision from the fp64 reference, the BF16 buffers on its rounding
        self.assertTrue(all(c["dist"]["base"]["kernel"] < 1e-6 and c["dist"]["on"]["x256"] < 1e-6 for c in fp8))
        self.assertTrue(all(c["dist"]["on"]["kernel"] > 1e-3 for c in fp8))
        self.assertTrue(all(c["dist_bf16_store"]["on"]["x256"] == 0.0 and c["elements_apart"]["on"] == 0 for c in fp8))
        self.assertEqual((r["first_reading"], r["store_equal_min"]), ("pass", 1.0))
        self.assertTrue(any(ln.startswith("tiny-sink") for ln in ck.lines(r)))

    def test_switch_without_effect_fails(self):
        arms = copy.deepcopy(self.arms)
        for key, cell in arms["on"]["cells"].items():
            cell.update({b: arms["base"]["cells"][key][b].clone() for b in ck.BUFFERS})
        r = self.run_check(arms)
        self.assertEqual(r["verdict"], "fail:6")
        for c in r["cells"]:
            self.assertEqual(c["fails"], ["no-match", "not-moved"] if c["pool"] == "fp8" else [])

    def test_off_arm_must_be_bit_equal(self):
        arms = copy.deepcopy(self.arms)
        for buf in ck.BUFFERS:
            arms = copy.deepcopy(self.arms)
            o = arms["off"]["cells"]["tiny|fp8|splitkv"][buf]
            o[0, 0, 0] = o[0, 0, 0] + 1.0
            r = self.run_check(arms)
            self.assertEqual(r["verdict"], "fail:1", buf)
            self.assertEqual([c["fails"] for c in r["cells"] if c["fails"]], [["off-differs"]])

    def test_bf16_pool_must_not_move(self):
        for buf in ck.BUFFERS:
            arms = copy.deepcopy(self.arms)
            o = arms["on"]["cells"]["tiny-sink|bf16|extend"][buf]
            o[1, 1, 1] = o[1, 1, 1] + 1.0
            r = self.run_check(arms)
            self.assertEqual(r["verdict"], "fail:1", buf)
            self.assertEqual([c["fails"] for c in r["cells"] if c["fails"]], [["on-differs"]])

    def test_bf16_store_noise_is_reported_not_gated(self):
        """one BF16 step on a few elements of the on arm's BF16 buffer: the first reading fails, the verdict holds"""
        arms = copy.deepcopy(self.arms)
        o = arms["on"]["cells"]["tiny|fp8|gqa8"]["o"]
        flat = o.view(-1).view(torch.int16)
        flat[::37] += 1
        r = self.run_check(arms)
        self.assertEqual((r["verdict"], r["first_reading"]), ("pass", "fail:1"))
        row = [c for c in r["cells"] if c["first_reading_fails"]][0]
        self.assertEqual((row["kernel"], row["first_reading_fails"], row["fails"]), ("gqa8", ["no-match"], []))
        self.assertEqual(row["elements_apart"], {"base": 0, "on": len(flat[::37])})
        self.assertLess(r["store_equal_min"], 1.0)
        self.assertTrue(any("first reading no-match" in ln for ln in ck.lines(r)))

    def test_fp32_buffer_off_the_scaled_reference_fails(self):
        arms = copy.deepcopy(self.arms)
        cell = arms["on"]["cells"]["tiny|fp8|extend"]
        cell["o32"] = cell["o32"] * (1.0 + 2e-4)
        r = self.run_check(arms)
        self.assertEqual(r["verdict"], "fail:1")
        self.assertEqual([c["fails"] for c in r["cells"] if c["fails"]], [["no-match"]])

    def test_emulation_that_cannot_tell_the_casts_apart_is_unresolved(self):
        self.assertEqual(ck.judge_fp8(1e-3, 3e-3, 1.5e-3, False, True), ["unresolved"])
        self.assertEqual(ck.judge_fp8(1e-5, 3e-3, 6.0e-5, False, True), [])
        self.assertEqual(ck.judge_fp8(1e-5, 3e-3, 6.2e-5, False, True), ["no-match"])
        self.assertEqual(ck.judge_fp8(1e-4, 3e-3, 1.9e-4, False, True), [])
        self.assertEqual(ck.judge_fp8(1e-4, 3e-3, 2.1e-4, False, True), ["no-match"])

    def test_invalid_arms(self):
        for name, edit, why in (("base", lambda m: m.update(tree=PATCHED), "invalid:base-tree"),
                                ("off", lambda m: m["switch"].update(verify_splitkv=True), "invalid:off-switch"),
                                ("on", lambda m: m["switch"].update(extend_attention=False), "invalid:on-switch"),
                                ("on", lambda m: m.update(upcast_k=True), "invalid:on-qk")):
            arms = copy.deepcopy(self.arms)
            edit(arms[name]["meta"])
            self.assertEqual(self.run_check(arms)["verdict"], why)
        arms = copy.deepcopy(self.arms)
        del arms["on"]["cells"]["tiny|fp8|gqa8"]
        self.assertEqual(self.run_check(arms)["verdict"], "invalid:cells-differ")
        arms = copy.deepcopy(self.arms)
        arms["on"]["cells"]["tiny|fp8|gqa8"]["n_splits"] = 8
        self.assertEqual(self.run_check(arms)["verdict"], "invalid:launch-differs-tiny|fp8|gqa8")

    def test_inputs_round_trip(self):
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "r97t-inputs-%d.pt" % os.getpid())
        torch.save(self.cases, path)
        back = torch.load(path, map_location="cpu", weights_only=True)
        os.unlink(path)
        self.assertEqual(back[0]["extend"]["pool"]["fp8"][0].dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(back[1]["sink"], self.cases[1]["sink"]))
        a, b = back[0]["turn"]["pool"]["fp8"][1], self.cases[0]["turn"]["pool"]["fp8"][1]
        self.assertTrue(torch.equal(a.to(torch.float32), b.to(torch.float32)))


if __name__ == "__main__":
    unittest.main()
