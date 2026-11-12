import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emu"))
import emulate_prefix as ep  # noqa: E402


def literal_extend(q, kp, vp, kc, vc, sink, scale, first_row, variant, fp8_qk=True):
    """extend_attention's loops transcribed tile by tile (prefix loop, then the chunk loop, then the sink)."""
    n, hq, d = q.shape
    g = hq // kc.shape[1]
    out = torch.zeros((n, hq, d), dtype=torch.float64)
    for i in range(n):
        for h in range(hq):
            kh = h // g
            e_max, deno, acc = -math.inf, 0.0, torch.zeros(d, dtype=torch.float64)
            qp = ep.e4m3(q[i, h]) if fp8_qk else q[i, h]
            for start in range(0, kp.shape[0], ep.EXTEND_TILE):
                k, v = kp[start:start + ep.EXTEND_TILE, kh], vp[start:start + ep.EXTEND_TILE, kh]
                qk = k @ qp * scale
                n_e_max = max(float(qk.max()), e_max)
                re = math.exp(e_max - n_e_max) if e_max != -math.inf else 0.0
                p = torch.exp(qk - n_e_max)
                deno = deno * re + float(p.sum())
                acc = acc * re + ep.cast_weight(p, variant) @ v
                e_max = n_e_max
            for start in range(0, kc.shape[0], ep.EXTEND_TILE):
                k, v = kc[start:start + ep.EXTEND_TILE, kh], vc[start:start + ep.EXTEND_TILE, kh]
                qk = k @ q[i, h] * scale
                qk = qk.masked_fill(start + torch.arange(k.shape[0]) > first_row + i, -math.inf)
                if float(qk.max()) == -math.inf:
                    continue
                n_e_max = max(float(qk.max()), e_max)
                re = math.exp(e_max - n_e_max) if e_max != -math.inf else 0.0
                p = torch.exp(qk - n_e_max)
                deno = deno * re + float(p.sum())
                acc = acc * re + ep.bf16(p) @ v
                e_max = n_e_max
            if sink is not None:
                deno += math.exp(float(sink[h]) - e_max)
            out[i, h] = acc / deno
    return out


def literal_splitkv(q, kp, vp, kc, vc, sink, scale, variant, n_splits):
    """verify_splitkv's stage 1 per split and its stage 2 merge, transcribed."""
    n, hq, d = q.shape
    g = hq // kc.shape[1]
    prefix = kp.shape[0]
    per = ep.split_len(prefix, n_splits)
    out = torch.zeros((n, hq, d), dtype=torch.float64)
    for i in range(n):
        for h in range(hq):
            kh = h // g
            qp = ep.e4m3(q[i, h])
            att, lse = [], []
            for s in range(n_splits):
                lo, hi = per * s, min(per * s + per, prefix)
                if hi <= lo:
                    lse.append(-math.inf)
                    att.append(torch.zeros(d, dtype=torch.float64))
                    continue
                e_max, e_sum, acc = -math.inf, 0.0, torch.zeros(d, dtype=torch.float64)
                for start in range(lo, hi, ep.SPLIT_TILE):
                    k, v = kp[start:min(start + ep.SPLIT_TILE, hi), kh], vp[start:min(start + ep.SPLIT_TILE, hi), kh]
                    qk = k @ qp * scale
                    n_e_max = max(float(qk.max()), e_max)
                    re = math.exp(e_max - n_e_max) if e_max != -math.inf else 0.0
                    p = torch.exp(qk - n_e_max)
                    acc = acc * re + ep.cast_weight(p, variant) @ v
                    e_sum = e_sum * re + float(p.sum())
                    e_max = n_e_max
                att.append(acc / e_sum)
                lse.append(e_max + math.log(e_sum))
            m_p = max(lse)
            w = [math.exp(x - m_p) if x != -math.inf else 0.0 for x in lse]
            o_prefix = sum(a * x for a, x in zip(att, w)) / sum(w)
            lse_prefix = m_p + math.log(sum(w))
            qk = kc[: i + 1, kh] @ q[i, h] * scale
            m_d = float(qk.max())
            pd = torch.exp(qk - m_d)
            o_draft = pd @ vc[: i + 1, kh] / float(pd.sum())
            lse_draft = m_d + math.log(float(pd.sum()))
            m = max(lse_prefix, lse_draft)
            ws = 0.0
            if sink is not None:
                m = max(m, float(sink[h]))
                ws = math.exp(float(sink[h]) - m)
            wp, wd = math.exp(lse_prefix - m), math.exp(lse_draft - m)
            out[i, h] = (o_prefix * wp + o_draft * wd) / (wp + wd + ws)
    return out


def tensors(prefix, m, n, seed, spread=4.0):
    gen = torch.Generator().manual_seed(seed)
    hq, hkv, d = 4, 2, 16
    q = ep.bf16(torch.randn((n, hq, d), generator=gen, dtype=torch.float64) * spread)
    kp = ep.e4m3(torch.randn((prefix, hkv, d), generator=gen, dtype=torch.float64))
    vp = ep.e4m3(torch.randn((prefix, hkv, d), generator=gen, dtype=torch.float64))
    kc = ep.bf16(torch.randn((m, hkv, d), generator=gen, dtype=torch.float64))
    vc = ep.bf16(torch.randn((m, hkv, d), generator=gen, dtype=torch.float64))
    sink = torch.randn(hq, generator=gen, dtype=torch.float64)
    return q, kp, vp, kc, vc, sink, d ** -0.5


class EmulatePrefixTest(unittest.TestCase):
    def test_split_plan_matches_the_kernel_arithmetic(self):
        self.assertEqual([ep.choose_n_splits(x) for x in (4095, 4096, 8191, 8192, 197000)], [4, 8, 8, 16, 16])
        self.assertEqual(ep.split_len(6000, 8), 768)
        self.assertEqual(ep.split_len(300, 4), 96)
        self.assertEqual(ep.split_len(197000, 16), 12320)

    def test_extend_walk_equals_the_transcribed_loops(self):
        for prefix, m, n, first in ((300, 150, 6, 144), (0, 200, 5, 195), (64, 10, 4, 3), (1, 14, 14, 0)):
            q, kp, vp, kc, vc, sink, scale = tensors(prefix, m, n, 11 + prefix)
            for sk in (sink, None):
                got, _ = ep.attend(q, kp, vp, kc, vc, sk, scale, "extend", first_row=first)
                for v in ep.VARIANTS:
                    ref = literal_extend(q, kp, vp, kc, vc, sk, scale, first, v)
                    self.assertLess(float((got[v] - ref).abs().max()), 1e-11, (prefix, v))

    def test_splitkv_walk_equals_the_transcribed_stages(self):
        for prefix, n, splits in ((300, 6, 4), (1000, 5, 8), (97, 4, 4)):
            q, kp, vp, kc, vc, sink, scale = tensors(prefix, n, n, 23 + prefix)
            for sk in (sink, None):
                got, _ = ep.attend(q, kp, vp, kc, vc, sk, scale, "splitkv", n_splits=splits)
                for v in ep.VARIANTS:
                    ref = literal_splitkv(q, kp, vp, kc, vc, sk, scale, v, splits)
                    self.assertLess(float((got[v] - ref).abs().max()), 1e-11, (prefix, v))

    def test_exact_weights_do_not_depend_on_the_walk(self):
        q, kp, vp, kc, vc, sink, scale = tensors(500, 8, 8, 5)
        a, _ = ep.attend(q, kp, vp, kc, vc, sink, scale, "splitkv", fp8_qk=False)
        b = ep.ideal(q, kp, vp, kc, vc, sink, scale)
        self.assertLess(float((a["exact"] - b).abs().max()), 1e-11)
        self.assertGreater(float((a["kernel"] - b).abs().max()), 1e-6)

    def test_the_cast_loses_tail_mass_and_the_scaled_cast_recovers_most(self):
        q, kp, vp, kc, vc, sink, scale = tensors(4000, 8, 8, 7, spread=10.0)
        got, shares = ep.attend(q, kp, vp, kc, vc, sink, scale, "extend")
        lost = float(torch.linalg.vector_norm(got["kernel"] - got["exact"]))
        kept = float(torch.linalg.vector_norm(got["x256"] - got["exact"]))
        self.assertGreater(lost, 0.0)
        self.assertLess(kept, lost)
        self.assertAlmostEqual(shares["normal"] + shares["subnormal"] + shares["zero"], shares["prefix"], places=9)
        self.assertGreater(shares["subnormal"] + shares["zero"], 0.0)

    def test_distance_rounds_the_reference_to_bf16(self):
        ref = torch.randn((3, 8), dtype=torch.float64)
        dump = ref.to(torch.bfloat16)
        d = ep.distance(dump, ref)
        self.assertEqual(d["rel_l2"], 0.0)
        self.assertEqual(d["equal_share"], 1.0)
        self.assertGreater(ep.distance(dump, ref * 1.01)["rel_l2"], 1e-3)


if __name__ == "__main__":
    unittest.main()
