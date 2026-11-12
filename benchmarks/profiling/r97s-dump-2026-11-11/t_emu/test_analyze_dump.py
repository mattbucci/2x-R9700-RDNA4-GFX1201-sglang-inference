import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emu"))
import analyze_dump as ad  # noqa: E402
import emulate_prefix as ep  # noqa: E402

D, TP, HQ, HKV = 16, 2, 4, 2
CHUNK, TAIL, MIN_PREFIX = 96, 8, 64
DEPTHS, LAYERS = (200, 330), (0, 3)


def prompts():
    gen = torch.Generator().manual_seed(1)
    out = {"tail": TAIL, "prompts": {}}
    for p in DEPTHS:
        out["prompts"][str(p)] = {"prefix": p, "ids": torch.randint(0, 1000, (p + TAIL,), generator=gen).tolist()}
    return out


def sinks():
    gen = torch.Generator().manual_seed(2)
    return {layer: torch.randn(HQ * TP, generator=gen, dtype=torch.float64) for layer in LAYERS}


def client(pr):
    reqs = []
    for p in DEPTHS:
        reqs += [{"depth": p, "state": "cold", "prompt_tokens": p, "cached_tokens": 0},
                 {"depth": p, "state": "turn", "prompt_tokens": p + TAIL, "cached_tokens": p},
                 {"depth": p, "state": "flushed", "prompt_tokens": p + TAIL, "cached_tokens": 0}]
    return {"requests": reqs}


def serve(root, pool, weights, pr, sk, turn_weights=None):
    """Write the dump a server would leave: chunked extend forwards, a split-KV turn, a foreign forward per rank."""
    cast = ep.e4m3 if pool == "fp8" else (lambda x: x)
    scale = D ** -0.5
    for rank in range(TP):
        d = os.path.join(root, "rank%d" % rank)
        os.makedirs(d)
        count = 0

        def write(ids, prefix, stages):
            nonlocal count
            meta = {"forward": count, "rank": rank, "pid": 1, "mode": "EXTEND", "rows": len(ids), "input_ids": list(ids),
                    "extend_prefix_lens": [prefix], "extend_seq_lens": [len(ids)], "rids": ["x"], "non_tensor": {}}
            torch.save({"meta": meta, "stages": stages}, os.path.join(d, "fwd%04d.pt" % count))
            count += 1
        write([7, 8, 9], 0, {})  # a health check
        for p in DEPTHS:
            ids = pr["prompts"][str(p)]["ids"]
            tab = {}
            for layer in LAYERS:
                gen = torch.Generator().manual_seed(100 * rank + layer + p)
                tab[layer] = (ep.bf16(torch.randn((p + TAIL, HQ, D), generator=gen, dtype=torch.float64) * 6),
                              ep.bf16(torch.randn((p + TAIL, HKV, D), generator=gen, dtype=torch.float64)),
                              ep.bf16(torch.randn((p + TAIL, HKV, D), generator=gen, dtype=torch.float64)))

            def forward(lo, hi, walk, w):
                st = {}
                for layer in LAYERS:
                    q, k, v = tab[layer]
                    out, _ = ep.attend(q[lo:hi], cast(k[:lo]), cast(v[:lo]), k[lo:hi], v[lo:hi],
                                       sk[layer].chunk(TP)[rank], scale, walk, fp8_qk=(pool == "fp8"), variants=(w,),
                                       n_splits=None)
                    name = "L%02d-" % layer
                    st[name + "q"] = q[lo:hi].reshape(hi - lo, -1).to(torch.bfloat16)
                    st[name + "k"] = k[lo:hi].reshape(hi - lo, -1).to(torch.bfloat16)
                    st[name + "v"] = v[lo:hi].reshape(hi - lo, -1).to(torch.bfloat16)
                    st[name + "attn"] = out[w].reshape(hi - lo, -1).to(torch.bfloat16)
                write(ids[lo:hi], lo, st)
            for lo in range(0, p, CHUNK):
                forward(lo, min(p, lo + CHUNK), "extend", weights)
            forward(p, p + TAIL, "splitkv", turn_weights or weights)
            write([5], 0, {})
            for lo in range(0, p + TAIL, CHUNK):
                forward(lo, min(p + TAIL, lo + CHUNK), "extend", weights)


class AnalyzeDumpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pr, self.sk = prompts(), sinks()

    def tearDown(self):
        self.tmp.cleanup()

    def run_arms(self, spec):
        arms = []
        for name, pool, weights in spec:
            root = os.path.join(self.tmp.name, name)
            serve(root, pool, weights, self.pr, self.sk)
            arms.append((name, root, client(self.pr), pool))
        return ad.analyze(arms, self.sk, self.pr, list(LAYERS), tp=TP, head_dim=D, tail=TAIL, min_prefix=MIN_PREFIX)

    def test_kernel_cast_dump_is_confirmed_against_the_control(self):
        rep = self.run_arms([("preset", "fp8", "kernel"), ("bf16-kv", "bf16", "bf16")])
        self.assertEqual(rep["verdict"], "confirmed")
        self.assertEqual(len(rep["cells"]), len(DEPTHS) * len(LAYERS))
        for c in rep["cells"]:
            self.assertEqual((c["state"], c["floor_from"], c["floor"]), ("confirmed", "bf16-kv", ad.FLOOR_MIN))
            self.assertEqual(c["kernel"], 0.0)
            self.assertGreater(c["exact"], 5 * ad.FLOOR_MIN)
        r0 = rep["arms"]["preset"]["ranks"]["rank0"]
        self.assertEqual(r0["foreign"], 1 + len(DEPTHS))
        layer = r0["depths"]["200"]["layers"]["0"]
        self.assertEqual((layer["turn"]["walk"], layer["turn"]["prefix"], layer["turn"]["rows"]), ("splitkv", 200, TAIL))
        self.assertEqual((layer["cold-tail"]["walk"], layer["cold-tail"]["prefix"]), ("extend", 192))
        self.assertEqual(layer["flushed-tail"]["prefix"], 192)
        self.assertEqual(layer["no-pool"]["variants"]["exact"]["num2"], 0.0)
        self.assertEqual(layer["first_chunk_repeat_equal_share"], 1.0)
        self.assertEqual(layer["cold-tail"]["variants"]["kernel"]["num2"], 0.0)
        self.assertGreater(layer["cold-tail"]["variants"]["exact"]["num2"], 0.0)
        text = "\n".join(ad.lines(rep))
        self.assertIn("R97S VERDICT confirmed", text)
        self.assertIn("R97S preset depth 200 L00 | no-pool[extend 0+8]", text)

    def test_exact_weight_dump_is_not_present(self):
        rep = self.run_arms([("preset", "fp8", "exact"), ("bf16-kv", "bf16", "bf16")])
        self.assertEqual(rep["verdict"], "not-present")
        self.assertTrue(all(c["exact"] == 0.0 for c in rep["cells"]))

    def test_a_dump_neither_reference_matches_is_unexplained(self):
        rep = self.run_arms([("preset", "fp8", "x256"), ("bf16-kv", "bf16", "bf16")])
        self.assertEqual(rep["verdict"], "unexplained")

    def test_a_control_that_misses_raises_the_floor(self):
        # the control's dump is not what the emulation predicts: its distance is the floor, and nothing is confirmed
        rep = self.run_arms([("preset", "fp8", "kernel"), ("bf16-kv", "bf16", "exact")])
        self.assertTrue(all(c["floor"] > ad.FLOOR_MIN for c in rep["cells"]))
        self.assertTrue(all(c["floor_from"] == "bf16-kv" for c in rep["cells"]))

    def test_without_a_control_the_floor_is_the_no_pool_residual(self):
        rep = self.run_arms([("preset", "fp8", "kernel")])
        self.assertEqual(rep["verdict"], "confirmed")
        self.assertTrue(all(c["floor_from"] == "no-pool" for c in rep["cells"]))

    def test_a_missing_forward_is_unreadable(self):
        root = os.path.join(self.tmp.name, "preset")
        serve(root, "fp8", "kernel", self.pr, self.sk)
        os.remove(os.path.join(root, "rank1", "fwd0002.pt"))
        rep = ad.analyze([("preset", root, client(self.pr), "fp8")], self.sk, self.pr, list(LAYERS), tp=TP, head_dim=D,
                         tail=TAIL, min_prefix=MIN_PREFIX)
        self.assertTrue(rep["verdict"].startswith("unreadable:preset-rank1-200-cold-stops-at-"), rep["verdict"])

    def test_the_server_token_count_must_match_the_ids(self):
        c = client(self.pr)
        c["requests"][0]["prompt_tokens"] += 1
        with self.assertRaises(ValueError):
            ad.build_requests(self.pr, c)


if __name__ == "__main__":
    unittest.main()
