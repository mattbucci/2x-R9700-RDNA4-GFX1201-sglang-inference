import enum
import os
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hook"))
import flint_stage_dump as fsd  # noqa: E402


class Mode(enum.IntEnum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 5


class Batch:
    def __init__(self, mode, prefix, rows):
        self.forward_mode = mode
        self.extend_prefix_lens_cpu = [prefix]
        self.extend_seq_lens_cpu = [rows]
        self.rids = ["r1"]


class Opaque:
    pass


class Radix(nn.Module):
    def forward(self, q, k, v, forward_batch, sinks=None):
        return q + 1.0


class OProj(nn.Module):
    def forward(self, x):
        return x * 2.0, None


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Radix()
        self.o_proj = OProj()

    def forward(self, positions, hidden_states, forward_batch):
        q = hidden_states + 0.25
        out, _ = self.o_proj(self.attn(q, q * 3.0, q * 5.0, forward_batch, sinks=None))
        return out


class Mlp(nn.Module):
    def forward(self, hidden_states, forward_batch):
        if isinstance(hidden_states, Opaque):
            return torch.zeros(3, 4)
        out = hidden_states - 7.0
        hidden_states.add_(100.0)  # a later in-place edit must not reach the dump
        return out


class Layer(nn.Module):
    def __init__(self, opaque=False):
        super().__init__()
        self.self_attn = Attn()
        self.mlp = Mlp()
        self.opaque = opaque

    def forward(self, positions, hidden_states, forward_batch, captured_last_layer_outputs=None):
        h = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        return self.mlp(Opaque() if self.opaque else h, forward_batch)


class Core(nn.Module):
    def __init__(self, opaque=False):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer(opaque)])

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        h = input_ids.float().unsqueeze(1).repeat(1, 4)
        for layer in self.layers:
            h = layer(positions, h, forward_batch, captured_last_layer_outputs=None)
        return h


class Model(nn.Module):
    def __init__(self, opaque=False):
        super().__init__()
        self.model = Core(opaque)

    def forward(self, input_ids, forward_batch):
        return self.model(input_ids, torch.arange(len(input_ids)), forward_batch, None, pp_proxy_tensors=None)


def files(root, rank=1):
    return sorted(n for n in os.listdir(os.path.join(root, "rank%d" % rank)) if n.startswith("fwd"))


class StageDumpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.environ.pop(fsd.ENV_DIR, None)
        os.environ.pop(fsd.ENV_ROWS, None)
        os.environ.pop(fsd.ENV_LAYERS, None)
        os.environ.pop(fsd.ENV_KINDS, None)
        os.environ.pop(fsd.ENV_FORWARDS, None)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop(fsd.ENV_DIR, None)
        os.environ.pop(fsd.ENV_ROWS, None)
        os.environ.pop(fsd.ENV_LAYERS, None)
        os.environ.pop(fsd.ENV_KINDS, None)
        os.environ.pop(fsd.ENV_FORWARDS, None)

    def test_switch_off_installs_nothing(self):
        m = Model()
        self.assertIsNone(fsd.maybe_install(m, 0))
        self.assertEqual(len(m.model._forward_pre_hooks), 0)
        self.assertEqual(len(m.model.layers[0].mlp._forward_hooks), 0)
        os.environ[fsd.ENV_DIR] = "  "
        self.assertIsNone(fsd.maybe_install(m, 0))

    def test_layer_and_kind_filters(self):
        os.environ[fsd.ENV_DIR] = self.root
        os.environ[fsd.ENV_LAYERS] = "1"
        os.environ[fsd.ENV_KINDS] = "q, k,v,attn"
        m = Model()
        fsd.maybe_install(m, 0)
        lines = open(os.path.join(self.root, "rank0", "ACTIVE")).read().splitlines()
        self.assertEqual(lines[0].split()[2:], ["rank", "0", "layers", "1", "max_rows", "64"])
        self.assertEqual(lines[1], "kinds q,k,v,attn")
        m(torch.arange(5) + 3, Batch(Mode.EXTEND, 0, 5))
        got = torch.load(os.path.join(self.root, "rank0", "fwd0000.pt"), weights_only=True)
        self.assertEqual(sorted(got["stages"]), ["L01-attn", "L01-k", "L01-q", "L01-v"])
        self.assertEqual(got["meta"]["non_tensor"], {})

    def test_unknown_kind_or_missing_layer_is_refused(self):
        with self.assertRaises(ValueError):
            fsd.install(Model(), 0, self.root, kinds=["q", "attention"])
        with self.assertRaises(ValueError):
            fsd.install(Model(), 0, self.root, layers=["0", "7"])

    def test_extend_forward_dumps_every_stage(self):
        os.environ[fsd.ENV_DIR] = self.root
        m = Model()
        with self.assertLogs(fsd.logger, "INFO") as logs:
            d = fsd.maybe_install(m, 1)
        self.assertIn("[flint-stage-dump] active: rank 1, 2 layers hooked, EXTEND forwards of 1..64 rows", logs.output[0])
        marker = open(os.path.join(self.root, "rank1", "ACTIVE")).read().splitlines()[0].split()
        self.assertEqual(marker[2:], ["rank", "1", "layers", "2", "max_rows", "64"])
        self.assertEqual(marker[1], str(os.getpid()))
        ids = torch.arange(15) + 3
        out = m(ids, Batch(Mode.EXTEND, 0, 15))
        self.assertEqual(files(self.root), ["fwd0000.pt"])
        got = torch.load(os.path.join(self.root, "rank1", "fwd0000.pt"))
        want = ["L%02d-%s" % (i, s) for i in (0, 1) for s in ("attn_in", "q", "k", "v", "attn", "oproj", "mlp_in", "mlp")]
        self.assertEqual(list(got["stages"]), want + ["final"])
        self.assertEqual(got["meta"]["rows"], 15)
        self.assertEqual(got["meta"]["input_ids"], ids.tolist())
        self.assertEqual(got["meta"]["extend_prefix_lens"], [0])
        self.assertEqual(got["meta"]["extend_seq_lens"], [15])
        self.assertEqual(got["meta"]["mode"], "EXTEND")
        self.assertEqual(got["meta"]["non_tensor"], {})
        h0 = ids.float().unsqueeze(1).repeat(1, 4)
        s = got["stages"]
        self.assertTrue(torch.equal(s["L00-attn_in"], h0))
        self.assertTrue(torch.equal(s["L00-q"], h0 + 0.25))
        self.assertTrue(torch.equal(s["L00-k"], (h0 + 0.25) * 3.0))
        self.assertTrue(torch.equal(s["L00-v"], (h0 + 0.25) * 5.0))
        self.assertTrue(torch.equal(s["L00-attn"], h0 + 1.25))
        self.assertTrue(torch.equal(s["L00-oproj"], (h0 + 1.25) * 2.0))
        # the MLP edits its input in place after the tap: the dump keeps the value at entry
        self.assertTrue(torch.equal(s["L00-mlp_in"], (h0 + 1.25) * 2.0))
        self.assertTrue(torch.equal(s["L00-mlp"], (h0 + 1.25) * 2.0 - 7.0))
        self.assertTrue(torch.equal(s["L01-attn_in"], s["L00-mlp"]))
        self.assertTrue(torch.equal(s["final"], out))
        self.assertEqual(d.count, 1)
        self.assertFalse(d.armed)

    def test_other_modes_and_large_batches_are_not_dumped(self):
        d = fsd.install(Model(), 1, self.root, max_rows=8)
        m = Model()
        d2 = fsd.install(m, 0, self.root, max_rows=8)
        m(torch.arange(4), Batch(Mode.DECODE, 0, 4))
        m(torch.arange(4), Batch(Mode.TARGET_VERIFY, 0, 4))
        m(torch.arange(9), Batch(Mode.EXTEND, 0, 9))
        self.assertEqual(files(self.root, 0), [])
        self.assertFalse(d2.armed)
        m(torch.arange(8), Batch(Mode.EXTEND, 0, 8))
        m(torch.arange(3), Batch(Mode.EXTEND, 5, 3))
        self.assertEqual(files(self.root, 0), ["fwd0000.pt", "fwd0001.pt"])
        got = torch.load(os.path.join(self.root, "rank0", "fwd0001.pt"))
        self.assertEqual(got["meta"]["extend_prefix_lens"], [5])
        self.assertEqual(got["meta"]["forward"], 1)
        self.assertEqual(files(self.root, 1), [])
        self.assertEqual(d.count, 0)

    def test_a_decode_after_an_extend_is_not_armed(self):
        m = Model()
        d = fsd.install(m, 0, self.root)
        m(torch.arange(4), Batch(Mode.EXTEND, 0, 4))
        m(torch.arange(1), Batch(Mode.DECODE, 0, 1))
        self.assertEqual(files(self.root, 0), ["fwd0000.pt"])
        self.assertFalse(d.armed)

    def test_forward_cap(self):
        m = Model()
        d = fsd.install(m, 0, self.root)
        d.max_forwards = 2
        for _ in range(4):
            m(torch.arange(4), Batch(Mode.EXTEND, 0, 4))
        self.assertEqual(files(self.root, 0), ["fwd0000.pt", "fwd0001.pt"])

    def test_forward_cap_from_the_environment(self):
        m = Model()
        os.environ[fsd.ENV_DIR] = self.root
        os.environ[fsd.ENV_FORWARDS] = "3"
        d = fsd.maybe_install(m, 1)
        self.assertEqual(d.max_forwards, 3)
        for _ in range(5):
            m(torch.arange(4), Batch(Mode.EXTEND, 0, 4))
        self.assertEqual(files(self.root, 1), ["fwd0000.pt", "fwd0001.pt", "fwd0002.pt"])
        with open(os.path.join(self.root, "rank1", "ACTIVE")) as f:
            self.assertEqual(f.read().splitlines()[2], "max_forwards 3")
        os.environ[fsd.ENV_FORWARDS] = "0"
        with self.assertRaises(ValueError):
            fsd.maybe_install(Model(), 0)

    def test_opaque_stage_input_is_named_not_dumped(self):
        m = Model(opaque=True)
        fsd.install(m, 0, self.root)
        m(torch.arange(3), Batch(Mode.EXTEND, 0, 3))
        got = torch.load(os.path.join(self.root, "rank0", "fwd0000.pt"))
        self.assertEqual(got["meta"]["non_tensor"], {"L01-mlp_in": "Opaque"})
        self.assertNotIn("L01-mlp_in", got["stages"])
        self.assertIn("L01-mlp", got["stages"])


if __name__ == "__main__":
    unittest.main()
