"""CPU checks for scripts/quantize/quantize_fp8_block.py: scale-grid shapes, ragged blocks, error bound, key rule."""
import importlib.util, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qb", ROOT / "scripts/quantize/quantize_fp8_block.py")
qb = importlib.util.module_from_spec(spec); sys.modules["qb"] = qb; spec.loader.exec_module(qb)


class _Slice:
    def __init__(self, shape, dtype="BF16"): self.s, self.d = shape, dtype
    def get_shape(self): return list(self.s)
    def get_dtype(self): return self.d


def test_shapes_match_vendor_grid():
    torch.manual_seed(0)
    for (o, i), grid in (((10240, 5120), (80, 40)), ((5120, 17408), (40, 136)), ((1024, 5120), (8, 40))):
        w = (torch.randn(o, i) * 0.02).to(torch.bfloat16)
        q, s = qb.block_quant(w)
        assert q.dtype == torch.float8_e4m3fn and tuple(q.shape) == (o, i)
        assert s.dtype == torch.bfloat16 and tuple(s.shape) == grid, (s.shape, grid)


def test_roundtrip_error_is_fp8_noise():
    torch.manual_seed(1)
    w = (torch.randn(2048, 1024) * 0.02).to(torch.bfloat16)
    q, s = qb.block_quant(w)
    d = qb.block_dequant(q, s)
    rel = ((d - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    assert 0.005 < rel < 0.08, rel          # e4m3 with per-block absmax scaling: a few percent RMS
    assert (d.abs() <= w.float().abs().max() * 1.01).all()


def test_ragged_last_block():
    w = (torch.randn(300, 200) * 0.05).to(torch.bfloat16)
    q, s = qb.block_quant(w)
    assert tuple(q.shape) == (300, 200) and tuple(s.shape) == (3, 2)
    d = qb.block_dequant(q, s)
    assert tuple(d.shape) == (300, 200)
    rel = ((d - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    assert rel < 0.08


def test_outlier_block_does_not_poison_neighbours():
    w = (torch.randn(256, 256) * 0.01)
    w[0, 0] = 30.0                          # one huge value in block (0,0)
    q, s = qb.block_quant(w.to(torch.bfloat16))
    d = qb.block_dequant(q, s)
    other = w[128:, 128:]
    rel = ((d[128:, 128:] - other).pow(2).mean().sqrt() / other.pow(2).mean().sqrt()).item()
    assert rel < 0.08, rel


def test_key_rule_matches_vendor_choice():
    yes = ["model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
           "model.language_model.layers.0.linear_attn.in_proj_z.weight",
           "model.language_model.layers.0.linear_attn.out_proj.weight",
           "model.language_model.layers.3.self_attn.q_proj.weight",
           "model.language_model.layers.3.mlp.down_proj.weight",
           "mtp.layers.0.self_attn.o_proj.weight"]
    no = ["model.language_model.layers.0.linear_attn.in_proj_a.weight",
          "model.language_model.layers.0.linear_attn.in_proj_b.weight",
          "model.language_model.layers.0.linear_attn.conv1d.weight",
          "model.language_model.layers.0.mlp.gate.weight",
          "model.language_model.layers.0.mlp.shared_expert_gate.weight",
          "model.language_model.layers.0.input_layernorm.weight",
          "mtp.fc.weight", "lm_head.weight", "model.language_model.embed_tokens.weight",
          "model.visual.blocks.0.attn.qkv.weight", "model.visual.merger.linear_fc1.weight"]
    for k in yes:
        assert qb.should_quant(k, _Slice((256, 256))), k
    for k in no:
        assert not qb.should_quant(k, _Slice((256, 256))), k
    assert not qb.should_quant(yes[0], _Slice((256, 1, 4)))   # conv-shaped, never


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("ok", n)
