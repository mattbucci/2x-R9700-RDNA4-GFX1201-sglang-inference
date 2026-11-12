"""R97-T: apply the fp8-pool weight-scale switch to an SGLang python tree (argv[1] = <tree>/python). Anchored edits,
each anchor must occur exactly once; refuses the serving tree and a tree that already has the switch."""
import os
import sys

TREE = sys.argv[1]
assert os.path.realpath(TREE) != "/data/sgl-v0522/python", "not the serving tree"
ATT = os.path.join(TREE, "sglang/kernels/ops/attention")
SCALED = "tl.dot((p * 256.0).to(v.dtype), v) * (1.0 / 256.0)"
ENV = 'os.environ.get("SGLANG_RDNA4_FP8_PV_SCALE", "0") == "1"\n'
STAGE1 = (
    "            acc *= re_scale[:, None]\n            acc += tl.dot(p.to(v.dtype), v)\n",
    "            acc *= re_scale[:, None]\n"
    "            if FP8_PV_SCALE:\n"
    "                acc += " + SCALED + "\n"
    "            else:\n"
    "                acc += tl.dot(p.to(v.dtype), v)\n",
)


def edit(path, pairs):
    s = open(path).read()
    assert "FP8_PV_SCALE" not in s, path + ": already patched"
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    with open(path + ".tmp", "w") as f:
        f.write(s)
    os.replace(path + ".tmp", path)
    print("edited", os.path.relpath(path, TREE), len(pairs))


edit(os.path.join(ATT, "extend_attention.py"), [
    ('_EXTEND_FP8_QK = os.environ.get("SGLANG_EXTEND_FP8_QK", "0") == "1"\n',
     '_EXTEND_FP8_QK = os.environ.get("SGLANG_EXTEND_FP8_QK", "0") == "1"\n'
     "# RDNA4 opt-in (R97-T): the value product on an fp8 pool casts the softmax weights to e4m3. Unscaled, a weight\n"
     "# under 2^-6 is subnormal and one under 2^-10 becomes zero (0.35-2.6 % of a cached turn's output, up to 10 % of\n"
     "# a prefill tail at 197K). Times 2^8 (a weight is at most 1, e4m3 reaches 448) the cast keeps three bits down to\n"
     "# 2^-14 and rounds weights above 2^-6 as before; the product is scaled back by 2^-8. Read by the extend prefix\n"
     "# loop here, verify_splitkv's stage 1 and verify_gqa_rdna4's grouped stage; inert on a BF16 pool.\n"
     "_FP8_PV_SCALE = " + ENV),
    ("    FP8_QK: tl.constexpr = False,\n    PAGE_SIZE: tl.constexpr = 1,\n    IDENTITY_KV_INDICES: tl.constexpr = False,\n",
     "    FP8_QK: tl.constexpr = False,\n    FP8_PV_SCALE: tl.constexpr = False,\n    PAGE_SIZE: tl.constexpr = 1,\n"
     "    IDENTITY_KV_INDICES: tl.constexpr = False,\n"),
    ("                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)\n"
     "                else:\n"
     "                    dot = tl.dot(p.to(v.dtype), v)\n"
     "                acc = acc * re_scale[:, None] + dot * v_scale\n",
     "                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)\n"
     "                elif FP8_PV_SCALE:\n"
     "                    dot = " + SCALED + "\n"
     "                else:\n"
     "                    dot = tl.dot(p.to(v.dtype), v)\n"
     "                acc = acc * re_scale[:, None] + dot * v_scale\n"),
    ("        FP8_QK=_EXTEND_FP8_QK,\n        STORE_TRANSPOSE=_is_hip,\n        USE_COMPACT_TILE_GRID=use_compact_tile_grid,\n",
     "        FP8_QK=_EXTEND_FP8_QK,\n"
     "        FP8_PV_SCALE=_FP8_PV_SCALE and v_buffer.dtype == torch.float8_e4m3fn,\n"
     "        STORE_TRANSPOSE=_is_hip,\n        USE_COMPACT_TILE_GRID=use_compact_tile_grid,\n"),
])

edit(os.path.join(ATT, "verify_splitkv.py"), [
    ('_UPCAST_K = os.environ.get("SGLANG_SPLITKV_UPCAST_K", "0") == "1"\n',
     '_UPCAST_K = os.environ.get("SGLANG_SPLITKV_UPCAST_K", "0") == "1"\n'
     "# RDNA4 opt-in (R97-T): scale the prefix weights by 2^8 ahead of their e4m3 cast (see extend_attention).\n"
     "_FP8_PV_SCALE = " + ENV),
    ("def _min_row_tile(device) -> int:\n",
     "def fp8_pv_scale(v_buffer) -> bool:\n"
     "    return _FP8_PV_SCALE and v_buffer.dtype == torch.float8_e4m3fn\n"
     "\n\n"
     "def _min_row_tile(device) -> int:\n"),
    ("    SLIDING_WINDOW_SIZE: tl.constexpr = -1,  # RDNA4 RAD-28: > 0 when kv_indptr / kv_indices span a causal window\n):\n",
     "    SLIDING_WINDOW_SIZE: tl.constexpr = -1,  # RDNA4 RAD-28: > 0 when kv_indptr / kv_indices span a causal window\n"
     "    FP8_PV_SCALE: tl.constexpr = False,  # RDNA4 R97-T: fp8 pool, weights x 2^8 ahead of the e4m3 cast\n):\n"),
    STAGE1,
    ("            UPCAST_K=self.upcast_k,\n            SLIDING_WINDOW_SIZE=self.sliding_window_size,\n",
     "            UPCAST_K=self.upcast_k,\n            SLIDING_WINDOW_SIZE=self.sliding_window_size,\n"
     "            FP8_PV_SCALE=fp8_pv_scale(v_buffer),\n"),
])

edit(os.path.join(ATT, "verify_gqa_rdna4.py"), [
    ("    choose_n_splits,\n    verify_splitkv_fwd,\n", "    choose_n_splits,\n    fp8_pv_scale,\n    verify_splitkv_fwd,\n"),
    ("    UPCAST_K: tl.constexpr = False,\n):\n",
     "    UPCAST_K: tl.constexpr = False,\n    FP8_PV_SCALE: tl.constexpr = False,  # R97-T, as in _verify_prefix_stage1\n):\n"),
    STAGE1,
    ("            UPCAST_K=self.upcast_k,\n            num_warps=self.num_warps,\n",
     "            UPCAST_K=self.upcast_k,\n"
     "            FP8_PV_SCALE=fp8_pv_scale(v_buffer),\n"
     "            num_warps=self.num_warps,\n"),
])

edit(os.path.join(TREE, "sglang/srt/layers/attention/triton_backend.py"), [
    ("            self.verify_gqa_fwd = torch.compiler.disable(verify_gqa_fwd)\n",
     "            self.verify_gqa_fwd = torch.compiler.disable(verify_gqa_fwd)\n"
     "        from sglang.kernels.ops.attention.extend_attention import _FP8_PV_SCALE\n"
     "\n"
     "        if _FP8_PV_SCALE and model_runner.kv_cache_dtype == torch.float8_e4m3fn:\n"
     "            logger.info(\n"
     '                "fp8 pool weight scale active: prefix softmax weights x 2^8 ahead of the e4m3 cast "\n'
     '                "(extend prefix loop, split-KV stage 1, grouped verify stage)"\n'
     "            )\n"),
])
