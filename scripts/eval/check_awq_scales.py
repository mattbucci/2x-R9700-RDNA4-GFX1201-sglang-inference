#!/usr/bin/env python3
"""Sanity-check AWQ safetensors for degenerate weights and scales.

Checks BOTH `*.scales` and `*.qweight` tensors, because the two failure
modes are independent:

  scales=0  (v2 disaster, Gemma-4-26B drop_images=False, 21B-REAP-v2):
    llmcompressor silently skipped quantization (degenerate Hessian) but
    the layer still got saved as `qweight + scales=0 + qzeros`. At
    inference the layer dequantizes to zero, propagates as zeros into
    the forward pass, and produces NaN logits downstream.

  qweight=0 (v3 disaster, 21B-REAP-v3 2026-05-08):
    GPTQ Hessian was *also* degenerate but produced non-zero scales with
    quantized-to-zero qweight (rare-expert under-calibration: 512 samples
    × top-k=8 ≈ 4K activations split across 128 experts × N layers means
    rare experts got ~0 activations, weight quantized to 0). Validator
    sees 1/4 PASS with empty `(reasoning)` placeholder content. Server
    log diagnostic shows `expert0_nonzero=False expert0_first4=[0,0,0,0]`.
    Audit script (.scales-only) MISSES this because the scales are fine.

Per-tensor flags:
  - all-zero       → ALL elements are zero (catastrophic)
  - majority-zero  → >50% zero (suspicious, rare-expert under-cal pattern)
  - any-NaN / Inf  → numerical blowup
  - all-tiny       → abs_max < 1e-8 (scales) or extreme outliers

Usage:
    python scripts/eval/check_awq_scales.py <model-dir-or-shard>
    python scripts/eval/check_awq_scales.py --hf mattbucci/Qwen3-VL-32B-AWQ
    python scripts/eval/check_awq_scales.py <path> --skip-qweight  # legacy
    python scripts/eval/check_awq_scales.py <awq-dir> --base <bf16-base-dir>

Exit code: 0 if clean, 1 if any scale or qweight tensor failed a check.

The optional `--base <bf16-dir>` enables the dead-channel comparator: a zero
scale over a *dead* base-weight block (the MoE structural-sparsity case where
the BF16 base already has the channel at ~7.8e-38) is benign and downgraded;
a zero scale over a *live* base block stays a DEFECT. See `_base_block_maxabs`.

The HF mode Range-fetches the safetensors header to enumerate tensor
names + shapes + dtypes without downloading the full weights, then for
any flagged tensor does a targeted Range-fetch of just that tensor
to confirm the values.  RAM-safe; doesn't load the full model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch  # for bf16-tolerant scale loading via safetensors framework="pt"


def _check_scale_tensor(name: str, arr: np.ndarray) -> list[str]:
    """Return list of human-readable failure messages for a scale tensor."""
    issues: list[str] = []

    # Coerce bf16 → float32 for nan/inf checks (numpy doesn't have native bf16)
    if arr.dtype == np.uint16:
        # bfloat16 stored as raw uint16 — reinterpret to fp32
        as_fp32 = (arr.astype(np.uint32) << 16).view(np.float32)
    elif arr.dtype in (np.float16, np.float32, np.float64):
        as_fp32 = arr.astype(np.float32, copy=False)
    else:
        return [f"[unexpected scales dtype {arr.dtype}]"]

    n = as_fp32.size
    if n == 0:
        issues.append(f"empty tensor (shape {arr.shape})")
        return issues

    n_nan = int(np.isnan(as_fp32).sum())
    n_inf = int(np.isinf(as_fp32).sum())
    n_zero = int((as_fp32 == 0).sum())
    abs_arr = np.abs(as_fp32)
    abs_min = float(abs_arr.min())
    abs_max = float(abs_arr.max())
    abs_mean = float(abs_arr.mean())

    if n_nan > 0:
        issues.append(f"{n_nan}/{n} NaN")
    if n_inf > 0:
        issues.append(f"{n_inf}/{n} Inf")
    if n_zero == n:
        issues.append(f"ALL-ZERO scales (n={n}, shape={tuple(arr.shape)})")
    elif n_zero / n > 0.5:
        issues.append(f"{100*n_zero/n:.1f}% zero scales (suspicious)")
    if abs_max < 1e-8 and n_zero != n:
        issues.append(f"all scales tiny: abs_max={abs_max:.2e}")
    elif abs_max > 1e6:
        issues.append(f"scale outlier: abs_max={abs_max:.2e}")

    return issues


def _check_qweight_tensor(name: str, arr: np.ndarray) -> list[str]:
    """Return list of human-readable failure messages for a qweight tensor.

    qweight is int32 packed (8 4-bit values per int32). int32 == 0 means
    all 8 packed 4-bit values are 0, i.e. the underlying weight is zero
    in that block. Per-element `==0` is sufficient because hitting all
    zeros across an entire packed-int32 block is the disaster signal.
    """
    issues: list[str] = []
    n = arr.size
    if n == 0:
        issues.append(f"empty tensor (shape {arr.shape})")
        return issues

    # int32 packed ints; np handles == 0 directly. Coerce to int64 to be
    # safe across platforms where dtype reads might come back as int32 vs
    # uint32.
    n_zero = int((arr == 0).sum())

    if n_zero == n:
        issues.append(f"ALL-ZERO qweight (n={n}, shape={tuple(arr.shape)})")
    elif n_zero / n > 0.5:
        issues.append(f"{100*n_zero/n:.1f}% zero qweight (rare-expert under-cal pattern)")
    return issues


# ---------------------------------------------------------------------------
# Base-model "dead channel" comparator (opt-in via --base).
#
# MoE bases such as Qwen3.6-35B-A3B ship with 50-72% of some layer-0 expert
# gate/up output channels at ~7.8e-38 (bf16 denormal) in the BF16 base. AWQ's
# fp16 group scale = max_abs(block)/15 underflows fp16 to exactly 0 over those
# blocks — a *faithful* encoding of a dead channel, not a defect. Without the
# base, the audit can't tell a dead-channel zero scale apart from the v2
# disaster (quantizer bailed on a LIVE block -> dequant-to-zero -> NaN). The
# comparator resolves it: a zero scale is benign iff the matching base weight
# block is dead (max_abs < DEAD_THRESH); over a LIVE block it stays a DEFECT.
#
# Conservative by construction. It only DOWNGRADES a flag when it can both
# (a) confidently map the AWQ tensor to a base weight block (validated by exact
# shape match) and (b) confirm deadness. Any name it can't map or any shape
# that doesn't validate is left flagged.
#
# DEAD_THRESH (1e-15) sits in the wide gap between dead and live base blocks.
# Dead channels come in (at least) two magnitudes — Qwen3.6 ships bf16
# denormals (~1e-38), while the Coder-30B REAP/REAM bases carry near-zero
# channels at ~1e-26 (one expert, ~52% of its gate/up). Both are >=24 orders
# of magnitude below live trained weights (block max_abs ~1e-2..1e-1, min
# ~1e-3). A v2-disaster zeroes a LIVE block (~1e-2), which stays far above
# 1e-15 -> still flagged DEFECT. So 1e-15 introduces no false negatives (no
# real defect has a sub-1e-15 base block) while correctly treating both
# dead-channel magnitudes as benign.
# ---------------------------------------------------------------------------

DEAD_THRESH = 1e-15


def _load_base_ctx(base_path: Path):
    """Open a base BF16 checkpoint for lazy per-tensor slicing.
    Returns (base_dir, weight_map, handle_cache) or None."""
    from safetensors import safe_open

    idx = base_path / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.load(open(idx))["weight_map"]
    else:
        shards = sorted(base_path.glob("*.safetensors"))
        if not shards:
            print(f"[warn] --base {base_path}: no safetensors found, comparator disabled", file=sys.stderr)
            return None
        weight_map = {}
        for s in shards:
            with safe_open(str(s), framework="pt") as h:
                for k in h.keys():
                    weight_map[k] = s.name
    return (base_path, weight_map, {})


def _base_handle(base_ctx, shard):
    base_dir, _, cache = base_ctx
    if shard not in cache:
        from safetensors import safe_open
        cache[shard] = safe_open(str(base_dir / shard), framework="pt")
    return cache[shard]


def _base_targets(scale_name: str):
    """Map an AWQ scale tensor name -> a list of candidate
    (base_tensor_name, kind, expert_idx) mappings, tried in order.

    kind: 'gate'|'up' (fused gate_up_proj, gate=rows[0:O], up=rows[O:2O]),
          'down' (per-expert down_proj fused as [E,...]), or 'weight' (plain
          2-D [out, in] tensor — a dense Linear OR an unfused per-expert weight).
    Routed experts get TWO candidates so both base layouts resolve:
      * fused base (Qwen3.5/3.6): one `experts.gate_up_proj`/`down_proj` 3-D param;
      * unfused base (Qwen3Moe/Coder): per-expert `experts.{e}.{proj}.weight`.
    Returns [] when the name isn't recognised (caller stays conservative).
    """
    if scale_name.endswith(".scales"):
        base = scale_name[: -len(".scales")]
    elif scale_name.endswith(".weight_scale"):
        base = scale_name[: -len(".weight_scale")]
    else:
        return []
    m = re.match(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)$", base)
    if m:
        prefix, e, proj = m.group(1), int(m.group(2)), m.group(3)
        cands = []
        if proj == "down_proj":
            cands.append((f"{prefix}.down_proj", "down", e))
        else:
            cands.append((f"{prefix}.gate_up_proj", "gate" if proj == "gate_proj" else "up", e))
        cands.append((f"{base}.weight", "weight", None))  # unfused per-expert fallback
        return cands
    # plain dense Linear: base stores <module>.weight as [out, in]
    return [(f"{base}.weight", "weight", None)]


def _base_block_maxabs(scale_name, G, O, group_size, base_ctx):
    """Return a [G, O] float32 grid of base-weight per-(group, out) block
    max-abs, aligned to the AWQ scale tensor, or None if no candidate mapping
    is present + shape-validated (caller stays conservative)."""
    _, weight_map, _ = base_ctx
    for bname, kind, e in _base_targets(scale_name):
        if bname not in weight_map:
            continue
        try:
            h = _base_handle(base_ctx, weight_map[bname])
            sl = h.get_slice(bname)
            sub = sl[e] if e is not None else sl[:]   # partial read: just this expert
            sub = sub.float().cpu().numpy()
        except Exception:
            continue
        if sub.ndim != 2:
            continue
        if kind == "gate":
            W = sub[0:O, :]
        elif kind == "up":
            W = sub[O:2 * O, :]
        else:  # 'down' or 'weight'
            W = sub
        if W.shape[0] != O or W.shape[1] != G * group_size:
            continue  # shape mismatch -> try next candidate
        return np.abs(W).reshape(O, G, group_size).max(axis=2).T.astype(np.float32)  # [G, O]
    return None


def _reclassify_scale_with_base(scale_name, t_fp32, issues, group_size, base_ctx):
    """Refine zero-scale flags against the base model. A zero scale over a DEAD
    base block is benign (dropped); over a LIVE base block it stays a DEFECT.
    NaN/Inf/outlier issues are passed through untouched."""
    zero_related = [i for i in issues if ("ZERO" in i) or ("zero scales" in i)]
    other = [i for i in issues if i not in zero_related]
    if not zero_related:
        return issues
    if t_fp32.ndim != 2:
        return other + [i + " [base: non-2D scale, unverified]" for i in zero_related]
    G, O = t_fp32.shape
    grid = _base_block_maxabs(scale_name, G, O, group_size, base_ctx)
    if grid is None:
        return other + [i + " [base: unmapped, unverified]" for i in zero_related]
    zero = (t_fp32 == 0)
    dead = (grid < DEAD_THRESH)
    n_zero = int(np.count_nonzero(zero))
    live_zero = int(np.count_nonzero(zero & ~dead))
    if live_zero > 0:
        return other + [f"{live_zero}/{n_zero} zero scales over LIVE base blocks (DEFECT)"]
    return other  # every zero scale sits over a dead base channel -> benign


def _quant_group_size(cfg_dir: Path) -> int:
    cfgp = cfg_dir / "config.json"
    if cfgp.exists():
        try:
            qc = json.load(open(cfgp)).get("quantization_config", {})
            gs = int(qc.get("group_size", 128))
            return gs if gs > 0 else 128
        except Exception:
            pass
    return 128


def check_local(path: Path, skip_qweight: bool = False, base_path: Path | None = None) -> tuple[int, int, int, list[tuple[str, str, list[str]]]]:
    """Return (scale_count, qweight_count, fail_count, failures[(file, name, issues)])."""
    from safetensors import safe_open

    if path.is_file() and path.suffix == ".safetensors":
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        print(f"[error] no .safetensors files found at {path}", file=sys.stderr)
        return 0, 0, 0, []

    scale_count = 0
    qweight_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    cfg_dir = path.parent if path.is_file() else path
    group_size = _quant_group_size(cfg_dir)
    base_ctx = _load_base_ctx(base_path) if base_path else None

    # Use framework="pt" for bf16 support — safetensors np backend raises
    # TypeError("data type 'bfloat16' not understood") on bf16 tensors which
    # are common in CT-format AWQ scales (e.g. gemma-4-26B-A4B-it-AWQ-4bit
    # ships scales in bf16). Torch handles bf16 natively; we cast to float
    # for the all-zero / NaN / Inf checks below.
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                if k.endswith(".scales") or k.endswith(".weight_scale"):
                    t_pt = h.get_tensor(k)
                    t = t_pt.float().cpu().numpy()
                    scale_count += 1
                    issues = _check_scale_tensor(k, t)
                    if issues and base_ctx is not None:
                        issues = _reclassify_scale_with_base(k, t, issues, group_size, base_ctx)
                    if issues:
                        failures.append((f.name, k, issues))
                elif (not skip_qweight) and k.endswith(".qweight"):
                    t_pt = h.get_tensor(k)
                    # qweight is int32; numpy handles natively
                    t = t_pt.cpu().numpy()
                    qweight_count += 1
                    issues = _check_qweight_tensor(k, t)
                    if issues:
                        failures.append((f.name, k, issues))

    return scale_count, qweight_count, len(failures), failures


def _hf_token() -> str | None:
    p = Path("~/.secrets/hf_token").expanduser()
    if p.exists():
        return p.read_text().strip()
    return os.environ.get("HF_TOKEN")


def _hf_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {}
    tok = _hf_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def _hf_resolve(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _hf_range_get(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(
        url, headers=_hf_headers({"Range": f"bytes={start}-{start+length-1}"})
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _list_repo_files(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [d["path"] for d in data if d.get("type") == "file"]


def check_hf(repo: str, skip_qweight: bool = False) -> tuple[int, int, int, list[tuple[str, str, list[str]]]]:
    """Range-fetch each safetensors header + scale/qweight tensor data."""
    files = [f for f in _list_repo_files(repo) if f.endswith(".safetensors")]
    if not files:
        print(f"[error] no safetensors in HF repo {repo}", file=sys.stderr)
        return 0, 0, 0, []

    scale_count = 0
    qweight_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    dtype_map = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # treat as raw uint16, _check_scale_tensor handles it
        "F64": (np.float64, 8),
        "I32": (np.int32, 4),
        "U32": (np.uint32, 4),
    }

    for fname in files:
        url = _hf_resolve(repo, fname)
        # safetensors header: first 8 bytes = u64 little-endian length, then JSON
        head = _hf_range_get(url, 0, 8)
        hdr_len = struct.unpack("<Q", head)[0]
        hdr_bytes = _hf_range_get(url, 8, hdr_len)
        hdr = json.loads(hdr_bytes)

        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            is_scale = name.endswith(".scales") or name.endswith(".weight_scale")
            is_qweight = name.endswith(".qweight")
            if not (is_scale or (is_qweight and not skip_qweight)):
                continue
            dtype = info.get("dtype", "")
            if dtype not in dtype_map:
                failures.append((fname, name, [f"unknown dtype {dtype}"]))
                continue
            np_dtype, elem_size = dtype_map[dtype]
            shape = info.get("shape", [])
            n_elem = 1
            for d in shape:
                n_elem *= d
            data_offsets = info.get("data_offsets", [0, 0])
            byte_start = 8 + hdr_len + data_offsets[0]
            byte_len = data_offsets[1] - data_offsets[0]
            if byte_len != n_elem * elem_size:
                failures.append((fname, name, [f"byte/elem mismatch: {byte_len} vs {n_elem*elem_size}"]))
                continue
            raw = _hf_range_get(url, byte_start, byte_len)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            if is_scale:
                scale_count += 1
                issues = _check_scale_tensor(name, arr)
            else:
                qweight_count += 1
                issues = _check_qweight_tensor(name, arr)
            if issues:
                failures.append((fname, name, issues))

    return scale_count, qweight_count, len(failures), failures


def main():
    ap = argparse.ArgumentParser(description="AWQ scales + qweight sanity check")
    ap.add_argument("path", nargs="?", help="local model dir OR single .safetensors file")
    ap.add_argument("--hf", help="HF repo (e.g. mattbucci/Qwen3-VL-32B-AWQ)")
    ap.add_argument("--skip-qweight", action="store_true",
                    help="legacy mode: only audit .scales (skip new qweight check)")
    ap.add_argument("--base", help="upstream BF16 base dir — enables the dead-channel "
                    "comparator (downgrades zero scales over dead base blocks; local mode only)")
    args = ap.parse_args()

    if args.hf:
        print(f"=== HF repo {args.hf} ===")
        if args.base:
            print("[warn] --base comparator is local-only; ignored in --hf mode", file=sys.stderr)
        scale_count, qweight_count, fail_count, failures = check_hf(args.hf, skip_qweight=args.skip_qweight)
    elif args.path:
        path = Path(args.path).expanduser().resolve()
        if not path.exists():
            print(f"[error] {path} does not exist", file=sys.stderr)
            sys.exit(1)
        base_path = None
        if args.base:
            base_path = Path(args.base).expanduser().resolve()
            if not base_path.exists():
                print(f"[error] --base {base_path} does not exist", file=sys.stderr)
                sys.exit(1)
            print(f"=== local {path}  (vs base {base_path}) ===")
        else:
            print(f"=== local {path} ===")
        scale_count, qweight_count, fail_count, failures = check_local(
            path, skip_qweight=args.skip_qweight, base_path=base_path)
    else:
        ap.print_usage()
        sys.exit(2)

    if scale_count == 0 and qweight_count == 0:
        # Not an AWQ build (BF16 base, FP8, full-precision checkpoint, etc.)
        print("[info] no *.scales / *.qweight tensors found — not an AWQ build (skipping audit)")
        sys.exit(0)

    summary_parts = [f"{scale_count} *.scales"]
    if qweight_count:
        summary_parts.append(f"{qweight_count} *.qweight")
    print(f"Scanned {' + '.join(summary_parts)} tensors, {fail_count} flagged.")
    if failures:
        for fname, name, issues in failures:
            print(f"  [FAIL] {fname}::{name}")
            for i in issues:
                print(f"         - {i}")
        sys.exit(1)
    else:
        print("All scales + qweight clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
