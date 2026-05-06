#!/usr/bin/env python3
"""Sanity-check AWQ safetensors for degenerate scales.

The v3 Gemma-4-26B drop_images=False calibration produced zero scales for
`embed_vision.embedding_projection`, which validate_capabilities couldn't
catch (model loaded, served, just produced NaN logits).  This script
inspects every `*.scales` tensor in an AWQ repo and flags:

  - all-zero       → llmcompressor silently skipped quantization
                     (degenerate Hessian) but the layer still got saved as
                     `qweight + scales=0 + qzeros`.  At inference the layer
                     dequantizes to zero, propagates as zeros into the
                     forward pass, and produces NaN logits downstream.
  - any-NaN        → numerical blowup somewhere during GPTQ.
  - any-Inf        → same.
  - all-tiny       → suspicious if the *.weight statistics in the BF16
                     base have meaningful magnitude; flag for review.

Usage:
    python scripts/eval/check_awq_scales.py <model-dir-or-shard>
    python scripts/eval/check_awq_scales.py --hf mattbucci/Qwen3-VL-32B-AWQ

Exit code: 0 if clean, 1 if any scale tensor failed a check.

The HF mode Range-fetches the safetensors header to enumerate tensor
names + shapes + dtypes without downloading the full weights, then for
any flagged scale tensor does a targeted Range-fetch of just that tensor
to confirm the values.  RAM-safe; doesn't load the full model.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np


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


def check_local(path: Path) -> tuple[int, int, list[tuple[str, str, list[str]]]]:
    """Return (scale_count, fail_count, failures[(file, name, issues)])."""
    from safetensors import safe_open

    if path.is_file() and path.suffix == ".safetensors":
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        print(f"[error] no .safetensors files found at {path}", file=sys.stderr)
        return 0, 0, []

    scale_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    for f in files:
        with safe_open(str(f), framework="np") as h:
            for k in h.keys():
                if not (k.endswith(".scales") or k.endswith(".weight_scale")):
                    continue
                t = h.get_tensor(k)
                scale_count += 1
                issues = _check_scale_tensor(k, t)
                if issues:
                    failures.append((f.name, k, issues))

    return scale_count, len(failures), failures


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


def check_hf(repo: str) -> tuple[int, int, list[tuple[str, str, list[str]]]]:
    """Range-fetch each safetensors header + scale tensor data."""
    files = [f for f in _list_repo_files(repo) if f.endswith(".safetensors")]
    if not files:
        print(f"[error] no safetensors in HF repo {repo}", file=sys.stderr)
        return 0, 0, []

    scale_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    dtype_map = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # treat as raw uint16, _check_scale_tensor handles it
        "F64": (np.float64, 8),
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
            if not (name.endswith(".scales") or name.endswith(".weight_scale")):
                continue
            scale_count += 1
            dtype = info.get("dtype", "")
            if dtype not in dtype_map:
                failures.append((fname, name, [f"unknown scale dtype {dtype}"]))
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
            issues = _check_scale_tensor(name, arr)
            if issues:
                failures.append((fname, name, issues))

    return scale_count, len(failures), failures


def main():
    ap = argparse.ArgumentParser(description="AWQ scales sanity check")
    ap.add_argument("path", nargs="?", help="local model dir OR single .safetensors file")
    ap.add_argument("--hf", help="HF repo (e.g. mattbucci/Qwen3-VL-32B-AWQ)")
    args = ap.parse_args()

    if args.hf:
        print(f"=== HF repo {args.hf} ===")
        scale_count, fail_count, failures = check_hf(args.hf)
    elif args.path:
        path = Path(args.path).expanduser().resolve()
        if not path.exists():
            print(f"[error] {path} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"=== local {path} ===")
        scale_count, fail_count, failures = check_local(path)
    else:
        ap.print_usage()
        sys.exit(2)

    if scale_count == 0:
        print("[error] no *.scales tensors found", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned {scale_count} *.scales tensors, {fail_count} flagged.")
    if failures:
        for fname, name, issues in failures:
            print(f"  [FAIL] {fname}::{name}")
            for i in issues:
                print(f"         - {i}")
        sys.exit(1)
    else:
        print("All scales clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
