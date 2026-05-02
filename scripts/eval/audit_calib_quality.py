#!/usr/bin/env python3
"""Calibration-quality audit across shipped mattbucci/*-AWQ repos.

Pure metadata audit (HF config.json + safetensors index, ~10-100 KB per repo).
NO model load, NO GPU. Safe to run while a calibration is in flight.

Checks:
  1. Architecture class declared (text-only vs multimodal)
  2. Vision tower presence + dtype (BF16 preserved vs INT4-quantized)
  3. Audio tower presence + dtype
  4. MoE router (mlp.gate) dtype — must be BF16, never INT4
  5. DeltaNet linear_attn.in_proj_a/b — must be BF16

For single-file safetensors repos (no index), Range-fetches the header
(<10 MB) to get the same tensor list without downloading weights.

History:
  2026-05-02 first run — all 11 shipped repos audited clean.
  Vision/audio towers preserved as BF16 across the board.
  MoE routers preserved as BF16 (Coder-30B explicit ignore list only had
  lm_head but llmcompressor targets=Linear correctly skipped the router).
  Gemma 4 audio not yet present in google/* BF16 base either.

Usage:
  python scripts/eval/audit_calib_quality.py
  python scripts/eval/audit_calib_quality.py --repo mattbucci/Devstral-24B-AWQ
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request

DEFAULT_REPOS = [
    "mattbucci/Qwen3.5-27B-AWQ",
    "mattbucci/Qwen3.6-27B-AWQ",
    "mattbucci/Qwen3.6-35B-A3B-AWQ",
    "mattbucci/Qwen3.6-REAM-A3B-AWQ",
    "mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ",
    "mattbucci/Devstral-24B-AWQ",
    "mattbucci/gemma-4-26B-AWQ",
    "mattbucci/Qwen3-Coder-30B-A3B-AWQ",
    "mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ",
    "mattbucci/Qwen3-Coder-Next-REAM-AWQ",
    "mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ",
]


def _hf_token() -> str | None:
    p = os.path.expanduser("~/.secrets/hf_token")
    if os.path.exists(p):
        return open(p).read().strip()
    return os.environ.get("HF_TOKEN")


def _get(url: str, *, range_header: str | None = None) -> bytes:
    headers = {}
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _tensor_keys(repo: str) -> list[str]:
    """Return list of tensor keys for a repo. Falls back to single-file header
    range fetch when model.safetensors.index.json is absent (small models)."""
    try:
        idx = json.loads(_get(f"https://huggingface.co/{repo}/raw/main/model.safetensors.index.json"))
        return list(idx.get("weight_map", {}).keys())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    # Single-file fallback: range-fetch safetensors header
    n = struct.unpack("<Q", _get(
        f"https://huggingface.co/{repo}/resolve/main/model.safetensors",
        range_header="bytes=0-7",
    ))[0]
    if n > 50_000_000:
        raise RuntimeError(f"{repo}: header is {n/1e6:.1f}MB — refusing to fetch (likely not safetensors)")
    header = json.loads(_get(
        f"https://huggingface.co/{repo}/resolve/main/model.safetensors",
        range_header=f"bytes=8-{8+n-1}",
    ))
    header.pop("__metadata__", None)
    return list(header.keys())


def audit(repo: str) -> dict:
    cfg = json.loads(_get(f"https://huggingface.co/{repo}/raw/main/config.json"))
    arch = ", ".join(cfg.get("architectures", []) or ["?"])
    qc = cfg.get("quantization_config", {})
    ignore = qc.get("ignore", []) or qc.get("modules_to_not_convert", [])
    keys = _tensor_keys(repo)

    def quantized(group):
        return [k for k in group if k.endswith((".qweight", ".scales", ".qzeros"))]

    def bf16(group):
        return [k for k in group if k.endswith((".weight", ".bias"))]

    vision = [k for k in keys if any(t in k for t in ("vision_tower", "visual.", "multi_modal_projector", "embed_vision"))]
    audio = [k for k in keys if any(t in k for t in ("audio_tower", "embed_audio"))]
    router = [k for k in keys if re.search(r"mlp\.gate(\.|$)", k) and "shared_expert" not in k]
    deltanet = [k for k in keys if "linear_attn.in_proj" in k]

    findings = []
    multimodal_arch = "ConditionalGeneration" in arch or "ForConditional" in arch

    if multimodal_arch and not vision:
        # text-only recipe stripped vision, OR base never had vision
        findings.append("multimodal arch but NO vision_tower keys — verify base/recipe")
    if vision and quantized(vision):
        findings.append(f"vision_tower has {len(quantized(vision))} INT4 keys — silent degradation")
    if audio and quantized(audio):
        findings.append(f"audio_tower has {len(quantized(audio))} INT4 keys — silent degradation")
    if router and quantized(router):
        findings.append(f"MoE router has {len(quantized(router))} INT4 keys — top-k accuracy degraded")
    if deltanet and quantized(deltanet):
        findings.append(f"DeltaNet in_proj has {len(quantized(deltanet))} INT4 keys — recurrent state will diverge")

    return {
        "repo": repo,
        "arch": arch,
        "ignore_count": len(ignore),
        "total_keys": len(keys),
        "vision": (len(vision), len(bf16(vision)), len(quantized(vision))),
        "audio": (len(audio), len(bf16(audio)), len(quantized(audio))),
        "router": (len(router), len(bf16(router)), len(quantized(router))),
        "deltanet": (len(deltanet), len(bf16(deltanet)), len(quantized(deltanet))),
        "findings": findings,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", action="append", help="repo id (default: shipped mattbucci set)")
    args = p.parse_args()
    repos = args.repo or DEFAULT_REPOS

    print(f"{'repo':<48} {'arch':<42} {'vis(b/q)':<11} {'aud(b/q)':<11} {'router(b/q)':<13} {'deltanet(b/q)'}")
    print("-" * 160)
    issues_total = []
    for repo in repos:
        try:
            r = audit(repo)
        except Exception as e:
            print(f"{repo:<48} ERROR: {e}")
            continue
        v, a, ro, dn = r["vision"], r["audio"], r["router"], r["deltanet"]
        print(f"{r['repo']:<48} {r['arch']:<42} "
              f"{v[0]}({v[1]}/{v[2]:<3}) {a[0]}({a[1]}/{a[2]:<3}) "
              f"{ro[0]}({ro[1]}/{ro[2]:<5}) {dn[0]}({dn[1]}/{dn[2]})")
        for f in r["findings"]:
            print(f"  ⚠ {f}")
            issues_total.append((repo, f))

    print(f"\n{'='*60}\nSUMMARY: {len(issues_total)} issues across {len(repos)} repos")
    for repo, f in issues_total:
        print(f"  {repo}: {f}")
    sys.exit(1 if issues_total else 0)


if __name__ == "__main__":
    main()
