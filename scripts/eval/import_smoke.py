#!/usr/bin/env python
"""Eager-import boot-chain smoke — the CPU merge-remnant catch after a patch rebase.

Imports every module a patch touches (quant infra + token_dispatcher + activation +
triton attention + every patched model class). A dropped guard-hunk or a merge remnant
(stale import of a since-deleted upstream symbol) surfaces here as an ImportError /
NameError / AttributeError *before* any GPU boot. Prints a PASS/FAIL table; exits 1 if
any import fails.

Usage (inside the target env, CPU-only — no GPU needed):
    HIP_VISIBLE_DEVICES="" python scripts/eval/import_smoke.py
"""
import importlib
import sys
import traceback

# (label, module dotted-path[, attr to touch])
CHECKS = [
    # --- quant infra ---
    ("quant: QUANTIZATION_METHODS", "sglang.srt.layers.quantization", "QUANTIZATION_METHODS"),
    ("quant: fp8_utils (005 _is_rdna4_device)", "sglang.srt.layers.quantization.fp8_utils", "_is_rdna4_device"),
    ("quant: fp8 method", "sglang.srt.layers.quantization.fp8", None),
    ("quant: moe_wna16 (031/033)", "sglang.srt.layers.quantization.moe_wna16", None),
    ("quant: awq", "sglang.srt.layers.quantization.awq", None),
    ("quant: modelopt (044)", "sglang.srt.layers.quantization.modelopt_quant", None),
    ("quant: quark mxfp4 moe (060)", "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe", None),
    # --- MoE / dispatch ---
    ("moe: topk (004 _is_hip fallback)", "sglang.srt.layers.moe.topk", None),
    ("moe: token_dispatcher (059 fuseep-drop)", "sglang.srt.layers.moe.token_dispatcher", None),
    ("moe: hybrid_w4a16 (032)", "sglang.srt.layers.moe.hybrid_w4a16_moe", None),
    # --- activation / attention / server-args ---
    ("layers: activation (063 relu2)", "sglang.srt.layers.activation", None),
    ("attn: triton_backend (011/065)", "sglang.srt.layers.attention.triton_backend", None),
    ("server_args (073 mamba-radix)", "sglang.srt.server_args", None),
    ("arg_groups.overrides (073 relocated)", "sglang.srt.arg_groups.overrides", "_mamba_radix_cache_resolution"),
    ("configs.model_config (062 hybrid-swa)", "sglang.srt.configs.model_config", "is_hybrid_swa_model"),
    ("entrypoint: launch_server", "sglang.launch_server", None),
    # --- patched models ---
    ("model: qwen3_5 (DeltaNet)", "sglang.srt.models.qwen3_5", None),
    ("model: qwen3_next", "sglang.srt.models.qwen3_next", None),
    # v0.5.15 folds the MoE class into qwen3_5.py (no separate qwen3_5_moe module)
    ("model: Qwen3_5MoeForCausalLM (in qwen3_5)", "sglang.srt.models.qwen3_5", "Qwen3_5MoeForCausalLM"),
    ("model: qwen3_vl (055 eagle3)", "sglang.srt.models.qwen3_vl", None),
    ("model: gemma4_mm (023-026/061)", "sglang.srt.models.gemma4_mm", None),
    ("model: gemma4_causal", "sglang.srt.models.gemma4_causal", None),
    ("model: gemma4_unified (072)", "sglang.srt.models.gemma4_unified", None),
    ("model: ministral3 (064 upstreamed)", "sglang.srt.models.ministral3", None),
    ("model: llama (007)", "sglang.srt.models.llama", None),
    ("model: glm4_moe (066)", "sglang.srt.models.glm4_moe", None),
    ("model: cohere2_moe (062)", "sglang.srt.models.cohere2_moe", None),
    ("model: nemotron_h (063/043/046/047)", "sglang.srt.models.nemotron_h", None),
    ("model: mistral (Mistral3/Devstral wrapper)", "sglang.srt.models.mistral", None),
]


def main():
    ok = 0
    fail = 0
    fails = []
    for entry in CHECKS:
        label, mod = entry[0], entry[1]
        attr = entry[2] if len(entry) > 2 else None
        try:
            m = importlib.import_module(mod)
            if attr is not None:
                getattr(m, attr)
            print(f"  PASS  {label}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {label}\n        {type(e).__name__}: {e}")
            tb = traceback.format_exc().strip().splitlines()
            for line in tb[-4:]:
                print(f"        {line}")
            fail += 1
            fails.append(label)
    total = ok + fail
    print(f"\n=== import smoke: {ok}/{total} PASS, {fail} FAIL ===")
    if fails:
        print("FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
