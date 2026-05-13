#!/usr/bin/env python3
"""Homegrown REAP (Router-aware Expert Pruning, arxiv:2510.13999) — pure
pytorch + transformers, no vLLM dependency.

Algorithm (per-layer):
  1. Forward `--num-samples` calibration prompts through the BF16 model.
  2. On each MoE layer, hook the router (`mlp.gate`) to capture per-token
     routing probabilities, and hook each expert's `down_proj` to capture
     per-token output L2 norms.
  3. For each expert E in layer L: accumulate its saliency as
        S_{L,E} = sum_t (gate_t_E * ||down_proj_E(intermediate_t_E)||_2)
     summing only over tokens that routed to expert E (top-k routing).
  4. Per layer, keep the top-`--keep-experts` highest-saliency experts;
     drop the rest.
  5. Re-pack the model: ModuleList[Qwen3MoeMLP] is filtered to surviving
     experts; router gate weight rows are sliced to the same surviving set.
     `config.num_experts` and `num_local_experts` are updated to the new size.
  6. Save the pruned BF16 + tokenizer + a `reap_report.json` describing
     which experts survived per layer and the saliency distribution.

Compared to Cerebras's REAP tool (github.com/CerebrasResearch/reap), this
implementation:
  * pure pytorch — no vLLM dependency (their tool requires vLLM 0.10 pinned)
  * controls memory via explicit `max_memory` per device (their tool uses
    accelerate's auto split which OOMs on R9700 with Coder-30B)
  * per-layer pruning (matches the Cerebras paper)
  * router renormalization is implicit (softmax over surviving rows)

Reusable across Qwen3MoE / Qwen3_5MoE / Gemma4MoE / Nemotron-3 — the hook
selectors only require `.mlp.gate` and `.mlp.experts.<i>.down_proj` paths,
which all common HF MoE archs use.

Usage (text-only):
    conda activate vllm   # (or any env with transformers + accelerate + datasets)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/quantize/run_reap.py \\
        --model ~/AI/models/Qwen3-Coder-30B-A3B-BF16 \\
        --save-path ~/AI/models/Qwen3-Coder-30B-A3B-REAP-BF16 \\
        --keep-experts 96 \\
        --dataset theblackcat102/evol-codealpaca-v1 \\
        --num-samples 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Apply Qwen3MoeExperts unfused-experts monkey-patch BEFORE from_pretrained.
# Required so per-expert nn.Modules survive — the fused 3D Parameter form
# can't be hook-instrumented per-expert.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATCH_DIR = os.path.join(_REPO_DIR, "patches")
if os.path.isfile(os.path.join(_PATCH_DIR, "qwen3moe_unfused_experts.py")):
    sys.path.insert(0, _PATCH_DIR)
    try:
        import qwen3moe_unfused_experts  # noqa: F401
        print("[run_reap] Qwen3MoeExperts → Qwen3MoeExpertsUnfused (per-expert ModuleList)")
    except ImportError as e:
        print(f"[run_reap] WARNING: failed to apply unfused patch: {e}")


class REAPSaliencyTracker:
    """Hook every MoE router + expert.down_proj; accumulate per-expert saliency.

    The saliency formula is a simplification of the Cerebras paper:
        S_{L,E} = sum_t (gate_t_E * ||down_proj_E(intermediate_t_E)||_2)
    where the sum is over calibration tokens that the router sent to expert E
    (i.e., E was in the top-k for token t).

    We accumulate per-expert across the full calibration corpus, then
    `top_k_survivors` returns the layer-wise pruned set.
    """

    def __init__(self, model: torch.nn.Module, top_k: int, verbose: bool = True):
        self.top_k = top_k
        self.verbose = verbose
        # layer_idx -> tensor[num_experts] (float64 for accumulation precision)
        self.saliency: dict[int, torch.Tensor] = {}
        # transient: layer_idx -> (topk_probs[B*T,top_k], topk_idx[B*T,top_k]) cpu
        self._routing_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks: list = []
        self._install(model)

    def _install(self, model: torch.nn.Module):
        n_routers = 0
        n_experts_hooked = 0
        for name, mod in model.named_modules():
            if name.endswith(".mlp.gate") and isinstance(mod, torch.nn.Linear):
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                self.saliency[layer_idx] = torch.zeros(mod.out_features, dtype=torch.float64)
                h = mod.register_forward_hook(self._make_router_hook(layer_idx))
                self.hooks.append(h)
                n_routers += 1
            elif ".mlp.experts." in name and name.endswith(".down_proj") and isinstance(mod, torch.nn.Linear):
                # name = model.layers.{L}.mlp.experts.{E}.down_proj
                try:
                    parts = name.split(".")
                    L = int(parts[parts.index("layers") + 1])
                    E = int(parts[parts.index("experts") + 1])
                except (IndexError, ValueError):
                    continue
                h = mod.register_forward_hook(self._make_expert_hook(L, E))
                self.hooks.append(h)
                n_experts_hooked += 1
        if self.verbose:
            print(f"[REAPSaliencyTracker] hooked {n_routers} routers + {n_experts_hooked} expert.down_proj modules")

    def _make_router_hook(self, layer_idx: int):
        def hook(module, inp, out):
            with torch.no_grad():
                logits = out.detach().float()
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.shape[-1])  # flatten batch + seq
                probs = logits.softmax(dim=-1)
                topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
                # Cache on CPU so expert hooks can read it without GPU sync overhead per call
                self._routing_cache[layer_idx] = (topk_probs.cpu(), topk_idx.cpu())
        return hook

    def _make_expert_hook(self, layer_idx: int, expert_idx: int):
        def hook(module, inp, out):
            with torch.no_grad():
                # out: [num_tokens_routed_to_this_expert, hidden_size]
                if out.numel() == 0:
                    return
                # Per-routed-token L2 norm of the expert's output
                output_norms = out.detach().float().norm(dim=-1)  # [num_routed]
                num_routed = output_norms.numel()
                if num_routed == 0:
                    return
                # Lookup average routing prob this expert received in the current batch
                # (we don't have token-by-token alignment between the gate hook's flat
                # output and the expert's filtered token batch, so we use the mean
                # routing prob for this expert as an approximation; sum_t gate_t * norm
                # collapses to mean(gate) * sum(norm) under that approximation, which
                # matches Cerebras's gating-weighted formula in expectation).
                cache = self._routing_cache.get(layer_idx)
                if cache is None:
                    weight = 1.0
                else:
                    topk_probs, topk_idx = cache
                    # mask of (token, k) positions where this expert was selected
                    mask = (topk_idx == expert_idx)
                    if mask.any():
                        weight = topk_probs[mask].mean().item()
                    else:
                        weight = 0.0
                contribution = float(weight) * float(output_norms.sum().item())
                self.saliency[layer_idx][expert_idx] += contribution
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self._routing_cache.clear()

    def survivors_per_layer(self, k: int) -> dict[int, list[int]]:
        """For each MoE layer return the top-k expert indices (sorted ascending)."""
        out = {}
        for L in sorted(self.saliency):
            sal = self.saliency[L]
            top = sal.topk(min(k, sal.numel())).indices.tolist()
            out[L] = sorted(top)
        return out


def prune_model(model: torch.nn.Module, survivors_per_layer: dict[int, list[int]]) -> None:
    """In-place: drop pruned experts and slice router gate weight rows to surviving set.

    Assumes Qwen3MoE-style ModuleList[Qwen3MoeMLP] for experts (the unfused-experts
    monkey-patch ensures this) and an `mlp.gate` Linear of shape [num_experts, hidden].
    """
    for layer_idx, keep in survivors_per_layer.items():
        layer_path = f"model.layers.{layer_idx}.mlp"
        # Walk the module tree to find this layer's mlp
        layer_mlp = model
        for part in layer_path.split("."):
            layer_mlp = getattr(layer_mlp, part)
        # Slice the gate Linear weight (rows = experts)
        gate = layer_mlp.gate
        keep_t = torch.tensor(keep, dtype=torch.long, device=gate.weight.device)
        new_gate = torch.nn.Linear(gate.in_features, len(keep), bias=(gate.bias is not None),
                                   device=gate.weight.device, dtype=gate.weight.dtype)
        with torch.no_grad():
            new_gate.weight.copy_(gate.weight.index_select(0, keep_t))
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias.index_select(0, keep_t))
        layer_mlp.gate = new_gate
        # Filter the experts ModuleList
        old_experts = layer_mlp.experts
        new_experts = torch.nn.ModuleList([old_experts[i] for i in keep])
        layer_mlp.experts = new_experts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or HF id of BF16 base model")
    p.add_argument("--save-path", required=True, help="Output dir for pruned BF16")
    p.add_argument("--keep-experts", type=int, default=96, help="Experts to KEEP per layer")
    p.add_argument("--dataset", default="theblackcat102/evol-codealpaca-v1")
    p.add_argument("--text-field", default=None,
                   help="Field name in the dataset for the prompt text. Default = auto-detect.")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-memory-per-gpu", default="28GiB",
                   help="Per-GPU max memory for accelerate device_map (e.g. 28GiB)")
    args = p.parse_args()

    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    print(f"Model:    {args.model}")
    print(f"Save:     {args.save_path}")
    print(f"Keep:     {args.keep_experts} experts per layer")
    print(f"Dataset:  {args.dataset} ({args.num_samples} samples × {args.max_length} tokens)")
    print(f"RAM:      {ram_gb:.1f} GB")

    # Build max_memory dict for accelerate
    n_visible = torch.cuda.device_count()
    max_memory = {i: args.max_memory_per_gpu for i in range(n_visible)}
    max_memory["cpu"] = "100GiB"
    print(f"max_memory: {max_memory} (n_visible_gpus={n_visible})")

    print("\n[1/4] Loading tokenizer + model on multi-GPU...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s ({type(model).__name__})")

    top_k = getattr(model.config, "num_experts_per_tok", 8)
    num_experts = getattr(model.config, "num_experts", None) or getattr(model.config, "num_local_experts", None)
    print(f"  num_experts={num_experts}, top_k={top_k}")
    if args.keep_experts >= num_experts:
        raise ValueError(f"--keep-experts {args.keep_experts} ≥ num_experts {num_experts}; nothing to prune")

    print("\n[2/4] Installing REAPSaliencyTracker hooks...")
    tracker = REAPSaliencyTracker(model, top_k=top_k)

    print(f"\n[3/4] Loading {args.dataset} + running observation pass...")
    ds = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
    print(f"  loaded {len(ds)} samples; columns: {ds.column_names[:6]}")

    # Auto-detect text field
    text_field = args.text_field
    if text_field is None:
        for cand in ("text", "instruction", "prompt", "content", "input", "messages"):
            if cand in ds.column_names:
                text_field = cand
                break
    if text_field is None:
        text_field = ds.column_names[0]
    print(f"  using text-field: {text_field!r}")

    t0 = time.time()
    n_processed = 0
    for i, sample in enumerate(ds):
        text = sample.get(text_field) or ""
        # Some datasets pair instruction+output; concatenate if available
        if "output" in sample and sample.get("output"):
            text = f"{text}\n{sample['output']}"
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt", max_length=args.max_length,
                        truncation=True).input_ids.to(next(model.parameters()).device)
        try:
            with torch.no_grad():
                model(ids)
            n_processed += 1
        except Exception as e:
            print(f"  WARN: sample {i} failed ({type(e).__name__}: {e})")
            continue
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / rate
            print(f"  {i+1}/{len(ds)}  ({rate:.1f} sample/s, ETA {eta/60:.1f}min)")
    elapsed = time.time() - t0
    print(f"  observation done: {n_processed}/{len(ds)} samples in {elapsed/60:.1f}min")

    tracker.remove()

    print(f"\n[4/4] Computing per-layer survivors (keep top-{args.keep_experts})...")
    survivors = tracker.survivors_per_layer(args.keep_experts)
    n_layers = len(survivors)
    print(f"  {n_layers} MoE layers; example survivors layer 0: {survivors[0][:10]}...")

    print(f"\nPruning experts + slicing router weights in-place...")
    prune_model(model, survivors)

    # Update config
    model.config.num_experts = args.keep_experts
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = args.keep_experts

    os.makedirs(args.save_path, exist_ok=True)
    print(f"\nSaving pruned BF16 to {args.save_path}...")
    model.save_pretrained(args.save_path, max_shard_size="2GB")
    tokenizer.save_pretrained(args.save_path)

    # Persist REAP report alongside the model
    report = {
        "args": vars(args),
        "num_layers_pruned": n_layers,
        "kept_experts_per_layer": args.keep_experts,
        "original_num_experts": num_experts,
        "saliency_per_layer": {
            str(L): tracker.saliency[L].tolist() for L in tracker.saliency
        },
        "survivors_per_layer": {str(L): survivors[L] for L in survivors},
    }
    with open(os.path.join(args.save_path, "reap_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("  reap_report.json written")
    print("\nDone.")


if __name__ == "__main__":
    main()
