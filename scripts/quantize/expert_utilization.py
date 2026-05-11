"""Expert utilization tracker for MoE calibration runs.

Implements the user's MoE calibration rule (memory feedback_moe_quant_best_practices.md):
"Monitor expert utilization during calibration — under-routed experts under-calibrate
and ship with degenerate scales."

Usage:
    from expert_utilization import ExpertUtilizationTracker
    tracker = ExpertUtilizationTracker(model, top_k=config.num_experts_per_tok)
    oneshot(model=model, ...)            # GPTQ forward passes accumulate counts
    report = tracker.report()             # dict {layer_idx -> stats}
    tracker.dump_json(path)               # persist alongside the AWQ output

Design notes:
- Hooks the router gate Linear (named `mlp.gate`) on every MoE layer.
- On each forward pass, top-k of the router logits gives the routed experts;
  bincount accumulates routing decisions per expert per layer.
- "Under-routed" = less than `min_frac` (default 0.5%) of total routing decisions.
- Hook fires CPU-side (we calibrate with device_map="cpu"), so no GPU tensor copies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class LayerStats:
    counts: torch.Tensor                  # int64 [num_experts]
    n_forward_calls: int = 0


class ExpertUtilizationTracker:
    """Forward-hook every `mlp.gate` Linear; record per-expert routing counts."""

    def __init__(
        self,
        model: nn.Module,
        top_k: int,
        min_frac: float = 0.005,
        verbose: bool = True,
    ):
        self.top_k = top_k
        self.min_frac = min_frac
        self.verbose = verbose
        self.layer_stats: dict[int, LayerStats] = {}
        self.hooks: list = []
        self._install(model)

    def _install(self, model: nn.Module):
        for name, mod in model.named_modules():
            # Match `model.layers.{N}.mlp.gate` — the router for Qwen3MoE / Qwen3_5MoE / Gemma4MoE
            if not name.endswith(".mlp.gate"):
                continue
            if not isinstance(mod, nn.Linear):
                continue
            try:
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            num_experts = mod.out_features
            self.layer_stats[layer_idx] = LayerStats(
                counts=torch.zeros(num_experts, dtype=torch.long)
            )
            h = mod.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(h)

        if self.verbose:
            print(
                f"[ExpertUtilizationTracker] hooked {len(self.layer_stats)} MoE router gates "
                f"(top_k={self.top_k}, min_frac threshold={self.min_frac:.1%})"
            )

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Linear, inp: tuple, out: torch.Tensor):
            with torch.no_grad():
                logits = out.detach().float()
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.shape[-1])
                topk = logits.topk(self.top_k, dim=-1).indices.flatten()
                bc = torch.bincount(topk, minlength=module.out_features)
                stats = self.layer_stats[layer_idx]
                stats.counts += bc.cpu().long()
                stats.n_forward_calls += 1
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def report(self) -> dict:
        out = {}
        for layer_idx in sorted(self.layer_stats):
            stats = self.layer_stats[layer_idx]
            counts = stats.counts
            total = int(counts.sum().item())
            if total == 0:
                out[layer_idx] = {"total_routed": 0, "warning": "no_routing_observed"}
                continue
            frac = counts.float() / total
            zero_routed = (counts == 0).nonzero().flatten().tolist()
            under_routed = ((frac < self.min_frac) & (counts > 0)).nonzero().flatten().tolist()
            out[layer_idx] = {
                "num_experts": int(counts.numel()),
                "total_routed": total,
                "n_forward_calls": stats.n_forward_calls,
                "min_frac": float(frac.min().item()),
                "min_frac_expert": int(frac.argmin().item()),
                "max_frac": float(frac.max().item()),
                "mean_frac": float(frac.mean().item()),
                "zero_routed_experts": zero_routed,
                "under_routed_experts": under_routed,
            }
        return out

    def dump_json(self, path: str) -> None:
        rep = self.report()
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        if self.verbose:
            print(f"[ExpertUtilizationTracker] report → {path}")

    def summary(self) -> str:
        rep = self.report()
        lines = ["=== Expert utilization report ==="]
        n_layers = len(rep)
        n_with_zero = sum(1 for s in rep.values() if s.get("zero_routed_experts"))
        n_with_under = sum(1 for s in rep.values() if s.get("under_routed_experts"))
        lines.append(
            f"Layers: {n_layers}  layers_with_zero_routed: {n_with_zero}  "
            f"layers_with_under_{self.min_frac:.1%}: {n_with_under}"
        )
        for layer_idx, s in rep.items():
            if s.get("zero_routed_experts") or s.get("under_routed_experts"):
                zr = s.get("zero_routed_experts", [])
                ur = s.get("under_routed_experts", [])
                lines.append(
                    f"  layer {layer_idx}: zero={zr}  "
                    f"under_{self.min_frac:.1%}={ur}  "
                    f"min_frac={s.get('min_frac', 0):.4%} (expert {s.get('min_frac_expert')})"
                )
        return "\n".join(lines)

    def has_blocking_issues(self) -> bool:
        """True if any expert in any layer was never routed during calibration."""
        rep = self.report()
        return any(s.get("zero_routed_experts") for s in rep.values())
