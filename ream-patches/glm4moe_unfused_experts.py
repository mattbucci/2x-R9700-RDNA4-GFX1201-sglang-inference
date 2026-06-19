"""Monkey-patch Glm4MoeNaiveMoe to use an unfused per-expert ModuleList.

Apply BEFORE any `from_pretrained` of a Glm4MoeForCausalLM checkpoint
(GLM-4.5-Air, etc.) on transformers >= 5.x. Fixes the silent garbage-init
that breaks REAM merging and any tooling that reads per-expert weights.

Why (mirrors the qwen3moe case — see ream-patches/qwen3moe_unfused_experts.py):
    transformers 5.x defines `Glm4MoeNaiveMoe` with FUSED 3D Parameters
    (`gate_up_proj [num_experts, 2*moe_intermediate, hidden]` and
    `down_proj   [num_experts, hidden, moe_intermediate]`). GLM-4.5-Air
    ships PER-EXPERT weights (`experts.{i}.{gate,up,down}_proj.weight`,
    17664 tensors for 128 experts x 45 MoE layers). transformers silently
    drops them as UNEXPECTED and random-inits the fused params → every
    expert is garbage → REAM saliency/merging operates on noise.

    Verified 2026-06-17 on /data/models/GLM-4.5-Air-BF16 + transformers
    5.8.1: meta-device state_dict shows model wants 90 fused keys (missing
    from ckpt) and ignores all 17664 per-expert keys. This patch makes the
    experts a ModuleList[Glm4MoeMLP] so the per-expert weights load cleanly.

Usage:
    import glm4moe_unfused_experts   # patches transformers in place
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("/data/models/GLM-4.5-Air-BF16", ...)
    # m.model.layers[1].mlp.experts is now ModuleList[Glm4MoeMLP].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.glm4_moe import modeling_glm4_moe
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP


class Glm4MoeExpertsUnfused(nn.ModuleList):
    """Drop-in replacement for `Glm4MoeNaiveMoe` that IS a `nn.ModuleList`
    of per-expert `Glm4MoeMLP` children, each sized to moe_intermediate_size.

    ModuleList specifically (not nn.Module) for two reasons:
    1. State_dict prefix matches the checkpoint: `0.gate_proj.weight`, ...
       combined with parent `mlp.experts.` → `mlp.experts.0.gate_proj.weight`.
    2. REAM merger.py / moe_utils gate on `isinstance(experts, nn.ModuleList)`
       to dispatch the per-expert (vs fused-3D) code path.
    """

    def __init__(self, config):
        num_experts = getattr(config, "num_local_experts", None) or config.n_routed_experts
        experts = [
            Glm4MoeMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(num_experts)
        ]
        super().__init__(experts)
        self.num_experts = num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # nn.ModuleList.forward raises NotImplementedError by default; override
        # so Glm4MoeMoE.forward can call self.experts(hidden, idx, weights).
        # Logic mirrors the stock Glm4MoeNaiveMoe.forward, dispatching per child.
        # Use len(self), NOT self.num_experts: the REAM merger deletes ModuleList entries
        # in-place (128 -> 96) without updating num_experts, so len(self) is the live count.
        n_experts = len(self)
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=n_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item() if expert_idx.dim() else int(expert_idx)
            if expert_idx == n_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


# ---- Monkey-patch the symbol so Glm4MoeMoE.__init__ picks it up ----
modeling_glm4_moe.Glm4MoeNaiveMoe = Glm4MoeExpertsUnfused

# Skip the upstream _init_weights for our unfused collection: the stock version
# touches `module.gate_up_proj` / `module.down_proj` 3D Parameters our class
# doesn't have. nn.Linear children init themselves.
_orig_init = modeling_glm4_moe.Glm4MoePreTrainedModel._init_weights


def _patched_init_weights(self, module):
    if isinstance(module, Glm4MoeExpertsUnfused):
        return
    return _orig_init(self, module)


modeling_glm4_moe.Glm4MoePreTrainedModel._init_weights = _patched_init_weights
