"""Monkey-patch Qwen3MoeExperts to use unfused per-expert ModuleList.

Apply BEFORE any `from_pretrained` of a Qwen3MoeForCausalLM checkpoint
(Coder-30B-A3B, Qwen3-30B-A3B, etc.). Fixes the silent garbage-init that
breaks REAM merging and any external tooling that reads checkpoint
per-expert weights (gate_proj/up_proj/down_proj as 384 separate tensors
per layer for Coder-30B with 128 experts).

Why:
    transformers 5.x defines `Qwen3MoeExperts` with fused 3D Parameters
    (`gate_up_proj [num_experts, 2*intermediate, hidden]` and
    `down_proj [num_experts, hidden, intermediate]`). When you load a
    checkpoint that stores per-expert weights, transformers silently
    drops them as UNEXPECTED and random-inits the fused 3D params.
    Saliency / merging / quantization downstream sees garbage.

    See `memory/project_ream_qwen3moe_root_cause.md` for the 2026-05-02
    investigation against task #62.

    The existing `transformers_disable_qwen3moe_fusion.patch` on
    `conversion_mapping.py` is necessary for some other paths but does
    NOT prevent class-level fusion — only this monkey-patch does.

Usage:
    # Place this file on PYTHONPATH or alongside your driver script.
    import qwen3moe_unfused_experts  # patches transformers in place
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct", ...)
    # m.model.layers[0].mlp.experts is now ModuleList[Qwen3MoeMLP], not 3D Parameters.
    # Per-expert weights load cleanly from the checkpoint (no UNEXPECTED).

Test (run after import):
    e0 = m.model.layers[0].mlp.experts[0]
    assert hasattr(e0, "gate_proj"), "experts[0] should be unfused MLP"
    assert e0.gate_proj.weight.shape[1] == m.config.hidden_size

Status (2026-05-02):
    Written but not yet exercised end-to-end through REAM `merge.py`.
    Once verified, the same import line at the top of
    `~/repos/ream/merge.py` (and any other Qwen3MoeForCausalLM tool we
    write) unblocks Coder-30B-A3B-REAM (#42, #62).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.models.qwen3_moe import modeling_qwen3_moe


class Qwen3MoeMLP(nn.Module):
    """Single-expert MLP matching the unfused checkpoint format.

    Module attributes match `experts.{i}.{gate,up,down}_proj.weight`
    naming so transformers' `_load_pretrained_model` finds them on the
    checkpoint side.
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3MoeExpertsUnfused(nn.ModuleList):
    """Drop-in replacement for `Qwen3MoeExperts` that IS a `nn.ModuleList`
    of per-expert `Qwen3MoeMLP` children.

    Two reasons for ModuleList specifically (not nn.Module + add_module):
    1. State_dict prefix matches checkpoint: `0.gate_proj.weight`, ... →
       combined with parent `mlp.experts.` gives the full
       `mlp.experts.0.gate_proj.weight` that Coder-30B ships.
    2. REAM merger.py + moe_utils.run_all_experts gate on
       `isinstance(moe_layer.experts, nn.ModuleList)` to dispatch
       per-expert vs fused-3D code paths. We need the per-expert path,
       so we MUST be a ModuleList instance — anything else (plain
       nn.Module, custom container) takes the fused-3D path that
       references missing `gate_up_proj` / `down_proj` 3D Parameters.
    """

    def __init__(self, config):
        super().__init__([Qwen3MoeMLP(config) for _ in range(config.num_experts)])
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # NOTE: do NOT keep self.act_fn here — ACT2FN values (e.g. nn.SiLU)
        # are nn.Module instances, and on a ModuleList subclass nn.Module
        # __setattr__ registers them as children. That inflates `len(self)`
        # from num_experts to num_experts+1 → REAM merger's get_num_experts()
        # returns 129 for a 128-expert model → router-gate view crashes with
        # `shape '[B, S, 129]' is invalid for input of size B*S*128`. Each
        # child Qwen3MoeMLP has its own act_fn, so the outer activation is
        # not needed for forward dispatch.

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # nn.ModuleList.forward raises NotImplementedError by default; override
        # so Qwen3MoeSparseMoeBlock can call self.experts(hidden, idx, w).
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item() if expert_idx.dim() else int(expert_idx)
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


# ---- Monkey-patch the symbol so AutoModelForCausalLM picks it up ----
modeling_qwen3_moe.Qwen3MoeExperts = Qwen3MoeExpertsUnfused

# Patch _init_weights too: the upstream version touches `module.gate_up_proj`
# and `module.down_proj` 3D Parameters which our unfused class doesn't have.
# Our nn.Linear children get sensible default init from nn.Linear itself; we
# just skip the manual initialization for our unfused expert collection.
_orig_init = modeling_qwen3_moe.Qwen3MoePreTrainedModel._init_weights


def _patched_init_weights(self, module):
    if isinstance(module, Qwen3MoeExpertsUnfused):
        return  # nn.Linear children already init themselves
    return _orig_init(self, module)


modeling_qwen3_moe.Qwen3MoePreTrainedModel._init_weights = _patched_init_weights
