"""Register a calibration module for Nemotron-H's `NemotronHMoE` so llmcompressor's
`moe_calibration_context` routes ALL tokens through ALL experts during calibration
(every expert's GPTQ hessian sees data — without this, rare experts get garbage
scales → `<pad>`/garbage output, per our MoE ship rules).

llmcompressor 0.11.x moved all-expert MoE calibration out of GPTQModifier into a
registry of per-arch MoECalibrationModule subclasses applied by oneshot's
`moe_calibration_context` (default `moe_calibrate_all_experts=True`). Nemotron-H's
`NemotronHMoE` (latent projections + group routing + shared experts) isn't shipped
in the registry, so we register it here. Import this module before calling oneshot.

The forward mirrors NemotronHMoE.forward EXACTLY (same routing, weights, latent
projs, shared experts) so the model output is numerically unchanged — the only
difference is that under `calibrate_all_experts` each expert is run on the full
token batch (for hessian capture) before the routed contribution is gathered.
"""
import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("NemotronHMoE")
class CalibrationNemotronHMoE(MoECalibrationModule):
    is_permanent = False

    def __init__(self, original, config, calibrate_all_experts: bool = True):
        super().__init__()
        self.calibrate_all_experts = calibrate_all_experts
        # routing scalars (copied so we don't depend on the replaced original)
        self.n_routed_experts = original.n_routed_experts
        self.n_group = original.n_group
        self.topk_group = original.topk_group
        self.norm_topk_prob = original.norm_topk_prob
        self.routed_scaling_factor = original.routed_scaling_factor
        self.top_k = original.top_k
        # submodules (references — these carry the weights GPTQ quantizes)
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.fc1_latent_proj = original.fc1_latent_proj
        self.fc2_latent_proj = original.fc2_latent_proj

    # verbatim from NemotronHMoE.route_tokens_to_experts
    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_inputs = self.fc1_latent_proj(hidden_states)
        expert_outputs = torch.zeros_like(expert_inputs, dtype=topk_weights.dtype)

        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (experts, top_k, tokens)

        for expert_idx in range(self.n_routed_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if self.calibrate_all_experts:
                # run the expert on ALL tokens so its hessian sees full data
                out_all = self.experts[expert_idx](expert_inputs)
                if token_idx.numel() == 0:
                    continue
                current = out_all[token_idx] * topk_weights[token_idx, top_k_pos, None]
            else:
                if token_idx.numel() == 0:
                    continue
                current = self.experts[expert_idx](expert_inputs[token_idx]) \
                    * topk_weights[token_idx, top_k_pos, None]
            expert_outputs.index_add_(0, token_idx, current.to(expert_outputs.dtype))

        expert_outputs = expert_outputs.to(expert_inputs.dtype)
        hidden_states = self.fc2_latent_proj(expert_outputs)
        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
