#!/usr/bin/env python3
"""GPTQ calibration for Gemma 4 26B-A4B — pure PyTorch, no external GPTQ libraries.

Monkey-patches HF Gemma4TextExperts to unfuse experts into per-expert nn.Linear,
then runs GPTQ calibration on ALL linear layers (including experts).

Outputs per-expert AWQ format (qweight/qzeros/scales) that our SGLang
load_weights() already supports.

Usage:
    cd /home/letsrtfm/AI/rdna4-inference-triton36
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/quantize_gemma4_gptq.py \
        --model-path ~/AI/models/gemma-4-26B-A4B-it-BF16 \
        --output-path ~/AI/models/gemma-4-26B-A4B-it-GPTQ-calibrated \
        --bits 4 --group-size 32 --num-samples 128
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPTQ core — pure PyTorch implementation
# ---------------------------------------------------------------------------

def quantize_weight_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
    damp_percent: float = 0.01,
    sym: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ quantize a single weight matrix.

    Args:
        W: [out_features, in_features] float weight matrix
        H: [in_features, in_features] Hessian (X^T X from calibration)
        bits: quantization bits (4)
        group_size: per-group quantization (32)

    Returns:
        qweight: [in_features // 8, out_features] int32 (AWQ packed)
        scales: [in_features // group_size, out_features] float16
        qzeros: [in_features // group_size, out_features // 8] int32 (AWQ packed)
    """
    rows, cols = W.shape  # [out, in]
    device = W.device
    dtype = W.dtype

    maxq = 2**bits - 1

    # Add dampening to Hessian diagonal
    diag = torch.diag(H)
    damp = damp_percent * torch.mean(diag)
    H = H + damp * torch.eye(cols, device=device, dtype=dtype)

    # Cholesky decomposition of H (for error propagation)
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        # Fallback: add more dampening
        H = H + 0.1 * torch.eye(cols, device=device, dtype=dtype)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)

    Hinv_diag = torch.diag(Hinv)

    W = W.clone().float()
    Q = torch.zeros_like(W)

    num_groups = math.ceil(cols / group_size)
    all_scales = torch.zeros(num_groups, rows, device=device, dtype=torch.float32)
    all_zeros = torch.zeros(num_groups, rows, device=device, dtype=torch.float32)

    # Process column groups
    for g in range(num_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, cols)
        group_cols = col_end - col_start

        # Compute scale and zero for this group
        w_group = W[:, col_start:col_end]
        w_min = w_group.min(dim=1).values
        w_max = w_group.max(dim=1).values

        if sym:
            w_abs_max = torch.max(w_min.abs(), w_max.abs())
            scale = w_abs_max / (maxq / 2)
            zero = torch.full_like(scale, maxq / 2)  # 8 for 4-bit
        else:
            scale = (w_max - w_min) / maxq
            zero = -w_min / scale

        scale = scale.clamp(min=1e-10)
        all_scales[g] = scale
        all_zeros[g] = zero

        # GPTQ: quantize each column and propagate error
        for j in range(col_start, col_end):
            w_col = W[:, j]
            q_col = torch.clamp(torch.round(w_col / scale + zero), 0, maxq)
            Q[:, j] = q_col

            # Quantization error
            err = (w_col - (q_col - zero) * scale)

            # Propagate error to remaining columns
            if j + 1 < cols:
                h_inv_j = Hinv_diag[j].clamp(min=1e-10)
                W[:, j + 1:] -= err.unsqueeze(1) * (Hinv[j, j + 1:].unsqueeze(0) / h_inv_j)

    # Pack into AWQ format
    qweight = pack_int4_awq(Q.to(torch.int32), cols, rows)
    scales_out = all_scales.to(torch.float16)  # [groups, out]
    qzeros = pack_int4_awq(
        all_zeros.to(torch.int32).unsqueeze(-1).expand(-1, -1, 1).squeeze(-1),
        num_groups, rows,
        is_zeros=True,
    )

    return qweight, scales_out, qzeros


def pack_int4_awq(
    intweight: torch.Tensor,
    in_features: int,
    out_features: int,
    is_zeros: bool = False,
) -> torch.Tensor:
    """Pack 4-bit integers into int32 AWQ format.

    AWQ packing order: [0, 4, 1, 5, 2, 6, 3, 7] within each group of 8.
    Input: [out_features, in_features] int values (0-15)
    Output: [in_features // 8, out_features] int32 packed
    """
    if is_zeros:
        # zeros: [groups, out] -> pack groups dim
        groups, out = intweight.shape
        packed = torch.zeros(groups, out // 8, dtype=torch.int32, device=intweight.device)
        awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
        for j in range(8):
            col_idx = awq_order[j]
            if col_idx < out % 8 if groups > 0 else False:
                continue
            # Pack groups of 8 output columns
            packed_cols = out // 8
            for pc in range(packed_cols):
                src_col = pc * 8 + awq_order[j]
                if src_col < out:
                    packed[:, pc] |= (intweight[:, src_col].to(torch.int32) & 0xF) << (j * 4)
        return packed

    # weights: [out, in] -> transpose to [in, out] then pack in-dim
    W = intweight.T.contiguous()  # [in, out]
    in_dim, out_dim = W.shape
    packed = torch.zeros(in_dim // 8, out_dim, dtype=torch.int32, device=W.device)

    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    for j in range(8):
        for row_block in range(in_dim // 8):
            src_row = row_block * 8 + awq_order[j]
            packed[row_block, :] |= (W[src_row, :].to(torch.int32) & 0xF) << (j * 4)

    return packed


# ---------------------------------------------------------------------------
# Monkey-patch: unfuse experts into per-expert nn.Linear
# ---------------------------------------------------------------------------

def patch_gemma4_experts():
    """Replace Gemma4TextExperts with per-expert nn.Linear for GPTQ calibration."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    OrigExperts = g4.Gemma4TextExperts

    class UnfusedExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = g4.ACT2FN[config.hidden_activation]

            self.gate_proj = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
                for _ in range(self.num_experts)
            ])
            self.up_proj = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
                for _ in range(self.num_experts)
            ])
            self.down_proj = nn.ModuleList([
                nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
                for _ in range(self.num_experts)
            ])

        def forward(self, hidden_states, routing_weights, selected_experts):
            batch_size = hidden_states.shape[0]
            final_hidden = torch.zeros_like(hidden_states)
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            for expert_idx in range(self.num_experts):
                idx, top_x = torch.where(expert_mask[expert_idx])
                if top_x.shape[0] == 0:
                    continue
                current_state = hidden_states[top_x]
                gate = self.act_fn(self.gate_proj[expert_idx](current_state))
                up = self.up_proj[expert_idx](current_state)
                current_hidden = gate * up
                current_hidden = self.down_proj[expert_idx](current_hidden)
                current_hidden *= routing_weights[top_x, idx].unsqueeze(-1)
                final_hidden.index_add_(0, top_x, current_hidden.to(final_hidden.dtype))
            return final_hidden

    g4.Gemma4TextExperts = UnfusedExperts
    logger.info("Patched Gemma4TextExperts → per-expert nn.Linear")
    return OrigExperts


# ---------------------------------------------------------------------------
# Activation collection via hooks
# ---------------------------------------------------------------------------

class ActivationCollector:
    """Collects input activations for each nn.Linear via forward hooks."""

    def __init__(self, model):
        self.hessians = {}
        self.nsamples = {}
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)
                self.hessians[name] = None
                self.nsamples[name] = 0

    def _make_hook(self, name):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            nsamples = x.shape[0]
            H = x.T @ x  # [in, in]
            if self.hessians[name] is None:
                self.hessians[name] = H
            else:
                self.hessians[name] += H
            self.nsamples[name] += nsamples
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_hessian(self, name):
        H = self.hessians.get(name)
        n = self.nsamples.get(name, 0)
        if H is not None and n > 0:
            return H / n
        return H


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

def get_calibration_data(tokenizer, num_samples=128, seq_len=512):
    from datasets import load_dataset
    logger.info(f"Loading C4 calibration data: {num_samples} samples, seq_len={seq_len}")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for example in ds:
        tokens = tokenizer(example["text"], return_tensors="pt",
                           truncation=True, max_length=seq_len)
        if tokens.input_ids.shape[1] >= seq_len // 2:
            samples.append(tokens.input_ids)
            if len(samples) >= num_samples:
                break
    logger.info(f"Collected {len(samples)} samples")
    return samples


def run_calibration(model, tokenizer, calibration_data):
    """Run calibration data through model to collect Hessians."""
    collector = ActivationCollector(model)
    model.eval()

    logger.info(f"Running {len(calibration_data)} calibration samples...")
    with torch.no_grad():
        for i, input_ids in enumerate(calibration_data):
            input_ids = input_ids.to(model.device)
            try:
                model(input_ids)
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
            if (i + 1) % 16 == 0:
                logger.info(f"  Calibration: {i+1}/{len(calibration_data)}")

    collector.remove_hooks()
    logger.info(f"Collected Hessians for {len(collector.hessians)} layers")
    return collector


def quantize_all_linears(model, collector, bits=4, group_size=32):
    """Quantize all nn.Linear layers using GPTQ with collected Hessians."""
    quantized_state = OrderedDict()
    total_layers = sum(1 for _ in model.named_modules() if isinstance(_[1], nn.Linear))
    done = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        W = module.weight.data.float()
        H = collector.get_hessian(name)

        if H is None:
            logger.warning(f"No Hessian for {name}, using RTN")
            H = torch.eye(W.shape[1], device=W.device, dtype=torch.float32)

        H = H.to(W.device)

        try:
            qweight, scales, qzeros = quantize_weight_gptq(
                W, H, bits=bits, group_size=group_size
            )
        except Exception as e:
            logger.warning(f"GPTQ failed for {name}: {e}, falling back to RTN")
            # RTN fallback
            H_eye = torch.eye(W.shape[1], device=W.device, dtype=torch.float32)
            qweight, scales, qzeros = quantize_weight_gptq(
                W, H_eye, bits=bits, group_size=group_size
            )

        quantized_state[f"{name}.qweight"] = qweight.cpu()
        quantized_state[f"{name}.scales"] = scales.cpu()
        quantized_state[f"{name}.qzeros"] = qzeros.cpu()

        done += 1
        if done % 50 == 0 or done == total_layers:
            logger.info(f"  Quantized {done}/{total_layers} layers")

    return quantized_state


def save_quantized_model(quantized_state, model, tokenizer, output_path, bits, group_size):
    """Save quantized weights + config in AWQ-compatible format."""
    os.makedirs(output_path, exist_ok=True)

    # Save non-linear parameters (norms, scalars, embeddings) as-is
    full_state = OrderedDict()
    for name, param in model.named_parameters():
        is_linear_weight = False
        for mod_name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name == f"{mod_name}.weight":
                is_linear_weight = True
                break
        if not is_linear_weight:
            full_state[name] = param.data.cpu()

    # Add quantized weights
    full_state.update(quantized_state)

    # Save as safetensors
    import safetensors.torch as st

    # Split into manageable shards (~4GB each)
    shard_size = 4 * 1024**3  # 4GB
    shards = {}
    current_shard = OrderedDict()
    current_size = 0
    shard_idx = 1

    for key, tensor in full_state.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
            shards[shard_name] = current_shard
            current_shard = OrderedDict()
            current_size = 0
            shard_idx += 1
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
        shards[shard_name] = current_shard

    # Fix shard names with actual total
    total = len(shards)
    weight_map = {}
    for old_name, shard_data in list(shards.items()):
        new_name = old_name.replace("TOTAL", f"{total:05d}")
        shard_path = os.path.join(output_path, new_name)
        st.save_file(shard_data, shard_path)
        for key in shard_data:
            weight_map[key] = new_name
        logger.info(f"Saved {new_name} ({len(shard_data)} tensors)")

    # Save index
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in full_state.values())},
             "weight_map": weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save config with quantization info
    import shutil
    config_src = model.config._name_or_path
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(config_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))

    # Update config with quantization info
    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
        "modules_to_not_convert": [],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    tokenizer.save_pretrained(output_path)
    logger.info(f"Saved quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    start_time = time.time()

    # 1. Patch experts
    patch_gemma4_experts()

    # 2. Load tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 3. Load model with unfused experts across GPUs
    logger.info("Loading BF16 model with device_map=auto...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    logger.info(f"Model loaded: {linear_count} nn.Linear layers to calibrate")

    # 4. Calibration data
    calibration_data = get_calibration_data(tokenizer, args.num_samples, args.seq_len)

    # 5. Collect Hessians
    logger.info("Collecting activation Hessians...")
    collector = run_calibration(model, tokenizer, calibration_data)

    # 6. Quantize
    logger.info("Running GPTQ quantization...")
    quantized_state = quantize_all_linears(model, collector, args.bits, args.group_size)

    # 7. Save
    save_quantized_model(quantized_state, model, tokenizer, args.output_path,
                         args.bits, args.group_size)

    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
