#!/usr/bin/env python3
"""Convert a vLLM speculators-library DSpark draft (RedHatAI/*-speculator.dspark) to the
flat config layout SGLang's DSparkDraftModel reads (the RadixArk/*-DSpark layout).

The weights need no change (tensor names are identical); only config.json differs:
  * the backbone geometry sits under `transformer_layer_config` -> lifted to the top level
  * `aux_hidden_state_layer_ids` -> `target_layer_ids` (+ num_target_layers from the target)
  * the custom `auto_map` config class (imports the `speculators` package) is dropped
  * `mrope_section` is dropped: for a text-only draft every mrope axis carries the same
    position, which is exactly plain 1-D RoPE
Everything else (block_size, mask_token_id, markov/confidence heads, sample_from_anchor,
sliding-window layer_types) is copied from the source config.

Usage:
  python scripts/convert_dspark_speculators_config.py \
      --src /data/models/Qwen3.8-27B-speculator.dspark \
      --template /data/models/Qwen3.8-27B-DSpark/config.json \
      --out /data/models/Qwen3.8-27B-speculator.dspark-sgl --num-target-layers 64
The output dir gets config.json plus a symlink to the source model.safetensors.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="speculators-format draft dir")
ap.add_argument("--template", required=True, help="a flat-layout DSpark config.json (RadixArk) for the same target")
ap.add_argument("--out", required=True)
ap.add_argument("--num-target-layers", type=int, required=True, help="target model num_hidden_layers")
a = ap.parse_args()

rh = json.load(open(os.path.join(a.src, "config.json")))
rx = json.load(open(a.template))
tl = rh["transformer_layer_config"]
ids = rh["aux_hidden_state_layer_ids"]
c = dict(rx)
over = {
    "architectures": ["DSparkDraftModel"], "model_type": tl.get("model_type", "qwen3"),
    "block_size": rh["block_size"], "mask_token_id": rh["mask_token_id"],
    "target_layer_ids": ids, "num_target_layers": a.num_target_layers,
    "head_dim": tl["head_dim"], "num_attention_heads": tl["num_attention_heads"],
    "num_key_value_heads": tl["num_key_value_heads"], "hidden_size": tl["hidden_size"],
    "intermediate_size": tl["intermediate_size"], "num_hidden_layers": tl["num_hidden_layers"],
    "layer_types": tl["layer_types"], "sliding_window": tl.get("sliding_window"),
    "use_sliding_window": tl.get("use_sliding_window", False), "max_window_layers": tl["num_hidden_layers"],
    "rms_norm_eps": tl["rms_norm_eps"], "hidden_act": tl["hidden_act"], "attention_bias": tl.get("attention_bias", False),
    "max_position_embeddings": tl["max_position_embeddings"],
    "rope_theta": tl["rope_parameters"]["rope_theta"],
    "rope_parameters": {"rope_type": "default", "rope_theta": tl["rope_parameters"]["rope_theta"]},
    "rope_scaling": None,
    "markov_rank": rh["markov_rank"], "markov_head_type": rh["markov_head_type"],
    "enable_confidence_head": rh["enable_confidence_head"], "confidence_head_with_markov": rh["confidence_head_with_markov"],
    "sample_from_anchor": rh.get("sample_from_anchor", True), "draft_vocab_size": rh["draft_vocab_size"],
    "vocab_size": tl["vocab_size"], "tie_word_embeddings": False, "training_block_size": rh["block_size"],
    "transformers_version": rh.get("transformers_version"),
    "_converted_from": f"{os.path.basename(a.src.rstrip('/'))} (speculators {rh.get('speculators_version')}) -> SGLang flat DSpark layout; mrope_section dropped (text-only: identical positions on all axes == plain 1D rope)",
}
c.update(over)
for sub in ("dflash_config", "dspark_config"):
    if sub in c:
        d = dict(c[sub]); d.update({"mask_token_id": rh["mask_token_id"], "target_layer_ids": ids,
                                    "markov_rank": rh["markov_rank"], "markov_head_type": rh["markov_head_type"],
                                    "enable_confidence_head": rh["enable_confidence_head"],
                                    "confidence_head_with_markov": rh["confidence_head_with_markov"]}); c[sub] = d
c.pop("auto_map", None)
os.makedirs(a.out, exist_ok=True)
json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=1)
link = os.path.join(a.out, "model.safetensors")
if not os.path.exists(link):
    os.symlink(os.path.join(os.path.abspath(a.src), "model.safetensors"), link)
print("wrote", a.out, "gamma", rh["block_size"], "target_layer_ids", ids)
