"""Calibration dataset builder for AWQ/GPTQ quantization.

Past calibrations used text-only Open-Platypus which silently regressed:
  - Qwen3.5 thinking: infinite <think> loop because no thinking traces in calibration
  - Devstral/Gemma4 vision: vision projector drifts because no image tokens

This module assembles mixed calibration sets that preserve all capabilities
a model ships with. Use via `build_calibration_dataset(...)` below.

Dataset choices (2026-04):
  - Thinking:  a-m-team/AM-Thinking-v1-Distilled   (Qwen3-generated, verified <think> tags)
  - Thinking:  glaiveai/reasoning-v1-20m           (native <think>, 22M rows, fallback)
  - Math+CoT:  AI-MO/NuminaMath-CoT                (+9.81% GPTQ accuracy vs WikiText2)
  - Chat:      HuggingFaceH4/ultrachat_200k        (general multi-turn dialogue)
  - Vision:    liuhaotian/LLaVA-Instruct-150K      (image + conversation pairs)
  - Code:      bigcode/the-stack-smol              (domain data for coder models)

Chat template is applied with `enable_thinking=True` when the model supports it,
so the resulting calibration text contains the exact `<think>...</think>` structure
the model must learn to terminate.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable

from datasets import Dataset, load_dataset


@dataclass
class Mix:
    """A weighted slice of a HF dataset with a formatter that turns it into messages."""
    name: str
    hf_name: str
    split: str
    weight: float
    format_fn: Callable[[dict], list[dict]]
    streaming: bool = False
    config: str | None = None


def _am_thinking(row: dict) -> list[dict]:
    """AM-Thinking-v1-Distilled: `conversations` list (from/value) + optional `system`."""
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    for turn in row.get("conversations", []):
        role = {"human": "user", "gpt": "assistant"}.get(turn.get("from"), turn.get("from", "user"))
        msgs.append({"role": role, "content": turn.get("value", "")})
    return msgs


def _glaive_reasoning(row: dict) -> list[dict]:
    """glaiveai/reasoning-v1-20m: `prompt` + `response` (response already has <think>)."""
    return [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]


def _numina_math(row: dict) -> list[dict]:
    """NuminaMath-CoT: `problem` + `solution`."""
    return [
        {"role": "user", "content": row["problem"]},
        {"role": "assistant", "content": row["solution"]},
    ]


def _ultrachat(row: dict) -> list[dict]:
    """HF ultrachat_200k: already in messages format."""
    return row["messages"]


def _llava_instruct(row: dict) -> list[dict]:
    """LLaVA-Instruct-150K: `conversations` list of {"from","value"}. Images handled separately."""
    msgs = []
    for turn in row.get("conversations", []):
        role = {"human": "user", "gpt": "assistant"}.get(turn["from"], turn["from"])
        msgs.append({"role": role, "content": turn["value"]})
    return msgs


def _thestack_code(row: dict) -> list[dict]:
    """bigcode/the-stack-smol: raw code as assistant response to 'show me code' prompt."""
    content = row.get("content", "")[:4000]
    return [
        {"role": "user", "content": "Show me a code example."},
        {"role": "assistant", "content": content},
    ]


# --- Registry of available mixes ---

MIXES: dict[str, Mix] = {
    "am_thinking": Mix(
        "am_thinking", "a-m-team/AM-Thinking-v1-Distilled",
        split="train", weight=0.0, format_fn=_am_thinking,
        streaming=True,  # Full download errors on DatasetGenerationError; streaming works.
    ),
    "glaive_reasoning": Mix(
        "glaive_reasoning", "glaiveai/reasoning-v1-20m",
        split="train", weight=0.0, format_fn=_glaive_reasoning, streaming=True,
    ),
    "numina_math": Mix(
        "numina_math", "AI-MO/NuminaMath-CoT",
        split="train", weight=0.0, format_fn=_numina_math,
    ),
    "ultrachat": Mix(
        "ultrachat", "HuggingFaceH4/ultrachat_200k",
        split="train_sft", weight=0.0, format_fn=_ultrachat,
    ),
    "llava_instruct": Mix(
        "llava_instruct", "liuhaotian/LLaVA-Instruct-150K",
        split="train", weight=0.0, format_fn=_llava_instruct,
    ),
    "thestack_code": Mix(
        "thestack_code", "bigcode/the-stack-smol",
        split="train", weight=0.0, format_fn=_thestack_code,
        config="data/python",
    ),
}


# --- Per-model recipes ---

# Fractions must sum to 1.0.  Keys must exist in MIXES.

RECIPE_THINKING_TEXT = {
    # Text-only thinking model (Qwen3.5-27B dense, Qwen3.5-35B MoE).
    "am_thinking": 0.50,
    "numina_math": 0.25,
    "ultrachat": 0.25,
}

RECIPE_THINKING_VISION = {
    # Thinking + vision (Gemma4 26B MoE, Gemma4 31B).
    # Image pairs MUST be present so the MM projector sees real vision activations.
    "am_thinking": 0.40,
    "llava_instruct": 0.30,
    "numina_math": 0.15,
    "ultrachat": 0.15,
}

RECIPE_CODE_VISION = {
    # Coder + vision (Devstral).  Code heavy, vision preserved, minimal thinking.
    "thestack_code": 0.45,
    "llava_instruct": 0.25,
    "ultrachat": 0.20,
    "numina_math": 0.10,
}

RECIPE_CODE_THINKING = {
    # Coder with optional thinking (Qwen3-Coder-30B, Coder-Next 80B).
    "thestack_code": 0.40,
    "am_thinking": 0.25,
    "numina_math": 0.20,
    "ultrachat": 0.15,
}

RECIPES = {
    "thinking_text": RECIPE_THINKING_TEXT,
    "thinking_vision": RECIPE_THINKING_VISION,
    "code_vision": RECIPE_CODE_VISION,
    "code_thinking": RECIPE_CODE_THINKING,
}


# --- Builder ---


def _load_slice(mix: Mix, n: int, seed: int) -> list[dict]:
    """Load `n` rows from a mix, returning raw HF rows.  Streaming-safe."""
    print(f"  [{mix.name}] loading {n} samples from {mix.hf_name}")
    try:
        if mix.streaming:
            ds = load_dataset(
                mix.hf_name,
                name=mix.config,
                split=mix.split,
                streaming=True,
            )
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            rows = []
            for row in ds:
                rows.append(row)
                if len(rows) >= n:
                    break
            return rows
        else:
            ds = load_dataset(
                mix.hf_name,
                name=mix.config,
                split=f"{mix.split}[:{max(n * 3, n + 100)}]",
            )
            ds = ds.shuffle(seed=seed)
            return [ds[i] for i in range(min(n, len(ds)))]
    except Exception as e:
        print(f"  [{mix.name}] FAILED to load: {e!r}")
        return []


def build_calibration_dataset(
    recipe: str | dict,
    num_samples: int,
    seed: int = 42,
    fallback_mix: str = "ultrachat",
) -> list[dict]:
    """Build a mixed calibration dataset as a list of rows.

    Each returned row is a dict with:
      - "messages": list[{"role", "content"}]  (always present)
      - "images":   list[PIL.Image or path]     (only for vision mixes, may be empty)
      - "source":   str                         (which mix it came from)

    Args:
      recipe: recipe name (e.g. "thinking_vision") or dict {mix_name: weight}
      num_samples: total rows across all mixes (rounded per-mix)
      seed: shuffle seed
      fallback_mix: if a mix fails to load, pad with this one
    """
    if isinstance(recipe, str):
        if recipe not in RECIPES:
            raise ValueError(f"Unknown recipe {recipe!r}.  Available: {list(RECIPES)}")
        weights = RECIPES[recipe]
    else:
        weights = recipe

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-3:
        raise ValueError(f"Recipe weights must sum to 1.0, got {total_weight}")

    print(f"Building calibration set ({num_samples} samples, recipe={recipe}):")
    for name, w in weights.items():
        print(f"  {name:<20} {w:.2%} -> {int(round(num_samples * w))} samples")

    rows: list[dict] = []
    for mix_name, weight in weights.items():
        if mix_name not in MIXES:
            print(f"  [{mix_name}] NOT in registry, skipping")
            continue
        n = int(round(num_samples * weight))
        raw = _load_slice(MIXES[mix_name], n, seed)
        mix = MIXES[mix_name]
        for row in raw:
            messages = mix.format_fn(row)
            images = row.get("image") or row.get("images") or []
            if images and not isinstance(images, list):
                images = [images]
            rows.append({
                "messages": messages,
                "images": images,
                "source": mix.name,
            })

    # Pad to num_samples with fallback if any mix under-delivered
    deficit = num_samples - len(rows)
    if deficit > 0 and fallback_mix in MIXES:
        print(f"  Padding {deficit} samples from {fallback_mix}")
        extra = _load_slice(MIXES[fallback_mix], deficit, seed + 1)
        fmt = MIXES[fallback_mix].format_fn
        for row in extra:
            rows.append({"messages": fmt(row), "images": [], "source": fallback_mix})

    random.Random(seed).shuffle(rows)
    rows = rows[:num_samples]

    # Report what we actually got
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print("Final mix:")
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {c:4d}  ({c/len(rows):.1%})")

    return rows


# --- Text rendering for text-only GPTQ (no multimodal) ---


def rows_to_text(
    rows: list[dict],
    tokenizer,
    enable_thinking: bool = True,
    drop_images: bool = True,
    max_samples: int | None = None,
) -> Dataset:
    """Render rows into a text-only HF Dataset for llm-compressor oneshot.

    Applies chat template with thinking enabled so that <think> tags appear in
    the calibration text.  Vision rows have images dropped here — use
    `rows_to_multimodal` for models with image encoders.
    """
    out = []
    for row in rows[:max_samples] if max_samples else rows:
        if row["images"] and drop_images:
            # Strip <image> placeholders from message content if present
            msgs = [
                {**m, "content": m["content"].replace("<image>", "").strip()}
                for m in row["messages"]
            ]
        else:
            msgs = row["messages"]
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers don't accept enable_thinking
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
        if text:
            out.append({"text": text})
    return Dataset.from_list(out)


def tokenize_text_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    """Pre-tokenize a text-only dataset for llm-compressor oneshot().

    Matches the pattern used by existing `quantize_qwen35_llmcompressor.py`:
    returns a dataset with `input_ids` + `attention_mask`, text column dropped.
    """
    def _tok(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    return dataset.map(_tok, remove_columns=dataset.column_names)


def verify_thinking_preserved(dataset: Dataset, min_fraction: float = 0.10) -> None:
    """Sanity check: confirm <think>...</think> appears in the rendered text.

    If the chat template silently dropped thinking tags, this catches it before
    we spend 6h calibrating on the wrong data.
    """
    n = 0
    for row in dataset:
        if "<think>" in row["text"] or "<|channel>" in row["text"]:
            n += 1
    frac = n / max(1, len(dataset))
    print(f"Thinking-tagged rows: {n}/{len(dataset)} ({frac:.1%})")
    if frac < min_fraction:
        raise RuntimeError(
            f"Only {frac:.1%} of calibration rows contain thinking tags "
            f"(expected >={min_fraction:.0%}).  "
            f"The chat template may be stripping them — check tokenizer config "
            f"and enable_thinking flag."
        )


if __name__ == "__main__":
    # Smoke test: load 64 samples of the thinking_text recipe
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default="thinking_text", choices=list(RECIPES))
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()

    rows = build_calibration_dataset(args.recipe, args.n)
    print(f"\nGot {len(rows)} rows")
    print("First row sample:")
    print(f"  source: {rows[0]['source']}")
    print(f"  images: {len(rows[0]['images'])}")
    print(f"  messages[0]: {str(rows[0]['messages'][0])[:200]}")
