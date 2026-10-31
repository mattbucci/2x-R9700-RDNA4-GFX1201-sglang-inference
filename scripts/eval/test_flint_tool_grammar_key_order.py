#!/usr/bin/env python3
"""CPU-only test: the Flint strict tool grammar must accept a call that writes the payload key first even
when the scaffold's schema declares a metadata key first (oh-my-pi: ``i`` before ``command``).

XGrammar 0.2.7 compiles a strict object schema in declaration order, so without the reorder in
``flint_structural_tags._as_strict`` the grammar forces ``<arg_key>i</arg_key>`` first and the model puts the
command in it (3090 rig, 2026-10-28). Compiles the real grammar with the served tokenizer and checks the
accepted / rejected strings token by token.

    python scripts/eval/test_flint_tool_grammar_key_order.py [--model /data/models/Flint-FP8]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/data/sgl-v0522/python")


def accepts(compiler, tag, tokenizer, text: str) -> bool:
    import xgrammar as xgr
    compiled = compiler.compile_structural_tag(tag.model_dump_json())
    matcher = xgr.GrammarMatcher(compiled)
    for tid in tokenizer.encode(text, add_special_tokens=False):
        if not matcher.accept_token(tid):
            return False
    return matcher.is_terminated() or matcher.accept_token(tokenizer.convert_tokens_to_ids("<|user|>"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FLINT_MODEL", "/data/models/Flint-FP8"))
    a = ap.parse_args()
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from sglang.srt.constrained import flint_structural_tags as fst

    tok = AutoTokenizer.from_pretrained(a.model)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok))
    compiler = xgr.GrammarCompiler(info)
    # strict=False is what SGLang's Tool.model_dump() sends (Function.strict defaults False)
    omp_bash = {"type": "function", "function": {"name": "bash", "description": "run a command", "strict": False, "parameters": {
        "type": "object",
        "properties": {"i": {"type": "string", "description": "intent"}, "command": {"type": "string"}},
        "required": ["i", "command"], "additionalProperties": False}}}
    payload_first = ('<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value>'
                     '<arg_key>i</arg_key><arg_value>list files</arg_value></tool_call>')
    missing_required = '<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>'
    fails = []
    tag = fst.build_flint_tool_grammar(tools=[omp_bash], tool_choice="required")
    if not accepts(compiler, tag, tok, payload_first):
        fails.append("payload-first omp call REJECTED under the strict grammar (key order still forced)")
    if accepts(compiler, tag, tok, missing_required):
        fails.append("call missing the required `i` key ACCEPTED (strict schema not enforced)")
    os.environ["FLINT_STRICT_TOOLS"] = "0"
    import importlib
    importlib.reload(fst)
    tag_loose = fst.build_flint_tool_grammar(tools=[omp_bash], tool_choice="required")
    if not accepts(compiler, tag_loose, tok, missing_required):
        fails.append("FLINT_STRICT_TOOLS=0 should be schema-free but rejected a call missing `i`")
    for f in fails:
        print("FAIL:", f)
    print("OK: strict grammar accepts payload-first, rejects missing required, loose mode is schema-free"
          if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
