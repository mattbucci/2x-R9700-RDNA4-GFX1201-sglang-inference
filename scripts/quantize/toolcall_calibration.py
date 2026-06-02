"""Synthesised tool-call calibration rows in each model's NATIVE tool format.

WHY: AWQ is activation-aware — it protects weights that the calibration data
exercises. The standard recipes (thinking_text / thinking_vision / code_thinking)
render via `apply_chat_template(msgs)` WITHOUT `tools=`, so the model's tool-call
emission pathway is never activated → int4 quantizes those weights hardest →
dense models emit malformed tool calls (qwen35-27b-AWQ: 0/6 SWE-bench, falls back
to broken JSON `{"x": "y".}` instead of its native `<tool_call><function=...>
<parameter=...>` XML; the same model in FP8 scores 83%).

This module renders (user-request -> assistant tool_call) conversations through
the model's OWN chat template WITH `tools=`, so the calibration text contains the
model's exact native tool-call tokens. Tools mirror the opencode coding-agent set
(read/edit/bash/grep/glob/write/list/todowrite) the SWE-bench harness actually uses,
and arguments deliberately include the tricky structures that broke formatting:
nested paths, code with quotes/newlines/escapes, regexes, multi-field objects.

Used by the recalibration scripts as an extra ~15-20% mix on top of a code-bearing
base recipe (balanced_thinking_text / balanced_thinking_vision). PROOF target:
re-run qwen35-27b SWE-bench smoke after recal; expect malformed-JSON -> valid.
"""
from __future__ import annotations
import json, random

# opencode-style coding-agent tool schemas (OpenAI function format)
TOOL_SCHEMAS = {
    "read": {"type": "function", "function": {"name": "read", "description": "Read a file's contents.",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string", "description": "absolute or repo-relative path"},
            "offset": {"type": "integer", "description": "start line (optional)"},
            "limit": {"type": "integer", "description": "number of lines (optional)"}},
            "required": ["filePath"]}}},
    "edit": {"type": "function", "function": {"name": "edit", "description": "Replace oldString with newString in a file (oldString must be unique).",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"}}, "required": ["filePath", "oldString", "newString"]}}},
    "write": {"type": "function", "function": {"name": "write", "description": "Write content to a file (overwrites).",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
    "bash": {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command"]}}},
    "grep": {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regex.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}},
            "required": ["pattern"]}}},
    "glob": {"type": "function", "function": {"name": "glob", "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    "list": {"type": "function", "function": {"name": "list", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    "todowrite": {"type": "function", "function": {"name": "todowrite", "description": "Record a task list.",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object",
            "properties": {"content": {"type": "string"}, "status": {"type": "string"}}}}}, "required": ["todos"]}}},
}

_PATHS = ["src/utils.py", "seaborn/_statistics.py", "django/db/models/query.py", "requests/models.py",
          "flask/config.py", "xarray/core/dataset.py", "pylint/checkers/base.py", "lib/parser/tokenizer.go",
          "internal/server/handler.py", "app/services/auth.ts", "pkg/cache/lru.rs", "tests/test_core.py",
          "astropy/units/quantity.py", "sympy/core/expr.py", "scikit/cluster/_kmeans.py"]
# code strings chosen to carry the punctuation that breaks naive JSON: quotes, braces, newlines, backslashes
_OLD = ['np.polyfit(x, y, self.order)', 'return {"status": "ok"}', 'if x == None:',
        'data = json.loads(resp.text)', 'logger.info("done")', 'def foo(a, b):\n    return a+b',
        're.compile(r"\\d+\\.\\d+")', 'self._cache = {}', 'raise ValueError("bad")']
_NEW = ['np.polynomial.polynomial.Polynomial.fit(x, y, self.order)', 'return {"status": "ok", "code": 200}',
        'if x is None:', 'data = resp.json()', 'logger.debug("done: %s", result)',
        'def foo(a, b, c=0):\n    return a + b + c', 're.compile(r"\\d+\\.\\d+([eE][+-]?\\d+)?")',
        'self._cache: dict[str, Any] = {}', 'raise ValueError(f"bad value: {x!r}")']
_PATTERNS = [r"def \w+\(", r"TODO|FIXME", r"import\s+numpy", r"class \w+\(.*\):", r"\bassert\b", r"raise \w+Error"]
_CMDS = ["python -m pytest tests/ -x -q", "grep -rn 'deprecated' src/", "python -c \"import app; print(app.__version__)\"",
         "ls -la build/ && cat build/log.txt", "ruff check . --fix", "git diff --stat"]
_REQS = ["Fix the bug in {p} where {desc}.", "The tests in {p} fail — investigate and patch.",
         "Refactor {p}: {desc}.", "There's a regression in {p}; locate and fix it.",
         "Update {p} to {desc}.", "Find where {desc} and correct it."]
_DESCS = ["a None check is missing", "the return type is wrong", "an exception isn't raised",
          "the regex doesn't match floats", "the cache is unbounded", "the API call is deprecated"]


def _call(name, args):
    return {"id": f"call_{random.randint(10**7,10**8)}", "type": "function",
            "function": {"name": name, "arguments": args}}


def _gen_one(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Return (messages, tools) for one synthesised tool-using turn."""
    p = rng.choice(_PATHS); desc = rng.choice(_DESCS)
    user = rng.choice(_REQS).format(p=p, desc=desc)
    # choose a tool + realistic args
    tool = rng.choice(["edit", "read", "bash", "grep", "glob", "write", "list", "todowrite"])
    if tool == "edit":
        args = {"filePath": p, "oldString": rng.choice(_OLD), "newString": rng.choice(_NEW)}
    elif tool == "read":
        args = {"filePath": p, **({"offset": rng.randint(1, 200), "limit": rng.choice([40, 80, 120])} if rng.random() < .5 else {})}
    elif tool == "bash":
        args = {"command": rng.choice(_CMDS), **({"description": "run checks"} if rng.random() < .5 else {})}
    elif tool == "grep":
        args = {"pattern": rng.choice(_PATTERNS), **({"path": p.rsplit("/", 1)[0]} if rng.random() < .6 else {}),
                **({"glob": "**/*.py"} if rng.random() < .4 else {})}
    elif tool == "glob":
        args = {"pattern": rng.choice(["**/*.py", "src/**/*.ts", "**/test_*.py", "*.cfg"])}
    elif tool == "write":
        args = {"filePath": p, "content": rng.choice(_NEW) + "\n"}
    elif tool == "list":
        args = {"path": p.rsplit("/", 1)[0] or "."}
    else:  # todowrite
        args = {"todos": [{"content": f"Inspect {p}", "status": "in_progress"},
                          {"content": f"Patch: {desc}", "status": "pending"}]}
    # offer a varied subset of tools (so the model sees different schemas), always including the used one
    pool = [t for t in TOOL_SCHEMAS if t != tool]
    rng.shuffle(pool)
    tool_names = [tool] + pool[: rng.randint(2, 5)]
    tools = [TOOL_SCHEMAS[t] for t in tool_names]
    preamble = rng.choice(["", "Let me look into this.", "I'll inspect the relevant code first.",
                            "Sure — locating the issue.", "I'll make the change."])
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": preamble, "tool_calls": [_call(tool, args)]}]
    return msgs, tools


def build_toolcall_text_rows(tokenizer, n: int, seed: int = 42, enable_thinking: bool = True) -> list[dict]:
    """Render n synthesised tool-call conversations to native-format text rows [{"text": ...}]."""
    rng = random.Random(seed)
    out, attempts = [], 0
    while len(out) < n and attempts < n * 4:
        attempts += 1
        msgs, tools = _gen_one(rng)
        try:
            text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False,
                                                  add_generation_prompt=False, enable_thinking=enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
        if text and ("<tool_call>" in text or "<function=" in text or "[TOOL_CALLS]" in text):
            out.append({"text": text})
    return out


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/letsrtfm/AI/models/Qwen3.5-27B-BF16"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    rows = build_toolcall_text_rows(tok, 8, seed=7)
    print(f"generated {len(rows)} rows")
    for i, r in enumerate(rows[:3]):
        print(f"\n===== sample {i} (tail) =====\n{r['text'][-700:]}")
    # diversity + format check
    blob = "\n".join(r["text"] for r in rows)
    print("\nnative-format rows:", sum("<function=" in r["text"] for r in rows), "/", len(rows))
    print("distinct tools exercised:", sorted({t for t in TOOL_SCHEMAS if f"<function={t}>" in blob}))
