"""Offline checks for the chat-format HumanEval path in scripts/eval/eval_and_chart.py."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eac", ROOT / "scripts/eval/eval_and_chart.py")
eac = importlib.util.module_from_spec(spec)
sys.modules["eac"] = eac
spec.loader.exec_module(eac)

PROMPT = (
    "from typing import List\n\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Check if any two numbers are closer than threshold.\n    """\n'
)
TEST = (
    "def check(candidate):\n"
    "    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n"
    "    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True\n"
)
BODY = (
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)


def run(program):
    g = {}
    exec(program + "\n" + TEST, g)
    g["check"](g["has_close_elements"])


def test_full_function_in_fence():
    content = "Here you go:\n```python\n" + PROMPT + BODY + "```\nDone."
    code = eac.extract_code(content)
    run(eac.assemble_program(PROMPT, code, "has_close_elements"))


def test_full_function_without_imports_keeps_prompt_header():
    content = "```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n" + BODY + "```"
    code = eac.extract_code(content)
    program = eac.assemble_program(PROMPT, code, "has_close_elements")
    assert program.startswith("from typing import List")
    run(program)


def test_body_only_is_appended_to_prompt():
    content = "```python\n" + BODY + "```"
    code = eac.extract_code(content)
    run(eac.assemble_program(PROMPT, code, "has_close_elements"))


def test_think_block_and_prose_are_dropped():
    content = "<think>plan</think>Sure.\n```py\n" + PROMPT + BODY + "```"
    code = eac.extract_code(content)
    assert "<think>" not in code and "Sure" not in code
    run(eac.assemble_program(PROMPT, code, "has_close_elements"))


def test_longest_fence_wins():
    content = "```python\nprint('hi')\n```\n```python\n" + PROMPT + BODY + "```"
    code = eac.extract_code(content)
    assert "has_close_elements" in code


def test_no_fence_falls_back_to_whole_text():
    code = eac.extract_code(PROMPT + BODY)
    run(eac.assemble_program(PROMPT, code, "has_close_elements"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
