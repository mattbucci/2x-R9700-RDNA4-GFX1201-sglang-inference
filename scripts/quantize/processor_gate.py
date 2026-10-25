#!/usr/bin/env python3
"""Post-save ship gate for the multimodal processor files.

``tokenizer.save_pretrained()`` never writes ``preprocessor_config.json``, and SGLang's
tokenizer_manager loads the image/video/audio processor from the served directory, so a
quantize script whose ``AutoProcessor.save_pretrained()`` failed used to finish with a WARN
and a checkpoint that boots text-only or not at all.  Every multimodal quantize script now
calls :func:`gate_processor_files` after its processor save: a missing or unparseable file is
a hard failure (exit 65), not a warning.

The expected set is base-relative: whichever of the known processor files the BF16 base
carries must exist in the output and parse as JSON.  A base given as a Hub id is resolved
from the local HF cache without downloading; when it cannot be resolved the gate still
requires at least one processor file in the output.

CLI::

    python scripts/quantize/processor_gate.py /path/to/output [--base /path/to/bf16-or-hub-id]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)
EX_PROCESSOR_GATE = 65  # EX_DATAERR: the output directory is not a servable multimodal checkpoint


class ProcessorGateError(RuntimeError):
    """The output directory is missing or carries a broken processor file."""


def resolve_base_dir(base: str | None) -> str | None:
    """Return a local directory for *base* (a path or a Hub id in the local cache), else None."""
    if not base:
        return None
    if os.path.isdir(base):
        return base
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(base, local_files_only=True)
    except Exception:
        return None


def processor_files_present(directory: str) -> list[str]:
    return [f for f in PROCESSOR_FILES if os.path.isfile(os.path.join(directory, f))]


def check_processor_files(output_dir: str, base: str | None = None) -> dict:
    """Describe the processor files of *output_dir* against *base* without raising.

    Returns ``{"present", "expected", "missing", "broken", "base_dir"}`` where ``broken`` maps
    a present file name to the JSON error it raised.
    """
    base_dir = resolve_base_dir(base)
    expected = processor_files_present(base_dir) if base_dir else []
    present = processor_files_present(output_dir) if os.path.isdir(output_dir) else []
    broken: dict[str, str] = {}
    for name in present:
        try:
            with open(os.path.join(output_dir, name), encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                broken[name] = f"top-level JSON is {type(data).__name__}, not an object"
        except (OSError, ValueError) as exc:
            broken[name] = repr(exc)
    return {
        "present": present,
        "expected": expected,
        "missing": [f for f in expected if f not in present],
        "broken": broken,
        "base_dir": base_dir,
    }


def require_processor_files(output_dir: str, base: str | None = None, *, log=print) -> list[str]:
    """Raise :class:`ProcessorGateError` unless *output_dir* carries every expected processor file.

    Returns the list of processor files present in *output_dir* on success.
    """
    report = check_processor_files(output_dir, base)
    problems = []
    if not os.path.isdir(output_dir):
        problems.append(f"output directory does not exist: {output_dir}")
    elif not report["present"]:
        problems.append(
            f"no processor config in {output_dir} (none of {', '.join(PROCESSOR_FILES)}); "
            "SGLang cannot load the multimodal processor from this checkpoint"
        )
    if report["missing"]:
        problems.append(
            f"base {report['base_dir']} carries {', '.join(report['missing'])} "
            f"but {output_dir} does not"
        )
    for name, err in report["broken"].items():
        problems.append(f"{os.path.join(output_dir, name)} is not a JSON object: {err}")
    if problems:
        raise ProcessorGateError("\n".join(problems))
    if report["base_dir"] is None and base:
        log(f"  Processor gate: base {base!r} not resolvable locally; checked presence only")
    log(f"  Processor gate OK: {', '.join(report['present'])}")
    return report["present"]


def gate_processor_files(output_dir: str, base: str | None = None) -> list[str]:
    """Script entry: run :func:`require_processor_files` and exit 65 on failure."""
    try:
        return require_processor_files(output_dir, base)
    except ProcessorGateError as exc:
        print(f"\nPROCESSOR GATE FAILED (exit {EX_PROCESSOR_GATE}):\n{exc}", file=sys.stderr)
        print(
            "  Re-run AutoProcessor.from_pretrained(<base>).save_pretrained(<output>) or copy the "
            "processor files from the base checkpoint, then re-run this gate.",
            file=sys.stderr,
        )
        sys.exit(EX_PROCESSOR_GATE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", help="quantized checkpoint directory to gate")
    parser.add_argument("--base", default=None, help="BF16 base directory or Hub id (local cache only)")
    args = parser.parse_args(argv)
    try:
        require_processor_files(args.output_dir, args.base)
    except ProcessorGateError as exc:
        print(f"PROCESSOR GATE FAILED:\n{exc}", file=sys.stderr)
        return EX_PROCESSOR_GATE
    return 0


if __name__ == "__main__":
    sys.exit(main())
