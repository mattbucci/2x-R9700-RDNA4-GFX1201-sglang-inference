#!/usr/bin/env python3
"""Contract tests for the post-save processor ship gate (scripts/quantize/processor_gate.py)."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "quantize" / "processor_gate.py"

spec = importlib.util.spec_from_file_location("processor_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def _write(directory: str, name: str, payload) -> None:
    with open(os.path.join(directory, name), "w", encoding="utf-8") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)


class ProcessorGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = os.path.join(self._tmp.name, "base")
        self.out = os.path.join(self._tmp.name, "out")
        os.makedirs(self.base)
        os.makedirs(self.out)
        self.logs: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def _log(self, msg):
        self.logs.append(msg)

    def test_passes_when_output_mirrors_base(self):
        for d in (self.base, self.out):
            _write(d, "preprocessor_config.json", {"image_processor_type": "X"})
            _write(d, "video_preprocessor_config.json", {"video_processor_type": "Y"})
        present = gate.require_processor_files(self.out, self.base, log=self._log)
        self.assertEqual(present, ["preprocessor_config.json", "video_preprocessor_config.json"])
        self.assertTrue(any("Processor gate OK" in m for m in self.logs))

    def test_fails_when_base_video_config_is_missing_from_output(self):
        _write(self.base, "preprocessor_config.json", {})
        _write(self.base, "video_preprocessor_config.json", {})
        _write(self.out, "preprocessor_config.json", {})
        with self.assertRaises(gate.ProcessorGateError) as cm:
            gate.require_processor_files(self.out, self.base, log=self._log)
        self.assertIn("video_preprocessor_config.json", str(cm.exception))
        self.assertIn("does not", str(cm.exception))

    def test_fails_when_output_has_no_processor_file(self):
        _write(self.base, "processor_config.json", {})
        with self.assertRaises(gate.ProcessorGateError) as cm:
            gate.require_processor_files(self.out, self.base, log=self._log)
        self.assertIn("no processor config", str(cm.exception))

    def test_fails_on_unparseable_json(self):
        _write(self.base, "preprocessor_config.json", {})
        _write(self.out, "preprocessor_config.json", "{not json")
        with self.assertRaises(gate.ProcessorGateError) as cm:
            gate.require_processor_files(self.out, self.base, log=self._log)
        self.assertIn("not a JSON object", str(cm.exception))

    def test_fails_on_non_object_json(self):
        _write(self.base, "preprocessor_config.json", {})
        _write(self.out, "preprocessor_config.json", [1, 2, 3])
        with self.assertRaises(gate.ProcessorGateError):
            gate.require_processor_files(self.out, self.base, log=self._log)

    def test_fails_when_output_dir_is_absent(self):
        with self.assertRaises(gate.ProcessorGateError) as cm:
            gate.require_processor_files(os.path.join(self._tmp.name, "nope"), None, log=self._log)
        self.assertIn("does not exist", str(cm.exception))

    def test_unresolvable_hub_base_checks_presence_only(self):
        _write(self.out, "processor_config.json", {"processor_class": "Z"})
        present = gate.require_processor_files(
            self.out, "definitely/not-a-cached-repo-xyz", log=self._log
        )
        self.assertEqual(present, ["processor_config.json"])
        self.assertTrue(any("checked presence only" in m for m in self.logs))

    def test_gemma_style_processor_config_only_passes(self):
        # Gemma 4 bases carry processor_config.json and no preprocessor_config.json.
        _write(self.base, "processor_config.json", {})
        _write(self.out, "processor_config.json", {})
        self.assertEqual(
            gate.require_processor_files(self.out, self.base, log=self._log),
            ["processor_config.json"],
        )

    def test_check_report_shape(self):
        _write(self.base, "preprocessor_config.json", {})
        report = gate.check_processor_files(self.out, self.base)
        self.assertEqual(report["expected"], ["preprocessor_config.json"])
        self.assertEqual(report["missing"], ["preprocessor_config.json"])
        self.assertEqual(report["present"], [])
        self.assertEqual(report["broken"], {})
        self.assertEqual(report["base_dir"], self.base)

    def test_gate_entry_exits_65(self):
        _write(self.base, "preprocessor_config.json", {})
        with self.assertRaises(SystemExit) as cm:
            gate.gate_processor_files(self.out, self.base)
        self.assertEqual(cm.exception.code, gate.EX_PROCESSOR_GATE)
        self.assertEqual(gate.EX_PROCESSOR_GATE, 65)

    def test_cli_exit_codes(self):
        _write(self.base, "preprocessor_config.json", {})
        bad = subprocess.run(
            [sys.executable, str(MODULE_PATH), self.out, "--base", self.base],
            capture_output=True, text=True,
        )
        self.assertEqual(bad.returncode, 65, bad.stderr)
        self.assertIn("PROCESSOR GATE FAILED", bad.stderr)
        _write(self.out, "preprocessor_config.json", {})
        good = subprocess.run(
            [sys.executable, str(MODULE_PATH), self.out, "--base", self.base],
            capture_output=True, text=True,
        )
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("Processor gate OK", good.stdout)


if __name__ == "__main__":
    unittest.main()
