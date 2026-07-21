#!/usr/bin/env python3
"""Focused, network-free gates for the R97-C calibration-source port."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "quantize" / "calibration_datasets.py"
SPEC = importlib.util.spec_from_file_location("calibration_datasets_r97c", MODULE_PATH)
assert SPEC and SPEC.loader
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


class RedirectRegistryTest(unittest.TestCase):
    def test_exact_redirects_and_recipe_surface(self):
        thinking = calibration.MIXES["am_thinking"]
        common_voice = calibration.MIXES["common_voice_audio"]
        covost = calibration.MIXES["covost2_audio"]

        self.assertEqual(thinking.hf_name, "glaiveai/reasoning-v1-20m")
        self.assertIs(thinking.format_fn, calibration._glaive_reasoning)
        self.assertEqual(
            (common_voice.hf_name, common_voice.config, common_voice.split),
            ("openslr/librispeech_asr", "clean", "train.100"),
        )
        self.assertEqual(
            (covost.hf_name, covost.config, covost.split),
            ("facebook/voxpopuli", "en", "train"),
        )
        self.assertTrue(common_voice.drop_audio)
        self.assertTrue(covost.drop_audio)

        self.assertEqual(len(calibration.RECIPES), 8)
        self.assertEqual(calibration.RECIPES["code_thinking"]["evol_code"], 0.40)
        self.assertEqual(calibration.RECIPES["code_vision"]["evol_code"], 0.45)
        for recipe in calibration.RECIPES.values():
            self.assertAlmostEqual(sum(recipe.values()), 1.0)

    def test_voxpopuli_formatter_uses_nonempty_transcript_precedence(self):
        messages = calibration._covost2_audio(
            {
                "raw_text": "raw transcript",
                "normalized_text": "normalized transcript",
                "sentence": "legacy transcript",
            }
        )
        self.assertIn("<|audio|>", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "raw transcript")


class AudioLoaderTest(unittest.TestCase):
    def test_audio_is_decode_disabled_and_removed_before_shuffle(self):
        calls = []

        class FakeIterable:
            def cast_column(self, name, feature):
                calls.append(("cast", name, feature.decode))
                return self

            def remove_columns(self, name):
                calls.append(("remove", name))
                return self

            def shuffle(self, *, seed, buffer_size):
                calls.append(("shuffle", seed, buffer_size))
                return self

            def __iter__(self):
                yield {"text": "a transcript"}

        mix = calibration.Mix(
            "audio",
            "example/audio",
            split="train",
            weight=0.0,
            format_fn=calibration._common_voice_audio,
            streaming=True,
            drop_audio=True,
        )
        with mock.patch.object(calibration, "load_dataset", return_value=FakeIterable()):
            rows = calibration._load_slice(mix, n=1, seed=42)

        self.assertEqual(rows, [{"text": "a transcript"}])
        self.assertEqual(calls[:2], [("cast", "audio", False), ("remove", "audio")])
        self.assertEqual(calls[2], ("shuffle", 42, 100))


if __name__ == "__main__":
    unittest.main()
