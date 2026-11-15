#!/usr/bin/env python3
"""Every launch.sh preset has a fleet_validate.sh row or a stated omission."""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "scripts" / "launch.sh"
FLEET = ROOT / "scripts" / "eval" / "fleet_validate.sh"


def launch_presets():
    """First name of each label in the preset case block (8-space indent)."""
    lines = LAUNCH.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l == '    case "$1" in')
    end = next(i for i in range(start, len(lines)) if lines[i] == "    esac")
    labels = (re.match(r" {8}([a-z0-9.|_-]+)\)$", l) for l in lines[start:end])
    return [m.group(1).split("|")[0] for m in labels if m]


def fleet_rows():
    text = FLEET.read_text()
    block = text.split("MODELS=(", 1)[1].split("\n)\n", 1)[0]
    return [l.strip().strip('"').split("|") for l in block.splitlines() if l.strip()]


def fleet_omitted():
    line = next(l for l in FLEET.read_text().splitlines() if l.startswith("# Omitted, checkpoint absent:"))
    names = line.split(":", 1)[1].rstrip(".").split(",")
    return [n.split("(")[0].strip() for n in names]


class FleetListTest(unittest.TestCase):
    def test_presets_parsed(self):
        presets = launch_presets()
        self.assertGreaterEqual(len(presets), 20)
        self.assertIn("flint", presets)

    def test_every_preset_listed_once(self):
        listed = [r[0] for r in fleet_rows()]
        omitted = fleet_omitted()
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(sorted(listed + omitted), sorted(launch_presets()))

    def test_rows_well_formed(self):
        for row in fleet_rows():
            self.assertEqual(len(row), 5, row)
            self.assertTrue(row[2].isdigit(), row)
            self.assertIn(row[4], ("0", "1"), row)


if __name__ == "__main__":
    unittest.main()
