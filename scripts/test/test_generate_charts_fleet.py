#!/usr/bin/env python3
"""Contract tests for the fleet charts: model classes, colours, markers and grouping."""

from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "bench" / "generate_charts.py"
HAS_CHART_DEPS = (
    importlib.util.find_spec("matplotlib") is not None
    and importlib.util.find_spec("numpy") is not None
)

charts = None
if HAS_CHART_DEPS:
    spec = importlib.util.spec_from_file_location("generate_charts_fleet", MODULE_PATH)
    assert spec and spec.loader
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)


def _lab(hex_color):
    """sRGB hex -> CIE L*a*b* (D65)."""
    rgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    xyz = (
        (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047,
        0.2126 * r + 0.7152 * g + 0.0722 * b,
        (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883,
    )
    fx, fy, fz = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in xyz]
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta_e(a, b):
    return sum((p - q) ** 2 for p, q in zip(_lab(a), _lab(b))) ** 0.5


def _sweep(points):
    return [{"context": ctx, "input_len": inp, "tok_per_sec": tps} for ctx, inp, tps in points]


def _fleet():
    """Four synthetic models, one per (class, quantization) cell, with real MODELS metadata."""
    shapes = {
        "north-mini": [(128, 22, 70.0), (8192, 7300, 66.0), (221184, 197000, 36.0)],        # moe fp8
        "coder-30b-awq": [(128, 22, 86.0), (8192, 7300, 80.0), (32768, 26000, 74.0)],       # moe 4bit
        "qwen38-27b-fp8": [(128, 24, 60.0), (8192, 7300, 81.0), (221184, 197000, 61.0)],    # dense fp8
        "devstral2-awq": [(128, 22, 52.0), (8192, 7300, 48.0), (221184, 198000, 17.0)],     # dense 4bit
    }
    data = {}
    for key, points in shapes.items():
        data[key] = (charts.MODELS[key], {
            "timestamp": "2026-11-01T00:00:00",
            "context_sweep": _sweep(points),
            "throughput_sweep": [{"concurrency": c, "tok_per_sec": points[0][2] * c * 0.8} for c in (1, 2, 4)],
        })
    return data


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class FleetMetadataTest(unittest.TestCase):
    def test_every_model_declares_class_quant_colour_and_marker(self):
        for key, meta in charts.MODELS.items():
            with self.subTest(model=key):
                self.assertIn(meta["cls"], charts.MODEL_CLASSES)
                self.assertIn(meta["quant"], charts.MODEL_QUANTS)
                self.assertIn(meta["color"], charts.FLEET_PALETTE.values())
                self.assertTrue(meta["marker"])
                self.assertTrue(meta["label"])

    def test_class_and_quant_follow_the_label(self):
        for key, meta in charts.MODELS.items():
            with self.subTest(model=key):
                if "MoE" in meta["label"] or "moe" in meta["label"]:
                    self.assertEqual(meta["cls"], "moe")
                if "Dense" in meta["label"]:
                    self.assertEqual(meta["cls"], "dense")
                self.assertEqual(meta["quant"] == "fp8", "FP8" in meta["label"])

    def test_colour_and_marker_are_unique_inside_a_class(self):
        for cls in charts.MODEL_CLASSES:
            members = [m for m in charts.MODELS.values() if m["cls"] == cls]
            self.assertGreater(len(members), 1)
            self.assertLessEqual(len(members), len(charts.FLEET_PALETTE))
            for field in ("color", "marker"):
                with self.subTest(cls=cls, field=field):
                    self.assertEqual(len({m[field] for m in members}), len(members))

    def test_fp8_models_have_distinct_colours(self):
        fp8 = [m["color"] for m in charts.MODELS.values() if m["quant"] == "fp8"]
        self.assertEqual(len(set(fp8)), len(fp8))

    def test_no_two_models_share_colour_and_marker(self):
        pairs = [(m["color"], m["marker"]) for m in charts.MODELS.values()]
        self.assertEqual(len(set(pairs)), len(pairs))

    def test_palette_colours_are_far_apart(self):
        colours = list(charts.FLEET_PALETTE.values())
        self.assertEqual(len(set(colours)), len(colours))
        closest = min(_delta_e(a, b) for a, b in itertools.combinations(colours, 2))
        # The palette this replaced put related models 9-12 apart (CIE76).
        self.assertGreaterEqual(closest, 30.0)

    def test_palette_reads_on_the_dark_panel(self):
        panel = charts.plt.rcParams["axes.facecolor"]
        for name, colour in charts.FLEET_PALETTE.items():
            with self.subTest(colour=name):
                self.assertGreaterEqual(_delta_e(colour, panel), 50.0)


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class FleetGroupingTest(unittest.TestCase):
    def test_default_grouping_is_by_class(self):
        self.assertEqual(charts.FLEET_DEFAULT_GROUPING, "class")
        self.assertEqual(sorted(charts.FLEET_GROUPINGS), ["class", "none", "quant"])

    def test_class_grouping_splits_moe_and_dense(self):
        groups = charts.fleet_groups(_fleet(), "class")
        self.assertEqual([title for title, _ in groups], ["MoE", "Dense"])
        self.assertEqual([[key for key, _, _ in members] for _, members in groups],
                         [["north-mini", "coder-30b-awq"], ["qwen38-27b-fp8", "devstral2-awq"]])

    def test_quant_grouping_splits_fp8_and_four_bit(self):
        groups = charts.fleet_groups(_fleet(), "quant")
        self.assertEqual([title for title, _ in groups], ["FP8", "4-bit (AWQ / GPTQ)"])
        self.assertEqual([[key for key, _, _ in members] for _, members in groups],
                         [["north-mini", "qwen38-27b-fp8"], ["coder-30b-awq", "devstral2-awq"]])

    def test_no_grouping_is_one_untitled_group(self):
        groups = charts.fleet_groups(_fleet(), "none")
        self.assertEqual(len(groups), 1)
        self.assertIsNone(groups[0][0])
        self.assertEqual(len(groups[0][1]), 4)

    def test_empty_groups_are_dropped(self):
        fleet = {k: v for k, v in _fleet().items() if v[0]["cls"] == "dense"}
        self.assertEqual([title for title, _ in charts.fleet_groups(fleet, "class")], ["Dense"])
        self.assertEqual(charts.fleet_groups({}, "none"), [])

    def test_unknown_grouping_is_rejected(self):
        with self.assertRaises(KeyError):
            charts.fleet_groups(_fleet(), "vendor")

    def test_every_real_model_lands_in_exactly_one_group(self):
        fleet = {key: (meta, {}) for key, meta in charts.MODELS.items()}
        for group_by in charts.FLEET_GROUPINGS:
            with self.subTest(group_by=group_by):
                keys = [key for _, members in charts.fleet_groups(fleet, group_by) for key, _, _ in members]
                self.assertEqual(sorted(keys), sorted(charts.MODELS))


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class FleetLineStyleTest(unittest.TestCase):
    def test_marker_fill_carries_the_class(self):
        panel = charts.plt.rcParams["axes.facecolor"]
        for key, meta in charts.MODELS.items():
            style = charts.fleet_line_style(meta)
            with self.subTest(model=key):
                self.assertEqual(style["color"], meta["color"])
                self.assertEqual(style["marker"], meta["marker"])
                if meta["cls"] == "dense":
                    self.assertEqual(style["markerfacecolor"], panel)
                    self.assertEqual(style["markeredgecolor"], meta["color"])
                else:
                    self.assertEqual(style["markerfacecolor"], meta["color"])

    def test_dash_carries_the_quantization(self):
        for key, meta in charts.MODELS.items():
            with self.subTest(model=key):
                solid = charts.fleet_line_style(meta)["linestyle"] == "-"
                self.assertEqual(solid, meta["quant"] == "fp8")


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class DecodeRowsTest(unittest.TestCase):
    def test_peak_and_deepest_point_with_actual_input_tokens(self):
        rows = {r["key"]: r for r in charts.decode_rows(_fleet())}
        dspark = rows["qwen38-27b-fp8"]
        self.assertEqual((dspark["peak"], dspark["peak_in"]), (81.0, 7300))
        self.assertEqual((dspark["deep"], dspark["deep_in"]), (61.0, 197000))
        self.assertTrue(dspark["reaches"])
        capped = rows["coder-30b-awq"]
        self.assertEqual((capped["deep"], capped["deep_in"]), (74.0, 26000))
        self.assertFalse(capped["reaches"])
        self.assertEqual(dspark["date"], "2026-11-01")

    def test_error_rows_and_single_point_sweeps_are_skipped(self):
        fleet = _fleet()
        fleet["north-mini"][1]["context_sweep"].append(
            {"context": 262144, "input_len": 250000, "tok_per_sec": 0, "error": "oom"})
        fleet["devstral2-awq"][1]["context_sweep"] = _sweep([(128, 22, 52.0)])
        rows = {r["key"]: r for r in charts.decode_rows(fleet)}
        self.assertNotIn("devstral2-awq", rows)
        self.assertEqual(rows["north-mini"]["deep_in"], 197000)


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class FleetRenderTest(unittest.TestCase):
    def _render(self, make, group_by):
        """Render one chart into a temp dir; returns the figure's axes as plt.close saw them."""
        seen = []
        real_close = charts.plt.close

        def close(fig):
            seen.append(fig)
            real_close(fig)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(charts.plt, "close", close):
            path = os.path.join(tmp, "chart.png")
            self.assertEqual(make(_fleet(), group_by, path), path)
            self.assertGreater(os.path.getsize(path), 10_000)
        self.assertEqual(len(seen), 1)
        return seen[0].axes

    def test_line_charts_draw_one_panel_per_group(self):
        for make in (charts.make_combined_context_chart, charts.make_combined_concurrency_chart):
            for group_by, panels in (("class", 2), ("quant", 2), ("none", 1)):
                with self.subTest(chart=make.__name__, group_by=group_by):
                    axes = self._render(make, group_by)
                    self.assertEqual(len(axes), panels)
                    self.assertEqual(sum(len(ax.get_lines()) for ax in axes), 4)

    def test_class_panels_are_titled_and_ordered_by_left_edge_speed(self):
        axes = self._render(charts.make_combined_context_chart, "class")
        self.assertEqual([ax.get_title(loc="left") for ax in axes], ["MoE · 2 models", "Dense · 2 models"])
        legend = [text.get_text() for text in axes[0].get_legend().get_texts()]
        self.assertEqual(legend, [charts.MODELS["coder-30b-awq"]["label"], charts.MODELS["north-mini"]["label"]])
        self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())

    def test_decode_bars_group_models_and_sort_by_peak(self):
        expected = {
            "class": ["coder-30b-awq", "north-mini", "qwen38-27b-fp8", "devstral2-awq"],
            "quant": ["qwen38-27b-fp8", "north-mini", "coder-30b-awq", "devstral2-awq"],
            "none": ["coder-30b-awq", "qwen38-27b-fp8", "north-mini", "devstral2-awq"],
        }
        for group_by, keys in expected.items():
            with self.subTest(group_by=group_by):
                (ax,) = self._render(charts.make_decode_bar_chart, group_by)
                labels = [tick.get_text() for tick in ax.get_yticklabels()]
                self.assertEqual(labels, [charts.MODELS[key]["label"] for key in keys])
                self.assertEqual(len(ax.patches), 8)
                hatched = [bar for bar in ax.patches if bar.get_hatch()]
                self.assertEqual(len(hatched), 1)   # coder-30b stops below the deep-input bar

    def test_charts_without_data_are_skipped(self):
        for make in (charts.make_combined_context_chart, charts.make_decode_bar_chart):
            with self.subTest(chart=make.__name__), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "chart.png")
                self.assertIsNone(make({}, "class", path))
                self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
