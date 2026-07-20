#!/usr/bin/env python3
"""Network-free contract tests for the prompt-profile control A/B generator."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "eval" / "profile_control_ab.py"
SPEC = importlib.util.spec_from_file_location("profile_control_ab_r97d", MODULE_PATH)
assert SPEC and SPEC.loader
ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ab
SPEC.loader.exec_module(ab)

probe = ab.probe


def _http_result(payload, status_code=200):
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _usage(prompt_tokens=64_801, completion_tokens=95, reasoning_tokens=62):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": None,
    }


def _tool_call(name="lookup_record", arguments='{"id":"BANANA42"}', call_id="call_u"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _chat_response(*, finish="tool_calls", tool_calls=None, content="",
                   usage=None, message_extra=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if message_extra:
        message.update(message_extra)
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": usage if usage is not None else _usage(),
    }


def _calibration_response(prompt_tokens):
    return _chat_response(
        finish="length",
        usage=_usage(prompt_tokens=prompt_tokens, completion_tokens=1,
                     reasoning_tokens=0),
    )


class FakeTokenizer:
    """Deterministic monotone chars -> tokens stand-in for a real server.

    ``tokens = ceil(filler_chars / chars_per_token) + overhead`` reproduces the two
    things that make exact-token calibration hard: a profile-dependent ratio, and a
    fixed rendering overhead the caller cannot see.
    """

    def __init__(self, chars_per_token=6.575, overhead=41):
        self.chars_per_token = chars_per_token
        self.overhead = overhead
        self.calls = []

    def tokens_for(self, prompt):
        self.calls.append(len(prompt))
        return int(len(prompt) / self.chars_per_token) + self.overhead

    def post(self, url, json=None, timeout=None):  # noqa: A002 - requests kwarg name
        prompt = json["messages"][0]["content"]
        tokens = self.tokens_for(prompt)
        if json.get("max_tokens") == 1:
            return _http_result(_calibration_response(tokens))
        return _http_result(_chat_response(
            tool_calls=[_tool_call()],
            usage=_usage(prompt_tokens=tokens),
        ))


class ExactTokenCalibrationTest(unittest.TestCase):
    def test_calibration_converges_on_the_exact_target(self):
        fake = FakeTokenizer()
        with mock.patch.object(ab.requests, "post", side_effect=fake.post):
            prompt, tokens, filler_chars, iterations = ab.calibrate_profile(
                "http://unit/v1/chat/completions", 64_801, "repeated",
                timeout=5, verbose=False,
            )

        self.assertEqual(tokens, 64_801)
        self.assertLessEqual(iterations, ab.CALIBRATION_MAX_ITERS)
        self.assertGreater(filler_chars, 0)
        # The returned prompt is the one that measured on target, and it is built by
        # probe.build_prompt, so the needle and task suffix are still in place.
        self.assertIn(probe.NEEDLE, prompt)
        self.assertTrue(prompt.endswith(probe.TASK))
        self.assertEqual(fake.tokens_for(prompt), 64_801)

    def test_heterogeneous_profile_converges_from_its_own_ratio(self):
        fake = FakeTokenizer(chars_per_token=2.8755, overhead=17)
        with mock.patch.object(ab.requests, "post", side_effect=fake.post):
            prompt, tokens, _, iterations = ab.calibrate_profile(
                "http://unit/v1/chat/completions", 115_806, "agentic",
                timeout=5, verbose=False,
            )

        self.assertEqual(tokens, 115_806)
        self.assertLessEqual(iterations, ab.CALIBRATION_MAX_ITERS)
        self.assertTrue(prompt.startswith("[000000] ts="))

    def test_both_profiles_land_on_identical_rendered_tokens(self):
        target = 64_801
        landed = {}
        for profile, cpt, overhead in (
            ("repeated", 6.575, 41),
            ("agentic", 2.8755, 17),
        ):
            fake = FakeTokenizer(chars_per_token=cpt, overhead=overhead)
            with mock.patch.object(ab.requests, "post", side_effect=fake.post):
                prompt, tokens, _, _ = ab.calibrate_profile(
                    "http://unit/v1/chat/completions", target, profile,
                    timeout=5, verbose=False,
                )
            landed[profile] = (tokens, len(prompt))

        self.assertEqual(landed["repeated"][0], target)
        self.assertEqual(landed["agentic"][0], target)
        self.assertEqual(landed["repeated"][0], landed["agentic"][0])
        # Same rendered length, different textures => different character counts.
        # That difference is the point of the control, not a defect.
        self.assertNotEqual(landed["repeated"][1], landed["agentic"][1])

    def test_unreachable_target_fails_loudly_instead_of_near_missing(self):
        # A coarse tokenizer: token count jumps in steps of 5, so most targets are
        # simply not renderable. The search must say so, not return a near miss.
        class Coarse(FakeTokenizer):
            def tokens_for(self, prompt):
                self.calls.append(len(prompt))
                return (int(len(prompt) / self.chars_per_token) // 5) * 5

        fake = Coarse()
        with mock.patch.object(ab.requests, "post", side_effect=fake.post):
            with self.assertRaises(ab.CalibrationError) as caught:
                ab.calibrate_profile(
                    "http://unit/v1/chat/completions", 64_803, "repeated",
                    timeout=5, verbose=False,
                )

        self.assertIn("64803", str(caught.exception))
        self.assertIn("adjacent", str(caught.exception))

    def test_iteration_cap_is_enforced(self):
        fake = FakeTokenizer()
        with mock.patch.object(ab.requests, "post", side_effect=fake.post):
            with self.assertRaises(ab.CalibrationError) as caught:
                ab.calibrate_profile(
                    "http://unit/v1/chat/completions", 64_801, "repeated",
                    timeout=5, max_iters=1, verbose=False,
                )

        self.assertIn("within 1 calibration requests", str(caught.exception))
        self.assertEqual(len(fake.calls), 1)

    def test_missing_usage_is_a_calibration_failure_not_a_silent_zero(self):
        response = _http_result({"choices": [], "usage": {}})
        with mock.patch.object(ab.requests, "post", return_value=response):
            with self.assertRaisesRegex(ab.CalibrationError, "usage.prompt_tokens"):
                ab.calibrate_profile(
                    "http://unit/v1/chat/completions", 1_000, "repeated",
                    timeout=5, verbose=False,
                )

    def test_chars_per_token_inversion_is_exact(self):
        for target_tokens in (64_801, 115_806, 997):
            for filler_chars in (1, 12_345, 426_064):
                with self.subTest(tokens=target_tokens, chars=filler_chars):
                    prompt = ab.build_profile_prompt(
                        target_tokens, filler_chars, "repeated")
                    receipt = probe._prompt_receipt(prompt, "repeated")
                    self.assertEqual(receipt["filler_chars"], filler_chars)

    def test_unknown_profile_is_rejected_before_any_request(self):
        with mock.patch.object(ab.requests, "post") as post:
            with self.assertRaisesRegex(ValueError, "unknown filler profile"):
                ab.calibrate_profile(
                    "http://unit/v1/chat/completions", 1_000, "nonsense", timeout=5)
        post.assert_not_called()


class ScoringTest(unittest.TestCase):
    def _score(self, payload):
        with mock.patch.object(
            ab.requests, "post", return_value=_http_result(payload)
        ):
            return ab.score_one(
                "http://unit/v1/chat/completions", "prompt",
                profile_label="repeated", target_tokens=64_801, seed=0,
                max_tokens=1024, timeout=5,
            )

    def test_correct_banana42_call_scores_and_omits_diagnostics(self):
        row = self._score(_chat_response(tool_calls=[_tool_call()]))

        self.assertTrue(row["valid_toolcall"])
        self.assertTrue(row["correct_action"])
        self.assertEqual(row["got_id"], "BANANA42")
        self.assertEqual(row["tool_name"], "lookup_record")
        self.assertEqual(row["finish_reason"], "tool_calls")
        self.assertEqual(row["http_status"], 200)
        self.assertEqual(row["profile"], "repeated")
        self.assertEqual(row["seed"], 0)
        self.assertEqual(row["target_rendered_tokens"], 64_801)
        self.assertEqual(
            row["usage"],
            {
                "completion_tokens": 95,
                "prompt_tokens": 64_801,
                "prompt_tokens_details": None,
                "reasoning_tokens": 62,
                "total_tokens": 64_896,
            },
        )
        self.assertNotIn("failure_diagnostics", row)
        self.assertNotIn("error", row)

    def test_wrong_id_is_structurally_valid_but_not_a_correct_action(self):
        row = self._score(_chat_response(
            tool_calls=[_tool_call(arguments='{"id":"APPLE01"}')]))

        self.assertTrue(row["valid_toolcall"])
        self.assertFalse(row["correct_action"])
        self.assertEqual(row["got_id"], "APPLE01")
        self.assertIn("failure_diagnostics", row)

    def test_wrong_tool_name_is_rejected(self):
        row = self._score(_chat_response(
            tool_calls=[_tool_call(name="delete_record")]))

        self.assertFalse(row["valid_toolcall"])
        self.assertFalse(row["correct_action"])
        self.assertIsNone(row["got_id"])
        self.assertEqual(row["tool_name"], "delete_record")

    def test_two_calls_are_not_one_exact_call(self):
        row = self._score(_chat_response(tool_calls=[_tool_call(), _tool_call()]))

        self.assertFalse(row["valid_toolcall"])
        self.assertFalse(row["correct_action"])

    def test_prose_answer_with_no_call_is_a_failure_with_bounded_diagnostics(self):
        # This is the shape patch 095 later recovered: a well-formed action block
        # emitted as prose because the parser did not know the `function` key.
        content = ('<|START_ACTION|>[\n    {"function": "lookup_record", '
                   '"parameters": {"id": "BANANA42"}}\n]<|END_ACTION|>')
        row = self._score(_chat_response(
            finish="stop", content=content, tool_calls=None,
            message_extra={"reasoning_content": "Let me call that function.",
                           "tool_calls": None},
        ))

        self.assertFalse(row["valid_toolcall"])
        self.assertFalse(row["correct_action"])
        self.assertIsNone(row["got_id"])
        self.assertIsNone(row["tool_name"])
        self.assertEqual(row["finish_reason"], "stop")
        diagnostics = row["failure_diagnostics"]
        self.assertEqual(diagnostics["content"]["chars"], len(content))
        self.assertEqual(diagnostics["content"]["head"], content)
        self.assertEqual(len(diagnostics["content"]["sha256"]), 64)
        self.assertEqual(diagnostics["tool_calls"]["head"], "null")
        self.assertIn("reasoning_content", diagnostics)

    def test_long_failure_output_is_bounded_head_tail_and_hashed(self):
        reasoning = "The archive contains routine maintenance logs. " * 200
        row = self._score(_chat_response(
            finish="length", content="", tool_calls=None,
            message_extra={"reasoning_content": reasoning, "tool_calls": None},
            usage=_usage(completion_tokens=1024, reasoning_tokens=1024),
        ))

        bounded = row["failure_diagnostics"]["reasoning_content"]
        self.assertEqual(bounded["chars"], len(reasoning))
        self.assertTrue(bounded["truncated"])
        self.assertEqual(len(bounded["head"]), probe.DIAGNOSTIC_EDGE_CHARS)
        self.assertEqual(len(bounded["tail"]), probe.DIAGNOSTIC_EDGE_CHARS)
        self.assertLess(len(json.dumps(bounded)), len(reasoning))

    def test_sampling_fields_are_pinned_on_the_wire(self):
        with mock.patch.object(
            ab.requests, "post", return_value=_http_result(
                _chat_response(tool_calls=[_tool_call()]))
        ) as post:
            ab.score_one(
                "http://unit/v1/chat/completions", "prompt",
                profile_label="repeated", target_tokens=64_801, seed=2,
                max_tokens=1024, timeout=5,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["top_k"], -1)
        self.assertEqual(payload["seed"], 2)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["tools"], probe.TOOLS)
        self.assertEqual(payload["messages"][0]["content"], "prompt")

    def test_transport_failure_yields_a_scored_row_not_a_crash(self):
        with mock.patch.object(
            ab.requests, "post", side_effect=OSError("connection reset")
        ):
            row = ab.score_one(
                "http://unit/v1/chat/completions", "prompt",
                profile_label="repeated", target_tokens=64_801, seed=1,
                max_tokens=1024, timeout=5,
            )

        self.assertFalse(row["valid_toolcall"])
        self.assertFalse(row["correct_action"])
        self.assertIsNone(row["http_status"])
        self.assertIn("connection reset", row["error"])
        self.assertEqual(row["usage"]["prompt_tokens"], None)


class PatchChainTest(unittest.TestCase):
    def test_patch_chain_is_computed_from_the_real_files_090_to_095(self):
        chain = ab.patch_chain()

        self.assertEqual([entry["number"] for entry in chain],
                         [90, 91, 92, 93, 94, 95])
        for entry in chain:
            path = REPO_ROOT / entry["file"]
            self.assertTrue(path.is_file(), entry["file"])
            self.assertTrue(entry["file"].startswith("patches/"))
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                entry["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_095_is_present_and_not_inherited_from_the_old_hardcoded_chain(self):
        chain = ab.patch_chain()
        entry = next(e for e in chain if e["number"] == 95)

        self.assertIn("cohere-command4-function-key-name-recovery", entry["file"])
        old_receipt = json.loads(
            (REPO_ROOT / "benchmarks" / "quality"
             / "north-mini-tooluse-profile-ab-post094-2026-07-19.json").read_text()
        )
        old_numbers = [e["number"] for e in old_receipt["patch_chain"]]
        self.assertNotIn(95, old_numbers)
        self.assertEqual([e["number"] for e in chain][:5], old_numbers)
        # The shared 090-094 files must still hash identically, or this receipt is
        # not comparable with the one it supersedes.
        for new, old in zip(chain, old_receipt["patch_chain"]):
            self.assertEqual(new["file"], old["file"])
            self.assertEqual(new["sha256"], old["sha256"])

    def test_missing_patch_number_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SystemExit) as caught:
                ab.patch_chain(90, 95, patches_dir=pathlib.Path(temp_dir))

        self.assertIn("090-*.patch", str(caught.exception))


class ServerReceiptTest(unittest.TestCase):
    def test_extra_serving_fields_are_captured(self):
        info = {
            "model_path": "/models/north-mini",
            "context_length": 262_144,
            "tp_size": 2,
            "kv_cache_dtype": "auto",
            "attention_backend": "triton",
            "tool_call_parser": "cohere_command4",
            "reasoning_parser": "cohere_command4",
            "enable_deterministic_inference": True,
            "server_args": {
                "dtype": "bfloat16",
                "quantization": "compressed-tensors",
                "sampling_backend": "pytorch",
                "chunked_prefill_size": 4096,
                "disable_overlap_schedule": True,
                "swa_full_tokens_ratio": 0.0625,
            },
        }
        receipt = ab.profile_server_receipt(info)

        self.assertEqual(receipt["dtype"], "bfloat16")
        self.assertEqual(receipt["quantization"], "compressed-tensors")
        self.assertEqual(receipt["sampling_backend"], "pytorch")
        self.assertEqual(receipt["chunked_prefill_size"], 4096)
        self.assertTrue(receipt["disable_overlap_schedule"])
        self.assertEqual(receipt["swa_full_tokens_ratio"], 0.0625)
        self.assertEqual(receipt["tool_call_parser"], "cohere_command4")

    def test_auto_kv_cache_dtype_resolves_to_the_model_dtype(self):
        self.assertEqual(
            ab.resolved_kv_cache_dtype(
                {"kv_cache_dtype": "auto", "dtype": "bfloat16"}),
            "bfloat16",
        )
        self.assertEqual(
            ab.resolved_kv_cache_dtype({"kv_cache_dtype": "fp8_e4m3",
                                        "dtype": "bfloat16"}),
            "fp8_e4m3",
        )


class ReceiptShapeTest(unittest.TestCase):
    OLD_RECEIPT = (REPO_ROOT / "benchmarks" / "quality"
                   / "north-mini-tooluse-profile-ab-post094-2026-07-19.json")

    def _document(self):
        prompts = [
            {"profile": "repeated", "rendered_tokens": 64_801,
             "target_rendered_tokens": 64_801, "user_chars": 426_348,
             "user_sha256": "a" * 64},
            {"profile": "heterogeneous_code_log_exact", "rendered_tokens": 64_801,
             "target_rendered_tokens": 64_801, "user_chars": 186_620,
             "user_sha256": "b" * 64},
        ]
        results = []
        for profile, correct_seeds in (
            ("repeated", {0}),
            ("heterogeneous_code_log_exact", {0, 1, 2}),
        ):
            for seed in (0, 1, 2):
                correct = seed in correct_seeds
                results.append({
                    "profile": profile, "seed": seed,
                    "target_rendered_tokens": 64_801,
                    "valid_toolcall": correct, "correct_action": correct,
                    "got_id": "BANANA42" if correct else None,
                    "tool_name": "lookup_record" if correct else None,
                    "finish_reason": "tool_calls" if correct else "stop",
                    "http_status": 200, "elapsed_s": 1.5,
                    "usage": ab._usage_receipt(_usage()),
                })
        return ab.assemble_receipt(
            prompts=prompts,
            results=results,
            sampling={"temperature": 1.0, "top_p": 0.95, "top_k": -1,
                      "max_tokens": 1024, "seeds": [0, 1, 2],
                      "seed_effective": True},
            server={"model_path": "/models/north-mini", "kv_cache_dtype": "auto",
                    "dtype": "bfloat16"},
            raw_log_sha256="c" * 64,
            captured_at="2026-07-19T15:49:00-07:00",
        )

    def test_top_level_keys_and_order_match_the_superseded_receipt(self):
        old = json.loads(self.OLD_RECEIPT.read_text())
        document = self._document()

        self.assertEqual(list(document), list(old))
        self.assertEqual(document["schema_version"], 1)

    def test_tag_advances_to_the_095_chain(self):
        document = self._document()

        self.assertEqual(
            document["tag"],
            "north-fixes-090-095-bf16kv-deterministic-profile-ab",
        )
        old = json.loads(self.OLD_RECEIPT.read_text())
        self.assertNotEqual(document["tag"], old["tag"])

    def test_generator_block_preserves_the_receipt_profile_labels(self):
        old = json.loads(self.OLD_RECEIPT.read_text())
        document = self._document()

        self.assertEqual(document["generator"], old["generator"])
        self.assertEqual(document["measurement"], old["measurement"])
        self.assertEqual(set(ab.PROFILE_LABELS.values()), set(old["generator"]))
        self.assertEqual(ab.PROFILE_LABELS["agentic"],
                         "heterogeneous_code_log_exact")

    def test_resolved_kv_cache_dtype_is_appended_to_the_server_block(self):
        document = self._document()

        self.assertEqual(document["server"]["resolved_kv_cache_dtype"], "bfloat16")
        self.assertEqual(list(document["server"])[-1], "resolved_kv_cache_dtype")

    def test_provenance_hashes_both_the_raw_log_and_the_extracted_core(self):
        document = self._document()
        provenance = document["provenance"]

        self.assertEqual(provenance["captured_at"], "2026-07-19T15:49:00-07:00")
        self.assertEqual(provenance["raw_log_sha256"], "c" * 64)
        self.assertRegex(provenance["extracted_json_sha256"], r"^[0-9a-f]{64}$")
        # The core digest must actually cover the measurements: change one row and
        # the digest must move.
        mutated = self._document()
        mutated_doc = ab.assemble_receipt(
            prompts=[{"profile": "repeated", "rendered_tokens": 1,
                      "target_rendered_tokens": 1, "user_chars": 2,
                      "user_sha256": "d" * 64}],
            results=[], sampling={}, server={},
            raw_log_sha256="c" * 64,
            captured_at="2026-07-19T15:49:00-07:00",
        )
        self.assertNotEqual(
            mutated["provenance"]["extracted_json_sha256"],
            mutated_doc["provenance"]["extracted_json_sha256"],
        )

    def test_summary_rates_match_the_old_receipts_integral_encoding(self):
        document = self._document()
        summary = {(e["profile"], e["rendered_tokens"]): e
                   for e in document["summary"]}

        heterogeneous = summary[("heterogeneous_code_log_exact", 64_801)]
        repeated = summary[("repeated", 64_801)]
        self.assertEqual(heterogeneous["correct"], 3)
        self.assertEqual(heterogeneous["samples"], 3)
        self.assertEqual(heterogeneous["correct_rate"], 1)
        self.assertIsInstance(heterogeneous["correct_rate"], int)
        self.assertEqual(repeated["correct"], 1)
        self.assertAlmostEqual(repeated["correct_rate"], 1 / 3)
        self.assertIn('"correct_rate": 1\n', json.dumps(document, indent=2))

    def test_summary_is_sorted_by_profile_then_depth(self):
        prompts = [
            {"profile": "repeated", "rendered_tokens": 115_806,
             "target_rendered_tokens": 115_806, "user_chars": 1, "user_sha256": "a"},
            {"profile": "repeated", "rendered_tokens": 64_801,
             "target_rendered_tokens": 64_801, "user_chars": 1, "user_sha256": "a"},
            {"profile": "heterogeneous_code_log_exact", "rendered_tokens": 115_806,
             "target_rendered_tokens": 115_806, "user_chars": 1, "user_sha256": "a"},
            {"profile": "heterogeneous_code_log_exact", "rendered_tokens": 64_801,
             "target_rendered_tokens": 64_801, "user_chars": 1, "user_sha256": "a"},
        ]
        summary = ab.build_summary(prompts, [])

        self.assertEqual(
            [(e["profile"], e["rendered_tokens"]) for e in summary],
            [("heterogeneous_code_log_exact", 64_801),
             ("heterogeneous_code_log_exact", 115_806),
             ("repeated", 64_801),
             ("repeated", 115_806)],
        )


class MainMatrixTest(unittest.TestCase):
    def _server_info(self, deterministic=True):
        return {
            "model_path": "/models/north-mini",
            "context_length": 262_144,
            "tp_size": 2,
            "kv_cache_dtype": "auto",
            "attention_backend": "triton",
            "tool_call_parser": "cohere_command4",
            "reasoning_parser": "cohere_command4",
            "enable_deterministic_inference": deterministic,
            "server_args": {"dtype": "bfloat16"},
        }

    def _run_main(self, argv_extra, deterministic=True, out_path=None):
        # Stub prompts stand in for the calibrated bodies. Their char counts differ
        # per profile exactly as the real ones do (426348 vs 186620 at 64801), which
        # is what the length-vs-texture assertions key on.
        calibrated = {
            ("repeated", 64_801): ("stub-repeated-64801", 64_801, 426_064, 3),
            ("agentic", 64_801): ("stub-heterogeneous-64801", 64_801, 186_336, 4),
            ("repeated", 115_806): ("stub-repeated-115806", 115_806, 762_698, 3),
            ("agentic", 115_806): ("stub-heterogeneous-115806", 115_806, 333_587, 5),
        }
        user_chars = {
            "stub-repeated-64801": 426_348,
            "stub-heterogeneous-64801": 186_620,
            "stub-repeated-115806": 762_982,
            "stub-heterogeneous-115806": 333_871,
        }

        def fake_calibrate(url, target, profile, **kwargs):
            return calibrated[(profile, target)]

        def fake_prompt_receipt(prompt, profile):
            return {"prompt_chars": user_chars[prompt],
                    "prompt_sha256": "e" * 64}

        argv = ["profile_control_ab.py"] + argv_extra
        if out_path is not None:
            argv += ["--out", str(out_path)]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            ab.probe, "read_server_info",
            return_value=self._server_info(deterministic)
        ), mock.patch.object(
            ab, "calibrate_profile", side_effect=fake_calibrate
        ), mock.patch.object(
            ab.probe, "_prompt_receipt", side_effect=fake_prompt_receipt
        ), mock.patch.object(
            ab.requests, "post",
            return_value=_http_result(_chat_response(tool_calls=[_tool_call()]))
        ) as post, contextlib.redirect_stdout(output):
            ab.main()
        return output.getvalue(), post

    def test_two_depths_two_profiles_three_seeds_is_twelve_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = pathlib.Path(temp_dir) / "receipt.json"
            _, post = self._run_main([], out_path=out_path)
            document = json.loads(out_path.read_text())
            raw_log = out_path.with_suffix(".json.raw.log")
            raw_text = raw_log.read_text()

        self.assertEqual(len(document["results"]), 12)
        self.assertEqual(len(document["prompts"]), 4)
        self.assertEqual(post.call_count, 12)
        self.assertEqual(
            sorted({(r["profile"], r["target_rendered_tokens"], r["seed"])
                    for r in document["results"]}),
            sorted([(p, d, s)
                    for p in ("heterogeneous_code_log_exact", "repeated")
                    for d in (64_801, 115_806)
                    for s in (0, 1, 2)]),
        )
        self.assertEqual(document["sampling"], {
            "temperature": 1.0, "top_p": 0.95, "top_k": -1,
            "max_tokens": 1024, "seeds": [0, 1, 2], "seed_effective": True,
        })
        self.assertEqual(len(document["patch_chain"]), 6)
        self.assertEqual(len(document["summary"]), 4)
        self.assertTrue(all(e["samples"] == 3 for e in document["summary"]))
        # The raw log the provenance digest attests really covers every scored call.
        self.assertEqual(raw_text.count('"kind": "scored_request"'), 12)
        self.assertEqual(
            document["provenance"]["raw_log_sha256"],
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        )

    def test_both_profiles_at_each_depth_share_one_rendered_token_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = pathlib.Path(temp_dir) / "receipt.json"
            self._run_main([], out_path=out_path)
            document = json.loads(out_path.read_text())

        by_depth = {}
        for entry in document["prompts"]:
            by_depth.setdefault(entry["target_rendered_tokens"], set()).add(
                entry["rendered_tokens"])
        self.assertEqual(by_depth, {64_801: {64_801}, 115_806: {115_806}})
        for entry in document["prompts"]:
            self.assertEqual(entry["rendered_tokens"],
                             entry["target_rendered_tokens"])
        # Same token count, different character counts: texture varies, length does not.
        chars = {e["profile"]: e["user_chars"] for e in document["prompts"]
                 if e["target_rendered_tokens"] == 64_801}
        self.assertNotEqual(chars["repeated"],
                            chars["heterogeneous_code_log_exact"])

    def test_length_mismatch_between_profiles_aborts_the_run(self):
        def fake_calibrate(url, target, profile, **kwargs):
            tokens = target if profile == "repeated" else target - 1
            return (profile, tokens, 10, 1)

        argv = ["profile_control_ab.py", "--depths", "64801"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            ab.probe, "read_server_info", return_value=self._server_info()
        ), mock.patch.object(
            ab, "calibrate_profile", side_effect=fake_calibrate
        ), mock.patch.object(
            ab.probe, "_prompt_receipt",
            return_value={"prompt_chars": 1, "prompt_sha256": "f" * 64}
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                ab.main()

        self.assertIn("confounded by length", str(caught.exception))

    def test_nondeterministic_server_warns_loudly_and_records_seed_effective_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = pathlib.Path(temp_dir) / "receipt.json"
            output, _ = self._run_main(
                ["--depths", "64801"], deterministic=False, out_path=out_path)
            document = json.loads(out_path.read_text())

        self.assertIn("WARN", output)
        self.assertIn("enable_deterministic_inference", output)
        self.assertIn("SILENTLY IGNORED", output)
        self.assertFalse(document["sampling"]["seed_effective"])
        self.assertEqual(document["server"]["enable_deterministic_inference"], False)

    def test_deterministic_server_does_not_warn(self):
        output, _ = self._run_main(["--depths", "64801"], deterministic=True)

        self.assertNotIn("SILENTLY IGNORED", output)

    def test_cli_overrides_reshape_the_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = pathlib.Path(temp_dir) / "receipt.json"
            _, post = self._run_main(
                ["--depths", "64801", "--seeds", "7", "--max-tokens", "256"],
                out_path=out_path)
            document = json.loads(out_path.read_text())

        self.assertEqual(len(document["results"]), 2)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(document["sampling"]["seeds"], [7])
        self.assertEqual(document["sampling"]["max_tokens"], 256)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 256)
        self.assertEqual(post.call_args.kwargs["json"]["seed"], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
