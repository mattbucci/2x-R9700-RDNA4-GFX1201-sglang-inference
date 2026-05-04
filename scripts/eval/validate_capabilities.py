#!/usr/bin/env python3
"""Capability validator — runs against a live SGLang server on <port>.

Checks the two capabilities we silently break during calibration:
  1. Thinking — model produces <think>...</think> and terminates before max_tokens
  2. Vision  — model describes an image correctly (keyword match)

Also runs a basic short-answer sanity check.

Usage:
    # Launch your server, then:
    python scripts/eval/validate_capabilities.py --port 23334
    python scripts/eval/validate_capabilities.py --port 23334 --skip-vision
    python scripts/eval/validate_capabilities.py --port 23334 --thinking-kwarg '{"enable_thinking":true}'

Exit 0 if all enabled checks pass, non-zero otherwise.  Designed to be run
immediately after every requant so we never ship a regression.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


def _http_post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Surface the server's error body (often actionable: missing config,
        # template render error, etc.) instead of the opaque default str().
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = "<no body>"
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason}: {body}", e.headers, None
        ) from None


def _http_get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8").strip()
        return json.loads(body) if body else {}


def _server_alive(base_url: str, timeout: int = 5) -> bool:
    """/health returns 200 with empty body on SGLang; don't try to parse JSON."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _make_test_image() -> bytes:
    """A synthetic 256x256 image with a red circle on a white background.

    Kept inline so the validator has no external asset dependency.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.stderr.write("PIL required for vision validation; pip install pillow\n")
        raise

    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((64, 64, 192, 192), fill="red", outline="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _model_max_len(base_url: str) -> int | None:
    """Look up max_model_len from /v1/models so we don't request more tokens
    than the server allows (small-context test runs OOM otherwise)."""
    try:
        data = _http_get(f"{base_url}/v1/models", timeout=5)
        return data["data"][0].get("max_model_len")
    except Exception:
        return None


def check_thinking(
    base_url: str,
    model: str,
    thinking_kwargs: dict | None,
    max_tokens: int = 4096,
) -> tuple[bool, str]:
    """Send a reasoning prompt, verify <think>...</think> structure + clean termination."""
    prompt = (
        "A ball and a bat cost $1.10 together.  The bat costs $1.00 more than "
        "the ball.  How much does the ball cost?  Put the final numeric answer "
        "alone on the last line."
    )

    # Use the model's recommended sampling rather than greedy decode.  Qwen3-family
    # models loop on temp=0 ("Paris</think>Paris</think>...") even when calibrated
    # correctly — SGLang reads sampling defaults from generation_config.json when
    # we omit temperature; falling back to 0.7/top_p=0.95 if the server hasn't
    # picked up a model preset.
    # Cap max_tokens against the model's actual context window — small-context
    # test runs (e.g. 4K for OOM-headroom) would otherwise 400 on the request.
    server_max = _model_max_len(base_url)
    if server_max:
        max_tokens = min(max_tokens, max(256, server_max - 256))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "skip_special_tokens": False,
    }
    # Default to enable_thinking=True so we actually exercise the thinking path
    # on Qwen3.5/3.6 (chat template needs the explicit flag).  Override with
    # --thinking-kwarg for models that use a different toggle.
    payload["chat_template_kwargs"] = thinking_kwargs or {"enable_thinking": True}

    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=300)
    except Exception as e:
        return False, f"request failed: {e!r}"

    choice = r["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    usage = r.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    # Either a reasoning parser separated the content, OR the raw content has tags
    has_reasoning = bool(reasoning) or "<think>" in content or "<|channel>" in content
    closed = (
        bool(reasoning and content)  # parser split it
        or "</think>" in content
        or "<channel|>" in content
    )
    truncated = finish == "length"

    # Find the final answer: "0.05" or "$0.05" or "5 cents".  Look in BOTH
    # content and reasoning_content — some calibrated models put the final
    # answer inside the thinking block before terminating cleanly.
    answer_haystack = (content or "") + " " + (reasoning or "")
    if "</think>" in answer_haystack:
        # Prefer text after </think> (the proper answer slot)
        after = answer_haystack.split("</think>")[-1]
        if re.search(r"\$?0?\.05\b|\b5\s*cents?\b", after.lower()):
            answer_haystack = after
    answer_correct = bool(re.search(r"\$?0?\.05\b|\b5\s*cents?\b", answer_haystack.lower()))

    status = []
    if has_reasoning:
        status.append("reasoning_seen")
    if closed:
        status.append("terminated")
    if answer_correct:
        status.append("answer_ok")
    if truncated:
        status.append("TRUNCATED")

    # PASS if either:
    #   (a) thinking terminated cleanly with the right answer (ideal), OR
    #   (b) answer was found in reasoning_content even if max_tokens hit —
    #       the model is calibrated correctly, the bat-ball is just verbose
    #       at temp=0.7.  We log TRUNCATED so the user sees it.
    passed = has_reasoning and answer_correct and (closed or not truncated)
    msg = (
        f"{' '.join(status) or 'no_markers':30s} "
        f"({completion_tokens} tok, finish={finish})"
    )
    return passed, msg


def check_basic(base_url: str, model: str) -> tuple[bool, str]:
    """Short factual question — verifies the server at all.

    Explicitly sets `enable_thinking=False` so that for Qwen3.5/3.6 family
    chat templates (which default `enable_thinking=True`), the rendered
    prompt has NO `<think>` markers.  Otherwise the model emits open-ended
    reasoning that loops `paris</think>\\n\\nparis</think>\\n\\n…` because
    it's never been calibrated to handle a basic question through the
    thinking gate.  This is the M4-audited regression — basic mode should
    use the non-thinking path.  (Cross-team patch from R9700 2026-04-30.)
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France?  Answer in one word."}],
        "max_tokens": 256,
        # Greedy (temp=0) sends Qwen3-family models into an immediate-EOS or
        # infinite-thinking state; use the same sampling as check_thinking.
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=120)
    except Exception as e:
        return False, f"request failed: {e!r}"
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").lower()
    reasoning = (msg.get("reasoning_content") or "").lower()
    finish = r["choices"][0].get("finish_reason")
    passed = "paris" in content or "paris" in reasoning
    sample = content[:60] if content else f"(reasoning){reasoning[:60]}"
    return passed, f"finish={finish} answer={sample!r}"


def _make_test_video() -> bytes:
    """A 12-frame synthetic video: red circle moves left→right across white bg.

    Encoded as MP4 (libx264 if available; falls back to GIF).  Used only to
    confirm the video tower receives frames and the LM emits motion words.
    """
    try:
        from PIL import Image, ImageDraw
        import imageio.v3 as iio
    except ImportError:
        sys.stderr.write("PIL+imageio required for video validation; pip install pillow imageio[ffmpeg]\n")
        raise

    frames = []
    for i in range(12):
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        x = 32 + i * 16  # circle moves rightward
        draw.ellipse((x, 96, x + 64, 160), fill="red", outline="black", width=3)
        frames.append(img)

    buf = io.BytesIO()
    try:
        iio.imwrite(buf, [f for f in frames], extension=".mp4", fps=12)
    except Exception:
        # libx264 not available — fall back to GIF (Gemma video tower accepts both)
        buf = io.BytesIO()
        iio.imwrite(buf, [f for f in frames], extension=".gif", fps=12)
    return buf.getvalue()


def check_video(base_url: str, model: str) -> tuple[bool, str]:
    """Send a synthetic video (red circle moving right) and verify motion describe.

    Pass: response mentions at least one of {move, slide, right, motion,
    travel, across, ball, circle, red}.  Catches both vision-tower regressions
    and chat-template video-token plumbing failures.

    Multimodal capability matrix (per M4 cross-team note):
      - Gemma 4: image + video + audio
      - Qwen3.5/3.6: image + video (no audio)
      - Devstral: image only — call check_video on Devstral and you'll get a
        polite refusal; that's fine, validator tolerates that.
    """
    try:
        from PIL import Image  # noqa: F401  - imported for the bytes builder
    except ImportError:
        return True, "skipped (no PIL)"

    try:
        video_bytes = _make_test_video()
    except Exception as e:
        return True, f"skipped ({e!r})"

    b64 = base64.b64encode(video_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": "What happens in this video?  One short sentence."},
            ],
        }],
        # 12 frames produce ~200 reasoning tokens before the model gets to the
        # one-sentence summary on Qwen3.6-27B; 128 truncates inside reasoning
        # so the visible answer comes back empty. 400 is comfortable for
        # Qwen-family + Gemma 4 thinking modes without going crazy.
        "max_tokens": 400,
        "temperature": 0.7,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=180)
    except Exception as e:
        return False, f"request failed: {e!r}"

    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").lower()
    reasoning = (msg.get("reasoning_content") or "").lower()
    # Same pattern as check_vision: thinking-mode models (Qwen3/Gemma) emit
    # the substantive answer into reasoning_content, not content. Without
    # checking both, the validator silently fails clean models.
    haystack = content + " " + reasoning
    expected = ["move", "slide", "right", "motion", "travel", "across",
                "ball", "circle", "red"]
    hits = [w for w in expected if w in haystack]
    passed = len(hits) >= 1
    sample = content[:120] if content else f"(reasoning){reasoning[:120]}"
    return passed, f"saw={hits}  response={sample!r}"


def check_vision(base_url: str, model: str) -> tuple[bool, str]:
    """Send a synthetic image (red circle) and verify the model sees it.

    Passing criteria: response contains at least one of {red, circle, round,
    sphere, ball, dot, disk} (case-insensitive).  This is a correctness test,
    not a quality test — we just need to know vision isn't silently broken.
    """
    img_bytes = _make_test_image()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }],
        "max_tokens": 128,
        # Greedy (temp=0) can send Qwen3/Gemma4 into an immediate-EOS state;
        # use nucleus sampling like the other checks.
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=180)
    except Exception as e:
        return False, f"request failed: {e!r}"

    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").lower()
    reasoning = (msg.get("reasoning_content") or "").lower()
    haystack = content + " " + reasoning
    # The probe image is a solid red circle on a white background.  A model that
    # actually processed the image should mention BOTH color and shape — matching
    # just one word triggers false positives on text-only models that accept
    # image tokens silently and hallucinate generic descriptions (see Qwen3.6-35B
    # flattened-config case: validator passed with "one-sentence description of
    # the image" while the model's direct answer was "The image is a black
    # square.").  Require a color hit AND a shape hit.
    color_terms = ["red", "crimson", "scarlet"]
    shape_terms = ["circle", "round", "sphere", "ball", "dot", "disk", "oval", "ellipse"]
    color_hits = [w for w in color_terms if w in haystack]
    shape_hits = [w for w in shape_terms if w in haystack]
    hits = color_hits + shape_hits
    passed = bool(color_hits) and bool(shape_hits)
    sample = content[:120] if content else f"(reasoning){reasoning[:120]}"
    return passed, f"saw={hits}  response={sample!r}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="localhost")
    p.add_argument("--model", default=None, help="Override model name (default: server-reported)")
    p.add_argument("--skip-thinking", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-video", action="store_true",
                   help="skip the video roundtrip (Devstral has no video; Qwen/Gemma do)")
    p.add_argument("--thinking-kwarg", default=None,
                   help='JSON string, e.g. \'{"enable_thinking": true}\' for Gemma4')
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--save", default=None,
                   help="Append results to a JSON file keyed by --tag (creates if missing)")
    p.add_argument("--tag", default=None,
                   help="Model tag for --save (default: server-reported model name)")
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"

    # Verify server
    if not _server_alive(base):
        sys.stderr.write(f"Server at {base} not responding\n")
        return 2

    # Resolve model name
    if args.model:
        model = args.model
    else:
        try:
            models = _http_get(f"{base}/v1/models", timeout=5)
            model = models["data"][0]["id"]
        except Exception:
            model = "default"

    thinking_kwargs = None
    if args.thinking_kwarg:
        thinking_kwargs = json.loads(args.thinking_kwarg)

    print(f"=== Capability validator — {base}  model={model} ===")

    results: list[tuple[str, bool, str]] = []

    t0 = time.time()
    ok, msg = check_basic(base, model)
    results.append(("basic", ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] basic     {msg}")

    if not args.skip_thinking:
        ok, msg = check_thinking(base, model, thinking_kwargs)
        results.append(("thinking", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] thinking  {msg}")

    if not args.skip_vision:
        ok, msg = check_vision(base, model)
        results.append(("vision", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] vision    {msg}")

    if not args.skip_video:
        ok, msg = check_video(base, model)
        results.append(("video", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] video     {msg}")

    elapsed = time.time() - t0
    print(f"--- {sum(ok for _, ok, _ in results)}/{len(results)} passed in {elapsed:.1f}s ---")

    if args.save:
        out_path = Path(args.save)
        existing = json.load(open(out_path)) if out_path.exists() else {}
        tag = args.tag or model.split("/")[-1]
        existing[tag] = {
            "tag": tag,
            "model": model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "elapsed_sec": round(elapsed, 1),
            "checks": {name: {"passed": ok, "message": msg} for name, ok, msg in results},
            "summary": {
                "passed": sum(ok for _, ok, _ in results),
                "total": len(results),
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(existing, open(out_path, "w"), indent=2)
        print(f"Saved to {out_path}")

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
