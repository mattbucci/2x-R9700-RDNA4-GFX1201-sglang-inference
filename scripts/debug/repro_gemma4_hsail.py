#!/usr/bin/env python
"""Clean reproduction of the gemma4 26B HSAIL 0x1016 sampler crash on R9700.

The crash signature (sglang v0.5.11 + patches 023+024):
    sampler.py:498  torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    sampler.py:235  top_k_top_p_min_p_sampling_from_probs_torch(...)
    sampler.py:167  self._sample_from_probs(probs, ...)
then Gloo connection-closed cascade from the worker subprocess.

HSAIL 0x1016 is an async GPU exception — the *real* failing kernel is upstream
of the line shown. Most likely culprits:
  - NaN/Inf in logits (LM forward output) → softmax-of-Inf produces NaN →
    torch.multinomial returns invalid index → torch.gather OOB → HSAIL
  - Or a Triton MoE/AWQ kernel issue on RDNA4 that the 3090 backend doesn't trip

3090 team confirms the same model + same patches work on Ampere, so this is
RDNA4-kernel-specific, not a model packing bug.

Usage:
    # Reproduce the crash (auto-launches its own sglang server):
    python scripts/debug/repro_gemma4_hsail.py

    # With aggressive sampler tracing enabled (set GEMMA4_DEBUG=1):
    GEMMA4_DEBUG=1 python scripts/debug/repro_gemma4_hsail.py

    # Point at an already-running server:
    python scripts/debug/repro_gemma4_hsail.py --port 23334 --no-launch
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("MODELS_DIR", Path.home() / "AI/models"))
MODEL_PATH = MODELS / "gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed"
TOKENIZER_PATH = MODELS / "gemma-4-26B-A4B-it-BF16"


def wait_port_open(port: int, host: str = "127.0.0.1", timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass
        time.sleep(1.0)
    return False


def launch_server(port: int, log_path: Path) -> subprocess.Popen:
    """Spawn sglang.launch_server as a subprocess matching scripts/launch.sh gemma4
    plus debug-friendly flags."""
    env = os.environ.copy()
    # Required RDNA4 env from scripts/common.sh
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    env.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
    env.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    env.setdefault("TRITON_HIP_USE_NEW_STREAM_PIPELINER", "1")
    # Surface every NaN/Inf check in our model code
    env.setdefault("GEMMA4_DEBUG", os.environ.get("GEMMA4_DEBUG", "1"))
    # Make HSAIL crashes synchronous so the stack trace points at the actual
    # offending kernel rather than the next sync barrier.
    env.setdefault("HIP_LAUNCH_BLOCKING", "1")
    # crash_on_warnings() reads SGLANG_IS_IN_CI — turn it on so the
    # `Detected NaN/Inf in logits` warning escalates to ValueError before
    # the HSAIL clobbers the worker.
    env.setdefault("SGLANG_IS_IN_CI", "1")
    # ROCm: don't bury async errors
    env.setdefault("AMD_LOG_LEVEL", env.get("AMD_LOG_LEVEL", "0"))

    cmd = [
        sys.executable, "-u", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_PATH),
        "--tokenizer-path", str(TOKENIZER_PATH),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--tp", "2",
        "--mem-fraction-static", "0.85",
        "--context-length", "4096",
        "--max-running-requests", "1",
        "--chunked-prefill-size", "2048",
        "--attention-backend", "torch_native",
        "--reasoning-parser", "gemma4",
        "--enable-multimodal",
        "--skip-server-warmup",
        "--enable-nan-detection",
        # Note: SGLANG_IS_IN_CI=1 (set in env above) makes crash_on_warnings()
        # return True, which escalates NaN/Inf detection to ValueError.
        "--watchdog-timeout", "1800",
        "--log-level", "info",
    ]
    # Allow overriding --quantization to test different MoE backends
    quant = os.environ.get("REPRO_QUANT")
    if quant:
        cmd.extend(["--quantization", quant])
    print(f"[repro] launching: {' '.join(cmd)}", flush=True)
    print(f"[repro] log: {log_path}", flush=True)
    log = open(log_path, "wb")
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        cwd=str(REPO),
    )


def send_prompt(port: int, greedy: bool = False) -> tuple[int, str]:
    """Send a tiny text-only chat request and return (status, body).

    If greedy=True, uses temperature=0 (argmax path, no multinomial), which
    isolates whether the crash is in the sampler or in the model forward.
    """
    payload = {
        "model": "gemma-4-26B",
        "messages": [
            {"role": "user", "content": "Say the single word 'hello' and stop."}
        ],
        "max_tokens": 8,
    }
    if greedy:
        payload["temperature"] = 0.0
    else:
        payload.update(temperature=1.0, top_p=0.95, top_k=40)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"").decode(errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--no-launch", action="store_true",
                   help="Skip launching the server (use an already-running one)")
    p.add_argument("--keep-running", action="store_true",
                   help="Don't kill the server after the test (for follow-up debugging)")
    args = p.parse_args()

    log_dir = Path("/tmp/gemma4-hsail-repro-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"

    proc = None
    if not args.no_launch:
        if not MODEL_PATH.exists():
            print(f"[repro] ERROR: model path missing: {MODEL_PATH}", file=sys.stderr)
            return 2
        proc = launch_server(args.port, log_path)
        print(f"[repro] server pid={proc.pid}", flush=True)
        print(f"[repro] waiting for port {args.port}...", flush=True)
        if not wait_port_open(args.port, timeout=420):
            print("[repro] FAIL: server never opened port", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            _tail(log_path, 100)
            return 3
        # Give the scheduler a moment past `port open` to finish init
        time.sleep(3.0)

    print(f"[repro] sending prompt to :{args.port}/v1/chat/completions ...", flush=True)
    t0 = time.time()
    status, body = send_prompt(args.port)
    dt = time.time() - t0
    print(f"[repro] response status={status} ({dt:.1f}s)", flush=True)
    print(f"[repro] body: {body[:500]}", flush=True)

    crashed = False
    if proc is not None:
        # Give the worker subprocess a beat to die if the request killed it
        time.sleep(2.0)
        rc = proc.poll()
        crashed = (rc is not None and rc != 0)
        if not args.keep_running:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    print(f"[repro] server crashed={crashed}, exit={proc.poll() if proc else 'n/a'}", flush=True)
    print(f"\n[repro] === last 200 log lines (HSAIL & NaN search) ===", flush=True)
    _tail(log_path, 200, grep=r"NaN|nan|HSAIL|0x1016|sampler|Sampler|GEMMA4|Traceback|Error|Exception")

    return 0 if (status == 200 and not crashed) else 1


def _tail(path: Path, n: int, grep: str | None = None):
    if not path.exists():
        print(f"[repro] no log file at {path}")
        return
    import re
    lines = path.read_bytes().splitlines()
    pat = re.compile(grep) if grep else None
    out = []
    for line in lines:
        s = line.decode(errors="replace")
        if pat is None or pat.search(s):
            out.append(s)
    for s in out[-n:]:
        print(s)


if __name__ == "__main__":
    sys.exit(main())
