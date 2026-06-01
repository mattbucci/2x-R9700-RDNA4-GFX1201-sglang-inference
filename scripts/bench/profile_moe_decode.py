#!/usr/bin/env python3
"""Profile a single-user decode step on a running SGLang server and break the
GPU time down by kernel family.

Drives SGLang's torch-profiler endpoint (`/start_profile` with `num_steps`,
which auto-stops and dumps a Kineto/chrome trace into SGLANG_TORCH_PROFILER_DIR),
then parses the trace: sums GPU-side kernel durations, groups by name + by
family (moe / awq-gemv / rocblas-gemm / nccl / attention / deltanet / other),
and reports the per-step breakdown + the memory roofline for the active weights.

This is the MoE analogue of the dense-GEMV decode profile that found the
v0.5.12 awq_gemv_bf16_kernel regression. For an A3B MoE the question is whether
the Triton moe_wna16 path runs far under the active-weight roofline at M=1.

Usage (with a server already running on --port, launched with
SGLANG_TORCH_PROFILER_DIR pointing at --trace-dir):
    python scripts/bench/profile_moe_decode.py --port 23334 \
        --trace-dir /tmp/prof --steps 40 --active-gb 1.5 \
        --label coder-30b@2k --out benchmarks/profiling/coder-30b-decode-profile.json
"""
import argparse, glob, gzip, json, os, sys, time
import requests


def newest_trace(trace_dir, since):
    cands = []
    for ext in ("*.json", "*.json.gz", "*.trace.json", "*.trace.json.gz"):
        cands += glob.glob(os.path.join(trace_dir, "**", ext), recursive=True)
    cands = [c for c in cands if os.path.getmtime(c) >= since - 1]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def family(name):
    n = name.lower()
    if "moe" in n or "fused_moe" in n or "grouped" in n or "expert" in n:
        return "moe"
    if "awq_gemv" in n:
        return "awq_gemv"
    if n.startswith("cijk") or "gemm" in n or "hgemm" in n or "f8f8" in n or "wmma" in n:
        return "rocblas_gemm"
    if "nccl" in n or "rccl" in n or "allreduce" in n:
        return "nccl"
    if "attn" in n or "attention" in n or "flash" in n or "decode_kernel" in n or "paged" in n:
        return "attention"
    if "deltanet" in n or "gated_delta" in n or "chunk_" in n or "recurrent" in n:
        return "deltanet"
    if "elementwise" in n or "norm" in n or "rope" in n or "rotary" in n or "silu" in n or "cast" in n or "vectorized" in n:
        return "elementwise_norm"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--steps", type=int, default=40, help="decode forward steps to capture")
    ap.add_argument("--prompt", default="Write a long detailed essay about GPU memory hierarchy.")
    ap.add_argument("--active-gb", type=float, default=0.0,
                    help="active int4 weight GB read per token PER GPU (for roofline); 0=skip")
    ap.add_argument("--bw", type=float, default=640.0, help="GDDR7 GB/s per card")
    ap.add_argument("--label", default="moe-decode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    model = requests.get(f"{base}/v1/models", timeout=30).json()["data"][0]["id"]

    # Warmup so weights/caches are hot and cuda-graph (if any) is replayed.
    requests.post(f"{base}/v1/chat/completions", timeout=120, json={
        "model": model, "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8, "temperature": 0})

    t0 = time.time()
    # num_steps auto-stops + dumps. activities GPU+CPU so we get device kernels.
    r = requests.post(f"{base}/start_profile", timeout=30, json={
        "output_dir": args.trace_dir, "num_steps": args.steps,
        "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False})
    if r.status_code != 200:
        print(f"start_profile HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)

    # Drive exactly the decode we want captured.
    gen = requests.post(f"{base}/v1/chat/completions", timeout=600, json={
        "model": model, "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.steps + 4, "temperature": 0})
    usage = gen.json().get("usage", {})

    # Give the profiler a moment to flush the trace to disk.
    trace = None
    for _ in range(40):
        time.sleep(1.5)
        p = newest_trace(args.trace_dir, t0)
        if p:
            try:
                trace = load_trace(p)
                tpath = p
                break
            except Exception:
                continue
    if trace is None:
        print("No trace file found in", args.trace_dir, file=sys.stderr)
        sys.exit(2)

    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    # GPU kernel events: kineto tags device ops cat in {"kernel","gpu_op","Kernel"}.
    by_name = {}
    gpu_total = 0.0
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = str(e.get("cat", "")).lower()
        if cat not in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(e.get("dur", 0.0))  # microseconds
        nm = e.get("name", "?")
        by_name[nm] = by_name.get(nm, 0.0) + dur
        gpu_total += dur

    by_family = {}
    for nm, us in by_name.items():
        f = family(nm)
        by_family[f] = by_family.get(f, 0.0) + us

    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:12]
    steps = max(args.steps, 1)
    per_step_gpu_ms = (gpu_total / 1000.0) / steps

    result = {
        "label": args.label,
        "model": model,
        "trace_file": tpath,
        "steps_captured": steps,
        "completion_tokens": usage.get("completion_tokens"),
        "gpu_busy_total_ms": round(gpu_total / 1000.0, 2),
        "per_step_gpu_busy_ms": round(per_step_gpu_ms, 3),
        "breakdown_pct_of_gpu_busy": {
            k: round(100.0 * v / gpu_total, 1)
            for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])
        } if gpu_total else {},
        "top_kernels_ms_total": {nm[:70]: round(us / 1000.0, 2) for nm, us in top},
    }
    if args.active_gb > 0:
        roof_ms = args.active_gb / args.bw * 1000.0
        result["active_weight_roofline_ms_per_step"] = round(roof_ms, 3)
        result["roofline_note"] = (f"{args.active_gb} GB active int4/card / {args.bw} GB/s; "
                                   f"per-step GPU busy runs {per_step_gpu_ms/roof_ms:.1f}x over roofline")

    js = json.dumps(result, indent=2)
    print(js)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(js + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
