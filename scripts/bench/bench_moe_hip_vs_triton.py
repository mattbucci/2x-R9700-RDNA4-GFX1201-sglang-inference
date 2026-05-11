#!/usr/bin/env python3
"""Microbench HIP awq_gemv_moe_hip vs Triton _fused_moe_kernel_sequence.

Bench-first decision for task #17 (wire HIP MoE kernel into SGLang).
Architecture caveat (from memory project_hip_awq_kernel_recovery.md):
HIP kernel is single-matmul, so MoE forward needs ≥2 calls + a silu+mul op
between gate+up and down. Triton path does it in one fused launch.
Phase 2 wiring only pays off if HIP×2 + silu_op_overhead < Triton_fused.

Dims: Qwen3.6-35B-A3B-REAM (192 experts, top-8, hidden=2048,
moe_intermediate=512, group_size=128).

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/bench/bench_moe_hip_vs_triton.py
    python scripts/bench/bench_moe_hip_vs_triton.py --m 1 4 8 --iters 100
"""
import argparse
import os
import statistics
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "components", "sglang", "python"))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, nargs="+", default=[1, 4, 8],
                   help="batch sizes to bench (token counts)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--moe-inter", type=int, default=512)
    p.add_argument("--num-experts", type=int, default=192)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def make_awq_weights_hip(E, K, N, group_size, device):
    """HIP kernel layout: qweight [E, K, N/8] int32 (on-disk AWQ format)."""
    PACK = 8
    num_groups = K // group_size
    qweight = torch.randint(0, 2**31 - 1, (E, K, N // PACK), dtype=torch.int32, device=device)
    scales = torch.randn(E, num_groups, N, dtype=torch.float16, device=device) * 0.01
    qzeros = torch.randint(0, 2**31 - 1, (E, num_groups, N // PACK), dtype=torch.int32, device=device)
    return qweight, scales, qzeros


def make_awq_weights_triton(E, K, N, group_size, device):
    """SGLang Triton MoE layout (post moe_wna16 convert_awq_tensor):
    qweight [E, N, K/8] int32, qzeros [E, N/8, K/G] int32, scales [E, K/G, N] fp16.
    Per `_fused_moe_kernel_sequence`: `E, N, _ = w1.shape`."""
    PACK = 8
    num_groups = K // group_size
    qweight = torch.randint(0, 2**31 - 1, (E, N, K // PACK), dtype=torch.int32, device=device)
    scales = torch.randn(E, num_groups, N, dtype=torch.float16, device=device) * 0.01
    qzeros = torch.randint(0, 2**31 - 1, (E, N // PACK, num_groups), dtype=torch.int32, device=device)
    return qweight, scales, qzeros


def time_fn(fn, warmup, iters, device):
    """Returns (median_us, min_us, max_us) over `iters` runs."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times_us.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times_us), min(times_us), max(times_us)


def bench_hip_path(m, args, weights):
    """HIP path: gate+up call → silu+mul → down call."""
    import awq_gemv_hip_ext
    device = args.device
    H, I = args.hidden, args.moe_inter
    K_w13, N_w13 = H, I * 2          # gate+up combined
    K_w2, N_w2 = I, H                # down

    w13_qw, w13_sc, w13_qz = weights["w13"]
    w2_qw, w2_sc, w2_qz = weights["w2"]

    num_slots = m * args.top_k
    activation = torch.randn(m, K_w13, dtype=torch.float16, device=device) * 0.1
    sorted_ids = torch.arange(num_slots, dtype=torch.int32, device=device)
    expert_ids = torch.randint(0, args.num_experts, (num_slots,), dtype=torch.int32, device=device)
    topk_w = torch.ones(num_slots, dtype=torch.float32, device=device) / args.top_k

    gate_up = torch.empty(num_slots, N_w13, dtype=torch.float16, device=device)
    intermediate = torch.empty(num_slots, I, dtype=torch.float16, device=device)
    output = torch.empty(num_slots, N_w2, dtype=torch.float16, device=device)

    def step():
        # Step 1: gate+up matmul (HIP, mul_routed_weight=False, split_k=0=auto)
        awq_gemv_hip_ext.awq_gemv_moe_hip(
            activation, w13_qw, w13_sc, w13_qz,
            gate_up, sorted_ids, expert_ids, topk_w,
            args.top_k, False, 0)
        # Step 2: silu(gate) * up — PyTorch op
        gate, up = gate_up.split(I, dim=-1)
        torch.nn.functional.silu(gate, inplace=True)
        torch.mul(gate, up, out=intermediate)
        # Step 3: down matmul (HIP, mul_routed_weight=True for routed scaling)
        awq_gemv_hip_ext.awq_gemv_moe_hip(
            intermediate, w2_qw, w2_sc, w2_qz,
            output, sorted_ids, expert_ids, topk_w,
            1, True, 0)

    return time_fn(step, args.warmup, args.iters, device)


def bench_triton_path(m, args, weights):
    """Triton path: _fused_moe_kernel_sequence does everything in one fused call."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _fused_moe_kernel_sequence,
    )
    device = args.device
    H, I = args.hidden, args.moe_inter

    w13_qw, w13_sc, w13_qz = weights["w13"]
    w2_qw, w2_sc, w2_qz = weights["w2"]

    activation = torch.randn(m, H, dtype=torch.float16, device=device) * 0.1
    topk_weights = torch.ones(m, args.top_k, dtype=torch.float32, device=device) / args.top_k
    topk_ids = torch.randint(0, args.num_experts, (m, args.top_k), dtype=torch.int32, device=device)
    # sorted_ids/expert_ids/num_tokens_post_padded normally come from a preceding
    # sort+pad call; for microbench purposes use straightforward layout.
    num_slots = m * args.top_k
    sorted_ids = torch.arange(num_slots, dtype=torch.int32, device=device)
    expert_ids = torch.randint(0, args.num_experts, (num_slots,), dtype=torch.int32, device=device)
    num_tokens_post = torch.tensor([num_slots], dtype=torch.int32, device=device)

    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}

    def step():
        out = _fused_moe_kernel_sequence(
            activation, w13_qw, w2_qw, topk_weights, topk_ids,
            sorted_ids, expert_ids, num_tokens_post,
            config, None, False,
            b1=None, b2=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=True,
            per_channel_quant=False,
            w1_scale=w13_sc, w2_scale=w2_sc,
            w1_zp=w13_qz, w2_zp=w2_qz,
            a1_scale=None, a2_scale=None,
            block_shape=[0, args.group_size],
            activation="silu",
            is_gated=True,
            no_combine=False,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=False,
        )
        return out

    return time_fn(step, args.warmup, args.iters, device)


def make_skinny_weights(E, K, N, group_size, device):
    """wvSplitK skinny ExLlama-shuffled layout: qweight [E, N, K/8] int32,
    scales [E, N, K/G] fp16. Per project_mgehre_moe_kernel_port.md memory."""
    PACK = 8
    num_groups = K // group_size
    qweight = torch.randint(0, 2**31 - 1, (E, N, K // PACK), dtype=torch.int32, device=device)
    scales = torch.randn(E, N, num_groups, dtype=torch.float16, device=device) * 0.01
    return qweight, scales


def bench_wvsplitk_path(m, args, weights):
    """wvSplitK HIP path: same 2-call gate+up → silu+mul → down as awq_gemv path.
    Implemented via the standalone hybrid_w4a16_moe_apply Python wrapper."""
    import skinny_gemms_int4_ext  # noqa: F401  (raises if not built)
    device = args.device
    H, I = args.hidden, args.moe_inter
    w13_qw, w13_sc = weights["w13"]
    w2_qw, w2_sc = weights["w2"]

    activation = torch.randn(m, H, dtype=torch.float16, device=device) * 0.1
    num_slots = m * args.top_k
    sorted_ids = torch.arange(num_slots, dtype=torch.int32, device=device)
    expert_ids = torch.randint(0, args.num_experts, (num_slots,), dtype=torch.int32, device=device)
    topk_w = torch.ones(num_slots, dtype=torch.float32, device=device) / args.top_k

    gate_up = torch.empty(num_slots, I * 2, dtype=torch.float16, device=device)
    output = torch.empty(num_slots, H, dtype=torch.float16, device=device)
    cu_count = torch.cuda.get_device_properties(device).multi_processor_count

    def step():
        # gate+up
        skinny_gemms_int4_ext.fused_moe_wvSplitK_int4_gemm(
            activation, w13_qw, w13_sc, gate_up, expert_ids,
            16, cu_count, args.group_size,
            torch.empty(0, dtype=torch.float16, device=device),
            sorted_ids, args.top_k,
        )
        # silu+mul
        gate, up = gate_up.split(I, dim=-1)
        torch.nn.functional.silu(gate, inplace=True)
        intermediate = gate * up
        # down
        skinny_gemms_int4_ext.fused_moe_wvSplitK_int4_gemm(
            intermediate, w2_qw, w2_sc, output, expert_ids,
            16, cu_count, args.group_size,
            torch.empty(0, dtype=torch.float16, device=device),
            sorted_ids, 1,
        )

    return time_fn(step, args.warmup, args.iters, device)


def main():
    args = get_args()
    print(f"Device: {torch.cuda.get_device_name(args.device)}")
    print(f"Dims: hidden={args.hidden} moe_inter={args.moe_inter} "
          f"experts={args.num_experts} top_k={args.top_k} gs={args.group_size}")
    print(f"Bench: warmup={args.warmup} iters={args.iters}\n")

    weights_hip = {
        "w13": make_awq_weights_hip(args.num_experts, args.hidden, args.moe_inter * 2,
                                    args.group_size, args.device),
        "w2": make_awq_weights_hip(args.num_experts, args.moe_inter, args.hidden,
                                   args.group_size, args.device),
    }
    weights_triton = {
        "w13": make_awq_weights_triton(args.num_experts, args.hidden, args.moe_inter * 2,
                                       args.group_size, args.device),
        "w2": make_awq_weights_triton(args.num_experts, args.moe_inter, args.hidden,
                                      args.group_size, args.device),
    }
    # wvSplitK uses skinny [E, N, K/8] layout — same total bytes but different stride.
    weights_skinny = {
        "w13": make_skinny_weights(args.num_experts, args.hidden, args.moe_inter * 2,
                                   args.group_size, args.device),
        "w2": make_skinny_weights(args.num_experts, args.moe_inter, args.hidden,
                                  args.group_size, args.device),
    }

    print(f"{'M':>3} | {'awq_gemv HIP (us)':>20} | {'wvSplitK HIP (us)':>20} | "
          f"{'Triton fused (us)':>20} | best")
    print("-" * 100)

    try:
        import awq_gemv_hip_ext  # noqa: F401
        hip_available = True
    except ImportError:
        hip_available = False
        print("WARN: awq_gemv_hip_ext not importable — awq_gemv path skipped")
    try:
        import skinny_gemms_int4_ext  # noqa: F401
        wv_available = True
    except ImportError:
        wv_available = False
        print("WARN: skinny_gemms_int4_ext not importable — wvSplitK path skipped")

    for m in args.m:
        if hip_available:
            try:
                hip_med, hip_min, hip_max = bench_hip_path(m, args, weights_hip)
                hip_str = f"{hip_med:>8.1f} ({hip_min:.1f}-{hip_max:.1f})"
            except Exception as e:
                hip_med, hip_str = float("inf"), f"CRASH: {type(e).__name__}: {e}"
        else:
            hip_med, hip_str = float("inf"), "n/a"

        if wv_available:
            try:
                wv_med, wv_min, wv_max = bench_wvsplitk_path(m, args, weights_skinny)
                wv_str = f"{wv_med:>8.1f} ({wv_min:.1f}-{wv_max:.1f})"
            except Exception as e:
                wv_med, wv_str = float("inf"), f"CRASH: {type(e).__name__}: {e}"
        else:
            wv_med, wv_str = float("inf"), "n/a"

        try:
            tri_med, tri_min, tri_max = bench_triton_path(m, args, weights_triton)
            tri_str = f"{tri_med:>8.1f} ({tri_min:.1f}-{tri_max:.1f})"
        except Exception as e:
            tri_med, tri_str = float("inf"), f"CRASH: {type(e).__name__}: {e}"

        # Pick best
        cands = [("awq_gemv", hip_med), ("wvSplitK", wv_med), ("triton", tri_med)]
        cands.sort(key=lambda x: x[1])
        winner, win_t = cands[0]
        runner = cands[1]
        if win_t == float("inf"):
            best = "all-crash"
        else:
            best = f"{winner} ({runner[1]/win_t:.2f}x vs {runner[0]})"

        print(f"{m:>3} | {hip_str:>20} | {wv_str:>20} | {tri_str:>20} | {best}")

    print("\nDecision rules:")
    print("  awq_gemv wins at M=1 → wire awq_gemv_moe_hip into MoeRunner (#17).")
    print("  wvSplitK wins at M=1 → wire HybridW4A16MoEExperts into MoeRunner (mgehre port).")
    print("  Triton wins at M=1 → skip both HIP paths; investigate Triton MoE further.")
    print("  Cutoff for HIP→Triton dispatch on M>1 informs SGLANG_MOE_HYBRID_W4A16_MAX_BATCH.")
    print("  Mgehre's vLLM defaults to M≤5 (RDNA 3.5 tuning). R9700 may differ — vary +/- 2 to confirm.")


if __name__ == "__main__":
    main()
