#!/usr/bin/env python3
"""Test individual Triton kernels on both torch 2.11 and 2.12 to isolate failures.

Run with each env:
  /home/letsrtfm/miniforge3/envs/sglang-clean/bin/python scripts/test_triton_kernels.py
  /home/letsrtfm/miniforge3/envs/sglang-triton36/bin/python scripts/test_triton_kernels.py
"""
import sys
import os
import torch
import triton
import triton.language as tl
import traceback

print(f"torch: {torch.__version__}")
print(f"triton: {triton.__version__}")
print(f"python: {sys.executable}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Clear triton cache for clean test
TRITON_CACHE = os.path.expanduser("~/.triton/cache")

def make_awq_weights(K, N, group_size=128, device="cuda:0"):
    """Create synthetic AWQ-formatted weights for testing."""
    # qweight: [K, N//8] int32 (8 x 4-bit values packed)
    qweight = torch.randint(-2**31, 2**31, (K, N // 8), dtype=torch.int32, device=device)
    # scales: [K//group_size, N] float16
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device=device) * 0.01
    # qzeros: [K//group_size, N//8] int32
    qzeros = torch.randint(-2**31, 2**31, (K // group_size, N // 8), dtype=torch.int32, device=device)
    return qweight, scales, qzeros


def test_awq_dequantize():
    """Test the AWQ dequantize kernel (standalone, no GEMM)."""
    print("=== Test 1: AWQ Dequantize Kernel ===")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang'))
    from python.sglang.srt.layers.quantization.awq_triton import awq_dequantize_triton

    K, N = 4096, 4096
    qweight, scales, qzeros = make_awq_weights(K, N)

    try:
        result = awq_dequantize_triton(qweight, scales, qzeros)
        # Check result is not all zeros or NaN
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        all_zero = (result == 0).all().item()
        std = result.std().item()
        print(f"  Shape: {result.shape}, dtype: {result.dtype}")
        print(f"  NaN: {has_nan}, Inf: {has_inf}, AllZero: {all_zero}, Std: {std:.6f}")
        print(f"  PASS" if not has_nan and not has_inf and not all_zero and std > 0.001 else "  FAIL")
    except Exception as e:
        print(f"  CRASH: {e}")
        traceback.print_exc()
    print()


def test_awq_gemm():
    """Test the AWQ GEMM kernel (dense model path)."""
    print("=== Test 2: AWQ GEMM Kernel (dense path) ===")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang'))
    from python.sglang.srt.layers.quantization.awq_triton import awq_gemm_triton

    K, N = 4096, 4096
    qweight, scales, qzeros = make_awq_weights(K, N)

    for M, split_k in [(1, 8), (4, 1), (16, 1), (32, 8)]:
        try:
            input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
            result = awq_gemm_triton(
                input_tensor, qweight, scales, qzeros,
                split_k_iters=split_k,
                block_size_m=16, block_size_n=64, block_size_k=64,
            )
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            all_zero = (result == 0).all().item()
            std = result.std().item()
            status = "PASS" if not has_nan and not has_inf and not all_zero and std > 0.001 else "FAIL"
            print(f"  M={M:3d}, split_k={split_k}: shape={result.shape}, std={std:.4f} -> {status}")
        except Exception as e:
            print(f"  M={M:3d}, split_k={split_k}: CRASH: {e}")
    print()


def test_awq_gemv():
    """Test the AWQ GEMV kernel (M=1 decode path)."""
    print("=== Test 3: AWQ GEMV Kernel (decode path) ===")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang'))
    from python.sglang.srt.layers.quantization.awq_triton import awq_gemv_triton

    K, N = 4096, 4096
    qweight, scales, qzeros = make_awq_weights(K, N)

    for split_k in [1, 4, 8, 16]:
        try:
            input_tensor = torch.randn(1, K, dtype=torch.float16, device="cuda:0")
            result = awq_gemv_triton(
                input_tensor, qweight, scales, qzeros,
                split_k_iters=split_k,
            )
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            all_zero = (result == 0).all().item()
            std = result.std().item()
            status = "PASS" if not has_nan and not has_inf and not all_zero and std > 0.001 else "FAIL"
            print(f"  split_k={split_k:2d}: shape={result.shape}, std={std:.4f} -> {status}")
        except Exception as e:
            print(f"  split_k={split_k:2d}: CRASH: {e}")
    print()


def test_dot_kernel():
    """Test a minimal tl.dot kernel — the fundamental matmul op."""
    print("=== Test 4: Minimal tl.dot Kernel ===")

    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            offs_k += BLOCK_K
        c = acc.to(tl.float16)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    for M in [1, 4, 16, 32]:
        K, N = 4096, 4096
        a = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda:0")
        c = torch.zeros(M, N, dtype=torch.float16, device="cuda:0")
        try:
            grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
            dot_kernel[grid](a, b, c, M, N, K, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
            ref = a @ b
            err = (c - ref).abs().max().item()
            status = "PASS" if err < 1.0 else f"FAIL (max_err={err:.2f})"
            print(f"  M={M:3d}: max_err={err:.4f} -> {status}")
        except Exception as e:
            print(f"  M={M:3d}: CRASH: {e}")
    print()


def test_interleave_shift():
    """Test interleave + shift pattern used in AWQ unpacking."""
    print("=== Test 5: Interleave + Shift Pattern (AWQ unpack) ===")

    @triton.jit
    def unpack_kernel(qweight_ptr, result_ptr, N_packed: tl.constexpr, N_full: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, N_packed)
        vals = tl.load(qweight_ptr + pid * N_packed + offs)
        vals = tl.interleave(vals, vals)
        vals = tl.interleave(vals, vals)
        vals = tl.interleave(vals, vals)
        # AWQ order: [0, 4, 1, 5, 2, 6, 3, 7]
        reverse_awq = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
        shifts = reverse_awq * 4
        shifts = tl.broadcast_to(shifts[None, :], (N_packed, 8))
        shifts = tl.reshape(shifts, (N_full,))
        unpacked = (vals >> shifts) & 0xF
        tl.store(result_ptr + pid * N_full + tl.arange(0, N_full), unpacked)

    N_packed = 64
    N_full = N_packed * 8
    rows = 128
    qw = torch.randint(-2**31, 2**31, (rows, N_packed), dtype=torch.int32, device="cuda:0")
    result = torch.zeros(rows, N_full, dtype=torch.int32, device="cuda:0")
    try:
        unpack_kernel[(rows,)](qw, result, N_packed=N_packed, N_full=N_full)
        # All values should be 0-15
        in_range = (result >= 0).all().item() and (result <= 15).all().item()
        nonzero = (result != 0).any().item()
        print(f"  Values in [0,15]: {in_range}, has non-zero: {nonzero}")
        print(f"  PASS" if in_range and nonzero else "  FAIL")
    except Exception as e:
        print(f"  CRASH: {e}")
        traceback.print_exc()
    print()


def test_fused_moe_pattern():
    """Test the fused MoE kernel pattern (grouped GEMM with expert routing)."""
    print("=== Test 6: Fused MoE Pattern (grouped GEMM) ===")

    @triton.jit
    def simple_moe_gemm(
        a_ptr, b_ptr, c_ptr,
        expert_ids_ptr, token_ids_ptr,
        M, N, K, num_tokens,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Simplified MoE GEMM: route tokens to experts, multiply."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Load token indices for this block
        token_mask = offs_m < num_tokens
        token_ids = tl.load(token_ids_ptr + offs_m, mask=token_mask, other=0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K
            # Gather input rows by token_id
            a_offs = token_ids[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptr + a_offs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            offs_k += BLOCK_K

        c = acc.to(tl.float16)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c, mask=token_mask[:, None] & (offs_n[None, :] < N))

    M, K, N = 32, 512, 256
    a = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda:0")
    c = torch.zeros(M, N, dtype=torch.float16, device="cuda:0")
    token_ids = torch.arange(M, dtype=torch.int32, device="cuda:0")  # identity mapping
    expert_ids = torch.zeros(M, dtype=torch.int32, device="cuda:0")

    try:
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
        simple_moe_gemm[grid](a, b, c, expert_ids, token_ids, M, N, K, M,
                              BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
        ref = a @ b
        err = (c - ref).abs().max().item()
        status = "PASS" if err < 1.0 else f"FAIL (max_err={err:.2f})"
        print(f"  max_err={err:.4f} -> {status}")
    except Exception as e:
        print(f"  CRASH: {e}")
        traceback.print_exc()
    print()


def test_awq_gemm_correctness():
    """Test AWQ GEMM correctness by comparing to manual dequant + matmul."""
    print("=== Test 7: AWQ GEMM Correctness (vs reference) ===")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang'))
    from python.sglang.srt.layers.quantization.awq_triton import (
        awq_gemm_triton, awq_dequantize_triton
    )

    K, N = 1024, 1024
    group_size = 128
    qweight, scales, qzeros = make_awq_weights(K, N, group_size)

    for M in [1, 4, 16]:
        try:
            input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
            # Reference: dequantize then matmul
            deq = awq_dequantize_triton(qweight, scales, qzeros)  # [K, N]
            ref = input_tensor @ deq

            # Test: fused GEMM
            split_k = 1 if M <= 4 else 8
            result = awq_gemm_triton(
                input_tensor, qweight, scales, qzeros,
                split_k_iters=split_k,
                block_size_m=16, block_size_n=64, block_size_k=64,
            )

            err = (result - ref).abs().max().item()
            rel_err = err / (ref.abs().max().item() + 1e-8)
            status = "PASS" if rel_err < 0.05 else f"FAIL (rel_err={rel_err:.4f})"
            print(f"  M={M:3d}: max_abs_err={err:.4f}, rel_err={rel_err:.6f} -> {status}")
        except Exception as e:
            print(f"  M={M:3d}: CRASH: {e}")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)

    test_dot_kernel()
    test_interleave_shift()
    test_awq_dequantize()
    test_awq_gemm()
    test_awq_gemv()
    test_awq_gemm_correctness()
    test_fused_moe_pattern()
