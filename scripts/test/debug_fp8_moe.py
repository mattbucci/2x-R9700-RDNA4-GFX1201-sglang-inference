#!/usr/bin/env python3
"""Step-by-step FP8 MoE debug: finds exactly which kernel crashes on gfx1201.

Monkey-patches key Triton kernel entry points with torch.cuda.synchronize()
calls to make async GPU errors synchronous, then runs a single forward pass.
"""
import os, sys, time

os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['SGLANG_USE_AITER'] = '0'
os.environ['SGLANG_USE_AITER_AR'] = '0'
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '1'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.cache/triton_rdna4_t36')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang', 'python'))

import torch
print(f"PyTorch {torch.__version__}, ROCm {torch.version.hip}")
print(f"GPU0: {torch.cuda.get_device_name(0)}")
print(f"GPU0 free: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")

# Step 1: Test basic FP8 WMMA operation (the core instruction that crashes)
print("\n=== Step 1: Test FP8 WMMA directly ===")
try:
    a_fp8 = torch.randn(16, 128, device='cuda:0').to(torch.float8_e4m3fn)
    b_fp8 = torch.randn(128, 128, device='cuda:0').to(torch.float8_e4m3fn)
    # This should use WMMA on RDNA4
    c = torch._scaled_mm(a_fp8, b_fp8, out_dtype=torch.float32,
                          scale_a=torch.tensor(1.0, device='cuda:0'),
                          scale_b=torch.tensor(1.0, device='cuda:0'))
    torch.cuda.synchronize()
    print(f"  _scaled_mm OK: {c.shape}, mean={c.float().mean():.4f}")
except Exception as e:
    print(f"  _scaled_mm FAILED: {e}")

# Step 2: Test the Triton FP8 block matmul kernel (dense linear)
print("\n=== Step 2: Test dense FP8 block matmul (Triton) ===")
try:
    from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul
    M, N, K = 4, 384, 512
    block_size = [128, 128]
    A = torch.randn(M, K, device='cuda:0').to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device='cuda:0').to(torch.float8_e4m3fn)
    As = torch.ones(M, K // block_size[1], device='cuda:0', dtype=torch.float32)
    Bs = torch.ones(N // block_size[0], K // block_size[1], device='cuda:0', dtype=torch.float32)

    C = w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"  w8a8_block_fp8_matmul OK: {C.shape}, mean={C.float().mean():.4f}")
except Exception as e:
    print(f"  w8a8_block_fp8_matmul FAILED: {e}")

# Step 3: Test per_token_group_quant_fp8
print("\n=== Step 3: Test per_token_group_quant_fp8 ===")
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import per_token_group_quant_fp8
    x = torch.randn(4, 512, device='cuda:0', dtype=torch.bfloat16)
    x_fp8, x_scale = per_token_group_quant_fp8(x, 128)
    torch.cuda.synchronize()
    print(f"  per_token_group_quant_fp8 OK: {x_fp8.shape} {x_fp8.dtype}, scale={x_scale.shape}")
except Exception as e:
    print(f"  per_token_group_quant_fp8 FAILED: {e}")

# Step 4: Test the fused MoE kernel with synthetic data
print("\n=== Step 4: Test fused MoE kernel with synthetic data ===")
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        fused_moe_kernel_rdna4, _is_rdna4
    )
    import triton

    print(f"  _is_rdna4: {_is_rdna4}")

    # Simulate Qwen3-Coder-30B MoE dimensions
    # E=128 experts, N=384 intermediate, K=2048 hidden (per TP)
    E, N, K = 128, 384, 2048
    num_tokens = 4
    top_k = 8  # Qwen3 MoE uses top-8

    # Create synthetic FP8 data
    A = torch.randn(num_tokens, K, device='cuda:0').to(torch.float8_e4m3fn)  # activations
    B = torch.randn(E, N, K, device='cuda:0').to(torch.float8_e4m3fn)  # expert weights

    # Block scales [128, 128]
    A_scale = torch.ones(num_tokens, K // 128, device='cuda:0', dtype=torch.float32)
    B_scale = torch.ones(E, N // 128, K // 128, device='cuda:0', dtype=torch.float32)

    # Routing: each token goes to top_k experts
    topk_weights = torch.ones(num_tokens * top_k, device='cuda:0', dtype=torch.float32) / top_k
    topk_ids = torch.randint(0, E, (num_tokens, top_k), device='cuda:0', dtype=torch.int32)

    # Sorted token IDs and expert IDs (simplified)
    sorted_token_ids = torch.arange(num_tokens * top_k, device='cuda:0', dtype=torch.int32)
    expert_ids = topk_ids.flatten()
    # Need to sort by expert
    sorted_order = expert_ids.argsort()
    sorted_token_ids = sorted_token_ids[sorted_order]
    expert_ids_sorted = expert_ids[sorted_order]

    # Pad to multiple of BLOCK_SIZE_M=16
    pad_to = ((num_tokens * top_k + 15) // 16) * 16
    if pad_to > num_tokens * top_k:
        sorted_token_ids = torch.cat([sorted_token_ids,
            torch.full((pad_to - num_tokens * top_k,), num_tokens * top_k, device='cuda:0', dtype=torch.int32)])

    # Expert IDs per block
    num_blocks = pad_to // 16
    expert_ids_block = torch.zeros(num_blocks, device='cuda:0', dtype=torch.int32)
    for i in range(num_blocks):
        idx = i * 16
        if idx < len(expert_ids_sorted):
            expert_ids_block[i] = expert_ids_sorted[idx]
        else:
            expert_ids_block[i] = -1  # padding

    num_tokens_post_padded = torch.tensor([pad_to], device='cuda:0', dtype=torch.int32)

    C = torch.zeros(num_tokens * top_k, N, device='cuda:0', dtype=torch.bfloat16)

    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    print(f"  Launching kernel: A={A.shape} B={B.shape} C={C.shape}")
    print(f"  A_scale={A_scale.shape} B_scale={B_scale.shape}")
    print(f"  sorted_tokens={sorted_token_ids.shape} experts={expert_ids_block.shape}")
    print(f"  config: {config}")

    fused_moe_kernel_rdna4[grid](
        A, B, C,
        A_scale, B_scale,
        sorted_token_ids, expert_ids_block, num_tokens_post_padded,
        N, K,
        sorted_token_ids.shape[0],
        num_tokens * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        128, 128,  # group_n, group_k (block_shape)
        top_k=top_k,
        compute_type=triton.language.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        per_channel_quant=False,
        **config,
    )
    torch.cuda.synchronize()
    print(f"  fused_moe_kernel_rdna4 OK! C mean={C.float().mean():.6f}, any_nan={C.isnan().any().item()}")

except Exception as e:
    import traceback
    print(f"  fused_moe_kernel_rdna4 FAILED: {e}")
    traceback.print_exc()

# Step 5: Test the ORIGINAL SGLang kernel (not rdna4 variant)
print("\n=== Step 5: Test original SGLang fused_moe_kernel ===")
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import fused_moe_kernel
    # Reuse vars from Step 4
    C2 = torch.zeros(num_tokens * top_k, N, device='cuda:0', dtype=torch.bfloat16)

    fused_moe_kernel[grid](
        A,
        None,  # a_desc
        B,
        None,  # b_desc
        None,  # bias
        C2,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids, expert_ids_block, num_tokens_post_padded,
        N, K,
        sorted_token_ids.shape[0],
        num_tokens * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        0, 0,  # bias strides
        C2.stride(0), C2.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        128, 128,  # group_n, group_k
        MUL_ROUTED_WEIGHT=True,
        top_k=top_k,
        compute_type=triton.language.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=True,
        c_sorted=False,
        filter_expert=True,
        swap_ab=False,
        FUSE_SUM_ALL_REDUCE=False,
        ROUTER_TOPK=top_k,
        **config,
    )
    torch.cuda.synchronize()
    print(f"  fused_moe_kernel (original) OK! C2 mean={C2.float().mean():.6f}")

except Exception as e:
    import traceback
    print(f"  fused_moe_kernel (original) FAILED: {e}")
    traceback.print_exc()

print("\n=== All steps completed ===")
