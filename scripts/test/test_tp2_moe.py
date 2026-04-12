#!/usr/bin/env python3
"""Minimal TP=2 MoE kernel test without the full SGLang server.
Tests if the FP8 MoE kernel works when launched on 2 GPUs via multiprocessing.
"""
import os, sys, torch, torch.multiprocessing as mp

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.cache/triton_clean')
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

sys.path.insert(0, '/home/letsrtfm/AI/rdna4-inference-triton36/components/sglang/python')
sys.path.insert(0, '/home/letsrtfm/AI/rdna4-inference-triton36/components/sglang/sgl-kernel/python')


def worker(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(rank)}", flush=True)

    import triton
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import fused_moe_kernel_rdna4

    E, N, K = 128, 384, 2048
    top_k = 8
    num_tokens = 4

    A = torch.randn(num_tokens, K, device=f'cuda:{rank}').to(torch.float8_e4m3fn)
    B = torch.randn(E, N, K, device=f'cuda:{rank}').to(torch.float8_e4m3fn)
    A_scale = torch.ones(num_tokens, K // 128, device=f'cuda:{rank}', dtype=torch.float32)
    B_scale = torch.ones(E, N // 128, K // 128, device=f'cuda:{rank}', dtype=torch.float32)

    total_tokens = num_tokens * top_k
    pad_to = ((total_tokens + 15) // 16) * 16
    sorted_token_ids = torch.arange(pad_to, device=f'cuda:{rank}', dtype=torch.int32)
    sorted_token_ids[total_tokens:] = total_tokens
    expert_ids_block = torch.zeros(pad_to // 16, device=f'cuda:{rank}', dtype=torch.int32)
    for i in range(pad_to // 16):
        expert_ids_block[i] = i % E if i * 16 < total_tokens else -1
    num_tokens_post_padded = torch.tensor([pad_to], device=f'cuda:{rank}', dtype=torch.int32)
    C = torch.zeros(total_tokens, N, device=f'cuda:{rank}', dtype=torch.bfloat16)

    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    grid = lambda META: (triton.cdiv(pad_to, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # Run MoE kernel
    fused_moe_kernel_rdna4[grid](
        A, B, C, A_scale, B_scale,
        sorted_token_ids, expert_ids_block, num_tokens_post_padded,
        N, K, pad_to, total_tokens,
        A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        128, 128, top_k=top_k, compute_type=triton.language.bfloat16,
        use_fp8_w8a8=True, use_int8_w8a8=False, per_channel_quant=False, **config,
    )
    torch.cuda.synchronize()
    print(f"[Rank {rank}] MoE kernel OK! C mean={C.float().mean():.4f}", flush=True)

    # Test NCCL all-reduce
    test_tensor = torch.ones(100, device=f'cuda:{rank}') * (rank + 1)
    torch.distributed.all_reduce(test_tensor)
    print(f"[Rank {rank}] all-reduce OK: sum={test_tensor[0].item()}", flush=True)

    # Test MoE kernel AFTER NCCL (this is the critical test)
    C2 = torch.zeros(total_tokens, N, device=f'cuda:{rank}', dtype=torch.bfloat16)
    fused_moe_kernel_rdna4[grid](
        A, B, C2, A_scale, B_scale,
        sorted_token_ids, expert_ids_block, num_tokens_post_padded,
        N, K, pad_to, total_tokens,
        A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
        C2.stride(0), C2.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        128, 128, top_k=top_k, compute_type=triton.language.bfloat16,
        use_fp8_w8a8=True, use_int8_w8a8=False, per_channel_quant=False, **config,
    )
    torch.cuda.synchronize()
    print(f"[Rank {rank}] MoE after NCCL OK! C2 mean={C2.float().mean():.4f}", flush=True)

    torch.distributed.destroy_process_group()
    print(f"[Rank {rank}] DONE", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("=== ALL RANKS COMPLETED ===")
