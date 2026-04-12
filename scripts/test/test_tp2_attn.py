#!/usr/bin/env python3
"""Test Triton attention kernel in TP=2 mode."""
import os, sys, torch, torch.multiprocessing as mp

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.cache/triton_clean')
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29502'

sys.path.insert(0, '/home/letsrtfm/AI/rdna4-inference-triton36/components/sglang/python')
sys.path.insert(0, '/home/letsrtfm/AI/rdna4-inference-triton36/components/sglang/sgl-kernel/python')


def worker(rank, world_size):
    os.environ['RANK'] = str(rank)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

    device = f'cuda:{rank}'
    # Qwen3-Coder dims: head_dim=128, num_q_heads=12 (24/TP=2), num_kv_heads=2 (4/TP=2)
    num_q_heads = 12
    num_kv_heads = 2
    head_dim = 128
    seq_len = 4
    max_seq = 64

    # Test extend attention
    q = torch.randn(seq_len, num_q_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(seq_len, num_kv_heads * head_dim, device=device, dtype=torch.float16)
    v = torch.randn(seq_len, num_kv_heads * head_dim, device=device, dtype=torch.float16)
    o = torch.zeros(seq_len, num_q_heads, head_dim, device=device, dtype=torch.float16)

    # KV cache
    k_cache = torch.randn(max_seq, num_kv_heads * head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(max_seq, num_kv_heads * head_dim, device=device, dtype=torch.float16)

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int64, device=device)

    try:
        extend_attention_fwd(
            q, k, v, o, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices,
            None,  # custom_mask
            True,  # causal
            None,  # mask_indptr
            seq_len,  # max_extend_len
            1.0, 1.0,  # k/v descale
            1.0 / (head_dim ** 0.5),  # sm_scale
        )
        torch.cuda.synchronize()
        print(f"[Rank {rank}] extend_attention OK! o mean={o.float().mean():.4f}", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] extend_attention FAILED: {e}", flush=True)

    # Test NCCL after attention
    t = torch.ones(100, device=device) * (rank + 1)
    torch.distributed.all_reduce(t)
    print(f"[Rank {rank}] all-reduce OK: {t[0].item()}", flush=True)

    torch.distributed.destroy_process_group()
    print(f"[Rank {rank}] DONE", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("=== ATTENTION TP=2 PASSED ===")
