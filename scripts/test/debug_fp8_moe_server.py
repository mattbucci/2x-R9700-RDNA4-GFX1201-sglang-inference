#!/usr/bin/env python3
"""Debug server: monkey-patches MoE kernel to log args and sync after each call.

Run with: TORCHDYNAMO_DISABLE=1 python scripts/debug_fp8_moe_server.py
Then send: curl http://localhost:23334/generate -H "Content-Type: application/json" \
           -d '{"text":"Hello","sampling_params":{"max_new_tokens":5,"temperature":0}}'
"""
import os, sys, torch, functools

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['SGLANG_USE_AITER'] = '0'
os.environ['SGLANG_USE_AITER_AR'] = '0'
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '1'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.cache/triton_rdna4_t36')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang', 'python'))

# Monkey-patch the invoke function to add sync + logging
import sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels as moe_kernels
_orig_invoke = None
_call_count = 0

def _patched_invoke(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    mul_routed_weight, top_k, use_fp8_w8a8, use_int8_w8a8,
                    use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False, config=None, compute_type=None,
                    block_shape=None, B_zp=None, bias=None, padding_size=0,
                    a_use_tma=False, b_use_tma=False, c_sorted=False,
                    filter_expert=True, fuse_sum_all_reduce=False,
                    router_topk=1):
    global _call_count
    _call_count += 1

    if _call_count <= 3:  # Only log first 3 calls
        print(f"\n>>> MoE INVOKE #{_call_count}:", flush=True)
        print(f"  A: shape={A.shape}, dtype={A.dtype}, stride={A.stride()}, contiguous={A.is_contiguous()}", flush=True)
        print(f"  B: shape={B.shape}, dtype={B.dtype}, stride={B.stride()}", flush=True)
        print(f"  C: shape={C.shape}, dtype={C.dtype}, stride={C.stride()}", flush=True)
        if A_scale is not None:
            print(f"  A_scale: shape={A_scale.shape}, dtype={A_scale.dtype}, stride={A_scale.stride()}", flush=True)
        if B_scale is not None:
            print(f"  B_scale: shape={B_scale.shape}, dtype={B_scale.dtype}, stride={B_scale.stride()}, ndim={B_scale.ndim}", flush=True)
        print(f"  sorted_tokens: shape={sorted_token_ids.shape}, min={sorted_token_ids.min()}, max={sorted_token_ids.max()}", flush=True)
        print(f"  expert_ids: shape={expert_ids.shape}, min={expert_ids.min()}, max={expert_ids.max()}", flush=True)
        print(f"  use_fp8={use_fp8_w8a8}, block_shape={block_shape}, config={config}", flush=True)
        print(f"  top_k={top_k}, mul_routed={mul_routed_weight}, compute_type={compute_type}", flush=True)

        # Check for NaN/Inf in inputs
        if A.is_floating_point():
            print(f"  A has nan={A.isnan().any().item()}, inf={A.isinf().any().item()}", flush=True)

    # Call original
    try:
        _orig_invoke(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                     sorted_token_ids, expert_ids, num_tokens_post_padded,
                     mul_routed_weight, top_k, use_fp8_w8a8, use_int8_w8a8,
                     use_int8_w8a16=use_int8_w8a16, use_int4_w4a16=use_int4_w4a16,
                     per_channel_quant=per_channel_quant, config=config,
                     compute_type=compute_type, block_shape=block_shape,
                     B_zp=B_zp, bias=bias, padding_size=padding_size,
                     a_use_tma=a_use_tma, b_use_tma=b_use_tma, c_sorted=c_sorted,
                     filter_expert=filter_expert, fuse_sum_all_reduce=fuse_sum_all_reduce,
                     router_topk=router_topk)

        # Sync to catch async errors AT THIS POINT
        torch.cuda.synchronize()

        if _call_count <= 3:
            print(f"  >>> KERNEL OK! C mean={C.float().mean():.6f}, nan={C.isnan().any().item()}", flush=True)
    except Exception as e:
        print(f"  >>> KERNEL FAILED: {e}", flush=True)
        raise

# Install the monkey-patch
_orig_invoke = moe_kernels.invoke_fused_moe_kernel
moe_kernels.invoke_fused_moe_kernel = _patched_invoke

# Also patch the dense FP8 matmul to add sync
import sglang.srt.layers.quantization.fp8_kernel as fp8_kernel
_orig_fp8_matmul = fp8_kernel.w8a8_block_fp8_matmul
_fp8_count = 0

def _patched_fp8_matmul(*args, **kwargs):
    global _fp8_count
    _fp8_count += 1
    result = _orig_fp8_matmul(*args, **kwargs)
    if _fp8_count <= 2:
        torch.cuda.synchronize()
        print(f"  [dense FP8 matmul #{_fp8_count}] OK, shape={result.shape}", flush=True)
    return result

fp8_kernel.w8a8_block_fp8_matmul = _patched_fp8_matmul

print("=== Monkey-patches installed. Starting server... ===", flush=True)

if __name__ == '__main__':
    from sglang.launch_server import run_server
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(
        model_path=os.path.expanduser('~/AI/models/Qwen3-Coder-30B-A3B-FP8'),
        tp_size=2, dtype='bfloat16', quantization='fp8',
        kv_cache_dtype='fp8_e4m3', context_length=4096,
        mem_fraction_static=0.90, disable_cuda_graph=True,
        max_running_requests=1, chunked_prefill_size=2048,
        attention_backend='triton', num_continuous_decode_steps=1,
        disable_custom_all_reduce=True, trust_remote_code=True,
        watchdog_timeout=1800, skip_server_warmup=True,
        port=23334, host='0.0.0.0', enable_metrics=True,
    )
    run_server(args)
