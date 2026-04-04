#!/usr/bin/env python3
"""Debug launcher for Qwen3-Coder-30B FP8 on single GPU."""
import os, sys
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SGLANG_USE_AITER'] = '0'
os.environ['SGLANG_USE_AITER_AR'] = '0'
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '1'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.cache/triton_rdna4_t36')
os.environ['USE_TRITON_W8A8_FP8_KERNEL'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components', 'sglang', 'python'))
from sglang.launch_server import run_server
from sglang.srt.server_args import ServerArgs
if __name__ == '__main__':
    args = ServerArgs(
        model_path=os.path.expanduser('~/AI/models/Qwen3-Coder-30B-A3B-FP8'),
        tp_size=1, dtype='float16', quantization='fp8',
        kv_cache_dtype='fp8_e4m3', context_length=4096,
        mem_fraction_static=0.95, disable_cuda_graph=True,
        max_running_requests=1, chunked_prefill_size=2048,
        attention_backend='triton', num_continuous_decode_steps=1,
        disable_custom_all_reduce=True, trust_remote_code=True,
        watchdog_timeout=1800, skip_server_warmup=True,
        port=23334, host='0.0.0.0', enable_metrics=True,
    )
    run_server(args)
