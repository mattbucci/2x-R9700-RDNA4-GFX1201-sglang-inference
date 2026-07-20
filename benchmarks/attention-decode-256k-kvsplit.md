# 256K decode attention — split-KV CU occupancy (patch 086)

2026-07-14. Status: **FIXED (patch 086) — validated ~2.14× 256K decode win.**

## Summary

At true 256K, single-user decode is ~85–90% attention (KV-read). The Triton flash-decode was
KV-read-bound at only **~21% of the 1280 GB/s roofline** because the upstream AMD default hard-set
`num_kv_splits=16` (`server_args._handle_amd_specifics`), and the decode grid is
`head_groups × num_kv_splits` — at batch=1 that's ~32 blocks on gfx1201's **64 CUs** (half idle).
Raising the default to 64 lets `get_num_kv_splits` scale up to fill the CUs at depth:
**256K decode 14.4 → 30.7 tok/s (2.14×)** on coder-reap-25b, short context unregressed, output coherent.

## Evidence chain (all reproducible; artifacts under benchmarks/profiling/)

1. **Decode profile @255K** (`profile_moe_decode.py --context`, radix-cache-warmed): attention ≈ 90% of
   decode GPU time. `_fwd_grouped_kernel_stage1` is the decode-attention kernel; `_fwd_kernel` is the
   *extend* op (exclude it — it's residual tail-extend contamination).
   → `coder-reap-25b-256k-decode-profile.json`
2. **Roofline sweep**: decode attention runs at a fixed ~275 GB/s = ~21% of 1280 across 64K/128K/255K
   → ~4.7× headroom. → `coder-reap-25b-attn-roofline-sweep.json`
3. **Root cause**: grid = `batch(1) × cdiv(head_num, min(16, kv_group_num)) × num_kv_splits`
   (`decode_attention.py:~603`). coder-reap TP2 = 16 q / 2 kv heads → head-groups = 2 → blocks =
   2 × num_kv_splits. The AMD hard-cap `num_kv_splits=16` → 32 blocks on 64 CUs.
4. **num_kv_splits A/B @255K**: 16/32/48/64 → 21/38/46/51% roofline (decode-attn 46.4/26.1/21.2/19.3 ms);
   knee ~48–64. → `coder-reap-25b-256k-kvsplit-ab.json`
5. **End-to-end tok/s** (non-static, heuristic scales): 227K **14.4 → 30.7 (2.14×)**; 128 90.0 → 90.8
   (no regression); 8K 78.6 → 82.6. Coherent both arms.

## Fix (patch 086)

`server_args._handle_amd_specifics`: AMD `num_kv_splits` default **16 → 64** (+ `SGLANG_KV_SPLITS_OVERRIDE`
env to force a value). Safe fleet-wide: `get_num_kv_splits` picks ≤ cap based on seq_len + CU count, so
short context / small models pick few splits (no overhead) while deep context fills the CUs. 64 targets
~2× CU oversubscription for the worst case (head_groups = 2).

## Next / caveats

- Validated on coder-reap-25b (full-attention MoE). Should help every deep-context model on the Triton
  decode path; a per-family deep-ctx re-bench would confirm the fleet-wide gain. Hybrid-SWA models
  (north/laguna) may benefit less (SWA caps their KV); full-attention 256K models benefit most.
- A CU-derived default (from `device_core_count`) would generalize beyond gfx1201; 64 is tuned for the
  64-CU R9700.
- Roofline is still ~51% at 64 splits — further headroom (KV access pattern / fp8 dequant) exists but is
  kernel work; the config fix captured the bulk (21 → 51%).
- The fleet `results.json` deep-context numbers predate 086 and understate 256K decode for full-attention
  models; refresh on the next sweep.

## Pointers

- Grid: `decode_attention.py` `_decode_grouped_att_m_fwd`. Heuristic: `triton_backend.py`
  `get_num_kv_splits`. Default: `server_args._handle_amd_specifics`.
- Deep-context profiler (fixed for this): `profile_moe_decode.py --context N`. A/B scripts in scratchpad.
