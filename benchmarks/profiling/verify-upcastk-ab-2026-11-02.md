# DSpark verify kernel: UPCAST_K A/B on `qwen38` (2026-11-02)

`SGLANG_SPLITKV_UPCAST_K=1` makes the split-KV verify kernel upcast the fp8 cache K to bf16 for the QK dot (the
gfx1201 extend kernel's pre-108 numerics) instead of the fp8 WMMA dot. Same session, vendor Qwen3.8-27B-FP8 on the
DSpark preset, `decode_ab --think-off --ignore-eos`, 80 tokens, 3-run medians, order default → upcast → default
([default1](verify-upcastk-ab-2026-11-02-default1.json), [upcast](verify-upcastk-ab-2026-11-02-upcast.json),
[default2](verify-upcastk-ab-2026-11-02-default2.json)); accept length is the mean of the server's decode-batch lines.

| input tokens | default (1st) | upcast | default (2nd) | upcast vs default |
|---:|---:|---:|---:|---:|
| 24 | 44.1 | **48.4** | 45.0 | +9 % |
| 7,331 | 35.4 | **37.7** | 35.8 | +6 % |
| 58,483 | 33.9 | 30.5 | 35.5 | −12 % |
| 197,326 | 28.8 | 22.3 | 29.0 | **−23 %** |
| accept length / rate | 2.85 / 0.232 | 2.94 / 0.244 | 2.85 / 0.232 | +3 % |

The upcast buys 3 % acceptance at a verify-kernel cost that grows with the prefix (bf16 K traffic + the extra convert
per tile), so it wins only where the prefix is short and loses 12–23 % at depth. **Default (fp8 dot) stays.** A
prefix-length switch (upcast below ~16K cached tokens) would be worth +6–9 % on short-context decode only; not
pursued, since short context is not the README goal.
