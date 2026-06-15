# TurboQuant KV on RDNA4 (gfx1201) — cheap de-risk result, 2026-06-15

**Question (README Open work #4, step 1):** does the `domvox/llama.cpp-turboquant-hip`
fork run TurboQuant KV **decode-neutral** on gfx1201 (only RDNA3/gfx1100 was tested
upstream)? This gates whether to undertake the multi-week SGLang fused-triton-decode
port. Gate: **M=1 decode tok/s must NOT regress.**

## Verdict: FAILS the decode-neutral gate — turbo3 is decode-NEGATIVE on RDNA4, worsening with depth.

### Build + run
- Fork **builds clean for gfx1201** (`cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx1201`, RC=0, amdclang, ROCm 7.2). Custom turbo HIP kernels (turbo-wht, mmvq-tq, turbo flash-attn) compile.
- **Baseline f16 KV runs great on gfx1201**: Llama-3.2-1B Q4_K_M tg64 = 325 t/s; Qwen2.5-1.5B Q4_K_M tg64 = 186 t/s.
- **turbo3 KV runs on head_dim-128 models** (Qwen2.5-1.5B — our real targets Qwen3.5/3.6-27B are head_dim 128). **Crashes on head_dim-64** (Llama-3.2-1B: `GGML_ASSERT(ggml_nelements(a)==ne0*ne1*ne2)` in `ggml_reshape_3d`←`build_attn` — a turbo-group/head_dim alignment edge case; turbo_group is 128-or-64 keyed on `ne[0]%128`).

### Decode @ depth — `llama-bench -fa 1 -d <depth> -n 64`, Qwen2.5-1.5B, gfx1201, 1 GPU

| depth | f16 tg/s | turbo3 tg/s | Δ |
|------:|---------:|------------:|----:|
| short (n64) | 185.5 | 165.3 | −11% |
| 8192  | 170.0 | 113.4 | −33% |
| 32768 | 142.5 |  62.1 | −56% |
| 65536 | 115.7 |  39.5 | **−66%** |

The regression **grows with KV depth** — the opposite of the fork's RDNA3 result
(−2%, decode-neutral) and exactly the long-context regime TurboQuant is meant to win.
On RDNA4 the per-step turbo dequant/WHT/flash-attn cost swamps the smaller-KV-read
benefit. (pp/prefill is fine: turbo3 12.6k vs f16 10.0k t/s — it's the *decode* path.)

## Implication
- The cheap de-risk path is **closed**: don't build the multi-week SGLang fused-triton-decode
  TurboQuant kernel on the assumption RDNA4 runs it decode-neutral — it does not.
- Making it viable on RDNA4 would require **RDNA4-specific optimization of the turbo
  flash-attn/dequant kernels** (the fork is RDNA3/wave32-tuned for gfx1100), a kernel-dev
  effort, and the worsening-with-depth trend makes the payoff doubtful.
- Reminder of narrow value (from the README audit): TurboQuant never gated single-user
  256K (AWQ-int4 already reaches it); its only real lever was FP8-256K-*agentic* for dense
  thinking models — which this result deprioritizes.

## Caveats (honest)
- Tested on small models (1B/1.5B). On a 27B the per-token *model* compute is far larger,
  so turbo3's overhead would be a **smaller fraction** of the step → the regression would
  be **milder than −66%**, but still a regression (turbo adds overhead, doesn't remove it)
  and still fails "must NOT regress". The KV-path economics (dequant-cost vs read-savings)
  are ~model-size-independent, so the *direction* transfers. A Qwen3.5-27B GGUF
  convert+bench would pin the exact 27B magnitude (optional, deprioritized).
- Out-of-the-box kernels (no RDNA4 tuning). turbo3 coherence not cleanly verified via the
  flaky `llama-cli` output path; the decisive gate metric (decode speed) fails regardless.

Artifacts: build at `/data/llama-turboquant` (gfx1201), GGUFs at `/data/gguf/`.
Fork: https://github.com/domvox/llama.cpp-turboquant-hip · upstream disc #21526.
