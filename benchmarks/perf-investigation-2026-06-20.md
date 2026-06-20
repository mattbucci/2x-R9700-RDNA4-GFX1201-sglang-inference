# 256K single-user performance investigation — 5 parallel code probes + serial test plan (2026-06-20)

Five parallel read-only investigations of the live SGLang tree (`/data/sgl-rebase`) to find
single-user 256K decode levers, then ranked into a serial test todo (tasks #29-34). The unifying
finding: **at true 256K depth, decode is COMPUTE-bound, not KV-read-bound** — so the levers live in
the attention-*work* and dispatch paths, not in KV-byte reduction.

## The five findings (condensed)

1. **Decode-attention precision (#1 → task #33).** Patch 011 made decode P@V FP32 + extend QK native-Q,
   but LEFT **decode-QK as BF16×BF16** (`decode_attention.py:363`) and extend P@V BF16. The decode-QK
   score path is exactly what the 3090 cross-team note (`da1a58e`) fingered for int4×triton long-KV
   agentic garble. Change: `tl.dot(q_k, k, out_dtype=tl.float32)` + drop the BF16 Q-cast (the FP32-QK
   construct already compiles in-tree at `extend_attention.py:493`). A **quality** lever (≈neutral
   speed). Kill criterion: confirm gfx1201 still emits the WMMA dot via `TRITON_KERNEL_DUMP`.

2. **KV-cache bandwidth (#2 → task #34, PARKED).** REFRAME: "256K decode is KV-read-bound" is
   **disproven** — `kvsplit_sweep` 16/32/64 → 12.21/12.18/12.21 t/s flat @244K; FP8 is the floor
   (sub-FP8 TurboQuant is decode-*negative*); layout already optimal (NHD, page-1, coalesced). No
   per-byte KV-read speed lever. KV4/FP4 pool exists but is **CUDA-only** (NVIDIA PTX, no triton
   unpack) — porting helps **context capacity, not tok/s**. Park unless a model can't reach 256K in FP8.

3. **Sparse/windowing (#3 → tasks #29/#32).** HEADLINE. SWA already works on triton (Gemma uses it);
   the decode path keys off `layer.sliding_window_size` (`triton_backend.py:1437`) and the SAME
   `decode_attention_fwd` consumes arbitrary `kv_indices` — O(N)→O(window) is already wired. Generalize
   via a `--force-decode-window N [--decode-sink S]` arg for full-attention models. **Biggest decode
   lever.** Quest/top-k KV (recall-preserving) is dead on gfx1201 (DeepSeek-only gate, no triton adaptor).

4. **MoE expert kernel (#4 → task #30).** LOW prize, prior = DON'T WIRE. The HIP MoE GEMV would
   replace ~20% of GPU-busy = ~9.7% of wall; cuda-graph already pulls the real lever (the 21ms launch
   gap → 1.67-2.54×); M=1 MoE is **dispatch-bound, not GEMV-bound**; splitting the fused Triton launch
   into 2 HIP launches + silu would add dispatches and risk breaking cuda-graph. Both `.so`s are built,
   so the bench is cheap — run to *settle*. Surfaced the real M=1 MoE compute split: **NCCL all-reduce
   ~24% + elementwise/norm ~30%** (a future TP-allreduce lever), expert-GEMV is a sideshow.

5. **Scheduler/cuda-graph/prefill (#5 → task #31).** Two corrections: (a) `num_continuous_decode_steps`
   is a **DEAD FLAG** in v0.5.13 (zero readers under `srt/`) — the launch.sh `DECODE_STEPS` tuning has
   NO effect; (b) the cuda-graph pool is **~0.3-0.5 GB, not 2+GiB**. The lever: `--cuda-graph-bs 1` for
   single-user presets → frees bs2-8 graph allocations (more contiguous VRAM for 256K KV), neutral
   decode speed, faster capture. A **capacity/fragmentation** win, not a speed win.

## #29 — windowing prize, sized from existing data (DONE 2026-06-20)

Pure full-attention models, short-ctx floor vs ~245K deep (from `v0513-resweep` + `gemma4-31b/results.json`):

| model | short-ctx | @~245K | recoverable (attn-over-deep-KV) | windowing ceiling |
|---|---|---|---|---|
| qwen3vl-32b (dense full-attn) | 25.9 t/s | 8.2 t/s | ~83ms of 122ms TPOT (~68%) | ~20-25 t/s (**~2.5-3×**) |
| devstral2 (dense full-attn) | 40.9 t/s | 13.1 t/s | ~52ms of 76ms TPOT (~68%) | ~30-35 t/s (**~2.5-3×**) |

**Caveat (Gemma):** Gemma-4-31B already windows 25/30 layers yet is the fleet-slowest at depth (5.2 t/s)
— **partial/recall-preserving windowing is bounded by the unwindowed layers.** The full ~2.5-3× needs
aggressive windowing (most/all layers), which costs long-range recall. The recall/speed Pareto (how few
layers can stay full while keeping most of the win) is the real design question for #32.

**Decision: prize clears the ≥2× bar → BUILD #32** (ceiling-bench window-all first to confirm ~2.5-3×,
then sweep the layer-subset/window-size recall tradeoff; gate quality on needle-in-haystack + probe trio).

## Serial test plan (ranked; cheap-and-decisive first)

| order | task | type | expected |
|---|---|---|---|
| ✅ #29 | size windowing prize (zero GPU) | analysis | ~2.5-3× ceiling → build #32 |
| #30 | MoE HIP-GEMV bench (no server) | settle | confirm don't-wire |
| #31 | `--cuda-graph-bs 1` A/B + dead-flag cleanup | capacity | more KV headroom @256K, neutral speed |
| #32 | build+bench `--force-decode-window` | **the speed win** | ~2.5-3× dense decode @256K (recall-gated) |
| #33 | decode-QK FP32 A/B | quality | may fix int4 agentic long-KV |
| #34 | KV4/FP4 capacity port | PARKED | capacity for FP8-tight dense, multi-day |

Measurement discipline (per `feedback-spec-decode-measure-serverlog`): server-log `gen throughput` AT
TRUE DEPTH on real content; never client TPOT or short-depth-on-a-big-server. One server at a time
(no concurrent serving/calib).
