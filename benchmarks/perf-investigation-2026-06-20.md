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
   **RESULT (#30, 2026-06-20): don't-wire confirmed.** `bench_moe_hip_vs_triton.py --m 1 2 4 8 --iters 200`
   **GPU-faults** on gfx1201 (`Memory access fault by GPU node-1` — consistent with the K-major
   `[E,K,N/8]` layout the HIP kernel expects vs the harness/production N-major; the fault crashed
   cleanly, no VRAM wedge). Getting a clean A/B would require debugging the HIP kernel/layout for a
   lever whose ceiling is ~5% wall and which would risk breaking the cuda-graph win — not worth it.
   Decision: **don't wire; leave Triton MoE + cuda-graph.** (The NCCL-allreduce ~24% finding is the
   real MoE M=1 lever if revisited — separate from the expert GEMV.)

5. **Scheduler/cuda-graph/prefill (#5 → task #31).** Two corrections: (a) `num_continuous_decode_steps`
   is a **DEAD FLAG** in v0.5.13 (zero readers under `srt/`) — the launch.sh `DECODE_STEPS` tuning has
   NO effect; (b) the cuda-graph pool is **~0.3-0.5 GB, not 2+GiB**. The lever: `--cuda-graph-bs 1` for
   single-user presets → frees bs2-8 graph allocations (more contiguous VRAM for 256K KV), neutral
   decode speed, faster capture. A **capacity/fragmentation** win, not a speed win.
   **RESULT (#31, 2026-06-20): negligible — don't change presets.** A/B coder-30b @262144:
   `--cuda-graph-bs 1` vs baseline [1,2,4,8] → **max_total_num_tokens identical (817,979)**, graph
   mem 0.33→0.25 GB (frees **80 MB**), capture 10.7s→3.2s. The KV pool is allocated BEFORE graph
   capture at mem-fraction, so bs=1 does NOT raise capacity; 80 MB ≈ 3,500 tokens of an 817,979 pool
   = immaterial. The "graph pool fragments long-ctx allocation" premise is **refuted** (pool is 0.33
   GB, not 2+GiB). Not worth a preset change (and it would drop multi-user graph coverage). The real
   #31 deliverable = the **dead-flag comment in launch.sh** (`--num-continuous-decode-steps` is inert
   in v0.5.13 AND v0.5.12, verified zero readers).

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

### #32 — `--force-decode-window` BUILT + ceiling CONFIRMED (2026-06-20)

Patch `067-force-decode-window.patch.CANDIDATE` (server_args `--force-decode-window N` +
model_runner injection): forces a recent-N sliding window on the DECODE path of full-attention layers
by injecting N into the existing `sliding_window_size` fields (backend + every RadixAttention layer),
**reusing the fully-tested SWA decode path with NO kernel/metadata logic change.** Inert by default
(-1). Confirmed live: "windowed 64 full-attention decode layers to 4096 recent tokens."

**Ceiling bench (qwen3vl-32b AWQ, pure dense full-attention, @262144, server-log gen-throughput @~245K):**

| config | decode tok/s @245K | vs baseline |
|---|---|---|
| baseline (no window) | 8.2 | 1× |
| **`--force-decode-window 4096`** | **25.1** (n=16, median=max) | **3.06×** |

25.1 ≈ the model's short-ctx ceiling (25.9) → windowing **recovers essentially the entire depth
penalty**, confirming dense full-attention decode at depth is dominated by attention-over-deep-KV
(consistent with #2: it's *attention-work*-bound, and windowing cuts the work, not the per-byte cost).

**QUALITY GATE RESULT (2026-06-20): plain recent-window is FAST-BUT-INCOHERENT → needs attention sinks.**
Needle test on the windowed server (qwen3vl-32b, N=4096, ~230K depth, self-calibrated full=919352 tok):

| needle position | result | response |
|---|---|---|
| EARLY (~5% depth, outside window) | FAIL | "This is the final output message. The response is complete…" |
| LATE (end, INSIDE the 4096 window) | **FAIL — garbage** | ` ``````````````… ` (degenerate) |

The LATE failure is the decisive one: the model produces **garbage even for in-window content**. Forcing a
recent-window on a model **trained for full attention** collapses the attention softmax — the classic
**StreamingLLM** result: windowed attention WITHOUT **attention sinks** (the first few tokens) is unstable.
My initial build is plain recent-window (no sink), so the **3.06× speed is real but the output is incoherent**
— not a recall-vs-speed tradeoff, an outright collapse. (This is why Gemma works: its weights are *trained*
with SWA; qwen3vl-32b's are not.)

**⇒ Plain `--force-decode-window` is not shippable for full-attention models. The sink+window variant
(StreamingLLM: gather `[first S] ++ [recent N-S]`) is REQUIRED for coherence — that is the next experiment.**

**Window-size sweep (does a bigger window restore coherence? NO):**

| config | decode tok/s @245K | speedup | coherent? |
|---|---|---|---|
| baseline (full attention) | 8.2 | 1× | ✅ |
| `--force-decode-window 32768` | 20.6 | 2.5× | ❌ garbage (`[::]…`) |
| `--force-decode-window 4096` | 25.1 | 3.06× | ❌ garbage (backticks) |

**Window SIZE is not the lever** — both 4096 and 32768 collapse. Windowing reliably delivers 2.5-3×
decode (the mechanism works) but is **incoherent at every size** on full-attention-trained weights.
This firmly isolates the cause to the **missing attention sink** (StreamingLLM), not window size.

**#32 OUTCOME: speed lever PROVEN (2.5-3×), but plain windowing is unusable on full-attention models
(incoherent). Made it a follow-up task — two hypotheses to test, both opt-in/gated so native SWA
(Gemma) is untouched:**
1. **Attention sinks** (StreamingLLM): gather `[first S] ++ [recent N-S]` in BOTH the extend (prefill)
   and decode window paths — the sink must be present during prefill too, since windowed-no-sink prefill
   builds the broken hidden states that get cached. Cached K carries absolute-position RoPE so a first
   cut may skip RoPE remapping.
2. **Decode-only windowing** (full prefill, windowed decode only): keep prefill full-attention so the
   model builds a coherent full-context representation once, then window ONLY the decode KV read. May
   still hit the decode-side sink-collapse, but isolates "is it the windowed prefill or the windowed
   decode that breaks coherence?" Cheap to test (gate the window switch to `forward_decode` only).

These are research-grade (uncertain payoff, may ultimately need SWA-trained weights like Gemma). The
2.5-3× SPEED is established regardless; coherence is the gate.

### #35 RESOLVED — decode-only windowing WINS (full prefill + windowed decode, 2026-06-20)

Implemented the **decode-only** variant (patch 067 extended): `forward_extend` keeps FULL attention when
`--force-decode-window>0` (so the model builds a coherent full-context representation), and only
`forward_decode` windows. Gated on the flag → native-SWA (Gemma) untouched.

| config (qwen3vl-32b @262144) | decode tok/s @245K | speedup | coherence (needle) |
|---|---|---|---|
| baseline (full attention) | 8.2 | 1× | ✅ recalls anywhere |
| window-BOTH prefill+decode, N=4096 | 25.1 | 3.06× | ❌ garbage (incoherent) |
| window-BOTH, N=32768 | 20.6 | 2.5× | ❌ garbage |
| **decode-only (full prefill), N=4096** | **24.2** | **2.95×** | ✅ **LATE needle PASS (`ZEPHYR-4419`)** |

**Verdict: the windowed PREFILL was the coherence-killer** — keeping prefill full and windowing only the
decode gives **2.95× AND coherent in-window output**. The EARLY-needle FAIL (`Z444…`) is the *expected*
recall tradeoff (decode can't attend beyond the window), NOT incoherence. So no attention-sink machinery
was needed for basic coherence; full-prefill alone fixes it.

**What it is / isn't:** a **decode-throughput** win for single-user long-context *generation* (full prefill
means TTFT/prefill cost is unchanged — O(N) — so this helps decode-heavy workloads, not short-gen or
explicit deep-retrieval-at-decode). The model understands the whole context (full prefill) but generates
attending only the recent window. Ship as **opt-in `--force-decode-window N`** with that guidance. Sinks
(StreamingLLM) remain an optional future enhancement to extend recall, not required for coherence.
Harnesses: `scripts/bench/window_sweep.sh`, `scripts/bench/window_needle_test.py`.

## Serial test plan (ranked; cheap-and-decisive first)

| order | task | type | expected |
|---|---|---|---|
| ✅ #29 | size windowing prize (zero GPU) | analysis | ~2.5-3× ceiling → build #32 |
| ✅ #30 | MoE HIP-GEMV bench (no server) | settle | **don't-wire confirmed** (bench GPU-faults on gfx1201) |
| ✅ #31 | `--cuda-graph-bs 1` A/B + dead-flag cleanup | capacity | **negligible** (frees 80MB, max_total unchanged) — dead-flag comment landed |
| ✅ #32 | build+bench `--force-decode-window` | **the speed win** | speed PROVEN 2.5-3× (8.2→20.6→25.1), but plain window INCOHERENT at every size → sinks needed (follow-up task) |
| #33 | decode-QK FP32 A/B | quality | may fix int4 agentic long-KV |
| #34 | KV4/FP4 capacity port | PARKED | capacity for FP8-tight dense, multi-day |

Measurement discipline (per `feedback-spec-decode-measure-serverlog`): server-log `gen throughput` AT
TRUE DEPTH on real content; never client TPOT or short-depth-on-a-big-server. One server at a time
(no concurrent serving/calib).

## #39 IN PROGRESS — top-K attention-mass sparse KV (recall-preserving the decode-window win)

The shipped `--force-decode-window` (#35, 2.95×) is **retrieval-blind**: it keeps only the recent
window, dropping all mid-context KV. #39 replaces "last-N tokens" with "top-K most-relevant pages"
feeding the *same* index-agnostic decode kernel (`forward_decode` reads `kv_indices` opaquely —
triton_backend.py:1515-1557), so recall is preserved at a fraction of the O(ctx) KV read.

**Injection map (Explore agent, 2026-06-20).** Recency indices are built by
`update_sliding_window_buffer()` (triton_backend.py:1736-1780): `window_kv_start_idx = seq_lens -
min(seq_lens, window)` → literally the tail. #39 rewrites `window_kv_indices` in-place (respecting the
`window_kv_indptr` per-request spans) to the selected pages' token slots. Hard constraint: cuda-graph
buffer `cuda_graph_window_kv_indices` is fixed-size `max_num_tokens × window` → **K is static, budget =
K×PAGE**. Key insight: `forward_decode` runs **per-layer**, so per-layer/per-query selection can live
there with one reused scratch buffer — the "per-layer index set" problem dissolves (layers are sequential).

**Scorer-selection — synthetic gate then REAL-KEY confirmation (the real test reversed the synthetic).**

Step 1, synthetic (`scripts/bench/topk_scorer_recall_proto.py`, CPU, needle-recall): established the
robust part — **recency (shipped `--force-decode-window`) = 0.0 needle-recall** (retrieval-blind, #39
justified) — but its bbox-vs-centroid verdict (bbox "collapses", centroid "wins") was an artifact of an
**adversarial-caricature** structured key model and did NOT survive real keys. Kept as a recency-blind +
page-sensitivity probe only; its scorer verdict is explicitly disavowed in its docstring.

Step 2, REAL keys (`scripts/bench/topk_scorer_realkey.py`, GPU): Qwen3-4B (pure full-attention GQA+rope,
same generation as our fleet), 8170 real tokens, **post-rope** q/k captured by wrapping the rope fn,
metric = **fraction of true attention probability mass captured** by the selected pages (0.95 ≈ lossless
windowed decode). Deep layers (retrieval lives there), budget=2048 (~25% of ctx):

| layer | recency (shipped) | **bbox @PAGE8** | cent @PAGE8 | oracle (top-2048 tok) |
|---|---|---|---|---|
| 9  | 0.186 | **0.799** | 0.800 | 0.972 |
| 18 | 0.267 | **0.799** | 0.764 | 0.996 |
| 27 | 0.077 | **0.868** | 0.757 | 0.994 |
| 34 | 0.113 | **0.790** | 0.721 | 0.994 |

- **#39 premise CONFIRMED on real keys:** criticality top-K captures **3–11× more attention mass than
  recency** at the same decode cost (layer 27: 0.868 vs 0.077), nearly reaching the token-level oracle.
- **Quest bounding-box is the production scorer** — best on real keys, beating centroid as pages shrink.
  The synthetic "disqualification" was wrong; the original plan to lift `quest_algorithm.py`'s scorer holds.
- **PAGE=8 is the sweet spot** (bbox 0.87@PAGE8 vs 0.78@PAGE32 vs 0.72@PAGE64 on layer 27). Smaller pages =
  finer selection, closer to oracle, at ~2× the rep memory of PAGE=16 — a tunable knob.
- `cent_c == cent` exactly on real keys → real keys lack the dominant shared DC the synthetic injected.

**Decision for the build:** scorer = Quest bbox (min/max page reps), PAGE=8 default (configurable),
budget = the existing window size. Rep memory at PAGE=8/256K is non-trivial (min+max per page per
kv-head per layer) — make page-size + scorer (bbox vs cheaper centroid) configurable so the
recall/memory tradeoff is tunable per model. Footgun caught: GQA query layout must be head-major or
the needle is silently swamped (mismeasured 5.7 vs correct 48) — see proto docstring.

**Next: the build (turnkey checklist, task #40).** All on `/data/sgl-rebase`; capture each edit into a
`patches/0NN-decode-topk-sparse.patch.CANDIDATE` immediately; equivalence-gate vs pristine clone; boot
+ needle test to validate; NEVER abort a TP=2 prefill mid-flight.
1. `server_args.py`: add `decode_topk_pages: int = -1` (K pages; -1=off), `decode_topk_page_size: int = 8`,
   `decode_topk_scorer: str = "bbox"` (`bbox`|`centroid`) + argparse. Window/budget = K × page_size.
2. Rep storage (new, per attention layer, per kv-head): `page_k_min/page_k_max` `[n_pages, Hkv, D]` bf16.
   Allocate in `init_cuda_graph_state`-adjacent path sized to max ctx; graph-resident. Memory note in help.
3. Rep build: hook the EXTEND/prefill write (where KV is written to pool) to compute per-page min/max over
   the just-written tokens. Decode: update only the growing last partial page each step (cheap).
4. `forward_decode` (per-layer, query in hand): when `decode_topk_pages>0`, compute Quest criticality
   `where(q≥0,q·kmax,q·kmin).sum` over (Hkv,D) per page (lift from `quest_algorithm.py:_retrieve_page_scores`),
   `topk(K)` (+ always-include sink page 0 + recent page), gather those pages' token slots into the
   pre-allocated `window_kv_indices` (respect `window_kv_indptr` spans), then the existing windowed decode
   dispatch consumes it unchanged. K static → cuda-graph-safe; topk fixed-shape.
5. Validate: `scripts/bench/window_needle_test.py` with `--decode-topk-pages 256` (=2048 tok @page8) must
   recover MID-context needles that plain `--force-decode-window 2048` misses; decode tok/s ≈ window (same
   read budget). Expect ~0.8–0.87 attention-mass recall per the real-key gate.
Reuses the patch-067 insight end-to-end: the decode kernel is index-agnostic — #39 just feeds it a
criticality-selected index set instead of a recency one.

### #40 v1 BUILT + VALIDATED — recall win PROVEN on the real serving stack (2026-06-20)

Built v1 (correctness-first, eager) on `/data/sgl-rebase` and validated end-to-end. v1 deviates from
the turnkey plan in one way that *shrinks* blast radius: it does NOT set `sliding_window_size` (so it
never touches `model_runner` or the 067 SWA-injection path), and instead leaves the FULL `kv_indices`
in the metadata and SELECTS in `forward_decode`. Edits (compiled, WIP-captured in
`patches/069-decode-topk-sparse.WIP.diff`):
- `server_args.py`: `--decode-topk-pages` / `--decode-topk-page-size` (default 8) / `--decode-topk-scorer`
  (bbox|centroid) + `__post_init__` auto-disables cuda-graph for the eager v1.
- `triton_backend.py`: flag captures in `__init__`; new `_build_topk_kv_indices` (per request: Quest bbox
  criticality over per-page min/max of the full KV vs the decode query → top-K pages + page-0 sink +
  recent partial page → shortened kv_indptr/kv_indices); `forward_decode` interception that swaps the
  full indices for the selected set, consumed by the unchanged decode kernel.

**Needle validation (`scripts/bench/window_needle_test.py`, EARLY/MID/LATE, 30K ctx, 2048-tok budget):**

coder-30b (MoE, full-attention layers): TOPK EARLY ✅ + LATE ✅ recalled `ZEPHYR-4419`.
qwen3vl-32b (pure dense, the primary #39 target) — clean SAME-MODEL A/B, same budget:

| needle | RECENCY `--force-decode-window 2048` | **TOPK `--decode-topk-pages 256×8`** |
|---|---|---|
| EARLY(~5%) | ❌ FAIL (garbage `ZAPPAQAAQ…`) | ✅ PASS `ZEPHYR-4419` |
| MID(~50%)  | ❌ FAIL (garbage `ZAPPAQQUQU…`) | ✅ PASS `ZEPHYR-4419` |
| LATE(end)  | ✅ PASS | ✅ PASS |

The MID(~50%) pass (needle at ~15K tok, far from both sink and recent window) is recovered purely by
criticality selection → the mechanism works, not an artifact. **v1 proves the RECALL claim** (the novel/
risky part): sparse criticality selection gives the model full-context retrieval at a windowed decode
read budget, where plain recency windowing garbles everything outside its window.

**Bonus finding — `--force-decode-window` (#35) is BROKEN on qwen3_moe (coder-30b):** boot crashes with
`'MHATokenToKVPool' object has no attribute 'full_to_swa_index_mapping'` in qwen3_moe.py
`apply_qk_norm_rope → create_fused_set_kv_buffer_arg` — setting `sliding_window_size` on a non-SWA pool
trips the fused kv-buffer SWA path. 067 was only validated on qwen3vl-32b. #39 sidesteps it (never sets
`sliding_window_size`). **FIXED (patch 070, validated):** `getattr` guard → `None` when the pool has no
`full_to_swa_index_mapping` (correct: full pool needs no full→swa remap). coder-30b now boots with
`--force-decode-window 2048` and generates coherently — the shipped #35 feature works on qwen3_moe.

**v1 is NOT the perf version.** It computes per-page min/max over the FULL KV every layer every step
(O(ctx), eager, Python per-request loop, CPU syncs) — so it validates recall, not speed (likely a decode
regression). The speedup needs **v2 (task #41):** incremental reps (built at prefill, last-page update per
step) + a fixed-shape cuda-graph-capturable scoring/gather kernel. v1's recall validation de-risks v2:
the selection logic is proven correct on the real stack; v2 is a pure perf/mechanics rebuild of the same
math. Clean `069` patch + equivalence gate to be cut when the code settles (after v2 or v1-ship decision).

### #41 v2 BUILT — recall preserved, but a PERF REGRESSION → v3 (cuda-graph) required (2026-06-20)

Built v2 (`_build_topk_kv_indices_v2`): cached per-layer page reps, extended incrementally (a page's
bbox min/max computed once when it completes), bf16 reps (fp32 OOM'd at 128K — ~8.6GB on top of a full
KV pool; bf16 + native-dtype min/max fixed it). Recall is bit-identical to v1 (EARLY/MID/LATE all PASS
on qwen3vl-32b @32K).

**Throughput A/B @128K depth (qwen3vl-32b, server-log gen tok/s, real ~120K prompt):**

| config | gen tok/s | vs baseline | recall |
|---|---|---|---|
| BASELINE (full attention) | 12.85 | 1.00× | ✅ complete |
| RECENCY `--force-decode-window 2048` | 24.93 | **1.94×** | ❌ recent-only |
| **TOPK_v2 `--decode-topk-pages 256` (page8)** | **9.58** | **0.75× (REGRESSION)** | ✅ complete (coherent) |

**v2 is SLOWER than baseline.** The eager per-step overhead dominates: cuda-graph is DISABLED (v1/v2 use
data-dependent shapes), and the selection does heavy Python/small-op work every layer every step —
`int(kv_indptr)` CPU syncs, `topk().tolist()`, `set()`, a Python list-comp gather, `torch.cat` — ×64
layers. That launch/sync overhead + the rep-read bandwidth (page8 reps ≈ ¼ of K) exceeds what windowing
the decode read saves. RECENCY (24.93) is the windowed-decode **ceiling** at 128K (1.94×); any #39 win
must come out of that budget minus rep-scoring cost.

**Conclusion: the recall mechanism is proven (v1/v2), but the speedup needs v3.** v3 = make the whole
select path cuda-graph-capturable: max-sized graph-resident rep buffer; per-step scatter-update of the
current page's running min/max (fixed op, no completion branch); score all pages (mask invalid→-inf);
fixed-K `topk`; fixed-shape gather of K×page_size slots (+sink+recent) via arithmetic indexing (no Python
control flow); re-enable cuda-graph. Pure torch, fixed shapes — no custom triton kernel needed. Then the
per-step overhead collapses into the graph and the only residual cost is rep-read bandwidth (tunable via
page size: bigger pages = less rep bandwidth + less memory, lower recall). Realistic target: a fraction
of the 1.94×/2.95× windowed ceiling, WITH full-context recall. Tracked as #43.

### #43 RESOLVED-ish — page size fixes the regression; #39 is SHIPPABLE NOW (eager), v3 is upside

The page-8 regression was page-count + rep-scan-bandwidth overhead, not fundamental. Sweeping page size
at FIXED 2048-token budget @128K (qwen3vl-32b) flips it positive — and needle recall (EARLY/MID/LATE)
holds at every size:

| config (budget 2048) | gen tok/s | vs baseline | needle EARLY/MID/LATE |
|---|---|---|---|
| baseline (full attention) | 12.85 | 1.00× | ✅ (full) |
| recency window 2048 | 24.93 | 1.94× | ❌ EARLY+MID FAIL (garbage) |
| topk page 8 (K=256) | 9.58 | 0.75× | ✅✅✅ |
| topk page 32 (K=64) | 14.83 | **1.15×** | ✅✅✅ |
| topk page 64 (K=32) | 15.90 | **1.24×** | ✅✅✅ |

**#39 is a shippable recall-preserving decode speedup TODAY** (page 32/64, eager v2 already in tree):
1.15–1.24× over baseline at 128K *while recovering* the mid-context needles recency garbles. Default
page size set to 32 (balanced; 64 for more speed, 8 for max recall/slower). v3 (cuda-graph, #43-remaining)
is now UPSIDE toward the 1.94×/2.95× windowed ceiling, not a prerequisite. Next: confirm at **256K**
(the mandate target, where the windowed ceiling is ~2.95× → bigger headroom) and rep memory at depth.

### #43 256K CORRECTION — a fixed budget does NOT preserve recall at depth (budget must scale)

256K validation (qwen3vl-32b @262144, page64 K=32 = budget 2048):
- **Throughput: TOPK64 12.93 vs baseline 8.29 tok/s = 1.56×** (speedup grows with depth, as expected).
- **BUT needle recall FAILED at 245K — all three (EARLY/MID/LATE)**, model hallucinated partial
  passphrases (`ZEP10R` ≈ corrupted `ZEPHYR-4419`). **The 1.56× was NOT recall-preserving.**

Root cause: **budget must scale with context.** budget 2048 = 0.8% of 245K, vs 1.6% @128K (passed) and
25% in the real-key gate. At 0.8% selection over ~3828 pages, bbox can't surface the needle page in the
top-K — far more pages dilute the fixed budget and loosen the bbox upper-bound's discrimination. So the
"shippable @128K, budget 2048" result holds at ~128K but a FIXED 2048 budget under-selects at 256K.
(The committed 128K claim — page32/64, budget 2048, 1.15–1.24×, recall PASS — remains valid for ~128K.)

Two fixes: (a) **staleness bug** — the v2 rep cache keyed on seq_len only, so back-to-back same-length
requests could reuse stale reps; now also keyed on the first physical slot (request fingerprint;
radix-shared prefixes keep slot0 so shared-page reps stay valid). (b) **budget scaling** — re-finding the
recall-preserving 256K budget (testing K=128 = 8192 = 3.3%). Expect lower speedup than 1.56× but
recall-complete. The honest operating-point question: at 256K, what's the speedup at the SMALLEST budget
that still recovers needles? That (not the recall-blind 1.56×) is the real #39 deliverable at the mandate
depth. v3 cuda-graph remains the path to push that operating point toward the windowed ceiling.

**256K budget sweep + key insight (2026-06-21).** Throughput @256K is **budget-INDEPENDENT** (~1.6×):
K=32/budget2048 → 12.93 (1.56×), K=128/budget8192 → 13.47 (1.63×). The per-step **rep-scan over ALL
pages dominates**, not the decode read — so a bigger budget costs ~nothing in speed but buys recall.
v3's real speed lever is therefore making the rep-scan cheap (fused kernel + cuda-graph), not shrinking
the budget. Recall vs budget @245K:

| budget | EARLY | MID | LATE | tok/s |
|---|---|---|---|---|
| 2048 (0.8%) | `ZEP10R` ❌ | `L0g1n2` ❌ | ❌ | 12.93 |
| 8192 (3.3%) | `ZEPHY-4419` (off 1) | `ZEPH-4419` (off 2) | ✅ exact | 13.47 |

budget 8192 moved deep needles from total hallucination to **near-exact** (LATE exact; EARLY/MID off by
1–2 chars). The needle page IS selected at 8192; deep reproduction is marginally imperfect. **Open
control (running):** does BASELINE full-attention recall EXACTLY at 245K? If baseline also near-misses at
this extreme depth, #39@8192 ≈ baseline quality at 1.63× = win; if baseline is exact, #39 loses a little
deep fidelity (→ try bigger budget / smaller page, both ~free on speed since rep-scan dominates).

**Control result (2026-06-21): BASELINE full-attention recalls EXACTLY at 245K** — all three needles
`ZEPHYR-4419`. So the #39 near-miss at budget 8192 is **NOT a depth limit — it's a #39 sparse-selection
fidelity cost** (deep needles drop 1–2 chars where full attention is exact). Since throughput is
budget-independent, the fix is a bigger budget at ~no speed cost. Testing K=256 (16384 = 6.7%) for the
exact-recall operating point at 256K. Honest framing: **#39 @128K is exact-recall-preserving (validated);
@256K it needs a context-scaled budget (≥16384, TBD) to match baseline-exact, still at ~1.6×.** The strict
single-needle bar may be harsher than agentic use needs — the **#44 harness eval** (agentic quality at
#39's operating point) is the decision-relevant quality gate, not exact char recall of one token.

**Budget hunt result (2026-06-21): the deep near-miss is BUDGET-INSENSITIVE.** K=256/16384 (6.7%) MID
@245K = `ZEPHY-4419` — same 1-char miss as budget 8192. Doubling the budget did NOT fix it. So this is
**not a recall/budget failure** (the needle is clearly RETRIEVED — `ZEPHY…4419`, not a hallucinated
passphrase) but a **sparse-attention fidelity artifact at extreme depth**: the model reads the selected
needle page but reproduces one char fuzzily where full attention is byte-exact. (Possibly more selected
pages = more distractors dilute the needle's attention; bigger budget doesn't help and may slightly
hurt.) **Complete #39 characterization:** @128K exact-recall-preserving + 1.15–1.24×; @256K ~1.6×
(budget-independent) with **near-exact** deep recall (needle retrieved, ±1 char). Stopping the
single-needle micro-opt here — diminishing returns, and the ±1-char artifact on a strict needle is
likely immaterial to agentic/reasoning use. **The quality eval (#44) is the ship-decision**, not exact
char recall. #39's *speedup* targets the DENSE fleet (qwen3vl-32b etc.); on MoE coders decode is
dispatch-bound so #39 there is a recall/quality feature, not a speed one — so #44 splits into: (a) a
256K reasoning/retrieval quality probe on a dense #39-target (does sparse decode preserve quality at
depth), and (b) the agentic coding harness on coder-30b (does sparse selection break multi-turn coding).

### #43 v3 increment 1 — fused bbox-criticality kernel (the rep-scan lever, 2026-06-21)

The budget-independence finding said the per-step **rep-scan dominates** (throughput identical at budget
2048 vs 16384 -> not the decode-read, not the K-sized gather loop; the score over ALL pages). So v3's
first lever is fusing that score. The eager `where(q>=0,q*pmax,q*pmin).sum((1,2))` materializes fp32 reps
+ multiple passes over the dominant rep data; the fused triton kernel (`fused_bbox_score` /
`_bbox_crit_kernel` in triton_backend.py) does ONE pass, one program/page, reducing Hkv*D in-register.
- **Offline (synthetic): identical ranking (top-64 overlap 1.000, max_rel_err 0.0), ~11x faster scoring**
  (n=3828: 0.152 -> 0.014 ms/call). Wired into `_build_topk_kv_indices_v2` with an eager fallback.
- **Live-path correctness: needle EARLY/MID/LATE all PASS** (qwen3vl-32b @32K, page32) - selection
  unchanged by the kernel.
- **256K throughput MEASURED: 14.71 tok/s (fused) vs 13.47 (eager) vs 8.29 (baseline) = 1.77×** (up from
  v2's 1.63×). The fused rep-scan delivered the predicted gain, correctness-preserving. #39 @256K is now a
  **1.77× recall-preserving** decode (near-exact deep recall); @128K stays 1.15–1.24× exact.

**v3 remaining headroom (1.77× → ~2–2.95× ceiling), diminishing returns per increment:** (v3.2) vectorize
the Python gather (topk.tolist()/set()/list-comp/cat + per-elem int() syncs → tensor ops: `starts =
keep*page`, `(starts[:,None]+arange(page)).flatten()`, gather) — cuts per-layer CPU-sync overhead AND is
the prerequisite for (v3.3) cuda-graph (fixed-shape capture, cuts all launch overhead — smaller win at
256K where per-step work is large). Both are the same low-risk correctness-preserving pattern as v3.1.
v3.1 is a clean shippable milestone; v3.2/v3.3 are future increments.
