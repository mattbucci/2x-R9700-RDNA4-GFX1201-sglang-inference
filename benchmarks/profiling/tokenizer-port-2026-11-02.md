# 3090 tokenizer-side ports on qwen38: null on this rig (2026-11-02)

The 3090 cut its qwen38 agentic 1-token turn 364 → 142 → 92 ms at 197K with two CPU-side `serving_chat.py` changes (an
exact incremental tokenizer for append-only prompts and a lazy prompt decode-back on multimodal-flagged models; their patch
072). Their patch applies cleanly to v0.5.22 here and was gated the same way, on the shipped `qwen38` DSpark preset
(route on), `measure_extend_cost.py --multi-turn`, warmed, median TTFT for a cache-hit turn
([raw JSON](tokenizer-port-2026-11-02/)):

| cached tokens | new tokens | baseline (patch reverted) | lazy decode-back only | incremental + verify | incremental, no verify |
|---:|---:|---:|---:|---:|---:|
| 51,816 | 1 / 64 | 108 / 160 | 108 / 159 | 108 / 158 | 116 / 178 (warm-up pass) |
| 104,446 | 1 / 64 | 132 / 214 | 133 / 215 | 131 / 216 | 142 / 226 (warm-up pass) |
| 175,808 | 1 / 64 | 162 / 264 | 161 / 263 | 163 / 267 | **163 / 263** |

Exactness: the incremental tokenizer logged 0 mismatches in verify mode across every served turn, and an offline fuzz on
the Qwen3.8, Flint, Laguna and Gemma 4 tokenizers (80 encodes each, 72 cache hits) matched the full encode on every call.
Capabilities 5/5 on every arm. Speed: nothing, at any depth or suffix length.

Why: `py-spy record` on the tokenizer-manager process during twelve 1-token / 64-token turns at 176K collected **17
non-idle samples in 300 s** — that process is busy for roughly 14 ms of a 163 ms turn, and `tokenizer.encode` is one of
those samples. Our agentic turn is GPU / scheduler time (the 16 routed full-attention layers at ~2.8 ms each at 176K, the
48 DeltaNet layers' state restore and 1-token extend, the M=1 MoE/GEMM pass, the all-reduces), not host CPU; the 3090's
host was the bottleneck at its faster GPU step. Both changes are reverted from the live tree (the patch file is kept in
the session notes for re-application if a host-bound case appears); issue #35 closes as null.

What this points at instead: the 1-token turn at 176K is ~1.3× the server's own 64-token turn ratio and ~6.6 decode
steps — the remaining tax is in the non-routed layers' extend path and the per-layer launch overhead, the territory of
the 3090's prefill-graph capture (small-extend HIP graph) rather than the tokenizer.
