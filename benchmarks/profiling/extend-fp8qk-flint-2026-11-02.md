# Flint: fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `flint` preset (FP8 KV, route on), `SGLANG_EXTEND_FP8_QK=0` then `1`; raw JSON in
[`extend-fp8qk-flint-2026-11-02/`](extend-fp8qk-flint-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,321 | 0.99–1.11 s | 0.93–1.04 s | −6 % (GEMM/launch-bound) |
| 58,473 | 17.5 s | **10.5 s** | **−40 %** |
| 197,315 | 146 s | **65 s** | **−55 %** |

Warmed 1-token / 64-token turns at 52K / 104K: 129 / 129 and 182 / 184 ms → 130 / 130 and 184 / 194 ms (the route serves
them; unchanged). Natural-haystack needle 32K / 131K × 0.1 / 0.5 / 0.9: **5/6 on both arms**, the same cell (32K @ 0.1)
running to the 2,048-token thinking cap on both, every other cell found. 256K reasoning probe (`probe_256k_quality.py`,
32K / 131K): **identical cell by cell on both arms** — vartrack and aggregate pass at 32K, multikey fails at 32K and all
three fail at 131K on both. That 33 % is Flint's own behaviour on this probe's repetitive filler (the NoPE model's known
weakness with low-entropy filler, cf. its needle receipt), not the QK dot; it is recorded as a Flint quality finding.
Text probes 3/3 on both arms (vision / video unsupported by the model).

Shipped: the `flint` block exports `SGLANG_EXTEND_FP8_QK=1` (`EXTEND_FP8_QK=0` restores the upcast).
