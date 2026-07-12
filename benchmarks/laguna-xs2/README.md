# Laguna XS.2 FP8 — 2× R9700

`poolside/Laguna-XS.2-FP8` is served TP=2 with Triton attention/MoE, graph capture at batch size 1,
checkpoint FP8 KV, and the Laguna preset's 0.01 full-attention pool ratio. The original corrected graph
baseline in [`results.json`](results.json) is 34.49 / 33.84 / 30.06 tok/s at inputs 62 / 7,403 /
58,785. A 219,596-token coherent run measured 22.93 tok/s.

The 2026-07-12 v0.5.15 campaign added tuned R9700 MoE configs and SGLang-internal router, collective,
RMSNorm, and fused FP8 K/V-cache-store changes. Conservative medians from the final reverse-confirmed
run are **48.999 / 47.485 / 39.959 tok/s** at inputs 62 / 7,403 / 58,785, versus **47.772 / 39.342 /
27.202** with fused stores disabled: **+2.57% / +20.70% / +46.89%**. The first five-run ON pass was
slightly higher at 49.875 / 48.597 / 41.597 and is retained as a first pass rather than used for the
headline. A final-stack run at input 220,277 was coherent at **29.270 tok/s**.

Final correctness was 34/36 comprehensive: the evaluator rejected a response that contained the exact
speed-of-light answer, and the pre-existing `-7 % 3` model miss remained. Capabilities passed 2/2 and
the tool-call probe passed. See the [full receipt](../north-laguna-v0515-r9700-2026-07-12.md) for
stage-by-stage numbers, method differences, correctness status, and rejected experiments.

The JSON in this directory remains the original graph-corrected baseline so historical chart inputs do
not silently change. Consolidated final-stage values live in
[`../north-laguna-v0515-r9700-2026-07-12.json`](../north-laguna-v0515-r9700-2026-07-12.json).
