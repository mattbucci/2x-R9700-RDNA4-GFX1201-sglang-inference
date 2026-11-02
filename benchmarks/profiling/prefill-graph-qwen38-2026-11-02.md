# Prefill HIP graph for small extends on qwen38: null (2026-11-02)

The 3090 cut its qwen38 1-token agentic turn 79 → 39 ms at 32K by capturing short extends as CUDA graphs
(`--cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 8 16 32 64 128 --cuda-graph-max-bs-prefill 1
--cuda-graph-max-seq-len-prefill 128`; CROSS-TEAM 2026-11-02). All four flags exist in v0.5.22 and the breakable backend
captures on gfx1201 (num_tokens 8–128, 4.5 s, 1.0 GB per rank, pool unchanged at 206,324 tokens). Same session on the
shipped `qwen38` DSpark preset (route on), `measure_extend_cost.py --multi-turn`, warmed, median TTFT for a cache-hit turn
([raw JSON](prefill-graph-qwen38-2026-11-02/)):

| cached tokens | new tokens | default (1st) | graph (1st, slow boot) | default (2nd) | graph (2nd) |
|---:|---:|---:|---:|---:|---:|
| 51,816 | 1 / 64 | 107 / 159 | 151 / 232 | 107 / 159 | 115 / 171 |
| 104,446 | 1 / 64 | 134 / 214 | 147 / 249 | 132 / 218 | 130 / 216 |
| 175,808 | 1 / 64 | 169 / 272 | 170 / 275 | 163 / 264 | 159 / 268 |

The first graph arm decoded 18–27 ms/token against 15–25 on every other arm (a slow boot, discarded); the repeat at normal
clocks is within ±7 % of the defaults (+7 % at 52K, −3 % at 176K). Needle across a turn 6/6 and capabilities 5/5 on every
arm. **Not adopted.** On this rig the short extend is not launch-bound once the split-KV route serves the full-attention
layers (patches 107 / 110); what remains of the 107–163 ms turn is the DeltaNet layers' state restore + 1-token extend, the
M=1 MoE/GEMM pass and the all-reduces, which a graph does not shorten here.
