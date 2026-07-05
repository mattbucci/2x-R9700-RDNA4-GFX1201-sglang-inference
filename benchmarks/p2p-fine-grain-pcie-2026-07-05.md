# `HSA_FORCE_FINE_GRAIN_PCIE` — measured no-op for TP=2 P2P on 2×R9700 (2026-07-05)

**Question** ([issue #2](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/issues/2), from ShafaAat): the ROCm runtime flag `HSA_FORCE_FINE_GRAIN_PCIE=1` is documented as enabling peer-to-peer transport over PCIe. It isn't listed in our prerequisites — did we need it, and would enabling it unlock more performance?

**Answer: it's already enabled, and it's a measured no-op on this box.** `scripts/common.sh:74` (`setup_rdna4_env`) has exported `HSA_FORCE_FINE_GRAIN_PCIE=1` since the first commit (`4bdd438`), so every `launch.sh` — and the live server — already runs with it on. An A/B holding the full production env constant and toggling only this flag shows **no change to the RCCL transport decision and no change to all-reduce bandwidth**. True P2P on this box is delivered by the **kernel `CONFIG_HSA_AMD_P2P` + `iommu=pt`** (the documented prerequisites), not by this flag.

## Method

TP=2 RCCL all-reduce microbench — the exact collective SGLang tensor-parallelism uses (our MoE presets run `--disable-custom-all-reduce`, so TP=2 all-reduce is pure RCCL over PCIe). Both GPUs free (North-Mini taken down for a clean measurement). Same env for every run via `source scripts/common.sh; setup_rdna4_env`; only `HSA_FORCE_FINE_GRAIN_PCIE` toggled (`=1` vs `env -u`, i.e. fully unset).

- Bench: [`scripts/bench/p2p_allreduce_bw.py`](../scripts/bench/p2p_allreduce_bw.py) (bf16 all-reduce, 1 MB → 512 MB, 8 warmup + 30 timed iters/size)
- A/B harness: [`scripts/bench/p2p_flag_ab.sh`](../scripts/bench/p2p_flag_ab.sh)
- Stack: RCCL 2.27.7-HEAD, HIP 7.2.26015, ROCm 7.2.0, torch 2.11.0+rocm7.2, kernel 7.0.9-zen (CONFIG_HSA_AMD_P2P=y, iommu=pt), gfx1201 ×2, PCIe Gen4 x8/GPU.

## Result 1 — transport is P2P/IPC with the flag ON *and* OFF

RCCL's own channel-setup log (`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P`), identical in both runs:

```
Check P2P Type isAllDirectP2p 1 directMode 0
Channel 00/0 : 0[3000] -> 1[7000] via P2P/IPC
Channel 00/0 : 1[7000] -> 0[3000] via P2P/IPC
Channel 01/0 : 0[3000] -> 1[7000] via P2P/IPC
Channel 01/0 : 1[7000] -> 0[3000] via P2P/IPC
```

`isAllDirectP2p 1` + `via P2P/IPC` with the flag unset proves P2P is already active without it — it's gated by the kernel P2P config + IOMMU passthrough, not `HSA_FORCE_FINE_GRAIN_PCIE`.

## Result 2 — bandwidth is flag-independent

All-reduce GB/s (algbw = busbw for 2 GPUs). Four runs, both orderings:

| size_MB | ON (run1) | OFF (run1) | OFF (run2, first) | ON (run2, second) |
|--------:|----------:|-----------:|------------------:|------------------:|
| 1   | 7.55  | 6.30  | 7.43  | 6.62  |
| 4   | 9.73  | 9.70  | 9.79  | 9.69  |
| 16  | 10.53 | 10.50 | 10.52 | 10.52 |
| 32  | 10.61 | 10.58 | 10.60 | 10.59 |
| 64  | 10.66 | 10.63 | 10.64 | 10.63 |
| 128 | 10.69 | 10.65 | 10.67 | 10.66 |
| 256 | 10.68 | 10.66 | 10.68 | 10.67 |
| 512 | 10.67 | 10.67 | 10.68 | 10.68 |

- **≥4 MB: identical within ~0.3%** (run-to-run noise) regardless of flag. The steady-state bus bandwidth that carries real TP traffic is **~10.68 GB/s** — ~81% of the 13.2 GB/s measured Gen4 x8 ceiling, healthy for a 2-GPU ring all-reduce, and consistent with P2P being active.
- **The only wiggle is the 1 MB point, and it tracks *run order*, not the flag**: whichever run goes *first* lands ~7.4–7.5, the *second* ~6.3–6.6, in both orderings. Pure warmup/clock ordering, latency-regime, ~0.14 ms absolute — irrelevant to serving.

## Conclusion / actions

1. **Keep the flag set** — it's harmless, matches AMD's general guidance for PCIe P2P, and may be load-bearing on other systems or allocation paths (fine-grained coherent device memory). On *this* box (kernel P2P + iommu=pt already in place) it changes nothing measurable.
2. **Document it in the README prerequisites** (the actual gap issue #2 flagged) — noted as already-set + measured-no-op, with the real P2P levers called out (kernel `CONFIG_HSA_AMD_P2P` + `iommu=pt`).
3. **No serving A/B needed** — the transport decision and collective bandwidth are the mechanism; both are flag-independent here, so long-context decode inherits the same null. (The IOMMU-passthrough effect the README documents — channel renegotiation collapsing long-context decode — is a *different, orthogonal* lever and remains load-bearing.)

Raw logs: `/tmp/p2p-ab/{on,off}.{out,log}`.
