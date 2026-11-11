#!/usr/bin/env python3
"""RAD-15 deferred-head preflight (no server): DeferredLmHead from a tree that carries lm_head_int8_tail, driven
through a stand-in for LogitsProcessor._get_logits and a stand-in decode-graph runner, one rank and two ranks.

Exits non-zero when a capture forward does not defer, a deferred replay serves a token other than the bf16
head's arg-max, a request the gate must refuse (sampling, logprobs, grammar, bias, penalties, a changed head
weight) does not get the full bf16 logits, a graph that did not defer is touched, or the patched modules do
not import.

    python scripts/test/preflight_lm_head_int8_tail.py --tree /data/sgl-v0522/python --dry-cpu          # torch reference filter
    python scripts/test/preflight_lm_head_int8_tail.py --tree /data/sgl-v0522/python --ext-dir build/lm_head_int8 --device 0
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ext-dir")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dry-cpu", action="store_true", help="CPU tensors, torch reference in place of the kernel")
    ap.add_argument("--synthetic", default="4096,512,96", help="N,K,T of the random head and inputs")
    ap.add_argument("--skip-import-check", action="store_true")
    ap.add_argument("--timing", action="store_true",
                    help="run with SGLANG_RDNA4_INT8_LM_HEAD_TIMING=1 (waits for the device, GPU only)")
    ap.add_argument("--out")
    a = ap.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = "" if a.dry_cpu else str(a.device)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["SGLANG_RDNA4_INT8_LM_HEAD"] = "1"
    os.environ.pop("SGLANG_RDNA4_INT8_LM_HEAD_FULL", None)
    os.environ.pop("SGLANG_RDNA4_INT8_LM_HEAD_TIMING", None)
    if a.timing:
        assert not a.dry_cpu, "--timing waits for the device"
        os.environ["SGLANG_RDNA4_INT8_LM_HEAD_TIMING"] = "1"
    if a.ext_dir:
        sys.path.insert(0, os.path.abspath(a.ext_dir))
    sys.path.insert(0, a.tree)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
    import torch
    import lm_head_lowbit_study as study
    from sglang.srt.layers import lm_head_int8_filter as flt
    from sglang.srt.layers import lm_head_int8_tail as tail_mod
    from sglang.srt.model_executor import cuda_graph_config

    tree = os.path.abspath(a.tree)
    assert os.path.abspath(tail_mod.__file__).startswith(tree), tail_mod.__file__
    assert tail_mod._ENABLED and not tail_mod._FORCE_FULL
    kernel = not a.dry_cpu
    if kernel:
        assert flt.lm_head_int8_argmax_hip is not None, "lm_head_int8_hip_ext is not importable"
    dev = torch.device("cuda:0" if kernel else "cpu")
    cuda_graph_config.check_cuda_graph_backend = lambda phase, backend: True
    w16, h16 = study.synthetic(torch, a.synthetic)
    n, k = w16.shape
    # an exact tie across the two halves of the head at step 2: the filter cannot prove a winner there
    tie_a = int(torch.argmax(torch.matmul(h16[2:3].float(), w16.float().T)[0]).item())
    w16[(tie_a + n // 2) % n] = w16[tie_a]
    w16, h16 = w16.to(dev).contiguous(), h16.to(dev).contiguous()
    steps = h16.shape[0]
    fails = []
    report = {"tree": tree, "kernel": kernel, "rows": n, "cols": k, "steps": steps}

    def check(cond, what):
        if not cond:
            fails.append(what)

    class Mode:
        def is_decode(self):
            return True

    class Backend_:
        pass

    Backend_.__name__ = "FullCudaGraphBackend"

    def sampling(**kw):
        base = dict(is_all_greedy=True, grammars=None, logit_bias=None, acc_additive_penalties=None,
                    acc_scaling_penalties=None, has_custom_logit_processor=False,
                    has_watermark_candidates=False, sampling_mask_batch_indices=None,
                    sampling_support_logprobs_capture_indices=None, penalizer_orchestrator=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def batch(si=None, **kw):
        base = dict(sampling_info=si if si is not None else sampling(), return_logprob=False, batch_size=1,
                    forward_mode=Mode())
        base.update(kw)
        return types.SimpleNamespace(**base)

    for world in (1, 2):
        shard = n // world
        heads = [types.SimpleNamespace(weight=w16[r * shard:(r + 1) * shard].contiguous()) for r in range(world)]
        full_t = torch.cat([h.weight for h in heads]).T.contiguous()
        cap = types.SimpleNamespace(is_capture_mode=False)
        tails, procs, metas, outs, bufs = [], [], [], [], []

        class Group:
            def __init__(self, rank):
                self.world_size = world
                self.rank = rank

            def all_gather_into_tensor(self, out, inp):
                for j in range(world):
                    out[4 * j:4 * j + 4] = inp if j == self.rank else run(tails[j].filter, tails[self.rank].hidden).clone()

        def run(f, x):
            return f.run(x) if kernel else f.run_reference(x.reshape(-1))

        class Processor:
            """The contract of the patched LogitsProcessor._get_logits: stash first, bf16 head otherwise."""

            def __init__(self, rank):
                self.vocab_size = n
                self.rank = rank
                self.full_calls = 0

            def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
                t = tails[self.rank]
                deferred = t.stash(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
                if deferred is not None:
                    return deferred
                self.full_calls += 1
                logits = torch.matmul(hidden_states.to(full_t.dtype), full_t)  # stands for matmul + TP gather
                logits_metadata.next_token_logits_buffer.copy_(logits)
                return logits_metadata.next_token_logits_buffer

        for r in range(world):
            f = flt.Int8HeadFilter(heads[r].weight, row_offset=r * shard)
            if not kernel:
                f.run = lambda x, _f=f: _f.run_reference(x.reshape(-1))
            proc = Processor(r)
            runner = types.SimpleNamespace(sampling_observer=None, watermark_state=None)
            t = tail_mod.DeferredLmHead(runner, proc, heads[r], f, flt.decide, Group(r), cap)
            buf = torch.zeros(1, n, dtype=torch.float32, device=dev)
            tails.append(t); procs.append(proc); bufs.append(buf)
            metas.append(types.SimpleNamespace(forward_mode=Mode(), next_token_logits_buffer=buf))
        graph_runner = types.SimpleNamespace(backend=Backend_())
        tag = f"world{world}"

        # outside capture mode the head runs in place
        x0 = h16[0:1]
        got = procs[0]._get_logits(x0, heads[0], metas[0])
        check(procs[0].full_calls == 1 and torch.isfinite(got).all().item(), f"{tag}: eager forward deferred")
        # capture: two warm-ups and the captured run defer, and the hook registers each output
        cap.is_capture_mode = True
        for r in range(world):
            for _ in range(3):
                res = procs[r]._get_logits(x0, heads[r], metas[r])
                out = types.SimpleNamespace(next_token_logits=res)
                tails[r].capture_hook(graph_runner, out, None, 1)
            outs.append(out)
            check(res is bufs[r] and torch.isinf(bufs[r]).all().item(), f"{tag}: capture did not defer on rank {r}")
            check(id(out) in tails[r]._deferred and procs[r].full_calls == (1 if r == 0 else 0),
                  f"{tag}: capture output not registered on rank {r}")
            # an extend-mode capture and a foreign head are left alone
            ext = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False),
                                        next_token_logits_buffer=bufs[r])
            check(tails[r].stash(procs[r], x0, heads[r], ext, None, True) is None, f"{tag}: extend capture deferred")
            check(tails[r].stash(procs[r], x0, types.SimpleNamespace(weight=heads[r].weight), metas[r], None, True)
                  is None, f"{tag}: foreign head deferred")
            check(tails[r]._pending is None, f"{tag}: pending state left behind")
        bad = types.SimpleNamespace(next_token_logits=bufs[0].clone())
        procs[0]._get_logits(x0, heads[0], metas[0])
        try:
            tails[0].capture_hook(graph_runner, bad, None, 1)
            check(False, f"{tag}: a capture output that is not the logits buffer was accepted")
        except RuntimeError:
            pass
        cap.is_capture_mode = False
        base_full = [p.full_calls for p in procs]

        def replay(i, fb):
            """What a graph replay leaves behind, then the eager tail on every rank."""
            for r in range(world):
                tails[r].hidden.copy_(h16[i:i + 1])
                bufs[r].fill_(float("-inf"))
            for r in range(world):
                tails[r].after_replay(outs[r], fb)

        wrong = fast = 0
        for i in range(steps):
            ref = torch.matmul(h16[i:i + 1], full_t)[0].float()
            replay(i, batch())
            for r in range(world):
                wrong += int(torch.argmax(bufs[r][0]).item() != torch.argmax(ref).item())
            finite = int(torch.isfinite(bufs[0][0]).sum().item())
            fast += int(finite == 1)
            if finite != 1:
                check(torch.equal(bufs[0][0], ref), f"{tag}: step {i} fell back without the bf16 logits")
            elif not torch.equal(bufs[0], bufs[world - 1]):
                check(False, f"{tag}: ranks disagree at step {i}")
        report[tag] = {"wrong_tokens": wrong, "fast_steps": fast, "fallback_steps": steps - fast,
                       "counters": [(t.steps, t.fast, t.unproven) for t in tails]}
        check(wrong == 0, f"{tag}: {wrong} wrong tokens")
        check(fast > steps // 2, f"{tag}: the int8 filter served only {fast} of {steps} steps")
        check(fast < steps, f"{tag}: the tied step was served by the int8 filter")
        check(all(t.fast == fast and t.steps == steps and t.unproven == steps - fast for t in tails),
              f"{tag}: counters {report[tag]['counters']}")
        if a.timing:
            laps = tails[0].laps or {}
            report[tag]["timing_us"] = {k: [round(1e6 * v[0] / v[1], 1), v[1]] for k, v in laps.items()}
            want = {"replay wait", "int8 filter", "readback and decision", "logit write", "bf16 head"}
            if world > 1:
                want.add("pack all-gather")
            check(want <= set(laps), f"{tag}: timing stages missing: {sorted(want - set(laps))}")
            check(laps.get("int8 filter", [0, 0])[1] == steps, f"{tag}: timing counted {laps.get('int8 filter')}")
        else:
            check(all(t.laps is None for t in tails), f"{tag}: the timing diagnostic is on without its switch")

        # requests the gate must refuse get the full bf16 logits
        ref = torch.matmul(h16[1:2], full_t)[0].float()
        refused = {
            "sampling": batch(sampling(is_all_greedy=False)),
            "logprobs": batch(return_logprob=True),
            "batch of two": batch(batch_size=2),
            "grammar": batch(sampling(grammars=[object()])),
            "logit bias": batch(sampling(logit_bias=torch.zeros(1))),
            "additive penalty": batch(sampling(acc_additive_penalties=torch.zeros(1))),
            "scaling penalty": batch(sampling(acc_scaling_penalties=torch.zeros(1))),
            "custom processor": batch(sampling(has_custom_logit_processor=True)),
            "watermark": batch(sampling(has_watermark_candidates=True)),
            "sampling mask": batch(sampling(sampling_mask_batch_indices=torch.zeros(1))),
            "support logprobs": batch(sampling(sampling_support_logprobs_capture_indices=torch.zeros(1))),
            "penalizer": batch(sampling(penalizer_orchestrator=types.SimpleNamespace(is_required=True))),
            "no sampling info": types.SimpleNamespace(sampling_info=None, return_logprob=False, batch_size=1,
                                                      forward_mode=Mode()),
        }
        for name, fb in refused.items():
            replay(1, fb)
            check(all(torch.equal(b[0], ref) for b in bufs), f"{tag}: '{name}' request did not get the bf16 logits")
        tails[0].model_runner.sampling_observer = object()
        replay(1, batch())
        check(torch.equal(bufs[0][0], ref), f"{tag}: an observer did not force the bf16 head")
        tails[0].model_runner.sampling_observer = None
        if world == 1:
            heads[0].weight.add_(0)  # an in-place write bumps the version: the int8 copy is stale
            replay(1, batch())
            check(torch.equal(bufs[0][0], ref), f"{tag}: a changed head weight did not force the bf16 head")
        # a graph that did not defer is not touched
        other = types.SimpleNamespace(next_token_logits=bufs[0])
        bufs[0].fill_(7.0)
        tails[0].after_replay(other, batch())
        check(bool((bufs[0] == 7.0).all().item()), f"{tag}: an unregistered graph output was rewritten")
        report[tag]["full_head_calls"] = [p.full_calls - b for p, b in zip(procs, base_full)]

    if not a.skip_import_check:
        import importlib

        for name in ("sglang.srt.layers.logits_processor", "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
                     "sglang.srt.model_executor.model_runner"):
            if not kernel:  # these modules need a GPU to import: compile the source and look for the hunk
                path = os.path.join(tree, *name.split(".")) + ".py"
                with open(path) as fh:
                    src = fh.read()
                compile(src, path, "exec")
                check("lm_head_int8_tail.ACTIVE" in src or "lm_head_int8_tail.maybe_build(self)" in src,
                      f"{name} does not carry the hunk")
                continue
            try:
                mod = importlib.import_module(name)
                check(getattr(mod, "lm_head_int8_tail", None) is tail_mod, f"{name} does not carry the hunk")
                check(os.path.abspath(mod.__file__).startswith(tree), f"{name} imported from {mod.__file__}")
            except Exception as ex:  # noqa: BLE001
                check(False, f"import {name}: {type(ex).__name__}: {str(ex)[:160]}")
        report["import_check"] = "imported" if kernel else "compiled"
    report["failures"] = fails
    report["gate"] = "PASS" if not fails else "FAIL"
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    print(f"LMH TAIL PREFLIGHT DONE gate={report['gate']}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
