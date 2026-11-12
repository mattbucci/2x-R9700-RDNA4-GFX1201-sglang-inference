"""Diagnosis of the first R97-T kernel run: where an arm's BF16 output differs from the BF16-rounded reference and
by how many BF16 steps. KDIR = a directory holding the first-rule check (saved here as check_kernels.first-rule.py)
under the name check_kernels.py, RUN = the run's output directory, R97S = the R97-S receipts."""
import os
import sys

import torch
K, RUN, R97S = os.environ["KDIR"], os.environ["RUN"], os.environ["R97S"]
sys.path.insert(0, K)
sys.path.insert(0, os.path.join(R97S, "emu"))
import check_kernels as ck  # noqa: E402
import emulate_prefix as ep  # noqa: E402

cases = {c["name"]: c for c in torch.load(RUN + "/inputs.pt", map_location="cpu", weights_only=True)}
arms = {n: torch.load(RUN + "/out-%s.pt" % n, map_location="cpu", weights_only=True) for n in ck.ARMS}
for key in arms["base"]["cells"]:
    name, p, kernel = key.split("|")
    if p != "fp8":
        continue
    info = {k: v for k, v in arms["base"]["cells"][key].items() if k != "o"}
    refs, _ = ck.references(ep, cases[name], p, kernel, info)
    line = []
    for arm, var in (("base", "kernel"), ("on", "x256")):
        o = arms[arm]["cells"][key]["o"]
        r = refs[var].to(torch.float32).to(torch.bfloat16)
        d = (o.double() - r.double())
        ne = d != 0
        # one BF16 step at the reference value: 2^(exponent - 7)
        step = torch.exp2(torch.floor(torch.log2(r.double().abs().clamp_min(1e-30))) - 7)
        steps = (d.abs() / step)[ne]
        vec = ne.reshape(-1, ne.shape[-1]).any(dim=1)
        line.append("%s~%s differ %5d of %7d el (%.1e), max %.1f steps, in %3d of %4d vectors" % (
            arm, var, int(ne.sum()), ne.numel(), float(ne.double().mean()), float(steps.max()) if steps.numel() else 0.0,
            int(vec.sum()), vec.numel()))
    sep = float(torch.linalg.vector_norm(refs["x256"] - refs["kernel"]) / torch.linalg.vector_norm(refs["kernel"]))
    on_k = ep.distance(arms["on"]["cells"][key]["o"], refs["kernel"])["rel_l2"]
    print("%-14s %-7s sep %.2e on~kernel %.2e | %s | %s" % (name, kernel, sep, on_k, line[0], line[1]))
