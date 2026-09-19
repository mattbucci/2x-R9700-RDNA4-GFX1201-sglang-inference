#!/usr/bin/env python3
"""Import every `sglang.srt.models` module on the patched tree, CPU-only.

A patch can apply byte-clean and still reference a symbol upstream removed
(3090, v0.5.20: an `EVSDataItem` import broke 90/238 model modules while the
apply/byte gates passed). The replay gate cannot see that; a registry import
can. Run it after every rebase, before any GPU boot:

    HIP_VISIBLE_DEVICES= python scripts/eval/registry_import_preflight.py

The GPU is hidden so this never touches a card that may be serving. The only
import-time device queries (`is_gfx1250_supported` -> `get_device_properties`)
are stubbed to a gfx1201. What still cannot import with no device is Triton's
AMD driver init (`hipGetErrorString`), which some modules trigger at import
through sglang.kernels; those are reported as `driver` and are NOT a patch
verdict -- they import on a real GPU (fleet boots are the receipt). Anything
else (ImportError, AttributeError, NameError, SyntaxError) is a real break and
fails the run. Known upstream-only misses (CUDA-only deps) are listed below.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys

import torch

# Upstream modules that need a dependency this stack never installs.
KNOWN_MISSING_DEPS = {"cutlass"}  # inkling (NVIDIA-only)


class _Props:
    name = "AMD Radeon AI PRO R9700"
    gcnArchName = "gfx1201"
    major, minor = 12, 0
    total_memory = 32 << 30
    multi_processor_count = 64
    warp_size = 32


def main() -> int:
    torch.cuda.get_device_properties = lambda *a, **k: _Props()
    torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
    torch.cuda.get_device_name = lambda *a, **k: _Props.name
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    torch.cuda.current_device = lambda: 0

    import sglang.srt.models as models

    ok, driver, dep, broken = [], [], [], []
    for mod in pkgutil.iter_modules(models.__path__):
        name = f"sglang.srt.models.{mod.name}"
        try:
            importlib.import_module(name)
            ok.append(mod.name)
        except ModuleNotFoundError as e:
            (dep if e.name in KNOWN_MISSING_DEPS else broken).append((mod.name, f"{type(e).__name__}: {e}"))
        except RuntimeError as e:
            if "hipGetErrorString" in str(e) or "No CUDA GPUs" in str(e):
                driver.append(mod.name)
            else:
                broken.append((mod.name, f"RuntimeError: {str(e)[:160]}"))
        except Exception as e:  # ImportError / AttributeError / NameError / SyntaxError ...
            broken.append((mod.name, f"{type(e).__name__}: {str(e)[:160]}"))

    total = len(ok) + len(driver) + len(dep) + len(broken)
    print(f"model modules: {total}  imported={len(ok)}  driver-init-only={len(driver)}  "
          f"known-missing-dep={len(dep)}  BROKEN={len(broken)}")
    if driver:
        print("  driver-init-only (need a real GPU; not a patch verdict):", " ".join(driver))
    for n, why in dep:
        print(f"  known-missing-dep {n}: {why}")
    for n, why in broken:
        print(f"  BROKEN {n}: {why}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
