"""Make compressed_tensors' set_forward_quantized tolerate non-bound (functools.partial)
module forwards. Import this before llmcompressor oneshot.

compressed_tensors 0.16.x set_forward_quantized does:
    @wraps(module.forward.__func__)
    def quantized_forward(self, input): ... output = self.__class__.forward(self, input) ...
    module.forward = quantized_forward.__get__(module)

The @wraps(...__func__) is cosmetic (copies name/doc onto quantized_forward), but it
crashes with AttributeError when module.forward is a functools.partial (no __func__) —
which it is for every quantized Linear under transformers 5.x + llmcompressor 0.11.x on
this model (the forwards are wrapped). Since set_forward_quantized REPLACES module.forward
with quantized_forward (which calls self.__class__.forward, not the instance forward), the
partial is discarded either way; we just temporarily expose a real bound method so the
cosmetic @wraps succeeds. Behavior-identical to the original intent, minus the crash.
"""
import compressed_tensors.quantization.lifecycle.forward as _ctf
import compressed_tensors.quantization.lifecycle.initialize as _init

_orig_set_forward_quantized = _ctf.set_forward_quantized


def _safe_set_forward_quantized(module):
    fwd = getattr(module, "forward", None)
    if fwd is not None and not hasattr(fwd, "__func__"):
        try:
            # expose the class-level forward as a real bound method so @wraps(.__func__)
            # works; set_forward_quantized overwrites module.forward immediately after.
            module.forward = module.__class__.forward.__get__(module)
        except Exception:
            pass
    return _orig_set_forward_quantized(module)


_ctf.set_forward_quantized = _safe_set_forward_quantized
_init.set_forward_quantized = _safe_set_forward_quantized


# Same partial-forward bug in compressed_tensors.offload.module.offload_module
# (line ~46: `original_forward_func = module.forward.__func__`), hit when a
# device_map'd model is converted to llmcompressor offloading (from_accelerate) and
# by moe_calibration_context. offload_module CALLS the captured func later as
# `self._original_forward_func(self, *args)`, so we expose the class-level forward as a
# real bound method first — its __func__ is a true unbound function the wrapper can call.
# Patch the source module + every binding that imported it (re-export + moe_context),
# before llmcompressor binds it.
import compressed_tensors.offload.module as _ctom
import compressed_tensors.offload as _cto

_orig_offload_module = _ctom.offload_module


def _safe_offload_module(module, *args, **kwargs):
    fwd = getattr(module, "forward", None)
    if fwd is not None and not hasattr(fwd, "__func__"):
        try:
            module.forward = module.__class__.forward.__get__(module)
        except Exception:
            pass
    return _orig_offload_module(module, *args, **kwargs)


_ctom.offload_module = _safe_offload_module
if hasattr(_cto, "offload_module"):
    _cto.offload_module = _safe_offload_module
