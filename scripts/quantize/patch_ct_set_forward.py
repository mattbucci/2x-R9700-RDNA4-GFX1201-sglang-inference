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
