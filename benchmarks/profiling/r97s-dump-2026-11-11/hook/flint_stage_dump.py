"""Per-stage dump of small Flint prefill forwards (diagnostic, off by default).

Active only when SGLANG_FLINT_STAGE_DUMP names a directory. Every EXTEND
forward of 1..SGLANG_FLINT_STAGE_DUMP_MAX_ROWS rows (default 64) is written as
<dir>/rank<r>/fwd<NNNN>.pt = {"meta": {...}, "stages": {name: cpu tensor}}:

  L<ii>-attn_in  attention input after the norm (the completed residual stream)
  L<ii>-q/-k/-v  inputs of RadixAttention (this rank's heads)
  L<ii>-attn     RadixAttention output (this rank's heads)
  L<ii>-oproj    o_proj output (this rank's partial sum)
  L<ii>-mlp_in   MLP / MoE input (attention sum completed, residual, norm)
  L<ii>-mlp      MLP / MoE output (this rank's partial sum)
  final          the model's output hidden states

SGLANG_FLINT_STAGE_DUMP_LAYERS (comma list of layer indices) and
SGLANG_FLINT_STAGE_DUMP_KINDS (comma list of attn_in, q, k, v, attn, oproj,
mlp_in, mlp, final) restrict the dump; empty means everything (R97-S dumps
q, k, v, attn of four layers at up to 4096 rows).
SGLANG_FLINT_STAGE_DUMP_MAX_FORWARDS caps the forwards written per rank (default 96).
Decode, verify and graph-capture forwards are never armed, so the hooks cost
one Python attribute test there and nothing in a replayed graph.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV_DIR = "SGLANG_FLINT_STAGE_DUMP"
ENV_ROWS = "SGLANG_FLINT_STAGE_DUMP_MAX_ROWS"
ENV_LAYERS = "SGLANG_FLINT_STAGE_DUMP_LAYERS"
ENV_KINDS = "SGLANG_FLINT_STAGE_DUMP_KINDS"
ENV_FORWARDS = "SGLANG_FLINT_STAGE_DUMP_MAX_FORWARDS"
KINDS = ("attn_in", "q", "k", "v", "attn", "oproj", "mlp_in", "mlp", "final")
MAX_FORWARDS = 96


def _first(value):
    while isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    return value


class StageDump:
    def __init__(self, root, rank, max_rows=64, max_forwards=MAX_FORWARDS, kinds=None):
        self.dir = os.path.join(root, "rank%d" % rank)
        os.makedirs(self.dir, exist_ok=True)
        self.rank = rank
        self.max_rows = max_rows
        self.max_forwards = max_forwards
        self.kinds = frozenset(kinds) if kinds else frozenset(KINDS)
        self.armed = False
        self.count = 0
        self.stages = None
        self.meta = None

    def keep(self, name, value):
        if not self.armed:
            return
        value = _first(value)
        if torch.is_tensor(value):
            self.stages[name] = value.detach().to("cpu", copy=True)
        else:
            self.meta["non_tensor"][name] = type(value).__name__

    # model-level hooks -------------------------------------------------
    def begin(self, module, args, kwargs):
        self.armed = False
        batch = None
        for value in list(args) + list(kwargs.values()):
            if hasattr(value, "forward_mode"):
                batch = value
                break
        if batch is None or getattr(batch.forward_mode, "name", "") != "EXTEND":
            return
        ids = args[0] if args else kwargs.get("input_ids")
        if not torch.is_tensor(ids):
            return
        rows = int(ids.shape[0])
        if rows < 1 or rows > self.max_rows or self.count >= self.max_forwards:
            return
        self.armed = True
        self.stages = {}
        self.meta = {
            "forward": self.count,
            "rank": self.rank,
            "pid": os.getpid(),
            "mode": batch.forward_mode.name,
            "rows": rows,
            "input_ids": [int(x) for x in ids.detach().to("cpu").tolist()],
            "extend_prefix_lens": [int(x) for x in (getattr(batch, "extend_prefix_lens_cpu", None) or [])],
            "extend_seq_lens": [int(x) for x in (getattr(batch, "extend_seq_lens_cpu", None) or [])],
            "rids": [str(x) for x in (getattr(batch, "rids", None) or [])],
            "non_tensor": {},
        }

    def end(self, module, args, kwargs, output):
        if not self.armed:
            return
        if "final" in self.kinds:
            self.keep("final", output)
        self.armed = False
        path = os.path.join(self.dir, "fwd%04d.pt" % self.count)
        torch.save({"meta": self.meta, "stages": self.stages}, path + ".tmp")
        os.replace(path + ".tmp", path)
        logger.info(
            "[flint-stage-dump] rank %d forward %d: %d rows, prefix %s, %d stages -> %s",
            self.rank, self.count, self.meta["rows"], self.meta["extend_prefix_lens"],
            len(self.stages), path,
        )
        self.count += 1
        self.stages = None
        self.meta = None

    # stage taps --------------------------------------------------------
    def tap_out(self, name):
        def hook(module, args, output):
            self.keep(name, output)
        return hook

    def tap_in(self, name, key, index):
        def hook(module, args, kwargs):
            if not self.armed:
                return
            if key in kwargs:
                self.keep(name, kwargs[key])
            elif len(args) > index:
                self.keep(name, args[index])
        return hook

    def tap_qkv(self, prefix):
        def hook(module, args, kwargs):
            if not self.armed:
                return
            for index, key in enumerate(("q", "k", "v")):
                if key not in self.kinds:
                    continue
                if key in kwargs:
                    self.keep(prefix + "-" + key, kwargs[key])
                elif len(args) > index:
                    self.keep(prefix + "-" + key, args[index])
        return hook


def _csv(text):
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def install(model, rank, root, max_rows=64, layers=None, kinds=None, max_forwards=MAX_FORWARDS):
    """Hook model.model and the chosen layers (all when layers is empty); returns the StageDump."""
    unknown = sorted(set(kinds or ()) - set(KINDS))
    if unknown:
        raise ValueError("%s: unknown stage kinds %s" % (ENV_KINDS, unknown))
    if max_forwards < 1:
        raise ValueError("%s: %d is not a positive count" % (ENV_FORWARDS, max_forwards))
    dump = StageDump(root, rank, max_rows, max_forwards=max_forwards, kinds=kinds)
    want = frozenset(int(x) for x in layers) if layers else None
    k = dump.kinds
    core = model.model
    core.register_forward_pre_hook(dump.begin, with_kwargs=True)
    core.register_forward_hook(dump.end, with_kwargs=True)
    hooked = 0
    for i, layer in enumerate(core.layers):
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attention is None or mlp is None or (want is not None and i not in want):
            continue
        name = "L%02d" % i
        if "attn_in" in k:
            attention.register_forward_pre_hook(dump.tap_in(name + "-attn_in", "hidden_states", 1), with_kwargs=True)
        if k & {"q", "k", "v"}:
            attention.attn.register_forward_pre_hook(dump.tap_qkv(name), with_kwargs=True)
        if "attn" in k:
            attention.attn.register_forward_hook(dump.tap_out(name + "-attn"))
        if "oproj" in k:
            attention.o_proj.register_forward_hook(dump.tap_out(name + "-oproj"))
        if "mlp_in" in k:
            mlp.register_forward_pre_hook(dump.tap_in(name + "-mlp_in", "hidden_states", 0), with_kwargs=True)
        if "mlp" in k:
            mlp.register_forward_hook(dump.tap_out(name + "-mlp"))
        hooked += 1
    # A marker beside the dumps: the driver's identity check does not depend on the log format.
    with open(os.path.join(dump.dir, "ACTIVE"), "w") as f:
        f.write("pid %d rank %d layers %d max_rows %d\n" % (os.getpid(), rank, hooked, max_rows))
        f.write("kinds %s\n" % ",".join(x for x in KINDS if x in k))
        f.write("max_forwards %d\n" % max_forwards)
    if want is not None and hooked != len(want):
        raise ValueError("%s: %d of %d requested layers exist" % (ENV_LAYERS, hooked, len(want)))
    logger.info(
        "[flint-stage-dump] active: rank %d, %d layers hooked, EXTEND forwards of 1..%d rows -> %s",
        rank, hooked, max_rows, dump.dir,
    )
    return dump


def maybe_install(model, rank):
    """Install the dump when its switch names a directory; None otherwise."""
    root = os.environ.get(ENV_DIR, "").strip()
    if not root:
        return None
    return install(
        model, int(rank), root, int(os.environ.get(ENV_ROWS, "64")),
        layers=_csv(os.environ.get(ENV_LAYERS)), kinds=_csv(os.environ.get(ENV_KINDS)),
        max_forwards=int(os.environ.get(ENV_FORWARDS, str(MAX_FORWARDS))),
    )
