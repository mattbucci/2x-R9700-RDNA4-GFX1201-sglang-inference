#!/usr/bin/env python3
"""RAD-13: time vllm-radiance's libr4d kernels against ours on the same inputs.

The m1, m16, tiled, quant and topk families pass their fp32 reference checks on gfx1201 with events timing
(receipts: benchmarks/profiling/libr4d-kernel-bench/); --timing graph and the stub families have not run.
The CPU-only parts (struct layout, ABI fingerprint, fragment permutation, scratch sizes) are checked by
--selfcheck against the built library and the vendor layout table.

Bench mode (opens the GPU; never run while a serving benchmark is timed). libr4d.so is third-party code:
run it through scripts/bench/libr4d_sandbox.sh (no network, home hidden, one writable directory):

    W=/data/tmp/rad13-run scripts/bench/libr4d_sandbox.sh \
        /home/letsrtfm/miniforge3/envs/sglang-triton36-v0522/bin/python -I scripts/bench/libr4d_kernel_bench.py \
        --lib /data/vllm-radiance-src/radiance-build/radiance_home/kernels/libr4d.so \
        --out /data/tmp/rad13-run/out/receipt.json [--families m1,m16,tiled,quant,topk] [--device 0] [--shapes qwen38]

CPU self-check (no torch import, no HIP call, no launch; safe inside a sandbox without GPU nodes):

    python3 -I libr4d_kernel_bench.py --selfcheck --lib <libr4d.so> \
        --abi-table <radiance>/tests/abi_layout_table.h --abi-rows <radiance>/abi/rad_abi_layout_rows.h

Calling convention (verified by reading libr4d/r4d.h, r4d_selftest.cpp and abi/*.h at 451b136):
  * plain entries: extern "C", device pointers as `long`, ints, `long stream` cast to hipStream_t
    (0 = HIP's default stream); they return 0 or a small negative int. Dimension order is M, K, N.
  * plugin entries: int launch(const RadArgs*, RadStream) + int64 scratch(const RadArgs*), reached
    through rad_kernel_count()/rad_kernel_at(i); operands are RadTensor rows, parameters RadParam
    rows. rowtopk_bf16 only exists in this form.
  * no module init call; libr4d has no rad_plugin_open.
Method mirrors scripts/bench/bench_fp8_block_gemm_m1.py (weight rotation over > 256 MB of distinct
copies so the 64 MB MALL cache cannot serve the weight) and the vendor's own st_time_us (as many
untimed launches as timed ones first, one event pair around the timed block).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import sys
import time
from ctypes import (CFUNCTYPE, POINTER, Structure, c_char, c_char_p, c_double, c_float, c_int,
                    c_int32, c_int64, c_long, c_longlong, c_uint32, c_uint64, c_void_p)

RAD_ABI_VERSION = 37            # abi/rad_abi.h at 451b136
GROUP = 128                     # weight_block_size [128, 128]; libr4d kBlock
ROTATE_BYTES_MIN = 256 * 2**20  # the MALL trap: rotate more than this per timed pass

# abi/rad_types.h dtype codes
RAD_F32, RAD_F16, RAD_BF16, RAD_F8E4M3 = 1, 2, 3, 4
RAD_I8, RAD_U8, RAD_I32, RAD_I64, RAD_U32 = 8, 9, 11, 12, 13
RAD_P_INT, RAD_P_STR, RAD_P_RANGE, RAD_P_F64 = 1, 2, 3, 4
RAD_PLANE_TILED, RAD_PLANE_TABLE = 0, 1
RAD_STATUS = {0: "RAD_OK", -1: "RAD_E_SHAPE", -2: "RAD_E_DTYPE", -3: "RAD_E_ALIGN",
              -4: "RAD_E_STRIDE", -5: "RAD_E_SCRATCH", -6: "RAD_E_NOMEM", -7: "RAD_E_NOKERNEL",
              -8: "RAD_E_NOSCHEMA", -9: "RAD_E_SCHEMA", -10: "RAD_E_ABI", -11: "RAD_E_DUPLICATE",
              -12: "RAD_E_INVAL", -13: "RAD_E_IO"}


# --------------------------------------------------------------------------- ABI mirror (ctypes)
class RadTensor(Structure):
    _fields_ = [("data", c_void_p), ("dtype", c_uint32), ("rank", c_uint32),
                ("shape", c_int64 * 6), ("stride", c_int64 * 6)]


class RadParam(Structure):
    _fields_ = [("key", c_char_p), ("kind", c_int), ("ival", c_longlong), ("ihi", c_longlong),
                ("sval", c_char_p), ("dval", c_double)]


class RadArgs(Structure):
    _fields_ = [("t", POINTER(RadTensor)), ("n_t", c_int), ("p", POINTER(RadParam)),
                ("n_p", c_int), ("scratch", c_void_p), ("scratch_bytes", c_int64),
                ("instance", c_void_p), ("rank", c_int), ("world_size", c_int),
                ("kv_used_max_h", c_int32), ("wgp_count", c_int32)]


class RadConstraint(Structure):
    _fields_ = [("key", c_char_p), ("op", c_int), ("ival", c_longlong), ("sval", c_char_p)]


class RadTunable(Structure):
    _fields_ = [("key", c_char_p), ("values", POINTER(c_int64)), ("n_values", c_int),
                ("deflt", c_int64), ("doc", c_char_p)]


class RadLayout(Structure):
    _fields_ = [("tag", c_char_p), ("dtype", c_uint32), ("rank", c_uint32),
                ("shape", c_int64 * 6), ("align", c_int64), ("bytes", c_int64)]


class RadPluginInfo(Structure):
    _fields_ = [("kind", c_uint32), ("name", c_char_p), ("version", c_char_p),
                ("description", c_char_p), ("build_target", c_char_p)]


class RadEncPlane(Structure):
    _fields_ = [("role", c_char * 24), ("dtype", c_uint32), ("kind", c_uint32),
                ("block", c_int64 * 2), ("extent", c_int64 * 2)]


class RadEncoding(Structure):
    _fields_ = [("scheme", c_char * 24), ("transform", c_char * 24), ("n_planes", c_int32),
                ("reserved", c_int32), ("plane", RadEncPlane * 8)]


class RadKernelInfo(Structure):
    _fields_ = [("name", c_char_p), ("op", c_char_p), ("family", c_char_p),
                ("computes", c_char_p), ("shape", c_char_p), ("dtypes", c_char_p),
                ("domain", c_int), ("priority", c_int),
                ("constraints", POINTER(RadConstraint)), ("n_constraints", c_int),
                ("tunables", POINTER(RadTunable)), ("n_tunables", c_int),
                ("tune_valid", c_void_p), ("init", c_void_p), ("fini", c_void_p),
                ("launch", c_void_p), ("scratch", c_void_p), ("layout", c_void_p),
                ("relayout", c_void_p), ("opd_shape", c_void_p), ("unrelayout", c_void_p),
                ("replaces", c_char_p), ("capabilities", c_uint64)]


MIRRORS = [RadTensor, RadParam, RadArgs, RadConstraint, RadTunable, RadLayout, RadPluginInfo,
           RadEncPlane, RadEncoding, RadKernelInfo]

LAUNCH_FN = CFUNCTYPE(c_int, POINTER(RadArgs), c_void_p)     # RadLaunchFn(const RadArgs*, RadStream)
SCRATCH_FN = CFUNCTYPE(c_int64, POINTER(RadArgs))            # RadScratchFn(const RadArgs*)

_I, _L, _F = c_int, c_long, c_float
# name: (restype, argtypes). Every argument is passed: the C++ defaults in r4d.h do not exist in C.
PROTOS = {
    # c[1][N] bf16 = a[1][K] bf16 @ dequant(wq[N][K] E4M3 row-major, ws[N/128][K/128] bf16)^T
    "r4d_gemm_fp8a16_nt_m1": (_I, [_L, _L, _L, _L, _I, _I, _I, _I, _I, _I, _L]),
    #   (a, wq, ws, c, M, K, N, w_ld, sw_ld, rb, stream)
    "r4d_gemm_fp8a16_nt_mt": (_I, [_L, _L, _L, _L, _I, _I, _I, _I, _I, _I, _I, _L]),
    #   (a, wq, ws, c, M, K, N, a_ld, c_ld, w_ld, sw_ld, stream), M <= 8
    "r4d_gemm_fp8a8_nt_m16": (_I, [_L, _L, _L, _L, _L, _I, _I, _I, _I, _I, _I, _I, _L, _L, _L]),
    #   (a, ascale, wq(frag64), ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, scratch, scratch_bytes, stream)
    "r4d_gemm_fp8a8_nt_m16_scratch": (_L, [_I, _I, _I]),          # (M, K, N) -> bytes, host only
    "r4d_gemm_fp8a8_nt_m16_max_m": (_I, []),
    "r4d_gemm_fp8a8_tiled": (_I, [_L, _L, _L, _L, _L, _I, _I, _I, _I, _I, _I, _I, _L]),
    #   (a, ascale, wq(frag64), ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream)
    "r4d_quant_act_fp8": (_I, [_L, _L, _L, _I, _I, _I, _I, _I, _L]),
    #   (a, q, s, M, K, x_ld, q_ld, s_ld, stream)
    "r4d_rmsnorm_quant_fp8": (_I, [_L, _L, _L, _L, _L, _L, _I, _I, _F, _F, _I,
                                   _I, _I, _I, _I, _I, _L]),
    #   (x, res, w, y, q, s, M, n, eps, wadd, w_f32, x_ld, r_ld, y_ld, q_ld, s_ld, stream)
    "r4d_gated_quant_fp8": (_I, [_L, _L, _L, _L, _I, _I, _I, _I, _I, _I, _I, _L]),
    #   (gu, y, q, s, M, n, mode, gu_ld, y_ld, q_ld, s_ld, stream)
    "r4d_rmsnorm_bf16": (_I, [_L, _L, _L, _I, _I, _F, _F, _I, _I, _I, _L]),
    #   (x, w, y, M, n, eps, wadd, w_f32, x_ld, y_ld, stream)
    "r4d_silu_mul_bf16": (_I, [_L, _L, _I, _I, _I, _I, _L]),      # (gu, y, M, n, gu_ld, y_ld, stream)
    "r4d_logits_gemm_bf16": (_I, [_L, _L, _L, _I, _I, _I, _I, _I, _L]),
    #   (a, w, c, M, K, N, a_ld, c_ld, stream), c is f32
    "r4d_rowtopk_bf16_chunks": (_I, [_I, _I]),                    # (N, NCH) host only
    "r4d_gemm_fp8a16_nt_m1_block": (_I, []),
    "r4d_gemm_fp8a16_nt_m1_max_m": (_I, []),
    "r4d_gemm_fp8a8_block": (_I, []),
    "r4d_quant_act_fp8_group": (_I, []),
    "rad_plugin_abi_version": (c_uint32, []),
    "rad_plugin_abi_layout": (c_uint64, []),
    "rad_plugin_info": (POINTER(RadPluginInfo), []),
    "rad_kernel_count": (_I, []),
    "rad_kernel_at": (POINTER(RadKernelInfo), [_I]),
    # host-side weight packer for the fp8a8 rows (r4d_fp8_layout.cpp)
    "r4d_rad_relayout_fp8_block": (_I, [POINTER(RadParam), _I, _I, POINTER(RadEncoding),
                                        POINTER(c_int), POINTER(RadTensor), _I, c_void_p, c_int64]),
}


def load_lib(path):
    """dlopen libr4d and attach prototypes. In bench mode torch must already be imported, so the
    loader resolves libr4d's NEEDED libamdhip64.so.7 to the copy torch mapped (one HIP runtime)."""
    lib = ctypes.CDLL(path)
    missing = []
    for name, (res, args) in PROTOS.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        fn.restype, fn.argtypes = res, args
    return lib, missing


def kernel_rows(lib):
    """{row name: RadKernelInfo} from the plugin's table (host-only calls)."""
    rows = {}
    for i in range(lib.rad_kernel_count()):
        info = lib.rad_kernel_at(i).contents
        rows[info.name.decode()] = info
    return rows


# --------------------------------------------------------------------------- plugin-ABI packing
_KEYS = {}


def _b(s):
    """Interned bytes: RadParam holds bare char pointers, so the keys must outlive every call."""
    return _KEYS.setdefault(s, s.encode())


def rad_tensor(data, dtype, shape):
    """The selftest's T(): contiguous strides in ELEMENTS (stride[last] = 1)."""
    t = RadTensor()
    t.data, t.dtype, t.rank = data, dtype, len(shape)
    st = 1
    for i in range(len(shape) - 1, -1, -1):
        t.shape[i], t.stride[i] = shape[i], st
        st *= shape[i]
    return t


def rad_absent(dtype=RAD_U8):
    """The selftest's absent(): data null and rank 0 (r4d_opt keys on the pointer)."""
    t = RadTensor()
    t.dtype = dtype
    return t


def PI(key, v):
    p = RadParam()
    p.key, p.kind, p.ival = _b(key), RAD_P_INT, int(v)
    return p


def PS(key, v):
    p = RadParam()
    p.key, p.kind, p.sval = _b(key), RAD_P_STR, _b(v)
    return p


def PF(key, v):
    p = RadParam()
    p.key, p.kind, p.dval = _b(key), RAD_P_F64, float(v)
    return p


class PackedArgs:
    """A RadArgs plus the arrays it points at, kept alive together."""

    def __init__(self, tensors, params, scratch=0, scratch_bytes=0, world_size=1, rank=0):
        self.t = (RadTensor * max(1, len(tensors)))(*tensors)
        self.p = (RadParam * max(1, len(params)))(*params)
        self.args = RadArgs()
        self.args.t = ctypes.cast(self.t, POINTER(RadTensor))
        self.args.n_t = len(tensors)
        self.args.p = ctypes.cast(self.p, POINTER(RadParam))
        self.args.n_p = len(params)
        self.args.scratch = scratch or None
        self.args.scratch_bytes = scratch_bytes
        self.args.rank, self.args.world_size = rank, world_size

    def ref(self):
        return ctypes.byref(self.args)


def fp8_affine_encoding():
    """RadEncoding of a block-FP8 linear as the checkpoint ships it: E4M3 codes + bf16 scale per
    128x128 tile (r4d_fp8_layout.cpp `affine(e, RAD_F8E4M3, RAD_BF16, 128, 128)`)."""
    e = RadEncoding()
    e.scheme, e.transform, e.n_planes = b"affine", b"", 2
    e.plane[0].role, e.plane[0].dtype, e.plane[0].kind = b"codes", RAD_F8E4M3, RAD_PLANE_TILED
    e.plane[0].block[0] = e.plane[0].block[1] = 1
    e.plane[1].role, e.plane[1].dtype, e.plane[1].kind = b"scale", RAD_BF16, RAD_PLANE_TILED
    e.plane[1].block[0] = e.plane[1].block[1] = GROUP
    return e


# --------------------------------------------------------------------------- layout table + hash
_ROW_RE = re.compile(r'\{\s*"(\w+)"\s*,\s*"(\w*)"\s*,\s*(\d+)\s*,\s*(\d+)\s*\}')


def parse_layout_table(path):
    """tests/abi_layout_table.h -> {(type, field): (offset, size)}; field "" is the struct."""
    table = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for m in _ROW_RE.finditer(f.read()):
            table[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    return table


def check_mirrors(table):
    """Every ctypes mirror against the vendor table: sizeof, each field's offset and size, and no
    field of the vendor struct missing from the mirror."""
    bad = []
    for cls in MIRRORS:
        name = cls.__name__
        if (name, "") not in table:
            bad.append(f"{name}: not in the layout table")
            continue
        if ctypes.sizeof(cls) != table[(name, "")][1]:
            bad.append(f"{name}: sizeof {ctypes.sizeof(cls)} != {table[(name, '')][1]}")
        mine = {f[0] for f in cls._fields_}
        theirs = {f for (t, f) in table if t == name and f}
        if mine != theirs:
            bad.append(f"{name}: fields differ, mirror-only {sorted(mine - theirs)}, "
                       f"table-only {sorted(theirs - mine)}")
        for fname in sorted(mine & theirs):
            d = getattr(cls, fname)
            if (d.offset, d.size) != table[(name, fname)]:
                bad.append(f"{name}.{fname}: ({d.offset}, {d.size}) != {table[(name, fname)]}")
    return bad


def abi_fingerprint(table, rows_path, version=RAD_ABI_VERSION):
    """abi/rad_abi_layout.h: FNV-1a over the ABI version, then sizeof per ABI_STRUCT and
    (offsetof, sizeof) per ABI_FIELD in rad_abi_layout_rows.h order, eight bytes a value."""
    mask = (1 << 64) - 1

    def mix(h, v):
        for i in range(8):
            h ^= (v >> (8 * i)) & 0xFF
            h = (h * 0x100000001B3) & mask
        return h

    h = mix(0xCBF29CE484222325, version)
    n_struct = n_field = 0
    with open(rows_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*ABI_STRUCT\((\w+)\)", line)
            if m:
                h = mix(h, table[(m.group(1), "")][1])
                n_struct += 1
                continue
            m = re.match(r"\s*ABI_FIELD\((\w+),\s*(\w+)\)", line)
            if m:
                off, size = table[(m.group(1), m.group(2))]
                h = mix(mix(h, off), size)
                n_field += 1
    return h, n_struct, n_field


# --------------------------------------------------------------------------- fp8a8 weight layout
def frag64_offsets(N, K):
    """Source byte offset of each 8-byte group of the "r4d.fp8.w.frag64" plane, in destination
    order (r4d_fp8_layout.cpp frag_walk). N % 16 == 0 and K % 64 == 0."""
    out = []
    for rt in range(N // 16):
        for ks in range(K // 64):
            for lane in range(32):
                for t in range(4):
                    out.append((rt * 16 + (lane & 15)) * K + ks * 64 + t * 16 + (lane >> 4) * 8)
    return out


def frag64_pack_bytes(src, N, K):
    return b"".join(src[o:o + 8] for o in frag64_offsets(N, K))


def frag64_torch(w):
    """The same permutation on a [N, K] E4M3 (or uint8) tensor, on its device. Axes of the view are
    (row tile, row in tile, k supertile, t, half, byte); destination order is
    (row tile, k supertile, half, row in tile, t, byte)."""
    import torch
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0, (n, k)
    u = w.contiguous().view(torch.uint8).view(n // 16, 16, k // 64, 4, 2, 8)
    return u.permute(0, 2, 4, 1, 3, 5).contiguous().view(n, k)


def vendor_frag64(lib, src, N, K):
    """The vendor's host packer on a host buffer (bytes in, bytes out). Returns (rc, bytes)."""
    buf = (c_char * (N * K)).from_buffer_copy(src)
    dst = (c_char * (N * K))()
    params = (RadParam * 2)(PI("N", N), PI("K", K))
    enc = fp8_affine_encoding()
    sel = (c_int * 1)(0)                                   # the "codes" plane
    pl = (RadTensor * 1)(rad_tensor(ctypes.addressof(buf), RAD_F8E4M3, [N, K]))
    rc = lib.r4d_rad_relayout_fp8_block(params, 2, 2, ctypes.byref(enc), sel, pl, 1,
                                        ctypes.cast(dst, c_void_p), N * K)
    return rc, bytes(dst)


# The narrow tile's split rule (r4d_gemm_fp8a8.hip narrow_mw / narrow_fk / narrow_ksplit),
# re-derived here so the receipt can say how many K partial planes a shape sums.
def narrow_ksplit(N, K, M, target=128, thin_cap=64):
    mw = 4
    if M > 16 and N % 128 == 0:
        mw = 8
    if M > 32 and N >= 2048 and N % 256 == 0:
        mw = 16
    gx = (N + mw * 16 - 1) // (mw * 16)
    if K < 512 or gx * mw >= target:
        return 1
    want = (target + gx * mw - 1) // (gx * mw)
    ckw = max(GROUP, ((K // want) + GROUP - 1) // GROUP * GROUP)
    k2 = (K + ckw - 1) // ckw
    if k2 > thin_cap and N * M < 8192:
        k2 = thin_cap
    if k2 < 2:
        return 1
    fk = 256 if K % 256 == 0 else 128
    ck = ((K + k2 - 1) // k2 + fk - 1) // fk * fk
    k2 = (K + ck - 1) // ck
    return k2 if k2 >= 2 else 1


# Per-rank block-FP8 linears of Qwen3.8-27B-FP8 at TP=2: (name, N, K, launches per decode step).
# "qwen38" is how SGLang fuses and splits them (DeltaNet replicated per rank); "qwen38-radiance"
# is the TP-split DeltaNet pair the vendor README times.
SHAPES = {
    "qwen38": [("mlp.gate_up_proj", 17408, 5120, 64), ("mlp.down_proj", 5120, 8704, 64),
               ("linear_attn.in_proj_qkvz", 16384, 5120, 48),
               ("linear_attn.out_proj", 5120, 6144, 48),
               ("self_attn.qkv_proj", 7168, 5120, 16), ("self_attn.o_proj", 5120, 3072, 16)],
    "qwen38-radiance": [("mlp.gate_up_proj", 17408, 5120, 64), ("mlp.down_proj", 5120, 8704, 64),
                        ("linear_attn.in_proj_qkvz/2", 8192, 5120, 48),
                        ("linear_attn.out_proj/2", 5120, 3072, 48),
                        ("self_attn.qkv_proj", 7168, 5120, 16),
                        ("self_attn.o_proj", 5120, 3072, 16)],
}

def rowtopk_chunks(lib, n, nch=0):
    """Stage-1 chunk count. r4d_rowtopk_bf16_chunks(N, NCH) divides by NCH (SIGFPE at 0): the
    tunable's 0 default is resolved to one chunk per 2048 columns by the plugin shim, not there."""
    nch = nch if nch > 0 else max(1, (n + 2047) // 2048)
    return lib.r4d_rowtopk_bf16_chunks(n, nch)


NEEDED_ROWS = ["gemm_fp8a16_nt_m1", "gemm_fp8a8_nt_m16", "gemm_fp8a8_tiled", "quant_act_fp8",
               "rmsnorm_quant_fp8", "gated_quant_fp8", "rowtopk_bf16", "logits_gemm_bf16",
               "logits_gemm_fp8"]


# --------------------------------------------------------------------------- CPU self-check
def selfcheck(a):
    """Host-only: struct layout, ABI fingerprint, row table, weight packer, scratch sizers.
    Never calls a launch hook or a HIP function, and never imports torch."""
    res = {"mode": "selfcheck", "ok": True, "checks": []}

    def check(name, ok, detail=None):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        res["ok"] = res["ok"] and bool(ok)

    tables = {}
    for path in a.abi_table or []:
        table = parse_layout_table(path)
        tables[path] = table
        bad = check_mirrors(table)
        n_fields = sum(len(c._fields_) for c in MIRRORS)
        check(f"ctypes mirrors == {os.path.basename(path)}", not bad,
              bad or f"{len(MIRRORS)} structs, {n_fields} fields, sizes "
              + ", ".join(f"{c.__name__} {ctypes.sizeof(c)}" for c in MIRRORS))
    if not a.lib:
        return res
    lib, missing = load_lib(a.lib)
    check("every prototype symbol is exported", not missing, missing or len(PROTOS))
    if missing:
        return res
    ver = lib.rad_plugin_abi_version()
    check("rad_plugin_abi_version == 37", ver == RAD_ABI_VERSION, ver)
    info = lib.rad_plugin_info().contents
    res["plugin"] = {k: (getattr(info, k) or b"").decode() for k in
                     ("name", "version", "description", "build_target")}
    if tables and a.abi_rows:
        # The vendor's snapshot table can lag its headers (at 451b136 it misses
        # RadEncoderDecl.bidirectional), so a table that cannot reproduce the fingerprint is
        # reported, and the check passes when at least one table does.
        got, fps = lib.rad_plugin_abi_layout(), []
        for path, table in tables.items():
            try:
                h, ns, nf = abi_fingerprint(table, a.abi_rows)
                fps.append({"table": path, "computed": hex(h), "match": h == got,
                            "structs": ns, "fields": nf})
            except KeyError as e:
                fps.append({"table": path, "match": False, "missing_row": list(e.args[0])})
        check("fingerprint(table, rows order) == rad_plugin_abi_layout()",
              any(f["match"] for f in fps), {"library": hex(got), "tables": fps})
    rows = kernel_rows(lib)
    res["n_rows"] = len(rows)
    absent = [r for r in NEEDED_ROWS if r not in rows]
    check("needed rows are in the plugin table", not absent, absent or NEEDED_ROWS)
    res["rows"] = {}
    for name in NEEDED_ROWS:
        if name not in rows:
            continue
        r = rows[name]
        tun = []
        for i in range(r.n_tunables):
            t = r.tunables[i]
            tun.append({"key": t.key.decode(), "default": t.deflt,
                        "values": [t.values[j] for j in range(t.n_values)]})
        res["rows"][name] = {"op": r.op.decode(), "family": r.family.decode(),
                             "priority": r.priority, "has_scratch": bool(r.scratch),
                             "has_layout": bool(r.layout), "has_relayout": bool(r.relayout),
                             "has_init": bool(r.init), "tunables": tun}
    res["constants"] = {
        "r4d_gemm_fp8a16_nt_m1_block": lib.r4d_gemm_fp8a16_nt_m1_block(),
        "r4d_gemm_fp8a16_nt_m1_max_m": lib.r4d_gemm_fp8a16_nt_m1_max_m(),
        "r4d_gemm_fp8a8_block": lib.r4d_gemm_fp8a8_block(),
        "r4d_gemm_fp8a8_nt_m16_max_m": lib.r4d_gemm_fp8a8_nt_m16_max_m(),
        "r4d_quant_act_fp8_group": lib.r4d_quant_act_fp8_group()}
    check("block sizes are 128", res["constants"]["r4d_gemm_fp8a8_block"] == GROUP
          and res["constants"]["r4d_gemm_fp8a16_nt_m1_block"] == GROUP
          and res["constants"]["r4d_quant_act_fp8_group"] == GROUP, res["constants"])

    # The fp8a8 weight packer: our permutation against the vendor's host hook.
    frag = []
    for n, k in ((16, 64), (32, 128), (48, 192), (128, 256)):
        src = os.urandom(n * k)
        rc, dst = vendor_frag64(lib, src, n, k)
        ok = rc == 0 and dst == frag64_pack_bytes(src, n, k)
        np_ok = None
        try:
            import numpy as np
            np_ok = (np.frombuffer(src, np.uint8).reshape(n // 16, 16, k // 64, 4, 2, 8)
                     .transpose(0, 2, 4, 1, 3, 5).tobytes() == dst)
        except ImportError:
            pass
        frag.append({"N": n, "K": k, "rc": rc, "walk_equal": ok, "permute_formula_equal": np_ok,
                     "identity": dst == src})
    check("frag64: frag_walk and the view/permute formula == r4d_rad_relayout_fp8_block",
          all(f["walk_equal"] and f["permute_formula_equal"] is not False for f in frag), frag)

    # Split-K scratch of the narrow tile: plain sizer, plugin scratch hook, our re-derivation.
    scr = []
    hook = SCRATCH_FN(rows["gemm_fp8a8_nt_m16"].scratch) if "gemm_fp8a8_nt_m16" in rows else None
    for name, n, k, _ in SHAPES["qwen38"] + SHAPES["qwen38-radiance"][2:4]:
        for m in (1, 8, 16, 64):
            plain = lib.r4d_gemm_fp8a8_nt_m16_scratch(m, k, n)
            ks = narrow_ksplit(n, k, m)
            mine = ks * m * n * 4 if ks > 1 else 0
            viahook = None
            if hook:
                pa = PackedArgs([], [PI("M", m), PI("N", n), PI("K", k), PI("group", GROUP),
                                     PS("dtype", "fp8a8")])
                viahook = hook(pa.ref())
            scr.append({"linear": name, "N": n, "K": k, "M": m, "bytes": plain, "ksplit": ks,
                        "rederived": mine, "plugin_hook": viahook})
    check("nt_m16 scratch: plain sizer == plugin hook (RadArgs packing) == re-derived rule",
          all(s["bytes"] == s["rederived"] and s["plugin_hook"] in (None, s["bytes"])
              for s in scr), scr)

    # rowtopk: chunk count and the plugin scratch hook over real host operands.
    topk = []
    hook = SCRATCH_FN(rows["rowtopk_bf16"].scratch) if "rowtopk_bf16" in rows else None
    for n in (15520, 124160):
        for m in (1, 8, 64):
            r = 32
            nch = rowtopk_chunks(lib, n)
            viahook = None
            if hook:
                x = (c_char * (m * n * 2))()
                idx = (c_char * (m * r * 4))()
                val = (c_char * (m * r * 4))()
                pa = PackedArgs([rad_tensor(ctypes.addressof(x), RAD_BF16, [m, n]),
                                 rad_tensor(ctypes.addressof(idx), RAD_I32, [m, r]),
                                 rad_tensor(ctypes.addressof(val), RAD_F32, [m, r])],
                                [PI("R", r), PI("M", m), PI("N", n)])
                viahook = hook(pa.ref())
            topk.append({"N": n, "M": m, "R": r, "chunks": nch, "expect": m * nch * r * 8,
                         "plugin_hook": viahook})
    check("rowtopk scratch hook == M * chunks * R * 8",
          all(t["plugin_hook"] in (None, t["expect"]) for t in topk), topk)
    return res


# =========================================================================== bench mode (GPU)
# Each launch wrapper raises on a non-zero return, and every family checks its output against an
# fp32 PyTorch reference before it is timed, so a wrong assumption (weight layout, pitch unit,
# stream handle) fails loudly instead of timing garbage.
def hip_runtime_maps():
    """Distinct libamdhip64 images mapped into this process. More than one means libr4d bound a
    second HIP runtime, whose launches cannot see torch's device memory."""
    seen = set()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libamdhip64" in line:
                seen.add(line.split()[-1])
    return sorted(seen)


def vram_used_gb():
    """Per-card VRAM in use from sysfs (a file read; no device is opened)."""
    import glob
    out = {}
    for p in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        try:
            with open(p) as f:
                out[p.split("/")[4]] = int(f.read()) / 2**30
        except (OSError, ValueError):
            pass
    return out


class Ctx:
    def __init__(self, a):
        # Pick the card before HIP initialises, so torch and libr4d agree that it is device 0:
        # a plain entry launched on the null stream goes to the thread's current HIP device.
        if a.device is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        self.torch, self.a = torch, a
        self.dev = torch.device("cuda:0")
        torch.zeros(1, device=self.dev)            # HIP is up through torch before libr4d loads
        self.lib, missing = load_lib(a.lib)
        if missing:
            raise SystemExit(f"libr4d lacks {missing}")
        self.rows = kernel_rows(self.lib)
        self.hip_maps = hip_runtime_maps()
        if len(self.hip_maps) != 1 and not a.allow_two_hip_runtimes:
            raise SystemExit(f"expected one libamdhip64 image, found {self.hip_maps}")
        self.matmul = self.quant = None
        self.triton_error = None
        if not a.no_triton:
            try:
                sys.path.insert(0, a.sglang_python)
                from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
                from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
                self.matmul, self.quant = w8a8_block_fp8_matmul_triton, per_token_group_quant_fp8
            except Exception as e:                 # vendor-only run still produces a receipt
                self.triton_error = repr(e)

    def stream(self):
        """hipStream_t of torch's current stream as an integer (0 = HIP's default stream)."""
        return int(self.torch.cuda.current_stream().cuda_stream or 0)

    def gen(self, seed):
        return self.torch.Generator(device=self.dev).manual_seed(seed)


def ptr(t):
    assert t.is_contiguous()
    return t.data_ptr()


def chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} returned {rc} ({RAD_STATUS.get(rc, 'see abi-notes.md')})")


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def time_events(ctx, launch, ncopies, iters, repeats):
    """The vendor's st_time_us under torch events: touch every copy, then per repeat `iters`
    untimed launches (the clocks ramp up from idle) and one event pair around `iters` timed
    ones. launch(i, stream) uses copy i. The output pointer never changes: the narrow tile keys a
    lazily allocated counter block on (stream, output)."""
    torch = ctx.torch
    s = ctx.stream()
    for i in range(ncopies):
        launch(i, s)
    torch.cuda.synchronize()
    us, host = [], []
    for _ in range(repeats):
        for i in range(iters):
            launch(i % ncopies, s)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        t0 = time.perf_counter()
        for i in range(iters):
            launch(i % ncopies, s)
        host.append((time.perf_counter() - t0) * 1e6 / iters)
        e1.record()
        e1.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3 / iters)
    return {"timing": "events", "us_median": _median(us), "us_min": min(us),
            "host_enqueue_us_median": _median(host), "iters": iters, "repeats": repeats,
            "copies": ncopies,
            # the GPU waits on Python: the number is a launcher cost, not a kernel cost
            "host_bound": _median(host) >= 0.8 * _median(us)}


def time_graph(ctx, launch, ncopies, replays, repeats):
    """bench_fp8_block_gemm_m1.py's method: one HIP graph holding a launch per copy, replayed.
    UNVERIFIED for libr4d launches (hipLaunchKernel on torch's capture stream); the narrow tile's
    split counters allocate and synchronise on first use, which a capture cannot hold."""
    torch = ctx.torch
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s = ctx.stream()
            for i in range(ncopies):
                launch(i, s)
        g.replay()
        torch.cuda.synchronize()
        us = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(replays):
                g.replay()
            torch.cuda.synchronize()
            us.append((time.perf_counter() - t0) * 1e6 / (replays * ncopies))
        return {"timing": "graph", "us_median": _median(us), "us_min": min(us),
                "replays": replays, "repeats": repeats, "copies": ncopies}
    except Exception as e:
        return {"timing": "graph", "error": repr(e)}


def timed(ctx, launch, ncopies, iters):
    a = ctx.a
    if a.timing == "graph":
        r = time_graph(ctx, launch, ncopies, a.graph_replays, a.repeats)
        if "error" not in r:
            return r
        fb = time_events(ctx, launch, ncopies, iters, a.repeats)
        fb["graph_error"] = r["error"]
        return fb
    return time_events(ctx, launch, ncopies, iters, a.repeats)


# --------------------------------------------------------------------------- references, accuracy
def bf16_ord(torch, t):
    """bf16 bit patterns as ordered integers, so a difference counts representable values."""
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def accuracy(torch, got, ref):
    """got (bf16 or f32) against an fp32 reference: absolute error, error relative to the
    reference's largest magnitude, cosine, and for bf16 the ULP distance to round(ref)."""
    g = got.float()
    out = {"finite": bool(torch.isfinite(g).all()),
           "max_abs_err": float((g - ref).abs().max()),
           "max_abs_ref": float(ref.abs().max()),
           "cosine": float(torch.nn.functional.cosine_similarity(
               g.flatten().double(), ref.flatten().double(), dim=0))}
    out["rel_max_err"] = out["max_abs_err"] / max(out["max_abs_ref"], 1e-30)
    if got.dtype == torch.bfloat16:
        d = (bf16_ord(torch, got) - bf16_ord(torch, ref.to(torch.bfloat16))).abs()
        n = d.numel()
        out["ulp_max"] = int(d.max())
        out["ulp_hist"] = {"0": int((d == 0).sum()) / n, "1": int((d == 1).sum()) / n,
                           "2": int((d == 2).sum()) / n, ">2": int((d > 2).sum()) / n}
    return out


def ulp_between(torch, a, b):
    d = (bf16_ord(torch, a) - bf16_ord(torch, b)).abs()
    return {"ulp_max": int(d.max()), "bit_identical": bool((d == 0).all()),
            "frac_equal": int((d == 0).sum()) / d.numel()}


def ref_quant(torch, x):
    """libr4d's stated rule (r4d.h): per (row, 128 columns) scale = amax/448, a dequant
    multiplier; an all-zero block gets scale 1; codes are x/scale clamped to +-448 in E4M3."""
    m, k = x.shape
    xf = x.float().view(m, k // GROUP, GROUP)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return q.contiguous(), scale.view(m, k // GROUP).contiguous()


def dequant_act(q, s):
    return q.float() * s.float().repeat_interleave(GROUP, dim=1)


def dequant_weight(w, ws):
    return w.float() * ws.float().repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)


def quant_agreement(torch, qa, sa, qb, sb):
    """Two (codes, scale) pairs for one activation: equal codes, scale error, dequant error."""
    da, db = dequant_act(qa, sa), dequant_act(qb, sb)
    return {"codes_equal_frac": float((qa.view(torch.uint8) == qb.view(torch.uint8)).float().mean()),
            "scale_max_rel": float(((sa.float() - sb.float()).abs()
                                    / sb.float().abs().clamp_min(1e-30)).max()),
            "dequant_max_abs_diff": float((da - db).abs().max()),
            "dequant_max_abs": float(db.abs().max())}


def make_weight(ctx, n, k, seed):
    """A block-FP8 linear in the checkpoint's layout: E4M3 [N, K] row-major and a bf16
    weight_scale_inv [N/128, K/128] (same value distribution as bench_fp8_block_gemm_m1.py)."""
    torch, g = ctx.torch, ctx.gen(seed)
    w = (torch.randn(n, k, device=ctx.dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    ws = (torch.rand(n // GROUP, k // GROUP, device=ctx.dev, generator=g) * 0.01
          + 0.001).to(torch.bfloat16)
    return w, ws


def n_copies(ctx, nbytes):
    return max(4, math.ceil(ctx.a.rotate_mb * 2**20 / nbytes))


def gbps(n, k, us):
    return n * k / us * 1e-3 if us else None


def shapes_for(a):
    if a.shape:
        return [(f"custom{i}", int(s.split(",")[0]), int(s.split(",")[1]), 0)
                for i, s in enumerate(a.shape)]
    return SHAPES[a.shapes]


# --------------------------------------------------------------------------- (a) M = 1 decode GEMM
def fam_m1(ctx):
    """r4d_gemm_fp8a16_nt_m1: bf16 activation read directly, checkpoint-layout weight, bf16 out,
    against our decode path: per_token_group_quant_fp8 launch + w8a8_block_fp8_matmul_triton."""
    torch, lib, a = ctx.torch, ctx.lib, ctx.a
    rows = []
    for name, n, k, per_step in shapes_for(a):
        w, ws = make_weight(ctx, n, k, seed=n * 131 + k)
        copies = [w] + [w.clone() for _ in range(n_copies(ctx, n * k) - 1)]
        x = torch.randn(1, k, device=ctx.dev, generator=ctx.gen(7)).to(torch.bfloat16)
        wd = dequant_weight(w, ws)
        ref = x.float() @ wd.T
        y = torch.zeros(1, n, device=ctx.dev, dtype=torch.bfloat16)
        wp = [ptr(c) for c in copies]
        xp, wsp, yp = ptr(x), ptr(ws), ptr(y)
        skb, rb = k // GROUP, a.rb

        def vendor(i, s):
            chk(lib.r4d_gemm_fp8a16_nt_m1(xp, wp[i], wsp, yp, 1, k, n, k, skb, rb, s),
                "r4d_gemm_fp8a16_nt_m1")

        row = {"linear": name, "N": n, "K": k, "M": 1, "launches_per_decode_step": per_step,
               "weight_bytes": n * k, "copies": len(copies), "rb": rb}
        vendor(0, ctx.stream())
        torch.cuda.synchronize()
        row["vendor_accuracy"] = accuracy(torch, y, ref)
        y_vendor = y.clone()
        vendor(len(copies) - 1, ctx.stream())
        torch.cuda.synchronize()
        row["vendor_last_copy_identical"] = bool((y == y_vendor).all())
        row["vendor"] = timed(ctx, vendor, len(copies), a.iters)
        row["vendor"]["gbps"] = gbps(n, k, row["vendor"]["us_median"])
        if ctx.matmul is not None:
            bs32 = ws.float().contiguous()
            q, xs = ctx.quant(x, GROUP, column_major_scales=False)
            ours_y = ctx.matmul(q, w, xs, bs32, [GROUP, GROUP], output_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            row["ours_accuracy_vs_unquantised_ref"] = accuracy(torch, ours_y, ref)
            row["ours_accuracy_vs_own_quantised_ref"] = accuracy(
                torch, ours_y, dequant_act(q, xs) @ wd.T)
            row["vendor_vs_ours"] = ulp_between(torch, y_vendor, ours_y)

            def ours_gemm(i, s, q=q, xs=xs, copies=copies):     # bound now: the names are deleted below
                ctx.matmul(q, copies[i], xs, bs32, [GROUP, GROUP], output_dtype=torch.bfloat16)

            def ours_quant_gemm(i, s, copies=copies):
                qi, si = ctx.quant(x, GROUP, column_major_scales=False)
                ctx.matmul(qi, copies[i], si, bs32, [GROUP, GROUP], output_dtype=torch.bfloat16)

            row["ours_gemm"] = timed(ctx, ours_gemm, len(copies), a.iters)
            row["ours_quant_gemm"] = timed(ctx, ours_quant_gemm, len(copies), a.iters)
            row["ours_gemm"]["gbps"] = gbps(n, k, row["ours_gemm"]["us_median"])
            row["speedup_vendor_over_ours_quant_gemm"] = (
                row["ours_quant_gemm"]["us_median"] / row["vendor"]["us_median"])
        rows.append(row)
        del copies, w, wd, ref
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- (b), (c) fp8a8 GEMMs
def _fam_a8(ctx, which, ms, iters):
    """r4d_gemm_fp8a8_nt_m16 (M <= 64) or r4d_gemm_fp8a8_tiled (any M): E4M3 activation + f32
    scale per (row, 128 K), weight codes in "r4d.fp8.w.frag64" order, bf16 checkpoint scale plane,
    bf16 out. Both sides get the same quantised activation (ref_quant), so this is GEMM against
    GEMM; the quantisers are compared in the quant family."""
    torch, lib, a = ctx.torch, ctx.lib, ctx.a
    rows = []
    for name, n, k, per_step in shapes_for(a):
        w, ws = make_weight(ctx, n, k, seed=n * 131 + k)
        nc = n_copies(ctx, n * k)
        copies = [w] + [w.clone() for _ in range(nc - 1)]
        frag0 = frag64_torch(w)
        layouts = {"frag64": [frag0] + [frag0.clone() for _ in range(nc - 1)], "rowmajor": copies}
        wd = dequant_weight(w, ws)
        bs32 = ws.float().contiguous()
        wsp, skb = ptr(ws), k // GROUP
        for m in ms:
            if which == "nt_m16" and m > lib.r4d_gemm_fp8a8_nt_m16_max_m():
                continue
            x = torch.randn(m, k, device=ctx.dev, generator=ctx.gen(11 + m)).to(torch.bfloat16)
            q, xs = ref_quant(torch, x)
            ref = dequant_act(q, xs) @ wd.T
            y = torch.zeros(m, n, device=ctx.dev, dtype=torch.bfloat16)
            assert ptr(y) % 16 == 0, "the tiled kernel needs a 16-byte aligned output"
            qp, xsp, yp = ptr(q), ptr(xs), ptr(y)
            row = {"linear": name, "N": n, "K": k, "M": m, "launches_per_decode_step": per_step,
                   "weight_bytes": n * k, "copies": nc}
            scr, scr_p, scr_b = None, 0, 0
            if which == "nt_m16":
                scr_b = int(lib.r4d_gemm_fp8a8_nt_m16_scratch(m, k, n))
                row["ksplit"], row["scratch_bytes"] = narrow_ksplit(n, k, m), scr_b
                if scr_b:
                    scr = torch.zeros(scr_b, device=ctx.dev, dtype=torch.uint8)
                    scr_p = ptr(scr)

            def make_vendor(wp):
                if which == "nt_m16":
                    def f(i, s):
                        chk(lib.r4d_gemm_fp8a8_nt_m16(qp, xsp, wp[i], wsp, yp, m, k, n, k, skb, k,
                                                      skb, scr_p, scr_b, s),
                            "r4d_gemm_fp8a8_nt_m16")
                else:
                    def f(i, s):
                        chk(lib.r4d_gemm_fp8a8_tiled(qp, xsp, wp[i], wsp, yp, m, k, n, k, skb, k,
                                                     skb, s), "r4d_gemm_fp8a8_tiled")
                return f

            # The weight layout is read from the code, not run: try it, and fall back to the
            # other layout if the reference disagrees, so the receipt records which one is real.
            order = ["frag64", "rowmajor"] if a.a8_layout == "auto" else [a.a8_layout]
            vendor = None
            for lay in order:
                f = make_vendor([ptr(c) for c in layouts[lay]])
                f(0, ctx.stream())
                torch.cuda.synchronize()
                acc = accuracy(torch, y, ref)
                row.setdefault("layout_trials", {})[lay] = acc["rel_max_err"]
                if acc["finite"] and acc["rel_max_err"] < 0.02:
                    vendor, row["weight_layout"], row["vendor_accuracy"] = f, lay, acc
                    break
            if vendor is None:
                row["error"] = "no weight layout reproduces the fp32 reference"
                rows.append(row)
                continue
            y_vendor = y.clone()
            row["vendor"] = timed(ctx, vendor, nc, iters)
            row["vendor"]["gbps"] = gbps(n, k, row["vendor"]["us_median"])
            if ctx.matmul is not None:
                ours_y = ctx.matmul(q, w, xs, bs32, [GROUP, GROUP], output_dtype=torch.bfloat16)
                torch.cuda.synchronize()
                row["ours_accuracy"] = accuracy(torch, ours_y, ref)
                row["vendor_vs_ours"] = ulp_between(torch, y_vendor, ours_y)
                del ours_y

                def ours(i, s, q=q, xs=xs, copies=copies):
                    ctx.matmul(q, copies[i], xs, bs32, [GROUP, GROUP], output_dtype=torch.bfloat16)

                row["ours"] = timed(ctx, ours, nc, iters)
                row["ours"]["gbps"] = gbps(n, k, row["ours"]["us_median"])
                row["speedup_vendor_over_ours"] = (row["ours"]["us_median"]
                                                   / row["vendor"]["us_median"])
            rows.append(row)
            del x, q, xs, ref, y, y_vendor, scr
            torch.cuda.empty_cache()
        del copies, layouts, frag0, w, wd
        torch.cuda.empty_cache()
    return rows


def fam_m16(ctx):
    return _fam_a8(ctx, "nt_m16", ctx.a.m16_m, ctx.a.iters)


def fam_tiled(ctx):
    return _fam_a8(ctx, "tiled", ctx.a.tiled_m, ctx.a.tiled_iters)


# --------------------------------------------------------------------------- (d) quantisers
def _qbufs(ctx, m, n):
    """Output planes of one quantiser launch: codes as uint8 storage (viewed E4M3) and f32 scales."""
    torch = ctx.torch
    q = torch.zeros(m, n, device=ctx.dev, dtype=torch.uint8)
    s = torch.zeros(m, n // GROUP, device=ctx.dev, dtype=torch.float32)
    return q, s


def _e4m3(torch, q):
    return q.view(torch.float8_e4m3fn)


def fam_quant(ctx):
    """The activation quantiser and its two fused producers (r4d_fused_quant_fp8.hip; there is no
    symbol named r4d_fused_quant_fp8): r4d_quant_act_fp8, r4d_rmsnorm_quant_fp8 and
    r4d_gated_quant_fp8, each against a torch reference, against the vendor's unfused pair (the
    source claims bit identity), and timed against our per_token_group_quant_fp8 launch. The
    "ours" norm and silu*mul are plain torch ops, NOT the server's kernels: read those two rows
    as a stand-in until the server's gemma_rmsnorm / SiluAndMul launches are wired in."""
    torch, lib, a = ctx.torch, ctx.lib, ctx.a
    F = torch.nn.functional
    rows, it = [], a.iters
    for m in a.quant_m:
        for k in a.quant_k:                                     # plain quantiser
            x = torch.randn(m, k, device=ctx.dev, generator=ctx.gen(3 + k)).to(torch.bfloat16)
            q, s = _qbufs(ctx, m, k)
            xp, qp, sp, skb = ptr(x), ptr(q), ptr(s), k // GROUP

            def vendor(i, st):
                chk(lib.r4d_quant_act_fp8(xp, qp, sp, m, k, k, k, skb, st), "r4d_quant_act_fp8")

            vendor(0, ctx.stream())
            torch.cuda.synchronize()
            rq, rs = ref_quant(torch, x)
            row = {"op": "quant_act_fp8", "M": m, "n": k,
                   "vendor_vs_ref": quant_agreement(torch, _e4m3(torch, q), s, rq, rs),
                   "vendor": timed(ctx, vendor, 1, it)}
            if ctx.quant is not None:
                oq, os_ = ctx.quant(x, GROUP, column_major_scales=False)
                torch.cuda.synchronize()
                row["ours_vs_ref"] = quant_agreement(torch, oq, os_, rq, rs)
                row["vendor_vs_ours"] = quant_agreement(torch, _e4m3(torch, q), s, oq, os_)
                row["ours"] = timed(
                    ctx, lambda i, st: ctx.quant(x, GROUP, column_major_scales=False), 1, it)
            rows.append(row)

        n = a.hidden                                            # rmsnorm -> quant
        x = torch.randn(m, n, device=ctx.dev, generator=ctx.gen(5)).to(torch.bfloat16)
        gain = (torch.randn(n, device=ctx.dev, generator=ctx.gen(6)) * 0.1).to(torch.bfloat16)
        y1 = torch.zeros(m, n, device=ctx.dev, dtype=torch.bfloat16)
        y2 = torch.zeros_like(y1)
        (q1, s1), (q2, s2) = _qbufs(ctx, m, n), _qbufs(ctx, m, n)
        xp, gp, skb = ptr(x), ptr(gain), n // GROUP
        y1p, q1p, s1p, y2p, q2p, s2p = ptr(y1), ptr(q1), ptr(s1), ptr(y2), ptr(q2), ptr(s2)
        eps, wadd = a.eps, a.wadd

        def fused(i, st, yp=y1p, resp=0):
            chk(lib.r4d_rmsnorm_quant_fp8(xp, resp, gp, yp, q1p, s1p, m, n, eps, wadd, 0,
                                          n, n, n, n, skb, st), "r4d_rmsnorm_quant_fp8")

        def unfused(i, st):
            chk(lib.r4d_rmsnorm_bf16(xp, gp, y2p, m, n, eps, wadd, 0, n, n, st),
                "r4d_rmsnorm_bf16")
            chk(lib.r4d_quant_act_fp8(y2p, q2p, s2p, m, n, n, n, skb, st), "r4d_quant_act_fp8")

        def norm_ref(v):
            vf = v.float()
            return vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps) * (gain.float() + wadd)

        fused(0, ctx.stream())
        unfused(0, ctx.stream())
        torch.cuda.synchronize()
        rq, rs = ref_quant(torch, y1)
        row = {"op": "rmsnorm_quant_fp8", "M": m, "n": n, "eps": eps, "wadd": wadd,
               "y_accuracy": accuracy(torch, y1, norm_ref(x)),
               "codes_vs_ref_quant_of_y": quant_agreement(torch, _e4m3(torch, q1), s1, rq, rs),
               "fused_vs_unfused": {"y_identical": bool((y1 == y2).all()),
                                    "codes_identical": bool((q1 == q2).all()),
                                    "scales_identical": bool((s1 == s2).all())}}
        # fused_add_rms_norm form: res += x (stored as bf16), then the norm of that sum
        res = torch.randn(m, n, device=ctx.dev, generator=ctx.gen(8)).to(torch.bfloat16)
        want_res = (res.float() + x.float()).to(torch.bfloat16)
        resp = ptr(res)
        fused(0, ctx.stream(), resp=resp)
        torch.cuda.synchronize()
        row["residual_form"] = {"res_identical": bool((res == want_res).all()),
                                "y_accuracy": accuracy(torch, y1, norm_ref(want_res))}
        row["vendor_fused"] = timed(ctx, fused, 1, it)
        row["vendor_fused_no_y"] = timed(ctx, lambda i, st: fused(i, st, yp=0), 1, it)
        row["vendor_unfused_pair"] = timed(ctx, unfused, 1, it)
        if ctx.quant is not None:
            def ours(i, st):
                v = x.float()
                yb = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
                      * (gain.float() + wadd)).to(torch.bfloat16)
                ctx.quant(yb, GROUP, column_major_scales=False)
            row["ours_torchnorm_plus_triton_quant"] = timed(ctx, ours, 1, it)
        rows.append(row)

        n = a.inter                                             # silu(gate) * up -> quant
        gu = torch.randn(m, 2 * n, device=ctx.dev, generator=ctx.gen(9)).to(torch.bfloat16)
        y1 = torch.zeros(m, n, device=ctx.dev, dtype=torch.bfloat16)
        y2 = torch.zeros_like(y1)
        (q1, s1), (q2, s2) = _qbufs(ctx, m, n), _qbufs(ctx, m, n)
        gup, skb = ptr(gu), n // GROUP
        y1p, q1p, s1p, y2p, q2p, s2p = ptr(y1), ptr(q1), ptr(s1), ptr(y2), ptr(q2), ptr(s2)

        def gfused(i, st, yp=y1p):
            chk(lib.r4d_gated_quant_fp8(gup, yp, q1p, s1p, m, n, 0, 2 * n, n, n, skb, st),
                "r4d_gated_quant_fp8")

        def gunfused(i, st):
            chk(lib.r4d_silu_mul_bf16(gup, y2p, m, n, 2 * n, n, st), "r4d_silu_mul_bf16")
            chk(lib.r4d_quant_act_fp8(y2p, q2p, s2p, m, n, n, n, skb, st), "r4d_quant_act_fp8")

        gfused(0, ctx.stream())
        gunfused(0, ctx.stream())
        torch.cuda.synchronize()
        rq, rs = ref_quant(torch, y1)
        row = {"op": "gated_quant_fp8", "M": m, "n": n, "mode": "silu",
               "y_accuracy": accuracy(torch, y1, F.silu(gu[:, :n].float()) * gu[:, n:].float()),
               "codes_vs_ref_quant_of_y": quant_agreement(torch, _e4m3(torch, q1), s1, rq, rs),
               "fused_vs_unfused": {"y_identical": bool((y1 == y2).all()),
                                    "codes_identical": bool((q1 == q2).all()),
                                    "scales_identical": bool((s1 == s2).all())},
               "vendor_fused": timed(ctx, gfused, 1, it),
               "vendor_fused_no_y": timed(ctx, lambda i, st: gfused(i, st, yp=0), 1, it),
               "vendor_unfused_pair": timed(ctx, gunfused, 1, it)}
        if ctx.quant is not None:
            def gours(i, st):
                ctx.quant(F.silu(gu[:, :n]) * gu[:, n:], GROUP, column_major_scales=False)
            row["ours_torch_silu_mul_plus_triton_quant"] = timed(ctx, gours, 1, it)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- (e) row top-k
def fam_topk(ctx):
    """rowtopk_bf16 (plugin ABI only: r4d_rad_rowtopk through the row table): the R largest of
    each bf16 row as (int32 column, f32 value), descending, ties to the lower column; two
    launches and M * chunks * R * 8 bytes of device scratch. Against torch.topk; the reference is
    a stable descending sort, which has the same tie rule."""
    torch, a = ctx.torch, ctx.a
    info = ctx.rows["rowtopk_bf16"]
    launch, scratch = LAUNCH_FN(info.launch), SCRATCH_FN(info.scratch)
    rows, r = [], a.topk_r
    for n in a.topk_n:
        for m in a.topk_m:
            x = (torch.randn(m, n, device=ctx.dev, generator=ctx.gen(13)) * 4).to(torch.bfloat16)
            idx = torch.zeros(m, r, device=ctx.dev, dtype=torch.int32)
            val = torch.zeros(m, r, device=ctx.dev, dtype=torch.float32)
            pa = PackedArgs([rad_tensor(ptr(x), RAD_BF16, [m, n]),
                             rad_tensor(ptr(idx), RAD_I32, [m, r]),
                             rad_tensor(ptr(val), RAD_F32, [m, r])],
                            [PI("R", r), PI("M", m), PI("N", n)])
            need = int(scratch(pa.ref()))
            if need < 0:
                rows.append({"N": n, "M": m, "R": r, "error": f"scratch hook returned {need}"})
                continue
            scr = torch.zeros(max(need, 8), device=ctx.dev, dtype=torch.uint8)
            pa.args.scratch, pa.args.scratch_bytes = ptr(scr), need
            ref_args = pa.ref()

            def vendor(i, st):
                chk(launch(ref_args, st or None), "rowtopk_bf16")

            vendor(0, ctx.stream())
            torch.cuda.synchronize()
            sv, si = torch.sort(x.float(), dim=-1, descending=True, stable=True)
            row = {"N": n, "M": m, "R": r, "scratch_bytes": need,
                   "values_identical": bool((val == sv[:, :r]).all()),
                   "indices_identical": bool((idx.long() == si[:, :r]).all()),
                   "vendor": timed(ctx, vendor, 1, a.iters),
                   "torch_topk_bf16": timed(ctx, lambda i, st: torch.topk(x, r, dim=-1), 1,
                                            a.iters)}
            xf = x.float()
            row["torch_topk_f32"] = timed(ctx, lambda i, st: torch.topk(xf, r, dim=-1), 1, a.iters)
            row["speedup_vendor_over_torch_topk_bf16"] = (row["torch_topk_bf16"]["us_median"]
                                                          / row["vendor"]["us_median"])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- TODO stubs
# Families not implemented, with the ABI facts a first implementation needs (abi-notes.md has
# the long form). They are copied into the receipt so a reader sees what was not measured.
STUBS = {
    "ar_oneshot_2rank_exact": {
        "symbols": ["r4d_ar_oneshot_2rank_exact", "r4d_ar_ipc_alloc", "r4d_ar_ipc_free",
                    "r4d_ar_ipc_enable_peer", "r4d_ar_max_blocks"],
        "signature": "void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long "
                     "peer_flags, long my_flags, long seq_ctrs, long slot_stride16, long inp, "
                     "long out, long n_elem, long dtype, long stream, long nblocks, long nthreads, "
                     "long drain, long acq, long stage)",
        "needs": "both cards in ONE process (a thread per rank, hipDeviceEnablePeerAccess, raw "
                 "peer pointers) or IPC handles from r4d_ar_ipc_alloc; scratch double-buffered by "
                 "parity, per-block flags, seq_ctrs = 2 * r4d_ar_max_blocks() zeroed words; both "
                 "ranks launch in lockstep; dtype 0/1/2 = bf16/fp16/fp32. Plugin-row tunable defaults: "
                 "ar_oneshot_2rank_exact nblocks 0 (library rule), nthreads 1024, drain 3, acq 0; "
                 "ar_oneshot_2rank_exact_lane nblocks 8, nthreads 256 (plugin-rows-rad13.txt). "
                 "The plain entries THROW std::runtime_error on a "
                 "HIP error, which aborts the process through ctypes: use the plugin rows, whose "
                 "entry wraps every call (r4d_ar_entry.hip R4D_TRY -> RAD_E_DEVICE) and "
                 "which need their init/fini hooks.",
        "our_baseline": "scripts/bench/p2p_allreduce_decode_shape.py (RCCL dist.all_reduce, "
                        "torchrun 2 procs, numel 1024..1M bf16)"},
    "logits_vocab": {
        "plugin_row": "logits_gemm_vocab_iqr",
        "symbols": ["r4d_rad_logits_gemm_vocab", "r4d_rad_logits_gemm_vocab_scratch",
                    "r4d_logits_gemm_bf16", "r4d_logits_gemm_fp8"],
        "needs": "the head stored as IQR trellis-coded width classes, produced by the vendor "
                 "converter (rad-convert, not built here). Our checkpoint's lm_head is BF16 "
                 "248320 x 5120, so the like-for-like vendor row is r4d_logits_gemm_bf16(a, w, c, "
                 "M, K, N, a_ld, c_ld, stream): bf16 row-major weight, f32 output, K % 8 == 0 "
                 "(prototype is in PROTOS). r4d_logits_gemm_fp8 wants rows of K codes + K/128 "
                 "bf16 scales padded to 16 bytes (5200 bytes at K = 5120).",
        "our_baseline": "the server's lm_head F.linear over the rank's vocabulary shard"},
    "gdn": {
        "symbols": ["r4d_gdn_conv_prep_w4_h128_bf16", "r4d_gdn_conv_update_w4_h128_bf16",
                    "r4d_gdn_kkt_solve_k128_c64_bf16", "r4d_gdn_chunk_scan_k128_v128_c64_bf16",
                    "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state", "r4d_gdn_dims"],
        "signature": "int r4d_gdn_chunk_scan_k128_v128_c64_bf16(q, k, v, A, g, beta, h0, o, ht, "
                     "cu, int N, int H, int Hg, int K, int V, int bt, float scale, int "
                     "state_fp16, void* stream); int r4d_gdn_kkt_solve_k128_c64_bf16(k, beta, g, "
                     "A, cu, int N, int T, int H, int Hg, int K, int bt, void* stream)",
        "needs": "geometry is compiled in (key/value head dim 128, chunk 64, conv width 4), which "
                 "Qwen3.8-27B matches (16 key heads, 48 value heads, conv kernel 4); returns 0, a "
                 "negative for an uncompiled geometry, or a positive hipError_t. Operand layouts "
                 "(packed sequences through cu, state dtype) must be read from r4d_gdn_args.h "
                 "and the selftest's gdn cases before a harness is written.",
        "our_baseline": "the DeltaNet chunk/recurrent kernels under sglang/kernels (prefill and "
                        "decode state update)"},
    "attn_prefill_h256_gqa6": {
        "symbols": ["r4d_attn_prefill_h256_gqa6_fp8kv", "r4d_attn_prefill_h256_gqa6_bf16kv"],
        "signature": "int r4d_attn_prefill_h256_gqa6_fp8kv(const R4DArgs*, hipStream_t)",
        "needs": "struct R4DArgs (r4d_args.h): q bf16, a PAGED KV pool (num_blocks, kv_heads, "
                 "block 16, 2 * head_dim) in fp8 or bf16, an int32 block table, seqused_k, "
                 "descales, scratch, cu_q for packed queries. SGLang's KV pool is not paged in "
                 "16-token blocks, so a bench must build the vendor pool from ours first.",
        "our_baseline": "the extend-attention Triton kernel (patch 108 fp8 QK path)"},
    "gemm_mxfp4a8": {
        "symbols": ["r4d_gemm_mxfp4a8_nt_m64", "r4d_gemm_mxfp4a8_tiled", "r4d_mxfp4a8_tile_a",
                    "r4d_rad_gemm_mxfp4a8_decode", "r4d_rad_layout_mxfp4"],
        "signature": "void r4d_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long "
                     "wref, long c, int M, int K, int N, int WV, int SK, int MB, int NPW, int "
                     "as_group, int as_rs, long stream, int lda, int ldws, int ldc)",
        "needs": "the packed MXFP4 weight layout (group 32, E8M0 scales) from "
                 "r4d_rad_layout_mxfp4 and its relayout hook; the decode band exists only as a "
                 "plugin row; the void entries report nothing on failure.",
        "our_baseline": "scripts/bench/test_mxfp4_w4a8_gemm.py and "
                        "sglang/kernels/ops/gemm/mxfp4_w4a8_rdna4.py (patch 117)"},
}

FAMILIES = {"m1": fam_m1, "m16": fam_m16, "tiled": fam_tiled, "quant": fam_quant, "topk": fam_topk}

ASSUMPTIONS = [
    "torch.cuda.current_stream().cuda_stream is a hipStream_t libr4d accepts (0 = default stream)",
    "libr4d resolves libamdhip64.so.7 to the image torch already mapped (checked at start-up)",
    "device pointers from tensor.data_ptr() are valid kernel arguments for libr4d launches",
    "fp8a8 weight codes are in frag64 order (auto layout detection records which layout passed)",
    "row pitches passed explicitly: bytes for E4M3 planes, elements for bf16 and f32 planes",
    "the narrow tile's scratch does not need zeroing and its counters warm on the first launch",
    "torch events bracket libr4d launches made on the same stream",
]


def _ints(s):
    return [int(v) for v in s.split(",") if v]


def bench(a):
    import hashlib
    import traceback
    busy = vram_used_gb()
    if any(v > a.max_busy_vram_gb for v in busy.values()) and not a.force_busy:
        raise SystemExit(f"a card is in use (VRAM GiB {busy}); this bench must not share the GPUs "
                         "with a timed run. --force-busy overrides.")
    ctx = Ctx(a)
    torch = ctx.torch
    with open(a.lib, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    info = ctx.lib.rad_plugin_info().contents
    receipt = {"experiment": "RAD-13 libr4d kernel bench",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "lib": {"path": os.path.abspath(a.lib), "sha256": sha,
                       "version": (info.version or b"").decode(),
                       "build_target": (info.build_target or b"").decode(),
                       "abi_version": ctx.lib.rad_plugin_abi_version()},
               "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
               "hip_runtime_images": ctx.hip_maps, "vram_used_gb_at_start": busy,
               "triton_error": ctx.triton_error, "args": vars(a), "families": {},
               "stubs": STUBS, "assumptions_behind_the_reference_gates": ASSUMPTIONS}
    for fam in a.families:
        try:
            receipt["families"][fam] = FAMILIES[fam](ctx)
        except Exception as e:      # one family's failure must not lose the others' rows
            receipt["families"][fam] = {"error": repr(e), "traceback": traceback.format_exc()}
        torch.cuda.empty_cache()
        with open(a.out, "w") as f:  # rewritten after every family: a crash keeps what finished
            json.dump(receipt, f, indent=1)
            f.write("\n")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", help="path to libr4d.so")
    ap.add_argument("--out", help="JSON receipt path")
    ap.add_argument("--selfcheck", action="store_true", help="CPU-only ABI checks, no GPU")
    ap.add_argument("--abi-table", action="append",
                    help="layout table in tests/abi_layout_table.h row format; repeatable. The "
                         "vendor snapshot or one generated from the headers (see abi-notes.md)")
    ap.add_argument("--abi-rows", help="<radiance>/abi/rad_abi_layout_rows.h")
    ap.add_argument("--list-stubs", action="store_true", help="print the unimplemented families")
    ap.add_argument("--families", default="m1,m16,tiled,quant,topk",
                    help="comma list of " + ",".join(FAMILIES))
    ap.add_argument("--device", type=int, help="card index, applied as HIP_VISIBLE_DEVICES")
    ap.add_argument("--shapes", default="qwen38", choices=sorted(SHAPES))
    ap.add_argument("--shape", action="append", help="N,K (repeatable); replaces --shapes")
    ap.add_argument("--iters", type=int, default=200, help="timed launches per repeat")
    ap.add_argument("--tiled-iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rotate-mb", type=int, default=320, help="distinct weight bytes per pass")
    ap.add_argument("--m16-m", type=_ints, default=[8, 16, 64])
    ap.add_argument("--tiled-m", type=_ints, default=[4096, 8192])
    ap.add_argument("--quant-m", type=_ints, default=[1, 8, 64])
    ap.add_argument("--quant-k", type=_ints, default=[5120, 8704, 6144, 3072])
    ap.add_argument("--hidden", type=int, default=5120, help="rmsnorm->quant width")
    ap.add_argument("--inter", type=int, default=8704, help="silu*mul->quant width per rank")
    ap.add_argument("--topk-n", type=_ints, default=[15520, 124160])
    ap.add_argument("--topk-m", type=_ints, default=[1, 8, 64])
    ap.add_argument("--topk-r", type=int, default=32)
    ap.add_argument("--rb", type=int, default=0, help="fp8a16 m1 tunable (0 = library rule)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--wadd", type=float, default=1.0, help="gain offset (1.0 = GemmaRMSNorm)")
    ap.add_argument("--timing", default="events", choices=["events", "graph"])
    ap.add_argument("--graph-replays", type=int, default=3)
    ap.add_argument("--a8-layout", default="auto", choices=["auto", "frag64", "rowmajor"])
    ap.add_argument("--sglang-python", default="/data/sgl-v0522/python")
    ap.add_argument("--no-triton", action="store_true", help="vendor side only")
    ap.add_argument("--allow-two-hip-runtimes", action="store_true")
    ap.add_argument("--max-busy-vram-gb", type=float, default=2.0)
    ap.add_argument("--force-busy", action="store_true")
    a = ap.parse_args(argv)
    if a.list_stubs:
        print(json.dumps(STUBS, indent=1))
        return 0
    if a.selfcheck:
        res = selfcheck(a)
        text = json.dumps(res, indent=1)
        if a.out:
            with open(a.out, "w") as f:
                f.write(text + "\n")
        print(text)
        return 0 if res["ok"] else 1
    if not a.lib or not a.out:
        ap.error("bench mode needs --lib and --out")
    a.families = [f for f in a.families.split(",") if f]
    unknown = [f for f in a.families if f not in FAMILIES]
    if unknown:
        ap.error(f"unknown families {unknown}; stubs are listed by --list-stubs")
    if a.rotate_mb * 2**20 <= ROTATE_BYTES_MIN:
        ap.error("--rotate-mb must exceed 256 (the MALL cache trap)")
    receipt = bench(a)
    failed = [k for k, v in receipt["families"].items() if isinstance(v, dict) and "error" in v]
    print(f"wrote {a.out}; families {list(receipt['families'])}; failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
