"""What the first dumped forward of each rank holds: check_r97s_dump.py <dump dir> <layers csv> <kinds csv>"""
import glob
import os
import sys

import torch

root, layers, kinds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3].split(",")
need = sorted("L%02d-%s" % (layer, k) for layer in layers for k in kinds)
ok = True
for r in (0, 1):
    files = sorted(glob.glob(os.path.join(root, "rank%d" % r, "fwd*.pt")))
    if not files:
        print("DUMP CHECK rank%d: no forward dumped" % r)
        ok = False
        continue
    d = torch.load(files[0], map_location="cpu", weights_only=True, mmap=True)
    st, meta = d["stages"], d["meta"]
    good = sorted(st) == need and all(st[n].dtype == torch.bfloat16 and st[n].shape[0] == meta["rows"] for n in st)
    ok = ok and good and not meta["non_tensor"]
    print("DUMP CHECK rank%d: %d forwards, first has %d stages (want %d), rows %d, prefix %s, rids %s, q %s k %s attn %s, "
          "non-tensor %s" % (r, len(files), len(st), len(need), meta["rows"], meta["extend_prefix_lens"], meta["rids"][:1],
                             *(list(st[need[0][:4] + x].shape) if need[0][:4] + x in st else "?" for x in ("q", "k", "attn")),
                             sorted(meta["non_tensor"]) or "none"))
print("DUMP CHECK %s" % ("ok" if ok else "FAIL"))
sys.exit(0 if ok else 1)
