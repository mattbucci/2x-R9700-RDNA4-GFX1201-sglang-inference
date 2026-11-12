"""Tensor-level comparison of two stage dumps: per forward file, every tensor leaf bit for bit."""
import os, sys, torch

def leaves(x, path=()):
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from leaves(x[k], path + (str(k),))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, path + (str(i),))
    else:
        yield path, x

def main():
    a, b, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
    torch.set_num_threads(4)
    files = sorted(os.path.join(r, f)[len(a) + 1:] for r, _, fs in os.walk(a) for f in fs if f.endswith(".pt"))
    if limit: files = files[::max(1, len(files) // limit)]
    eq = ne = 0; kinds = {}; meta = {}; first = []
    for f in files:
        x = dict(leaves(torch.load(os.path.join(a, f), map_location="cpu", weights_only=True)))
        y = dict(leaves(torch.load(os.path.join(b, f), map_location="cpu", weights_only=True)))
        same = set(x) == set(y)
        bad = []
        for k in sorted(set(x) & set(y)):
            u, v = x[k], y[k]
            if torch.is_tensor(u):
                ok = torch.is_tensor(v) and u.shape == v.shape and u.dtype == v.dtype and torch.equal(u, v)
                kinds.setdefault(k[-1], [0, 0])[0 if ok else 1] += 1
                if not ok: bad.append("/".join(k))
            elif u != v:
                meta["/".join(k[-1:])] = meta.get("/".join(k[-1:]), 0) + 1
        if same and not bad: eq += 1
        else:
            ne += 1
            if len(first) < 3: first.append((f, bad[:6], same))
    print("TENSORS %s vs %s: files %d, all tensors equal in %d, differ in %d" % (a, b, len(files), eq, ne))
    print("per leaf name [equal, differ]:", {k: v for k, v in sorted(kinds.items())})
    print("non-tensor leaves that differ (count by name):", meta)
    for row in first: print("first differing:", row)

main()
