"""Behaviour check for text_diff_lcs. Usage: check_text_diff_lcs.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load(sys.argv[1])
    a = ["a", "b", "c", "d", "e"]
    b = ["a", "c", "x", "e", "f"]
    assert m.lcs_length(a, b) == 3
    assert m.longest_common_subsequence(a, b) == ["a", "c", "e"]
    ops = m.diff_ops(a, b)
    assert m.apply_ops(ops) == b and m.original_from_ops(ops) == a
    assert m.diff_stats(ops) == (2, 2)
    assert abs(m.similarity_ratio(a, b) - 0.6) < 1e-9
    assert m.similarity_ratio([], []) == 1.0 and m.lcs_length([], a) == 0
    assert m.unified_diff(a, a) == [] and m.group_hunks(m.diff_ops(a, a)) == []
    assert m.diff_text("x\ny", "x\ny") == ""

    old = ["line%d" % i for i in range(1, 11)]
    new = list(old)
    new[4] = "changed"
    hunks = m.group_hunks(m.diff_ops(old, new), context=2)
    assert len(hunks) == 1
    assert hunks[0].lines == [" line3", " line4", "-line5", "+changed", " line6", " line7"], hunks[0].lines
    assert hunks[0].header() == "@@ -3,5 +3,5 @@"
    assert m.unified_diff(old, new, context=1, from_name="old", to_name="new") == [
        "--- old", "+++ new", "@@ -4,3 +4,3 @@", " line4", "-line5", "+changed", " line6"]

    # context 0 keeps only the changes
    h0 = m.group_hunks(m.diff_ops(old, new), context=0)
    assert [h.lines for h in h0] == [["-line5", "+changed"]]

    # change at the very end and at the very start
    tail = old + ["extra"]
    ht = m.group_hunks(m.diff_ops(old, tail), context=3)
    assert ht[0].lines == [" line8", " line9", " line10", "+extra"]
    assert ht[0].header() == "@@ -8,3 +8,4 @@"
    hs = m.group_hunks(m.diff_ops(old, old[1:]), context=1)
    assert hs[0].lines == ["-line1", " line2"] and hs[0].header() == "@@ -1,2 +1,1 @@"

    # pure insertion into an empty file
    he = m.group_hunks(m.diff_ops([], ["x", "y"]))
    assert he[0].header() == "@@ -0,0 +1,2 @@" and he[0].lines == ["+x", "+y"]

    # far apart changes give two hunks, near ones merge
    big = ["r%d" % i for i in range(30)]
    far = list(big)
    far[3] = "X"
    far[25] = "Y"
    hf = m.group_hunks(m.diff_ops(big, far), context=2)
    assert [h.header() for h in hf] == ["@@ -2,5 +2,5 @@", "@@ -24,5 +24,5 @@"]
    near = list(big)
    near[3] = "X"
    near[8] = "Y"
    hn = m.group_hunks(m.diff_ops(big, near), context=2)
    assert len(hn) == 1 and hn[0].header() == "@@ -2,10 +2,10 @@"
    for hunk in hf + hn:
        assert hunk.a_count == sum(1 for l in hunk.lines if l[0] != "+")
        assert hunk.b_count == sum(1 for l in hunk.lines if l[0] != "-")
    try:
        m.group_hunks(ops, context=-1)
        raise SystemExit("negative context accepted")
    except ValueError:
        pass
    print("ok")


if __name__ == "__main__":
    main()
