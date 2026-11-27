"""Behaviour checks for json_pointer_patch. Usage: check_json_pointer_patch.py <module path>."""
import importlib.util
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raises(exc, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc:
        return True
    return False



def main():
    m = load(sys.argv[1], "json_pointer_patch_under_test")

    doc = {
        "foo": ["bar", "baz"], "": 0, "a/b": 1, "c%d": 2, "e^f": 3,
        "g|h": 4, "i\\j": 5, "k\"l": 6, " ": 7, "m~n": 8,
    }
    assert m.resolve(doc, "") is doc
    assert m.resolve(doc, "/foo") == ["bar", "baz"]
    assert m.resolve(doc, "/foo/0") == "bar"
    assert m.resolve(doc, "/") == 0
    assert m.resolve(doc, "/a~1b") == 1
    assert m.resolve(doc, "/m~0n") == 8
    assert m.resolve(doc, "/ ") == 7
    assert m.resolve(doc, "/missing", default=None) is None
    assert raises(m.PatchError, m.resolve, doc, "/foo/2")
    assert raises(m.PatchError, m.resolve, doc, "/foo/-")
    assert raises(m.PointerError, m.resolve, doc, "foo")
    assert raises(m.PointerError, m.resolve, doc, "/foo/01")

    assert m.parse_pointer("/a~1b/m~0n/") == ["a/b", "m~n", ""]
    assert m.format_pointer(["a/b", "m~n", ""]) == "/a~1b/m~0n/"
    assert m.escape_token("~1") == "~01"
    assert m.unescape_token("~01") == "~1"
    for tokens in (["~1"], ["~0", "/~", "~/"], ["x~1y", "~~", "//"]):
        assert m.parse_pointer(m.format_pointer(tokens)) == tokens
    assert m.resolve({"~1": "tilde-one", "/": "slash"}, "/~01") == "tilde-one"

    d = {"a": [1, 2, 3], "b": {"c": 1}}
    assert m.add(d, "/a/1", 9) is d and d["a"] == [1, 9, 2, 3]
    m.add(d, "/a/-", 7)
    assert d["a"] == [1, 9, 2, 3, 7]
    m.add(d, "/a/5", 8)
    assert d["a"] == [1, 9, 2, 3, 7, 8]
    assert raises(m.PatchError, m.add, d, "/a/7", 0)
    m.add(d, "/b/d", 2)
    assert d["b"] == {"c": 1, "d": 2}
    assert m.add(d, "", [1]) == [1]

    m.remove(d, "/a/0")
    assert d["a"] == [9, 2, 3, 7, 8]
    m.remove(d, "/b/c")
    assert d["b"] == {"d": 2}
    assert raises(m.PatchError, m.remove, d, "/b/c")
    assert raises(m.PatchError, m.remove, d, "")

    m.replace(d, "/a/1", "x")
    assert d["a"] == [9, "x", 3, 7, 8]
    assert raises(m.PatchError, m.replace, d, "/b/zzz", 1)
    assert raises(m.PatchError, m.replace, d, "/a/5", 1)

    d = {"a": [1, 2, 3], "b": {"c": 1}}
    m.move(d, "/a/0", "/a/2")
    assert d["a"] == [2, 3, 1]
    m.move(d, "/b/c", "/b/e")
    assert d["b"] == {"e": 1}
    m.move(d, "/b", "/a/0")
    assert d == {"a": [{"e": 1}, 2, 3, 1]}
    assert raises(m.PatchError, m.move, d, "/a", "/a/0")

    original = {"items": [1, 2], "meta": {"n": 2}}
    patched = m.apply_patch(original, [
        {"op": "add", "path": "/items/-", "value": 3},
        {"op": "replace", "path": "/meta/n", "value": 3},
        {"op": "copy", "from": "/meta", "path": "/backup"},
        {"op": "move", "from": "/items/0", "path": "/first"},
        {"op": "remove", "path": "/meta"},
    ])
    assert patched == {"items": [2, 3], "backup": {"n": 3}, "first": 1}
    assert original == {"items": [1, 2], "meta": {"n": 2}}
    assert raises(m.PatchError, m.apply_patch, original, [{"op": "frobnicate", "path": ""}])
    assert raises(m.PatchError, m.apply_patch, original, [{"op": "add", "path": "/x"}])
    print("ok")


if __name__ == "__main__":
    main()
