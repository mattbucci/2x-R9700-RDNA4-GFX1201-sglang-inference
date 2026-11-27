"""Behaviour check for trie_autocomplete. Usage: check_trie_autocomplete.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

t = m.Trie()
assert t.insert("car", 5) is True
assert t.insert("cart", 2) is True
assert t.insert("cat", 9) is True
assert t.insert("dog", 1) is True
assert t.insert("car", 6) is False
assert len(t) == 4
assert "car" in t and "ca" not in t and 7 not in t
assert t.weight("car") == 6.0 and t.weight("ca") is None
assert t.complete("ca", k=2) == ["cat", "car"]
assert t.complete("ca", k=10) == ["cat", "car", "cart"]
assert t.complete("x") == [] and t.complete("ca", k=0) == []
assert t.count_prefix("car") == 2
assert [w for w, _ in t.iter_words()] == ["car", "cart", "cat", "dog"]
assert t.increment("dog", 2) == 3.0 and t.increment("emu") == 1.0
assert t.delete("emu") is True and not t.has_prefix("e")
assert t.delete("ca") is False and t.delete("zebra") is False

# deleting a word that is a prefix of another keeps the longer word
assert t.delete("car") is True
assert "cart" in t and "car" not in t and len(t) == 3
t.insert("car", 6)

# deleting a word that extends another keeps the shorter word
assert t.delete("cart") is True
assert len(t) == 3
assert "car" in t, "shorter word lost when its extension was deleted"
assert t.weight("car") == 6.0
assert t.complete("ca", k=5) == ["cat", "car"]
assert not t.has_prefix("cart")

u = m.Trie(case_sensitive=False)
u.insert("Apple", 3)
u.insert("apricot", 3)
assert "APPLE" in u and u.complete("AP") == ["apple", "apricot"]
assert u.longest_common_prefix() == "ap"
u.clear()
assert len(u) == 0 and not u.has_prefix("")
try:
    u.insert("")
except ValueError:
    pass
else:
    raise AssertionError("empty word accepted")
print("ok")
