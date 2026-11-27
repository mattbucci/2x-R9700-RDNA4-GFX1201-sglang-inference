"""Behaviour check for semver_range. Usage: check_semver_range.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

V = m.Version.parse
v = V("v1.2.3-rc.1+build.5")
assert v.core == (1, 2, 3) and v.prerelease == ("rc", "1") and v.build == ("build", "5")
assert str(v) == "1.2.3-rc.1+build.5" and v.is_prerelease()
for bad in ("1.2", "01.2.3", "1.2.3-", "1.2.3-01", "a.b.c", ""):
    try:
        V(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad!r}")
assert m.compare(V("1.2.3"), V("1.2.3+other")) == 0
assert m.compare(V("1.10.0"), V("1.9.0")) == 1
assert m.compare(V("1.0.0-rc.10"), V("1.0.0-rc.9")) == 1
assert m.compare(V("1.0.0-alpha.1"), V("1.0.0-alpha.beta")) == -1
assert m.compare(V("1.0.0-alpha"), V("1.0.0-alpha.1")) == -1

# a release outranks its own pre-releases
assert m.compare(V("1.0.0"), V("1.0.0-alpha")) == 1
assert m.compare(V("1.0.0-rc.1"), V("1.0.0")) == -1
chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
         "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
assert m.sort_versions(reversed(chain)) == chain
assert m.satisfies("1.0.0", ">=1.0.0-rc.1")
assert not m.satisfies("1.0.0-rc.1", ">=1.0.0")
assert m.max_satisfying(["1.2.0-beta", "1.2.0", "1.1.9"], "^1.0.0") == "1.2.0"

assert m.satisfies("1.4.0", "^1.2.3") and not m.satisfies("2.0.0", "^1.2.3")
assert not m.satisfies("1.2.2", "^1.2.3")
assert m.satisfies("0.2.9", "^0.2.3") and not m.satisfies("0.3.0", "^0.2.3")
assert m.satisfies("0.0.3", "^0.0.3") and not m.satisfies("0.0.4", "^0.0.3")
assert m.satisfies("1.2.9", "~1.2.3") and not m.satisfies("1.3.0", "~1.2.3")
assert m.satisfies("1.5.0", ">= 1.2.0 <2.0.0") and not m.satisfies("2.0.0", ">=1.2.0 <2.0.0")
assert m.satisfies("3.1.0", "<1.0.0 || >3.0.0") and not m.satisfies("2.0.0", "<1.0.0 || >3.0.0")
assert m.satisfies("1.0.0", "=1.0.0") and m.satisfies("1.0.0", "1.0.0") and m.satisfies("9.9.9", "*")
assert m.satisfies("1.0.0", "<=1.0.0") and not m.satisfies("1.0.1", "<=1.0.0")
assert m.max_satisfying(["1.0.0", "1.2.0", "2.0.0"], "~3.0.0") is None
print("ok")
