import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

parse = m.parse

assert parse("a,b,c\r\n1,2,3\r\n") == [["a", "b", "c"], ["1", "2", "3"]]
assert parse("a,b\n1,2") == [["a", "b"], ["1", "2"]]
assert parse("a\rb\r") == [["a"], ["b"]]
assert parse("") == [] and parse("\n\r\n") == []
assert parse('"x, y","say ""hi""",z\n') == [["x, y", 'say "hi"', "z"]]
assert parse('"line1\nline2",b\n"c\r\nd"\n') == [["line1\nline2", "b"], ["c\r\nd"]]
assert parse("a,,c\n,\n") == [["a", "", "c"], ["", ""]]
assert parse('a\n\nb\n""\n') == [["a"], ["b"], [""]]
assert parse("a,b,\n") == [["a", "b", ""]]

# last record without a line break
assert parse("x,y\n1,") == [["x", "y"], ["1", ""]], "trailing empty field at end of input"
assert parse('k\n""') == [["k"], [""]]
assert parse(",") == [["", ""]]

for bad in ('"abc', 'a"b,c\n', '"a"b\n'):
    try:
        parse(bad)
    except m.CsvError as err:
        assert err.line == 1
    else:
        raise AssertionError("expected CsvError for %r" % bad)
try:
    parse('a\n"b\nc\n"x"y')
except m.CsvError as err:
    assert err.line == 4
else:
    raise AssertionError("expected CsvError")
lenient = m.Dialect(strict=False)
assert parse('a"b,"c"d\n', lenient) == [['a"b', "cd"]]

assert parse("a\tb,c\n", m.TSV) == [["a", "b,c"]]
assert m.parse_dicts("id,name\n1,ann\n2,\"b,ob\"\n") == [{"id": "1", "name": "ann"}, {"id": "2", "name": "b,ob"}]
assert m.parse_dicts("") == []
for bad in ("a,a\n1,2\n", "a,b\n1\n"):
    try:
        m.parse_dicts(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

assert m.format_field('he said "no"') == '"he said ""no"""'
assert m.format_row(["a", "b,c", ""]) == 'a,"b,c",'
rows = [["a", "b"], [""], ["x,y", 'q"q', "multi\nline"], ["", ""], ["cr\rlf", " pad "]]
for dialect in (m.EXCEL, m.UNIX, m.TSV):
    assert parse(m.dumps(rows, dialect), dialect) == rows
assert m.dumps([["a", "b"]], m.UNIX) == "a,b\n"
assert m.column([["a", "b"], ["c"]], 1, default="-") == ["b", "-"]
for bad in (lambda: m.Dialect(delimiter=",,"), lambda: m.Dialect(delimiter='"'), lambda: m.format_row([])):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
print("ok")
