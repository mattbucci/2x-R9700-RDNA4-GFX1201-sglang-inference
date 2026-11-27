"""Behaviour checks for markdown_table_format. Usage: check_markdown_table_format.py <module path>."""
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
    m = load(sys.argv[1], "markdown_table_format_under_test")
    Table, TableError = m.Table, m.TableError

    assert m.split_row("| a | b |") == ["a", "b"]
    assert m.split_row("a|b") == ["a", "b"]
    assert m.split_row("| a \\| b | c |") == ["a | b", "c"]
    assert m.split_row("| a | \\|") == ["a", "|"]
    assert m.split_row("|  | x |") == ["", "x"]
    assert m.split_row("|") == []
    assert m.escape_cell("a|b") == "a\\|b"

    assert m.parse_alignment(":---") == m.ALIGN_LEFT
    assert m.parse_alignment("---:") == m.ALIGN_RIGHT
    assert m.parse_alignment(" :-: ") == m.ALIGN_CENTER
    assert m.parse_alignment("-") == m.ALIGN_NONE
    assert raises(TableError, m.parse_alignment, ":")
    assert raises(TableError, m.parse_alignment, "-:-")

    source = "\n".join([
        "|Name|Qty|Price|Note|",
        "|:--|:-:|--:|---|",
        "|apple|3|1.25|",
        "",
        "|kiwi \\| gold|12|10.00|ripe|extra|",
    ])
    table = Table.parse(source)
    assert table.headers == ["Name", "Qty", "Price", "Note"]
    assert table.alignments == [m.ALIGN_LEFT, m.ALIGN_CENTER, m.ALIGN_RIGHT, m.ALIGN_NONE]
    assert table.rows == [["apple", "3", "1.25", ""], ["kiwi | gold", "12", "10.00", "ripe"]]
    assert table.column("Qty") == ["3", "12"]
    assert table.column_widths() == [12, 3, 5, 4]

    expected = "\n".join([
        "| Name         | Qty | Price | Note |",
        "| :----------- | :-: | ----: | ---- |",
        "| apple        |  3  |  1.25 |      |",
        "| kiwi \\| gold | 12  | 10.00 | ripe |",
    ])
    rendered = table.render()
    assert rendered == expected, "\n" + rendered
    assert len({len(line) for line in rendered.splitlines()}) == 1
    assert m.format_table(rendered) == rendered
    assert Table.parse(rendered).rows == table.rows

    small = Table(["a", "b"], [m.ALIGN_RIGHT, m.ALIGN_CENTER], [["1", "2"]])
    assert small.render() == "|   a |  b  |\n| --: | :-: |\n|   1 |  2  |"
    small.set_alignment("a", m.ALIGN_LEFT)
    small.add_row(["wide cell"])
    assert small.render() == "\n".join([
        "| a         |  b  |",
        "| :-------- | :-: |",
        "| 1         |  2  |",
        "| wide cell |     |",
    ])
    assert raises(TableError, small.add_row, ["1", "2", "3"])
    assert raises(KeyError, small.column, "zzz")
    assert raises(TableError, small.set_alignment, "a", "justify")
    assert raises(TableError, Table.parse, "| a | b |")
    assert raises(TableError, Table.parse, "| a | b |\n| --- |")
    assert raises(TableError, Table.parse, "| a | b |\n| --- | abc |")
    assert raises(TableError, Table, [])
    print("ok")


if __name__ == "__main__":
    main()
