"""JSON Pointer (RFC 6901) resolution and JSON Patch (RFC 6902) operations.

Documents are plain Python structures as produced by ``json.loads``: dicts,
lists, strings, numbers, booleans and ``None``.

The single-operation helpers (:func:`add`, :func:`remove`, :func:`replace`,
:func:`move`, :func:`copy_value`) modify the document in place and return the
document root, which is a different object only when the pointer addresses
the whole document.  :func:`apply_patch` works on a deep copy and is therefore
all-or-nothing: the caller's document is never left half patched.
"""

from __future__ import annotations

import copy
import re
from typing import Any, List, Mapping, Sequence

_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
_MISSING = object()


class PointerError(ValueError):
    """Raised when a pointer string or an array index token is malformed."""


class PatchError(ValueError):
    """Raised when an operation cannot be applied to the document."""


def escape_token(token: str) -> str:
    """Encode one reference token for use inside a pointer string."""
    return token.replace("~", "~0").replace("/", "~1")


def unescape_token(token: str) -> str:
    """Decode one reference token taken from a pointer string."""
    return token.replace("~0", "~").replace("~1", "/")


def parse_pointer(pointer: str) -> List[str]:
    """Split a pointer string into its decoded reference tokens.

    The empty string addresses the whole document and yields no tokens.
    """
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise PointerError(f"pointer must be empty or start with '/': {pointer!r}")
    return [unescape_token(token) for token in pointer[1:].split("/")]


def format_pointer(tokens: Sequence[str]) -> str:
    """Build a pointer string from decoded reference tokens."""
    return "".join("/" + escape_token(token) for token in tokens)


def _list_index(token: str, length: int, allow_end: bool = False) -> int:
    """Convert an array reference token to an index into a list of ``length``.

    With ``allow_end`` the position one past the last element is accepted,
    either as that number or as ``-``; this is the insertion form used by add.
    """
    if token == "-":
        if allow_end:
            return length
        raise PatchError("'-' refers to a nonexistent array element")
    if not _INDEX_RE.fullmatch(token):
        raise PointerError(f"invalid array index {token!r}")
    index = int(token)
    limit = length if allow_end else length - 1
    if index > limit:
        raise PatchError(f"array index {index} out of range (length {length})")
    return index


def _step(container: Any, token: str) -> Any:
    """Follow one reference token from ``container``."""
    if isinstance(container, dict):
        if token not in container:
            raise PatchError(f"member {token!r} not found")
        return container[token]
    if isinstance(container, list):
        return container[_list_index(token, len(container))]
    raise PatchError(
        f"cannot look up {token!r} in a {type(container).__name__} value"
    )


def _walk(document: Any, tokens: Sequence[str]) -> Any:
    """Follow every token in ``tokens`` starting at the document root."""
    current = document
    for token in tokens:
        current = _step(current, token)
    return current


def resolve(document: Any, pointer: str, default: Any = _MISSING) -> Any:
    """Return the value addressed by ``pointer``.

    A missing target raises :class:`PatchError` unless ``default`` is given.
    A malformed pointer always raises :class:`PointerError`.
    """
    tokens = parse_pointer(pointer)
    try:
        return _walk(document, tokens)
    except PatchError:
        if default is _MISSING:
            raise
        return default


def _insert(document: Any, tokens: Sequence[str], value: Any) -> Any:
    """Add ``value`` at the location named by ``tokens``; return the root."""
    if not tokens:
        return value
    parent = _walk(document, tokens[:-1])
    token = tokens[-1]
    if isinstance(parent, dict):
        parent[token] = value
    elif isinstance(parent, list):
        parent.insert(_list_index(token, len(parent), allow_end=True), value)
    else:
        raise PatchError(f"cannot add a member to a {type(parent).__name__} value")
    return document


def _detach(document: Any, tokens: Sequence[str]) -> Any:
    """Remove the value named by ``tokens`` and return the removed value."""
    if not tokens:
        raise PatchError("cannot remove the document root")
    parent = _walk(document, tokens[:-1])
    token = tokens[-1]
    if isinstance(parent, dict):
        if token not in parent:
            raise PatchError(f"member {token!r} not found")
        return parent.pop(token)
    if isinstance(parent, list):
        return parent.pop(_list_index(token, len(parent)))
    raise PatchError(f"cannot remove from a {type(parent).__name__} value")


def add(document: Any, pointer: str, value: Any) -> Any:
    """Add ``value`` at ``pointer``.

    An object member is created or overwritten.  An array index inserts
    before the element currently at that index; ``-`` appends.
    """
    return _insert(document, parse_pointer(pointer), value)


def remove(document: Any, pointer: str) -> Any:
    """Remove the value at ``pointer``, which must exist."""
    _detach(document, parse_pointer(pointer))
    return document


def replace(document: Any, pointer: str, value: Any) -> Any:
    """Replace the existing value at ``pointer`` with ``value``."""
    tokens = parse_pointer(pointer)
    if not tokens:
        return value
    parent = _walk(document, tokens[:-1])
    token = tokens[-1]
    if isinstance(parent, dict):
        if token not in parent:
            raise PatchError(f"member {token!r} not found")
        parent[token] = value
    elif isinstance(parent, list):
        parent[_list_index(token, len(parent))] = value
    else:
        raise PatchError(f"cannot replace inside a {type(parent).__name__} value")
    return document


def move(document: Any, from_pointer: str, pointer: str) -> Any:
    """Remove the value at ``from_pointer`` and add it at ``pointer``.

    The target index of an array move is interpreted after the removal, as
    the RFC specifies.  A value cannot be moved into one of its own children.
    """
    source = parse_pointer(from_pointer)
    target = parse_pointer(pointer)
    if source == target:
        _walk(document, source)
        return document
    if len(target) > len(source) and target[: len(source)] == source:
        raise PatchError("cannot move a value into one of its own children")
    value = _detach(document, source)
    return _insert(document, target, value)


def copy_value(document: Any, from_pointer: str, pointer: str) -> Any:
    """Add an independent copy of the value at ``from_pointer`` at ``pointer``."""
    value = copy.deepcopy(_walk(document, parse_pointer(from_pointer)))
    return _insert(document, parse_pointer(pointer), value)


def _member(operation: Mapping[str, Any], key: str, position: int) -> Any:
    """Return a required member of a patch operation object."""
    if key not in operation:
        raise PatchError(f"operation {position} is missing {key!r}")
    return operation[key]


def apply_patch(document: Any, operations: Sequence[Mapping[str, Any]]) -> Any:
    """Apply a sequence of patch operations and return the new document.

    Supported operations are ``add``, ``remove``, ``replace``, ``move`` and
    ``copy``.  The input document and the operation values are not modified
    or shared with the result; any failure leaves the input untouched.
    """
    result = copy.deepcopy(document)
    for position, operation in enumerate(operations):
        name = _member(operation, "op", position)
        path = _member(operation, "path", position)
        if name == "add":
            value = copy.deepcopy(_member(operation, "value", position))
            result = add(result, path, value)
        elif name == "remove":
            result = remove(result, path)
        elif name == "replace":
            value = copy.deepcopy(_member(operation, "value", position))
            result = replace(result, path, value)
        elif name == "move":
            result = move(result, _member(operation, "from", position), path)
        elif name == "copy":
            result = copy_value(result, _member(operation, "from", position), path)
        else:
            raise PatchError(f"operation {position} has unknown op {name!r}")
    return result
