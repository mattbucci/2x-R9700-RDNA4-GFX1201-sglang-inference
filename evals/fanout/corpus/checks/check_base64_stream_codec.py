"""Behaviour check for base64_stream_codec. Usage: check_base64_stream_codec.py <module path>."""
import base64
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
    samples = [bytes((i * 37 + 11) % 256 for i in range(n)) for n in range(0, 40)]
    samples.append(bytes(range(256)))
    for raw in samples:
        ref = base64.b64encode(raw).decode("ascii")
        assert m.encode(raw) == ref
        assert m.decode(ref) == raw
        assert m.encoded_length(len(raw)) == len(ref)
        assert m.decoded_length(ref) == len(raw)
        assert m.decode(ref.rstrip("="), strict=False) == raw
    assert m.encode(b"") == "" and m.decode("") == b""
    assert m.encode(b"f") == "Zg==" and m.encode(b"fo") == "Zm8=" and m.encode(b"foo") == "Zm9v"
    assert m.decode("Zm9v\nYmFy \r\n") == b"foobar"
    for bad in ("Zg=", "Z", "Zg==Zg==", "Zm=v", "Zm9*", "Zh==", "====", "Zg==Z"):
        try:
            m.decode(bad)
            raise SystemExit("accepted invalid input %r" % bad)
        except m.Base64Error:
            pass
    try:
        m.decode("Zg")
        raise SystemExit("strict mode accepted missing padding")
    except m.Base64Error:
        pass

    raw = bytes(range(256)) * 2
    ref = base64.b64encode(raw).decode("ascii")
    for size in (1, 2, 3, 4, 5, 7, 64):
        dec = m.StreamDecoder()
        got = b"".join(dec.update(ref[p:p + size]) for p in range(0, len(ref), size))
        assert got + dec.finish() == raw, size
        pieces = [ref[p:p + size] for p in range(0, len(ref), size)]
        assert b"".join(m.decode_chunks(pieces)) == raw
    for size in (3, 6, 300):
        enc = m.StreamEncoder()
        got = "".join(enc.update(raw[p:p + size]) for p in range(0, len(raw), size))
        assert got + enc.finish() == ref, size
    for size in (1, 2, 4, 5, 7, 100):
        enc = m.StreamEncoder()
        got = "".join(enc.update(raw[p:p + size]) for p in range(0, len(raw), size))
        assert got + enc.finish() == ref, size
    assert "".join(m.encode_chunks([b"a", b"bcd", b"", b"ef", b"g"])) == base64.b64encode(b"abcdefg").decode()
    enc = m.StreamEncoder()
    enc.finish()
    try:
        enc.update(b"x")
        raise SystemExit("update after finish accepted")
    except m.Base64Error:
        pass
    assert m.wrap_lines("A" * 10, 4) == "AAAA\nAAAA\nAA"
    print("ok")


if __name__ == "__main__":
    main()
