"""Base64 encoding and decoding with streaming support.

A small, dependency-free implementation of the standard base64 alphabet
(RFC 4648, section 4).  Besides the one-shot :func:`encode` and
:func:`decode` helpers it offers :class:`StreamEncoder` and
:class:`StreamDecoder`, which accept input in arbitrary chunks and produce
exactly the output a one-shot call over the concatenated input would give.

Decoding ignores ASCII whitespace, so wrapped text (for example 76-column
MIME bodies) can be fed directly.  In strict mode the final quantum must be
padded with ``=``; in lenient mode missing padding is accepted.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
PAD = "="

_DECODE_MAP = {symbol: value for value, symbol in enumerate(ALPHABET)}
_WHITESPACE = " \t\r\n"


class Base64Error(ValueError):
    """Raised for malformed base64 input or misuse of a finished stream."""


def encoded_length(byte_count: int) -> int:
    """Return the number of characters ``byte_count`` bytes encode to."""
    if byte_count < 0:
        raise ValueError("byte_count must not be negative")
    return 4 * ((byte_count + 2) // 3)


def _encode_block(block: bytes) -> str:
    """Encode a block whose length is a multiple of three, without padding."""
    out: List[str] = []
    for pos in range(0, len(block), 3):
        n = (block[pos] << 16) | (block[pos + 1] << 8) | block[pos + 2]
        out.append(ALPHABET[n >> 18])
        out.append(ALPHABET[(n >> 12) & 0x3F])
        out.append(ALPHABET[(n >> 6) & 0x3F])
        out.append(ALPHABET[n & 0x3F])
    return "".join(out)


def _encode_tail(tail: bytes) -> str:
    """Encode the last zero, one or two bytes of a message with padding."""
    if not tail:
        return ""
    if len(tail) == 1:
        n = tail[0] << 16
        return ALPHABET[n >> 18] + ALPHABET[(n >> 12) & 0x3F] + PAD * 2
    if len(tail) == 2:
        n = (tail[0] << 16) | (tail[1] << 8)
        return (
            ALPHABET[n >> 18]
            + ALPHABET[(n >> 12) & 0x3F]
            + ALPHABET[(n >> 6) & 0x3F]
            + PAD
        )
    raise ValueError("tail must be shorter than three bytes")


def encode(data: bytes) -> str:
    """Encode ``data`` in one call."""
    cut = len(data) - len(data) % 3
    return _encode_block(data[:cut]) + _encode_tail(data[cut:])


class StreamEncoder:
    """Incremental encoder.

    Call :meth:`update` for every chunk and :meth:`finish` once at the end;
    the concatenation of all returned strings is the encoded message.
    """

    def __init__(self) -> None:
        self._pending = b""
        self._finished = False

    def update(self, data: bytes) -> str:
        """Encode as much of the input seen so far as forms whole triples."""
        if self._finished:
            raise Base64Error("encoder already finished")
        buf = self._pending + data
        cut = len(buf) - len(buf) % 3
        self._pending = data[cut:]
        return _encode_block(buf[:cut])

    def finish(self) -> str:
        """Encode the bytes still held back and add padding."""
        if self._finished:
            raise Base64Error("encoder already finished")
        self._finished = True
        tail, self._pending = self._pending, b""
        return _encode_tail(tail)


def _decode_quantum(quad: str, final: bool) -> bytes:
    """Decode one four-character quantum.

    Padding is accepted only in the final quantum.  The bits that padding
    leaves unused must be zero.
    """
    if len(quad) != 4:
        raise Base64Error("a quantum has exactly four characters")
    body = quad.rstrip(PAD)
    pad = len(quad) - len(body)
    if pad > 2:
        raise Base64Error("too much padding")
    if pad and not final:
        raise Base64Error("padding inside the data")
    n = 0
    for symbol in body:
        value = _DECODE_MAP.get(symbol)
        if value is None:
            raise Base64Error(f"invalid character {symbol!r}")
        n = (n << 6) | value
    n <<= 6 * pad
    raw = n.to_bytes(3, "big")
    if any(raw[3 - pad:]):
        raise Base64Error("non-zero bits in the padded quantum")
    return raw[: 3 - pad]


class StreamDecoder:
    """Incremental decoder, the counterpart of :class:`StreamEncoder`."""

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict
        self._pending = ""
        self._ended = False
        self._finished = False

    def update(self, text: str) -> bytes:
        """Decode every complete quantum available so far."""
        if self._finished:
            raise Base64Error("decoder already finished")
        chars = self._pending + "".join(ch for ch in text if ch not in _WHITESPACE)
        cut = len(chars) - len(chars) % 4
        out = bytearray()
        for pos in range(0, cut, 4):
            if self._ended:
                raise Base64Error("data after padding")
            quad = chars[pos:pos + 4]
            if PAD in quad:
                self._ended = True
                out += _decode_quantum(quad, final=True)
            else:
                out += _decode_quantum(quad, final=False)
        self._pending = chars[cut:]
        return bytes(out)

    def finish(self) -> bytes:
        """Flush the decoder and validate the end of the message."""
        if self._finished:
            raise Base64Error("decoder already finished")
        self._finished = True
        rest, self._pending = self._pending, ""
        if not rest:
            return b""
        if self._ended:
            raise Base64Error("data after padding")
        if self._strict or len(rest) == 1 or PAD in rest:
            raise Base64Error("truncated base64 input")
        return _decode_quantum(rest + PAD * (4 - len(rest)), final=True)


def decode(text: str, strict: bool = True) -> bytes:
    """Decode ``text`` in one call."""
    decoder = StreamDecoder(strict=strict)
    head = decoder.update(text)
    return head + decoder.finish()


def decoded_length(text: str) -> int:
    """Return the number of bytes a padded base64 string decodes to."""
    chars = "".join(ch for ch in text if ch not in _WHITESPACE)
    if len(chars) % 4:
        raise Base64Error("length is not a multiple of four")
    pad = len(chars) - len(chars.rstrip(PAD))
    if pad > 2:
        raise Base64Error("too much padding")
    return 3 * (len(chars) // 4) - pad


def encode_chunks(chunks: Iterable[bytes]) -> Iterator[str]:
    """Lazily encode an iterable of byte chunks."""
    encoder = StreamEncoder()
    for chunk in chunks:
        piece = encoder.update(chunk)
        if piece:
            yield piece
    last = encoder.finish()
    if last:
        yield last


def decode_chunks(chunks: Iterable[str], strict: bool = True) -> Iterator[bytes]:
    """Lazily decode an iterable of text chunks."""
    decoder = StreamDecoder(strict=strict)
    for chunk in chunks:
        piece = decoder.update(chunk)
        if piece:
            yield piece
    last = decoder.finish()
    if last:
        yield last


def wrap_lines(encoded: str, width: int = 76) -> str:
    """Break an encoded string into lines of at most ``width`` characters."""
    if width <= 0:
        raise ValueError("width must be a positive integer")
    return "\n".join(encoded[pos:pos + width] for pos in range(0, len(encoded), width))
