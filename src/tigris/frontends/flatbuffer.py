"""Read-only access to FlatBuffers tables without generated code.

A format built on FlatBuffers describes each table as a list of fields; a
schema module names the field slots it reads and this module resolves them.
Every offset is bounds-checked, so a truncated or hostile file raises
ValueError instead of reading past the buffer.
"""

import math
import re
import struct


class Table:
    """One FlatBuffers table at `position` inside `data`."""

    def __init__(self, data: bytes, position: int):
        self.data = data
        self.position = position
        vtable = position - self._i32(position)
        self._vtable = vtable
        self._vtable_size = self._u16(vtable)
        if self._vtable_size < 4 or self._vtable_size % 2:
            raise ValueError(f"malformed vtable at byte {vtable}")

    @classmethod
    def root(cls, data: bytes) -> "Table":
        if len(data) < 8:
            raise ValueError("buffer too short for a FlatBuffer")
        return cls(data, struct.unpack_from("<I", data, 0)[0])

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or offset + size > len(self.data):
            raise ValueError(f"FlatBuffer offset {offset} runs past the buffer")

    def _u16(self, offset: int) -> int:
        self._check(offset, 2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def _i32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<i", self.data, offset)[0]

    def _field(self, slot: int) -> int:
        """Absolute offset of field `slot`, or 0 when the field is absent."""
        entry = 4 + 2 * slot
        if entry + 2 > self._vtable_size:
            return 0
        relative = self._u16(self._vtable + entry)
        return self.position + relative if relative else 0

    def present(self) -> dict[int, int]:
        """Absolute offset of every field the table sets, by slot."""
        fields = {}
        for slot in range((self._vtable_size - 4) // 2):
            offset = self._field(slot)
            if offset:
                fields[slot] = offset
        return fields

    @property
    def size(self) -> int:
        """Bytes of the table's inline part: its vtable offset and its fields."""
        return self._u16(self._vtable + 2)

    def scalar(self, slot: int, fmt: str, default=0):
        offset = self._field(slot)
        if not offset:
            return default
        self._check(offset, struct.calcsize(fmt))
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def _indirect(self, offset: int) -> int:
        return offset + struct.unpack_from("<I", self.data, offset)[0]

    def table(self, slot: int) -> "Table | None":
        offset = self._field(slot)
        if not offset:
            return None
        self._check(offset, 4)
        return Table(self.data, self._indirect(offset))

    def _vector(self, slot: int) -> tuple[int, int] | None:
        offset = self._field(slot)
        if not offset:
            return None
        self._check(offset, 4)
        start = self._indirect(offset)
        self._check(start, 4)
        length = struct.unpack_from("<I", self.data, start)[0]
        return start + 4, length

    def vector_start(self, slot: int) -> int | None:
        """Absolute offset of the first element of vector field `slot`, or None
        when the field is absent."""
        vector = self._vector(slot)
        return None if vector is None else vector[0]

    def tables(self, slot: int) -> list["Table"]:
        vector = self._vector(slot)
        if vector is None:
            return []
        start, length = vector
        self._check(start, 4 * length)
        return [Table(self.data, self._indirect(start + 4 * i)) for i in range(length)]

    def scalars(self, slot: int, fmt: str) -> list:
        vector = self._vector(slot)
        if vector is None:
            return []
        start, length = vector
        size = struct.calcsize(fmt)
        self._check(start, size * length)
        return list(struct.unpack_from(f"<{length}{fmt}", self.data, start))

    def bytes(self, slot: int) -> bytes:
        vector = self._vector(slot)
        if vector is None:
            return b""
        start, length = vector
        self._check(start, length)
        return self.data[start:start + length]

    def string(self, slot: int) -> str:
        return self.bytes(slot).decode("utf-8", errors="replace")

    def text(self, slot: int) -> str:
        """A string field held to the format: UTF-8 followed by a zero byte."""
        vector = self._vector(slot)
        if vector is None:
            return ""
        start, length = vector
        self._check(start, length + 1)
        if self.data[start + length]:
            raise ValueError(f"the string at byte {start} is not terminated")
        try:
            return self.data[start:start + length].decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"the string at byte {start} is not UTF-8") from None


# FlexBuffers value types the readers below know.
_FLEX_NULL, _FLEX_INT, _FLEX_UINT, _FLEX_FLOAT = 0, 1, 2, 3
_FLEX_INDIRECT = {6: _FLEX_INT, 7: _FLEX_UINT, 8: _FLEX_FLOAT}
_FLEX_STRING, _FLEX_MAP, _FLEX_VECTOR, _FLEX_BOOL = 5, 9, 10, 26


def _flex_read(data: bytes, position: int, width: int, kind: int):
    if width not in (1, 2, 4, 8) or position < 0 or position + width > len(data):
        raise ValueError(f"FlexBuffer value at byte {position} runs past the buffer")
    if kind == _FLEX_FLOAT:
        if width < 4:
            raise ValueError(f"FlexBuffer float of {width} bytes")
        return struct.unpack_from("<f" if width == 4 else "<d", data, position)[0]
    letter = {1: "b", 2: "h", 4: "i", 8: "q"}[width]
    return struct.unpack_from("<" + (letter if kind == _FLEX_INT else letter.upper()), data, position)[0]


def _flex_target(data: bytes, position: int, width: int) -> int:
    return position - _flex_read(data, position, width, _FLEX_UINT)


def _flex_map(data: bytes) -> tuple[int, int, int]:
    """Where the values of a FlexBuffer whose root is a map start, how many
    there are and the byte width of each."""
    if len(data) < 3:
        raise ValueError("buffer too short for a FlexBuffer")
    root_width, packed = data[-1], data[-2]
    root = len(data) - 2 - root_width
    if packed >> 2 != _FLEX_MAP:
        raise ValueError("FlexBuffer root is not a map")
    width = 1 << (packed & 3)
    values = _flex_target(data, root, root_width)
    length = _flex_read(data, values - width, width, _FLEX_UINT)
    if values + length * (width + 1) > len(data):
        raise ValueError("FlexBuffer map runs past the buffer")
    return values, length, width


def _flex_entries(data: bytes):
    """Each key of a FlexBuffer root map with its value's type byte, position
    and the map's byte width, in key order."""
    values, length, width = _flex_map(data)
    keys = _flex_target(data, values - 3 * width, width)
    key_width = _flex_read(data, values - 2 * width, width, _FLEX_UINT)
    for i in range(length):
        start = _flex_target(data, keys + i * key_width, key_width)
        end = data.find(b"\0", start) if 0 <= start < len(data) else -1
        if end < 0:
            raise ValueError("FlexBuffer key runs past the buffer")
        yield (data[start:end].decode("utf-8", errors="replace"), data[values + length * width + i],
               values + i * width, width)


def flexbuffer_map(data: bytes) -> dict:
    """The scalar entries of a FlexBuffer whose root is a map, as TFLite stores
    custom operator options: int, unsigned, float and bool values by key, None
    for a null or a value of another type."""
    entries = {}
    for key, element, position, width in _flex_entries(data):
        kind, element_width = element >> 2, width
        if kind in _FLEX_INDIRECT:
            position = _flex_target(data, position, width)
            kind, element_width = _FLEX_INDIRECT[kind], 1 << (element & 3)
        if kind in (_FLEX_INT, _FLEX_UINT, _FLEX_FLOAT):
            value = _flex_read(data, position, element_width, kind)
        elif kind == _FLEX_BOOL:
            value = bool(_flex_read(data, position, element_width, _FLEX_UINT))
        else:
            value = None
        entries[key] = value
    return entries


def flexbuffer_slots(data: bytes, count: int) -> list[int]:
    """The first `count` value slots of a FlexBuffer whose root is a map, in
    key order, each read as a signed integer of the map's width and cut to
    32 bits, as TFLite Micro's FlexbufferWrapper reads options by position."""
    values, _, width = _flex_map(data)
    return [_int32(_flex_read(data, values + i * width, width, _FLEX_INT)) for i in range(count)]


def flexbuffer_int(data: bytes, key: str) -> int:
    """The value of `key` in a FlexBuffer root map as flexbuffers' AsInt32
    converts it: numbers and bools as stored (floats truncated), a string
    parsed as a decimal integer, an untyped vector as its length, anything
    else and a missing key as 0."""
    for name, element, position, width in _flex_entries(data):
        if name != key:
            continue
        kind, element_width = element >> 2, width
        if kind in _FLEX_INDIRECT:
            position = _flex_target(data, position, width)
            kind, element_width = _FLEX_INDIRECT[kind], 1 << (element & 3)
        if kind in (_FLEX_INT, _FLEX_UINT, _FLEX_BOOL):
            return _int32(_flex_read(data, position, element_width,
                                     _FLEX_UINT if kind == _FLEX_UINT else _FLEX_INT))
        if kind == _FLEX_FLOAT:
            value = _flex_read(data, position, element_width, _FLEX_FLOAT)
            # x86-64 converts a NaN or out-of-range double to INT64_MIN, whose low 32 bits are 0.
            return _int32(int(value)) if math.isfinite(value) and abs(value) < 2 ** 63 else 0
        if kind == _FLEX_STRING:
            start = _flex_target(data, position, width)
            size = _flex_read(data, start - (1 << (element & 3)), 1 << (element & 3), _FLEX_UINT)
            text = data[start:start + size].split(b"\0", 1)[0].decode("latin-1")
            if re.fullmatch(r"[ \t\n\v\f\r]*[+-]?[0-9]+", text) is None:
                return 0
            value = int(text)
            return _int32(value) if -2 ** 63 <= value < 2 ** 63 else 0
        if kind == _FLEX_VECTOR:
            start = _flex_target(data, position, width)
            vector_width = 1 << (element & 3)
            return _int32(_flex_read(data, start - vector_width, vector_width, _FLEX_UINT))
        return 0
    return 0


def _int32(value: int) -> int:
    return (value + 2 ** 31) % 2 ** 32 - 2 ** 31
