"""Read-only access to FlatBuffers tables without generated code.

A format built on FlatBuffers describes each table as a list of fields; a
schema module names the field slots it reads and this module resolves them.
Every offset is bounds-checked, so a truncated or hostile file raises
ValueError instead of reading past the buffer.
"""

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
