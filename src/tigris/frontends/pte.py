"""ExecuTorch programs (.pte) as records that can be edited and written back.

A .pte is a FlatBuffer (identifier ET12) followed by segments: blocks of
constant or delegate data that tables refer to by index. When segments follow,
an extended header after the identifier records the program's size, where the
segments start and how many bytes they take.

`read` takes every field of the ExecuTorch 1.6.0 program schema and refuses a
file that sets any other, so `write` never drops part of a program. `write`
builds the FlatBuffer table by table in the order ExecuTorch's serializer does,
with the alignment, padding and vtable sharing of the Python FlatBuffers
builder it uses, so a program read and written unchanged comes back byte for
byte; it then inserts the extended header and lays the segments out at their
recorded offsets.

Some of that layout depends on how the file was serialized: the force_align of
tensor and delegate byte vectors, the padding of the extended header, the
alignment of the segment data, and the vtable sharing rule (newer FlatBuffers
builders share a vtable only between tables of the same inline size). `read`
finds the settings that rebuild the file exactly, records them on the Program
for `write`, and refuses a file that no setting rebuilds.

A field that is absent reads as None, a field that is present but empty as an
empty list or string, and a scalar as its value or the schema default.
"""

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from tigris.frontends.flatbuffer import Table
from tigris.frontends.pte_schema import (
    DATA_ALIGNMENT, FIELDS, HEADER, HEADER_MAGIC, HEADER_OFFSET, INSTRUCTION_TYPES, MAGIC,
    SCALAR_TYPES, SEGMENT_ALIGNMENT, VALUE_TYPES, is_pte,
)


class Refusal(ValueError):
    """A program that tigris export does not rewrite. The message reads after
    the file name: "delegates to EthosUBackend; ..."."""


@dataclass
class AllocationDetails:
    memory_id: int = 0
    memory_offset_low: int = 0
    memory_offset_high: int = 0

    @property
    def offset(self) -> int:
        return self.memory_offset_high << 32 | self.memory_offset_low


@dataclass
class ExtraTensorInfo:
    mutable_data_segments_idx: int = 0
    fully_qualified_name: str | None = None
    location: int = 0
    device_type: int = 0
    device_index: int = 0


@dataclass
class Tensor:
    scalar_type: int = 0
    storage_offset: int = 0
    sizes: list | None = None
    dim_order: list | None = None
    requires_grad: bool = False
    data_buffer_idx: int = 0
    allocation_info: AllocationDetails | None = None
    layout: int = 0
    shape_dynamism: int = 0
    extra_tensor_info: ExtraTensorInfo | None = None

    @property
    def nbytes(self) -> int:
        if self.scalar_type not in SCALAR_TYPES:
            raise ValueError(f"unknown scalar type {self.scalar_type}")
        return math.prod(self.sizes or []) * SCALAR_TYPES[self.scalar_type][1]


@dataclass
class Null:
    pass


@dataclass
class Int:
    int_val: int = 0


@dataclass
class Bool:
    bool_val: bool = False


@dataclass
class Double:
    double_val: float = 0.0


@dataclass
class String:
    string_val: str | None = None


@dataclass
class IntList:
    items: list | None = None


@dataclass
class DoubleList:
    items: list | None = None


@dataclass
class BoolList:
    items: list | None = None


@dataclass
class TensorList:
    items: list | None = None  # value indices


@dataclass
class OptionalTensorList:
    items: list | None = None  # value indices, -1 for an absent tensor


@dataclass
class KernelCall:
    op_index: int = 0
    args: list | None = None  # value indices


@dataclass
class DelegateScratch:
    allocation: AllocationDetails | None = None
    size: int = 0


@dataclass
class DelegateCall:
    delegate_index: int = 0
    args: list | None = None
    scratch: list | None = None  # DelegateScratch


@dataclass
class MoveCall:
    move_from: int = 0
    move_to: int = 0


@dataclass
class JumpFalseCall:
    cond_value_index: int = 0
    destination_instruction: int = 0


@dataclass
class FreeCall:
    value_index: int = 0


@dataclass
class Frame:
    filename: str | None = None
    lineno: int = 0
    name: str | None = None
    context: str | None = None


@dataclass
class FrameList:
    items: list | None = None  # Frame


@dataclass
class Chain:
    inputs: list | None = None
    outputs: list | None = None
    instructions: list | None = None  # KernelCall, DelegateCall, ... or None
    stacktrace: list | None = None  # FrameList


@dataclass
class Operator:
    name: str | None = None
    overload: str | None = None


@dataclass
class ContainerMetadata:
    encoded_inp_str: str | None = None
    encoded_out_str: str | None = None


@dataclass
class BackendDelegateDataReference:
    location: int = 0  # 0 inline, 1 segment
    index: int = 0


@dataclass
class CompileSpec:
    key: str | None = None
    value: bytes | None = None


@dataclass
class BackendDelegate:
    id: str | None = None
    processed: BackendDelegateDataReference | None = None
    compile_specs: list | None = None  # CompileSpec


@dataclass
class NonConstBufferDevice:
    buffer_idx: int = 0
    device_type: int = 0
    device_index: int = 0


@dataclass
class ExecutionPlan:
    name: str | None = None
    container_meta_type: ContainerMetadata | None = None
    values: list | None = None  # Tensor, Int, ... or None
    inputs: list | None = None
    outputs: list | None = None
    chains: list | None = None
    operators: list | None = None
    delegates: list | None = None
    non_const_buffer_sizes: list | None = None
    non_const_buffer_device: list | None = None


@dataclass
class Buffer:
    storage: bytes | None = None


@dataclass
class BackendDelegateInlineData:
    data: bytes | None = None


@dataclass
class DataSegment:
    offset: int = 0  # from the start of the segment data
    size: int = 0


@dataclass
class SubsegmentOffsets:
    segment_index: int = 0
    offsets: list | None = None


@dataclass
class NamedData:
    key: str | None = None
    segment_index: int = 0


@dataclass
class ExtendedHeader:
    program_size: int
    segment_base_offset: int
    segment_data_size: int


@dataclass
class Program:
    """The program table, the payload of each entry of `segments`, and the
    layout it is written with: the size of the extended header with its
    padding, the alignment of Buffer and BackendDelegateInlineData bytes, the
    alignment of the segment data and the vtable sharing rule. `header` is the
    extended header as read; `write` derives its own from the layout."""
    version: int = 0
    execution_plan: list | None = None
    constant_buffer: list | None = None  # Buffer
    backend_delegate_data: list | None = None  # BackendDelegateInlineData
    segments: list | None = None  # DataSegment
    constant_segment: SubsegmentOffsets | None = None
    mutable_data_segments: list | None = None  # SubsegmentOffsets
    named_data: list | None = None  # NamedData
    segment_data: list = field(default_factory=list)  # bytes, one per segment
    header_size: int = HEADER.size
    tensor_alignment: int = DATA_ALIGNMENT
    delegate_alignment: int = DATA_ALIGNMENT
    segment_alignment: int = SEGMENT_ALIGNMENT
    vtables_by_size: bool = False
    header: ExtendedHeader | None = None

    def plan(self, name: str = "forward") -> ExecutionPlan:
        for plan in self.execution_plan or ():
            if plan.name == name:
                return plan
        raise ValueError(f"the program has no method {name!r}")


# Each record's table and its fields as (attribute, slot, format), in slot
# order. Formats: a struct letter for a scalar, "s" a string, "[" plus a letter
# a vector of scalars, "bytes" a byte vector, "tensor data" and "delegate data"
# a byte vector at the program's tensor or delegate alignment, a record class a
# table, [class] a vector of tables, and "values" or "instructions" a vector of
# union tables.
_SPEC = {
    AllocationDetails: ("allocation details", (
        ("memory_id", 0, "I"), ("memory_offset_low", 1, "I"), ("memory_offset_high", 2, "I"))),
    ExtraTensorInfo: ("extra tensor info", (
        ("mutable_data_segments_idx", 0, "Q"), ("fully_qualified_name", 1, "s"),
        ("location", 2, "b"), ("device_type", 3, "b"), ("device_index", 4, "b"))),
    Tensor: ("tensor", (
        ("scalar_type", 0, "b"), ("storage_offset", 1, "i"), ("sizes", 2, "[i"),
        ("dim_order", 3, "[B"), ("requires_grad", 4, "?"), ("data_buffer_idx", 5, "I"),
        ("allocation_info", 6, AllocationDetails), ("layout", 7, "b"),
        ("shape_dynamism", 8, "b"), ("extra_tensor_info", 9, ExtraTensorInfo))),
    Null: ("null", ()),
    Int: ("int", (("int_val", 0, "q"),)),
    Bool: ("bool", (("bool_val", 0, "?"),)),
    Double: ("double", (("double_val", 0, "d"),)),
    String: ("string", (("string_val", 0, "s"),)),
    IntList: ("int list", (("items", 0, "[q"),)),
    DoubleList: ("double list", (("items", 0, "[d"),)),
    BoolList: ("bool list", (("items", 0, "[?"),)),
    TensorList: ("tensor list", (("items", 0, "[i"),)),
    OptionalTensorList: ("optional tensor list", (("items", 0, "[i"),)),
    KernelCall: ("kernel call", (("op_index", 0, "i"), ("args", 1, "[i"))),
    DelegateScratch: ("delegate scratch", (
        ("allocation", 0, AllocationDetails), ("size", 1, "Q"))),
    DelegateCall: ("delegate call", (
        ("delegate_index", 0, "i"), ("args", 1, "[i"), ("scratch", 2, [DelegateScratch]))),
    MoveCall: ("move call", (("move_from", 0, "i"), ("move_to", 1, "i"))),
    JumpFalseCall: ("jump false call", (
        ("cond_value_index", 0, "i"), ("destination_instruction", 1, "i"))),
    FreeCall: ("free call", (("value_index", 0, "i"),)),
    Frame: ("frame", (
        ("filename", 0, "s"), ("lineno", 1, "i"), ("name", 2, "s"), ("context", 3, "s"))),
    FrameList: ("frame list", (("items", 0, [Frame]),)),
    Chain: ("chain", (
        ("inputs", 0, "[i"), ("outputs", 1, "[i"), ("instructions", 2, "instructions"),
        ("stacktrace", 3, [FrameList]))),
    Operator: ("operator", (("name", 0, "s"), ("overload", 1, "s"))),
    ContainerMetadata: ("container metadata", (
        ("encoded_inp_str", 0, "s"), ("encoded_out_str", 1, "s"))),
    BackendDelegateDataReference: ("delegate data reference", (
        ("location", 0, "b"), ("index", 1, "I"))),
    CompileSpec: ("compile spec", (("key", 0, "s"), ("value", 1, "bytes"))),
    BackendDelegate: ("backend delegate", (
        ("id", 0, "s"), ("processed", 1, BackendDelegateDataReference),
        ("compile_specs", 2, [CompileSpec]))),
    NonConstBufferDevice: ("non-const buffer device", (
        ("buffer_idx", 0, "i"), ("device_type", 1, "b"), ("device_index", 2, "b"))),
    ExecutionPlan: ("execution plan", (
        ("name", 0, "s"), ("container_meta_type", 1, ContainerMetadata),
        ("values", 2, "values"), ("inputs", 3, "[i"), ("outputs", 4, "[i"),
        ("chains", 5, [Chain]), ("operators", 6, [Operator]),
        ("delegates", 7, [BackendDelegate]), ("non_const_buffer_sizes", 8, "[q"),
        ("non_const_buffer_device", 9, [NonConstBufferDevice]))),
    Buffer: ("buffer", (("storage", 0, "tensor data"),)),
    BackendDelegateInlineData: ("inline data", (("data", 0, "delegate data"),)),
    DataSegment: ("data segment", (("offset", 0, "Q"), ("size", 1, "Q"))),
    SubsegmentOffsets: ("subsegment offsets", (
        ("segment_index", 0, "I"), ("offsets", 1, "[Q"))),
    NamedData: ("named data", (("key", 0, "s"), ("segment_index", 1, "I"))),
    Program: ("program", (
        ("version", 0, "I"), ("execution_plan", 1, [ExecutionPlan]),
        ("constant_buffer", 2, [Buffer]),
        ("backend_delegate_data", 3, [BackendDelegateInlineData]),
        ("segments", 4, [DataSegment]), ("constant_segment", 5, SubsegmentOffsets),
        ("mutable_data_segments", 6, [SubsegmentOffsets]),
        ("named_data", 7, [NamedData]))),
}
# The members of each union, by type index; index 0 is the empty member.
_UNIONS = {
    "values": ("evalue", (None,) + tuple(
        globals()[name] for name in VALUE_TYPES)),
    "instructions": ("instruction", (None,) + tuple(
        globals()[name] for name in INSTRUCTION_TYPES)),
}
_NUMPY = {"b": "i1", "B": "u1", "?": "u1", "i": "<i4", "I": "<u4", "q": "<i8", "Q": "<u8",
          "d": "<f8"}
_SCALARS = frozenset("bB?iIqQd")


def _is_scalar(fmt) -> bool:
    return isinstance(fmt, str) and fmt in _SCALARS


def _check_fields(table: Table, kind: str) -> dict[int, int]:
    present = table.present()
    unknown = sorted(slot for slot in present if slot >= FIELDS[kind])
    if unknown:
        raise ValueError(f"the {kind} table sets field {unknown[0]}, past the {FIELDS[kind]} "
                         f"fields of the ExecuTorch 1.6 schema")
    return present


def _union(table: Table, members: tuple, kind: str):
    _check_fields(table, kind)
    index = table.scalar(0, "B")
    member = table.table(1)
    if index >= len(members):
        raise ValueError(f"the {kind} table holds union type {index}, past the {len(members) - 1} "
                         f"types of the ExecuTorch 1.6 schema")
    if (index == 0) != (member is None):
        raise ValueError(f"the {kind} table's union type and value disagree")
    return None if member is None else _record(member, members[index])


def _field(table: Table, present: dict, slot: int, fmt, kind: str):
    if isinstance(fmt, type):
        child = table.table(slot)
        return None if child is None else _record(child, fmt)
    if slot not in present:
        return None if not _is_scalar(fmt) else {"?": False, "d": 0.0}.get(fmt, 0)
    if isinstance(fmt, list):
        return [_record(t, fmt[0]) for t in table.tables(slot)]
    if fmt in _UNIONS:
        union_kind, members = _UNIONS[fmt]
        return [_union(t, members, union_kind) for t in table.tables(slot)]
    if fmt == "s":
        return table.text(slot)
    if fmt in ("bytes", "tensor data", "delegate data"):
        return table.bytes(slot)
    if fmt.startswith("["):
        values = table.scalars(slot, "B" if fmt == "[?" else fmt[1:])
        if fmt == "[?":
            if any(v > 1 for v in values):
                raise ValueError(f"a bool of the {kind} table is neither 0 nor 1")
            values = [bool(v) for v in values]
        return values
    if fmt == "?":
        value = table.scalar(slot, "B")
        if value > 1:
            raise ValueError(f"a bool of the {kind} table is neither 0 nor 1")
        return bool(value)
    return table.scalar(slot, fmt)


def _record(table: Table, cls):
    kind, fields = _SPEC[cls]
    present = _check_fields(table, kind)
    return cls(**{attr: _field(table, present, slot, fmt, kind) for attr, slot, fmt in fields})


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _power_of_two(value: int) -> bool:
    return value > 0 and not value & (value - 1)


def _header(data: bytes) -> ExtendedHeader | None:
    if data[HEADER_OFFSET:HEADER_OFFSET + 4] != HEADER_MAGIC:
        return None
    if len(data) < HEADER_OFFSET + HEADER.size:
        raise ValueError("the file ends inside the extended header")
    _, length, program_size, base, size = HEADER.unpack_from(data, HEADER_OFFSET)
    if length != HEADER.size:
        raise ValueError(f"an extended header of {length} bytes, where ExecuTorch 1.6 writes "
                         f"{HEADER.size}")
    if not HEADER_OFFSET + HEADER.size < program_size <= base or base + size != len(data):
        raise ValueError(f"the extended header (program {program_size} bytes, segments at "
                         f"{base}, {size} bytes) does not match the file of {len(data)} bytes")
    if size == 0:
        raise ValueError("an extended header with no segment data; ExecuTorch 1.6 writes one only "
                         "before segments")
    if any(data[program_size:base]):
        raise ValueError("the bytes between the program and its segments are not zero")
    return ExtendedHeader(program_size, base, size)


def _segment_alignment(program: Program, header: ExtendedHeader) -> int:
    """The largest alignment the segment data can have been laid out for.
    ExecuTorch aligns every segment to one alignment and a named-data segment
    to a multiple of it, and aligns the segment data to the largest of them; a
    segment at offset 0 bounds neither."""
    base = header.segment_base_offset
    top = base & -base

    def bound(offset: int) -> int:
        return min(top, offset & -offset) if offset else top

    named = {n.segment_index for n in program.named_data or ()}
    segments = list(enumerate(program.segments or ()))
    shared = min((bound(s.offset) for i, s in segments if i not in named), default=top)
    alignment = max([shared] + [bound(s.offset) for i, s in segments if i in named])
    if _align(header.program_size, alignment) != base:
        raise ValueError(f"the segment data at {base} is not at the first aligned offset "
                         f"after the program of {header.program_size} bytes")
    return alignment


def _segments(program: Program, data: bytes, header: ExtendedHeader | None) -> None:
    records = program.segments or []
    if header is None:
        if any(s.size for s in records):
            raise ValueError("the program has segments but no extended header")
        program.segment_data = [b""] * len(records)
        return
    base, end = header.segment_base_offset, len(data)
    cursor = base
    for s in records:
        start = base + s.offset
        if start < cursor or start + s.size > end:
            raise ValueError(f"segment at {s.offset} ({s.size} bytes) overlaps another "
                             f"or runs past the file")
        if any(data[cursor:start]):
            raise ValueError("the padding between segments is not zero")
        program.segment_data.append(data[start:start + s.size])
        cursor = start + s.size
    if cursor != end:
        raise ValueError(f"{end - cursor} bytes follow the last segment")
    program.segment_alignment = _segment_alignment(program, header)


def _alignments(root: Table, slot: int) -> list[int]:
    """The alignments the byte vectors of the program's tables in `slot` can
    have been written with: the schema's first, then from the largest that
    every vector start satisfies down to 1."""
    starts = [start for table in root.tables(slot)
              if (start := table.vector_start(0)) is not None]
    if not starts:
        return [DATA_ALIGNMENT]
    common = 0
    for start in starts:
        common |= start
    largest = common & -common
    ordered = [DATA_ALIGNMENT] if DATA_ALIGNMENT <= largest else []
    while largest:
        if largest != DATA_ALIGNMENT:
            ordered.append(largest)
        largest >>= 1
    return ordered


def _program_bytes(flat: bytes, header_size: int, header: ExtendedHeader) -> bytes:
    """The FlatBuffer with the extended header, padded to `header_size`, after
    its identifier."""
    root = struct.unpack_from("<I", flat, 0)[0] + header_size
    raw = HEADER.pack(HEADER_MAGIC, HEADER.size, header.program_size,
                      header.segment_base_offset, header.segment_data_size)
    return b"".join((struct.pack("<I", root), flat[4:8], raw.ljust(header_size, b"\0"),
                     flat[8:]))


def read(data: bytes, layout: bool = True) -> Program:
    """The program in `data` as records; raises ValueError for a truncated or
    malformed file and for one that sets a field `write` does not reproduce,
    and Refusal, a ValueError, for a layout `write` does not reproduce. With
    `layout` False the records come back without that layout, for a reader
    that does not write the program."""
    data = bytes(data)
    if not is_pte(data):
        raise ValueError(f"not an ExecuTorch program (no {MAGIC.decode()} identifier)")
    header = _header(data)
    program_data = data[:header.program_size] if header else data
    root = Table.root(program_data)
    program = _record(root, Program)
    program.header = header
    _segments(program, data, header)
    if not layout:
        return program
    for by_size in (False, True):
        for tensor in _alignments(root, 2):
            for delegate in _alignments(root, 3):
                program.vtables_by_size = by_size
                program.tensor_alignment, program.delegate_alignment = tensor, delegate
                flat = _flatbuffer(program)
                if header is None:
                    if flat == data:
                        program.header_size = max(HEADER.size, tensor, delegate)
                        return program
                    continue
                size = header.program_size - len(flat)
                if (_header_fits(size, tensor, delegate)
                        and _program_bytes(flat, size, header) == program_data):
                    program.header_size = size
                    return program
    raise Refusal("is laid out differently from the programs ExecuTorch 1.6 writes; tigris "
                  "export rewrites programs it writes back byte for byte")


def _header_fits(size: int, tensor_alignment: int, delegate_alignment: int) -> bool:
    """Whether a header padded to `size` bytes keeps the FlatBuffer after it
    aligned."""
    return _power_of_two(size) and size >= max(HEADER.size, tensor_alignment,
                                               delegate_alignment)


class _Builder:
    """A FlatBuffer written back to front with the alignment, padding and vtable
    sharing of the Python FlatBuffers builder, independent of the installed
    flatbuffers package. Chunks are kept in write order and joined reversed."""

    def __init__(self, vtables_by_size: bool):
        self.chunks: list[bytes] = []
        self.size = 0
        self.minalign = 1
        self.vtables: dict[tuple, int] = {}
        self.by_size = vtables_by_size
        self.fields: list[int] = []
        self.object_end = 0

    def _put(self, raw: bytes) -> None:
        self.chunks.append(raw)
        self.size += len(raw)

    def prep(self, size: int, additional: int) -> None:
        """Pads so that `size`-aligned data fits after `additional` more bytes."""
        self.minalign = max(self.minalign, size)
        pad = -(self.size + additional) & (size - 1)
        if pad:
            self._put(bytes(pad))

    def scalar(self, fmt: str, value) -> None:
        try:
            raw = struct.pack("<" + fmt, value)
        except struct.error:
            raise ValueError(f"value {value!r} does not fit a field of format {fmt}") from None
        self.prep(len(raw), 0)
        self._put(raw)

    def uoffset(self, target: int) -> None:
        self.prep(4, 0)
        self._put(struct.pack("<I", self.size - target + 4))

    def vector(self, raw: bytes, count: int, alignment: int) -> int:
        if not _power_of_two(alignment):
            raise ValueError(f"a vector alignment of {alignment} is not a power of two")
        self.prep(4, len(raw))
        self.prep(alignment, len(raw))
        self._put(raw)
        self._put(struct.pack("<I", count))
        return self.size

    def string(self, value: str) -> int:
        raw = value.encode("utf-8")
        self.prep(4, len(raw) + 1)
        self._put(raw + b"\0")
        self._put(struct.pack("<I", len(raw)))
        return self.size

    def offsets(self, targets: list[int]) -> int:
        self.prep(4, 4 * len(targets))
        for target in reversed(targets):
            self.uoffset(target)
        self._put(struct.pack("<I", len(targets)))
        return self.size

    def start(self, count: int) -> None:
        self.fields = [0] * count
        self.object_end = self.size

    def add_scalar(self, slot: int, fmt: str, value, default) -> None:
        if value != default:
            self.scalar(fmt, value)
            self.fields[slot] = self.size

    def add_offset(self, slot: int, target: int) -> None:
        self.uoffset(target)
        self.fields[slot] = self.size

    def end(self) -> int:
        self.prep(4, 0)
        patch = len(self.chunks)
        self._put(bytes(4))
        position = self.size
        count = len(self.fields)
        while count and not self.fields[count - 1]:
            count -= 1
        entries = [position - f if f else 0 for f in self.fields[:count]]
        object_size = position - self.object_end
        key = (*entries, object_size) if self.by_size else tuple(entries)
        shared = self.vtables.get(key)
        if shared is None:
            self._put(struct.pack(f"<{count + 2}H", 2 * (count + 2), object_size, *entries))
            self.vtables[key] = shared = self.size
            soffset = self.size - position
        else:
            soffset = shared - position
        self.chunks[patch] = struct.pack("<i", soffset)
        return position

    def finish(self, root: int, identifier: bytes) -> bytes:
        self.prep(self.minalign, 8)
        self.prep(4, 4)
        self._put(identifier)
        self.uoffset(root)
        return b"".join(reversed(self.chunks))


class _Writer:
    def __init__(self, program: Program):
        self.b = _Builder(program.vtables_by_size)
        self.alignment = {"tensor data": program.tensor_alignment,
                          "delegate data": program.delegate_alignment, "bytes": 1}

    def offset(self, value, fmt) -> int:
        b = self.b
        if isinstance(fmt, type):
            return self.record(value)
        if isinstance(fmt, list):
            return b.offsets([self.record(v) for v in value])
        if fmt in _UNIONS:
            return b.offsets([self.union(v, _UNIONS[fmt]) for v in value])
        if fmt == "s":
            return b.string(value)
        if fmt in self.alignment:
            return b.vector(bytes(value), len(value), self.alignment[fmt])
        array = np.asarray(value, _NUMPY[fmt[1:]]).reshape(-1)
        return b.vector(array.tobytes(), array.size, array.itemsize)

    def union(self, value, union) -> int:
        kind, members = union
        child = None if value is None else self.record(value)
        self.b.start(FIELDS[kind])
        self.b.add_scalar(0, "B", 0 if value is None else members.index(type(value)), 0)
        if child is not None:
            self.b.add_offset(1, child)
        return self.b.end()

    def record(self, rec) -> int:
        """The table of `rec`: its offset fields built in slot order, then the
        table with every field added in slot order, defaults left out."""
        kind, fields = _SPEC[type(rec)]
        offsets = {}
        for attr, slot, fmt in fields:
            value = getattr(rec, attr)
            if not _is_scalar(fmt) and value is not None:
                offsets[slot] = self.offset(value, fmt)
        b = self.b
        b.start(FIELDS[kind])
        for attr, slot, fmt in fields:
            if slot in offsets:
                b.add_offset(slot, offsets[slot])
            elif _is_scalar(fmt):
                b.add_scalar(slot, fmt, getattr(rec, attr), False if fmt == "?" else 0)
        return b.end()


def _flatbuffer(program: Program) -> bytes:
    w = _Writer(program)
    return w.b.finish(w.record(program), MAGIC)


def _segment_data(program: Program) -> bytes:
    records = program.segments or []
    if len(records) != len(program.segment_data):
        raise ValueError(f"{len(records)} segment records but {len(program.segment_data)} "
                         f"segment payloads")
    out = bytearray()
    for s, payload in zip(records, program.segment_data):
        if len(payload) != s.size:
            raise ValueError(f"segment at {s.offset} records {s.size} bytes but holds "
                             f"{len(payload)}")
        if s.offset < len(out):
            raise ValueError(f"segment at {s.offset} overlaps the one before it")
        out += bytes(s.offset - len(out)) + payload
    return bytes(out)


def write(program: Program) -> bytes:
    """The program as a .pte file."""
    flat = _flatbuffer(program)
    segments = _segment_data(program)
    if not segments:
        return flat
    size, alignment = program.header_size, program.segment_alignment
    if not _header_fits(size, program.tensor_alignment, program.delegate_alignment):
        raise ValueError(f"an extended header padded to {size} bytes does not keep the "
                         f"program aligned")
    if not _power_of_two(alignment):
        raise ValueError(f"a segment alignment of {alignment} is not a power of two")
    program_size = size + len(flat)
    header = ExtendedHeader(program_size, _align(program_size, alignment), len(segments))
    return b"".join((_program_bytes(flat, size, header),
                     bytes(header.segment_base_offset - program_size), segments))
