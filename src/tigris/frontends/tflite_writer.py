"""TFLite models as records that can be edited and written back.

`read` takes every field of the TFLite schema it knows and refuses a file that
sets any other, so `write` never drops part of a model. `write` serializes the
records with the FlatBuffers builder. An operator options table that holds only
scalars is copied field by field from the source file; the few option tables
that hold vectors or strings are read into values and rebuilt.
"""

import struct
from dataclasses import dataclass, field

import flatbuffers
import numpy as np

from tigris.frontends.flatbuffer import Table
from tigris.frontends.tflite import (
    _B_DATA, _B_OFFSET, _B_SIZE, _MODEL_BUFFERS, _MODEL_DESCRIPTION, _MODEL_OPCODES,
    _MODEL_SUBGRAPHS, _MODEL_VERSION, _OC_CODE, _OC_CUSTOM, _OC_DEPRECATED_CODE, _OC_VERSION,
    _OP_CUSTOM_OPTIONS, _OP_INPUTS, _OP_OPCODE, _OP_OPTIONS, _OP_OUTPUTS, _Q_DIMENSION, _Q_SCALE,
    _Q_ZERO_POINT, _SG_INPUTS, _SG_NAME, _SG_OPERATORS, _SG_OUTPUTS, _SG_TENSORS, _T_BUFFER,
    _T_NAME, _T_QUANT, _T_SHAPE, _T_TYPE, _T_VARIABLE, MAGIC, is_tflite,
)

# Field slots in the TFLite schema beyond those the reader names.
_MODEL_METADATA_BUFFER, _MODEL_METADATA, _MODEL_SIGNATURES = 5, 6, 7
_SG_DEBUG = 5
_T_SHAPE_SIGNATURE, _T_HAS_RANK = 7, 8
_Q_MIN, _Q_MAX, _Q_DETAILS_TYPE, _Q_DETAILS = 0, 1, 4, 5
_OP_OPTIONS_TYPE, _OP_CUSTOM_FORMAT, _OP_MUTATING, _OP_INTERMEDIATES = 3, 6, 7, 8
_OP_OPTIONS2_TYPE, _OP_OPTIONS2, _OP_DEBUG = 11, 12, 13
_META_NAME, _META_BUFFER = 0, 1
_SIG_INPUTS, _SIG_OUTPUTS, _SIG_KEY, _SIG_TAG, _SIG_SUBGRAPH = 0, 1, 2, 3, 4
_MAP_NAME, _MAP_TENSOR = 0, 1

# The slots each table of the schema has that this module reads and writes.
_KNOWN = {
    "model": {0, 1, 2, 3, 4, 5, 6, 7},
    "subgraph": {0, 1, 2, 3, 4, 5},
    "tensor": {0, 1, 2, 3, 4, 5, 7, 8},
    "quantization": {0, 1, 2, 3, 4, 5, 6},
    "operator": {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13},
    "operator code": {0, 1, 2, 3},
    "buffer": {0, 1, 2},
    "metadata": {0, 1},
    "signature": {0, 1, 2, 3, 4},
    "tensor map": {0, 1},
    "custom quantization": {0},
}

# Field count of each BuiltinOptions and BuiltinOptions2 table, by union type.
_OPTION_FIELDS = bytes((
    0, 7, 7, 3, 1, 6, 3, 2, 6, 1, 2, 2, 1, 4, 5, 4, 1, 1, 3, 1, 1, 1, 0, 2, 0, 0, 0, 1, 2, 1,
    1, 3, 6, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 1, 0, 1, 4, 2,
    0, 1, 0, 0, 2, 0, 0, 0, 0, 6, 4, 6, 0, 0, 2, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 2, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
))
_OPTION2_FIELDS = bytes((0, 1, 1, 3, 17, 6, 2, 7, 2, 1, 3, 1, 5, 6, 3, 2, 6, 1, 0, 1, 1, 5, 0, 1))
# Option tables with vectors or strings: (slot, format) per field, "[" vector, "s" string.
_VECTOR_OPTIONS = {
    3: ((0, "i"), (1, "[i"), (2, "[i")),  # ConcatEmbeddingsOptions
    8: ((0, "b"), (1, "b"), (2, "?"), (3, "?"), (4, "b"), (5, "[B")),  # FullyConnectedOptions
    17: ((0, "[i"),),  # ReshapeOptions
    30: ((0, "[i"),),  # SqueezeOptions
    111: ((0, "s"), (1, "s")),  # VarHandleOptions
    115: ((0, "[f"),),  # BucketizeOptions
}
# BuiltinOptions2 tables holding vectors or strings; read refuses them.
_VECTOR_OPTIONS2 = frozenset((2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 16, 17, 21, 23))
_CUSTOM_QUANTIZATION = 1
_NUMPY_FORMATS = {"i": "<i4", "q": "<i8", "f": "<f4", "?": "?", "B": "u1"}


@dataclass
class Options:
    """A builtin options table: its union type, the source table its scalar
    fields are copied from, and fields set over it by slot as (format, value)."""
    kind: int
    source: Table | None = None
    fields: dict = field(default_factory=dict)


@dataclass
class Quantization:
    scale: list | None = None
    zero_point: list | None = None
    quantized_dimension: int = 0
    min: list | None = None
    max: list | None = None
    custom: bytes | None = None


@dataclass
class Tensor:
    shape: list | None
    type: int
    buffer: int = 0
    name: bytes | None = None
    quantization: Quantization | None = None
    is_variable: bool = False
    shape_signature: list | None = None
    has_rank: bool = False


@dataclass
class Operator:
    opcode: int
    inputs: list
    outputs: list
    options: Options | None = None
    custom_options: bytes | None = None
    custom_options_format: int = 0
    mutating_variable_inputs: list | None = None
    intermediates: list | None = None
    options2: Options | None = None
    debug_metadata_index: int | None = None


@dataclass
class Subgraph:
    tensors: list
    inputs: list
    outputs: list
    operators: list
    name: bytes | None = None
    debug_metadata_index: int | None = None


@dataclass
class OperatorCode:
    builtin_code: int
    version: int = 1
    custom_code: bytes | None = None
    deprecated_builtin_code: int = 0


@dataclass
class Signature:
    inputs: list  # (name, tensor index)
    outputs: list
    key: bytes | None = None
    tag: bytes | None = None
    subgraph_index: int = 0


@dataclass
class Model:
    version: int
    operator_codes: list
    subgraphs: list
    buffers: list  # bytes, or None for a buffer without data
    description: bytes | None = None
    metadata_buffer: list | None = None
    metadata: list | None = None  # (name, buffer index)
    signatures: list | None = None


def _check_fields(table: Table, kind: str) -> dict[int, int]:
    present = table.present()
    unknown = sorted(set(present) - _KNOWN[kind])
    if unknown:
        raise ValueError(f"the {kind} table sets field {unknown[0]}, which is not supported")
    return present


def _vector(table: Table, present: dict, slot: int, fmt: str) -> list | None:
    return table.scalars(slot, fmt) if slot in present else None


def _string(table: Table, present: dict, slot: int) -> bytes | None:
    return table.bytes(slot) if slot in present else None


def _options(table: Table | None, kind: int, second: bool) -> Options | None:
    if table is None:
        return None
    counts = _OPTION2_FIELDS if second else _OPTION_FIELDS
    if not 0 < kind < len(counts) or (second and kind in _VECTOR_OPTIONS2):
        raise ValueError(f"operator options of type {kind} are not supported")
    present = table.present()
    if any(slot >= counts[kind] for slot in present):
        raise ValueError(f"operator options of type {kind} set an unknown field")
    if second or kind not in _VECTOR_OPTIONS:
        return Options(kind, table)
    fields = {}
    for slot, fmt in _VECTOR_OPTIONS[kind]:
        if slot not in present:
            continue
        if fmt == "s":
            fields[slot] = (fmt, table.bytes(slot))
        elif fmt.startswith("["):
            fields[slot] = (fmt, table.scalars(slot, fmt[1:]))
        else:
            fields[slot] = (fmt, table.scalar(slot, fmt))
    return Options(kind, None, fields)


def _quantization(table: Table | None) -> Quantization | None:
    if table is None:
        return None
    present = _check_fields(table, "quantization")
    custom = None
    details_type = table.scalar(_Q_DETAILS_TYPE, "B")
    if details_type or _Q_DETAILS in present:
        details = table.table(_Q_DETAILS)
        if details_type != _CUSTOM_QUANTIZATION or details is None:
            raise ValueError(f"quantization details of type {details_type} are not supported")
        _check_fields(details, "custom quantization")
        custom = details.bytes(0)
    return Quantization(
        scale=_vector(table, present, _Q_SCALE, "f"),
        zero_point=_vector(table, present, _Q_ZERO_POINT, "q"),
        quantized_dimension=table.scalar(_Q_DIMENSION, "i"),
        min=_vector(table, present, _Q_MIN, "f"),
        max=_vector(table, present, _Q_MAX, "f"),
        custom=custom)


def _tensor(table: Table) -> Tensor:
    present = _check_fields(table, "tensor")
    return Tensor(
        shape=_vector(table, present, _T_SHAPE, "i"),
        type=table.scalar(_T_TYPE, "b"),
        buffer=table.scalar(_T_BUFFER, "I"),
        name=_string(table, present, _T_NAME),
        quantization=_quantization(table.table(_T_QUANT)),
        is_variable=bool(table.scalar(_T_VARIABLE, "B")),
        shape_signature=_vector(table, present, _T_SHAPE_SIGNATURE, "i"),
        has_rank=bool(table.scalar(_T_HAS_RANK, "B")))


def _operator(table: Table) -> Operator:
    present = _check_fields(table, "operator")
    return Operator(
        opcode=table.scalar(_OP_OPCODE, "I"),
        inputs=table.scalars(_OP_INPUTS, "i"),
        outputs=table.scalars(_OP_OUTPUTS, "i"),
        options=_options(table.table(_OP_OPTIONS), table.scalar(_OP_OPTIONS_TYPE, "B"), False),
        custom_options=_string(table, present, _OP_CUSTOM_OPTIONS),
        custom_options_format=table.scalar(_OP_CUSTOM_FORMAT, "b"),
        mutating_variable_inputs=_vector(table, present, _OP_MUTATING, "?"),
        intermediates=_vector(table, present, _OP_INTERMEDIATES, "i"),
        options2=_options(table.table(_OP_OPTIONS2), table.scalar(_OP_OPTIONS2_TYPE, "B"), True),
        debug_metadata_index=table.scalar(_OP_DEBUG, "i") if _OP_DEBUG in present else None)


def _tensor_maps(tables: list[Table]) -> list[tuple]:
    maps = []
    for table in tables:
        present = _check_fields(table, "tensor map")
        maps.append((_string(table, present, _MAP_NAME), table.scalar(_MAP_TENSOR, "I")))
    return maps


def read(data: bytes) -> Model:
    """The model in `data` as records; raises ValueError for a file that sets
    a field or holds a construct the writer does not reproduce."""
    if not is_tflite(data):
        raise ValueError("not a TFLite FlatBuffer")
    root = Table.root(data)
    present = _check_fields(root, "model")
    codes = []
    for table in root.tables(_MODEL_OPCODES):
        code_fields = _check_fields(table, "operator code")
        codes.append(OperatorCode(
            builtin_code=table.scalar(_OC_CODE, "i"), version=table.scalar(_OC_VERSION, "i", 1),
            custom_code=_string(table, code_fields, _OC_CUSTOM),
            deprecated_builtin_code=table.scalar(_OC_DEPRECATED_CODE, "b")))
    subgraphs = []
    for table in root.tables(_MODEL_SUBGRAPHS):
        sg_fields = _check_fields(table, "subgraph")
        subgraphs.append(Subgraph(
            tensors=[_tensor(t) for t in table.tables(_SG_TENSORS)],
            inputs=table.scalars(_SG_INPUTS, "i"),
            outputs=table.scalars(_SG_OUTPUTS, "i"),
            operators=[_operator(t) for t in table.tables(_SG_OPERATORS)],
            name=_string(table, sg_fields, _SG_NAME),
            debug_metadata_index=table.scalar(_SG_DEBUG, "i") if _SG_DEBUG in sg_fields else None))
    buffers = []
    for table in root.tables(_MODEL_BUFFERS):
        buffer_fields = _check_fields(table, "buffer")
        if table.scalar(_B_OFFSET, "Q") > 1 or table.scalar(_B_SIZE, "Q"):
            raise ValueError("buffers stored outside the FlatBuffer are not supported")
        buffers.append(table.bytes(_B_DATA) if _B_DATA in buffer_fields else None)
    metadata = [] if _MODEL_METADATA in present else None
    for table in root.tables(_MODEL_METADATA):
        meta_fields = _check_fields(table, "metadata")
        metadata.append((_string(table, meta_fields, _META_NAME), table.scalar(_META_BUFFER, "I")))
    signatures = [] if _MODEL_SIGNATURES in present else None
    for table in root.tables(_MODEL_SIGNATURES):
        sig_fields = _check_fields(table, "signature")
        signatures.append(Signature(
            inputs=_tensor_maps(table.tables(_SIG_INPUTS)),
            outputs=_tensor_maps(table.tables(_SIG_OUTPUTS)),
            key=_string(table, sig_fields, _SIG_KEY), tag=_string(table, sig_fields, _SIG_TAG),
            subgraph_index=table.scalar(_SIG_SUBGRAPH, "I")))
    return Model(
        version=root.scalar(_MODEL_VERSION, "I"), operator_codes=codes, subgraphs=subgraphs,
        buffers=buffers, description=_string(root, present, _MODEL_DESCRIPTION),
        metadata_buffer=_vector(root, present, _MODEL_METADATA_BUFFER, "i"),
        metadata=metadata, signatures=signatures)


class _Writer:
    def __init__(self, size: int):
        self.b = flatbuffers.Builder(size)

    def string(self, value: bytes | None) -> int | None:
        return None if value is None else self.b.CreateString(value)

    def numbers(self, values, fmt: str) -> int | None:
        if values is None:
            return None
        return self.b.CreateNumpyVector(np.asarray(values, _NUMPY_FORMATS[fmt]).reshape(-1))

    def tables(self, offsets: list[int]) -> int:
        self.b.StartVector(4, len(offsets), 4)
        for offset in reversed(offsets):
            self.b.PrependUOffsetTRelative(offset)
        return self.b.EndVector()

    def data(self, value: bytes | None) -> int | None:
        if value is None:
            return None
        # Buffer contents start on a 16-byte boundary, as the schema asks.
        self.b.Prep(16, len(value))
        return self.b.CreateByteVector(bytes(value))

    def table(self, fields: list) -> int:
        """A table from (slot, format, value) fields: a format of "o" is an
        offset written before, "raw" bytes copied as one scalar, any other a
        struct format; None values are left out."""
        fields = [f for f in fields if f[2] is not None]
        b = self.b
        prepend = {1: b.PrependUint8, 2: b.PrependUint16, 4: b.PrependUint32, 8: b.PrependUint64}
        b.StartObject(max((f[0] for f in fields), default=-1) + 1)
        # Widest first keeps the table free of padding.
        for slot, fmt, value in sorted(fields, key=lambda f: -_width(f[1], f[2])):
            if fmt == "o":
                b.PrependUOffsetTRelative(value)
            else:
                raw = value if fmt == "raw" else struct.pack("<" + fmt, value)
                prepend[len(raw)](int.from_bytes(raw, "little"))
            b.Slot(slot)
        return b.EndObject()

    def options(self, options: Options | None) -> int | None:
        if options is None:
            return None
        fields = []
        if options.source is not None:
            fields = [(slot, "raw", raw) for slot, raw in _scalar_fields(options.source).items()
                      if slot not in options.fields]
        for slot, (fmt, value) in options.fields.items():
            if fmt == "s":
                fields.append((slot, "o", self.string(value)))
            elif fmt.startswith("["):
                fields.append((slot, "o", self.numbers(value, fmt[1:])))
            else:
                fields.append((slot, fmt, value))
        return self.table(fields)


def _width(fmt: str, value) -> int:
    if fmt == "o":
        return 4
    return len(value) if fmt == "raw" else struct.calcsize("<" + fmt)


def _scalar_fields(table: Table) -> dict[int, bytes]:
    """The bytes of each scalar field of a table that holds only scalars. A
    field's width is the widest of 8, 4, 2 or 1 bytes that fits before the
    next field and divides its offset; bytes past the true width are padding."""
    present = sorted(table.present().items(), key=lambda kv: kv[1])
    end = table.position + table.size
    fields = {}
    for k, (slot, start) in enumerate(present):
        stop = present[k + 1][1] if k + 1 < len(present) else end
        width = 8
        while width > 1 and (width > stop - start or start % width):
            width //= 2
        if stop <= start or start + width > len(table.data):
            raise ValueError("malformed options table")
        fields[slot] = bytes(table.data[start:start + width])
    return fields


def _quantization_table(w: _Writer, q: Quantization | None) -> int | None:
    if q is None:
        return None
    details = None
    if q.custom is not None:
        custom = w.numbers(np.frombuffer(q.custom, np.uint8), "B")
        details = w.table([(0, "o", custom)])
    return w.table([
        (_Q_MIN, "o", w.numbers(q.min, "f")), (_Q_MAX, "o", w.numbers(q.max, "f")),
        (_Q_SCALE, "o", w.numbers(q.scale, "f")), (_Q_ZERO_POINT, "o", w.numbers(q.zero_point, "q")),
        (_Q_DETAILS_TYPE, "B", _CUSTOM_QUANTIZATION if details is not None else None),
        (_Q_DETAILS, "o", details),
        (_Q_DIMENSION, "i", q.quantized_dimension or None)])


def _tensor_table(w: _Writer, t: Tensor) -> int:
    return w.table([
        (_T_SHAPE, "o", w.numbers(t.shape, "i")), (_T_TYPE, "b", t.type or None),
        (_T_BUFFER, "I", t.buffer or None), (_T_NAME, "o", w.string(t.name)),
        (_T_QUANT, "o", _quantization_table(w, t.quantization)),
        (_T_VARIABLE, "?", True if t.is_variable else None),
        (_T_SHAPE_SIGNATURE, "o", w.numbers(t.shape_signature, "i")),
        (_T_HAS_RANK, "?", True if t.has_rank else None)])


def _operator_table(w: _Writer, op: Operator) -> int:
    custom = (None if op.custom_options is None
              else w.numbers(np.frombuffer(op.custom_options, np.uint8), "B"))
    return w.table([
        (_OP_OPCODE, "I", op.opcode or None),
        (_OP_INPUTS, "o", w.numbers(op.inputs, "i")), (_OP_OUTPUTS, "o", w.numbers(op.outputs, "i")),
        (_OP_OPTIONS_TYPE, "B", op.options.kind if op.options else None),
        (_OP_OPTIONS, "o", w.options(op.options)), (_OP_CUSTOM_OPTIONS, "o", custom),
        (_OP_CUSTOM_FORMAT, "b", op.custom_options_format or None),
        (_OP_MUTATING, "o", w.numbers(op.mutating_variable_inputs, "?")),
        (_OP_INTERMEDIATES, "o", w.numbers(op.intermediates, "i")),
        (_OP_OPTIONS2_TYPE, "B", op.options2.kind if op.options2 else None),
        (_OP_OPTIONS2, "o", w.options(op.options2)),
        (_OP_DEBUG, "i", op.debug_metadata_index)])


def _tensor_map_tables(w: _Writer, maps: list[tuple]) -> int:
    return w.tables([w.table([(_MAP_NAME, "o", w.string(name)), (_MAP_TENSOR, "I", index or None)])
                     for name, index in maps])


def write(model: Model) -> bytes:
    """The model as a TFLite FlatBuffer."""
    size = sum(len(b) for b in model.buffers if b) + 1024
    w = _Writer(size)
    buffers = [w.table([(_B_DATA, "o", w.data(data))]) for data in model.buffers]
    subgraphs = []
    for sg in model.subgraphs:
        tensors = [_tensor_table(w, t) for t in sg.tensors]
        operators = [_operator_table(w, op) for op in sg.operators]
        subgraphs.append(w.table([
            (_SG_TENSORS, "o", w.tables(tensors)), (_SG_INPUTS, "o", w.numbers(sg.inputs, "i")),
            (_SG_OUTPUTS, "o", w.numbers(sg.outputs, "i")), (_SG_OPERATORS, "o", w.tables(operators)),
            (_SG_NAME, "o", w.string(sg.name)), (_SG_DEBUG, "i", sg.debug_metadata_index)]))
    codes = [w.table([
        (_OC_DEPRECATED_CODE, "b", c.deprecated_builtin_code or None),
        (_OC_CUSTOM, "o", w.string(c.custom_code)),
        (_OC_VERSION, "i", c.version if c.version != 1 else None),
        (_OC_CODE, "i", c.builtin_code or None)]) for c in model.operator_codes]
    metadata = [w.table([(_META_NAME, "o", w.string(name)), (_META_BUFFER, "I", buffer or None)])
                for name, buffer in model.metadata or ()]
    signatures = [w.table([
        (_SIG_INPUTS, "o", _tensor_map_tables(w, s.inputs)),
        (_SIG_OUTPUTS, "o", _tensor_map_tables(w, s.outputs)),
        (_SIG_KEY, "o", w.string(s.key)), (_SIG_TAG, "o", w.string(s.tag)),
        (_SIG_SUBGRAPH, "I", s.subgraph_index or None)]) for s in model.signatures or ()]
    root = w.table([
        (_MODEL_VERSION, "I", model.version or None), (_MODEL_OPCODES, "o", w.tables(codes)),
        (_MODEL_SUBGRAPHS, "o", w.tables(subgraphs)),
        (_MODEL_DESCRIPTION, "o", w.string(model.description)),
        (_MODEL_BUFFERS, "o", w.tables(buffers)),
        (_MODEL_METADATA_BUFFER, "o", w.numbers(model.metadata_buffer, "i")),
        (_MODEL_METADATA, "o", None if model.metadata is None else w.tables(metadata)),
        (_MODEL_SIGNATURES, "o", None if model.signatures is None else w.tables(signatures))])
    w.b.Finish(root, file_identifier=MAGIC)
    return bytes(w.b.Output())
