"""TFLite FlatBuffer models: inspection and conversion to QDQ ONNX.

Inspection reads any TFLite file. Conversion covers the int8 operators whose
TFLite semantics a QDQ ONNX graph states exactly and refuses everything else
by name, so a model is never compiled with a reinterpreted operator.
"""

import json
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import onnx
from onnx import TensorProto, helper

from tigris.frontends.flatbuffer import Table, flexbuffer_map
from tigris.frontends.qdq import GraphBuilder, same_padding
from tigris.frontends.tflite_schema import (
    BUILTIN_OPERATORS, B_DATA, B_OFFSET, B_SIZE, MODEL_BUFFERS, MODEL_DESCRIPTION, MODEL_OPCODES,
    MODEL_SUBGRAPHS, MODEL_VERSION, OC_CODE, OC_CUSTOM, OC_DEPRECATED_CODE, OC_VERSION,
    OP_CUSTOM_OPTIONS, OP_INPUTS, OP_OPCODE, OP_OPTIONS, OP_OUTPUTS, Q_DIMENSION, Q_MAX, Q_MIN,
    Q_SCALE, Q_ZERO_POINT, SAME, SG_INPUTS, SG_NAME, SG_OPERATORS, SG_OUTPUTS, SG_TENSORS,
    TENSOR_TYPES, T_BUFFER, T_NAME, T_QUANT, T_SHAPE, T_TYPE, T_VARIABLE, VALID, WINDOWED,
    is_tflite,
)

# Metadata stating that every model input and output of rank 3 or more is held
# channels-last, as the TFLite tensor is laid out.
BOUNDARY_LAYOUT_KEY = "tigris.boundary_layout"
# Variables a model keeps across invocations: a JSON list of
# {"input", "output", "initial"} value names, one entry per variable.
STATE_KEY = "tigris.state"

_ACTIVATIONS = {0: "none", 1: "relu", 2: "relu_n1_to_1", 3: "relu6", 4: "tanh", 5: "sign_bit"}
_NUMPY = {"FLOAT32": np.float32, "INT32": np.int32, "INT8": np.int8, "UINT8": np.uint8,
          "INT16": np.int16, "INT64": np.int64, "BOOL": np.bool_}
# ONNX element types of model boundaries and operator state; the checks admit
# no other type there.
_ONNX_TYPES = {"FLOAT32": TensorProto.FLOAT, "INT32": TensorProto.INT32, "INT64": TensorProto.INT64,
               "INT16": TensorProto.INT16, "INT8": TensorProto.INT8, "UINT8": TensorProto.UINT8,
               "BOOL": TensorProto.BOOL}


class _Model:
    """The parts of a TFLite model the frontend reads, resolved once."""

    def __init__(self, data: bytes):
        if not is_tflite(data):
            raise ValueError("not a TFLite FlatBuffer")
        self.data = data
        root = Table.root(data)
        self.version = root.scalar(MODEL_VERSION, "I")
        self.description = root.string(MODEL_DESCRIPTION)
        self.buffers = root.tables(MODEL_BUFFERS)
        self.subgraphs = root.tables(MODEL_SUBGRAPHS)
        self.opcodes = []
        for code in root.tables(MODEL_OPCODES):
            builtin = max(code.scalar(OC_CODE, "i"), code.scalar(OC_DEPRECATED_CODE, "b"))
            custom = code.string(OC_CUSTOM)
            name = (f"CUSTOM:{custom}" if custom else
                    BUILTIN_OPERATORS[builtin] if 0 <= builtin < len(BUILTIN_OPERATORS)
                    else f"BUILTIN_{builtin}")
            self.opcodes.append((name, code.scalar(OC_VERSION, "i", 1)))

    def buffer(self, index: int) -> bytes:
        if not 0 <= index < len(self.buffers):
            return b""
        entry = self.buffers[index]
        offset = entry.scalar(B_OFFSET, "Q")
        if offset > 1:
            size = entry.scalar(B_SIZE, "Q")
            if offset + size > len(self.data):
                raise ValueError(f"buffer {index} runs past the file")
            return self.data[offset:offset + size]
        return entry.bytes(B_DATA)


class _Tensor:
    def __init__(self, model: _Model, table: Table):
        self.name = table.string(T_NAME)
        self.shape = table.scalars(T_SHAPE, "i")
        code = table.scalar(T_TYPE, "b")
        self.type = TENSOR_TYPES[code] if 0 <= code < len(TENSOR_TYPES) else f"TYPE_{code}"
        self.buffer = table.scalar(T_BUFFER, "I")
        quant = table.table(T_QUANT)
        self.scale = np.asarray(quant.scalars(Q_SCALE, "f") if quant else [], np.float32)
        self.zero_point = np.asarray(quant.scalars(Q_ZERO_POINT, "q") if quant else [], np.int64)
        self.quantized_dimension = quant.scalar(Q_DIMENSION, "i") if quant else 0
        self.minimum = quant.scalars(Q_MIN, "f") if quant else []
        self.maximum = quant.scalars(Q_MAX, "f") if quant else []
        # A variable tensor keeps what an operator wrote into it for the next invocation.
        self.variable = bool(table.scalar(T_VARIABLE, "B"))
        self._model = model
        # The value of a constant the frontend computed itself.
        self.folded: bytes | None = None

    @property
    def data(self) -> bytes:
        return self.folded if self.folded is not None else self._model.buffer(self.buffer)

    def array(self) -> np.ndarray:
        dtype = _NUMPY.get(self.type)
        if dtype is None:
            raise ValueError(f"tensor {self.name!r} has unsupported type {self.type}")
        return np.frombuffer(self.data, dtype=dtype).reshape(self.shape)


class _Operator:
    def __init__(self, model: _Model, table: Table, index: int):
        opcode = table.scalar(OP_OPCODE, "I")
        self.kind, self.version = (model.opcodes[opcode] if opcode < len(model.opcodes)
                                   else (f"OPCODE_{opcode}", 1))
        self.inputs = table.scalars(OP_INPUTS, "i")
        self.outputs = table.scalars(OP_OUTPUTS, "i")
        self.options = table.table(OP_OPTIONS)
        self.custom_options = table.bytes(OP_CUSTOM_OPTIONS)
        self.index = index

    def option(self, slot: int, fmt: str, default=0):
        """Field `slot` of the options table: a scalar of struct format `fmt`,
        a string for "s", a byte vector for "[B"; `default` without a table."""
        if not self.options:
            return default
        if fmt == "s":
            return self.options.string(slot)
        return self.options.bytes(slot) if fmt == "[B" else self.options.scalar(slot, fmt, default)


def _check_indices(owner: str, item: str, indices, count: int, lowest: int = 0) -> None:
    """Raise ValueError when one of `indices` lies outside [lowest, count)."""
    for index in indices:
        if not lowest <= index < count:
            raise ValueError(f"{owner}: {item} index {index} is out of range for {count} "
                             f"{item}{'s' if count != 1 else ''}")


def _missing_operand(op: _Operator) -> str:
    """The first operand `op` reads that the file leaves out, as -1 or past the
    end of the list; "" when there is none or the kind is not in _OPERANDS."""
    if op.kind not in _OPERANDS:
        return ""
    inputs, outputs = _OPERANDS[op.kind]
    optional = _OPTIONAL_OPERANDS.get(op.kind, ())
    for position in range(max(inputs, len(op.inputs))):
        if (position >= len(op.inputs) or op.inputs[position] < 0) and position not in optional:
            return f"input {position} is missing"
    return f"output {len(op.outputs)} is missing" if len(op.outputs) < outputs else ""


def _read(data: bytes):
    model = _Model(data)
    graphs = []
    for subgraph in model.subgraphs:
        tensors = [_Tensor(model, t) for t in subgraph.tables(SG_TENSORS)]
        operators = [_Operator(model, op, i) for i, op in enumerate(subgraph.tables(SG_OPERATORS))]
        graphs.append((subgraph.string(SG_NAME), tensors, subgraph.scalars(SG_INPUTS, "i"),
                       subgraph.scalars(SG_OUTPUTS, "i"), operators))
    for position, (_, tensors, inputs, outputs, operators) in enumerate(graphs):
        prefix = f"subgraph {position} " if position else ""
        _check_indices(f"{prefix or 'model '}inputs", "tensor", inputs, len(tensors))
        _check_indices(f"{prefix or 'model '}outputs", "tensor", outputs, len(tensors))
        for op in operators:
            owner = f"{prefix}operator {op.index} {op.kind}"
            # -1 is an operand left out.
            _check_indices(owner, "tensor", op.inputs, len(tensors), -1)
            _check_indices(owner, "tensor", op.outputs, len(tensors))
            _check_indices(owner, "subgraph", _subgraphs(op), len(graphs))
            reason = _missing_operand(op)
            if reason:
                raise ValueError(f"{owner}: {reason}")
    # The converter passes a constant an IF branch reads as one of its
    # operands; inside the branch it is that constant, from the same buffer.
    # Each graph is folded before the constants it passes on are read.
    for index in range(len(graphs)):
        graphs[index] = _fold_constants(graphs[index])
        _, outer, _, _, operators = graphs[index]
        for op in operators:
            if op.kind != "IF":
                continue
            for position in _subgraphs(op):
                if not 0 < position < len(graphs):
                    continue
                _, branch_tensors, branch_inputs, _, _ = graphs[position]
                for operand, entering in zip(op.inputs[1:], branch_inputs):
                    if operand >= 0 and _is_constant(outer[operand]):
                        branch_tensors[entering].buffer = outer[operand].buffer
                        branch_tensors[entering].folded = outer[operand].folded
    return model, graphs


def _constant_result(op: "_Operator", tensors) -> "np.ndarray | None":
    """What an operator computes from static shapes and constants alone, or
    None when it reads an activation's values."""
    ins = [tensors[i] for i in op.inputs if i >= 0]
    out = tensors[op.outputs[0]]
    if op.kind == "SHAPE":
        return np.asarray(ins[0].shape, _NUMPY[out.type])
    if op.kind == "ZEROS_LIKE":
        # TFLite Micro writes zero bytes, whatever the quantization.
        return np.zeros(out.shape, _NUMPY[out.type])
    if not ins or not all(_is_constant(t) for t in ins):
        return None
    if op.kind == "TRANSPOSE":
        return ins[0].array().transpose(ins[1].array().reshape(-1))
    if op.kind == "QUANTIZE" and ins[0].type == "FLOAT32" and out.type == "INT8" and len(out.scale) == 1:
        # TFLite Micro's QUANTIZE: a float32 division, rounded half away from zero.
        ratio = (ins[0].array() / np.float32(out.scale[0])).astype(np.float32).astype(np.float64)
        rounded = np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)
        return np.clip(rounded + int(out.zero_point[0]), -128, 127).astype(np.int8)
    if op.kind == "FILL":
        return np.full(ins[0].array().reshape(-1), ins[1].array().reshape(-1)[0], _NUMPY[out.type])
    if op.kind == "BROADCAST_ARGS":
        return np.asarray(np.broadcast_shapes(*(tuple(t.array().reshape(-1)) for t in ins)),
                          _NUMPY[out.type])
    if op.kind in _CONSTANT_MOVES and _same_quantization([ins[0], out]):
        if op.kind in _OPTION_SHAPED and _option_shape_reason(op, ins, [out]):
            return None  # kept, so that the operator check refuses the shape
        return _constant_move(op, ins)
    if (op.kind in _CONSTANT_MATH and all(t.type in ("INT32", "INT64") and not len(t.scale)
                                          for t in (*ins, out))
            and not _option(op, "fused_activation_function")):
        a, b = (t.array().astype(np.int64) for t in ins)
        value = _CONSTANT_MATH[op.kind](a, b)
        info = np.iinfo(_NUMPY[out.type])
        return value if np.all((value >= info.min) & (value <= info.max)) else None
    return None


# Data movement the frontend computes on constants of any dtype, and the integer
# arithmetic of shape computations (a Keras Reshape that keeps the batch axis
# exports SHAPE, STRIDED_SLICE and PACK ahead of the RESHAPE).
_CONSTANT_MOVES = ("STRIDED_SLICE", "SLICE", "PACK", "CONCATENATION", "RESHAPE", "SQUEEZE",
                   "EXPAND_DIMS")
_CONSTANT_MATH = {"ADD": np.add, "SUB": np.subtract, "MUL": np.multiply}


def _dropped_index(begin: int, extent: int, stride: int, masked: bool) -> "int | None":
    """The element a STRIDED_SLICE takes from an axis it drops, as TFLite Micro
    picks it (its begin_mask bit selects the first or last element), or None
    when that index is outside the axis."""
    index = (0 if stride > 0 else extent - 1) if masked else begin + extent if begin < 0 else begin
    return index if 0 <= index < extent else None


class _StridedSliceOptions(NamedTuple):
    """The StridedSliceOptions table of a STRIDED_SLICE."""
    begin_mask: int
    end_mask: int
    ellipsis_mask: int
    new_axis_mask: int
    shrink_axis_mask: int
    offset: bool

    @classmethod
    def of(cls, op: "_Operator") -> "_StridedSliceOptions":
        return cls(*(_option(op, field) for field in cls._fields))


def _strided_slice_axes(options: _StridedSliceOptions, shape, begins, ends, strides):
    """Per axis of `shape`, (start, stop, step, dropped) of a STRIDED_SLICE, with
    TensorFlow's clamping as slice().indices() gives it; a dropped axis is
    (index, index + 1, 1, True). None where TFLite Micro gives no result: begin,
    end or strides with fewer entries than `shape` has axes (it reads one per
    axis), a zero stride, a dropped axis with a negative stride (it writes
    nothing) or a dropped index outside its axis."""
    if min(len(begins), len(ends), len(strides)) < len(shape):
        return None
    axes = []
    for axis, extent in enumerate(shape):
        step, begin_masked = strides[axis], bool(options.begin_mask >> axis & 1)
        if step == 0:
            return None
        if options.shrink_axis_mask >> axis & 1:
            if step < 0:
                return None
            index = _dropped_index(begins[axis], extent, step, begin_masked)
            if index is None:
                return None
            axes.append((index, index + 1, 1, True))
            continue
        begin = None if begin_masked else begins[axis]
        end = (None if options.end_mask >> axis & 1
               else ends[axis] + (begins[axis] if options.offset else 0))
        axes.append((*slice(begin, end, step).indices(extent), False))
    return axes


def _constant_move(op: "_Operator", ins) -> "np.ndarray | None":
    x = ins[0].array()
    if op.kind in ("RESHAPE", "SQUEEZE", "EXPAND_DIMS"):
        return x  # _fold_constants gives it the output's static shape
    if op.kind == "PACK":
        if _option(op, "values_count") != len(ins):
            return None  # kept, so that the operator check refuses the count
        return np.stack([t.array() for t in ins], axis=_option(op, "axis"))
    if op.kind == "CONCATENATION":
        if _option(op, "fused_activation_function") or not _same_quantization(ins):
            return None
        return np.concatenate([t.array() for t in ins], axis=_option(op, "axis"))
    if op.kind == "SLICE":
        begin = [int(v) for v in ins[1].array().reshape(-1)]
        size = [int(v) for v in ins[2].array().reshape(-1)]
        if min(len(begin), len(size)) < x.ndim:
            return None  # TFLite Micro reads one entry per input axis from each
        ends = [extent if s == -1 else b + s for b, s, extent in zip(begin, size, x.shape)]
        if any(not 0 <= b <= e <= extent for b, e, extent in zip(begin, ends, x.shape)):
            return None
        return x[tuple(slice(b, e) for b, e in zip(begin, ends))]
    # STRIDED_SLICE; ellipsis and new axes are not folded.
    options = _StridedSliceOptions.of(op)
    if options.ellipsis_mask or options.new_axis_mask:
        return None
    begins, ends, strides = ([int(v) for v in ins[i].array().reshape(-1)] for i in (1, 2, 3))
    axes = _strided_slice_axes(options, x.shape, begins, ends, strides)
    if axes is None:
        return None
    for axis, (start, stop, step, _) in enumerate(axes):
        x = np.take(x, list(range(start, stop, step)), axis=axis)
    return x.reshape([n for n, (_, _, _, dropped) in zip(x.shape, axes) if not dropped])


def _fold_constants(graph):
    """Operators computed from static shapes and constants become constants:
    the shape and fill operators the converter folds in a static model, the
    shape arithmetic it leaves in, and a transpose of a weight it passed into
    a branch."""
    name, tensors, inputs, outputs, operators = graph
    kept = []
    for op in operators:
        result = tensors[op.outputs[0]] if len(op.outputs) == 1 else None
        value = (_constant_result(op, tensors)
                 if result is not None and op.outputs[0] not in outputs and result.type in _NUMPY
                 else None)
        if value is None or value.size != int(np.prod(result.shape)):
            kept.append(op)
            continue
        value = value.reshape(result.shape)
        result.folded = np.ascontiguousarray(value.astype(_NUMPY[result.type])).tobytes()
        result.buffer = 1 << 30
    return name, tensors, inputs, outputs, kept


def _dtype_name(tensor: _Tensor) -> str:
    return tensor.type.lower()


def _narrow_index_inputs(graph):
    """Keep an integer input and its index-only movement chain in int32."""
    _, tensors, inputs, outputs, operators = graph
    for entering in inputs:
        if tensors[entering].type != "INT64":
            continue
        reached, pending, valid = set(), [entering], True
        while pending and valid:
            index = pending.pop()
            if index in reached:
                continue
            reached.add(index)
            t = tensors[index]
            if (index in outputs or any(v < -2147483648 for v in t.minimum)
                    or any(v > 2147483647 for v in t.maximum)):
                valid = False
                break
            uses = [op for op in operators if index in op.inputs]
            if not uses:
                valid = False
                break
            for op in uses:
                slot = _INDEX_SLOTS.get(op.kind)
                if slot is not None and op.inputs[slot] == index:
                    continue
                if (op.kind not in {"CAST", "RESHAPE", "TRANSPOSE"} or op.inputs[0] != index
                        or tensors[op.outputs[0]].type not in {"INT32", "INT64"}):
                    valid = False
                    break
                pending.extend(op.outputs)
        if valid:
            for index in reached:
                tensors[index].type = "INT32"
    return graph


def unsupported(data: bytes) -> list[str]:
    """Why the model cannot be converted, one reason per line; empty when it can."""
    _, graphs = _read(data)
    return _reasons([_narrow_index_inputs(g) for g in graphs])


def _reasons(graphs) -> list[str]:
    """The reasons of unsupported() for the subgraphs `_read` gave, after
    `_narrow_index_inputs`."""
    reasons = []
    init = _initializer_subgraphs(graphs)
    branches = _branch_subgraphs(graphs)
    for position in range(1, len(graphs)):
        if position in branches:
            reasons += [f"subgraph {position}: {r}"
                        for r in _branch_reasons(graphs[position], branches[position])]
        elif position not in init:
            reasons.append(f"subgraph {position}: control flow other than IF and WHILE does not convert yet")
        elif _variable_initials(graphs, {position}) is None:
            reasons.append(f"subgraph {position}: only constant variable initial values convert")
    for _, tensors, inputs, outputs, operators in graphs[:1]:
        indices = {i for op in operators if op.kind in _INDEX for i in op.outputs}
        slots = {op.inputs[_INDEX_SLOTS[op.kind]] for op in operators if op.kind in _INDEX_SLOTS}
        index_inputs = {i for i in inputs if tensors[i].type == "INT32"}
        # int32 arithmetic results are values a run computes, usable as indices.
        computed = {op.outputs[0] for op in operators
                    if op.kind in ("ADD", "SUB", "MUL", "CAST", "RESHAPE", "TRANSPOSE") and tensors[op.outputs[0]].type == "INT32"}
        for index in list(inputs) + list(outputs):
            if (tensors[index].type not in ("INT8", "UINT8", "FLOAT32", "BOOL") and index not in indices
                    and index not in index_inputs):
                reasons.append(f"model boundary {tensors[index].name!r} is {tensors[index].type}")
        for index in sorted(indices | index_inputs):
            uses = [op for op in operators if index in op.inputs]
            if any((op.kind not in _INDEX_SLOTS or op.inputs[_INDEX_SLOTS[op.kind]] != index)
                   and not _integer_use(op)
                   for op in uses):
                reasons.append(f"index {tensors[index].name!r} feeds an operand other than indices")
        held = {op.inputs[slot] for op in operators if op.kind in _STATEFUL
                for slot in _STATEFUL[op.kind]}
        for index, tensor in enumerate(tensors):
            if not tensor.variable:
                continue
            uses = [op for op in operators if index in op.inputs]
            if (index in inputs or index in outputs or len(uses) != 1 or index not in held
                    or any(index in op.outputs for op in operators)):
                reasons.append(f"variable tensor {tensor.name!r} is not the state of one "
                               "SVDF or LSTM")
        for index in sorted(held):
            if not tensors[index].variable:
                reasons.append(f"state {tensors[index].name!r} is not a variable tensor")
        for index in sorted(slots - indices - index_inputs - computed):
            if not _is_constant(tensors[index]):
                reasons.append(f"indices {tensors[index].name!r} are computed by an operator "
                               "other than ARG_MAX, ARG_MIN or int32 arithmetic")
        grouped: dict[str, list[_Operator]] = {}
        for op in operators:
            reason = _operator_reason(op, tensors)
            if reason:
                grouped.setdefault(reason, []).append(op)
        for reason, ops in grouped.items():
            if len(ops) == 1:
                reasons.append(f"operator {ops[0].index} {ops[0].kind}: {reason}")
            else:
                kinds = ", ".join(sorted({op.kind for op in ops}))
                reasons.append(f"{reason}: {len(ops)} operators ({kinds})")
    return reasons


def _branch_subgraphs(graphs) -> dict[int, str]:
    """Subgraphs control flow runs, from the main graph down, by role: IF
    branches, WHILE conditions and bodies."""
    roles = {}
    pending = [0] if graphs else []
    while pending:
        for op in graphs[pending.pop()][4]:
            if op.kind not in ("IF", "WHILE"):
                continue
            found = dict(zip(_subgraphs(op), ("branch", "branch") if op.kind == "IF"
                             else ("condition", "body")))
            for position, role in found.items():
                if 0 < position < len(graphs) and position not in roles:
                    roles[position] = role
                    pending.append(position)
    return roles


def _branch_reasons(graph, role: str = "branch") -> list[str]:
    """Why a subgraph cannot run: its operators, and anything beyond a plain
    float32 computation of its inputs; a condition gives one bool."""
    _, tensors, inputs, outputs, operators = graph
    reasons = []
    for op in operators:
        if op.kind in (*_VARIABLES, *_STATEFUL):
            reasons.append(f"operator {op.index} {op.kind}: state inside a subgraph")
            continue
        reason = _operator_reason(op, tensors)
        if reason:
            reasons.append(f"operator {op.index} {op.kind}: {reason}")
    if role == "condition":
        if len(outputs) != 1 or tensors[outputs[0]].type != "BOOL" or int(np.prod(tensors[outputs[0]].shape)) != 1:
            reasons.append("a condition other than one bool")
        outputs = []
    for index in [*inputs, *outputs]:
        if tensors[index].type not in ("FLOAT32", "INT32") and not _is_constant(tensors[index]):
            reasons.append(f"subgraph boundary {tensors[index].name!r} other than float32 or int32")
    return reasons


_BOUNDARY_TYPES = {"FLOAT32": TensorProto.FLOAT, "BOOL": TensorProto.BOOL, "INT32": TensorProto.INT32}


def _control_flow_tensors(operators) -> set[int]:
    """The operands and results of IF and WHILE operators."""
    return {i for op in operators if op.kind in ("IF", "WHILE") for i in (*op.inputs, *op.outputs)
            if i >= 0}


def _branch_graph(graphs, position: int, name: str) -> onnx.GraphProto:
    """One subgraph as an ONNX graph whose inputs are the operands it receives
    and whose outputs are its results, each held by an operator of it."""
    _, tensors, inputs, outputs, operators = graphs[position]
    consumed = {i for op in operators for i in op.inputs}
    converter = _Converter(tensors, (), consumed)
    converter.graphs = graphs
    converter.float_boundary = {*inputs, *outputs, *_control_flow_tensors(operators)}
    b = converter.b
    graph_inputs = []
    for position, index in enumerate(inputs):
        if _is_constant(tensors[index]):
            continue
        input_name = b.unique(f"{name}_in{position}")
        graph_inputs.append(helper.make_tensor_value_info(
            input_name, _BOUNDARY_TYPES[tensors[index].type], _onnx_shape(tensors[index].shape)))
        converter.values[index] = input_name
    for op in operators:
        converter.convert(op)
    graph_outputs = []
    for position, index in enumerate(outputs):
        result = b.node("Identity", [converter.value(index)], f"{name}_out{position}")
        graph_outputs.append(helper.make_tensor_value_info(
            result, _BOUNDARY_TYPES[tensors[index].type], _onnx_shape(tensors[index].shape)))
    return helper.make_graph(b.nodes, name, graph_inputs, graph_outputs, b.initializers,
                             value_info=converter.value_info)


def _var_name(op: "_Operator") -> str:
    """The variable a VAR_HANDLE names: its container and shared name."""
    if op.options is None:
        return ""
    return f"{_option(op, 'container')}/{_option(op, 'shared_name')}"


def _subgraphs(op: "_Operator") -> list[int]:
    """The subgraph indices an IF, WHILE or CALL_ONCE names, in field order;
    none for other kinds."""
    if op.kind not in ("IF", "WHILE", "CALL_ONCE"):
        return []
    return [_option(op, field) for field in _KINDS[op.kind].options]


def _initializer_subgraphs(graphs) -> set[int]:
    """Subgraphs the main graph runs once through CALL_ONCE."""
    return ({position for op in graphs[0][4] if op.kind == "CALL_ONCE" for position in _subgraphs(op)}
            if graphs else set())


def _variable_initials(graphs, positions):
    """{variable: array} assigned by the CALL_ONCE subgraphs, or None when one
    does anything but assign constants to variables."""
    initials = {}
    for position in positions:
        if not 0 < position < len(graphs):
            return None
        _, tensors, _, _, operators = graphs[position]
        handles = {}
        for op in operators:
            if op.kind == "VAR_HANDLE":
                handles[op.outputs[0]] = _var_name(op)
            elif (op.kind == "ASSIGN_VARIABLE" and op.inputs[0] in handles
                  and _is_constant(tensors[op.inputs[1]])
                  and tensors[op.inputs[1]].type == "FLOAT32"):
                initials[handles[op.inputs[0]]] = tensors[op.inputs[1]].array()
            else:
                return None
    return initials


def _activation(code: int) -> str:
    return _ACTIVATIONS.get(code, f"activation {code}")


_UNARY = {"LOGISTIC": "Sigmoid", "TANH": "Tanh", "HARD_SWISH": "HardSwish", "RELU": "Relu",
          "ELU": "Elu"}
# Elementwise math: the ONNX operator, or a composition the compiler folds back.
_ELEMENTWISE_UNARY = {"ABS": "Abs", "NEG": "Neg", "EXP": "Exp", "LOG": "Log", "SQRT": "Sqrt",
                      "FLOOR": "Floor", "CEIL": "Ceil", "ROUND": "Round", "SIN": "Sin",
                      "COS": "Cos"}
_ELEMENTWISE_BINARY = ("DIV", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE", "FLOOR_DIV",
                       "FLOOR_MOD")
# Data movement built from Split, Concat, Reshape and Transpose.
_DATA_MOVEMENT = ("SLICE", "STRIDED_SLICE", "GATHER", "PACK", "UNPACK", "SPACE_TO_DEPTH",
                  "DEPTH_TO_SPACE", "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND", "BROADCAST_TO",
                  "GATHER_ND", "MIRROR_PAD", "REVERSE_V2", "EMBEDDING_LOOKUP",
                  "DYNAMIC_UPDATE_SLICE")
# Comparisons write bool; the logical operators read and write it.
_COMPARISONS = {"EQUAL": "Equal", "NOT_EQUAL": "Equal", "LESS": "Less", "LESS_EQUAL": "LessOrEqual",
                "GREATER": "Greater", "GREATER_EQUAL": "GreaterOrEqual"}
_LOGICAL = {"LOGICAL_AND": "And", "LOGICAL_OR": "Or", "LOGICAL_NOT": "Not"}
# Input positions that hold bool, and whether the output does.
_BOOL_SLOTS = {**{kind: ((), True) for kind in _COMPARISONS},
               "LOGICAL_AND": ((0, 1), True), "LOGICAL_OR": ((0, 1), True),
               "LOGICAL_NOT": ((0,), True), "SELECT_V2": ((0,), False), "CAST": ((0,), False),
               "REDUCE_ALL": ((0,), True)}
# Operators that keep state in a variable tensor, at this operand.
_STATEFUL = {"SVDF": (4,), "UNIDIRECTIONAL_SEQUENCE_LSTM": (18, 19)}
# Resource variables: a handle names a variable, which is read and assigned.
_VARIABLES = ("CALL_ONCE", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE")
# Index outputs: int32 positions, as model outputs or the indices of the operators below.
_INDEX = {"ARG_MAX": "ArgMax", "ARG_MIN": "ArgMin"}
# The operand holding indices or start positions, which may be computed at run time.
_INDEX_SLOTS = {"GATHER": 1, "GATHER_ND": 1, "EMBEDDING_LOOKUP": 0, "DYNAMIC_UPDATE_SLICE": 2}
# Reductions over one run of adjacent axes, and the prefix sum over one axis.
_REDUCTIONS = {"REDUCE_MAX": "ReduceMax", "REDUCE_MIN": "ReduceMin", "SUM": "ReduceSum",
               "REDUCE_ALL": "ReduceAll"}
# The SSD box decoding and non-max suppression TFLite Micro registers as a
# custom operator.
_DETECTION = "CUSTOM:TFLite_Detection_PostProcess"
# Operators whose options set their output shapes from the input shapes.
_OPTION_SHAPED = ("PACK", "UNPACK", "CONCATENATION", "GATHER", "BATCH_MATMUL", "STRIDED_SLICE")
# Positions of the data, weights and bias operands of the weighted operators.
_WEIGHTED = {"CONV_2D": (0, 1, 2), "DEPTHWISE_CONV_2D": (0, 1, 2), "FULLY_CONNECTED": (0, 1, 2),
             "TRANSPOSE_CONV": (2, 1, 3)}
_CONSTANT_OPERANDS = {"MEAN": (1,), "TRANSPOSE": (1,), "SPLIT": (0,), "SPLIT_V": (1, 2),
                      "PAD": (1,), "PADV2": (1, 2), "RESHAPE": (1,), "EXPAND_DIMS": (1,),
                      "RESIZE_NEAREST_NEIGHBOR": (1,), "RESIZE_BILINEAR": (1,),
                      "TRANSPOSE_CONV": (0,), "SLICE": (1, 2), "STRIDED_SLICE": (1, 2, 3),
                      "GATHER": (1,), "SPACE_TO_BATCH_ND": (1, 2), "BATCH_TO_SPACE_ND": (1, 2),
                      "BROADCAST_TO": (1,), "PRELU": (1,), "REDUCE_MAX": (1,),
                      "REDUCE_MIN": (1,), "SUM": (1,), "CUMSUM": (1,), "GATHER_ND": (1,),
                      "MIRROR_PAD": (1,), "REVERSE_V2": (1,), "EMBEDDING_LOOKUP": (0,),
                      "DYNAMIC_UPDATE_SLICE": (2,), "ARG_MAX": (1,), "ARG_MIN": (1,),
                      "REDUCE_ALL": (1,), "SVDF": (1, 2, 3)}
# Operands each kind the importer converts or folds reads, (inputs, outputs); for a
# variadic kind the fewest. The detection operator checks its own.
_OPERANDS = {
    **dict.fromkeys((*_UNARY, *_ELEMENTWISE_UNARY, "RSQRT", "SQUARE", "RELU6", "SOFTMAX",
                     "LOG_SOFTMAX", "LEAKY_RELU", "L2_NORMALIZATION", "GELU", "QUANTIZE",
                     "DEQUANTIZE", "CAST", "LOGICAL_NOT", "SQUEEZE", "SPACE_TO_DEPTH",
                     "DEPTH_TO_SPACE", "AVERAGE_POOL_2D", "MAX_POOL_2D", "L2_POOL_2D",
                     "READ_VARIABLE", "CONCATENATION", "PACK", "ADD_N", "UNPACK", "IF", "WHILE",
                     "SHAPE", "ZEROS_LIKE"), (1, 1)),
    **dict.fromkeys((*_CONSTANT_MATH, *_ELEMENTWISE_BINARY, *_COMPARISONS, "LOGICAL_AND",
                     "LOGICAL_OR", "PRELU", "BATCH_MATMUL", "MEAN", *_REDUCTIONS, *_INDEX,
                     "TRANSPOSE", "RESHAPE", "EXPAND_DIMS", "GATHER", "GATHER_ND",
                     "BROADCAST_TO", "MIRROR_PAD", "REVERSE_V2", "EMBEDDING_LOOKUP", "PAD",
                     "SPLIT", "RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR", "CUMSUM", "FILL",
                     "BROADCAST_ARGS"), (2, 1)),
    **dict.fromkeys(("SLICE", "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND", "DYNAMIC_UPDATE_SLICE",
                     "SELECT_V2", "PADV2", "SPLIT_V", "CONV_2D", "DEPTHWISE_CONV_2D",
                     "FULLY_CONNECTED"), (3, 1)),
    "STRIDED_SLICE": (4, 1), "TRANSPOSE_CONV": (4, 1), "SVDF": (5, 1),
    "UNIDIRECTIONAL_SEQUENCE_LSTM": (24, 1), "CALL_ONCE": (0, 0), "VAR_HANDLE": (0, 1), "ASSIGN_VARIABLE": (2, 0)}
# Operands that may be left out: biases, the constants the conversion takes from the
# stated shapes or pads with zero, and what _svdf_reason and _lstm_reason check.
_OPTIONAL_OPERANDS = {"CONV_2D": (2,), "DEPTHWISE_CONV_2D": (2,), "FULLY_CONNECTED": (2,),
                      "TRANSPOSE_CONV": (0, 3), "RESHAPE": (1,), "SPLIT_V": (1,),
                      "PADV2": (2,), "SVDF": (3,),
                      "UNIDIRECTIONAL_SEQUENCE_LSTM": (1, 5, 9, 10, 11, 12, 16, 17, 20, 21,
                                                       22, 23)}


class Window(NamedTuple):
    """The padding, filter, strides and dilations of a windowed operator, height first."""
    padding: int
    kernel: tuple
    strides: tuple
    dilations: tuple

    @property
    def spans(self) -> tuple:
        """The rows and columns the filter covers with its dilation."""
        return tuple((k - 1) * d + 1 for k, d in zip(self.kernel, self.dilations))

    def outputs(self, extents) -> list[int]:
        """The output height and width the padding gives an input of `extents`."""
        return [(e + s - 1) // s if self.padding == SAME else (e - span + s) // s
                for e, span, s in zip(extents, self.spans, self.strides)]

    def pads(self, extents, outputs) -> list[tuple[int, int]]:
        """Per axis (begin, end) padding that turns `extents` into `outputs`."""
        return [same_padding(e, o, span, s)
                for e, o, span, s in zip(extents, outputs, self.spans, self.strides)]


def window(kind: str, option, weights=None) -> Window:
    """The window of a `WINDOWED` operator. `option(slot, fmt, default)` reads
    its options table; a weighted operator's filter is `weights`' dimensions 1 and 2."""
    slot_w, slot_h, *dilations = WINDOWED[kind]
    kernel = tuple(weights[1:3]) if kind in _WEIGHTED else (option(4, "i", 0), option(3, "i", 0))
    dilation_w, dilation_h = (1 if slot is None else option(slot, "i", 1) for slot in dilations)
    return Window(option(0, "b", SAME), kernel, (option(slot_h, "i", 0), option(slot_w, "i", 0)),
                  (dilation_h, dilation_w))


def _is_constant(tensor: _Tensor) -> bool:
    return tensor.buffer > 0 and bool(tensor.data)


def _axes(values, rank: int) -> list:
    """The distinct axes in `values`, each taken modulo `rank`, ascending."""
    return sorted({a % rank for a in values})


def _option(op: _Operator, name: str):
    """Option `name` of the operator, by the slot, format and absent value
    its kind's entry in _KINDS gives."""
    slot, fmt, default = _KINDS[op.kind].options[name]
    return op.option(slot, fmt, default)


def _operator_reason(op: _Operator, tensors: list[_Tensor]) -> str:
    """Why `op` does not convert, or "" when it does: the first failing check
    of the constant operands, the kind's early checks, the dtypes and
    quantization of the data operands and outputs, the fused activation and
    the kind's late checks, in this order."""
    entry = _KINDS.get(op.kind)
    if entry is None:
        return "not supported"
    ins = [tensors[i] if i >= 0 else None for i in op.inputs]
    outs = [tensors[i] for i in op.outputs]
    for position in _CONSTANT_OPERANDS.get(op.kind, ()):
        if position < len(ins) and ins[position] is not None and not _is_constant(ins[position]):
            if position != _INDEX_SLOTS.get(op.kind):
                return f"input {position} must be a constant"
            if ins[position].type != "INT32":
                return f"{ins[position].type} run-time indices; the runtime takes int32"
    for check in entry.early:
        reason = check(op, ins, outs)
        if reason is not None:
            return reason
    data, checked = _checked_operands(op, ins, outs)
    reason = entry.dtypes(data, checked)
    if reason:
        return reason
    if any(len(t.scale) != 1 for t in [*data, *checked] if t.type in ("INT8", "UINT8")):
        return "activations must be quantized per tensor"
    if entry.fused:
        fused = _activation(_option(op, "fused_activation_function"))
        if fused not in ("none", "relu", "relu6"):
            return f"fused {fused}"
    for check in entry.late:
        reason = check(op, ins, outs)
        if reason:
            return reason
    return ""


def _checked_operands(op: _Operator, ins: list, outs: list[_Tensor]) -> tuple[list, list]:
    """The data operands and outputs the dtype and quantization checks read:
    no constant operand, weights, bias, bool operand or output, or index output."""
    if op.kind in _WEIGHTED:
        return [ins[_WEIGHTED[op.kind][0]]], outs
    positions, bool_output = _BOOL_SLOTS.get(op.kind, ((), False))
    data = [t for i, t in enumerate(ins) if t is not None and i not in positions
            and i not in _CONSTANT_OPERANDS.get(op.kind, ())]
    return data, [] if bool_output or op.kind in _INDEX else outs


def _float_or_int8(data: list, outs: list) -> str:
    tensors = [*data, *outs]
    if all(t.type == "FLOAT32" for t in tensors) or all(t.type == "INT8" for t in tensors):
        return ""
    return "activations must be all int8 or all float32"


def _float_only(data: list, outs: list) -> str:
    if all(t.type == "FLOAT32" for t in [*data, *outs]):
        return ""
    return _float_or_int8(data, outs) or "TFLite Micro runs it in float32 only"


def _quantize_dtypes(data: list, outs: list) -> str:
    # uint8 only enters and leaves an int8 model, as the converter writes
    # its boundaries for inference_input_type and inference_output_type uint8.
    if (data[0].type, outs[0].type) not in (("FLOAT32", "INT8"), ("INT8", "INT8"),
                                            ("UINT8", "INT8"), ("INT8", "UINT8")):
        return "only float or int8 to int8 converts"
    return ""


def _dequantize_dtypes(data: list, outs: list) -> str:
    if data[0].type != "INT8" or outs[0].type != "FLOAT32":
        return "only int8 to float converts"
    return ""


def _quant_spec_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    """A non-empty quant_spec names a quantization contract a runtime must
    refuse unless it knows it."""
    return "a non-standard quantization contract in quant_spec" if _option(op, "quant_spec") else None


def _count_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    field, side = ("values_count", "input") if op.kind == "PACK" else ("num", "output")
    stated, operands = _option(op, field), len(ins) if op.kind == "PACK" else len(outs)
    if stated != operands:
        return f"{field} {stated} for {operands} {side}{'s' if operands != 1 else ''}"
    return None


def _variables_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    values = [t for t in [*ins, *outs] if t is not None and t.type != "RESOURCE"]
    if any(t.type != "FLOAT32" for t in values):
        # The converter keeps variables in float even in int8 models.
        return "variables other than float32"
    return ""


def _if_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    if ins[0].type != "BOOL" or int(np.prod(ins[0].shape)) != 1:
        return "a condition other than one bool"
    if any(t.type not in ("FLOAT32", "INT32")
           for t in [*(t for t in ins[1:] if not _is_constant(t)), *outs]):
        return "operands other than float32 or int32"
    return ""


def _while_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    if any(t.type not in ("FLOAT32", "INT32") for t in [*ins, *outs]):
        return "loop variables other than float32 or int32"
    return ""


def _kept_constant_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    """For an operator on constants only, which the fold kept: its operands
    or options are the cause, not its dtype."""
    if not all(t is not None and _is_constant(t) for t in ins):
        return None
    return ((op.kind in _DATA_MOVEMENT and _data_movement_reason(op, ins, outs))
            or (op.kind in _OPTION_SHAPED and _option_shape_reason(op, ins, outs)) or None)


def _has_int32(ins: list, outs: list[_Tensor]) -> bool:
    return any(t is not None and t.type == "INT32" for t in [*ins, *outs])


def _int32_arithmetic(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    if not _has_int32(ins, outs):
        return None
    if any(t.type != "INT32" for t in [*ins, *outs]):
        return "int32 mixed with other operands"
    if _activation(_option(op, "fused_activation_function")) != "none":
        return "a fused activation on int32"
    if all(_is_constant(t) for t in ins):
        return "only constant operands"
    return ""


def _int32_comparison(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    if not _has_int32(ins, outs):
        return None
    return "int32 compared with another dtype" if any(t.type != "INT32" for t in ins) else ""


def _int32_movement(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    return "" if ins[0].type == outs[0].type == "INT32" else None


def _int32_cast(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    if not _has_int32(ins, outs):
        return None
    return "" if outs[0].type in {"FLOAT32", "INT32"} else "int32 cast to other than float32"


def _weights_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    source, position, bias_position = _WEIGHTED[op.kind]
    weights = ins[position]
    bias = ins[bias_position] if bias_position < len(ins) else None
    if ins[source].type == "FLOAT32":
        if weights.type != "FLOAT32" or not _is_constant(weights):
            reason = "weights of a float32 operator must be constant float32"
            if weights.type == "INT8" and _is_constant(weights):
                # Dynamic-range quantization: int8 weights, float activations.
                reason += ("; convert to full int8 or to float32 without dynamic-range "
                           "quantization")
            return reason
        if bias is not None and bias.type != "FLOAT32":
            return "bias of a float32 operator must be float32"
        return None
    # TFLite's int8 convolutions ignore the filter zero point, which the
    # converter's own QUANTIZE of a weight can leave nonzero.
    symmetric = op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D") or not np.any(weights.zero_point != 0)
    if weights.type != "INT8" or not symmetric or not _is_constant(weights):
        return "weights must be constant symmetric int8"
    if bias is not None and bias.type != "INT32":
        return "bias must be int32"
    return None


def _bool_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    positions, bool_output = _BOOL_SLOTS[op.kind]
    if any((t.type == "BOOL") != (i in positions) for i, t in enumerate(ins) if t is not None):
        return "operands of the wrong dtype"
    if (outs[0].type == "BOOL") != bool_output:
        return "an output of the wrong dtype"
    return None


def _cast_encoding_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    if outs[0].type == "INT8" and (
            outs[0].scale.size != 1 or outs[0].scale[0] != 1 or outs[0].zero_point[0] != 0):
        # TFLite copies the raw 0 or 1 into the int8 output.
        return "int8 output encoding other than scale 1 and zero point 0"
    return None


def _index_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    return f"{outs[0].type} indices; TFLite Micro writes int32" if outs[0].type != "INT32" else None


def _constant_only_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    return "only constant operands" if all(_is_constant(t) for t in ins) else ""


def _concat_quantization_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    if outs[0].type == "INT8" and any(t.scale[0] != outs[0].scale[0]
                                      or t.zero_point[0] != outs[0].zero_point[0] for t in ins):
        return "inputs quantized differently from the output"
    return ""


def _l2_fused_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    fused = _option(op, "fused_activation_function")
    return f"fused {_activation(fused)}" if fused != 0 else ""


def _reduction_axes_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    axes = _axes(ins[1].array().reshape(-1).tolist(), len(ins[0].shape))
    if not axes or (op.kind != "MEAN" and axes != list(range(axes[0], axes[-1] + 1))):
        return "axes that are not adjacent"
    return ""


def _beta_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    return "beta other than 1" if _option(op, "beta") != 1.0 else ""


def _weights_format_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    return "shuffled weights format" if _option(op, "weights_format") != 0 else ""


def _rank_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    return "operands of different rank" if len(ins[0].shape) != len(ins[1].shape) else ""


def _resize_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    if _option(op, "align_corners") and _option(op, "half_pixel_centers"):
        return "align_corners and half_pixel_centers together"
    return ""


def _integer_use(op: _Operator) -> bool:
    """Operators that take an int32 model input as a value."""
    return op.kind in ("ADD", "SUB", "MUL", "CAST", "RESHAPE", "TRANSPOSE", "IF", "WHILE", *_COMPARISONS)


def _same_quantization(tensors) -> bool:
    first = tensors[0]
    return all(t.type == first.type and np.array_equal(t.scale, first.scale)
               and np.array_equal(t.zero_point, first.zero_point) for t in tensors[1:])


def _strided_slice_bounds(op: _Operator, ins: list[_Tensor]):
    """Per axis (begin, size, shrink) of a stride-1 STRIDED_SLICE, or None."""
    shape, options = ins[0].shape, _StridedSliceOptions.of(op)
    begins, ends, strides = ([int(v) for v in ins[i].array().reshape(-1)] for i in (1, 2, 3))
    if (options.ellipsis_mask or options.new_axis_mask or options.offset
            or len(begins) != len(shape) or any(v != 1 for v in strides)):
        return None
    axes = _strided_slice_axes(options, shape, begins, ends, [1] * len(shape))
    if axes is None or any(stop <= start for start, stop, _, _ in axes):
        return None
    return [(start, stop - start, dropped) for start, stop, _, dropped in axes]


def _gather_run(op: _Operator, ins: list[_Tensor]):
    """(axis, first, count, scalar) when a GATHER takes a contiguous run of one
    axis, or None."""
    indices = [int(v) for v in ins[1].array().reshape(-1)]
    axis = _option(op, "axis") % len(ins[0].shape)
    extent = ins[0].shape[axis]
    indices = [i + extent if i < 0 else i for i in indices]
    if (_option(op, "batch_dims") or not indices or any(not 0 <= i < extent for i in indices)
            or indices != list(range(indices[0], indices[0] + len(indices)))):
        return None
    return axis, indices[0], len(indices), len(ins[1].shape) == 0


def _svdf_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    x, feature, time, bias, state = ins
    y = outs[0]
    if bias is None:
        # TFLite Micro's Prepare reads the bias whether or not it is there.
        return "no bias, which TFLite Micro requires"
    if len(x.shape) != 2:
        return "input of rank other than 2"
    rank, filters = _option(op, "rank"), feature.shape[0] if feature.shape else 0
    if rank <= 0 or filters % rank:
        return f"rank {rank} for {filters} filters"
    reason = _shape_reason(f"rank {rank}", y.shape, [x.shape[0], filters // rank])
    if reason:
        return reason
    if x.type == "FLOAT32":
        if any(t.type != "FLOAT32" for t in (feature, time, bias, state, y)):
            return "float32 input with operands of another dtype"
        fused = _activation(_option(op, "fused_activation_function"))
        if fused not in ("none", "relu", "relu6"):
            return f"fused {fused}"
        return ""
    if x.type != "INT8" or y.type != "INT8" or feature.type != "INT8" or bias.type != "INT32":
        return "activations must be all int8 or all float32"
    if time.type != "INT16" or state.type != "INT16":
        return "int8 state; the converter writes int16"
    if any(len(t.scale) != 1 for t in (x, feature, time, state, y)):
        return "operands must be quantized per tensor"
    return ""


# Operand positions of the LSTM's gate weights and biases, gates in order i, f, c, o.
_LSTM_CONSTANTS = (*range(1, 9), *range(12, 16))


def _lstm_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    """What TFLite Micro runs: every gate present, no peepholes, projection or
    layer normalization, and a tanh cell activation."""
    ins = ins + [None] * (24 - len(ins))
    if any(ins[i] is None for i in (0, *_LSTM_CONSTANTS, 18, 19)):
        return "a missing gate, which TFLite Micro requires"
    if any(ins[i] is not None for i in (9, 10, 11, 16, 17, 20, 21, 22, 23)):
        return "peepholes, projection or layer normalization, which TFLite Micro does not run"
    if len(ins[0].shape) != 3:
        return "input of rank other than 3"
    if any(len(ins[i].shape) == 1 for i in range(5, 9)):
        return "diagonal recurrent weights, which TFLite Micro does not run"
    activation = _activation(_option(op, "fused_activation_function"))
    if activation != "tanh":
        return f"cell activation {activation}"
    if any(not _is_constant(ins[i]) for i in _LSTM_CONSTANTS):
        return "weights and biases must be constant"
    # time_major picks the batch axis of the input, which the states are sized by.
    time_major = _option(op, "time_major")
    given = [ins[0].shape[1 if time_major else 0], ins[18].shape[-1] if ins[18].shape else 0]
    for role, state in (("hidden state", ins[18]), ("cell state", ins[19])):
        reason = _shape_reason(f"time_major {str(time_major).lower()}", state.shape, given, role)
        if reason:
            return reason
    tensors = [ins[i] for i in (0, *_LSTM_CONSTANTS, 18, 19)] + outs
    if all(t.type == "FLOAT32" for t in tensors):
        return ""
    x, hidden, cell, y = ins[0], ins[18], ins[19], outs[0]
    if (x.type != "INT8" or hidden.type != "INT8" or y.type != "INT8"
            or any(ins[i].type != "INT8" for i in range(1, 9))
            or any(ins[i].type != "INT32" for i in range(12, 16))):
        return "activations must be all int8 or all float32"
    if cell.type != "INT16" or cell.zero_point.size != 1 or cell.zero_point[0] != 0:
        return "cell state other than symmetric int16"
    if any(len(t.scale) != 1 for t in (x, hidden, cell, y, *(ins[i] for i in range(1, 9)))):
        return "operands must be quantized per tensor"
    return ""


def _detection_options(op: _Operator) -> dict:
    """The options as TFLite Micro reads them, with its defaults."""
    found = flexbuffer_map(op.custom_options)

    def number(key, kind, default=None):
        value = found.get(key)
        if value is None:
            if default is None:
                raise ValueError(f"option {key} is missing")
            return default
        return kind(value)

    return {"max_detections": number("max_detections", int),
            "max_classes_per_detection": number("max_classes_per_detection", int),
            "detections_per_class": number("detections_per_class", int, 100),
            "use_regular_nms": number("use_regular_nms", bool, False),
            "score_threshold": float(np.float32(number("nms_score_threshold", float))),
            "iou_threshold": float(np.float32(number("nms_iou_threshold", float))),
            "num_classes": number("num_classes", int),
            "scales": [float(np.float32(number(f"{axis}_scale", float))) for axis in "yxhw"]}


def _detection_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    if len(ins) != 3 or any(t is None for t in ins) or len(outs) != 4:
        return "operands other than box encodings, scores and anchors, and four outputs"
    try:
        options = _detection_options(op)
    except ValueError as error:
        return str(error)
    boxes, scores, anchors = ins
    if any(t.type != "FLOAT32" for t in (*ins, *outs)):
        # TFLite Micro reads float operands; an int8 model dequantizes them first.
        return "operands or outputs other than float32"
    if not _is_constant(anchors):
        return "anchors that are not a constant"
    count, classes, detections = (len(boxes.shape) == 3 and boxes.shape[1], options["num_classes"],
                                  options["max_detections"])
    if (tuple(boxes.shape) != (1, count, 4) or len(scores.shape) != 3
            or tuple(scores.shape[:2]) != (1, count) or tuple(anchors.shape) != (count, 4)):
        return "shapes other than box encodings [1, boxes, 4], scores [1, boxes, classes] and anchors [boxes, 4]"
    if not 1 <= classes <= scores.shape[2] <= classes + 1:
        return "scores for other than the classes and at most one background column"
    if not 0.0 < options["iou_threshold"] <= 1.0:
        return "an IoU threshold outside (0, 1]"
    if options["use_regular_nms"]:
        if options["detections_per_class"] <= 0:
            return "no detections per class"
    elif options["max_classes_per_detection"] != 1:
        # TFLite Micro's fast form writes past its outputs for more than one.
        return "more than one class per detection in the fast form"
    if detections <= 0 or [tuple(t.shape) for t in outs] != [
            (1, detections, 4), (1, detections), (1, detections), (1,)]:
        return "outputs other than [1, detections, 4], [1, detections], [1, detections] and [1]"
    return ""


def _spatial_mean(op: _Operator, tensors) -> bool:
    """A MEAN over the height and width of a feature map, which the compiler
    runs as a global average pool."""
    source = tensors[op.inputs[0]]
    axes = _axes(tensors[op.inputs[1]].array().reshape(-1).tolist(), len(source.shape))
    return len(source.shape) == 4 and axes == [1, 2]


def _window_reason(op: _Operator, ins: list[_Tensor], outs: list[_Tensor]) -> str:
    """TFLite Micro pads a window by its padding option and the input size and
    maps depthwise channels by depth_multiplier, but runs over the output the
    file states; the conversion pads by the stated shapes, so both must agree."""
    if op.kind == "DEPTHWISE_CONV_2D":
        multiplier, channels, produced = _option(op, "depth_multiplier"), ins[0].shape[3], outs[0].shape[3]
        if multiplier * channels != produced:
            return f"depth_multiplier {multiplier} for {channels} input and {produced} output channels"
    win = window(op.kind, op.option, ins[1].shape if op.kind in _WEIGHTED else None)
    if win.padding not in (SAME, VALID):
        return f"padding {win.padding}, neither SAME nor VALID"
    if min(win.strides) < 1:
        return f"a {win.strides[0]}x{win.strides[1]} stride; strides must be at least 1"
    if min(win.kernel) < 1:
        return f"a {win.kernel[0]}x{win.kernel[1]} filter; filter sizes must be at least 1"
    # A transposed convolution's input is what a convolution of its output gives.
    source, target = (outs[0], ins[2]) if op.kind == "TRANSPOSE_CONV" else (ins[0], outs[0])
    given = win.outputs(source.shape[1:3])
    name = ("SAME", "VALID")[win.padding]
    if op.kind == "TRANSPOSE_CONV":
        return _shape_reason(f"{name} padding of the output", target.shape[1:3], given, "input")
    return _shape_reason(f"{name} padding", target.shape[1:3], given)


def _data_movement_reason(op: _Operator, ins: list[_Tensor], outs: list[_Tensor]) -> str:
    slot = _INDEX_SLOTS.get(op.kind)
    data = [t for i, t in enumerate(ins) if not _is_constant(t) and i != slot]
    if not _same_quantization([*data, *outs]):
        return "inputs and outputs quantized differently"
    if op.kind == "SLICE" and any(t.array().size < len(ins[0].shape) for t in ins[1:3]):
        # TFLite Micro reads one entry per input axis from each.
        return "begin or size with fewer entries than the input has axes"
    if op.kind == "STRIDED_SLICE":
        options = _StridedSliceOptions.of(op)
        if options.ellipsis_mask or options.new_axis_mask or options.offset:
            return "ellipsis, new axes or offset slices"
        rank = len(ins[0].shape)
        if rank > 4:
            return "slices of rank above 4"
        if any(ins[i].array().size < rank for i in (1, 2, 3)):
            # TFLite Micro reads one entry per input axis from each, and no more.
            return "begin, end or strides with fewer entries than the input has axes"
        steps = [int(v) for v in ins[3].array().reshape(-1)[:rank]]
        if any(step == 0 for step in steps):
            return "a zero stride"
        if any(options.shrink_axis_mask >> axis & 1 and step < 0 for axis, step in enumerate(steps)):
            # TFLite Micro leaves such an output unwritten.
            return "a dropped axis with a negative stride"
    if op.kind == "REVERSE_V2":
        axes = _axes(ins[1].array().reshape(-1).tolist(), len(ins[0].shape))
        if not axes or axes != list(range(axes[0], axes[-1] + 1)):
            return "axes that are not adjacent"
    if op.kind in ("SPACE_TO_DEPTH", "DEPTH_TO_SPACE", "SPACE_TO_BATCH_ND",
                   "BATCH_TO_SPACE_ND") and len(ins[0].shape) != 4:
        return "only rank-4 inputs convert"
    if op.kind in ("SPACE_TO_DEPTH", "DEPTH_TO_SPACE"):
        return _block_reason(op, ins[0], outs[0])
    return ""


def _block_reason(op: _Operator, source: _Tensor, target: _Tensor) -> str:
    """TFLite Micro shapes the output by block_size; the conversion reshapes to
    the output the file states, so both must agree."""
    block, (n, h, w, c) = _option(op, "block_size"), source.shape
    if op.kind == "SPACE_TO_DEPTH":
        valid = block > 0 and h % block == 0 and w % block == 0
        given = [n, h // block, w // block, c * block * block] if valid else []
    else:
        valid = block > 0 and c % (block * block) == 0
        given = [n, h * block, w * block, c // (block * block)] if valid else []
    if not valid:
        return f"block_size {block} for a {_dims(source.shape)} input"
    return _shape_reason(f"block_size {block}", target.shape, given)


def _dims(shape) -> str:
    return "x".join(map(str, shape)) or "scalar"


def _shape_reason(setting: str, stated, given, role: str = "output") -> str:
    """Why a stated shape is not the one `setting` gives, or "" when it is."""
    if list(stated) == list(given):
        return ""
    return f"a {_dims(stated)} {role} where {setting} gives {_dims(given)}"


def _option_shape_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    """The options give the output shapes from the input shapes, and TFLite
    Micro computes by them; the conversion takes the stated shapes, so both
    must agree."""
    shapes = [list(t.shape) for t in ins if t is not None]
    first, inputs = shapes[0], ", ".join(_dims(s) for s in shapes)
    if op.kind == "STRIDED_SLICE":
        axes = _strided_slice_axes(_StridedSliceOptions.of(op), first,
                                   *([int(v) for v in t.array().reshape(-1)] for t in ins[1:4]))
        if axes is None:
            # The operator check refuses every other cause before this one.
            return "a dropped axis at an index outside the axis"
        given = [[len(range(start, stop, step)) for start, stop, step, dropped in axes if not dropped]]
        setting = "the slice"
    elif op.kind in ("PACK", "UNPACK", "CONCATENATION"):
        axis = _option(op, "axis")
        setting, rank = f"axis {axis}", len(first) + (op.kind == "PACK")
        a = axis + rank if axis < 0 else axis
        if op.kind == "UNPACK":
            if not 0 <= a < rank or first[a] != len(outs):
                count = f"{len(outs)} output{'s' if len(outs) != 1 else ''}"
                return f"{setting} for a {_dims(first)} input and {count}"
            given = [first[:a] + first[a + 1:]] * len(outs)
        elif op.kind == "PACK":
            if not 0 <= a < rank or any(s != first for s in shapes):
                return f"{setting} for inputs {inputs}"
            given = [first[:a] + [len(shapes)] + first[a:]]
        else:
            if not 0 <= a < rank or any(len(s) != rank or s[:a] + s[a + 1:] != first[:a] + first[a + 1:]
                                        for s in shapes):
                return f"{setting} for inputs {inputs}"
            given = [first[:a] + [sum(s[a] for s in shapes)] + first[a + 1:]]
    elif op.kind == "GATHER":
        axis, batch = _option(op, "axis"), _option(op, "batch_dims")
        setting, data, indices = f"axis {axis}, batch_dims {batch}", shapes[0], shapes[1]
        a = axis + len(data) if axis < 0 else axis
        b = batch + len(indices) if batch < 0 else batch
        if not (0 <= b <= a < len(data) and b <= len(indices) and data[:b] == indices[:b]):
            return f"{setting} for inputs {inputs}"
        given = [data[:a] + indices[b:] + data[a + 1:]]
    else:
        adj_x, adj_y = _option(op, "adj_x"), _option(op, "adj_y")
        setting = f"adj_x {str(adj_x).lower()}, adj_y {str(adj_y).lower()}"
        rank = max(len(s) for s in shapes[:2])
        lhs, rhs = ([1] * (rank - len(s)) + s for s in shapes[:2])
        batches = list(zip(lhs[:-2], rhs[:-2]))
        if (min(len(s) for s in shapes[:2]) < 2
                or (lhs[-2] if adj_x else lhs[-1]) != (rhs[-1] if adj_y else rhs[-2])
                or any(left != right and 1 not in (left, right) for left, right in batches)):
            return f"{setting} for inputs {inputs}"
        given = [[right if left == 1 else left for left, right in batches]
                 + [lhs[-1] if adj_x else lhs[-2], rhs[-2] if adj_y else rhs[-1]]]
    for out, shape in zip(outs, given):
        reason = _shape_reason(setting, out.shape, shape)
        if reason:
            return reason
    return ""


def describe(data: bytes) -> dict:
    """The model's own facts, readable whether or not it converts."""
    model, graphs = _read(data)
    name, tensors, inputs, outputs, operators = graphs[0] if graphs else ("", [], [], [], [])

    def interface(index):
        tensor = tensors[index]
        return {"name": tensor.name, "kind": "tensor", "dtype": _dtype_name(tensor),
                "shape": list(tensor.shape)}

    constants = [t for t in tensors if _is_constant(t)]
    facts = {
        "name": name or model.description,
        "tflite_version": model.version,
        "description": model.description,
        "subgraphs": len(graphs),
        "graph": {
            "name": name,
            "inputs": [interface(i) for i in inputs],
            "outputs": [interface(i) for i in outputs],
            "operators": [{"name": f"{op.kind.lower()}_{op.index}", "type": op.kind, "domain": "",
                           "version": op.version,
                           "inputs": [tensors[i].name for i in op.inputs if i >= 0],
                           "outputs": [tensors[i].name for i in op.outputs]} for op in operators],
            "initializers": [{"name": t.name, "dtype": _dtype_name(t), "shape": list(t.shape),
                              "size_bytes": len(t.data), "storage": "embedded",
                              "scales": len(t.scale),
                              "symmetric": bool(len(t.zero_point)) and not np.any(t.zero_point != 0)}
                             for t in constants],
            # Every tensor the operators produce or read that is not a constant.
            "activations": [{"name": t.name, "dtype": _dtype_name(t), "shape": list(t.shape),
                             "variable": t.variable}
                            for t in tensors if not _is_constant(t)],
            "sparse_initializers": [],
        },
    }
    # After the facts: narrowing rewrites the types of index inputs.
    facts["unsupported"] = _reasons([_narrow_index_inputs(g) for g in graphs])
    return facts


# Ranks 3 and 4 are channels-first in ONNX, as the runtime stores them
# channels-last; other ranks keep TFLite's order.
def _to_first(rank: int) -> list[int]:
    return [0, rank - 1, *range(1, rank - 1)] if rank in (3, 4) else list(range(rank))


def _to_last(rank: int) -> list[int]:
    return [0, *range(2, rank), 1] if rank in (3, 4) else list(range(rank))


def _onnx_axis(axis: int, rank: int) -> int:
    return _to_last(rank)[axis % rank]


def _onnx_shape(shape) -> list[int]:
    return [shape[i] for i in _to_first(len(shape))]


def _layout_free(shape) -> bool:
    """True when both axis orders hold the elements in the same sequence."""
    return (len(shape) not in (3, 4) or shape[-1] == 1
            or int(np.prod(shape[1:-1])) == 1)


class _Converter:
    """Builds the QDQ ONNX graph operator by operator; `values` maps a TFLite
    tensor index to the ONNX value holding it, rank-4 values in NCHW."""

    def __init__(self, tensors: list[_Tensor], outputs=(), consumed=()):
        self.b = GraphBuilder()
        self.tensors = tensors
        self.values: dict[int, str] = {}
        # int8 and uint8 model outputs are the QuantizeLinear itself, as TFLite
        # hands them over.
        self.int8_outputs = {i for i in outputs if tensors[i].type in ("INT8", "UINT8")}
        self.consumed = set(consumed)
        # Output shapes of the operators ONNX has no standard form for.
        self.value_info = []
        # Resource handles by tensor, and per variable its state input and
        # latest value.
        self.handles: dict[int, str] = {}
        self.variables: dict[str, dict] = {}
        # Per variable tensor, its state input and the value written for the next run.
        self.held_state: dict[int, dict] = {}
        self.detection_operands: set[int] = set()
        # Tensors crossing a control-flow boundary, which an int8 model keeps in
        # float32: control-flow operands and results, a subgraph's own inputs
        # and outputs. A QUANTIZE or DEQUANTIZE there stays an operator.
        self.float_boundary: set[int] = set()

    def value(self, index: int, rank: int | None = None) -> str:
        """The ONNX value of a tensor; a constant is broadcast-aligned to `rank`."""
        if index in self.values and rank is None:
            return self.values[index]
        tensor = self.tensors[index]
        array = tensor.array()
        rank = rank or array.ndim
        array = array.reshape((1,) * (rank - array.ndim) + array.shape)
        # A scalar has no axis to quantize along.
        axis = _onnx_axis(tensor.quantized_dimension + rank - len(tensor.shape), rank) if rank else 0
        array = array.transpose(_to_first(rank))
        if tensor.type in ("FLOAT32", "BOOL", "INT32"):
            return self.b.constant(array, tensor.name)
        return self.b.dequantized_constant(array, tensor.scale, tensor.zero_point, tensor.name,
                                           axis=axis)

    def operands(self, op: _Operator) -> list[str]:
        """A broadcasting operator's operands at the output's rank: a constant
        aligned when it is built, a lower-rank tensor reshaped with leading
        ones in TFLite's axis order, as TFLite broadcasts from the right."""
        rank = len(self.tensors[op.outputs[0]].shape)
        result = []
        for i in op.inputs:
            if i not in self.values:
                result.append(self.value(i, rank))
                continue
            shape = list(self.tensors[i].shape)
            if len(shape) == rank:
                result.append(self.values[i])
                continue
            tag = f"{self.tensors[i].name}_rank{rank}"
            full = [1] * (rank - len(shape)) + shape
            result.append(self.held(self.reshape(self.values[i], shape, full, tag, i), i, tag + "_q"))
        return result

    def slice_axis(self, x: str, shape, axis: int, begin: int, size: int, tag: str,
                   index: int, hold: bool) -> str:
        """[begin, begin + size) of TFLite axis `axis` of `x`: the middle part of
        a Split. Every part is held as tensor `index` is quantized except the
        one returned when `hold` is False."""
        extent = shape[axis]
        if begin == 0 and size == extent:
            return x
        sizes = [part for part in (begin, size, extent - begin - size) if part > 0]
        names = [f"{tag}_part{i}" for i in range(len(sizes))]
        parts = self.b.multi_node("Split", [x, self.b.constant(np.asarray(sizes, np.int64),
                                                                tag + "_sizes")],
                                  names, axis=_onnx_axis(axis, len(shape)))
        chosen = 1 if begin > 0 else 0
        held = [part if (i == chosen and not hold) else self.held(part, index, part + "_q")
                for i, part in enumerate(parts)]
        return held[chosen]

    def last_to_last(self, x: str, rank: int, tag: str, index: int, to_tflite: bool) -> str:
        """`x` moved between the ONNX and the TFLite axis order, held."""
        if rank not in (3, 4):
            return x
        perm = _to_last(rank) if to_tflite else _to_first(rank)
        return self.held(self.b.node("Transpose", [x], tag, perm=perm), index, tag + "_q")

    def ints(self, index: int) -> list[int]:
        return [int(v) for v in self.tensors[index].array().reshape(-1)]

    def held(self, source: str, index: int, name: str) -> str:
        """`source` requantized as tensor `index` is, so a data-movement step
        between two operators stays in int8."""
        tensor = self.tensors[index]
        if tensor.type in ("FLOAT32", "BOOL", "INT32"):
            return source
        return self.b.requantized(source, float(tensor.scale[0]), int(tensor.zero_point[0]), name)

    def finish(self, source: str, index: int) -> None:
        tensor = self.tensors[index]
        if index in self.int8_outputs:
            b = self.b
            scale = b.constant(np.float32(tensor.scale[0]), tensor.name + "_scale")
            zero = b.constant(np.array(int(tensor.zero_point[0]), _NUMPY[tensor.type]),
                              tensor.name + "_zero_point")
            quantized = b.node("QuantizeLinear", [source, scale, zero], tensor.name)
            self.values[index] = (b.node("DequantizeLinear", [quantized, scale, zero],
                                         tensor.name + "_float")
                                  if index in self.consumed else quantized)
        elif tensor.type in ("FLOAT32", "BOOL", "INT32"):
            self.values[index] = self.b.node("Identity", [source], tensor.name)
        else:
            self.values[index] = self.b.requantized(source, float(tensor.scale[0]),
                                                    int(tensor.zero_point[0]), tensor.name)

    def reshape(self, source: str, source_shape, shape, tag: str, index: int) -> str:
        """TFLite's row-major reshape of a channels-last tensor, between ONNX
        layouts; intermediate steps are held as tensor `index` is quantized."""
        b = self.b
        if not _layout_free(source_shape):
            source = self.held(b.node("Transpose", [source], tag + "_last",
                                      perm=_to_last(len(source_shape))), index, tag + "_last_q")
        free_out = _layout_free(shape)
        target = _onnx_shape(shape) if free_out else list(shape)
        y = b.node("Reshape", [source, b.constant(np.asarray(target, np.int64), tag + "_shape")], tag)
        if not free_out:
            y = b.node("Transpose", [self.held(y, index, tag + "_q")], tag + "_first",
                       perm=_to_first(len(shape)))
        return y

    def convert(self, op: _Operator) -> None:
        """Adds `op` by its kind's converter; a result it returns is finished
        as the output, after the fused activation."""
        entry = _KINDS[op.kind]
        tag = self.tensors[op.outputs[0]].name + "_float" if op.outputs else ""
        y = entry.convert(self, op, tag)
        if y is not None:
            fused = _activation(_option(op, "fused_activation_function")) if entry.fused else "none"
            self.finish(self.b.fused_activation(y, fused, tag), op.outputs[0])

    def _variable(self, op: _Operator, tag: str) -> None:
        """A read sees the variable's latest value: the state input, or what
        this invocation assigned before it."""
        ins, outs = op.inputs, op.outputs
        if op.kind == "VAR_HANDLE":
            self.handles[outs[0]] = _var_name(op)
        elif op.kind == "READ_VARIABLE":
            self.values[outs[0]] = self.variables[self.handles[ins[0]]]["current"]
        elif op.kind == "ASSIGN_VARIABLE":
            self.variables[self.handles[ins[0]]]["current"] = self.value(ins[1])

    def _index(self, op: _Operator, tag: str) -> None:
        """The index in TFLite's axis order, cast to the int32 TFLite writes."""
        ins, b = op.inputs, self.b
        rank = len(self.tensors[ins[0]].shape)
        x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
        y = b.node(_INDEX[op.kind], [x], tag, axis=self.ints(ins[1])[0] % rank, keepdims=0,
                   select_last_index=0)
        self.values[op.outputs[0]] = b.node("Cast", [y], self.tensors[op.outputs[0]].name,
                                            to=TensorProto.INT32)

    def _quantization(self, op: _Operator, tag: str) -> str | None:
        ins, outs, kind = op.inputs, op.outputs, op.kind
        if ((ins[0] if kind == "QUANTIZE" else outs[0]) in self.float_boundary
                and self.tensors[ins[0] if kind == "DEQUANTIZE" else outs[0]].type == "INT8"):
            return self.custom("Quantize" if kind == "QUANTIZE" else "Dequantize",
                               [self.value(ins[0])], tag, _onnx_shape(self.tensors[outs[0]].shape))
        if kind == "DEQUANTIZE" and outs[0] in self.detection_operands:
            # The detection operator reads the int8 tensor through its quantization.
            self.values[outs[0]] = self.value(ins[0])
            return None
        return self.value(ins[0])

    def _pool(self, op: _Operator, tag: str) -> str:
        attributes = {"count_include_pad": 0} if op.kind == "AVERAGE_POOL_2D" else {}
        return self.b.node("MaxPool" if op.kind == "MAX_POOL_2D" else "AveragePool",
                           [self.value(op.inputs[0])], tag, **self._pool_window(op), **attributes)

    def _l2_pool(self, op: _Operator, tag: str) -> str:
        b = self.b
        x = self.value(op.inputs[0])
        pooled = b.node("AveragePool", [b.node("Mul", [x, x], tag + "_square")], tag + "_mean",
                        count_include_pad=0, **self._pool_window(op))
        return b.node("Sqrt", [pooled], tag)

    def _concatenation(self, op: _Operator, tag: str) -> str:
        out = self.tensors[op.outputs[0]]
        return self._concat([(self.value(i), list(self.tensors[i].shape), i) for i in op.inputs],
                            _option(op, "axis") % len(out.shape), op.outputs[0], tag)

    def _comparison(self, op: _Operator, tag: str) -> str:
        y = self.b.node(_COMPARISONS[op.kind], self.operands(op),
                        tag + "_equal" if op.kind == "NOT_EQUAL" else tag)
        return self.b.node("Not", [y], tag) if op.kind == "NOT_EQUAL" else y

    def _cast(self, op: _Operator, tag: str) -> str:
        # A bool cast to int8 is the float 0.0 or 1.0 quantized into the output.
        x = self.value(op.inputs[0])
        if self.tensors[op.outputs[0]].type == "INT32":
            return self.b.node("Identity", [x], tag)
        return self.b.node("Cast", [x], tag, to=TensorProto.FLOAT)

    def _softmax(self, op: _Operator, tag: str) -> str:
        return self.b.node("Softmax" if op.kind == "SOFTMAX" else "LogSoftmax",
                           [self.value(op.inputs[0])], tag,
                           axis=_onnx_axis(-1, len(self.tensors[op.outputs[0]].shape)))

    def _mean(self, op: _Operator, tag: str) -> str:
        if not _spatial_mean(op, self.tensors):
            return self._reduction(op, tag)
        ins, out = op.inputs, self.tensors[op.outputs[0]]
        source = self.tensors[ins[0]]
        rank = len(source.shape)
        axes = sorted(_onnx_axis(a, rank) for a in self.ints(ins[1]))
        y = self.b.node("ReduceMean", [self.value(ins[0])], tag, axes=axes, keepdims=1)
        if len(out.shape) != rank:
            reduced = _axes(self.ints(ins[1]), rank)
            kept = [1 if i in reduced else d for i, d in enumerate(source.shape)]
            y = self.reshape(self.held(y, op.outputs[0], tag + "_kept"), kept, out.shape,
                             tag + "_squeezed", op.outputs[0])
        return y

    def _shape_only(self, op: _Operator, tag: str) -> str:
        return self.reshape(self.value(op.inputs[0]), self.tensors[op.inputs[0]].shape,
                            self.tensors[op.outputs[0]].shape, tag, op.outputs[0])

    def _transpose(self, op: _Operator, tag: str) -> str:
        rank = len(self.tensors[op.outputs[0]].shape)
        perm = [p % rank for p in self.ints(op.inputs[1])]
        first, last = _to_first(rank), _to_last(rank)
        perm = [last[perm[first[i]]] for i in range(rank)]
        return self.b.node("Transpose", [self.value(op.inputs[0])], tag, perm=perm)

    def _split(self, op: _Operator, tag: str) -> None:
        ins, outs = op.inputs, op.outputs
        source_index, axis_index = (ins[1], ins[0]) if op.kind == "SPLIT" else (ins[0], ins[2])
        rank = len(self.tensors[source_index].shape)
        axis = self.ints(axis_index)[0] % rank
        sizes = self.b.constant(np.asarray([self.tensors[i].shape[axis] for i in outs], np.int64),
                                self.tensors[outs[0]].name + "_sizes")
        names = [self.tensors[i].name + "_float" for i in outs]
        parts = self.b.multi_node("Split", [self.value(source_index), sizes], names,
                                  axis=_onnx_axis(axis, rank))
        for index, part in zip(outs, parts):
            self.finish(part, index)

    def _pad(self, op: _Operator, tag: str) -> str:
        ins = op.inputs
        pads = self._pads_constant(op, tag)
        source = self.tensors[ins[0]]
        if op.kind != "PADV2" or len(ins) < 3 or ins[2] < 0:
            fill = 0.0
        elif source.type == "FLOAT32":
            fill = float(self.tensors[ins[2]].array().reshape(-1)[0])
        else:
            fill = float((int(self.ints(ins[2])[0]) - source.zero_point[0]) * source.scale[0])
        return self.b.node("Pad", [self.value(ins[0]), pads,
                                   self.b.constant(np.float32(fill), tag + "_fill")],
                           tag, mode="constant")

    def _mirror_pad(self, op: _Operator, tag: str) -> str:
        pads = self._pads_constant(op, tag)
        if _option(op, "mode") == 0:
            return self.b.node("Pad", [self.value(op.inputs[0]), pads], tag, mode="reflect")
        return self.custom("MirrorPad", [self.value(op.inputs[0]), pads], tag,
                           _onnx_shape(self.tensors[op.outputs[0]].shape), mode="symmetric")

    def _gather(self, op: _Operator, tag: str) -> str:
        """A GATHER of a contiguous run of constant indices slices the axis."""
        ins = op.inputs
        run = (_is_constant(self.tensors[ins[1]])
               and _gather_run(op, [self.tensors[i] for i in ins]))
        if not run:
            return self._gather_node("Gather", ins[0], ins[1], _option(op, "axis"), op, tag)
        axis, first, count, scalar = run
        bounds = [(0, extent, False) for extent in self.tensors[ins[0]].shape]
        bounds[axis] = (first, count, scalar)
        return self._cut(op, bounds, tag)

    def _strided_slice(self, op: _Operator, tag: str) -> str:
        bounds = _strided_slice_bounds(op, [self.tensors[i] for i in op.inputs])
        return self._slice_steps(op, tag) if bounds is None else self._cut(op, bounds, tag)

    def _slice(self, op: _Operator, tag: str) -> str:
        source = self.tensors[op.inputs[0]]
        bounds = [(begin, (extent - begin) if size == -1 else size, False)
                  for begin, size, extent in zip(self.ints(op.inputs[1]), self.ints(op.inputs[2]),
                                                 source.shape)]
        return self._cut(op, bounds, tag)

    def _cut(self, op: _Operator, bounds, tag: str) -> str:
        """Per axis (begin, size, dropped) of the first input, each cut axis a
        Split part, reshaped to the output's shape."""
        ins, b = op.inputs, self.b
        source, out = self.tensors[ins[0]], self.tensors[op.outputs[0]]
        cut = [axis for axis, (begin, size, _) in enumerate(bounds)
               if not (begin == 0 and size == source.shape[axis])]
        y, shape = self.value(ins[0]), list(source.shape)
        for step, axis in enumerate(cut):
            begin, size, _ = bounds[axis]
            y = self.slice_axis(y, shape, axis, begin, size, f"{tag}_slice{step}", ins[0],
                                hold=step < len(cut) - 1)
            shape[axis] = size
        if list(shape) != list(out.shape):
            if cut:
                y = self.held(y, ins[0], tag + "_sliced")
            y = self.reshape(y, shape, out.shape, tag, op.outputs[0])
        elif not cut:
            y = b.node("Identity", [y], tag)
        return y

    def _reverse(self, op: _Operator, tag: str) -> str:
        ins = op.inputs
        rank = len(self.tensors[op.outputs[0]].shape)
        x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
        axes = self.b.constant(np.asarray(self.ints(ins[1]), np.int64), tag + "_axes")
        return self.tflite_order_out(self.custom("ReverseV2", [x, axes], tag + "_reversed",
                                                 self.tensors[op.outputs[0]].shape),
                                     op.outputs[0], tag)

    def _dynamic_update_slice(self, op: _Operator, tag: str) -> str:
        ins, out = op.inputs, self.tensors[op.outputs[0]]
        rank = len(out.shape)
        if not _is_constant(self.tensors[ins[2]]):
            # Run-time starts are in TFLite's axis order, so the update runs there.
            x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
            u = self.last_to_last(self.value(ins[1]), rank, tag + "_update", ins[1], True)
            return self.tflite_order_out(self.custom("DynamicUpdateSlice", [x, u, self.value(ins[2])],
                                                     tag + "_updated", out.shape), op.outputs[0], tag)
        starts = [self.ints(ins[2])[a] for a in _to_first(rank)]
        return self.custom("DynamicUpdateSlice",
                           [self.value(ins[0]), self.value(ins[1]),
                            self.b.constant(np.asarray(starts, np.int64), tag + "_starts")],
                           tag, _onnx_shape(out.shape))

    def _pack(self, op: _Operator, tag: str) -> str:
        axis = _option(op, "axis") % len(self.tensors[op.outputs[0]].shape)
        parts = []
        for step, i in enumerate(op.inputs):
            shape = list(self.tensors[i].shape)
            expanded = shape[:axis] + [1] + shape[axis:]
            parts.append((self.held(self.reshape(self.value(i), shape, expanded,
                                                 f"{tag}_part{step}", i), i,
                                    f"{tag}_part{step}_q"), expanded, i))
        return self._concat(parts, axis, op.outputs[0], tag)

    def _unpack(self, op: _Operator, tag: str) -> str | None:
        ins, outs = op.inputs, op.outputs
        source = self.tensors[ins[0]]
        if len(outs) == 1:
            # The one part is the input without its axis.
            return self.reshape(self.value(ins[0]), list(source.shape), self.tensors[outs[0]].shape,
                                tag, outs[0])
        axis = _option(op, "axis") % len(source.shape)
        names = [f"{self.tensors[i].name}_float_part" for i in outs]
        sizes = self.b.constant(np.ones(len(outs), np.int64), tag + "_sizes")
        parts = self.b.multi_node("Split", [self.value(ins[0]), sizes], names,
                                  axis=_onnx_axis(axis, len(source.shape)))
        shape = list(source.shape)
        shape[axis] = 1
        for part, index in zip(parts, outs):
            name = self.tensors[index].name + "_float"
            held = self.held(part, index, name + "_part_q")
            self.finish(self.reshape(held, shape, self.tensors[index].shape, name, index), index)
        return None

    def _depth_space(self, op: _Operator, tag: str) -> str:
        """Reshape to six axes, swap the two in the middle, reshape back, all
        in TFLite's own axis order."""
        ins = op.inputs
        block = _option(op, "block_size")
        n, h, w, c = self.tensors[ins[0]].shape
        if op.kind == "SPACE_TO_DEPTH":
            split = [n, h // block, block, w // block, block, c]
        else:
            split = [n, h, w, block, block, c // (block * block)]
        y = self.last_to_last(self.value(ins[0]), 4, tag + "_nhwc", ins[0], True)
        y = self._six(y, split, [0, 1, 3, 2, 4, 5], self.tensors[op.outputs[0]].shape, tag, ins[0],
                      "_swap")
        return self.b.node("Transpose", [y], tag, perm=_to_first(4))

    def _broadcast_to(self, op: _Operator, tag: str) -> str:
        """Leading axes by reshape, then each broadcast axis as copies."""
        ins, b, out = op.inputs, self.b, self.tensors[op.outputs[0]]
        source = list(self.tensors[ins[0]].shape)
        shape = [1] * (len(out.shape) - len(source)) + source
        y = self.value(ins[0])
        if shape != source:
            y = self.held(self.reshape(y, source, shape, tag + "_rank", ins[0]), ins[0],
                          tag + "_rank_q")
        axes = [a for a, (d, t) in enumerate(zip(shape, out.shape)) if d != t]
        for step, axis in enumerate(axes):
            y = b.node("Concat", [y] * out.shape[axis], f"{tag}_axis{axis}",
                       axis=_onnx_axis(axis, len(shape)))
            if step < len(axes) - 1:
                y = self.held(y, ins[0], f"{tag}_axis{axis}_q")
        if not axes:
            y = b.node("Identity", [y], tag)
        return y

    def _batch_matmul(self, op: _Operator, tag: str) -> str:
        """The matrices are the last two TFLite axes, so the product runs in
        TFLite's own axis order."""
        b = self.b
        rank = len(self.tensors[op.outputs[0]].shape)
        operands = []
        for i, adjoint in zip(op.inputs, (_option(op, "adj_x"), _option(op, "adj_y"))):
            perm = list(range(rank))
            if adjoint:
                perm[-2:] = perm[-1], perm[-2]
            perm = [_to_last(rank)[p] for p in perm]
            x = self.value(i)
            if perm != list(range(rank)):
                x = self.held(b.node("Transpose", [x], tag + "_operand", perm=perm), i,
                              tag + "_operand_q")
            operands.append(x)
        y = b.node("MatMul", operands, tag)
        if rank in (3, 4):
            y = b.node("Transpose", [self.held(y, op.outputs[0], tag + "_q")], tag + "_first",
                       perm=_to_first(rank))
        return y

    def _resize(self, op: _Operator, tag: str) -> str:
        b = self.b
        sizes = b.constant(np.asarray(_onnx_shape(self.tensors[op.outputs[0]].shape), np.int64),
                           tag + "_sizes")
        align, half_pixel = _option(op, "align_corners"), _option(op, "half_pixel_centers")
        if op.kind == "RESIZE_BILINEAR":
            coordinate = "align_corners" if align else "half_pixel" if half_pixel else "asymmetric"
            attributes = {"mode": "linear", "coordinate_transformation_mode": coordinate}
        else:
            coordinate = "align_corners" if align else "tf_half_pixel_for_nn" if half_pixel else "asymmetric"
            attributes = {"mode": "nearest", "coordinate_transformation_mode": coordinate,
                          "nearest_mode": "round_prefer_ceil" if align else "floor"}
        return b.node("Resize", [self.value(op.inputs[0]), "", "", sizes], tag, **attributes)

    def _concat(self, parts, axis: int, index: int, tag: str) -> str:
        """Concatenation along TFLite axis `axis` of (value, TFLite shape,
        tensor index) parts. The runtime concatenates ranks 3 and 4 along any
        axis but the first; elsewhere every part becomes [1, outer, run], the
        outer axes before `axis` and the run from it, joined along the run
        and reshaped back, which places the elements alike."""
        b = self.b
        shape = list(self.tensors[index].shape)
        rank = len(shape)
        if rank in (3, 4) and axis > 0:
            return b.node("Concat", [value for value, _, _ in parts], tag,
                          axis=_onnx_axis(axis, rank))
        if axis == 0 and rank > 1:
            # Preserve the trailing run as an independent axis.
            width = int(np.prod(shape[1:]))
            flat = [self.held(self.reshape(value, part_shape, [1, part_shape[0], width],
                                           f"{tag}_run{step}", part), part, f"{tag}_run{step}_q")
                    for step, (value, part_shape, part) in enumerate(parts)]
            joined = self.held(b.node("Concat", flat, tag + "_runs", axis=_onnx_axis(1, 3)),
                               index, tag + "_runs_q")
            return self.reshape(joined, [1, shape[0], width], shape, tag, index)
        outer = int(np.prod(shape[:axis]))
        flat = []
        for step, (value, part_shape, part) in enumerate(parts):
            run = [1, outer, int(np.prod(part_shape[axis:]))]
            flat.append(self.held(self.reshape(value, part_shape, run, f"{tag}_run{step}", part),
                                  part, f"{tag}_run{step}_q"))
        joined = self.held(b.node("Concat", flat, tag + "_runs", axis=_onnx_axis(2, 3)), index,
                           tag + "_runs_q")
        total = [1, outer, int(np.prod(shape[axis:]))]
        return self.reshape(joined, total, shape, tag, index)

    def _pool_window(self, op: _Operator) -> dict:
        """The kernel_shape, strides and pads of a pooling operator's ONNX node."""
        win = window(op.kind, op.option)
        (top, bottom), (left, right) = win.pads(self.tensors[op.inputs[0]].shape[1:3],
                                                self.tensors[op.outputs[0]].shape[1:3])
        return {"kernel_shape": list(win.kernel), "strides": list(win.strides),
                "pads": [top, left, bottom, right]}

    def _pads_constant(self, op: _Operator, tag: str) -> str:
        """The paddings of a PAD, PADV2 or MIRROR_PAD as an ONNX Pad's pads input."""
        rank = len(self.tensors[op.outputs[0]].shape)
        pairs = np.asarray(self.ints(op.inputs[1])).reshape(rank, 2)
        order = _to_first(rank)
        return self.b.constant(np.asarray([pairs[a, 0] for a in order] + [pairs[a, 1] for a in order],
                                          np.int64), tag + "_pads")

    def _l2_normalization(self, op: _Operator, tag: str) -> str:
        """x / max(sqrt(sum(x * x)), 1e-6) over the last TFLite axis, as TFLite
        Micro computes it; the compiler folds it back into one operator."""
        b = self.b
        rank = len(self.tensors[op.inputs[0]].shape)
        x = self.value(op.inputs[0])
        axes = b.constant(np.asarray([_onnx_axis(-1, rank)], np.int64), tag + "_axes")
        total = b.node("ReduceSum", [b.node("Mul", [x, x], tag + "_square"), axes], tag + "_sum",
                       keepdims=1)
        norm = b.node("Max", [b.node("Sqrt", [total], tag + "_norm"),
                              b.constant(np.float32(1e-6), tag + "_epsilon")], tag + "_floor")
        return b.node("Div", [x, norm], tag)

    def _reduction(self, op: _Operator, tag: str) -> str:
        """Group mean axes without reordering a sum, then reduce the three
        axes [before, reduced, after] and restore the TFLite output shape."""
        b = self.b
        ins, source = op.inputs, self.tensors[op.inputs[0]]
        out = self.tensors[op.outputs[0]]
        shape = list(source.shape)
        axes = _axes(self.ints(ins[1]), len(shape))
        # Unit axes hold one element each, so reducing over them or not
        # gives the same values; a span they close needs no transpose.
        extent = [a for a in axes if shape[a] != 1] or axes[:1]
        span = list(range(extent[0], extent[-1] + 1))
        if all(a in axes or shape[a] == 1 for a in span):
            axes = span
        x = self.value(ins[0])
        if axes != list(range(axes[0], axes[-1] + 1)):
            # Keep both sets in their original order so each sum visits the
            # same elements in the same sequence as TFLite.
            kept_axes = [a for a in range(len(shape)) if a not in axes]
            perm = kept_axes + axes
            x = self.last_to_last(x, len(shape), tag + "_last", ins[0], True)
            x = self.held(b.node("Transpose", [x], tag + "_grouped", perm=perm),
                          ins[0], tag + "_grouped_q")
            grouped = [shape[a] for a in perm]
            if not _layout_free(grouped):
                x = self.held(b.node("Transpose", [x], tag + "_first",
                                    perm=_to_first(len(shape))), ins[0], tag + "_first_q")
            axes = list(range(len(kept_axes), len(shape)))
            shape = grouped
        first, last = axes[0], axes[-1] + 1
        rows = [int(np.prod(shape[:first])), int(np.prod(shape[first:last])),
                int(np.prod(shape[last:]))]
        if rows != shape:
            x = self.held(self.reshape(x, shape, rows, tag + "_rows", ins[0]), ins[0],
                          tag + "_rows_q")
        axis = _onnx_axis(1, 3)
        if op.kind == "CUMSUM":
            operands = [x, b.constant(np.asarray(axis, np.int64), tag + "_axis")]
            options = {"exclusive": int(_option(op, "exclusive")),
                       "reverse": int(_option(op, "reverse"))}
            if source.type == "INT8":
                # TFLite seeds the int8 sum with the input zero point, which
                # ONNX's CumSum over dequantized values does not; the
                # compiler's own CumSum states it.
                y = self.custom("CumSum", operands, tag + "_scan", _onnx_shape(rows), **options)
            else:
                y = b.node("CumSum", operands, tag + "_scan", **options)
            kept = rows
        elif op.kind == "SUM":
            y = b.node("ReduceSum", [x, b.constant(np.asarray([axis], np.int64), tag + "_axes")],
                       tag + "_reduced", keepdims=1)
            kept = [rows[0], 1, rows[2]]
        elif op.kind == "REDUCE_ALL":
            # ONNX has no bool reduction: the minimum of the values as uint8,
            # which the compiler folds back into one operator.
            widened = b.node("Cast", [x], tag + "_widened", to=TensorProto.UINT8)
            lowest = b.node("ReduceMin", [widened], tag + "_lowest", axes=[axis], keepdims=1)
            y = b.node("Cast", [lowest], tag + "_reduced", to=TensorProto.BOOL)
            kept = [rows[0], 1, rows[2]]
        else:
            kind = "ReduceMean" if op.kind == "MEAN" else _REDUCTIONS[op.kind]
            y = b.node(kind, [x], tag + "_reduced", axes=[axis], keepdims=1)
            kept = [rows[0], 1, rows[2]]
        if kept == list(out.shape):
            return b.node("Identity", [y], tag)
        return self.reshape(self.held(y, op.outputs[0], tag + "_kept"), kept, out.shape, tag,
                            op.outputs[0])

    def custom(self, kind: str, inputs: list[str], tag: str, shape, **attributes) -> str:
        """A node of the compiler's own domain, with the output shape stated."""
        y = self.b.unique(tag)
        self.b.nodes.append(helper.make_node(kind, inputs, [y], domain="tigris", **attributes))
        self.value_info.append(helper.make_tensor_value_info(y, TensorProto.FLOAT, list(shape)))
        return y

    def tflite_order_out(self, y: str, index: int, tag: str) -> str:
        """`y`, held in TFLite's axis order as tensor `index`, moved to ONNX's."""
        rank = len(self.tensors[index].shape)
        if rank not in (3, 4):
            return y
        return self.b.node("Transpose", [self.held(y, index, tag + "_q")], tag,
                           perm=_to_first(rank))

    def _gather_node(self, kind: str, data: int, ids: int, axis, op: _Operator, tag: str) -> str:
        """GATHER, GATHER_ND or EMBEDDING_LOOKUP with constant indices, run in
        TFLite's axis order, where its axis and index tuples are stated."""
        rank = len(self.tensors[data].shape)
        if _is_constant(self.tensors[data]):
            tensor = self.tensors[data]
            x = (self.b.constant(tensor.array(), tensor.name) if tensor.type == "FLOAT32" else
                 self.b.dequantized_constant(tensor.array(), tensor.scale, tensor.zero_point,
                                             tensor.name, axis=tensor.quantized_dimension))
        else:
            x = self.last_to_last(self.value(data), rank, tag + "_last", data, True)
        indices = (self.b.constant(np.asarray(self.tensors[ids].array(), np.int64), tag + "_indices")
                   if _is_constant(self.tensors[ids]) else self.last_to_last(
                       self.value(ids), len(self.tensors[ids].shape), tag + "_index_order", ids, True))
        attributes = {} if axis is None else {"axis": axis % rank}
        batch = _option(op, "batch_dims") if op.kind == "GATHER" else 0
        if batch or (kind == "GatherND" and not _is_constant(self.tensors[ids])):
            # ONNX GatherND takes int64 indices; this form keeps runtime int32.
            y = self.custom(kind, [x, indices], tag + "_gathered",
                            self.tensors[op.outputs[0]].shape, batch_dims=batch, **attributes)
        else:
            y = self.b.node(kind, [x, indices], tag + "_gathered", **attributes)
        return self.tflite_order_out(y, op.outputs[0], tag)

    def _svdf(self, op: _Operator, tag: str) -> str:
        """SVDF in the compiler's own form, its state passed in and out. An int8
        SVDF takes its constants as stored, with their scales stated; its int16
        state stays raw. TFLite Micro applies no activation to an int8 SVDF."""
        b = self.b
        x, feature, time, bias, state = op.inputs
        quantized = self.tensors[x].type == "INT8"
        activation = "none" if quantized else _activation(_option(op, "fused_activation_function"))
        attributes = {"rank": _option(op, "rank"), "activation": activation}
        if quantized:
            attributes.update(feature_scale=float(self.tensors[feature].scale[0]),
                              time_scale=float(self.tensors[time].scale[0]),
                              state_scale=float(self.tensors[state].scale[0]),
                              state_zero_point=int(self.tensors[state].zero_point[0]))
        constants = [b.constant(self.tensors[i].array(), f"{tag}_{part}")
                     for i, part in ((feature, "feature"), (time, "time"), (bias, "bias"))]
        y, kept = b.unique(tag), b.unique(tag + "_state")
        b.nodes.append(helper.make_node("Svdf", [self.value(x), *constants,
                                                 self.held_state[state]["input"]],
                                        [y, kept], domain="tigris", **attributes))
        held = self.tensors[state]
        self.value_info += [
            helper.make_tensor_value_info(y, TensorProto.FLOAT,
                                          list(self.tensors[op.outputs[0]].shape)),
            helper.make_tensor_value_info(kept, _ONNX_TYPES[held.type], list(held.shape))]
        self.held_state[state]["output"] = kept
        return y

    def _detection(self, op: _Operator, tag: str) -> None:
        """TFLite_Detection_PostProcess in the compiler's own form; its four
        float32 outputs are the model's detections."""
        b = self.b
        options = _detection_options(op)
        boxes, scores, anchors = op.inputs
        names = [b.unique(f"{tag}_{part}") for part in ("boxes", "classes", "scores", "count")]
        b.nodes.append(helper.make_node(
            "DetectionPostProcess",
            [self.value(boxes), self.value(scores),
             b.constant(self.tensors[anchors].array(), tag + "_anchors")],
            names, domain="tigris",
            max_detections=options["max_detections"],
            detections_per_class=options["detections_per_class"],
            use_regular_nms=int(options["use_regular_nms"]),
            score_threshold=options["score_threshold"], iou_threshold=options["iou_threshold"],
            num_classes=options["num_classes"], scales=options["scales"]))
        for name, index in zip(names, op.outputs):
            self.value_info.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, _onnx_shape(self.tensors[index].shape)))
            self.finish(name, index)

    def _control_flow(self, op: _Operator, tag: str) -> None:
        """IF or WHILE in the compiler's own form, each subgraph an ONNX graph
        whose inputs are the operands it receives: an IF's operands after the
        condition, a WHILE's loop variables."""
        b = self.b
        kind, roles = (("If", ("then_branch", "else_branch")) if op.kind == "IF"
                       else ("While", ("cond_branch", "body_branch")))
        graphs = {role: _branch_graph(self.graphs, position, f"{tag}_{role}")
                  for position, role in zip(_subgraphs(op), roles)}
        results = [b.unique(f"{tag}_{n}") for n in range(len(op.outputs))]
        # A constant operand is read inside the branch itself.
        operands = [i for n, i in enumerate(op.inputs)
                    if op.kind != "IF" or n == 0 or not _is_constant(self.tensors[i])]
        b.nodes.append(helper.make_node(
            kind, [self.value(i) for i in operands], results, domain="tigris", **graphs))
        for index, name in zip(op.outputs, results):
            self.value_info.append(helper.make_tensor_value_info(
                name, _BOUNDARY_TYPES[self.tensors[index].type], _onnx_shape(self.tensors[index].shape)))
            self.finish(name, index)

    def _lstm(self, op: _Operator, tag: str) -> str:
        """UNIDIRECTIONAL_SEQUENCE_LSTM in the compiler's own form, run in
        TFLite's axis order, its hidden and cell states passed in and out."""
        b = self.b
        ins = op.inputs
        x = self.last_to_last(self.value(ins[0]), 3, tag + "_last", ins[0], True)
        constants = [b.constant(self.tensors[ins[i]].array(), f"{tag}_w{i}") for i in _LSTM_CONSTANTS]
        attributes = {"time_major": int(_option(op, "time_major")),
                      "cell_clip": float(_option(op, "cell_clip"))}
        if self.tensors[ins[0]].type == "INT8":
            # Integer weights pass as stored, their scales stated; the int16
            # cell state stays raw.
            attributes.update(weight_scales=[float(self.tensors[ins[i]].scale[0]) for i in range(1, 9)],
                              cell_scale=float(self.tensors[ins[19]].scale[0]))
        y = b.unique(tag + "_sequence")
        kept = [b.unique(tag + "_hidden"), b.unique(tag + "_cell")]
        b.nodes.append(helper.make_node(
            "Lstm", [x, *constants, self.value(ins[18]), self.held_state[ins[19]]["input"]],
            [y, *kept], domain="tigris", **attributes))
        out = self.tensors[op.outputs[0]]
        self.value_info.append(helper.make_tensor_value_info(y, TensorProto.FLOAT, list(out.shape)))
        for index, name in zip(ins[18:20], kept):
            held = self.tensors[index]
            if held.type == "INT8":
                self.value_info.append(helper.make_tensor_value_info(
                    name, TensorProto.FLOAT, list(held.shape)))
                scale, point = self.held_state[index]["quantization"]
                name = b.node("QuantizeLinear", [name, scale, point], name + "_q")
            else:
                self.value_info.append(helper.make_tensor_value_info(
                    name, _ONNX_TYPES[held.type], list(held.shape)))
            self.held_state[index]["output"] = name
        return self.tflite_order_out(self.held(y, op.outputs[0], tag + "_q"), op.outputs[0], tag)

    def _slice_steps(self, op: _Operator, tag: str) -> str:
        """A STRIDED_SLICE with any strides, as an ONNX Slice in TFLite's axis
        order; a dropped axis is a slice of one, reshaped away."""
        b = self.b
        ins, source = op.inputs, self.tensors[op.inputs[0]]
        rank = len(source.shape)
        axes = _strided_slice_axes(_StridedSliceOptions.of(op), source.shape,
                                   *(self.ints(i) for i in ins[1:4]))
        starts, stops, steps, sliced = [], [], [], []
        for start, stop, step, _ in axes:
            sliced.append(len(range(start, stop, step)))
            starts.append(start)
            # Below zero, a reverse slice runs past the first element.
            stops.append(stop if stop >= 0 else -(2**63))
            steps.append(step)
        x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
        y = b.node("Slice", [x, b.constant(np.asarray(starts, np.int64), tag + "_starts"),
                             b.constant(np.asarray(stops, np.int64), tag + "_ends"),
                             b.constant(np.arange(rank, dtype=np.int64), tag + "_axes"),
                             b.constant(np.asarray(steps, np.int64), tag + "_steps")],
                   tag + "_sliced")
        out = self.tensors[op.outputs[0]]
        if list(out.shape) != sliced:
            y = b.node("Reshape", [self.held(y, op.outputs[0], tag + "_sliced_q"),
                                   b.constant(np.asarray(out.shape, np.int64), tag + "_shape")],
                       tag + "_dropped")
        return self.tflite_order_out(y, op.outputs[0], tag)

    def _six(self, x: str, split, perm, joined, tag: str, index: int,
             middle: str = "_perm") -> str:
        """Reshape to six axes, permute, reshape back, in TFLite's axis order,
        each step held as tensor `index` is quantized; `middle` suffixes the
        permutation's node names."""
        b = self.b
        y = self.held(b.node("Reshape", [x, b.constant(np.asarray(split, np.int64),
                                                       tag + "_split")], tag + "_six"),
                      index, tag + "_six_q")
        y = self.held(b.node("Transpose", [y], tag + middle, perm=list(perm)), index,
                      tag + middle + "_q")
        return self.held(b.node("Reshape", [y, b.constant(np.asarray(joined, np.int64),
                                                          tag + "_joined")], tag + "_four"),
                         index, tag + "_four_q")

    def _batch_space(self, op: _Operator, tag: str) -> str:
        """SPACE_TO_BATCH_ND pads then folds blocks into the batch;
        BATCH_TO_SPACE_ND unfolds them and crops."""
        b = self.b
        ins, source = op.inputs, self.tensors[op.inputs[0]]
        out = self.tensors[op.outputs[0]]
        bh, bw = self.ints(ins[1])
        edges = self.ints(ins[2])
        x = self.value(ins[0])
        if op.kind == "SPACE_TO_BATCH_ND":
            top, bottom, left, right = edges
            if any(edges):
                pads = b.constant(np.asarray([0, 0, top, left, 0, 0, bottom, right], np.int64),
                                  tag + "_pads")
                x = self.held(b.node("Pad", [x, pads, b.constant(np.float32(0.0), tag + "_fill")],
                                     tag + "_padded", mode="constant"), ins[0], tag + "_padded_q")
            n, h, w, c = source.shape
            h, w = h + top + bottom, w + left + right
            x = self.last_to_last(x, 4, tag + "_nhwc", ins[0], True)
            x = self._six(x, [n, h // bh, bh, w // bw, bw, c], [2, 4, 0, 1, 3, 5], out.shape,
                          tag, ins[0])
            return b.node("Transpose", [x], tag, perm=_to_first(4))
        n, h, w, c = source.shape
        batch = n // (bh * bw)
        x = self.last_to_last(x, 4, tag + "_nhwc", ins[0], True)
        whole = [batch, h * bh, w * bw, c]
        x = self._six(x, [bh, bw, batch, h, w, c], [2, 3, 0, 4, 1, 5], whole, tag, ins[0])
        x = b.node("Transpose", [x], tag + "_first", perm=_to_first(4))
        top, bottom, left, right = edges
        shape = list(whole)
        for axis, (begin, end) in ((1, (top, bottom)), (2, (left, right))):
            if begin or end:
                x = self.held(x, ins[0], f"{tag}_crop{axis}_q")
                x = self.slice_axis(x, shape, axis, begin, shape[axis] - begin - end,
                                    f"{tag}_crop{axis}", ins[0], hold=False)
                shape[axis] -= begin + end
        return x

    def _floor_mod(self, a: str, divisor: str, tag: str) -> str:
        """TFLite's FLOOR_MOD: the truncated remainder, plus the divisor where
        the remainder is non-zero and its sign differs from the divisor's."""
        b = self.b
        zero = b.constant(np.float32(0.0), tag + "_zero")
        remainder = b.node("Mod", [a, divisor], tag + "_remainder", fmod=1)
        nonzero = b.node("Not", [b.node("Equal", [remainder, zero], tag + "_is_zero")],
                         tag + "_nonzero")
        signs = b.node("Xor", [b.node("Less", [divisor, zero], tag + "_divisor_negative"),
                               b.node("Less", [remainder, zero], tag + "_remainder_negative")],
                       tag + "_signs_differ")
        condition = b.node("And", [nonzero, signs], tag + "_adjust")
        shifted = b.node("Add", [remainder, divisor], tag + "_shifted")
        return b.node("Where", [condition, shifted, remainder], tag)

    def weights(self, tensor: _Tensor, array: np.ndarray, axis: int,
                zero_dtype=np.int8) -> str:
        """A weight or bias: float32 as it is, int8 or int32 behind a
        DequantizeLinear with symmetric zero points."""
        if tensor.type == "FLOAT32":
            return self.b.constant(array.astype(np.float32), tensor.name)
        return self.b.dequantized_constant(array, tensor.scale, np.zeros_like(tensor.zero_point),
                                           tensor.name, axis=axis, zero_dtype=zero_dtype)

    def _bias(self, op: _Operator, position: int, name: str) -> str | None:
        if position >= len(op.inputs) or op.inputs[position] < 0:
            return None
        tensor = self.tensors[op.inputs[position]]
        return self.weights(tensor, tensor.array(), 0, np.int32)

    def _conv(self, op: _Operator, tag: str) -> str:
        depthwise = op.kind == "DEPTHWISE_CONV_2D"
        source, weights_tensor = self.tensors[op.inputs[0]], self.tensors[op.inputs[1]]
        out = self.tensors[op.outputs[0]]
        weights = np.transpose(weights_tensor.array(), (3, 0, 1, 2) if depthwise else (0, 3, 1, 2))
        win = window(op.kind, op.option, weights_tensor.shape)
        (top, bottom), (left, right) = win.pads(source.shape[1:3], out.shape[1:3])
        w = self.weights(weights_tensor, weights, 0)
        operands = [self.value(op.inputs[0]), w]
        bias = self._bias(op, 2, tag)
        if bias:
            operands.append(bias)
        return self.b.node("Conv", operands, tag, kernel_shape=list(win.kernel),
                           strides=list(win.strides), dilations=list(win.dilations),
                           pads=[top, left, bottom, right],
                           group=source.shape[3] if depthwise else 1)

    def _transpose_conv(self, op: _Operator, tag: str) -> str:
        source, weights_tensor = self.tensors[op.inputs[2]], self.tensors[op.inputs[1]]
        out = self.tensors[op.outputs[0]]
        weights = np.transpose(weights_tensor.array(), (3, 0, 1, 2))
        win = window(op.kind, op.option, weights_tensor.shape)
        (top, bottom), (left, right) = win.pads(out.shape[1:3], source.shape[1:3])
        w = self.weights(weights_tensor, weights, 1)
        operands = [self.value(op.inputs[2]), w]
        bias = self._bias(op, 3, tag)
        if bias:
            operands.append(bias)
        return self.b.node("ConvTranspose", operands, tag, kernel_shape=list(win.kernel),
                           strides=list(win.strides), pads=[top, left, bottom, right])

    def _fully_connected(self, op: _Operator, tag: str) -> str:
        source, weights_tensor = self.tensors[op.inputs[0]], self.tensors[op.inputs[1]]
        out = self.tensors[op.outputs[0]]
        w = self.weights(weights_tensor, weights_tensor.array(), 0)
        bias = self._bias(op, 2, tag) or self.b.constant(
            np.zeros(weights_tensor.shape[0], np.float32), out.name + "_bias")
        x = self.value(op.inputs[0])
        rows = int(np.prod(source.shape)) // weights_tensor.shape[1]
        flat = [rows, weights_tensor.shape[1]]
        if list(source.shape) != flat:
            x = self.held(self.reshape(x, source.shape, flat, tag + "_rows", op.inputs[0]),
                          op.inputs[0], tag + "_rows_q")
        y = self.b.node("Gemm", [x, w, bias], tag, transB=1)
        if list(out.shape) != [rows, weights_tensor.shape[0]]:
            # Held again so a fused activation after the reshape reads int8.
            y = self.held(self.reshape(self.held(y, op.outputs[0], tag + "_q"),
                                       [rows, weights_tensor.shape[0]], out.shape,
                                       tag + "_shape", op.outputs[0]),
                          op.outputs[0], tag + "_shape_q")
        return y


class _Kind(NamedTuple):
    """How the importer checks and converts one operator kind.

    convert(converter, op, tag): the ONNX value of the output, or None when it
    stored its results itself. options: the fields read from the options
    table, by name, as (slot, format, value when absent). early: checks after
    the constant operands; the first to return a string gives the result, ""
    accepting the operator. dtypes: the dtype check of the data operands and
    outputs. fused: fused_activation_function is checked, then applied after
    the operator. late: checks after it; the first non-empty reason refuses
    the operator."""
    convert: Callable
    options: dict = {}
    early: tuple = ()
    late: tuple = ()
    dtypes: Callable = _float_or_int8
    fused: bool = False


def _unary(onnx_type: str) -> Callable:
    return lambda c, op, tag: c.b.node(onnx_type, [c.value(op.inputs[0])], tag)


def _broadcast(onnx_type: str) -> Callable:
    return lambda c, op, tag: c.b.node(onnx_type, c.operands(op), tag)


def _fused(slot: int) -> dict:
    return {"fused_activation_function": (slot, "b", 0)}


_MOVEMENT = (_data_movement_reason,)
_SHAPED_MOVEMENT = (_data_movement_reason, _option_shape_reason)
_KINDS: dict[str, _Kind] = {
    **{kind: _Kind(_unary(name)) for kind, name in _UNARY.items()},
    # dtypes=_float_only: elementwise math TFLite Micro runs in float32 only.
    **{kind: _Kind(_unary(name), dtypes=_float_or_int8 if kind == "ABS" else _float_only)
       for kind, name in _ELEMENTWISE_UNARY.items()},
    "RSQRT": _Kind(lambda c, op, tag: c.b.node(
        "Reciprocal", [c.b.node("Sqrt", [c.value(op.inputs[0])], tag + "_sqrt")], tag)),
    "SQUARE": _Kind(lambda c, op, tag: c.b.node("Mul", [c.value(op.inputs[0])] * 2, tag),
                    dtypes=_float_only),
    "RELU6": _Kind(lambda c, op, tag: c.b.fused_activation(c.value(op.inputs[0]), "relu6", tag)),
    "LEAKY_RELU": _Kind(lambda c, op, tag: c.b.node("LeakyRelu", [c.value(op.inputs[0])], tag,
                                                    alpha=_option(op, "alpha")),
                        {"alpha": (0, "f", 0.0)}),
    "PRELU": _Kind(_broadcast("PRelu")),
    "GELU": _Kind(lambda c, op, tag: c.custom(
        "Gelu", [c.value(op.inputs[0])], tag, _onnx_shape(c.tensors[op.outputs[0]].shape),
        approximate="tanh" if _option(op, "approximate") else "none"),
        {"approximate": (0, "?", False)}),
    "SOFTMAX": _Kind(_Converter._softmax, {"beta": (0, "f", 0)}, late=(_beta_reason,)),
    "LOG_SOFTMAX": _Kind(_Converter._softmax),
    "L2_NORMALIZATION": _Kind(_Converter._l2_normalization, _fused(0), late=(_l2_fused_reason,)),
    **{kind: _Kind(_broadcast(name), _fused(0), early=(_int32_arithmetic,), fused=True)
       for kind, name in (("ADD", "Add"), ("SUB", "Sub"), ("MUL", "Mul"))},
    "DIV": _Kind(_broadcast("Div"), _fused(0), late=(_constant_only_reason,), fused=True),
    **{kind: _Kind(_broadcast(name), late=(_constant_only_reason,))
       for kind, name in (("MAXIMUM", "Max"), ("MINIMUM", "Min"))},
    "SQUARED_DIFFERENCE": _Kind(lambda c, op, tag: c.b.node(
        "Mul", [c.b.node("Sub", c.operands(op), tag + "_difference")] * 2, tag),
        late=(_constant_only_reason,)),
    "FLOOR_DIV": _Kind(lambda c, op, tag: c.b.node(
        "Floor", [c.b.node("Div", c.operands(op), tag + "_quotient")], tag),
        late=(_constant_only_reason,), dtypes=_float_only),
    "FLOOR_MOD": _Kind(lambda c, op, tag: c._floor_mod(*c.operands(op), tag),
                       late=(_constant_only_reason,), dtypes=_float_only),
    "ADD_N": _Kind(lambda c, op, tag: c.b.node("Sum", [c.value(i) for i in op.inputs], tag)),
    **{kind: _Kind(_Converter._comparison, early=(_int32_comparison, _bool_reason))
       for kind in _COMPARISONS},
    **{kind: _Kind(_broadcast(name), early=(_bool_reason,)) for kind, name in _LOGICAL.items()},
    "SELECT_V2": _Kind(_broadcast("Where"), early=(_bool_reason,)),
    "CAST": _Kind(_Converter._cast, early=(_int32_cast, _bool_reason, _cast_encoding_reason)),
    **{kind: _Kind(_Converter._index, early=(_index_reason,)) for kind in _INDEX},
    **{kind: _Kind(_Converter._reduction, late=(_reduction_axes_reason,))
       for kind in ("REDUCE_MAX", "REDUCE_MIN", "SUM")},
    "REDUCE_ALL": _Kind(_Converter._reduction, early=(_bool_reason,),
                        late=(_reduction_axes_reason,)),
    "MEAN": _Kind(_Converter._mean, late=(_reduction_axes_reason,)),
    "CUMSUM": _Kind(_Converter._reduction,
                    {"exclusive": (0, "?", False), "reverse": (1, "?", False)}),
    # Weighted and windowed operators.
    "CONV_2D": _Kind(_Converter._conv, _fused(3), early=(_weights_reason,),
                     late=(_window_reason,), fused=True),
    "DEPTHWISE_CONV_2D": _Kind(_Converter._conv, {"depth_multiplier": (3, "i", 0), **_fused(4)},
                               early=(_weights_reason,), late=(_window_reason,), fused=True),
    "TRANSPOSE_CONV": _Kind(_Converter._transpose_conv, _fused(3), early=(_weights_reason,),
                            late=(_window_reason,), fused=True),
    "FULLY_CONNECTED": _Kind(_Converter._fully_connected,
                             {**_fused(0), "weights_format": (1, "b", 0), "quant_spec": (5, "[B", b"")},
                             early=(_quant_spec_reason, _weights_reason),
                             late=(_weights_format_reason,), fused=True),
    **{kind: _Kind(_Converter._pool, _fused(5), late=(_window_reason,), fused=True)
       for kind in ("MAX_POOL_2D", "AVERAGE_POOL_2D")},
    "L2_POOL_2D": _Kind(_Converter._l2_pool, _fused(5), late=(_window_reason,), dtypes=_float_only,
                        fused=True),
    "BATCH_MATMUL": _Kind(_Converter._batch_matmul,
                          {"adj_x": (0, "?", False), "adj_y": (1, "?", False)},
                          late=(_rank_reason, _option_shape_reason)),
    # Shapes and data movement.
    **{kind: _Kind(_Converter._shape_only, early=(_kept_constant_reason,))
       for kind in ("SQUEEZE", "EXPAND_DIMS")},
    "RESHAPE": _Kind(_Converter._shape_only, early=(_kept_constant_reason, _int32_movement)),
    "TRANSPOSE": _Kind(_Converter._transpose, early=(_int32_movement,)),
    "CONCATENATION": _Kind(_Converter._concatenation, {"axis": (0, "i", 0), **_fused(1)},
                           early=(_kept_constant_reason,),
                           late=(_concat_quantization_reason, _option_shape_reason), fused=True),
    "PACK": _Kind(_Converter._pack, {"values_count": (0, "i", 0), "axis": (1, "i", 0)},
                  early=(_count_reason, _kept_constant_reason), late=_SHAPED_MOVEMENT),
    "UNPACK": _Kind(_Converter._unpack, {"num": (0, "i", 0), "axis": (1, "i", 0)},
                    early=(_count_reason,), late=_SHAPED_MOVEMENT),
    "SPLIT": _Kind(_Converter._split),
    "SPLIT_V": _Kind(_Converter._split),
    "SLICE": _Kind(_Converter._slice, early=(_kept_constant_reason,), late=_MOVEMENT),
    "STRIDED_SLICE": _Kind(_Converter._strided_slice,
                           {"begin_mask": (0, "i", 0), "end_mask": (1, "i", 0),
                            "ellipsis_mask": (2, "i", 0), "new_axis_mask": (3, "i", 0),
                            "shrink_axis_mask": (4, "i", 0), "offset": (5, "?", False)},
                           early=(_kept_constant_reason,), late=_SHAPED_MOVEMENT),
    "GATHER": _Kind(_Converter._gather, {"axis": (0, "i", 0), "batch_dims": (1, "i", 0)},
                    late=_SHAPED_MOVEMENT),
    "GATHER_ND": _Kind(lambda c, op, tag: c._gather_node("GatherND", op.inputs[0], op.inputs[1],
                                                         None, op, tag), late=_MOVEMENT),
    "EMBEDDING_LOOKUP": _Kind(lambda c, op, tag: c._gather_node("Gather", op.inputs[1],
                                                                op.inputs[0], 0, op, tag),
                              late=_MOVEMENT),
    "DYNAMIC_UPDATE_SLICE": _Kind(_Converter._dynamic_update_slice, late=_MOVEMENT),
    **{kind: _Kind(_Converter._depth_space, {"block_size": (0, "i", 0)}, late=_MOVEMENT)
       for kind in ("SPACE_TO_DEPTH", "DEPTH_TO_SPACE")},
    **{kind: _Kind(_Converter._batch_space, late=_MOVEMENT)
       for kind in ("SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND")},
    "BROADCAST_TO": _Kind(_Converter._broadcast_to, late=_MOVEMENT),
    "REVERSE_V2": _Kind(_Converter._reverse, late=_MOVEMENT),
    "PAD": _Kind(_Converter._pad),
    "PADV2": _Kind(_Converter._pad),
    "MIRROR_PAD": _Kind(_Converter._mirror_pad, {"mode": (0, "b", 0)}, late=_MOVEMENT),
    "RESIZE_BILINEAR": _Kind(_Converter._resize, {"align_corners": (2, "?", False),
                                                  "half_pixel_centers": (3, "?", False)},
                             late=(_resize_reason,)),
    "RESIZE_NEAREST_NEIGHBOR": _Kind(_Converter._resize, {"align_corners": (0, "?", False),
                                                          "half_pixel_centers": (1, "?", False)},
                                     late=(_resize_reason,)),
    "QUANTIZE": _Kind(_Converter._quantization, dtypes=_quantize_dtypes),
    "DEQUANTIZE": _Kind(_Converter._quantization, dtypes=_dequantize_dtypes),
    # State, control flow and detection.
    "SVDF": _Kind(_Converter._svdf, {"rank": (0, "i", 0), **_fused(1)}, early=(_svdf_reason,)),
    "UNIDIRECTIONAL_SEQUENCE_LSTM": _Kind(
        _Converter._lstm, {**_fused(0), "cell_clip": (1, "f", 0.0), "time_major": (3, "?", False)},
        early=(_lstm_reason,)),
    "CALL_ONCE": _Kind(_Converter._variable, {"init_subgraph_index": (0, "i", 0)},
                       early=(_variables_reason,)),
    "VAR_HANDLE": _Kind(_Converter._variable, {"container": (0, "s", ""), "shared_name": (1, "s", "")},
                        early=(_variables_reason,)),
    **{kind: _Kind(_Converter._variable, early=(_variables_reason,))
       for kind in ("READ_VARIABLE", "ASSIGN_VARIABLE")},
    "IF": _Kind(_Converter._control_flow, {"then_subgraph_index": (0, "i", 0),
                                           "else_subgraph_index": (1, "i", 0)}, early=(_if_reason,)),
    "WHILE": _Kind(_Converter._control_flow, {"cond_subgraph_index": (0, "i", 0),
                                              "body_subgraph_index": (1, "i", 0)},
                   early=(_while_reason,)),
    _DETECTION: _Kind(_Converter._detection, early=(_detection_reason,)),
}


def to_onnx(data: bytes, name: str) -> onnx.ModelProto:
    """The model as QDQ ONNX with its TFLite tensor names; raises ValueError
    naming every construct that does not convert."""
    _, graphs = _read(data)
    original_types = {i: graphs[0][1][i].type for i in graphs[0][2]}
    graphs = [_narrow_index_inputs(g) for g in graphs]
    reasons = _reasons(graphs)
    if reasons:
        raise ValueError("TFLite model does not convert: " + "; ".join(reasons))
    _, tensors, inputs, outputs, operators = graphs[0]
    consumed = {i for op in operators for i in op.inputs}
    converter = _Converter(tensors, outputs, consumed)
    converter.graphs = graphs
    converter.float_boundary = _control_flow_tensors(operators)
    # Dequantized tensors only a detection operator reads.
    readers = {}
    for op in operators:
        for i in op.inputs:
            readers.setdefault(i, set()).add(op.kind)
    converter.detection_operands = {
        op.outputs[0] for op in operators
        if op.kind == "DEQUANTIZE" and readers.get(op.outputs[0]) == {_DETECTION}
        and op.outputs[0] not in outputs}
    b = converter.b
    indices = {i for op in operators if op.kind in _INDEX for i in op.outputs}

    # The interface keeps TFLite's dtypes: an int8 input is dequantized straight
    # off the graph input.
    onnx_inputs = []
    for index in inputs:
        tensor = tensors[index]
        b._names.add(tensor.name)
        onnx_inputs.append(helper.make_tensor_value_info(
            tensor.name, _ONNX_TYPES[original_types[index]], _onnx_shape(tensor.shape)))
        if original_types[index] == "INT64" and tensor.type == "INT32":
            converter.values[index] = b.node("Cast", [tensor.name], tensor.name + "_i32", to=TensorProto.INT32)
        elif tensor.type in ("FLOAT32", "BOOL", "INT32"):
            converter.values[index] = tensor.name
        else:
            scale = b.constant(np.float32(tensor.scale[0]), tensor.name + "_scale")
            zero = b.constant(np.array(int(tensor.zero_point[0]), _NUMPY[tensor.type]),
                              tensor.name + "_zero_point")
            converter.values[index] = b.node("DequantizeLinear", [tensor.name, scale, zero],
                                             tensor.name + "_float")
    # Each variable enters as a state input holding its value from the last
    # invocation, initially what the CALL_ONCE subgraph assigns.
    initials = _variable_initials(graphs, _initializer_subgraphs(graphs)) or {}
    handles = {op.outputs[0]: _var_name(op) for op in operators if op.kind == "VAR_HANDLE"}
    shapes = {}
    for op in operators:
        if op.kind == "READ_VARIABLE":
            shapes.setdefault(handles[op.inputs[0]], list(tensors[op.outputs[0]].shape))
        elif op.kind == "ASSIGN_VARIABLE":
            shapes.setdefault(handles[op.inputs[0]], list(tensors[op.inputs[1]].shape))
    state = []
    for position, (variable, shape) in enumerate(shapes.items()):
        name = f"state{position}_{variable.strip('/')}"
        initial = initials.get(variable, np.zeros(shape, np.float32)).reshape(shape)
        onnx_inputs.append(helper.make_tensor_value_info(name + "_in", TensorProto.FLOAT,
                                                         _onnx_shape(shape)))
        b._names.add(name + "_in")
        converter.variables[variable] = {"current": name + "_in", "name": name, "shape": shape}
        state.append({"input": name + "_in", "output": None,
                      "initial": b.constant(initial.astype(np.float32).transpose(
                          _to_first(len(shape))), name + "_initial")})
    # A variable tensor enters as a state input, zero before the first run as
    # TFLite Micro resets it; the operator's write leaves as the state output.
    for index in [op.inputs[slot] for op in operators if op.kind in _STATEFUL
                  for slot in _STATEFUL[op.kind]]:
        tensor = tensors[index]
        name = b.unique(f"state{len(state)}_{tensor.name}_in")
        onnx_inputs.append(helper.make_tensor_value_info(
            name, _ONNX_TYPES[tensor.type], list(tensor.shape)))
        # An int8 variable starts at its zero point and is read through its
        # quantization; an int16 one is passed to its operator as stored.
        zero = int(tensor.zero_point[0]) if tensor.type == "INT8" else 0
        initial = b.constant(np.full(tensor.shape, zero, _NUMPY[tensor.type]),
                             name.removesuffix("_in") + "_initial")
        converter.held_state[index] = {"input": name, "output": None}
        if tensor.type == "INT8":
            scale = b.constant(np.float32(tensor.scale[0]), name + "_scale")
            point = b.constant(np.array(zero, np.int8), name + "_zero_point")
            converter.values[index] = b.node("DequantizeLinear", [name, scale, point],
                                             name + "_float")
            converter.held_state[index]["quantization"] = (scale, point)
        else:
            converter.values[index] = name
        state.append({"input": name, "output": None, "initial": initial})
    for op in operators:
        converter.convert(op)
    for entry, held in zip(state[len(state) - len(converter.held_state):],
                           converter.held_state.values()):
        entry["output"] = held["output"]
    state_outputs = []
    for index, held in converter.held_state.items():
        tensor = tensors[index]
        state_outputs.append(helper.make_tensor_value_info(
            held["output"], _ONNX_TYPES[tensor.type], list(tensor.shape)))
    for entry, variable in zip(state, shapes):
        current = converter.variables[variable]
        if current["current"] != entry["input"]:
            entry["output"] = b.node("Identity", [current["current"]], current["name"] + "_out")
            state_outputs.append(helper.make_tensor_value_info(
                entry["output"], TensorProto.FLOAT, _onnx_shape(current["shape"])))
    # An index is stored as the operator writes it, in TFLite's axis order.
    onnx_outputs = [helper.make_tensor_value_info(
        tensors[index].name if index in converter.int8_outputs else converter.values[index],
        _ONNX_TYPES[tensors[index].type],
        list(tensors[index].shape) if index in indices else _onnx_shape(tensors[index].shape))
        for index in outputs] + state_outputs
    opsets = [helper.make_opsetid("", 17)]
    if converter.value_info:
        opsets.append(helper.make_opsetid("tigris", 1))
    model = helper.make_model(helper.make_graph(b.nodes, name, onnx_inputs, onnx_outputs,
                                                b.initializers, value_info=converter.value_info),
                              opset_imports=opsets)
    model.ir_version = 8
    props = {BOUNDARY_LAYOUT_KEY: "channels_last"}
    if state:
        props[STATE_KEY] = json.dumps(state)
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    return model
