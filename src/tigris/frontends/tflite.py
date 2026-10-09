"""TFLite FlatBuffer models: inspection and conversion to QDQ ONNX.

Inspection reads any TFLite file. Conversion covers the int8 operators whose
TFLite semantics a QDQ ONNX graph states exactly and refuses everything else
by name, so a model is never compiled with a reinterpreted operator.
"""

import json

import numpy as np
import onnx
from onnx import TensorProto, helper

from tigris.frontends.flatbuffer import Table, flexbuffer_map
from tigris.frontends.qdq import GraphBuilder, same_padding

MAGIC = b"TFL3"
# Metadata stating that every model input and output of rank 3 or more is held
# channels-last, as the TFLite tensor is laid out.
BOUNDARY_LAYOUT_KEY = "tigris.boundary_layout"
# Variables a model keeps across invocations: a JSON list of
# {"input", "output", "initial"} value names, one entry per variable.
STATE_KEY = "tigris.state"

# Field slots in the TFLite schema (tensorflow/lite/schema/schema.fbs).
_MODEL_VERSION, _MODEL_OPCODES, _MODEL_SUBGRAPHS, _MODEL_DESCRIPTION, _MODEL_BUFFERS = 0, 1, 2, 3, 4
_SG_TENSORS, _SG_INPUTS, _SG_OUTPUTS, _SG_OPERATORS, _SG_NAME = 0, 1, 2, 3, 4
_T_SHAPE, _T_TYPE, _T_BUFFER, _T_NAME, _T_QUANT, _T_VARIABLE = 0, 1, 2, 3, 4, 5
_Q_SCALE, _Q_ZERO_POINT, _Q_DIMENSION = 2, 3, 6
_B_DATA, _B_OFFSET, _B_SIZE = 0, 1, 2
_OP_OPCODE, _OP_INPUTS, _OP_OUTPUTS, _OP_OPTIONS, _OP_CUSTOM_OPTIONS = 0, 1, 2, 4, 5
_OC_DEPRECATED_CODE, _OC_CUSTOM, _OC_VERSION, _OC_CODE = 0, 1, 2, 3

_BUILTIN_OPERATORS = (
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D",
    "DEPTH_TO_SPACE", "DEQUANTIZE", "EMBEDDING_LOOKUP", "FLOOR", "FULLY_CONNECTED",
    "HASHTABLE_LOOKUP", "L2_NORMALIZATION", "L2_POOL_2D",
    "LOCAL_RESPONSE_NORMALIZATION", "LOGISTIC", "LSH_PROJECTION", "LSTM", "MAX_POOL_2D",
    "MUL", "RELU", "RELU_N1_TO_1", "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RNN",
    "SOFTMAX", "SPACE_TO_DEPTH", "SVDF", "TANH", "CONCAT_EMBEDDINGS", "SKIP_GRAM",
    "CALL", "CUSTOM", "EMBEDDING_LOOKUP_SPARSE", "PAD", "UNIDIRECTIONAL_SEQUENCE_RNN",
    "GATHER", "BATCH_TO_SPACE_ND", "SPACE_TO_BATCH_ND", "TRANSPOSE", "MEAN", "SUB",
    "DIV", "SQUEEZE", "UNIDIRECTIONAL_SEQUENCE_LSTM", "STRIDED_SLICE",
    "BIDIRECTIONAL_SEQUENCE_RNN", "EXP", "TOPK_V2", "SPLIT", "LOG_SOFTMAX", "DELEGATE",
    "BIDIRECTIONAL_SEQUENCE_LSTM", "CAST", "PRELU", "MAXIMUM", "ARG_MAX", "MINIMUM",
    "LESS", "NEG", "PADV2", "GREATER", "GREATER_EQUAL", "LESS_EQUAL", "SELECT", "SLICE",
    "SIN", "TRANSPOSE_CONV", "SPARSE_TO_DENSE", "TILE", "EXPAND_DIMS", "EQUAL",
    "NOT_EQUAL", "LOG", "SUM", "SQRT", "RSQRT", "SHAPE", "POW", "ARG_MIN", "FAKE_QUANT",
    "REDUCE_PROD", "REDUCE_MAX", "PACK", "LOGICAL_OR", "ONE_HOT", "LOGICAL_AND",
    "LOGICAL_NOT", "UNPACK", "REDUCE_MIN", "FLOOR_DIV", "REDUCE_ANY", "SQUARE",
    "ZEROS_LIKE", "FILL", "FLOOR_MOD", "RANGE", "RESIZE_NEAREST_NEIGHBOR", "LEAKY_RELU",
    "SQUARED_DIFFERENCE", "MIRROR_PAD", "ABS", "SPLIT_V", "UNIQUE", "CEIL",
    "REVERSE_V2", "ADD_N", "GATHER_ND", "COS", "WHERE", "RANK", "ELU",
    "REVERSE_SEQUENCE", "MATRIX_DIAG", "QUANTIZE", "MATRIX_SET_DIAG", "ROUND",
    "HARD_SWISH", "IF", "WHILE", "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5",
    "SCATTER_ND", "SELECT_V2", "DENSIFY", "SEGMENT_SUM", "BATCH_MATMUL",
    "PLACEHOLDER_FOR_GREATER_OP_CODES", "CUMSUM", "CALL_ONCE", "BROADCAST_TO", "RFFT2D",
    "CONV_3D", "IMAG", "REAL", "COMPLEX_ABS", "HASHTABLE", "HASHTABLE_FIND",
    "HASHTABLE_IMPORT", "HASHTABLE_SIZE", "REDUCE_ALL", "CONV_3D_TRANSPOSE",
    "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "BROADCAST_ARGS",
    "RANDOM_STANDARD_NORMAL", "BUCKETIZE", "RANDOM_UNIFORM", "MULTINOMIAL", "GELU",
    "DYNAMIC_UPDATE_SLICE", "RELU_0_TO_1", "UNSORTED_SEGMENT_PROD",
    "UNSORTED_SEGMENT_MAX", "UNSORTED_SEGMENT_SUM", "ATAN2", "UNSORTED_SEGMENT_MIN",
    "SIGN", "BITCAST", "BITWISE_XOR", "RIGHT_SHIFT", "STABLEHLO_LOGISTIC",
    "STABLEHLO_ADD", "STABLEHLO_DIVIDE", "STABLEHLO_MULTIPLY", "STABLEHLO_MAXIMUM",
    "STABLEHLO_RESHAPE", "STABLEHLO_CLAMP", "STABLEHLO_CONCATENATE",
    "STABLEHLO_BROADCAST_IN_DIM", "STABLEHLO_CONVOLUTION", "STABLEHLO_SLICE",
    "STABLEHLO_CUSTOM_CALL", "STABLEHLO_REDUCE", "STABLEHLO_ABS", "STABLEHLO_AND",
    "STABLEHLO_COSINE", "STABLEHLO_EXPONENTIAL", "STABLEHLO_FLOOR", "STABLEHLO_LOG",
    "STABLEHLO_MINIMUM", "STABLEHLO_NEGATE", "STABLEHLO_OR", "STABLEHLO_POWER",
    "STABLEHLO_REMAINDER", "STABLEHLO_RSQRT", "STABLEHLO_SELECT", "STABLEHLO_SUBTRACT",
    "STABLEHLO_TANH", "STABLEHLO_SCATTER", "STABLEHLO_COMPARE", "STABLEHLO_CONVERT",
    "STABLEHLO_DYNAMIC_SLICE", "STABLEHLO_DYNAMIC_UPDATE_SLICE", "STABLEHLO_PAD",
    "STABLEHLO_IOTA", "STABLEHLO_DOT_GENERAL", "STABLEHLO_REDUCE_WINDOW",
    "STABLEHLO_SORT", "STABLEHLO_WHILE", "STABLEHLO_GATHER", "STABLEHLO_TRANSPOSE",
    "DILATE", "STABLEHLO_RNG_BIT_GENERATOR", "REDUCE_WINDOW", "STABLEHLO_COMPOSITE",
    "STABLEHLO_SHIFT_LEFT", "STABLEHLO_CBRT",
)
_TENSOR_TYPES = (
    "FLOAT32", "FLOAT16", "INT32", "UINT8", "INT64", "STRING", "BOOL", "INT16",
    "COMPLEX64", "INT8", "FLOAT64", "COMPLEX128", "UINT64", "RESOURCE", "VARIANT",
    "UINT32", "UINT16", "INT4", "BFLOAT16",
)


_ACTIVATIONS = {0: "none", 1: "relu", 2: "relu_n1_to_1", 3: "relu6", 4: "tanh", 5: "sign_bit"}
_NUMPY = {"FLOAT32": np.float32, "INT32": np.int32, "INT8": np.int8, "UINT8": np.uint8,
          "INT16": np.int16, "INT64": np.int64, "BOOL": np.bool_}


def is_tflite(data: bytes) -> bool:
    return len(data) >= 8 and data[4:8] == MAGIC


class _Model:
    """The parts of a TFLite model the frontend reads, resolved once."""

    def __init__(self, data: bytes):
        if not is_tflite(data):
            raise ValueError("not a TFLite FlatBuffer")
        self.data = data
        root = Table.root(data)
        self.version = root.scalar(_MODEL_VERSION, "I")
        self.description = root.string(_MODEL_DESCRIPTION)
        self.buffers = root.tables(_MODEL_BUFFERS)
        self.subgraphs = root.tables(_MODEL_SUBGRAPHS)
        self.opcodes = []
        for code in root.tables(_MODEL_OPCODES):
            builtin = max(code.scalar(_OC_CODE, "i"), code.scalar(_OC_DEPRECATED_CODE, "b"))
            custom = code.string(_OC_CUSTOM)
            name = (f"CUSTOM:{custom}" if custom else
                    _BUILTIN_OPERATORS[builtin] if 0 <= builtin < len(_BUILTIN_OPERATORS)
                    else f"BUILTIN_{builtin}")
            self.opcodes.append((name, code.scalar(_OC_VERSION, "i", 1)))

    def buffer(self, index: int) -> bytes:
        if not 0 <= index < len(self.buffers):
            return b""
        entry = self.buffers[index]
        offset = entry.scalar(_B_OFFSET, "Q")
        if offset > 1:
            size = entry.scalar(_B_SIZE, "Q")
            if offset + size > len(self.data):
                raise ValueError(f"buffer {index} runs past the file")
            return self.data[offset:offset + size]
        return entry.bytes(_B_DATA)


class _Tensor:
    def __init__(self, model: _Model, table: Table):
        self.name = table.string(_T_NAME)
        self.shape = table.scalars(_T_SHAPE, "i")
        code = table.scalar(_T_TYPE, "b")
        self.type = _TENSOR_TYPES[code] if 0 <= code < len(_TENSOR_TYPES) else f"TYPE_{code}"
        self.buffer = table.scalar(_T_BUFFER, "I")
        quant = table.table(_T_QUANT)
        self.scale = np.asarray(quant.scalars(_Q_SCALE, "f") if quant else [], np.float32)
        self.zero_point = np.asarray(quant.scalars(_Q_ZERO_POINT, "q") if quant else [], np.int64)
        self.quantized_dimension = quant.scalar(_Q_DIMENSION, "i") if quant else 0
        # A variable tensor keeps what an operator wrote into it for the next invocation.
        self.variable = bool(table.scalar(_T_VARIABLE, "B"))
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
        opcode = table.scalar(_OP_OPCODE, "I")
        self.kind, self.version = (model.opcodes[opcode] if opcode < len(model.opcodes)
                                   else (f"OPCODE_{opcode}", 1))
        self.inputs = table.scalars(_OP_INPUTS, "i")
        self.outputs = table.scalars(_OP_OUTPUTS, "i")
        self.options = table.table(_OP_OPTIONS)
        self.custom_options = table.bytes(_OP_CUSTOM_OPTIONS)
        self.index = index

    def option(self, slot: int, fmt: str, default=0):
        return self.options.scalar(slot, fmt, default) if self.options else default


def _read(data: bytes):
    model = _Model(data)
    graphs = []
    for subgraph in model.subgraphs:
        tensors = [_Tensor(model, t) for t in subgraph.tables(_SG_TENSORS)]
        operators = [_Operator(model, op, i) for i, op in enumerate(subgraph.tables(_SG_OPERATORS))]
        graphs.append((subgraph.string(_SG_NAME), tensors, subgraph.scalars(_SG_INPUTS, "i"),
                       subgraph.scalars(_SG_OUTPUTS, "i"), operators))
    # The converter passes a constant an IF branch reads as one of its
    # operands; inside the branch it is that constant, from the same buffer.
    # Each graph is folded before the constants it passes on are read.
    for index in range(len(graphs)):
        graphs[index] = _fold_constants(graphs[index])
        _, outer, _, _, operators = graphs[index]
        for op in operators:
            if op.kind != "IF":
                continue
            for slot in (0, 1):
                position = op.option(slot, "i")
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
    return None


def _fold_constants(graph):
    """Operators computed from static shapes and constants become constants:
    the shape and fill operators the converter folds in a static model, and a
    transpose of a weight it passed into a branch."""
    name, tensors, inputs, outputs, operators = graph
    kept = []
    for op in operators:
        result = tensors[op.outputs[0]] if len(op.outputs) == 1 else None
        value = (_constant_result(op, tensors)
                 if result is not None and op.outputs[0] not in outputs and result.type in _NUMPY
                 else None)
        if value is None:
            kept.append(op)
            continue
        result.folded = np.ascontiguousarray(value.astype(_NUMPY[result.type])).tobytes()
        result.buffer = 1 << 30
    return name, tensors, inputs, outputs, kept


def _dtype_name(tensor: _Tensor) -> str:
    return tensor.type.lower()


def unsupported(data: bytes) -> list[str]:
    """Why the model cannot be converted, one reason per line; empty when it can."""
    model, graphs = _read(data)
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
                    if op.kind in ("ADD", "SUB", "MUL") and tensors[op.outputs[0]].type == "INT32"}
        for index in list(inputs) + list(outputs):
            if (tensors[index].type not in ("INT8", "FLOAT32", "BOOL") and index not in indices
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
            found = ({op.option(0, "i"): "branch", op.option(1, "i"): "branch"} if op.kind == "IF"
                     else {op.option(0, "i"): "condition", op.option(1, "i"): "body"})
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
    return f"{op.options.string(0)}/{op.options.string(1)}"


def _initializer_subgraphs(graphs) -> set[int]:
    """Subgraphs the main graph runs once through CALL_ONCE."""
    return {op.option(0, "i") for op in graphs[0][4] if op.kind == "CALL_ONCE"} if graphs else set()


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


# Slot of the fused activation in each operator's options table.
_FUSED_SLOT = {"ADD": 0, "SUB": 0, "MUL": 0, "DIV": 0, "FULLY_CONNECTED": 0, "CONCATENATION": 1,
               "CONV_2D": 3, "TRANSPOSE_CONV": 3, "DEPTHWISE_CONV_2D": 4,
               "AVERAGE_POOL_2D": 5, "MAX_POOL_2D": 5, "L2_POOL_2D": 5}
_UNARY = {"LOGISTIC": "Sigmoid", "TANH": "Tanh", "HARD_SWISH": "HardSwish", "RELU": "Relu",
          "ELU": "Elu"}
_SHAPE_ONLY = ("RESHAPE", "SQUEEZE", "EXPAND_DIMS")
# Elementwise math: the ONNX operator, or a composition the compiler folds back.
_ELEMENTWISE_UNARY = {"ABS": "Abs", "NEG": "Neg", "EXP": "Exp", "LOG": "Log", "SQRT": "Sqrt",
                      "FLOOR": "Floor", "CEIL": "Ceil", "ROUND": "Round", "SIN": "Sin",
                      "COS": "Cos"}
_ELEMENTWISE_BINARY = ("DIV", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE", "FLOOR_DIV",
                       "FLOOR_MOD")
_ELEMENTWISE = (*_ELEMENTWISE_UNARY, *_ELEMENTWISE_BINARY, "RSQRT", "SQUARE")
# The ones TFLite Micro also runs in int8; the others are float only there.
_INT8_ELEMENTWISE = ("ABS", "RSQRT", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "DIV")
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
_STATE_TYPES = {"FLOAT32": TensorProto.FLOAT, "INT16": TensorProto.INT16, "INT8": TensorProto.INT8}
# Resource variables: a handle names a variable, which is read and assigned.
_VARIABLES = ("CALL_ONCE", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE")
# Index outputs: int32 positions, as model outputs or the indices of the operators below.
_INDEX = {"ARG_MAX": "ArgMax", "ARG_MIN": "ArgMin"}
# The operand holding indices or start positions, which may be computed at run time.
_INDEX_SLOTS = {"GATHER": 1, "GATHER_ND": 1, "EMBEDDING_LOOKUP": 0, "DYNAMIC_UPDATE_SLICE": 2}
# Reductions over one run of adjacent axes, and the prefix sum over one axis.
_REDUCTIONS = {"REDUCE_MAX": "ReduceMax", "REDUCE_MIN": "ReduceMin", "SUM": "ReduceSum",
               "REDUCE_ALL": "ReduceAll"}
_SUPPORTED = (*_FUSED_SLOT, *_UNARY, *_SHAPE_ONLY, *_ELEMENTWISE, *_DATA_MOVEMENT, *_REDUCTIONS,
              *_INDEX, *_BOOL_SLOTS, *_VARIABLES, *_STATEFUL, "ADD_N", "IF", "WHILE",
              "RELU6", "SOFTMAX", "LOG_SOFTMAX", "LEAKY_RELU", "PRELU", "L2_NORMALIZATION",
              "CUMSUM", "MEAN",
              "TRANSPOSE",
              "SPLIT", "SPLIT_V", "PAD", "PADV2", "BATCH_MATMUL", "RESIZE_NEAREST_NEIGHBOR",
              "RESIZE_BILINEAR", "QUANTIZE", "DEQUANTIZE")
# The SSD box decoding and non-max suppression TFLite Micro registers as a
# custom operator.
_DETECTION = "CUSTOM:TFLite_Detection_PostProcess"
_SUPPORTED = (*_SUPPORTED, _DETECTION)
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


def _is_constant(tensor: _Tensor) -> bool:
    return tensor.buffer > 0 and bool(tensor.data)


def _operator_reason(op: _Operator, tensors: list[_Tensor]) -> str:
    if op.kind not in _SUPPORTED:
        return "not supported"
    ins = [tensors[i] if i >= 0 else None for i in op.inputs]
    outs = [tensors[i] for i in op.outputs]
    if op.kind == _DETECTION:
        return _detection_reason(op, ins, outs)
    if op.kind in _VARIABLES:
        values = [t for t in [*ins, *outs] if t is not None and t.type != "RESOURCE"]
        if any(t.type != "FLOAT32" for t in values):
            # The converter keeps variables in float even in int8 models.
            return "variables other than float32"
        return ""
    for position in _CONSTANT_OPERANDS.get(op.kind, ()):
        if position < len(ins) and ins[position] is not None and not _is_constant(ins[position]):
            if position != _INDEX_SLOTS.get(op.kind):
                return f"input {position} must be a constant"
            if ins[position].type != "INT32":
                return f"{ins[position].type} run-time indices; the runtime takes int32"
    if op.kind == "IF":
        if ins[0].type != "BOOL" or int(np.prod(ins[0].shape)) != 1:
            return "a condition other than one bool"
        if any(t.type not in ("FLOAT32", "INT32")
               for t in [*(t for t in ins[1:] if not _is_constant(t)), *outs]):
            return "operands other than float32 or int32"
        return ""
    if op.kind == "WHILE":
        if any(t.type not in ("FLOAT32", "INT32") for t in [*ins, *outs]):
            return "loop variables other than float32 or int32"
        return ""
    if any(t is not None and t.type == "INT32" for t in [*ins, *outs]):
        reason = _int32_reason(op, ins, outs)
        if reason is not None:
            return reason
    if op.kind == "SVDF":
        return _svdf_reason(op, ins, outs)
    if op.kind == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        return _lstm_reason(op, ins, outs)
    data = [t for i, t in enumerate(ins) if t is not None and i not in _CONSTANT_OPERANDS.get(op.kind, ())]
    if op.kind in _WEIGHTED:
        source, position, bias_position = _WEIGHTED[op.kind]
        weights = ins[position]
        bias = ins[bias_position] if bias_position < len(ins) else None
        data = [ins[source]]
        if data[0].type == "FLOAT32":
            if weights.type != "FLOAT32" or not _is_constant(weights):
                return "weights of a float32 operator must be constant float32"
            if bias is not None and bias.type != "FLOAT32":
                return "bias of a float32 operator must be float32"
        else:
            # TFLite's int8 convolutions ignore the filter zero point, which the
            # converter's own QUANTIZE of a weight can leave nonzero.
            symmetric = op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D") or not np.any(weights.zero_point != 0)
            if weights.type != "INT8" or not symmetric or not _is_constant(weights):
                return "weights must be constant symmetric int8"
            if bias is not None and bias.type != "INT32":
                return "bias must be int32"
    if op.kind in _BOOL_SLOTS:
        positions, bool_output = _BOOL_SLOTS[op.kind]
        if any((t.type == "BOOL") != (i in positions) for i, t in enumerate(ins) if t is not None):
            return "operands of the wrong dtype"
        if (outs[0].type == "BOOL") != bool_output:
            return "an output of the wrong dtype"
        if op.kind == "CAST" and outs[0].type == "INT8" and (
                outs[0].scale.size != 1 or outs[0].scale[0] != 1 or outs[0].zero_point[0] != 0):
            # TFLite copies the raw 0 or 1 into the int8 output.
            return "int8 output encoding other than scale 1 and zero point 0"
        data = [t for i, t in enumerate(ins) if t is not None and i not in positions
                and i not in _CONSTANT_OPERANDS.get(op.kind, ())]
        if bool_output:
            outs = []
    if op.kind in _INDEX:
        if outs[0].type != "INT32":
            return f"{outs[0].type} indices; TFLite Micro writes int32"
        outs = []
    if op.kind == "QUANTIZE":
        if data[0].type not in ("INT8", "FLOAT32") or outs[0].type != "INT8":
            return "only float or int8 to int8 converts"
    elif op.kind == "DEQUANTIZE":
        if data[0].type != "INT8" or outs[0].type != "FLOAT32":
            return "only int8 to float converts"
    elif all(t.type == "FLOAT32" for t in [*data, *outs]):
        pass
    elif any(t.type != "INT8" for t in [*data, *outs]):
        return "activations must be all int8 or all float32"
    elif (op.kind in _ELEMENTWISE and op.kind not in _INT8_ELEMENTWISE) or op.kind == "L2_POOL_2D":
        return "TFLite Micro runs it in float32 only"
    if any(len(t.scale) != 1 for t in [*data, *outs] if t.type == "INT8"):
        return "activations must be quantized per tensor"
    if op.kind in _FUSED_SLOT:
        fused = _activation(op.option(_FUSED_SLOT[op.kind], "b"))
        if fused not in ("none", "relu", "relu6"):
            return f"fused {fused}"
    if op.kind in _ELEMENTWISE_BINARY and all(_is_constant(t) for t in ins):
        return "only constant operands"
    if op.kind == "CONCATENATION" and outs[0].type == "INT8":
        if any(t.scale[0] != outs[0].scale[0] or t.zero_point[0] != outs[0].zero_point[0] for t in data):
            return "inputs quantized differently from the output"
    if op.kind in _DATA_MOVEMENT:
        reason = _data_movement_reason(op, ins, outs)
        if reason:
            return reason
    if op.kind == "L2_NORMALIZATION" and op.option(0, "b") != 0:
        return f"fused {_activation(op.option(0, 'b'))}"
    if op.kind in _REDUCTIONS or op.kind == "MEAN":
        axes = sorted({a % len(ins[0].shape) for a in ins[1].array().reshape(-1).tolist()})
        if not axes or axes != list(range(axes[0], axes[-1] + 1)):
            return "axes that are not adjacent"
    if op.kind == "SOFTMAX" and op.option(0, "f", 1.0) != 1.0:
        return "beta other than 1"
    if op.kind == "FULLY_CONNECTED" and op.option(1, "b") != 0:
        return "shuffled weights format"
    if op.kind == "BATCH_MATMUL" and len(ins[0].shape) != len(ins[1].shape):
        return "operands of different rank"
    if op.kind in ("RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"):
        align, half_pixel = ((op.option(2, "?", False), op.option(3, "?", False))
                             if op.kind == "RESIZE_BILINEAR" else
                             (op.option(0, "?", False), op.option(1, "?", False)))
        if align and half_pixel:
            return "align_corners and half_pixel_centers together"
    return ""


def _integer_use(op: _Operator) -> bool:
    """Operators that take an int32 model input as a value."""
    return op.kind in ("ADD", "SUB", "MUL", "CAST", "IF", "WHILE", *_COMPARISONS)


def _int32_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> "str | None":
    """Why an int32 computation does not convert, "" when it does, None when
    the operator takes int32 only as indices or axes."""
    if op.kind in ("ADD", "SUB", "MUL"):
        if any(t.type != "INT32" for t in [*ins, *outs]):
            return "int32 mixed with other operands"
        if _activation(op.option(_FUSED_SLOT[op.kind], "b")) != "none":
            return "a fused activation on int32"
        if all(_is_constant(t) for t in ins):
            return "only constant operands"
        return ""
    if op.kind in _COMPARISONS:
        if any(t.type != "INT32" for t in ins):
            return "int32 compared with another dtype"
        return ""
    if op.kind == "CAST":
        return "" if outs[0].type == "FLOAT32" else "int32 cast to other than float32"
    return None


def _same_quantization(tensors) -> bool:
    first = tensors[0]
    return all(t.type == first.type and np.array_equal(t.scale, first.scale)
               and np.array_equal(t.zero_point, first.zero_point) for t in tensors[1:])


def _strided_slice_bounds(op: _Operator, ins: list[_Tensor]):
    """Per axis (begin, size, shrink) of a stride-1 STRIDED_SLICE, or None."""
    shape = ins[0].shape
    begins, ends, strides = (list(ins[i].array().reshape(-1)) for i in (1, 2, 3))
    begin_mask, end_mask = op.option(0, "i"), op.option(1, "i")
    shrink_mask = op.option(4, "i")
    if (op.option(2, "i") or op.option(3, "i") or op.option(5, "?", False)
            or len(begins) != len(shape) or any(int(v) != 1 for v in strides)):
        return None
    bounds = []
    for axis, extent in enumerate(shape):
        begin = 0 if begin_mask >> axis & 1 else int(begins[axis])
        end = extent if end_mask >> axis & 1 else int(ends[axis])
        begin = min(max(begin + extent if begin < 0 else begin, 0), extent)
        end = min(max(end + extent if end < 0 else end, 0), extent)
        shrink = bool(shrink_mask >> axis & 1)
        if shrink:
            end = begin + 1
        if end <= begin:
            return None
        bounds.append((begin, end - begin, shrink))
    return bounds


def _gather_run(op: _Operator, ins: list[_Tensor]):
    """(axis, first, count, scalar) when a GATHER takes a contiguous run of one
    axis, or None."""
    indices = [int(v) for v in ins[1].array().reshape(-1)]
    axis = op.option(0, "i") % len(ins[0].shape)
    extent = ins[0].shape[axis]
    indices = [i + extent if i < 0 else i for i in indices]
    if (op.option(1, "i") or not indices or any(not 0 <= i < extent for i in indices)
            or indices != list(range(indices[0], indices[0] + len(indices)))):
        return None
    return axis, indices[0], len(indices), len(ins[1].shape) == 0


def _svdf_reason(op: _Operator, ins: list, outs: list[_Tensor]) -> str:
    x, feature, time, bias, state = ins
    y = outs[0]
    if bias is None:
        # TFLite Micro's Prepare reads the bias whether or not it is there.
        return "no bias, which TFLite Micro requires"
    if len(x.shape) != 2 or op.option(0, "i") <= 0:
        return "input of rank other than 2"
    if x.type == "FLOAT32":
        if any(t.type != "FLOAT32" for t in (feature, time, bias, state, y)):
            return "float32 input with operands of another dtype"
        fused = _activation(op.option(1, "b"))
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
    if _activation(op.option(0, "b")) != "tanh":
        return f"cell activation {_activation(op.option(0, 'b'))}"
    if any(not _is_constant(ins[i]) for i in _LSTM_CONSTANTS):
        return "weights and biases must be constant"
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
    axes = sorted({a % len(source.shape) for a in tensors[op.inputs[1]].array().reshape(-1).tolist()})
    return len(source.shape) == 4 and axes == [1, 2]


def _data_movement_reason(op: _Operator, ins: list[_Tensor], outs: list[_Tensor]) -> str:
    slot = _INDEX_SLOTS.get(op.kind)
    data = [t for i, t in enumerate(ins) if not _is_constant(t) and i != slot]
    if not _same_quantization([*data, *outs]):
        return "inputs and outputs quantized differently"
    if op.kind == "STRIDED_SLICE":
        if op.option(2, "i") or op.option(3, "i") or op.option(5, "?", False):
            return "ellipsis, new axes or offset slices"
        if len(ins[0].shape) > 4:
            return "slices of rank above 4"
        steps = [int(v) for v in ins[3].array().reshape(-1)]
        if any(step == 0 for step in steps):
            return "a zero stride"
        if any(op.option(4, "i") >> axis & 1 and step < 0 for axis, step in enumerate(steps)):
            # TFLite Micro leaves such an output unwritten.
            return "a dropped axis with a negative stride"
    if op.kind == "REVERSE_V2":
        axes = sorted({a % len(ins[0].shape) for a in ins[1].array().reshape(-1).tolist()})
        if not axes or axes != list(range(axes[0], axes[-1] + 1)):
            return "axes that are not adjacent"
    if op.kind in ("SPACE_TO_DEPTH", "DEPTH_TO_SPACE", "SPACE_TO_BATCH_ND",
                   "BATCH_TO_SPACE_ND") and len(ins[0].shape) != 4:
        return "only rank-4 inputs convert"
    return ""


def describe(data: bytes) -> dict:
    """The model's own facts, readable whether or not it converts."""
    model, graphs = _read(data)
    name, tensors, inputs, outputs, operators = graphs[0] if graphs else ("", [], [], [], [])

    def interface(index):
        tensor = tensors[index]
        return {"name": tensor.name, "kind": "tensor", "dtype": _dtype_name(tensor),
                "shape": list(tensor.shape)}

    constants = [t for t in tensors if t.buffer > 0 and t.data]
    return {
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
                              "size_bytes": len(t.data), "storage": "embedded"} for t in constants],
            "sparse_initializers": [],
        },
        "unsupported": unsupported(data),
    }


# ONNX carries tensors of rank 3 and 4 channels-first; TFLite's are
# channels-last, which is how the compiler stores such a tensor.
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
        # int8 model outputs are the QuantizeLinear itself, as TFLite hands them over.
        self.int8_outputs = {i for i in outputs if tensors[i].type == "INT8"}
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
            zero = b.constant(np.array(int(tensor.zero_point[0]), np.int8), tensor.name + "_zero_point")
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
        ins, outs = op.inputs, op.outputs
        if op.kind in _VARIABLES:
            # A read sees the variable's latest value: the state input, or what
            # this invocation assigned before it.
            if op.kind == "VAR_HANDLE":
                self.handles[outs[0]] = _var_name(op)
            elif op.kind == "READ_VARIABLE":
                self.values[outs[0]] = self.variables[self.handles[ins[0]]]["current"]
            elif op.kind == "ASSIGN_VARIABLE":
                self.variables[self.handles[ins[0]]]["current"] = self.value(ins[1])
            return
        out = self.tensors[outs[0]]
        tag = out.name + "_float"
        kind = op.kind
        fused = _activation(op.option(_FUSED_SLOT[kind], "b")) if kind in _FUSED_SLOT else "none"
        b = self.b
        if kind in _INDEX:
            # The index in TFLite's axis order, cast to the int32 TFLite writes.
            rank = len(self.tensors[ins[0]].shape)
            x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
            y = b.node(_INDEX[kind], [x], tag, axis=self.ints(ins[1])[0] % rank, keepdims=0,
                       select_last_index=0)
            self.values[outs[0]] = b.node("Cast", [y], out.name, to=TensorProto.INT32)
            return
        if kind in ("IF", "WHILE"):
            self._control_flow(op, tag)
            return
        if kind in ("QUANTIZE", "DEQUANTIZE") and (
                (ins[0] if kind == "QUANTIZE" else outs[0]) in self.float_boundary
                and self.tensors[ins[0] if kind == "DEQUANTIZE" else outs[0]].type == "INT8"):
            y = self.custom("Quantize" if kind == "QUANTIZE" else "Dequantize", [self.value(ins[0])],
                            tag, _onnx_shape(out.shape))
            self.finish(y, outs[0])
            return
        if kind == "DEQUANTIZE" and outs[0] in self.detection_operands:
            # The detection operator reads the int8 tensor through its quantization.
            self.values[outs[0]] = self.value(ins[0])
            return
        if kind == _DETECTION:
            self._detection(op, tag)
            return
        if kind == "SVDF":
            self.finish(self._svdf(op, tag), outs[0])
            return
        if kind == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            self.finish(self._lstm(op, tag), outs[0])
            return
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            y = self._conv(op, tag)
        elif kind == "TRANSPOSE_CONV":
            y = self._transpose_conv(op, tag)
        elif kind == "FULLY_CONNECTED":
            y = self._fully_connected(op, tag)
        elif kind in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            source = self.tensors[ins[0]]
            kh, kw = op.option(4, "i", 1), op.option(3, "i", 1)
            sh, sw = op.option(2, "i", 1), op.option(1, "i", 1)
            top, bottom = same_padding(source.shape[1], out.shape[1], kh, sh)
            left, right = same_padding(source.shape[2], out.shape[2], kw, sw)
            attributes = {"count_include_pad": 0} if kind == "AVERAGE_POOL_2D" else {}
            y = b.node("MaxPool" if kind == "MAX_POOL_2D" else "AveragePool", [self.value(ins[0])],
                       tag, kernel_shape=[kh, kw], strides=[sh, sw],
                       pads=[top, left, bottom, right], **attributes)
        elif kind in ("ADD", "SUB", "MUL"):
            y = b.node({"ADD": "Add", "SUB": "Sub", "MUL": "Mul"}[kind], self.operands(op), tag)
        elif kind == "CONCATENATION":
            y = self._concat([(self.value(i), list(self.tensors[i].shape), i) for i in ins],
                             op.option(0, "i") % len(out.shape), outs[0], tag)
        elif kind in _ELEMENTWISE_UNARY:
            y = b.node(_ELEMENTWISE_UNARY[kind], [self.value(ins[0])], tag)
        elif kind == "RSQRT":
            y = b.node("Reciprocal", [b.node("Sqrt", [self.value(ins[0])], tag + "_sqrt")], tag)
        elif kind == "SQUARE":
            x = self.value(ins[0])
            y = b.node("Mul", [x, x], tag)
        elif kind in ("DIV", "MAXIMUM", "MINIMUM"):
            y = b.node({"DIV": "Div", "MAXIMUM": "Max", "MINIMUM": "Min"}[kind],
                       self.operands(op), tag)
        elif kind == "SQUARED_DIFFERENCE":
            difference = b.node("Sub", self.operands(op), tag + "_difference")
            y = b.node("Mul", [difference, difference], tag)
        elif kind == "FLOOR_DIV":
            y = b.node("Floor", [b.node("Div", self.operands(op), tag + "_quotient")], tag)
        elif kind == "FLOOR_MOD":
            y = self._floor_mod(*self.operands(op), tag)
        elif kind in _UNARY:
            y = b.node(_UNARY[kind], [self.value(ins[0])], tag)
        elif kind in _COMPARISONS:
            y = b.node(_COMPARISONS[kind], self.operands(op), tag + "_equal" if kind == "NOT_EQUAL" else tag)
            if kind == "NOT_EQUAL":
                y = b.node("Not", [y], tag)
        elif kind in _LOGICAL:
            y = b.node(_LOGICAL[kind], self.operands(op), tag)
        elif kind == "SELECT_V2":
            y = b.node("Where", self.operands(op), tag)
        elif kind == "CAST":
            # A bool cast to int8 is the float 0.0 or 1.0 quantized into the output.
            y = b.node("Cast", [self.value(ins[0])], tag, to=TensorProto.FLOAT)
        elif kind == "ADD_N":
            y = b.node("Sum", [self.value(i) for i in ins], tag)
        elif kind == "RELU6":
            y = b.fused_activation(self.value(ins[0]), "relu6", tag)
        elif kind in ("SOFTMAX", "LOG_SOFTMAX"):
            y = b.node("Softmax" if kind == "SOFTMAX" else "LogSoftmax", [self.value(ins[0])], tag,
                       axis=_onnx_axis(-1, len(out.shape)))
        elif kind == "LEAKY_RELU":
            y = b.node("LeakyRelu", [self.value(ins[0])], tag, alpha=op.option(0, "f", 0.0))
        elif kind == "PRELU":
            y = b.node("PRelu", self.operands(op), tag)
        elif kind == "L2_NORMALIZATION":
            y = self._l2_normalization(op, tag)
        elif kind == "L2_POOL_2D":
            source = self.tensors[ins[0]]
            kh, kw = op.option(4, "i", 1), op.option(3, "i", 1)
            sh, sw = op.option(2, "i", 1), op.option(1, "i", 1)
            top, bottom = same_padding(source.shape[1], out.shape[1], kh, sh)
            left, right = same_padding(source.shape[2], out.shape[2], kw, sw)
            x = self.value(ins[0])
            pooled = b.node("AveragePool", [b.node("Mul", [x, x], tag + "_square")], tag + "_mean",
                            kernel_shape=[kh, kw], strides=[sh, sw],
                            pads=[top, left, bottom, right], count_include_pad=0)
            y = b.node("Sqrt", [pooled], tag)
        elif kind in _REDUCTIONS or kind == "CUMSUM":
            y = self._reduction(op, tag)
        elif kind == "MEAN" and not _spatial_mean(op, self.tensors):
            y = self._reduction(op, tag)
        elif kind == "MEAN":
            source = self.tensors[ins[0]]
            rank = len(source.shape)
            axes = sorted(_onnx_axis(a, rank) for a in self.ints(ins[1]))
            y = b.node("ReduceMean", [self.value(ins[0])], tag, axes=axes, keepdims=1)
            if len(out.shape) != rank:
                kept = [1 if i in [a % rank for a in self.ints(ins[1])] else d
                        for i, d in enumerate(source.shape)]
                y = self.reshape(self.held(y, outs[0], tag + "_kept"), kept, out.shape,
                                 tag + "_squeezed", outs[0])
        elif kind in _SHAPE_ONLY:
            y = self.reshape(self.value(ins[0]), self.tensors[ins[0]].shape, out.shape, tag, outs[0])
        elif kind == "TRANSPOSE":
            rank = len(out.shape)
            perm = [p % rank for p in self.ints(ins[1])]
            first, last = _to_first(rank), _to_last(rank)
            perm = [last[perm[first[i]]] for i in range(rank)]
            y = b.node("Transpose", [self.value(ins[0])], tag, perm=perm)
        elif kind in ("SPLIT", "SPLIT_V"):
            source_index, axis_index = (ins[1], ins[0]) if kind == "SPLIT" else (ins[0], ins[2])
            rank = len(self.tensors[source_index].shape)
            axis = self.ints(axis_index)[0] % rank
            sizes = b.constant(np.asarray([self.tensors[i].shape[axis] for i in outs], np.int64),
                               out.name + "_sizes")
            names = [self.tensors[i].name + "_float" for i in outs]
            parts = b.multi_node("Split", [self.value(source_index), sizes], names,
                                 axis=_onnx_axis(axis, rank))
            for index, part in zip(outs, parts):
                self.finish(part, index)
            return
        elif kind in ("PAD", "PADV2"):
            rank = len(out.shape)
            pairs = np.asarray(self.ints(ins[1])).reshape(rank, 2)
            order = _to_first(rank)
            pads = b.constant(np.asarray([pairs[a, 0] for a in order] + [pairs[a, 1] for a in order],
                                         np.int64), tag + "_pads")
            source = self.tensors[ins[0]]
            if kind != "PADV2" or len(ins) < 3:
                fill = 0.0
            elif source.type == "FLOAT32":
                fill = float(self.tensors[ins[2]].array().reshape(-1)[0])
            else:
                fill = float((int(self.ints(ins[2])[0]) - source.zero_point[0]) * source.scale[0])
            y = b.node("Pad", [self.value(ins[0]), pads, b.constant(np.float32(fill), tag + "_fill")],
                       tag, mode="constant")
        elif kind == "GATHER" and (not _is_constant(self.tensors[ins[1]])
                                   or _gather_run(op, [self.tensors[i] for i in ins]) is None):
            y = self._gather("Gather", ins[0], ins[1], op.option(0, "i"), op, tag)
        elif kind == "EMBEDDING_LOOKUP":
            y = self._gather("Gather", ins[1], ins[0], 0, op, tag)
        elif kind == "GATHER_ND":
            y = self._gather("GatherND", ins[0], ins[1], None, op, tag)
        elif (kind == "STRIDED_SLICE"
              and _strided_slice_bounds(op, [self.tensors[i] for i in ins]) is None):
            y = self._strided_slice(op, tag)
        elif kind == "MIRROR_PAD":
            rank = len(out.shape)
            pairs = np.asarray(self.ints(ins[1])).reshape(rank, 2)
            order = _to_first(rank)
            pads = b.constant(np.asarray([pairs[a, 0] for a in order] + [pairs[a, 1] for a in order],
                                         np.int64), tag + "_pads")
            if op.option(0, "b") == 0:
                y = b.node("Pad", [self.value(ins[0]), pads], tag, mode="reflect")
            else:
                y = self.custom("MirrorPad", [self.value(ins[0]), pads], tag,
                                _onnx_shape(out.shape), mode="symmetric")
        elif kind == "REVERSE_V2":
            rank = len(out.shape)
            x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
            axes = b.constant(np.asarray(self.ints(ins[1]), np.int64), tag + "_axes")
            y = self.tflite_order_out(self.custom("ReverseV2", [x, axes], tag + "_reversed",
                                                  out.shape), outs[0], tag)
        elif kind == "DYNAMIC_UPDATE_SLICE" and not _is_constant(self.tensors[ins[2]]):
            # Run-time starts are in TFLite's axis order, so the update runs there.
            rank = len(out.shape)
            x = self.last_to_last(self.value(ins[0]), rank, tag + "_last", ins[0], True)
            u = self.last_to_last(self.value(ins[1]), rank, tag + "_update", ins[1], True)
            y = self.tflite_order_out(self.custom("DynamicUpdateSlice", [x, u, self.value(ins[2])],
                                                  tag + "_updated", out.shape), outs[0], tag)
        elif kind == "DYNAMIC_UPDATE_SLICE":
            rank = len(out.shape)
            starts = [self.ints(ins[2])[a] for a in _to_first(rank)]
            y = self.custom("DynamicUpdateSlice",
                            [self.value(ins[0]), self.value(ins[1]),
                             b.constant(np.asarray(starts, np.int64), tag + "_starts")],
                            tag, _onnx_shape(out.shape))
        elif kind in ("SLICE", "STRIDED_SLICE", "GATHER"):
            source = self.tensors[ins[0]]
            if kind == "SLICE":
                sizes = self.ints(ins[2])
                bounds = [(begin, (extent - begin) if size == -1 else size, False)
                          for begin, size, extent in zip(self.ints(ins[1]), sizes, source.shape)]
            elif kind == "STRIDED_SLICE":
                bounds = _strided_slice_bounds(op, [self.tensors[i] for i in ins])
            else:
                axis, first, count, scalar = _gather_run(op, [self.tensors[i] for i in ins])
                bounds = [(0, extent, False) for extent in source.shape]
                bounds[axis] = (first, count, scalar)
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
                y = self.reshape(y, shape, out.shape, tag, outs[0])
            elif not cut:
                y = b.node("Identity", [y], tag)
        elif kind == "PACK":
            axis = op.option(1, "i") % len(out.shape)
            parts = []
            for step, i in enumerate(ins):
                shape = list(self.tensors[i].shape)
                expanded = shape[:axis] + [1] + shape[axis:]
                parts.append((self.held(self.reshape(self.value(i), shape, expanded,
                                                     f"{tag}_part{step}", i), i,
                                        f"{tag}_part{step}_q"), expanded, i))
            y = self._concat(parts, axis, outs[0], tag)
        elif kind == "UNPACK":
            source = self.tensors[ins[0]]
            axis = op.option(1, "i") % len(source.shape)
            names = [f"{self.tensors[i].name}_float_part" for i in outs]
            sizes = b.constant(np.ones(len(outs), np.int64), tag + "_sizes")
            parts = b.multi_node("Split", [self.value(ins[0]), sizes], names,
                                 axis=_onnx_axis(axis, len(source.shape)))
            shape = list(source.shape)
            shape[axis] = 1
            for part, index in zip(parts, outs):
                name = self.tensors[index].name + "_float"
                held = self.held(part, index, name + "_part_q")
                self.finish(self.reshape(held, shape, self.tensors[index].shape, name, index),
                            index)
            return
        elif kind in ("SPACE_TO_DEPTH", "DEPTH_TO_SPACE"):
            # Reshape to six axes, swap the two in the middle, reshape back,
            # all in TFLite's own axis order.
            block = op.option(0, "i")
            n, h, w, c = self.tensors[ins[0]].shape
            if kind == "SPACE_TO_DEPTH":
                split = [n, h // block, block, w // block, block, c]
            else:
                split = [n, h, w, block, block, c // (block * block)]
            y = self.last_to_last(self.value(ins[0]), 4, tag + "_nhwc", ins[0], True)
            y = self.held(b.node("Reshape", [y, b.constant(np.asarray(split, np.int64),
                                                           tag + "_split")], tag + "_six"),
                          ins[0], tag + "_six_q")
            y = self.held(b.node("Transpose", [y], tag + "_swap", perm=[0, 1, 3, 2, 4, 5]),
                          ins[0], tag + "_swap_q")
            y = self.held(b.node("Reshape", [y, b.constant(np.asarray(out.shape, np.int64),
                                                           tag + "_joined")], tag + "_four"),
                          ins[0], tag + "_four_q")
            y = b.node("Transpose", [y], tag, perm=_to_first(4))
        elif kind in ("SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"):
            y = self._batch_space(op, tag)
        elif kind == "BROADCAST_TO":
            # Leading axes by reshape, then each broadcast axis as copies.
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
        elif kind == "BATCH_MATMUL":
            # The matrices are the last two TFLite axes, so the product runs
            # in TFLite's own axis order.
            rank = len(out.shape)
            operands = []
            for i, adjoint in zip(ins, (op.option(0, "?", False), op.option(1, "?", False))):
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
                y = b.node("Transpose", [self.held(y, outs[0], tag + "_q")], tag + "_first",
                           perm=_to_first(rank))
        elif kind in ("RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"):
            sizes = b.constant(np.asarray(_onnx_shape(out.shape), np.int64), tag + "_sizes")
            align = op.option(2 if kind == "RESIZE_BILINEAR" else 0, "?", False)
            half_pixel = op.option(3 if kind == "RESIZE_BILINEAR" else 1, "?", False)
            if kind == "RESIZE_BILINEAR":
                coordinate = "align_corners" if align else "half_pixel" if half_pixel else "asymmetric"
                attributes = {"mode": "linear", "coordinate_transformation_mode": coordinate}
            else:
                coordinate = "align_corners" if align else "tf_half_pixel_for_nn" if half_pixel else "asymmetric"
                attributes = {"mode": "nearest", "coordinate_transformation_mode": coordinate,
                              "nearest_mode": "round_prefer_ceil" if align else "floor"}
            y = b.node("Resize", [self.value(ins[0]), "", "", sizes], tag, **attributes)
        elif kind in ("QUANTIZE", "DEQUANTIZE"):
            y = self.value(ins[0])
        else:
            raise ValueError(f"no conversion for {kind}")
        self.finish(b.fused_activation(y, fused, tag), outs[0])

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
        """A reduction or prefix sum over adjacent axes, run on the three axes
        [before, reduced, after] of the TFLite shape and reshaped back."""
        b = self.b
        ins, source = op.inputs, self.tensors[op.inputs[0]]
        out = self.tensors[op.outputs[0]]
        shape = list(source.shape)
        axes = sorted({a % len(shape) for a in self.ints(ins[1])})
        first, last = axes[0], axes[-1] + 1
        rows = [int(np.prod(shape[:first])), int(np.prod(shape[first:last])),
                int(np.prod(shape[last:]))]
        x = self.value(ins[0])
        if rows != shape:
            x = self.held(self.reshape(x, shape, rows, tag + "_rows", ins[0]), ins[0],
                          tag + "_rows_q")
        axis = _onnx_axis(1, 3)
        if op.kind == "CUMSUM":
            operands = [x, b.constant(np.asarray(axis, np.int64), tag + "_axis")]
            options = {"exclusive": int(op.option(0, "?", False)),
                       "reverse": int(op.option(1, "?", False))}
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

    def _gather(self, kind: str, data: int, ids: int, axis, op: _Operator, tag: str) -> str:
        """GATHER, GATHER_ND or EMBEDDING_LOOKUP with constant indices, run in
        TFLite's axis order, where its axis and index tuples are stated."""
        rank = len(self.tensors[data].shape)
        x = self.last_to_last(self.value(data), rank, tag + "_last", data, True)
        indices = (self.b.constant(np.asarray(self.tensors[ids].array(), np.int64), tag + "_indices")
                   if _is_constant(self.tensors[ids]) else self.value(ids))
        attributes = {} if axis is None else {"axis": axis % rank}
        batch = op.option(1, "i") if op.kind == "GATHER" else 0
        if batch:
            # ONNX Gather has no batch dimensions; the compiler's own form does.
            y = self.custom("Gather", [x, indices], tag + "_gathered",
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
        attributes = {"rank": op.option(0, "i"),
                      "activation": "none" if quantized else _activation(op.option(1, "b"))}
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
            helper.make_tensor_value_info(kept, _STATE_TYPES[held.type], list(held.shape))]
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
        graphs = {role: _branch_graph(self.graphs, op.option(slot, "i"), f"{tag}_{role}")
                  for slot, role in enumerate(roles)}
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
        attributes = {"time_major": int(op.option(3, "?", False)),
                      "cell_clip": float(op.option(1, "f", 0.0))}
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
                    name, _STATE_TYPES[held.type], list(held.shape)))
            self.held_state[index]["output"] = name
        return self.tflite_order_out(self.held(y, op.outputs[0], tag + "_q"), op.outputs[0], tag)

    def _strided_slice(self, op: _Operator, tag: str) -> str:
        """A STRIDED_SLICE with any strides, as an ONNX Slice in TFLite's axis
        order; a dropped axis is a slice of one, reshaped away."""
        b = self.b
        ins, source = op.inputs, self.tensors[op.inputs[0]]
        shape, rank = list(source.shape), len(source.shape)
        begins, ends, steps = (self.ints(i) for i in ins[1:4])
        begin_mask, end_mask, shrink = op.option(0, "i"), op.option(1, "i"), op.option(4, "i")
        starts, stops, sliced = [], [], []
        for axis, extent in enumerate(shape):
            if shrink >> axis & 1:
                start = begins[axis] + extent if begins[axis] < 0 else begins[axis]
                stop, step = start + 1, 1
                steps[axis] = 1
            else:
                start, stop, step = slice(None if begin_mask >> axis & 1 else begins[axis],
                                          None if end_mask >> axis & 1 else ends[axis],
                                          steps[axis]).indices(extent)
            sliced.append(len(range(start, stop, step)))
            starts.append(start)
            # Below zero, a reverse slice runs past the first element.
            stops.append(stop if stop >= 0 else -(2**63))
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

    def _six(self, x: str, split, perm, joined, tag: str, index: int) -> str:
        """Reshape to six axes, permute, reshape back, in TFLite's axis order,
        each step held as tensor `index` is quantized."""
        b = self.b
        y = self.held(b.node("Reshape", [x, b.constant(np.asarray(split, np.int64),
                                                       tag + "_split")], tag + "_six"),
                      index, tag + "_six_q")
        y = self.held(b.node("Transpose", [y], tag + "_perm", perm=list(perm)), index,
                      tag + "_perm_q")
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
        kh, kw = weights.shape[2], weights.shape[3]
        sw, sh = op.option(1, "i", 1), op.option(2, "i", 1)
        dw, dh = (op.option(5, "i", 1), op.option(6, "i", 1)) if depthwise else (
            op.option(4, "i", 1), op.option(5, "i", 1))
        top, bottom = same_padding(source.shape[1], out.shape[1], (kh - 1) * dh + 1, sh)
        left, right = same_padding(source.shape[2], out.shape[2], (kw - 1) * dw + 1, sw)
        w = self.weights(weights_tensor, weights, 0)
        operands = [self.value(op.inputs[0]), w]
        bias = self._bias(op, 2, tag)
        if bias:
            operands.append(bias)
        return self.b.node("Conv", operands, tag, kernel_shape=[kh, kw], strides=[sh, sw],
                           dilations=[dh, dw], pads=[top, left, bottom, right],
                           group=source.shape[3] if depthwise else 1)

    def _transpose_conv(self, op: _Operator, tag: str) -> str:
        source, weights_tensor = self.tensors[op.inputs[2]], self.tensors[op.inputs[1]]
        out = self.tensors[op.outputs[0]]
        weights = np.transpose(weights_tensor.array(), (3, 0, 1, 2))
        kh, kw = weights.shape[2], weights.shape[3]
        sw, sh = op.option(1, "i", 1), op.option(2, "i", 1)
        top, bottom = same_padding(out.shape[1], source.shape[1], kh, sh)
        left, right = same_padding(out.shape[2], source.shape[2], kw, sw)
        w = self.weights(weights_tensor, weights, 1)
        operands = [self.value(op.inputs[2]), w]
        bias = self._bias(op, 3, tag)
        if bias:
            operands.append(bias)
        return self.b.node("ConvTranspose", operands, tag, kernel_shape=[kh, kw],
                           strides=[sh, sw], pads=[top, left, bottom, right])

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
            y = self.reshape(self.held(y, op.outputs[0], tag + "_q"), [rows, weights_tensor.shape[0]],
                             out.shape, tag + "_shape", op.outputs[0])
        return y


def to_onnx(data: bytes, name: str) -> onnx.ModelProto:
    """The model as QDQ ONNX with its TFLite tensor names; raises ValueError
    naming every construct that does not convert."""
    reasons = unsupported(data)
    if reasons:
        raise ValueError("TFLite model does not convert: " + "; ".join(reasons))
    _, graphs = _read(data)
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
    boundary_type = {"INT8": TensorProto.INT8, "FLOAT32": TensorProto.FLOAT,
                     "INT32": TensorProto.INT32, "BOOL": TensorProto.BOOL}
    indices = {i for op in operators if op.kind in _INDEX for i in op.outputs}

    # The interface keeps TFLite's dtypes: an int8 input is dequantized straight
    # off the graph input.
    onnx_inputs = []
    for index in inputs:
        tensor = tensors[index]
        b._names.add(tensor.name)
        onnx_inputs.append(helper.make_tensor_value_info(
            tensor.name, boundary_type[tensor.type], _onnx_shape(tensor.shape)))
        if tensor.type in ("FLOAT32", "BOOL", "INT32"):
            converter.values[index] = tensor.name
        else:
            scale = b.constant(np.float32(tensor.scale[0]), tensor.name + "_scale")
            zero = b.constant(np.array(int(tensor.zero_point[0]), np.int8),
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
            name, _STATE_TYPES[tensor.type], list(tensor.shape)))
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
            held["output"], _STATE_TYPES[tensor.type], list(tensor.shape)))
    for entry, variable in zip(state, shapes):
        current = converter.variables[variable]
        if current["current"] != entry["input"]:
            entry["output"] = b.node("Identity", [current["current"]], current["name"] + "_out")
            state_outputs.append(helper.make_tensor_value_info(
                entry["output"], TensorProto.FLOAT, _onnx_shape(current["shape"])))
    # An index is stored as the operator writes it, in TFLite's axis order.
    onnx_outputs = [helper.make_tensor_value_info(
        tensors[index].name if index in converter.int8_outputs else converter.values[index],
        boundary_type[tensors[index].type],
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
