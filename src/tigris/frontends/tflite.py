"""TFLite FlatBuffer models: inspection and conversion to QDQ ONNX.

Inspection reads any TFLite file. Conversion covers the int8 operators whose
TFLite semantics a QDQ ONNX graph states exactly and refuses everything else
by name, so a model is never compiled with a reinterpreted operator.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

from tigris.frontends.flatbuffer import Table
from tigris.frontends.qdq import GraphBuilder, same_padding

MAGIC = b"TFL3"
# Metadata stating that every model input and output of rank 3 or more is held
# channels-last, as the TFLite tensor is laid out.
BOUNDARY_LAYOUT_KEY = "tigris.boundary_layout"

# Field slots in the TFLite schema (tensorflow/lite/schema/schema.fbs).
_MODEL_VERSION, _MODEL_OPCODES, _MODEL_SUBGRAPHS, _MODEL_DESCRIPTION, _MODEL_BUFFERS = 0, 1, 2, 3, 4
_SG_TENSORS, _SG_INPUTS, _SG_OUTPUTS, _SG_OPERATORS, _SG_NAME = 0, 1, 2, 3, 4
_T_SHAPE, _T_TYPE, _T_BUFFER, _T_NAME, _T_QUANT = 0, 1, 2, 3, 4
_Q_SCALE, _Q_ZERO_POINT, _Q_DIMENSION = 2, 3, 6
_B_DATA, _B_OFFSET, _B_SIZE = 0, 1, 2
_OP_OPCODE, _OP_INPUTS, _OP_OUTPUTS, _OP_OPTIONS = 0, 1, 2, 4
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
          "INT16": np.int16, "INT64": np.int64}


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
        self._model = model

    @property
    def data(self) -> bytes:
        return self._model.buffer(self.buffer)

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
    return model, graphs


def _dtype_name(tensor: _Tensor) -> str:
    return tensor.type.lower()


def unsupported(data: bytes) -> list[str]:
    """Why the model cannot be converted, one reason per line; empty when it can."""
    model, graphs = _read(data)
    reasons = []
    if len(graphs) != 1:
        reasons.append(f"{len(graphs)} subgraphs; only single-subgraph models convert")
    for _, tensors, inputs, outputs, operators in graphs[:1]:
        for index in list(inputs) + list(outputs):
            if tensors[index].type not in ("INT8", "FLOAT32"):
                reasons.append(f"model boundary {tensors[index].name!r} is {tensors[index].type}")
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


def _activation(code: int) -> str:
    return _ACTIVATIONS.get(code, f"activation {code}")


# Slot of the fused activation in each operator's options table.
_FUSED_SLOT = {"ADD": 0, "SUB": 0, "MUL": 0, "FULLY_CONNECTED": 0, "CONCATENATION": 1,
               "CONV_2D": 3, "TRANSPOSE_CONV": 3, "DEPTHWISE_CONV_2D": 4,
               "AVERAGE_POOL_2D": 5, "MAX_POOL_2D": 5}
_UNARY = {"LOGISTIC": "Sigmoid", "TANH": "Tanh", "HARD_SWISH": "HardSwish", "RELU": "Relu"}
_SHAPE_ONLY = ("RESHAPE", "SQUEEZE", "EXPAND_DIMS")
_SUPPORTED = (*_FUSED_SLOT, *_UNARY, *_SHAPE_ONLY, "RELU6", "SOFTMAX", "MEAN", "TRANSPOSE",
              "SPLIT", "SPLIT_V", "PAD", "PADV2", "BATCH_MATMUL", "RESIZE_NEAREST_NEIGHBOR",
              "RESIZE_BILINEAR", "QUANTIZE", "DEQUANTIZE")
# Positions of the data, weights and bias operands of the weighted operators.
_WEIGHTED = {"CONV_2D": (0, 1, 2), "DEPTHWISE_CONV_2D": (0, 1, 2), "FULLY_CONNECTED": (0, 1, 2),
             "TRANSPOSE_CONV": (2, 1, 3)}
_CONSTANT_OPERANDS = {"MEAN": (1,), "TRANSPOSE": (1,), "SPLIT": (0,), "SPLIT_V": (1, 2),
                      "PAD": (1,), "PADV2": (1, 2), "RESHAPE": (1,), "EXPAND_DIMS": (1,),
                      "RESIZE_NEAREST_NEIGHBOR": (1,), "RESIZE_BILINEAR": (1,),
                      "TRANSPOSE_CONV": (0,)}


def _is_constant(tensor: _Tensor) -> bool:
    return tensor.buffer > 0 and bool(tensor.data)


def _operator_reason(op: _Operator, tensors: list[_Tensor]) -> str:
    if op.kind not in _SUPPORTED:
        return "not supported"
    ins = [tensors[i] if i >= 0 else None for i in op.inputs]
    outs = [tensors[i] for i in op.outputs]
    for position in _CONSTANT_OPERANDS.get(op.kind, ()):
        if position < len(ins) and ins[position] is not None and not _is_constant(ins[position]):
            return f"input {position} must be a constant"
    data = [t for i, t in enumerate(ins) if t is not None and i not in _CONSTANT_OPERANDS.get(op.kind, ())]
    if op.kind in _WEIGHTED:
        source, position, bias_position = _WEIGHTED[op.kind]
        weights = ins[position]
        if weights.type != "INT8" or np.any(weights.zero_point != 0) or not _is_constant(weights):
            return "weights must be constant symmetric int8"
        bias = ins[bias_position] if bias_position < len(ins) else None
        if bias is not None and bias.type != "INT32":
            return "bias must be int32"
        data = [ins[source]]
    if op.kind == "QUANTIZE":
        if data[0].type not in ("INT8", "FLOAT32") or outs[0].type != "INT8":
            return "only float or int8 to int8 converts"
    elif op.kind == "DEQUANTIZE":
        if data[0].type != "INT8" or outs[0].type != "FLOAT32":
            return "only int8 to float converts"
    elif any(t.type != "INT8" for t in [*data, *outs]):
        return "only int8 activations convert"
    if any(len(t.scale) != 1 for t in [*data, *outs] if t.type == "INT8"):
        return "activations must be quantized per tensor"
    if op.kind in _FUSED_SLOT:
        fused = _activation(op.option(_FUSED_SLOT[op.kind], "b"))
        if fused not in ("none", "relu", "relu6"):
            return f"fused {fused}"
    rank = len(outs[0].shape)
    if op.kind in ("ADD", "SUB", "MUL"):
        for operand in ins:
            if not _is_constant(operand) and len(operand.shape) != rank:
                return "operands of different rank"
    if op.kind == "CONCATENATION":
        if any(t.scale[0] != outs[0].scale[0] or t.zero_point[0] != outs[0].zero_point[0] for t in data):
            return "inputs quantized differently from the output"
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
        if align:
            return "align_corners"
        if op.kind == "RESIZE_NEAREST_NEIGHBOR" and half_pixel and any(
                o % i for i, o in zip(ins[0].shape[1:3], outs[0].shape[1:3])):
            return "half-pixel centers with a non-integer scale"
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
def _to_first(rank: int) -> list[int]:
    return [0, rank - 1, *range(1, rank - 1)] if rank >= 3 else list(range(rank))


def _to_last(rank: int) -> list[int]:
    return [0, *range(2, rank), 1] if rank >= 3 else list(range(rank))


def _onnx_axis(axis: int, rank: int) -> int:
    return _to_last(rank)[axis % rank]


def _onnx_shape(shape) -> list[int]:
    return [shape[i] for i in _to_first(len(shape))]


def _layout_free(shape) -> bool:
    """True when both axis orders hold the elements in the same sequence."""
    return len(shape) < 3 or shape[-1] == 1 or int(np.prod(shape[1:-1])) == 1


class _Converter:
    """Builds the QDQ ONNX graph operator by operator; `values` maps a TFLite
    tensor index to the ONNX value holding it, rank-4 values in NCHW."""

    def __init__(self, tensors: list[_Tensor]):
        self.b = GraphBuilder()
        self.tensors = tensors
        self.values: dict[int, str] = {}

    def value(self, index: int, rank: int | None = None) -> str:
        """The ONNX value of a tensor; a constant is broadcast-aligned to `rank`."""
        if index in self.values and rank is None:
            return self.values[index]
        tensor = self.tensors[index]
        array = tensor.array()
        rank = rank or array.ndim
        array = array.reshape((1,) * (rank - array.ndim) + array.shape)
        axis = _onnx_axis(tensor.quantized_dimension + rank - len(tensor.shape), rank)
        array = array.transpose(_to_first(rank))
        if tensor.type == "FLOAT32":
            return self.b.constant(array, tensor.name)
        return self.b.dequantized_constant(array, tensor.scale, tensor.zero_point, tensor.name,
                                           axis=axis)

    def ints(self, index: int) -> list[int]:
        return [int(v) for v in self.tensors[index].array().reshape(-1)]

    def held(self, source: str, index: int, name: str) -> str:
        """`source` requantized as tensor `index` is, so a data-movement step
        between two operators stays in int8."""
        tensor = self.tensors[index]
        if tensor.type == "FLOAT32":
            return source
        return self.b.requantized(source, float(tensor.scale[0]), int(tensor.zero_point[0]), name)

    def finish(self, source: str, index: int) -> None:
        tensor = self.tensors[index]
        if tensor.type == "FLOAT32":
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
        out = self.tensors[outs[0]]
        tag = out.name + "_float"
        kind = op.kind
        fused = _activation(op.option(_FUSED_SLOT[kind], "b")) if kind in _FUSED_SLOT else "none"
        b = self.b
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
            rank = len(out.shape)
            y = b.node({"ADD": "Add", "SUB": "Sub", "MUL": "Mul"}[kind],
                       [self.value(i, rank if i not in self.values else None) for i in ins], tag)
        elif kind == "CONCATENATION":
            y = b.node("Concat", [self.value(i) for i in ins], tag,
                       axis=_onnx_axis(op.option(0, "i"), len(out.shape)))
        elif kind in _UNARY:
            y = b.node(_UNARY[kind], [self.value(ins[0])], tag)
        elif kind == "RELU6":
            y = b.fused_activation(self.value(ins[0]), "relu6", tag)
        elif kind == "SOFTMAX":
            y = b.node("Softmax", [self.value(ins[0])], tag, axis=_onnx_axis(-1, len(out.shape)))
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
            fill = (float((int(self.ints(ins[2])[0]) - source.zero_point[0]) * source.scale[0])
                    if kind == "PADV2" and len(ins) > 2 else 0.0)
            y = b.node("Pad", [self.value(ins[0]), pads, b.constant(np.float32(fill), tag + "_fill")],
                       tag, mode="constant")
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
            if rank >= 3:
                y = b.node("Transpose", [self.held(y, outs[0], tag + "_q")], tag + "_first",
                           perm=_to_first(rank))
        elif kind in ("RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"):
            sizes = b.constant(np.asarray(_onnx_shape(out.shape), np.int64), tag + "_sizes")
            half_pixel = op.option(3 if kind == "RESIZE_BILINEAR" else 1, "?", False)
            if kind == "RESIZE_BILINEAR":
                attributes = {"mode": "linear", "coordinate_transformation_mode":
                              "half_pixel" if half_pixel else "asymmetric"}
            else:
                # With an integer scale, half-pixel centers pick the same source
                # pixel as the asymmetric floor rule.
                attributes = {"mode": "nearest", "coordinate_transformation_mode": "asymmetric",
                              "nearest_mode": "floor"}
            y = b.node("Resize", [self.value(ins[0]), "", "", sizes], tag, **attributes)
        elif kind in ("QUANTIZE", "DEQUANTIZE"):
            y = self.value(ins[0])
        else:
            raise ValueError(f"no conversion for {kind}")
        self.finish(b.fused_activation(y, fused, tag), outs[0])

    def _bias(self, op: _Operator, position: int, name: str) -> str | None:
        if position >= len(op.inputs) or op.inputs[position] < 0:
            return None
        tensor = self.tensors[op.inputs[position]]
        return self.b.dequantized_constant(tensor.array(), tensor.scale,
                                           np.zeros_like(tensor.zero_point), tensor.name,
                                           axis=0, zero_dtype=np.int32)

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
        w = self.b.dequantized_constant(weights, weights_tensor.scale,
                                        np.zeros_like(weights_tensor.zero_point),
                                        weights_tensor.name, axis=0)
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
        w = self.b.dequantized_constant(weights, weights_tensor.scale,
                                        np.zeros_like(weights_tensor.zero_point),
                                        weights_tensor.name, axis=1)
        operands = [self.value(op.inputs[2]), w]
        bias = self._bias(op, 3, tag)
        if bias:
            operands.append(bias)
        return self.b.node("ConvTranspose", operands, tag, kernel_shape=[kh, kw],
                           strides=[sh, sw], pads=[top, left, bottom, right])

    def _fully_connected(self, op: _Operator, tag: str) -> str:
        source, weights_tensor = self.tensors[op.inputs[0]], self.tensors[op.inputs[1]]
        out = self.tensors[op.outputs[0]]
        w = self.b.dequantized_constant(weights_tensor.array(), weights_tensor.scale,
                                        np.zeros_like(weights_tensor.zero_point),
                                        weights_tensor.name, axis=0)
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
    converter = _Converter(tensors)
    b = converter.b

    onnx_inputs = []
    for index in inputs:
        tensor = tensors[index]
        b._names.add(tensor.name)
        onnx_inputs.append(helper.make_tensor_value_info(tensor.name, TensorProto.FLOAT,
                                                         _onnx_shape(tensor.shape)))
        if tensor.type == "FLOAT32":
            converter.values[index] = tensor.name
        else:
            converter.values[index] = b.requantized(tensor.name, float(tensor.scale[0]),
                                                    int(tensor.zero_point[0]),
                                                    tensor.name + "_int8")
    for op in operators:
        converter.convert(op)
    onnx_outputs = [helper.make_tensor_value_info(converter.values[index], TensorProto.FLOAT,
                                                  _onnx_shape(tensors[index].shape)) for index in outputs]
    model = helper.make_model(helper.make_graph(b.nodes, name, onnx_inputs, onnx_outputs,
                                                b.initializers),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.helper.set_model_props(model, {BOUNDARY_LAYOUT_KEY: "channels_last"})
    onnx.checker.check_model(model)
    return model
