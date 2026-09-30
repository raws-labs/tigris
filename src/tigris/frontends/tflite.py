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
_SUPPORTED = ("ADD", "AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
              "RESHAPE", "SOFTMAX")


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
            if tensors[index].type != "INT8":
                reasons.append(f"model boundary {tensors[index].name!r} is {tensors[index].type}, not INT8")
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


def _operator_reason(op: _Operator, tensors: list[_Tensor]) -> str:
    if op.kind not in _SUPPORTED:
        return "not supported"
    ins = [tensors[i] for i in op.inputs if i >= 0]
    out = tensors[op.outputs[0]]
    if ins[0].type != "INT8" or out.type != "INT8":
        return "only int8 activations convert"
    if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
        if ins[1].type != "INT8" or np.any(ins[1].zero_point != 0):
            return "weights must be symmetric int8"
        if len(ins) > 2 and ins[2].type != "INT32":
            return "bias must be int32"
    if op.kind == "CONV_2D":
        if op.option(4, "i", 1) != 1 or op.option(5, "i", 1) != 1:
            return "dilation other than 1"
        if _activation(op.option(3, "b")) not in ("none", "relu", "relu6"):
            return f"fused {_activation(op.option(3, 'b'))}"
    if op.kind == "DEPTHWISE_CONV_2D":
        if op.option(5, "i", 1) != 1 or op.option(6, "i", 1) != 1:
            return "dilation other than 1"
        if ins[0].shape[3] != out.shape[3]:
            return "depth multiplier other than 1"
        if _activation(op.option(4, "b")) not in ("none", "relu", "relu6"):
            return f"fused {_activation(op.option(4, 'b'))}"
    if op.kind == "AVERAGE_POOL_2D":
        if (op.option(4, "i"), op.option(3, "i")) != tuple(ins[0].shape[1:3]) or list(out.shape[1:3]) != [1, 1]:
            return "only a pool over the whole map converts"
        if _activation(op.option(5, "b")) != "none":
            return "fused activation"
    if op.kind == "RESHAPE" and len(out.shape) != 2:
        return "only a flattening reshape converts"
    if op.kind == "FULLY_CONNECTED":
        if op.option(1, "b") != 0 or op.option(2, "?", False) or len(ins[0].shape) != 2:
            return "only a rank-2 product with default weights format converts"
        if _activation(op.option(0, "b")) not in ("none", "relu", "relu6"):
            return f"fused {_activation(op.option(0, 'b'))}"
    if op.kind == "ADD":
        if ins[0].shape != ins[1].shape:
            return "only an add of equal shapes converts"
        if _activation(op.option(0, "b")) not in ("none", "relu", "relu6"):
            return f"fused {_activation(op.option(0, 'b'))}"
    if op.kind == "SOFTMAX" and op.option(0, "f", 1.0) != 1.0:
        return "beta other than 1"
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


def to_onnx(data: bytes, name: str) -> onnx.ModelProto:
    """The model as QDQ ONNX with its TFLite tensor names; raises ValueError
    naming every construct that does not convert."""
    reasons = unsupported(data)
    if reasons:
        raise ValueError("TFLite model does not convert: " + "; ".join(reasons))
    _, graphs = _read(data)
    _, tensors, inputs, outputs, operators = graphs[0]
    b = GraphBuilder()
    values: dict[int, str] = {}

    def qparams(tensor):
        return float(tensor.scale[0]), int(tensor.zero_point[0])

    onnx_inputs = []
    for index in inputs:
        tensor = tensors[index]
        n, h, w, c = tensor.shape
        b._names.add(tensor.name)
        onnx_inputs.append(helper.make_tensor_value_info(tensor.name, TensorProto.FLOAT, [n, c, h, w]))
        values[index] = b.requantized(tensor.name, *qparams(tensor), tensor.name + "_int8")

    def finish(source, op):
        out = tensors[op.outputs[0]]
        values[op.outputs[0]] = b.requantized(source, *qparams(out), out.name)

    for op in operators:
        ins = [tensors[i] for i in op.inputs if i >= 0]
        out = tensors[op.outputs[0]]
        tag = out.name + "_float"
        x = values[op.inputs[0]]
        if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            depthwise = op.kind == "DEPTHWISE_CONV_2D"
            weights = ins[1].array()
            weights = np.transpose(weights, (3, 0, 1, 2) if depthwise else (0, 3, 1, 2))
            kh, kw = weights.shape[2], weights.shape[3]
            sw, sh = op.option(1, "i", 1), op.option(2, "i", 1)
            top, bottom = same_padding(ins[0].shape[1], out.shape[1], kh, sh)
            left, right = same_padding(ins[0].shape[2], out.shape[2], kw, sw)
            w = b.dequantized_constant(weights, ins[1].scale, np.zeros_like(ins[1].zero_point),
                                       ins[1].name, axis=0)
            operands = [x, w]
            if len(ins) > 2:
                operands.append(b.dequantized_constant(
                    ins[2].array(), ins[2].scale, np.zeros_like(ins[2].zero_point), ins[2].name,
                    axis=0, zero_dtype=np.int32))
            y = b.node("Conv", operands, tag, kernel_shape=[kh, kw], strides=[sh, sw],
                       pads=[top, left, bottom, right], group=weights.shape[0] if depthwise else 1)
            finish(b.fused_activation(y, _activation(op.option(4 if depthwise else 3, "b")), tag), op)
        elif op.kind == "AVERAGE_POOL_2D":
            h, w = ins[0].shape[1:3]
            y = b.node("AveragePool", [x], tag, kernel_shape=[h, w], strides=[1, 1])
            finish(y, op)
        elif op.kind == "ADD":
            y = b.node("Add", [x, values[op.inputs[1]]], tag)
            finish(b.fused_activation(y, _activation(op.option(0, "b")), tag), op)
        elif op.kind == "RESHAPE":
            finish(b.node("Flatten", [x], tag, axis=1), op)
        elif op.kind == "FULLY_CONNECTED":
            w = b.dequantized_constant(ins[1].array(), ins[1].scale,
                                       np.zeros_like(ins[1].zero_point), ins[1].name, axis=0)
            if len(ins) > 2:
                bias = b.dequantized_constant(ins[2].array(), ins[2].scale,
                                              np.zeros_like(ins[2].zero_point), ins[2].name,
                                              axis=0, zero_dtype=np.int32)
            else:
                bias = b.constant(np.zeros(ins[1].shape[0], np.float32), out.name + "_bias")
            y = b.node("Gemm", [x, w, bias], tag, transB=1)
            finish(b.fused_activation(y, _activation(op.option(0, "b")), tag), op)
        elif op.kind == "SOFTMAX":
            finish(b.node("Softmax", [x], tag, axis=1), op)

    onnx_outputs = []
    for index in outputs:
        tensor = tensors[index]
        onnx_outputs.append(helper.make_tensor_value_info(values[index], TensorProto.FLOAT,
                                                          list(tensor.shape)))
    model = helper.make_model(helper.make_graph(b.nodes, name, onnx_inputs, onnx_outputs,
                                                b.initializers),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model
