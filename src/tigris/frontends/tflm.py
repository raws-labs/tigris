"""TFLite Micro's memory on its reference kernels: the arena head its greedy
planner places for one subgraph, and the persistent allocations of its arena
tail, record by record, transcribed from the TFLite Micro sources."""

import bisect
import functools
import math
from typing import NamedTuple

_ELEMENT_BYTES = {"FLOAT32": 4, "INT32": 4, "UINT32": 4, "INT8": 1, "UINT8": 1, "BOOL": 1,
                  "INT16": 2, "UINT16": 2, "FLOAT16": 2, "INT64": 8, "UINT64": 8, "FLOAT64": 8}
# Reference kernels whose scratch buffers tflm_arena places.
_TFLM_REDUCTIONS = ("MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_ALL")


def tflm_tensor_bytes(tensors: list, operators: list = ()) -> list[int] | None:
    """The bytes TFLite Micro's planner places for each (storage type, shape,
    placed) tensor, 0 for one not placed, once each RESHAPE among `operators`,
    (kind, inputs, outputs) in execution order, has resolved a -1 in its output
    shape from its input's element count; None when a tensor has a storage type
    the model does not know."""
    shapes = [list(shape) for _, shape, _ in tensors]
    for kind, inputs, outputs in operators:
        if kind != "RESHAPE":
            continue
        shape = shapes[outputs[0]]
        known = math.prod(d for d in shape if d != -1)
        if shape.count(-1) == 1 and known:
            shape[shape.index(-1)] = math.prod(shapes[inputs[0]]) // known
    sizes = []
    for (dtype, _, placed), shape in zip(tensors, shapes):
        if dtype not in _ELEMENT_BYTES:
            return None
        sizes.append(math.prod(shape) * _ELEMENT_BYTES[dtype] if placed else 0)
    return sizes


def _tflm_scratch(kind: str, tensors: list, inputs: list, outputs: list) -> list[int]:
    """The bytes of each scratch buffer TFLite Micro's reference kernel for the
    operator requests, in request order, as TFLite Micro's reduction kernels size them."""
    if kind not in _TFLM_REDUCTIONS:
        return []
    (dtype, shape), axis = tensors[inputs[0]][:2], tensors[inputs[1]][1]
    # One int32 accumulator per output for int8 and int16 MEAN and SUM.
    accumulators = [4 * math.prod(tensors[outputs[0]][1])] if kind in ("MEAN", "SUM") and (
        dtype in ("INT8", "INT16")) else []
    return [*accumulators, 4 * len(shape), 4 * math.prod(axis)]


def tflm_arena(tensors: list, operators: list, inputs, outputs) -> int | None:
    """TFLite Micro's arena head for one subgraph on its reference kernels:
    `tensors` holds each tensor's (storage type, shape, placed), placed False
    for a constant or a variable, and `operators` each operator's (kind,
    inputs, outputs) in execution order. Its greedy planner places every placed
    tensor, aligned to 16 bytes and live from the operator that writes it
    (inputs from the start) to the last that reads it (outputs to the end),
    and the scratch buffers of MEAN, SUM, REDUCE_MAX, REDUCE_MIN and
    REDUCE_ALL. Other kernels' scratch buffers and TFLite Micro's persistent
    allocations come on top. None when a tensor has a storage type the model
    does not know."""
    sizes = tflm_tensor_bytes(tensors, operators)
    if sizes is None:
        return None
    scratch = [(k, size) for k, (kind, ins, outs) in enumerate(operators)
               for size in _tflm_scratch(kind, tensors, ins, outs)]
    return _greedy_arena(sizes, [(ins, outs) for _, ins, outs in operators], inputs, outputs,
                         scratch)


def tflm_lifetimes(count: int, operators: list[tuple], inputs, outputs) -> tuple[list, list]:
    """The first and last scope of each of `count` tensors in TFLite Micro's
    planner: inputs live from scope 0, a tensor from the operator that writes
    it, every operand to its operator's scope and outputs to the end.
    `operators` holds each operator's (inputs, outputs) in execution order,
    operator k at scope k + 1. Both are -1 where nothing sets them."""
    first = [-1] * count
    last = [-1] * count

    def created(index, scope):
        if first[index] == -1:
            first[index] = scope

    for index in inputs:
        created(index, 0)
        last[index] = 0
    for scope, (op_inputs, op_outputs) in enumerate(operators, start=1):
        for index in op_outputs:
            created(index, scope)
        for index in [*op_inputs, *op_outputs]:
            if index >= 0:
                last[index] = scope
    for index in outputs:
        created(index, len(operators))
        last[index] = len(operators)
    return first, last


def _greedy_arena(sizes: list[int], operators: list[tuple], inputs, outputs, scratch) -> int:
    """TFLite Micro's GreedyMemoryPlanner over one subgraph: `sizes` holds the
    bytes of each tensor, 0 for one it does not place, `operators` the
    (inputs, outputs) tensor indices of each operator in execution order, and
    `scratch` the (operator, bytes) of each kernel scratch buffer in request
    order, live while its operator runs."""
    first, last = tflm_lifetimes(len(sizes), operators, inputs, outputs)
    # Requests in tensor order: (size, first, last).
    requests = [((size + 15) // 16 * 16, first[index], last[index])
                for index, size in enumerate(sizes) if size]
    requests += [((size + 15) // 16 * 16, op + 1, op + 1) for op, size in scratch if size]
    if not requests:
        return 0
    # GreedyMemoryPlanner: requests listed in reverse, a stable sort by
    # decreasing size, then each placed at the lowest offset clear of every
    # placed buffer whose lifetime overlaps its own.
    order = sorted(reversed(range(len(requests))), key=lambda i: -requests[i][0])
    placed = []  # (offset, size, first, last), ascending offset, insertion-stable
    offsets = []
    for i in order:
        size, start, end = requests[i]
        offset = 0
        for other_offset, other_size, other_start, other_end in placed:
            if other_start > end or start > other_end:
                continue
            if other_offset - offset >= size:
                break
            offset = max(offset, other_offset + other_size)
        at = bisect.bisect_right(offsets, offset)
        offsets.insert(at, offset)
        placed.insert(at, (offset, size, start, end))
    return max(offset + size for offset, size, _, _ in placed)


class TflmTarget(NamedTuple):
    """The C ABI TFLite Micro is built for: the bytes of a pointer and of a
    size_t, and whether an enum without a fixed type takes the smallest integer
    that holds its values (every such enum TFLite Micro stores fits one byte)."""
    pointer: int
    short_enums: bool


TFLM_TARGETS = {"arm-none-eabi": TflmTarget(4, True), "ilp32": TflmTarget(4, False),
                "lp64": TflmTarget(8, False)}

# Members of each record TFLite Micro allocates in its arena tail, in declaration order.
_TFLM_RECORDS = {
    "SingleArenaBufferAllocator": "pp ppppp z i",
    "RecordingSingleArenaBufferAllocator": "pp ppppp z i zzzz",
    "GreedyMemoryPlanner": "p ii pppp ii p c",
    "MicroAllocator": "p pp p p c z p z",
    "RecordingMicroAllocator": "p pp p p c z p z p (zzz)7",
    "MicroBuiltinDataAllocator": "pp",
    "SubgraphAllocations": "pp",
    "TfLiteEvalTensor": "pp e",
    "NodeAndRegistration": "(pppppp i) p",
    "TfLiteTensor": "(e p) (i i) p p z e e c",
    "TfLiteAffineQuantization": "pp i",
    "ScratchBufferHandle": "p",
    "TfLiteAddParams": "e c",
    "TfLiteArgMaxParams": "e",
    "TfLiteArgMinParams": "e",
    "TfLiteBatchMatMulParams": "ccc",
    "TfLiteCastParams": "ee",
    "TfLiteConcatenationParams": "i e",
    "TfLiteConvParams": "e i i e i i e",
    "TfLiteCumsumParams": "cc",
    "TfLiteDepthToSpaceParams": "i",
    "TfLiteDepthwiseConvParams": "e i i i e i i",
    "TfLiteDivParams": "e",
    "TfLiteFullyConnectedParams": "e e c c e p i",
    "TfLiteGatherParams": "ii",
    "TfLiteL2NormParams": "e",
    "TfLiteLeakyReluParams": "i",
    "TfLiteMulParams": "e",
    "TfLitePackParams": "ii",
    "TfLitePoolParams": "e i i i i e (iiii)",
    "TfLiteReducerParams": "c",
    "TfLiteReshapeParams": "i8 i",
    "TfLiteResizeBilinearParams": "cc",
    "TfLiteResizeNearestNeighborParams": "cc",
    "TfLiteShapeParams": "e",
    "TfLiteSoftmaxParams": "i",
    "TfLiteSpaceToDepthParams": "i",
    "TfLiteSplitParams": "i",
    "TfLiteSplitVParams": "i",
    "TfLiteSqueezeParams": "i8 i",
    "TfLiteStridedSliceParams": "i5 c",
    "TfLiteSubParams": "e c",
    "TfLiteUnpackParams": "ii",
    "Comparison OpData": "(i7 c)",
    "Concatenation OpData": "(i p p i i i)",
    "Cumsum OpData": "i9",
    "DequantizeOpData": "(q i) i i i",
    "Elu OpData": "c256",
    "EmbeddingLookup OpData": "i z",
    "HardSwishParams": "h h h i h i",
    "L2NormalizationParams": "i",
    "LeakyReluOpData": "i6",
    "LogSoftmaxOpData": "i5 zz",
    "OpDataAbsRsqrt": "iiii c e e",
    "OpDataAdd": "c i14",
    "OpDataBatchMatmul": "ppp ccc",
    "OpDataConv": "(iiii) i5 pp i3",
    "OpDataDiv": "i7",
    "OpDataFullyConnected": "i9 pp c",
    "OpDataLogistic": "i4",
    "OpDataMul": "i9",
    "OpDataPooling": "(iiii) i4",
    "OpDataQuantizeReference": "(i q) i i i",
    "OpDataReduce": "i11",
    "OpDataSub": "c i12",
    "Tanh OpData": "i4",
    "Pad OpData": "(c i5 c i5 c) i",
    "PreluParams": "i7",
    "QuantizationOpDataBatchMatmul": "i7",
    "Relu6OpData": "ii",
    "ReluOpData": "c i i i4",
    "Select OpData": "c",
    "SoftmaxParams": "q i7 p5",
    "SpaceToBatchParams": "i",
    "SquaredDifference OpData": "c (c i14 q2 h2 i5)",
    "StridedSliceParams": "c i5 c i5 c i5 h5 c",
}
# Members a build with USE_TFLM_COMPRESSION appends to a record.
_TFLM_COMPRESSION = {"SubgraphAllocations": "p", "OpDataConv": "ii", "OpDataFullyConnected": "ii",
                     "Concatenation OpData": "i10", "RecordingMicroAllocator": "(zzz)"}


@functools.cache
def _tflm_layout(name: str, target: TflmTarget, compression: bool = False) -> tuple[int, int]:
    """sizeof and alignof of a TFLite Micro record on the target. Member codes:
    c one byte, h two, i four (int, int32_t, float), q eight (int64_t, double),
    p a pointer, z a size_t or intptr_t, e an enum without a fixed type, (...)
    a nested struct; a count after a member repeats it. A class with virtual
    functions starts with its vtable pointers."""
    fields = _TFLM_RECORDS[name] + (" " + _TFLM_COMPRESSION.get(name, "") if compression else "")
    fields = fields.replace(" ", "")
    width = {"c": 1, "h": 2, "i": 4, "q": 8, "p": target.pointer, "z": target.pointer,
             "e": 1 if target.short_enums else 4}

    def members(pos):
        size, align = 0, 1
        while pos < len(fields) and fields[pos] != ")":
            if fields[pos] == "(":
                (member, member_align), pos = members(pos + 1)
            else:
                member = member_align = width[fields[pos]]
            pos += 1
            count = ""
            while pos < len(fields) and fields[pos].isdigit():
                count, pos = count + fields[pos], pos + 1
            size = -(-size // member_align) * member_align + member * int(count or 1)
            align = max(align, member_align)
        return (-(-size // align) * align, align), pos

    return members(0)[0]


def _tflm_dims(rank: int) -> int:
    """Bytes of a TfLiteIntArray of `rank` elements."""
    return 4 + 4 * rank


def _tflm_quantized(tensor) -> bool:
    """Whether TFLite Micro gives the tensor affine quantization parameters."""
    return tensor[3] > 0 and tensor[4] > 0


def _tflm_output_dims(tensors, inputs, outputs, options, target):
    return [_tflm_dims(len(tensors[outputs[0]][1]))]


def _tflm_conv(axis):
    def prepare(tensors, inputs, outputs, options, target):
        return [4 * tensors[inputs[1]][1][axis]] * 2
    return prepare


def _tflm_fully_connected(tensors, inputs, outputs, options, target):
    filter_ = tensors[inputs[1]]
    if tensors[inputs[0]][0] == "FLOAT32" or not _tflm_quantized(filter_) or filter_[3] < 2:
        return []
    return [4 * filter_[3]] * 2


def _tflm_softmax(tensors, inputs, outputs, options, target):
    # Two int16 lookup tables of 513 entries for int16 input.
    return [2 * 513] * 2 if tensors[inputs[0]][0] == "INT16" else []


def _tflm_log_softmax(tensors, inputs, outputs, options, target):
    return [_tflm_layout("LogSoftmaxOpData", target)[0]] if tensors[inputs[0]][0] == "INT8" else []


def _tflm_cumsum(tensors, inputs, outputs, options, target):
    return [_tflm_layout("Cumsum OpData", target)[0]] if tensors[outputs[0]][0] == "INT8" else []


def _tflm_embedding_lookup(tensors, inputs, outputs, options, target):
    return [_tflm_dims(len(tensors[outputs[0]][1])),
            _tflm_layout("EmbeddingLookup OpData", target)[0]]


def _tflm_reshape(tensors, inputs, outputs, options, target):
    shape = tensors[outputs[0]][1]
    return [_tflm_dims(len(shape))] if -1 in shape else []


def _tflm_batch_matmul(tensors, inputs, outputs, options, target):
    lhs, rhs = tensors[inputs[0]], tensors[inputs[1]]
    sizes = [_tflm_layout("QuantizationOpDataBatchMatmul", target)[0]] if lhs[0] in (
        "INT8", "INT16") else []
    for (dtype, shape, *_), transposed in ((lhs, options.get("adj_x", False)),
                                          (rhs, not options.get("adj_y", False))):
        if transposed:
            sizes += [_tflm_layout("TfLiteEvalTensor", target)[0], _tflm_dims(len(shape)),
                      math.prod(shape) * _ELEMENT_BYTES[dtype]]
    return sizes + [_tflm_dims(len(tensors[outputs[0]][1]))]


# Per kind: the parser's builtin data record, the init record, and the prepare allocations.
TFLM_KERNELS = {
    "ABS": (None, "OpDataAbsRsqrt", None),
    "ADD": ("TfLiteAddParams", "OpDataAdd", None),
    "ARG_MAX": ("TfLiteArgMaxParams", None, None),
    "ARG_MIN": ("TfLiteArgMinParams", None, None),
    "AVERAGE_POOL_2D": ("TfLitePoolParams", "OpDataPooling", None),
    "BATCH_MATMUL": ("TfLiteBatchMatMulParams", "OpDataBatchMatmul", _tflm_batch_matmul),
    "BATCH_TO_SPACE_ND": (None, None, None),
    "BROADCAST_ARGS": (None, None, None),
    "BROADCAST_TO": (None, None, None),
    "CAST": ("TfLiteCastParams", None, None),
    "CEIL": (None, None, None),
    "CONCATENATION": ("TfLiteConcatenationParams", "Concatenation OpData", None),
    "CONV_2D": ("TfLiteConvParams", "OpDataConv", _tflm_conv(0)),
    "COS": (None, None, None),
    "CUMSUM": ("TfLiteCumsumParams", None, _tflm_cumsum),
    "DEPTH_TO_SPACE": ("TfLiteDepthToSpaceParams", None, _tflm_output_dims),
    "DEPTHWISE_CONV_2D": ("TfLiteDepthwiseConvParams", "OpDataConv", _tflm_conv(3)),
    "DEQUANTIZE": (None, "DequantizeOpData", None),
    "DIV": ("TfLiteDivParams", "OpDataDiv", None),
    "DYNAMIC_UPDATE_SLICE": (None, None, None),
    "ELU": (None, "Elu OpData", None),
    "EMBEDDING_LOOKUP": (None, None, _tflm_embedding_lookup),
    "EQUAL": (None, "Comparison OpData", None),
    "EXP": (None, None, None),
    "EXPAND_DIMS": (None, None, None),
    "FILL": (None, None, None),
    "FLOOR": (None, None, None),
    "FLOOR_DIV": (None, None, None),
    "FLOOR_MOD": (None, None, None),
    "FULLY_CONNECTED": ("TfLiteFullyConnectedParams", "OpDataFullyConnected",
                        _tflm_fully_connected),
    "GATHER": ("TfLiteGatherParams", None, _tflm_output_dims),
    "GATHER_ND": (None, None, _tflm_output_dims),
    "GREATER": (None, "Comparison OpData", None),
    "GREATER_EQUAL": (None, "Comparison OpData", None),
    "HARD_SWISH": (None, "HardSwishParams", None),
    "L2_NORMALIZATION": ("TfLiteL2NormParams", "L2NormalizationParams", None),
    "L2_POOL_2D": ("TfLitePoolParams", None, _tflm_output_dims),
    "LEAKY_RELU": ("TfLiteLeakyReluParams", "LeakyReluOpData", None),
    "LESS": (None, "Comparison OpData", None),
    "LESS_EQUAL": (None, "Comparison OpData", None),
    "LOG": (None, None, None),
    "LOGICAL_AND": (None, None, None),
    "LOGICAL_NOT": (None, None, None),
    "LOGICAL_OR": (None, None, None),
    "LOGISTIC": (None, "OpDataLogistic", None),
    "LOG_SOFTMAX": (None, None, _tflm_log_softmax),
    "MAXIMUM": (None, None, None),
    "MAX_POOL_2D": ("TfLitePoolParams", "OpDataPooling", None),
    "MEAN": ("TfLiteReducerParams", "OpDataReduce", None),
    "MINIMUM": (None, None, None),
    "MUL": ("TfLiteMulParams", "OpDataMul", None),
    "NEG": (None, None, None),
    "NOT_EQUAL": (None, "Comparison OpData", None),
    "PACK": ("TfLitePackParams", None, None),
    "PAD": (None, "Pad OpData", None),
    "PADV2": (None, "Pad OpData", None),
    "PRELU": (None, "PreluParams", None),
    "QUANTIZE": (None, "OpDataQuantizeReference", None),
    "REDUCE_ALL": ("TfLiteReducerParams", "OpDataReduce", None),
    "REDUCE_MAX": ("TfLiteReducerParams", "OpDataReduce", None),
    "REDUCE_MIN": ("TfLiteReducerParams", "OpDataReduce", None),
    "RELU": (None, "ReluOpData", None),
    "RELU6": (None, "Relu6OpData", None),
    "RESHAPE": ("TfLiteReshapeParams", None, _tflm_reshape),
    "RESIZE_BILINEAR": ("TfLiteResizeBilinearParams", None, None),
    "RESIZE_NEAREST_NEIGHBOR": ("TfLiteResizeNearestNeighborParams", None, None),
    "REVERSE_V2": (None, None, None),
    "ROUND": (None, None, None),
    "RSQRT": (None, "OpDataAbsRsqrt", None),
    "SELECT_V2": (None, "Select OpData", None),
    "SHAPE": ("TfLiteShapeParams", None, None),
    "SIN": (None, None, None),
    "SLICE": (None, None, None),
    "SOFTMAX": ("TfLiteSoftmaxParams", "SoftmaxParams", _tflm_softmax),
    "SPACE_TO_BATCH_ND": (None, "SpaceToBatchParams", None),
    "SPACE_TO_DEPTH": ("TfLiteSpaceToDepthParams", None, _tflm_output_dims),
    "SPLIT": ("TfLiteSplitParams", None, None),
    "SPLIT_V": ("TfLiteSplitVParams", None, None),
    "SQRT": (None, None, None),
    "SQUARE": (None, None, None),
    "SQUARED_DIFFERENCE": (None, "SquaredDifference OpData", None),
    "SQUEEZE": ("TfLiteSqueezeParams", None, None),
    "STRIDED_SLICE": ("TfLiteStridedSliceParams", "StridedSliceParams", None),
    "SUB": ("TfLiteSubParams", "OpDataSub", None),
    "SUM": ("TfLiteReducerParams", "OpDataReduce", None),
    "TANH": (None, "Tanh OpData", None),
    "TRANSPOSE": (None, None, None),
    "UNPACK": ("TfLiteUnpackParams", None, None),
    "ZEROS_LIKE": (None, None, None),
}


def tflm_persistent(tensors: list, operators: list, inputs, outputs, target: TflmTarget,
                    compression: bool = False, recording: bool = False) -> int | None:
    """TFLite Micro's persistent (tail) bytes for a one-subgraph model on its
    reference kernels, as a MicroInterpreter built for `target` allocates
    them: `tensors` holds each tensor's (storage type, shape, variable, scale
    count, zero point count) and `operators` each operator's (kind, inputs,
    outputs, options), options holding adj_x and adj_y of BATCH_MATMUL.
    `compression` is a build with USE_TFLM_COMPRESSION, `recording` a
    RecordingMicroAllocator in place of MicroAllocator. The arena end is
    16-byte aligned. None when an operator kind or a tensor's storage type is
    not modeled."""
    if any(kind not in TFLM_KERNELS for kind, *_ in operators) or any(
            t[0] not in _ELEMENT_BYTES for t in tensors):
        return None

    def record(name, count=1):
        size, align = _tflm_layout(name, target, compression)
        return size * count, align

    recording_prefix = "Recording" if recording else ""
    allocations = [record(f"{recording_prefix}SingleArenaBufferAllocator"),
                   record("GreedyMemoryPlanner"), record(f"{recording_prefix}MicroAllocator"),
                   record("MicroBuiltinDataAllocator"), record("SubgraphAllocations"),
                   record("TfLiteEvalTensor", len(tensors)),
                   record("NodeAndRegistration", len(operators))]
    allocations += [record(TFLM_KERNELS[kind][0]) for kind, *_ in operators
                    if TFLM_KERNELS[kind][0]]
    allocations += [(record(TFLM_KERNELS[kind][1])[0], 16) for kind, *_ in operators
                    if TFLM_KERNELS[kind][1]]
    for kind, ins, outs, options in operators:
        prepare = TFLM_KERNELS[kind][2]
        if prepare is not None:
            allocations += [(size, 16) for size in prepare(tensors, ins, outs, options, target)]
    requests = sum(len(_tflm_scratch(kind, tensors, ins, outs)) for kind, ins, outs, _ in operators)
    if requests:
        allocations.append(record("ScratchBufferHandle", requests))
    allocations += [(math.prod(t[1]) * _ELEMENT_BYTES[t[0]], 16) for t in tensors if t[2]]
    for boundary in (inputs, outputs):
        allocations.append((target.pointer * len(boundary), 16))
        for index in boundary:
            allocations.append(record("TfLiteTensor"))
            if _tflm_quantized(tensors[index]):
                allocations += [record("TfLiteAffineQuantization"),
                                (_tflm_dims(tensors[index][3]), 4)]
    used = 0
    for size, align in allocations:
        used = -(-(used + size) // align) * align
    return used
