"""TFLite Micro's memory on its reference kernels: the arena head its greedy
planner places for one subgraph, and the persistent allocations of its arena
tail, record by record, transcribed from the TFLite Micro sources."""

import bisect
import functools
import math
from typing import NamedTuple

_ELEMENT_BYTES = {"FLOAT32": 4, "INT32": 4, "UINT32": 4, "INT8": 1, "UINT8": 1, "BOOL": 1,
                  "INT16": 2, "UINT16": 2, "FLOAT16": 2, "INT64": 8, "UINT64": 8, "FLOAT64": 8}
# TfLiteTypeSizeOf: RESOURCE takes 4 bytes, INT4 one per element. STRING, VARIANT,
# INT2 and UINT4 have no size, so TFLite Micro fails to allocate them; they count 0.
_TFLM_TYPE_BYTES = {**_ELEMENT_BYTES, "COMPLEX64": 8, "COMPLEX128": 16, "RESOURCE": 4, "INT4": 1,
                    "BFLOAT16": 2, "FLOAT8_E4M3FN": 1, "FLOAT8_E5M2": 1}
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
    "AllocationInfo": "z p i i i c",
    "ScratchBufferRequest": "z i i",
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
    "TfLiteUnidirectionalSequenceLSTMParams": "e i i c c c",
    "OpDataLSTM": "(c i4) (i h i) e ((i9 ccc)2)4 ((c i14 q2 h2 i5)3) i4",
    "TfLiteSVDFParams": "i e c",
    "OpDataSvdf": "i9",
    "TfLiteTransposeConvParams": "e i i e e",
    "TransposeConv OpData": "(c (hhhh) h4 i9) i i pp",
    "TfLiteMirrorPaddingParams": "e",
    "OpDataMirrorPad": "i5",
    "AddN OpData": "i10",
    "TfLiteIfParams": "ii",
    "TfLiteWhileParams": "ii",
    "TfLiteCallOnceParams": "i",
    "TfLiteVarHandleParams": "pp",
    "If OpData": "ii",
    "While OpData": "ii",
    "CallOnce OpData": "i c",
    "VarHandle OpData": "i",
    "AssignVariable OpData": "",
    "MicroResourceVariables": "p i i",
    "MicroResourceVariable": "ppp z c",
    "CompressionTensorData": "c p",
    # Each bit-field counted as a byte of its own: an upper bound.
    "LookupTableData": "p c c4",
    "DetectionPostprocess OpData": "i6 c (i4) i11 (ii)3",
    "DecodeStateLut": "p pp p z z z p c c c",
    "DecodeStatePrune": "p pp p z z z p p c c",
    "DecodeStateHuffman": "p pp p z p c c",
    "OpDataCircularBuffer": "ii",
    "CircularBuffer": "z z z i i p",
    "TFLMSignalFrontendDelayParams": "i i i p p",
    "TFLMSignalEnergyParams": "ii",
    "TFLMSignalFilterBankParams": "(i ppppp ii) p",
    "TFLMSignalLogParams": "ii",
    "TFLMSignalSpectralSubtractionParams": "(i8 c) p z",
    "TFLMSignalFramerParams": "i4 c p p",
    "TfLiteAudioFrontendIrfftParams": "i4 e p",
    "IrfftFloatState": "i p",
    "TFLMSignalOverlapAddParams": "i4 e p",
    "TfLitePcanParams": "i",
    "TfLiteAudioFrontendRfftParams": "i4 e p i p",
    "TFLMSignalStackerParams": "i4 z z c p p",
    "TFLMSignalWindowParams": "ii",
    "kiss_fftr_state": "ppp",
    "kiss_fft_state": "ii i64 (ii)",
    "kiss_fft_state int16": "ii i64 (hh)",
}
# kMaxScratchBuffersPerOp: the scratch buffer requests one operator may make.
_TFLM_SCRATCH_PER_OP = 12
# GreedyMemoryPlanner::per_buffer_size(): a BufferRequirements, two sort keys, a
# ListEntry and an offset.
_TFLM_PLANNER_BUFFER = 16 + 4 + 4 + 12 + 4
# Members a build with USE_TFLM_COMPRESSION appends to a record.
_TFLM_COMPRESSION = {"SubgraphAllocations": "p", "OpDataConv": "ii", "OpDataFullyConnected": "ii",
                     "Concatenation OpData": "i10", "RecordingMicroAllocator": "(zzz)",
                     "TransposeConv OpData": "ii", "AssignVariable OpData": "i"}


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
                      math.prod(shape) * _TFLM_TYPE_BYTES.get(dtype, 0)]
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


def tflm_scratch_requests(tensors: list, operators: list) -> int:
    """The scratch buffers TFLite Micro's reference kernels request for
    `operators`, each (kind, inputs, outputs, ...), over `tensors`, each
    (storage type, shape, ...)."""
    return sum(len(_tflm_scratch(kind, tensors, ins, outs)) for kind, ins, outs, *_ in operators)


def tflm_planning(tensors: int, scratch: int, target: TflmTarget) -> int:
    """An upper bound of the temporary bytes TFLite Micro allocates past the
    arena head while it plans a subgraph of `tensors` tensors and `scratch`
    scratch buffers: an AllocationInfo and the GreedyMemoryPlanner's working
    arrays for each tensor and scratch buffer, the scratch buffer requests with
    room for one more operator's, and 64 bytes for aligning the arena start and
    each of these blocks."""
    info = _tflm_layout("AllocationInfo", target)[0]
    request = _tflm_layout("ScratchBufferRequest", target)[0]
    return ((info + _TFLM_PLANNER_BUFFER) * (tensors + scratch)
            + request * (scratch + _TFLM_SCRATCH_PER_OP) + 64)


def tflm_preparing(temps: int, scratch: int, target: TflmTarget) -> int:
    """An upper bound of the bytes TFLite Micro takes past the tail while a
    kernel prepares: the scratch buffer requests of `scratch` buffers with
    room for one more operator's, the `temps` bytes of temporary tensors the
    kernel holds, and 64 bytes for aligning the arena start and end and the
    request block. The head is planned only after every kernel prepared."""
    return _tflm_layout("ScratchBufferRequest", target)[0] * (scratch + _TFLM_SCRATCH_PER_OP) + (
        temps + 64)


def tflm_persistent(tensors: list, operators: list, inputs, outputs, target: TflmTarget,
                    compression: bool = False, recording: bool = False,
                    prepared: bool = False) -> int | None:
    """TFLite Micro's persistent (tail) bytes for a one-subgraph model on its
    reference kernels, as a MicroInterpreter built for `target` allocates
    them: `tensors` holds each tensor's (storage type, shape, variable, scale
    count, zero point count) and `operators` each operator's (kind, inputs,
    outputs, options), options holding adj_x and adj_y of BATCH_MATMUL.
    `compression` is a build with USE_TFLM_COMPRESSION, `recording` a
    RecordingMicroAllocator in place of MicroAllocator. The arena end is
    16-byte aligned. With `prepared`, the tail once every kernel prepared,
    before the scratch buffer handles, the variables and the graph boundary.
    None when an operator kind or a tensor's storage type is not modeled."""
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
    requests = tflm_scratch_requests(tensors, operators)
    if prepared:
        requests, tensors, inputs, outputs = 0, [], [], []
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


# Kernels whose Prepare allocates a temporary TfLiteTensor for an operand twice:
# a helper it calls takes the operands again.
_TFLM_TEMPS_TWICE = ("CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "MEAN", "SUM", "REDUCE_MAX",
                     "REDUCE_MIN", "REDUCE_ALL", "TANH", "CONCATENATION")
# Operand slots each kernel takes a decompression scratch buffer for, None for every input.
_TFLM_DECOMPRESSED = {"CONV_2D": (1, 2), "DEPTHWISE_CONV_2D": (1, 2), "FULLY_CONNECTED": (1, 2),
                      "TRANSPOSE_CONV": (1, 3), "ASSIGN_VARIABLE": (1,), "CONCATENATION": None}
# Kernels that request a scratch buffer of the filter for int4 weights.
_TFLM_INT4_FILTERS = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")


def _tflm_elements(shape) -> int:
    """Elements of a shape; a negative count is 0, since TFLite Micro wraps it
    to a planner size of at most 0 or a request no arena satisfies."""
    return max(0, math.prod(shape))


def _tflm_bytes(tensor) -> int:
    return _tflm_elements(tensor[1]) * _TFLM_TYPE_BYTES.get(tensor[0], 0)


def _c_div(a: int, b: int) -> int:
    """C integer division, truncating toward zero."""
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def _tflm_out_size(padding: int, size: int, filter_: int, stride: int) -> int:
    """ComputeOutSize for padding 0 SAME or 1 VALID: 0 where its checks fail."""
    if padding not in (0, 1) or size < 0 or filter_ <= 0 or stride <= 0:
        return 0
    return max(0, (size + stride - 1) // stride if padding == 0 else
               _c_div(size + stride - filter_, stride))


def tflm_written_shape(kind: str, shapes: list, inputs, outputs, options: dict) -> list | None:
    """The output shape a kernel that rewrites its output's dims in Prepare
    writes, from the input shapes as Prepare sees them; None for any other
    kernel and where the kernel's checks fail first. `options` holds
    block_size, axis, batch_dims, adj_x, adj_y, padding, stride_w, stride_h,
    filter_w and filter_h where the kind has them."""
    if not outputs:
        return None
    out = list(shapes[outputs[0]])
    ins = [shapes[i] if i >= 0 else None for i in inputs]
    if kind == "RESHAPE":
        if len(ins) == 1 and out == [0]:
            out = []
        known = math.prod(d for d in out if d != -1)
        if out.count(-1) != 1 or not known:
            return None
        out[out.index(-1)] = _c_div(math.prod(ins[0]), known)
        return out
    if kind in ("DEPTH_TO_SPACE", "SPACE_TO_DEPTH"):
        block = options.get("block_size", 0)
        if len(ins[0]) != 4 or len(out) != 4 or block <= 0:
            return None
        n, h, w, c = ins[0]
        if kind == "DEPTH_TO_SPACE":
            return [n, h * block, w * block, _c_div(_c_div(c, block), block)]
        return [n, _c_div(h, block), _c_div(w, block), c * block * block]
    if kind == "L2_POOL_2D":
        if len(ins[0]) != 4 or len(out) != 4:
            return None
        n, h, w, c = ins[0]
        padding = options.get("padding", 0)
        return [n,
                _tflm_out_size(padding, h, options.get("filter_h", 0), options.get("stride_h", 0)),
                _tflm_out_size(padding, w, options.get("filter_w", 0), options.get("stride_w", 0)),
                c]
    if kind == "GATHER":
        if len(ins) < 2 or ins[1] is None:
            return None
        params, coords = ins[0], ins[1]
        axis, batch = options.get("axis", 0), options.get("batch_dims", 0)
        axis += len(params) if axis < 0 else 0
        batch += len(coords) if batch < 0 else 0
        if not (0 <= axis < len(params) and 0 <= batch <= axis and batch < len(params)
                and batch <= len(coords)):
            return None
        return [*params[:axis], *coords[batch:], *params[axis + 1:]]
    if kind == "GATHER_ND":
        if len(ins) < 2 or ins[1] is None or not ins[0] or not ins[1]:
            return None
        params, indices = ins[0], ins[1]
        if indices[-1] > len(params):
            return None
        return [*indices[:-1], *params[max(indices[-1], 0):]]
    if kind == "EMBEDDING_LOOKUP":
        if len(ins) < 2 or not ins[0] or len(ins[1]) < 2 or len(out) < len(ins[1]):
            return None
        return [ins[0][0], *ins[1][1:], *out[len(ins[1]):]]
    if kind == "BATCH_MATMUL":
        if len(ins) < 2 or len(ins[0]) < 2 or len(ins[1]) < 2:
            return None
        rank = max(len(ins[0]), len(ins[1]))
        lhs, rhs = ([1] * (rank - len(s)) + list(s) for s in ins[:2])
        batch = [b if a == 1 and a != b else a for a, b in zip(lhs[:-2], rhs[:-2])]
        return [*batch, lhs[-1] if options.get("adj_x") else lhs[-2],
                rhs[-2] if options.get("adj_y") else rhs[-1]]
    return None


def tflm_temps(kind: str, tensors: list, inputs, outputs, target: TflmTarget) -> int:
    """An upper bound of the temporary bytes TFLite Micro's kernel for `kind`
    holds at once while it prepares: a TfLiteTensor for each operand it was
    given, twice for kernels whose helpers take them again, and for a
    quantized operand its TfLiteAffineQuantization and zero point array.
    They start pointer-aligned and none needs more alignment, so each
    allocation rounded up to a pointer bounds its padding. They are released
    after each operator's Prepare."""
    def up(size):
        return -(-size // target.pointer) * target.pointer

    total = 0
    for index in [*inputs, *outputs]:
        if index < 0:
            continue
        total += up(_tflm_layout("TfLiteTensor", target)[0])
        if _tflm_quantized(tensors[index]):
            total += (up(_tflm_layout("TfLiteAffineQuantization", target)[0])
                      + up(_tflm_dims(tensors[index][3])))
    return total * (2 if kind in _TFLM_TEMPS_TWICE else 1)


def _shape(tensors, inputs, slot):
    return tensors[inputs[slot]][1] if slot < len(inputs) and inputs[slot] >= 0 else None


def _lstm_prepare(tensors, inputs, outputs, options, target):
    # Two per-channel multiplier arrays for each per-channel weight of the integer path.
    return [4 * tensors[inputs[k]][3] for k in range(1, 9)
            if k < len(inputs) and inputs[k] >= 0 and tensors[inputs[k]][3] > 1 for _ in range(2)]


def _lstm_scratch(tensors, inputs, outputs, options, target):
    if _shape(tensors, inputs, 18) is None or _shape(tensors, inputs, 19) is None:
        return []
    cell = tensors[inputs[19]][0]
    return [_tflm_elements(tensors[inputs[18]][1]) * _TFLM_TYPE_BYTES.get(cell, 0)] * 4


def _svdf_scratch(tensors, inputs, outputs, options, target):
    batch_shape, feature_shape = _shape(tensors, inputs, 0), _shape(tensors, inputs, 1)
    output = _shape(tensors, outputs, 0)
    if not batch_shape or not feature_shape or output is None or len(output) < 2:
        return []
    batch, filters, units = batch_shape[0], feature_shape[0], output[1]
    return [max(0, 4 * batch * filters), max(0, 4 * batch * units)]


def _transpose_conv_prepare(tensors, inputs, outputs, options, target):
    filter_ = _shape(tensors, inputs, 1)
    return [4 * filter_[0]] * 2 if filter_ else []


def _transpose_conv_scratch(tensors, inputs, outputs, options, target):
    if _shape(tensors, inputs, 2) is None or _shape(tensors, outputs, 0) is None:
        return []
    kind, size = tensors[inputs[2]][0], _tflm_elements(tensors[outputs[0]][1])
    sizes = {"INT8": [4 * size], "INT16": [8 * size]}.get(kind, [])
    bias = _shape(tensors, inputs, 3)
    if kind != "FLOAT32" and bias is not None and tensors[inputs[3]][0] == "INT16":
        sizes.append(8 * _tflm_elements(bias))
    return sizes


def _mirror_pad_scratch(tensors, inputs, outputs, options, target):
    shape = _shape(tensors, inputs, 0)
    return [4 * len(shape)] * 2 if shape is not None else []


def _add_n_prepare(tensors, inputs, outputs, options, target):
    if _shape(tensors, outputs, 0) is None or tensors[outputs[0]][0] == "FLOAT32":
        return []
    return [_tflm_layout("AddN OpData", target)[0]]


def _add_n_scratch(tensors, inputs, outputs, options, target):
    return [target.pointer * len(inputs)]


def _assign_variable_prepare(tensors, inputs, outputs, options, target):
    # The resource variable's buffer, allocated once per variable, of the value's declared bytes.
    if _shape(tensors, inputs, 1) is None:
        return []
    return [_tflm_bytes((tensors[inputs[1]][0], tensors[inputs[1]][6]))]


def _custom(options) -> bytes:
    return options.get("custom") or b""


def _detection_scratch(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_int

    data = _custom(options)
    detections = flexbuffer_int(data, "max_detections")
    classes = flexbuffer_int(data, "num_classes")
    boxes_shape, scores_shape = _shape(tensors, inputs, 0), _shape(tensors, inputs, 1)
    if (boxes_shape is None or scores_shape is None or len(boxes_shape) != 3
            or len(scores_shape) != 3):
        return []
    boxes = boxes_shape[1]
    sizes = [boxes, 16 * boxes, 4 * scores_shape[1] * scores_shape[2], 4 * boxes, 4 * boxes,
             4 * detections * boxes, 4 * detections * boxes, 4 * boxes, 4 * detections * boxes,
             4 * max(classes, detections) * boxes, 4 * min(boxes, detections) * boxes]
    return [max(0, size) for size in sizes]


def _decode_prepare(tensors, inputs, outputs, options, target):
    # The largest DecodeState of the three types each output may take, and
    # Prune's zero point copy for a per-channel output.
    state = max(_tflm_layout(name, target)[0]
                for name in ("DecodeStateLut", "DecodeStatePrune", "DecodeStateHuffman"))
    sizes = [target.pointer * len(outputs)]
    for index in outputs:
        sizes.append(state)
        if _tflm_quantized(tensors[index]) and tensors[index][3] > 1:
            sizes.append(tensors[index][3])
    return sizes


def _circular_state(capacity: int, target: TflmTarget) -> int:
    """CircularBufferGetNeededMemory: the record and two int16 per element."""
    return max(0, _tflm_layout("CircularBuffer", target)[0] + 4 * capacity)


def _outer(shape, inner: int) -> int:
    """FlatSize over the given innermost elements, 0 where Prepare would divide by 0."""
    return _c_div(math.prod(shape), inner) if inner else 0


def _delay_prepare(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    delay = flexbuffer_slots(_custom(options), 1)[0]
    shape = _shape(tensors, inputs, 0) or []
    frame = shape[-1] if shape else 0
    outer = max(0, _outer(shape, frame))
    return [target.pointer * outer] * 2 + [_circular_state(frame + delay, target)] * outer


def _framer_prepare(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    size, step = flexbuffer_slots(_custom(options), 2)
    shape = _shape(tensors, inputs, 0) or []
    outer = max(0, _outer(shape, shape[-1] if shape else 0))
    capacity = _c_div(size + step - 1, step) * step if step else 0
    return [target.pointer * outer] * 2 + [_circular_state(capacity, target)] * outer


def _filter_bank_init(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    return [max(0, 8 * (flexbuffer_slots(_custom(options), 1)[0] + 1))]


def _spectral_subtraction_init(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    return [max(0, 4 * flexbuffer_slots(_custom(options), 5)[4])]


def _stacker_init(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    channels, left, right = flexbuffer_slots(_custom(options), 3)
    return [_circular_state(channels * (left + right + 1), target)]


# TensorType codes of the signal kernels' T option and the bytes of an element.
_TFLM_SIGNAL_TYPES = {7: 2, 2: 4, 0: 4}


def _fft_state(length: int, element: int, target: TflmTarget) -> int:
    """kiss_fftr_alloc's memory for a real FFT of `length` whose complex
    element pairs two scalars of `element` bytes; 0 for an odd length."""
    if length % 2:
        return 0
    half = _c_div(length, 2)
    state = _tflm_layout("kiss_fft_state int16" if element == 2 else "kiss_fft_state", target)[0]
    pair = 2 * element
    return max(0, _tflm_layout("kiss_fftr_state", target)[0] + state + pair * (half - 1)
               + pair * _c_div(half * 3, 2))


def _fft_type(options) -> int:
    from tigris.frontends.flatbuffer import flexbuffer_int

    return flexbuffer_int(_custom(options), "T")


def _rfft_init(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    kind = _fft_type(options)
    if kind not in _TFLM_SIGNAL_TYPES:
        return []
    length = flexbuffer_slots(_custom(options), 2)[1]
    return [_tflm_layout("TfLiteAudioFrontendRfftParams", target)[0],
            _fft_state(length, _TFLM_SIGNAL_TYPES[kind], target)]


def _rfft_scratch(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    kind = _fft_type(options)
    if kind not in _TFLM_SIGNAL_TYPES:
        return []
    return [max(0, flexbuffer_slots(_custom(options), 2)[1] * _TFLM_SIGNAL_TYPES[kind])]


def _irfft_init(tensors, inputs, outputs, options, target):
    from tigris.frontends.flatbuffer import flexbuffer_slots

    kind = _fft_type(options)
    if kind not in _TFLM_SIGNAL_TYPES:
        return []
    state = _fft_state(flexbuffer_slots(_custom(options), 2)[1], _TFLM_SIGNAL_TYPES[kind], target)
    if kind == 0:
        state += _tflm_layout("IrfftFloatState", target)[0]
    return [_tflm_layout("TfLiteAudioFrontendIrfftParams", target)[0], state]


def _overlap_add_prepare(tensors, inputs, outputs, options, target):
    kind = _fft_type(options)
    if kind not in (0, 7):
        return []
    shape = _shape(tensors, inputs, 0) or []
    frame = shape[-1] if shape else 0
    outer = max(0, _outer(shape, frame * shape[-2]) if len(shape) > 1 else 0)
    return ([_tflm_layout("TFLMSignalOverlapAddParams", target)[0], target.pointer * outer]
            + [max(0, frame * _TFLM_SIGNAL_TYPES[kind])] * outer)


# Per kind TFLite Micro's tflite-micro package registers beyond TFLM_KERNELS,
# builtins by name and custom operators by custom code: the parser's builtin
# data record, the init record, the other persistent allocations of Init and
# Prepare, and the scratch buffers Prepare requests. Each function takes the
# tensors as Prepare sees them: (storage type, shape its dims hold, variable,
# scale count, zero point count, constant, declared shape its bytes come from).
TFLM_BOUND_KERNELS = {
    "UNIDIRECTIONAL_SEQUENCE_LSTM": ("TfLiteUnidirectionalSequenceLSTMParams", "OpDataLSTM",
                                     _lstm_prepare, _lstm_scratch),
    "SVDF": ("TfLiteSVDFParams", "OpDataSvdf", None, _svdf_scratch),
    "TRANSPOSE_CONV": ("TfLiteTransposeConvParams", "TransposeConv OpData",
                       _transpose_conv_prepare, _transpose_conv_scratch),
    "MIRROR_PAD": ("TfLiteMirrorPaddingParams", "OpDataMirrorPad", None, _mirror_pad_scratch),
    "ADD_N": (None, None, _add_n_prepare, _add_n_scratch),
    "IF": ("TfLiteIfParams", "If OpData", None, None),
    "WHILE": ("TfLiteWhileParams", "While OpData", None, None),
    "CALL_ONCE": ("TfLiteCallOnceParams", "CallOnce OpData", None, None),
    "VAR_HANDLE": ("TfLiteVarHandleParams", "VarHandle OpData", None, None),
    "READ_VARIABLE": (None, None, None, None),
    "ASSIGN_VARIABLE": (None, "AssignVariable OpData", _assign_variable_prepare, None),
    "TFLite_Detection_PostProcess": (None, "DetectionPostprocess OpData", None,
                                     _detection_scratch),
    "TFLM_DECODE": (None, None, _decode_prepare, None),
    "BasicClassifier": (None, None, None, None),
    "CIRCULAR_BUFFER": (None, "OpDataCircularBuffer", None, None),
    "SignalDelay": (None, "TFLMSignalFrontendDelayParams", _delay_prepare, None),
    "SignalEnergy": (None, "TFLMSignalEnergyParams", None, None),
    "SignalFftAutoScale": (None, None, None, None),
    "SignalFilterBank": (None, "TFLMSignalFilterBankParams", _filter_bank_init, None),
    "SignalFilterBankLog": (None, "TFLMSignalLogParams", None, None),
    "SignalFilterBankSpectralSubtraction": (None, "TFLMSignalSpectralSubtractionParams",
                                            _spectral_subtraction_init, None),
    "SignalFilterBankSquareRoot": (None, None, None, None),
    "SignalFramer": (None, "TFLMSignalFramerParams", _framer_prepare, None),
    "SignalIrfft": (None, None, _irfft_init, None),
    "SignalOverlapAdd": (None, None, _overlap_add_prepare, None),
    "SignalPCAN": (None, "TfLitePcanParams", None, None),
    "SignalRfft": (None, None, _rfft_init, _rfft_scratch),
    "SignalStacker": (None, "TFLMSignalStackerParams", _stacker_init, None),
    "SignalWindow": (None, "TFLMSignalWindowParams", None, None),
}


def _tflm_bound_scratch(kind, tensors, inputs, outputs, options, target, compressed) -> list:
    """The scratch buffers the kernel requests: a reduction's, a bound kind's,
    an int4 filter's copy, and in a model with compressed tensors a
    decompression buffer for each operand the kernel may decompress."""
    sizes = list(_tflm_scratch(kind, tensors, inputs, outputs))
    if kind in TFLM_BOUND_KERNELS and TFLM_BOUND_KERNELS[kind][3] is not None:
        sizes += TFLM_BOUND_KERNELS[kind][3](tensors, inputs, outputs, options, target)
    if kind in _TFLM_INT4_FILTERS and _shape(tensors, inputs, 1) is not None and (
            tensors[inputs[1]][0] == "INT4"):
        sizes.append(_tflm_elements(tensors[inputs[1]][1]))
    if compressed and kind in _TFLM_DECOMPRESSED:
        slots = _TFLM_DECOMPRESSED[kind] or range(len(inputs))
        sizes += [_tflm_bytes(tensors[inputs[k]]) for k in slots
                  if _shape(tensors, inputs, k) is not None]
    return sizes


def tflm_bound(subgraphs: list, target: TflmTarget, compression: bool = False,
               recording: bool = False, offline: list | None = None,
               compressed: bool = False) -> int:
    """An upper bound of the arena TFLite Micro needs for a model on the
    kernels of the tflite-micro package. `subgraphs` holds each subgraph's
    (tensors, operators, inputs, outputs): each tensor (storage type, shape,
    variable, scale count, zero point count, constant), each operator (kind,
    inputs, outputs, options), kind a builtin name or a custom code, options
    what tflm_written_shape and BATCH_MATMUL read plus "custom", the custom
    options. `offline` holds the offline planned offset of every tensor of
    every subgraph, -1 for one planned online; `compressed` marks a model with
    compressed tensors.

    The head holds every placed tensor of every subgraph at once and every
    scratch buffer, 16-byte aligned, after the offline planned buffers: the
    greedy planner puts each buffer at 0 or at the end of one placed before.
    Planning takes tflm_planning's room for all tensors and scratch buffers
    and an offset per subgraph on top of it. The tail holds every persistent
    allocation, each with its alignment padding. Before planning, while one
    kernel prepares, the tail holds only what Init and Prepare allocated and
    tflm_preparing's bytes sit past it, so the larger of the two counts. A
    kind the package does not register adds nothing: TFLite Micro stops when
    it looks it up. Raises ValueError for a kernel that requests scratch
    buffers in a subgraph no IF, WHILE or CALL_ONCE reaches from the first:
    TFLite Micro plans those from memory it never sets, so no arena is
    enough. The options of IF, WHILE and CALL_ONCE hold "subgraphs", the
    indices they invoke."""
    pointer = target.pointer
    reached, pending = {0}, [0]
    while pending:
        for kind, _, _, options in subgraphs[pending.pop()][1]:
            if kind in ("IF", "WHILE", "CALL_ONCE"):
                for index in options.get("subgraphs", ()):
                    if 0 <= index < len(subgraphs) and index not in reached:
                        reached.add(index)
                        pending.append(index)

    def record(name, count=1):
        size, align = _tflm_layout(name, target, compression)
        return [(size * count, align)] if size * count else []

    prefix = "Recording" if recording else ""
    tail = [*record(f"{prefix}SingleArenaBufferAllocator"), *record("GreedyMemoryPlanner"),
            *record(f"{prefix}MicroAllocator"), *record("MicroBuiltinDataAllocator"),
            *record("SubgraphAllocations", len(subgraphs))]
    handles = sum(kind == "VAR_HANDLE" for _, operators, *_ in subgraphs for kind, *_ in operators)
    if handles:
        # The package creates one per distinct shared name, at most one per VAR_HANDLE.
        tail += [(_tflm_layout("MicroResourceVariables", target)[0], 16),
                 (_tflm_layout("MicroResourceVariable", target)[0] * handles, 16)]
    head = scratch_count = temps = offline_end = tensor_total = 0
    variables = []
    for number, (tensors, operators, _, _) in enumerate(subgraphs):
        tail += [*record("TfLiteEvalTensor", len(tensors)),
                 *record("NodeAndRegistration", len(operators))]
        if compressed:
            tail += [*record("CompressionTensorData", len(tensors)),
                     *record("LookupTableData", len(tensors)), (pointer * len(tensors), pointer)]
        shapes = [list(t[1]) for t in tensors]
        for kind, inputs, outputs, options in operators:
            entry = TFLM_KERNELS.get(kind) or TFLM_BOUND_KERNELS.get(kind)
            if entry is None:
                continue
            view = [(t[0], shape, *t[2:6], t[1]) for t, shape in zip(tensors, shapes)]
            params, init, prepare = entry[:3]
            if params:
                tail += record(params)
            if init and _tflm_layout(init, target, compression)[0]:
                tail.append((_tflm_layout(init, target, compression)[0], 16))
            if prepare is not None:
                tail += [(max(0, size), 16) for size in prepare(view, inputs, outputs, options,
                                                                target)]
            scratch = _tflm_bound_scratch(kind, view, inputs, outputs, options, target, compressed)
            if scratch and number not in reached:
                raise ValueError(f"no IF, WHILE or CALL_ONCE reaches subgraph {number}, so "
                                 f"TFLite Micro plans the scratch buffers of its {kind} from "
                                 "uninitialized memory")
            head += sum((max(0, size) + 15) // 16 * 16 for size in scratch)
            scratch_count += len(scratch)
            temps = max(temps, tflm_temps(kind, view, inputs, outputs, target))
            written = tflm_written_shape(kind, shapes, inputs, outputs, options)
            if written is not None:
                shapes[outputs[0]] = written
        for k, (tensor, shape) in enumerate(zip(tensors, shapes)):
            size = _tflm_bytes((tensor[0], shape))
            planned = offline[tensor_total + k] if offline else -1
            if tensor[2]:
                variables.append((size, 16))
            if tensor[5]:
                continue
            if planned >= 0:
                offline_end = max(offline_end, planned + (size + 15) // 16 * 16)
            elif not tensor[2]:
                head += (size + 15) // 16 * 16
        tensor_total += len(tensors)
    prepared = sum(size + align - 1 for size, align in tail)
    tail = [*variables, *record("ScratchBufferHandle", scratch_count)]
    tensors, _, inputs, outputs = subgraphs[0]
    for boundary in (inputs, outputs):
        tail.append((pointer * len(boundary), 16))
        for index in boundary:
            tail += record("TfLiteTensor")
            if _tflm_quantized(tensors[index]):
                tail += [*record("TfLiteAffineQuantization"), (_tflm_dims(tensors[index][3]), 4)]
    planning = tflm_planning(tensor_total, scratch_count, target) + pointer * len(subgraphs)
    return prepared + max(sum(size + align - 1 for size, align in tail) + offline_end + head
                          + planning, tflm_preparing(temps, scratch_count, target))
