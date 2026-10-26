"""Canonical operator records for the compiler and the plan loader."""

from dataclasses import dataclass

DATA = 0
DATA_OR_BOOL = 255
STATE = 254
DATA_OR_INT32 = 253
BOOL_OR_INT32 = 252
CONTROL_FLOW_VALUE = 251
FLOAT_OR_DATA = 250


@dataclass(frozen=True)
class DTypeSignature:
    inputs: tuple[int, ...] = (DATA, DATA, DATA)
    outputs: tuple[int, ...] = (DATA,)


@dataclass(frozen=True)
class OperatorContract:
    opcode: int
    c_name: str
    inputs: tuple[int, int]
    outputs: tuple[int, int]
    signature: DTypeSignature
    weight: str
    bias: str
    attributes: tuple[int, ...]
    required_attributes: tuple[int, ...]
    modes: frozenset[str]
    backends: frozenset[str]
    executor: bool
    validator: str | None


@dataclass(frozen=True)
class AttributeContract:
    """Payload lengths from minimum to maximum in steps of step; lengths lists them when few."""

    name: str
    lengths: tuple[int, ...]
    minimum: int
    maximum: int
    step: int


ATTRIBUTES: dict[int, AttributeContract] = {
    1: AttributeContract('TRANSPOSE_PERM', (), 0, 255, 1),
    2: AttributeContract('EPSILON', (4,), 4, 4, 1),
    3: AttributeContract('ALPHA', (4,), 4, 4, 1),
    4: AttributeContract('CLIP_BOUNDS', (8,), 8, 8, 1),
    5: AttributeContract('PADS', (), 8, 248, 8),
    6: AttributeContract('AXES', (), 1, 255, 1),
    7: AttributeContract('BINARY_REQUANT', (24,), 24, 24, 1),
    8: AttributeContract('POOL_ROUNDING', (1,), 1, 1, 1),
    9: AttributeContract('CONSTANT_OPERAND', (), 4, 252, 4),
    10: AttributeContract('RESIZE_SCALES', (8,), 8, 8, 1),
    11: AttributeContract('CUMSUM_OPTIONS', (2,), 2, 2, 1),
    12: AttributeContract('MOVEMENT', (), 4, 76, 4),
    13: AttributeContract('COMPARISON_REQUANT', (20,), 20, 20, 1),
    14: AttributeContract('CONSTANTS', (), 2, 254, 2),
    15: AttributeContract('SVDF', (4, 24), 4, 24, 20),
    16: AttributeContract('LSTM', (8, 112), 8, 112, 104),
    17: AttributeContract('SUBGRAPHS', (4,), 4, 4, 1),
    18: AttributeContract('DETECTION', (32,), 32, 32, 1),
    19: AttributeContract('GELU_APPROXIMATE', (1,), 1, 1, 1),
    20: AttributeContract('TABLE_SHAPE', (), 4, 24, 4),
}


OPERATOR_CONTRACT: dict[str, OperatorContract] = {
    'Conv': OperatorContract(1, 'TIGRIS_OP_CONV', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (), (), frozenset(('spatial', 'fused')), frozenset(('reference', 's8_ref', 'esp-nn', 'cmsis-nn')), False, 'validate_contract_conv'),
    'DepthwiseConv': OperatorContract(2, 'TIGRIS_OP_DEPTHWISE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (), (), frozenset(('spatial', 'fused')), frozenset(('reference', 's8_ref', 'esp-nn', 'cmsis-nn')), False, 'validate_contract_conv'),
    'Relu': OperatorContract(3, 'TIGRIS_OP_RELU', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_relu'),
    'Relu6': OperatorContract(4, 'TIGRIS_OP_RELU6', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_relu'),
    'MaxPool': OperatorContract(5, 'TIGRIS_OP_MAX_POOL', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('spatial',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_max_pool'),
    'AveragePool': OperatorContract(6, 'TIGRIS_OP_AVG_POOL', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('spatial',)), frozenset(('reference', 's8_ref', 'esp-nn', 'cmsis-nn')), False, 'validate_contract_max_pool'),
    'Add': OperatorContract(7, 'TIGRIS_OP_ADD', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (253,)), 'optional', 'none', (7, 9), (), frozenset(('binary', 'row', 'axis_binary', 'height', 'fused')), frozenset(('reference', 's8_ref')), False, 'validate_contract_add'),
    'Mul': OperatorContract(8, 'TIGRIS_OP_MUL', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (253,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref', 'esp-nn', 'cmsis-nn')), False, 'validate_contract_add'),
    'Gemm': OperatorContract(9, 'TIGRIS_OP_FULLY_CONN', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (), (), frozenset(('row', 'whole_operand', 'fused')), frozenset(('reference', 's8_ref', 'esp-nn', 'cmsis-nn')), False, 'validate_contract_fully_conn'),
    # Softmax normalizes along the final stored dimension, the channel axis in
    # both NHWC and NLC. Both tile axes are ahead of it, so a band holds whole
    # normalization rows and needs nothing from its neighbors: shape-preserving
    # and halo-free, which is what the height mode means, although the operator
    # itself is a reduction.
    'Softmax': OperatorContract(10, 'TIGRIS_OP_SOFTMAX', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_relu'),
    'Clip': OperatorContract(11, 'TIGRIS_OP_CLIP', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (4,), (), frozenset(()), frozenset(()), False, None),
    'Sigmoid': OperatorContract(12, 'TIGRIS_OP_SIGMOID', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_relu'),
    'Concat': OperatorContract(13, 'TIGRIS_OP_CONCAT', (1, 255), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (), (), frozenset(('height',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_concat'),
    'Pad': OperatorContract(14, 'TIGRIS_OP_PAD', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (5,), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_pad'),
    'GlobalAveragePool': OperatorContract(15, 'TIGRIS_OP_GLOBAL_AVG', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (8,), (), frozenset(()), frozenset(('reference', 's8_ref', 'cmsis-nn')), False, 'validate_contract_global_max'),
    'Flatten': OperatorContract(16, 'TIGRIS_OP_FLATTEN', (1, 1), (1, 1), DTypeSignature((255, 255, 255), (255,)), 'none', 'none', (), (), frozenset(('row',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_reshape'),
    'Reshape': OperatorContract(17, 'TIGRIS_OP_RESHAPE', (1, 1), (1, 1), DTypeSignature((255, 255, 255), (255,)), 'none', 'none', (), (), frozenset(('row',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_reshape'),
    'Sub': OperatorContract(18, 'TIGRIS_OP_SUB', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (253,)), 'optional', 'none', (7, 9), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_add'),
    'Div': OperatorContract(19, 'TIGRIS_OP_DIV', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_div'),
    'Tanh': OperatorContract(20, 'TIGRIS_OP_TANH', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_relu'),
    'LeakyRelu': OperatorContract(21, 'TIGRIS_OP_LEAKY_RELU', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (3,), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_leaky_relu'),
    'BatchNormalization': OperatorContract(22, 'TIGRIS_OP_BATCH_NORM', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(()), False, None),
    'InstanceNormalization': OperatorContract(23, 'TIGRIS_OP_INST_NORM', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(()), False, None),
    'ConvTranspose': OperatorContract(24, 'TIGRIS_OP_CONV_TRANSPOSE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_conv_transpose'),
    'MatMul': OperatorContract(25, 'TIGRIS_OP_MATMUL', (2, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('row', 'whole_operand')), frozenset(('reference', 's8_ref')), False, 'validate_contract_matmul'),
    'ReduceMean': OperatorContract(26, 'TIGRIS_OP_REDUCE_MEAN', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'Squeeze': OperatorContract(27, 'TIGRIS_OP_SQUEEZE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(()), False, None),
    'Unsqueeze': OperatorContract(28, 'TIGRIS_OP_UNSQUEEZE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(()), False, None),
    'Transpose': OperatorContract(29, 'TIGRIS_OP_TRANSPOSE', (1, 1), (1, 1), DTypeSignature((255, 255, 255), (255,)), 'none', 'none', (1,), (1,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_reshape'),
    # Resampling bands only along the height axis and only on its own stage,
    # because the chain executor does not compose a fractional stride through
    # the rest of a run, so it carries the resample mode and not the height mode.
    'Resize': OperatorContract(30, 'TIGRIS_OP_RESIZE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (10,), (), frozenset(('resample',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_resize_linear'),
    'GlobalMaxPool': OperatorContract(31, 'TIGRIS_OP_GLOBAL_MAX', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_global_max'),
    'Conv1D': OperatorContract(32, 'TIGRIS_OP_CONV1D', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (), (), frozenset(('fused',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_conv1d'),
    # Reduces along the final stored dimension, as Softmax does, so it carries
    # the height mode for the same reason.
    'LayerNormalization': OperatorContract(33, 'TIGRIS_OP_LAYER_NORM', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'required', 'optional', (2,), (2,), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_layer_norm'),
    'Erf': OperatorContract(34, 'TIGRIS_OP_ERF', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_hardswish'),
    'Split': OperatorContract(35, 'TIGRIS_OP_SPLIT', (1, 1), (2, 255), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_split'),
    'ResizeLinear': OperatorContract(36, 'TIGRIS_OP_RESIZE_LINEAR', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (10,), (), frozenset(('resample',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_resize_linear'),
    'HardSwish': OperatorContract(37, 'TIGRIS_OP_HARDSWISH', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_hardswish'),
    'Abs': OperatorContract(38, 'TIGRIS_OP_ABS', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_abs'),
    'Rsqrt': OperatorContract(39, 'TIGRIS_OP_RSQRT', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_abs'),
    'SquaredDifference': OperatorContract(40, 'TIGRIS_OP_SQUARED_DIFFERENCE', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_div'),
    'Max': OperatorContract(41, 'TIGRIS_OP_MAXIMUM', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref', 'cmsis-nn')), False, 'validate_contract_div'),
    'Min': OperatorContract(42, 'TIGRIS_OP_MINIMUM', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref', 'cmsis-nn')), False, 'validate_contract_div'),
    'Neg': OperatorContract(43, 'TIGRIS_OP_NEG', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Exp': OperatorContract(44, 'TIGRIS_OP_EXP', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Log': OperatorContract(45, 'TIGRIS_OP_LOG', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Sqrt': OperatorContract(46, 'TIGRIS_OP_SQRT', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Square': OperatorContract(47, 'TIGRIS_OP_SQUARE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Floor': OperatorContract(48, 'TIGRIS_OP_FLOOR', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Ceil': OperatorContract(49, 'TIGRIS_OP_CEIL', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Round': OperatorContract(50, 'TIGRIS_OP_ROUND', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Sin': OperatorContract(51, 'TIGRIS_OP_SIN', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'Cos': OperatorContract(52, 'TIGRIS_OP_COS', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference',)), False, 'validate_contract_abs'),
    'FloorDiv': OperatorContract(53, 'TIGRIS_OP_FLOOR_DIV', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference',)), False, 'validate_contract_div'),
    'FloorMod': OperatorContract(54, 'TIGRIS_OP_FLOOR_MOD', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference',)), False, 'validate_contract_div'),
    'PRelu': OperatorContract(55, 'TIGRIS_OP_PRELU', (1, 2), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_div'),
    'Elu': OperatorContract(56, 'TIGRIS_OP_ELU', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_leaky_relu'),
    'LogSoftmax': OperatorContract(57, 'TIGRIS_OP_LOG_SOFTMAX', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_leaky_relu'),
    'L2Normalization': OperatorContract(58, 'TIGRIS_OP_L2_NORMALIZATION', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (2,), (2,), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_leaky_relu'),
    'L2Pool': OperatorContract(59, 'TIGRIS_OP_L2_POOL', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('spatial',)), frozenset(('reference',)), False, 'validate_contract_max_pool'),
    'ReduceMax': OperatorContract(60, 'TIGRIS_OP_REDUCE_MAX', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'ReduceMin': OperatorContract(61, 'TIGRIS_OP_REDUCE_MIN', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'ReduceSum': OperatorContract(62, 'TIGRIS_OP_REDUCE_SUM', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'CumSum': OperatorContract(63, 'TIGRIS_OP_CUMSUM', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (6, 11), (6, 11), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_cumsum'),
    'ArgMax': OperatorContract(64, 'TIGRIS_OP_ARG_MAX', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (6,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'ArgMin': OperatorContract(65, 'TIGRIS_OP_ARG_MIN', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (6,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'Gather': OperatorContract(66, 'TIGRIS_OP_GATHER', (0, 2), (1, 1), DTypeSignature((0, 6, 6), (0,)), 'optional', 'none', (12, 20), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'GatherND': OperatorContract(67, 'TIGRIS_OP_GATHER_ND', (0, 2), (1, 1), DTypeSignature((0, 6, 6), (0,)), 'none', 'none', (12, 20), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'StridedSlice': OperatorContract(68, 'TIGRIS_OP_STRIDED_SLICE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (12,), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'MirrorPad': OperatorContract(69, 'TIGRIS_OP_MIRROR_PAD', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (12,), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'ReverseV2': OperatorContract(70, 'TIGRIS_OP_REVERSE_V2', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (12,), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'EmbeddingLookup': OperatorContract(71, 'TIGRIS_OP_EMBEDDING_LOOKUP', (0, 2), (1, 1), DTypeSignature((0, 6, 6), (0,)), 'none', 'none', (12, 20), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'DynamicUpdateSlice': OperatorContract(72, 'TIGRIS_OP_DYNAMIC_UPDATE_SLICE', (1, 3), (1, 1), DTypeSignature((0, 0, 6), (0,)), 'optional', 'none', (12,), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_gather'),
    'Equal': OperatorContract(73, 'TIGRIS_OP_EQUAL', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (9,)), 'optional', 'none', (9, 13), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Less': OperatorContract(74, 'TIGRIS_OP_LESS', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (9,)), 'optional', 'none', (9, 13), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'LessOrEqual': OperatorContract(75, 'TIGRIS_OP_LESS_EQUAL', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (9,)), 'optional', 'none', (9, 13), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Greater': OperatorContract(76, 'TIGRIS_OP_GREATER', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (9,)), 'optional', 'none', (9, 13), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'GreaterOrEqual': OperatorContract(77, 'TIGRIS_OP_GREATER_EQUAL', (1, 2), (1, 1), DTypeSignature((253, 253, 253), (9,)), 'optional', 'none', (9, 13), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'And': OperatorContract(78, 'TIGRIS_OP_LOGICAL_AND', (1, 2), (1, 1), DTypeSignature((9, 9, 9), (9,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Or': OperatorContract(79, 'TIGRIS_OP_LOGICAL_OR', (1, 2), (1, 1), DTypeSignature((9, 9, 9), (9,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Not': OperatorContract(80, 'TIGRIS_OP_LOGICAL_NOT', (1, 1), (1, 1), DTypeSignature((9, 9, 9), (9,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Where': OperatorContract(81, 'TIGRIS_OP_SELECT_V2', (2, 3), (1, 1), DTypeSignature((9, 0, 0), (0,)), 'optional', 'none', (9,), (), frozenset(('binary', 'row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Cast': OperatorContract(82, 'TIGRIS_OP_CAST', (1, 1), (1, 1), DTypeSignature((252, 252, 252), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'Sum': OperatorContract(83, 'TIGRIS_OP_ADD_N', (1, 255), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (), (), frozenset(('row', 'axis_binary', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_equal'),
    'ReduceAll': OperatorContract(84, 'TIGRIS_OP_REDUCE_ALL', (1, 1), (1, 1), DTypeSignature((9, 9, 9), (9,)), 'none', 'none', (6,), (6,), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_arg_max'),
    'Svdf': OperatorContract(85, 'TIGRIS_OP_SVDF', (2, 2), (2, 2), DTypeSignature((0, 254, 254), (254,)), 'specific', 'specific', (14, 15), (), frozenset(('fused_float',)), frozenset(('reference', 's8_ref')), False, 'validate_contract_svdf'),
    'Lstm': OperatorContract(86, 'TIGRIS_OP_LSTM', (3, 3), (3, 3), DTypeSignature((0, 254, 254), (254,)), 'specific', 'specific', (14, 16), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_lstm'),
    'If': OperatorContract(87, 'TIGRIS_OP_IF', (1, 255), (1, 255), DTypeSignature((9, 251, 251), (251,)), 'none', 'none', (17,), (), frozenset(()), frozenset(('reference', 's8_ref')), True, 'validate_contract_if'),
    'While': OperatorContract(88, 'TIGRIS_OP_WHILE', (1, 255), (1, 255), DTypeSignature((251, 251, 251), (251,)), 'specific', 'specific', (14, 17), (), frozenset(()), frozenset(('reference', 's8_ref')), True, 'validate_contract_if'),
    'DetectionPostProcess': OperatorContract(89, 'TIGRIS_OP_DETECTION_POSTPROCESS', (2, 2), (5, 5), DTypeSignature((0, 0, 0), (1,)), 'specific', 'specific', (14, 18), (), frozenset(()), frozenset(('reference', 's8_ref')), False, 'validate_contract_detection_postprocess'),
    'Quantize': OperatorContract(90, 'TIGRIS_OP_QUANTIZE', (1, 1), (1, 1), DTypeSignature((250, 250, 250), (0,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_quantize'),
    'Dequantize': OperatorContract(91, 'TIGRIS_OP_DEQUANTIZE', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (1,)), 'none', 'none', (), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_quantize'),
    'Gelu': OperatorContract(92, 'TIGRIS_OP_GELU', (1, 1), (1, 1), DTypeSignature((0, 0, 0), (0,)), 'none', 'none', (19,), (), frozenset(('axis_unary', 'row', 'height')), frozenset(('reference', 's8_ref')), False, 'validate_contract_gelu'),
}

OP_TYPE_MAP = {name: row.opcode for name, row in OPERATOR_CONTRACT.items()}
OP_TYPE_BY_CODE = {opcode: name for name, opcode in OP_TYPE_MAP.items()}
OP_C_NAMES = {row.opcode: row.c_name.removeprefix("TIGRIS_OP_")
              for row in OPERATOR_CONTRACT.values()}
OP_DTYPE_SIGNATURES = {name: row.signature for name, row in OPERATOR_CONTRACT.items()}
EXECUTOR_OPERATORS = frozenset(name for name, row in OPERATOR_CONTRACT.items() if row.executor)


def native_operators(backend: str) -> frozenset[str]:
    return frozenset(name for name, row in OPERATOR_CONTRACT.items() if backend in row.backends)


def operators_with(mode: str) -> frozenset[str]:
    """The operators whose record carries one execution mode."""
    return frozenset(name for name, row in OPERATOR_CONTRACT.items() if mode in row.modes)
