"""Binary plan format - constants, enums, and struct sizes.

Keep in sync with tigris-runtime/include/tigris.h
"""

# Magic
MAGIC = b"TGRS"

# Section type IDs
SEC_TENSORS = 1
SEC_OPS = 2
SEC_STAGES = 3
SEC_TILE_PLANS = 4
SEC_INDEX_POOL = 5
SEC_SHAPE_POOL = 6
SEC_STRINGS = 7
SEC_WEIGHTS = 8
SEC_QUANT_PARAMS = 9
SEC_WEIGHT_BLOCKS = 10
SEC_OP_ATTRIBUTES = 11

# Per-operator attribute kinds stored in SEC_OP_ATTRIBUTES.
OP_ATTR_TRANSPOSE_PERM = 1

# Compression types
COMPRESS_NONE = 0
COMPRESS_LZ4 = 1

# Header flags
FLAG_XIP = 0x01  # weights are execute-in-place from flash

# Op type enum (ONNX op_type string -> uint8)
OP_TYPE_MAP: dict[str, int] = {
    "Conv": 1,
    "DepthwiseConv": 2,
    "Relu": 3,
    "Relu6": 4,
    "MaxPool": 5,
    "AveragePool": 6,
    "Add": 7,
    "Mul": 8,
    "Gemm": 9,
    "Softmax": 10,
    "Clip": 11,
    "Sigmoid": 12,
    "Concat": 13,
    "Pad": 14,
    "GlobalAveragePool": 15,
    "Flatten": 16,
    "Reshape": 17,
    "Sub": 18,
    "Div": 19,
    "Tanh": 20,
    "LeakyRelu": 21,
    "BatchNormalization": 22,
    "InstanceNormalization": 23,
    "ConvTranspose": 24,
    "MatMul": 25,
    "ReduceMean": 26,
    "Squeeze": 27,
    "Unsqueeze": 28,
    "Transpose": 29,
    "Resize": 30,
    "GlobalMaxPool": 31,
    "Conv1D": 32,
}
OP_TYPE_UNKNOWN = 255

# Tensor flags
TENSOR_FLAG_CONSTANT = 0x01
TENSOR_FLAG_MODEL_INPUT = 0x02
TENSOR_FLAG_MODEL_OUTPUT = 0x04

# Struct sizes (layout retained by schema v3)
HEADER_SIZE = 48
SECTION_ENTRY_SIZE = 8
TENSOR_SIZE = 16
OP_SIZE = 38  # spatial attrs pad/dilation widened u8->u16 (12->18 bytes)
STAGE_SIZE = 28
TILE_PLAN_SIZE = 24
WEIGHT_ENTRY_SIZE = 12
QUANT_PARAM_SIZE = 16
WEIGHT_BLOCK_SIZE = 20

# Sentinel for no weight/bias
NO_WEIGHT = 0xFFFF
NO_QUANT_PARAM = 0xFFFF

# Fused activation enum
ACT_NONE = 0
ACT_RELU = 1
ACT_RELU6 = 2

# Spatial attr keys we extract from ONNX op attrs
_SPATIAL_KEYS = ("kernel_shape", "strides", "pads", "dilations", "group")
