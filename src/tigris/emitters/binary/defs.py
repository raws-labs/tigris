"""Binary plan format - constants, enums, and canonical wire layouts.

Keep in sync with tigris-runtime/include/tigris.h
"""

import struct

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

SECTION_TYPES = (
    SEC_TENSORS,
    SEC_OPS,
    SEC_STAGES,
    SEC_TILE_PLANS,
    SEC_INDEX_POOL,
    SEC_SHAPE_POOL,
    SEC_STRINGS,
    SEC_WEIGHTS,
    SEC_QUANT_PARAMS,
    SEC_WEIGHT_BLOCKS,
    SEC_OP_ATTRIBUTES,
)

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

# Stage flags - packed into the head stage's _reserved1 field.
STAGE_FLAG_LINE_BUFFERED = 0x0001

# Canonical little-endian wire layouts.  Writer, reader, and size estimates all
# consume these definitions so a schema edit cannot silently leave one of them
# using a stale hand-maintained byte count.
HEADER_STRUCT = struct.Struct("<4sIIIHHHHIIIHBBHHI")
SECTION_ENTRY_STRUCT = struct.Struct("<II")
TENSOR_STRUCT = struct.Struct("<IIHBBBHx")

# An operator is emitted in four pieces because its spatial fields are built by
# a dedicated validator.  Their combined size is the packed tigris_op_t size.
OP_PREFIX_STRUCT = struct.Struct("<IBBBBHH")
SPATIAL_ATTRS_STRUCT = struct.Struct("<4B7H")
OP_WEIGHT_BIAS_STRUCT = struct.Struct("<HH")
OP_ACTIVATION_STRUCT = struct.Struct("<Bbbx")

STAGE_STRUCT = struct.Struct("<I12H")
TILE_PLAN_STRUCT = struct.Struct("<BB5H3I")
WEIGHT_ENTRY_STRUCT = struct.Struct("<III")
QUANT_PARAM_STRUCT = struct.Struct("<fiHHHH")
WEIGHT_BLOCK_STRUCT = struct.Struct("<HHHHIII")
OP_ATTRIBUTE_STRUCT = struct.Struct("<HBBI")

# Section-local headers and primitive pool elements are part of the wire
# contract too, even though they are not named C record types.
QUANT_SECTION_HEADER_STRUCT = struct.Struct("<HH")
WEIGHT_BLOCK_SECTION_HEADER_STRUCT = struct.Struct("<HH")
OP_ATTRIBUTE_SECTION_HEADER_STRUCT = struct.Struct("<HH")
INDEX_STRUCT = struct.Struct("<H")
SHAPE_DIM_STRUCT = struct.Struct("<i")
QUANT_DATA_STRUCT = struct.Struct("<i")

# Every section begins on this boundary.  The weights section receives an
# additional pre-padding adjustment so its blob begins on the same boundary.
PLAN_SECTION_ALIGNMENT = 16

# Derived sizes remain available under the established public names.
HEADER_SIZE = HEADER_STRUCT.size
SECTION_ENTRY_SIZE = SECTION_ENTRY_STRUCT.size
TENSOR_SIZE = TENSOR_STRUCT.size
OP_SIZE = (
    OP_PREFIX_STRUCT.size
    + SPATIAL_ATTRS_STRUCT.size
    + OP_WEIGHT_BIAS_STRUCT.size
    + OP_ACTIVATION_STRUCT.size
)
STAGE_SIZE = STAGE_STRUCT.size
TILE_PLAN_SIZE = TILE_PLAN_STRUCT.size
WEIGHT_ENTRY_SIZE = WEIGHT_ENTRY_STRUCT.size
QUANT_PARAM_SIZE = QUANT_PARAM_STRUCT.size
WEIGHT_BLOCK_SIZE = WEIGHT_BLOCK_STRUCT.size
OP_ATTRIBUTE_SIZE = OP_ATTRIBUTE_STRUCT.size

# Derived field offsets used for staged decoding/patching.
OP_SPATIAL_OFFSET = OP_PREFIX_STRUCT.size
OP_WEIGHT_BIAS_OFFSET = OP_SPATIAL_OFFSET + SPATIAL_ATTRS_STRUCT.size
OP_ACTIVATION_OFFSET = OP_WEIGHT_BIAS_OFFSET + OP_WEIGHT_BIAS_STRUCT.size
STAGE_TILE_PLAN_INDEX_OFFSET = struct.calcsize("<I6H")

# Sentinel for no weight/bias
NO_WEIGHT = 0xFFFF
NO_QUANT_PARAM = 0xFFFF

# Fused activation enum
ACT_NONE = 0
ACT_RELU = 1
ACT_RELU6 = 2

# Spatial attr keys we extract from ONNX op attrs
_SPATIAL_KEYS = ("kernel_shape", "strides", "pads", "dilations", "group")
