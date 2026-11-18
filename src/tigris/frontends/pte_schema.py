"""Facts of the ExecuTorch program format (schema/program.fbs, schema/scalar_type.fbs
and schema/extended_header.h of ExecuTorch 1.6.0) that the .pte reader and writer
share: the file identifier, the field count of each table, the union members,
the scalar types, and the layout of the extended header. The field slots of
each table are listed with the records in pte.py."""

import struct

MAGIC = b"ET12"


def is_pte(data: bytes) -> bool:
    return len(data) >= 8 and data[4:8] == MAGIC


# The extended header sits at byte 8, after the root offset and the identifier:
# magic, its own length (u32), program size, segment base offset and segment
# data size (u64 each). It is zero-padded to a power of two no smaller than
# itself or than any force_align of the schema, so the FlatBuffer after it
# keeps its alignment.
HEADER_OFFSET = 8
HEADER_MAGIC = b"eh00"
HEADER = struct.Struct("<4sIQQQ")
# The force_align of Buffer.storage and BackendDelegateInlineData.data; a
# program can be serialized with larger ones.
DATA_ALIGNMENT = 16
# ExecuTorch's default alignment of each segment and of the segment data.
SEGMENT_ALIGNMENT = 128

# Field count of each table, as its generated builder starts it; a union takes
# two slots, its type and its value.
FIELDS = {
    "program": 8, "execution plan": 10, "container metadata": 2, "evalue": 2,
    "null": 0, "int": 1, "bool": 1, "double": 1, "string": 1, "int list": 1,
    "double list": 1, "bool list": 1, "tensor list": 1, "optional tensor list": 1,
    "tensor": 10, "allocation details": 3, "extra tensor info": 5, "operator": 2,
    "chain": 4, "instruction": 2, "kernel call": 2, "delegate call": 3,
    "delegate scratch": 2, "move call": 2, "jump false call": 2, "free call": 1,
    "frame list": 1, "frame": 4, "backend delegate": 3, "delegate data reference": 2,
    "compile spec": 2, "non-const buffer device": 3, "buffer": 1,
    "inline data": 1, "data segment": 2, "subsegment offsets": 2, "named data": 2,
}

# Members of the KernelTypes union (EValue) and the InstructionArguments union.
VALUE_TYPES = ("Null", "Int", "Bool", "Double", "Tensor", "String", "IntList",
               "DoubleList", "BoolList", "TensorList", "OptionalTensorList")
INSTRUCTION_TYPES = ("KernelCall", "DelegateCall", "MoveCall", "JumpFalseCall", "FreeCall")

# ScalarType values and the byte size of one element.
SCALAR_TYPES = {
    0: ("BYTE", 1), 1: ("CHAR", 1), 2: ("SHORT", 2), 3: ("INT", 4), 4: ("LONG", 8),
    5: ("HALF", 2), 6: ("FLOAT", 4), 7: ("DOUBLE", 8), 11: ("BOOL", 1), 12: ("QINT8", 1),
    13: ("QUINT8", 1), 14: ("QINT32", 4), 15: ("BFLOAT16", 2), 16: ("QUINT4X2", 1),
    17: ("QUINT2X4", 1), 22: ("BITS16", 2), 23: ("FLOAT8E5M2", 1), 24: ("FLOAT8E4M3FN", 1),
    25: ("FLOAT8E5M2FNUZ", 1), 26: ("FLOAT8E4M3FNUZ", 1), 27: ("UINT16", 2),
    28: ("UINT32", 4), 29: ("UINT64", 8),
}
