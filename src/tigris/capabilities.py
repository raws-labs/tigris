"""Audited runtime kernel capabilities used by codegen and documentation.

The native operator sets mirror the dispatch switches in tigris-runtime:

* ``src/tigris_kernels.c`` (reference float32)
* ``src/tigris_kernels_s8.c`` (reference int8)
* ``src/tigris_kernels_esp_nn.c`` (ESP-NN with explicit s8_ref fallback)
* ``src/tigris_kernels_cmsis_nn.c`` (CMSIS-NN with explicit s8_ref fallback)

Keep this as the single compiler-side capability source.  Accelerated
backends list only operators with a native adapter; effective support follows
their explicit runtime fallback chain.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from tigris.emitters.binary.defs import OP_TYPE_MAP


DTypeMode = Literal["float32", "int8"]


@dataclass(frozen=True)
class KernelCapabilities:
    """Native operators and explicit fallback for one runtime dispatcher."""

    name: str
    dtype: DTypeMode
    native_operators: frozenset[str]
    fallback: str | None = None


_FLOAT_REFERENCE_OPERATORS = frozenset({
    "Pad",
    "Conv",
    "ConvTranspose",
    "DepthwiseConv",
    "Relu",
    "Relu6",
    "Sigmoid",
    "Tanh",
    "Add",
    "Sub",
    "Mul",
    "Conv1D",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "AveragePool",
    "Gemm",
    "MatMul",
    "Reshape",
    "Flatten",
    "MaxPool",
    "Concat",
    "Resize",
    "ResizeLinear",
    "Softmax",
    "LeakyRelu",
    "PRelu",
    "Elu",
    "LogSoftmax",
    "L2Normalization",
    "L2Pool",
    "Transpose",
    "LayerNormalization",
    "Erf",
    "HardSwish",
    "Abs",
    "Rsqrt",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Square",
    "Floor",
    "Ceil",
    "Round",
    "Sin",
    "Cos",
    "Div",
    "SquaredDifference",
    "Max",
    "Min",
    "FloorDiv",
    "FloorMod",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceSum",
    "CumSum",
    "ArgMax",
    "ArgMin",
    "Gather",
    "GatherND",
    "StridedSlice",
    "MirrorPad",
    "ReverseV2",
    "EmbeddingLookup",
    "DynamicUpdateSlice",
    "Equal",
    "Less",
    "LessOrEqual",
    "Greater",
    "GreaterOrEqual",
    "And",
    "Or",
    "Not",
    "Where",
    "Cast",
    "Sum",
    "ReduceAll",
    "Split",
    "Svdf",
    "Lstm",
    "If",
})

# Operators the executor runs itself rather than a kernel dispatcher.
EXECUTOR_OPERATORS = frozenset({"If"})

_S8_REFERENCE_OPERATORS = _FLOAT_REFERENCE_OPERATORS - frozenset({
    "If",
    "L2Pool",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Square",
    "Floor",
    "Ceil",
    "Round",
    "Sin",
    "Cos",
    "FloorDiv",
    "FloorMod",
})


# Native accelerated adapters sometimes route a supported variant through the
# portable int8 dispatcher to preserve semantics.  Keep these conditions next
# to the operator sets so generated documentation cannot imply that every
# variant reaches vendor code.
CONDITIONAL_FALLBACKS: dict[str, dict[str, str]] = {
    "esp-nn": {
        "Conv": (
            "falls back for dilation other than 1; asymmetric padding falls "
            "back when its preparation-time workspace is insufficient"
        ),
        "DepthwiseConv": (
            "falls back for dilation other than 1; channel multipliers require batch 1"
        ),
        "Mul": (
            "falls back for constants, broadcasting, non-per-tensor quantization, "
            "requantization shifts outside [-31, 15], or width, row or rolled tiles"
        ),
        "AveragePool": (
            "falls back when tiled or when input/output quantization differs"
        ),
    },
    "cmsis-nn": {
        "Conv": "falls back for dilation other than 1",
        "DepthwiseConv": (
            "falls back for tiled dilation other than 1; channel multipliers "
            "fall back with CMSIS_NN_USE_SINGLE_ROUNDING"
        ),
        "Mul": (
            "falls back for constants, broadcasting, non-per-tensor quantization, "
            "requantization shifts outside [-31, 15], width, row or rolled tiles, "
            "or CMSIS_NN_USE_SINGLE_ROUNDING"
        ),
        "Max": (
            "falls back unless dynamic operands have identical shapes and both inputs "
            "and output have equal per-tensor scales and zero points; "
            "width, row and rolled tiles fall back"
        ),
        "Min": (
            "falls back unless dynamic operands have identical shapes and both inputs "
            "and output have equal per-tensor scales and zero points; "
            "width, row and rolled tiles fall back"
        ),
        "AveragePool": (
            "falls back when tiled or when input/output quantization differs"
        ),
        "GlobalAveragePool": (
            "falls back when tiled or when input/output quantization differs"
        ),
    },
}


# Operator-level qualifications that are part of the implemented plan
# contract.  The compiler's semantic validator remains authoritative for
# individual models; these concise notes prevent the public matrix from being
# mistaken for support for every ONNX attribute combination.
_BROADCAST = (
    "each operand has the output's shape or broadcasts to it, a tensor operand at the "
    "output's rank; tiled only when every operand is dense, a repeating constant, or one "
    "value per channel as the second operand"
)
_CONSTANT_OPERAND = (
    "a constant operand, on either side, float or per-tensor int8; one per element runs "
    "untiled"
)

OPERATOR_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    **{kind: ("data inputs, bool output; broadcasting and one constant operand; int8 input scales strictly between 0 and 1",)
       for kind in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual")},
    "And": ("bool inputs and output; broadcasting and one constant operand",),
    "Or": ("bool inputs and output; broadcasting and one constant operand",),
    "Not": ("one dynamic bool input and bool output",),
    "Where": ("bool condition and data branches; three-operand broadcasting through rank 5; at most one constant; int8 branches and output share quantization",),
    "Cast": ("dynamic bool to float32, or to int8 quantizing 0 and 1 into the output encoding",),
    "Sum": ("equal-shaped dynamic data inputs; int8 input quantization must match, with all partial sums proven within int32",),
    "Pad": (
        "constant mode with non-negative pads and a constant fill; int8 keeps its "
        "quantization; untiled execution",
    ),
    "Add": (
        f"{_BROADCAST}; {_CONSTANT_OPERAND}",
        "standalone rank-3 pointwise length tiling on serialized axis 1",
    ),
    "Sub": (
        f"{_BROADCAST}; {_CONSTANT_OPERAND}",
    ),
    "Mul": (
        f"{_BROADCAST}; {_CONSTANT_OPERAND}",
        "standalone rank-3 pointwise length tiling on serialized axis 1",
    ),
    "AveragePool": (
        "explicit padding, floor output sizing, unit dilation, and count_include_pad=0 wherever padding is non-zero",
    ),
    "MaxPool": (
        "explicit padding, floor output sizing, unit dilation, and no indices output",
    ),
    "Concat": (
        "rank-3 and rank-4 concatenation on any stored axis but the batch "
        "axis, tiled only on the last one; a constant part only as the "
        "leading operand of a float concatenation on the last stored axis",
    ),
    "DepthwiseConv": (
        "one group per input channel with a channel multiplier",
    ),
    "Conv1D": (
        "standalone rank-3 length tiling on serialized axis 1; may compose with "
        "shape-preserving unary pointwise operators",
    ),
    "GlobalAveragePool": (
        "rank-4 height tiling: each band of rows is added into the per-channel sums",
    ),
    "Split": (
        "parts along any stored axis; untiled execution",
    ),
    "ReduceMax": ("one axis of a rank-3 tensor; independent height or row bands; rank-4 spatial max with keepdims uses GlobalMaxPool; int8 quantization must match",),
    "ReduceMin": ("one axis of a rank-3 tensor; independent height or row bands; int8 quantization must match",),
    "ReduceAll": ("one axis of a rank-3 bool tensor; independent height or row bands",),
    "If": ("one bool condition; branches run as their own stages; float32 operands of rank 2 at most; no nesting",),
    "Svdf": ("state kept between runs; int8 with int16 state and time weights; untiled",),
    "Lstm": ("hidden and cell state kept between runs; every gate, no peepholes, projection or layer normalization; tanh cell activation; int8 with an int16 cell state; untiled",),
    "ReduceSum": ("one axis of a rank-3 tensor; independent height or row bands; int8 rejects reference arithmetic overflow",),
    "Gather": ("constant or runtime int32 indices; independent bands with constant indices; identical int8 quantization",),
    "GatherND": ("constant or runtime int32 indices; independent bands with constant indices; identical int8 quantization",),
    "StridedSlice": ("constant indices or bounds; bands along an unchanged axis; identical int8 quantization",),
    "MirrorPad": ("constant indices or bounds; bands along an unchanged axis; identical int8 quantization",),
    "ReverseV2": ("constant indices or bounds; bands along an unchanged axis; identical int8 quantization",),
    "EmbeddingLookup": ("constant or runtime int32 indices; independent bands with constant indices; identical int8 quantization",),
    "DynamicUpdateSlice": ("constant or runtime int32 starts, clamped to the update domain; bands require constant starts and a dynamic update spanning the band axis; identical int8 quantization",),
    "ArgMax": ("one axis of a tensor of rank 1 to 6; independent height or row bands; int32 indices in model outputs or index operands; first index wins ties",),
    "ArgMin": ("one axis of a tensor of rank 1 to 6; independent height or row bands; int32 indices in model outputs or index operands; first index wins ties",),
    "CumSum": ("one axis of a rank-3 tensor; independent height or row bands; exclusive and reverse; int8 input zero point 0",),
    "ReduceMean": (
        "rank-3 mean over one axis; independent height or row bands. A rank-4 mean "
        "over both spatial axes is rewritten to GlobalAveragePool instead",
    ),
    "Resize": (
        "rank-4 nearest-neighbor H/W resizing with asymmetric floor, "
        "half-pixel centers or align-corners rounding; height tiling as the "
        "stage's single spatial op, never in a chain",
    ),
    "ResizeLinear": (
        "rank-4 bilinear H/W resizing with half-pixel, asymmetric or align-corners "
        "coordinates; height tiling as the stage's single spatial op, never "
        "in a chain",
    ),
    "PRelu": ("constant broadcast alpha, rank at most 4; " + _BROADCAST,),
    "Elu": ("alpha=1",),
    "L2Pool": ("float32 square root of the mean of valid squared samples",),
    "L2Normalization": ("final stored axis; int8 output scale 1/128 and zero point 0",),
    "LogSoftmax": ("final stored axis; int8 output scale 1/16 and zero point 127",),
    "Softmax": (
        "final axis only; tiling on serialized axis 1, each tile holding whole "
        "rows",
    ),
    "Relu": ("rank-3 pointwise length tiling on serialized axis 1",),
    "Relu6": ("rank-3 pointwise length tiling on serialized axis 1",),
    "Sigmoid": ("rank-3 pointwise length tiling on serialized axis 1",),
    "Tanh": ("rank-3 pointwise length tiling on serialized axis 1",),
    "HardSwish": ("rank-3 pointwise length tiling on serialized axis 1",),
    "Abs": ("int8 requantization must stay within the 32-bit reference arithmetic domain",),
    "Rsqrt": ("int8 requires nonnegative centered inputs and reference shifts in range",),
    "Div": (
        f"{_BROADCAST}; {_CONSTANT_OPERAND}; int8 requires per-tensor quantization",
        "int8 rejects zero centered denominators and negative rounding exponents; "
        "rounding exponents >= 32 use wide mask/remainder/threshold arithmetic",
    ),
    "SquaredDifference": (
        f"{_BROADCAST}; {_CONSTANT_OPERAND}",
        "int8 requantization must stay within the 32-bit reference arithmetic domain",
    ),
    "Max": (f"{_BROADCAST}; {_CONSTANT_OPERAND}; int8 quantization must match",),
    "Min": (f"{_BROADCAST}; {_CONSTANT_OPERAND}; int8 quantization must match",),
    "FloorDiv": (f"{_BROADCAST}; {_CONSTANT_OPERAND}; zero denominators are rejected",),
    "FloorMod": (f"{_BROADCAST}; {_CONSTANT_OPERAND}",),
    "Transpose": ("a concrete, valid permutation is stored in schema 4+ plans",),
}


KERNEL_CAPABILITIES: dict[str, KernelCapabilities] = {
    "reference": KernelCapabilities(
        name="reference",
        dtype="float32",
        native_operators=_FLOAT_REFERENCE_OPERATORS,
    ),
    "s8_ref": KernelCapabilities(
        name="s8_ref",
        dtype="int8",
        native_operators=_S8_REFERENCE_OPERATORS,
    ),
    "esp-nn": KernelCapabilities(
        name="esp-nn",
        dtype="int8",
        native_operators=frozenset({
            "Conv",
            "DepthwiseConv",
            "Gemm",
            "AveragePool",
            "Mul",
        }),
        fallback="s8_ref",
    ),
    "cmsis-nn": KernelCapabilities(
        name="cmsis-nn",
        dtype="int8",
        native_operators=frozenset({
            "Conv",
            "DepthwiseConv",
            "Gemm",
            "AveragePool",
            "GlobalAveragePool",
            "Mul",
            "Max",
            "Min",
        }),
        fallback="s8_ref",
    ),
}


CODEGEN_BACKENDS = ("reference", "esp-nn", "cmsis-nn")

_CODEGEN_KERNEL_SELECTION: dict[tuple[str, DTypeMode], str] = {
    ("reference", "float32"): "reference",
    ("reference", "int8"): "s8_ref",
    # ESP-NN and CMSIS-NN accelerate int8 only. Their float harnesses use the
    # portable reference dispatcher explicitly rather than implying acceleration.
    ("esp-nn", "float32"): "reference",
    ("esp-nn", "int8"): "esp-nn",
    ("cmsis-nn", "float32"): "reference",
    ("cmsis-nn", "int8"): "cmsis-nn",
}

OP_TYPE_BY_CODE = {opcode: name for name, opcode in OP_TYPE_MAP.items()}


def resolve_kernel_backend(codegen_backend: str, dtype: DTypeMode) -> str:
    """Resolve a CLI deployment backend and plan dtype to a dispatcher."""
    try:
        return _CODEGEN_KERNEL_SELECTION[(codegen_backend, dtype)]
    except KeyError as exc:
        choices = ", ".join(CODEGEN_BACKENDS)
        raise ValueError(
            f"Unknown backend/dtype combination: {codegen_backend!r}/{dtype}. "
            f"Choose a backend from: {choices}"
        ) from exc


@lru_cache(maxsize=None)
def effective_operators(kernel_backend: str) -> frozenset[str]:
    """Return operators executable through native code or explicit fallback."""
    try:
        capability = KERNEL_CAPABILITIES[kernel_backend]
    except KeyError as exc:
        raise ValueError(f"Unknown kernel backend: {kernel_backend!r}") from exc

    operators = capability.native_operators
    if capability.fallback is not None:
        operators = operators | effective_operators(capability.fallback)
    return operators


def operator_route(kernel_backend: str, operator: str) -> str | None:
    """Return the dispatcher that implements an operator, following fallback."""
    capability = KERNEL_CAPABILITIES[kernel_backend]
    if operator in capability.native_operators:
        return kernel_backend
    if capability.fallback is not None:
        return operator_route(capability.fallback, operator)
    return None


def describe_codegen_route(codegen_backend: str, dtype: DTypeMode) -> str:
    """Return a user-facing, fallback-explicit kernel route description."""
    kernel_backend = resolve_kernel_backend(codegen_backend, dtype)
    capability = KERNEL_CAPABILITIES[kernel_backend]
    if kernel_backend != codegen_backend:
        if codegen_backend == "reference":
            return kernel_backend
        return (
            f"{kernel_backend} (explicit {dtype} fallback; "
            f"{codegen_backend} acceleration is int8-only)"
        )
    if capability.fallback is not None:
        return f"{kernel_backend} -> {capability.fallback} fallback"
    return kernel_backend


def capability_rows() -> list[dict[str, str | int]]:
    """Return stable rows suitable for CLI tables or generated documentation."""
    rows: list[dict[str, str | int]] = []
    for operator, opcode in sorted(OP_TYPE_MAP.items(), key=lambda item: item[1]):
        row: dict[str, str | int] = {"operator": operator, "opcode": opcode}
        for backend in KERNEL_CAPABILITIES:
            route = operator_route(backend, operator)
            if route is None:
                row[backend] = "unsupported"
            elif route == backend:
                row[backend] = "native"
            else:
                row[backend] = f"fallback:{route}"
        rows.append(row)
    return rows
