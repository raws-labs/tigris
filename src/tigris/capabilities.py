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
    "Conv",
    "DepthwiseConv",
    "Relu",
    "Relu6",
    "Sigmoid",
    "Tanh",
    "Add",
    "Mul",
    "Conv1D",
    "GlobalAveragePool",
    "AveragePool",
    "Gemm",
    "Reshape",
    "Flatten",
    "MaxPool",
    "Concat",
    "Resize",
    "Softmax",
    "Transpose",
})

_S8_REFERENCE_OPERATORS = _FLOAT_REFERENCE_OPERATORS


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
        "DepthwiseConv": "falls back for dilation other than 1",
        "AveragePool": (
            "falls back when tiled or when input/output quantization differs"
        ),
    },
    "cmsis-nn": {
        "Conv": "falls back when tiled; non-tiled dilation remains native",
        "DepthwiseConv": (
            "falls back when tiled; non-tiled dilation remains native"
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
OPERATOR_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "Add": ("dynamic operands must have identical shapes; no general broadcasting",),
    "Mul": ("dynamic operands must have identical shapes; no general broadcasting",),
    "AveragePool": (
        "explicit padding, floor output sizing, unit dilation, and count_include_pad=0",
    ),
    "MaxPool": (
        "explicit padding, floor output sizing, unit dilation, and no indices output",
    ),
    "Concat": ("rank-4 channel-axis concatenation",),
    "Conv1D": ("standalone rank-3 length tiling on serialized axis 1",),
    "GlobalAveragePool": ("untiled execution",),
    "Resize": (
        "rank-4 nearest-neighbor integer H/W upscaling; untiled execution",
    ),
    "Softmax": ("final axis only; untiled execution",),
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
