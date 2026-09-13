"""Fail-closed validation for ConvTranspose subset and the general
capabilities cross-check.

validate_operator_support already rejects operators missing from
OP_TYPE_MAP (wire-encodability). It did not, until this change, also
require a runtime route to exist for a wire-encodable operator, so an
operator like Pad (present in OP_TYPE_MAP, no dispatcher in
tigris.capabilities) compiled successfully and only failed on-device. This
file also covers the ConvTranspose-specific subset the runtime kernels
implement: group == 1 and unit dilation only.
"""

from onnx import TensorProto

from tigris.analysis.validation import validate_operator_support
from tigris.capabilities import KERNEL_CAPABILITIES, effective_operators
from tigris.emitters.binary.defs import OP_TYPE_MAP
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo


def build_convtranspose_graph(group: int = 1, dilation: int = 1) -> AnalyzedGraph:
    """A single-op graph: one ConvTranspose with a configurable group/dilation."""
    op = OpNode(
        name="conv_transpose",
        op_type="ConvTranspose",
        inputs=["input"],
        outputs=["output"],
        attrs={
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [0, 0, 0, 0],
            "dilations": [dilation, dilation],
            "group": group,
        },
    )
    return AnalyzedGraph(
        ops=[op],
        tensors={
            "input": TensorInfo("input", (1, 4, 4, 4), TensorProto.FLOAT),
            "output": TensorInfo("output", (1, 4, 9, 9), TensorProto.FLOAT),
        },
    )


def build_single_op_graph(op_type: str) -> AnalyzedGraph:
    """A single-op graph with no operator-specific attrs, for gate tests."""
    op = OpNode(
        name="lone_op",
        op_type=op_type,
        inputs=["input"],
        outputs=["output"],
    )
    return AnalyzedGraph(
        ops=[op],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
    )


def test_convtranspose_group_rejected():
    ag = build_convtranspose_graph(group=2)

    result = validate_operator_support(ag)

    assert not result.supported
    assert any("group" in issue.reason.lower() for issue in result.issues)


def test_convtranspose_dilation_rejected():
    ag = build_convtranspose_graph(dilation=2)

    result = validate_operator_support(ag)

    assert not result.supported
    assert any("dilation" in issue.reason.lower() for issue in result.issues)


def test_convtranspose_default_group_and_dilation_is_supported():
    ag = build_convtranspose_graph()

    result = validate_operator_support(ag)

    assert result.supported, result.describe()


def test_encodable_op_without_route_rejected():
    # Cross-check: an op present in OP_TYPE_MAP (wire-encodable) but with no
    # route on any backend must still be rejected at compile. Confirm at
    # least one such op currently exists (it does: Pad, Sub, Div,
    # LeakyRelu, BatchNormalization, InstanceNormalization, MatMul,
    # ReduceMean, Squeeze, Unsqueeze, GlobalMaxPool, Clip); if that ever
    # becomes empty this assertion documents the gate must be exercised via
    # a synthetic op_type instead.
    routed = frozenset().union(
        *(effective_operators(backend) for backend in KERNEL_CAPABILITIES)
    )
    unrouted = sorted(set(OP_TYPE_MAP) - routed)
    assert unrouted, "no wire-encodable-but-unrouted op found; use a synthetic op_type"

    ag = build_single_op_graph(op_type=unrouted[0])

    result = validate_operator_support(ag)

    assert not result.supported
    assert any(
        "no runtime kernel" in issue.reason.lower()
        or "unsupported" in issue.reason.lower()
        for issue in result.issues
    )


def test_routed_operator_is_not_rejected_by_capabilities_gate():
    # Sanity check the new gate does not reject an op that does have a route.
    ag = build_single_op_graph(op_type="Relu")

    result = validate_operator_support(ag)

    assert result.supported, result.describe()
