"""Tests for 2D (height + width) receptive field computation."""

from tigris import TILE_AXIS_HW
from tigris.analysis.partition_spatial import compute_receptive_field
from tigris.graph.ir import OpNode


def make_conv_op(kernel, stride, dilation):
    """Build the minimal OpNode a Conv-like op needs for receptive field analysis."""
    return OpNode(
        name="conv",
        op_type="Conv",
        inputs=[],
        outputs=[],
        attrs={
            "kernel_shape": list(kernel),
            "strides": list(stride),
            "dilations": list(dilation),
        },
    )


def test_hw_axis_constant():
    assert TILE_AXIS_HW == 3


def test_receptive_field_returns_both_axes():
    # A single 3x3 stride-1 conv: rf_h == rf_w == 3.
    ops = [make_conv_op(kernel=(3, 3), stride=(1, 1), dilation=(1, 1))]
    rf_h, rf_w = compute_receptive_field(ops)
    assert (rf_h, rf_w) == (3, 3)


def test_receptive_field_asymmetric_kernel():
    # 5x3 kernel, stride 2x1: rf_h = 5, rf_w = 3.
    ops = [make_conv_op(kernel=(5, 3), stride=(2, 1), dilation=(1, 1))]
    rf_h, rf_w = compute_receptive_field(ops)
    assert (rf_h, rf_w) == (5, 3)
