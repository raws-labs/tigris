"""Tests for 2D (height + width) receptive field computation and tile solving."""

from onnx import TensorProto

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW
from tigris.analysis.partition_spatial import compute_receptive_field, partition_spatial, solve_2d_tile
from tigris.graph.ir import AnalyzedGraph, MemoryBudget, OpNode, Stage, TensorInfo


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


# 2D tile solver
#
# solve_2d_tile mirrors the 1D proportional model already used by
# partition_spatial() for HEIGHT_OR_LENGTH: tiled_peak(th, tw) scales the
# stage's whole peak_bytes (which includes weights/bias/scratch that stay
# resident regardless of tile shape) by the fraction of the haloed input
# area the tile covers. This is deliberately conservative rather than the
# activation-only "(th+hh)(tw+hw)*C_in + th*tw*C_out" formula, because that
# formula omits resident non-activation memory and would let a 2D shape
# through that overflows the fast arena at runtime.


def test_solve_2d_tile_fits_budget():
    # peak_bytes models a [512,512] int8 activation with C=256 baked in
    # (512*512*256 = 67,108,864). Halo 2x2. A full 1-row tile does not fit
    # 32K, but a small square core does.
    peak_bytes = 512 * 512 * 256
    shape = solve_2d_tile(
        budget=32 * 1024,
        peak=peak_bytes,
        input_h=512,
        input_w=512,
        halo_h=2,
        halo_w=2,
    )
    assert shape is not None
    th, tw = shape
    assert th >= 1 and tw >= 1
    tiled_peak = int(peak_bytes * (th + 2) * (tw + 2) / (512 * 512))
    assert tiled_peak <= 32 * 1024


def test_solve_2d_tile_infeasible_returns_none():
    # peak_bytes models a [64,64] int8 activation with C=4096 baked in.
    # Even the 1x1 core (with 2x2 halo) overflows a 1K budget.
    peak_bytes = 64 * 64 * 4096
    assert (
        solve_2d_tile(
            budget=1024,
            peak=peak_bytes,
            input_h=64,
            input_w=64,
            halo_h=2,
            halo_w=2,
        )
        is None
    )


# Axis selection: HW only kicks in once the 1D height solve is infeasible
# even at a single-row tile.


def build_high_res_conv_graph(h: int, w: int, c: int, mem_budget: int) -> AnalyzedGraph:
    """Build a single-stage AnalyzedGraph for one 3x3 stride-1 Conv on an
    int8 [1,c,h,w] activation.

    peak_bytes is the whole-tensor working set (channels included), so a
    single-row height tile costs exactly c*w bytes, matching the 1D
    solver's proportional row-byte model and the comments below.
    """
    op = OpNode(
        name="conv",
        op_type="Conv",
        inputs=["input"],
        outputs=["output"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]},
    )
    stage = Stage(
        stage_id=0,
        op_indices=[0],
        input_tensors=["input"],
        output_tensors=["output"],
        peak_bytes=c * h * w,
    )
    return AnalyzedGraph(
        ops=[op],
        stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget),
    )


def plan_for_single_stage(ag: AnalyzedGraph):
    assert len(ag.stages) == 1
    return ag.stages[0].tile_plan


def test_axis_is_hw_only_when_height_infeasible():
    # [1,256,256] row = 64K; a single row does not fit a 24K budget.
    ag = build_high_res_conv_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    ag = partition_spatial(ag)
    stage_plan = plan_for_single_stage(ag)
    assert stage_plan.axis == TILE_AXIS_HW
    assert stage_plan.tileable is True


def test_axis_stays_height_when_1d_feasible():
    # [1,64,32] row = 2K, fits a 64K budget without needing HW tiling.
    ag = build_high_res_conv_graph(h=256, w=64, c=32, mem_budget=64 * 1024)
    ag = partition_spatial(ag)
    stage_plan = plan_for_single_stage(ag)
    assert stage_plan.axis == TILE_AXIS_HEIGHT_OR_LENGTH
