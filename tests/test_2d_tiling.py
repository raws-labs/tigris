"""Tests for 2D (height + width) receptive field computation and tile solving."""

from onnx import TensorProto

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW
from tigris.analysis.partition_spatial import (
    _op_supports_axis,
    _stage_2d_eligible,
    compute_receptive_field,
    partition_spatial,
    solve_2d_tile,
)
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
# stage's activation-only peak_bytes (compute_lifetimes skips constant
# tensors, so this never includes weights/bias) by the fraction of the
# haloed input area the tile covers, instead of the brief's
# "(th+hh)(tw+hw)*C_in + th*tw*C_out" formula. Resident weight/scratch bytes
# are not modeled against the tiled budget by either the 1D or 2D solver (a
# known pre-existing gap); the runtime backstops this by validating the
# emitted tile shape against the real fast arena and failing closed.


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


# Op eligibility for the HW axis. Conv1D shares the CONV category with Conv
# in _OP_CATEGORY, but it is rank-3 and has no width axis to tile, so it
# must be excluded from TILE_AXIS_HW explicitly rather than relying on the
# _OP_CATEGORY membership check alone.


def test_op_supports_axis_excludes_conv1d_for_hw():
    conv1d = OpNode(name="c1d", op_type="Conv1D", inputs=[], outputs=[], attrs={})
    conv2d = OpNode(name="c2d", op_type="Conv", inputs=[], outputs=[], attrs={})
    assert _op_supports_axis(conv1d, TILE_AXIS_HW) is False
    assert _op_supports_axis(conv2d, TILE_AXIS_HW) is True


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


# 2D eligibility must exclude binary ops (Add, Mul).
#
# exec_stage_tiled_2d loads every stage input with the conv's own
# input-halo rectangle. A binary op's second operand (e.g. a residual skip
# tensor) is a separate full-resolution tensor that is not guaranteed to be
# co-tiled with the conv's output at that rectangle, so a stage like
# [Conv, Add(conv_out, skip)] admitted as 2D would load the skip operand
# with the wrong region and size and silently produce a wrong result. This
# mirrors the precedent already enforced on the rank-3 axis-1 path
# (_BINARY_OPS / _stage_tile_axis), which forbids combining a Conv1D with
# any binary op for the same co-tiling reason.


def build_high_res_conv_add_graph(h: int, w: int, c: int, mem_budget: int) -> AnalyzedGraph:
    """Single-stage graph: Conv followed by Add against a separate
    full-resolution skip tensor, at the same [1,c,h,w] shape and peak_bytes
    model as build_high_res_conv_graph, so the same budget that forces 2D
    tiling for a bare Conv applies here too.
    """
    conv = OpNode(
        name="conv",
        op_type="Conv",
        inputs=["input"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]},
    )
    add = OpNode(
        name="add",
        op_type="Add",
        inputs=["conv_out", "skip"],
        outputs=["output"],
        attrs={},
    )
    stage = Stage(
        stage_id=0,
        op_indices=[0, 1],
        input_tensors=["input", "skip"],
        output_tensors=["output"],
        peak_bytes=c * h * w,
    )
    return AnalyzedGraph(
        ops=[conv, add],
        stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "conv_out": TensorInfo("conv_out", (1, c, h, w), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget),
    )


def build_high_res_conv_sigmoid_graph(h: int, w: int, c: int, mem_budget: int) -> AnalyzedGraph:
    """Single-stage graph: Conv followed by a unary Sigmoid wrapper, at the
    same [1,c,h,w] shape and peak_bytes model as build_high_res_conv_graph.

    Used as the control: a spatial op plus a UNARY pointwise wrapper must
    still be admitted for 2D tiling, proving the binary-op guard is scoped
    to binary ops only and does not over-exclude.
    """
    conv = OpNode(
        name="conv",
        op_type="Conv",
        inputs=["input"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]},
    )
    sigmoid = OpNode(
        name="sigmoid",
        op_type="Sigmoid",
        inputs=["conv_out"],
        outputs=["output"],
        attrs={},
    )
    stage = Stage(
        stage_id=0,
        op_indices=[0, 1],
        input_tensors=["input"],
        output_tensors=["output"],
        peak_bytes=c * h * w,
    )
    return AnalyzedGraph(
        ops=[conv, sigmoid],
        stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "conv_out": TensorInfo("conv_out", (1, c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget),
    )


def test_stage_2d_eligible_excludes_binary_op():
    # Hand-built stage with a binary op: must not be 2D eligible.
    ag = build_high_res_conv_add_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    stage = ag.stages[0]
    assert _stage_2d_eligible(ag, stage, ag.ops) is False


def test_stage_2d_eligible_allows_spatial_plus_unary_op():
    # Hand-built stage with only a spatial op plus a unary wrapper: must
    # still be 2D eligible, proving the guard does not over-exclude.
    ag = build_high_res_conv_sigmoid_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    stage = ag.stages[0]
    assert _stage_2d_eligible(ag, stage, ag.ops) is True


def test_binary_op_stage_does_not_go_hw():
    # End to end through partition_spatial(): same budget that forces a bare
    # Conv stage to TILE_AXIS_HW (see test_axis_is_hw_only_when_height_infeasible)
    # must NOT do so once a binary Add against an uncotiled skip tensor is
    # in the stage. It falls back to the 1D height axis instead.
    ag = build_high_res_conv_add_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    ag = partition_spatial(ag)
    stage_plan = plan_for_single_stage(ag)
    assert stage_plan.axis != TILE_AXIS_HW


def test_conv_plus_unary_stage_still_goes_hw():
    # Control: the same budget still drives a spatial-plus-unary stage to
    # TILE_AXIS_HW, proving the binary-op guard is scoped to binary ops only.
    ag = build_high_res_conv_sigmoid_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    ag = partition_spatial(ag)
    stage_plan = plan_for_single_stage(ag)
    assert stage_plan.axis == TILE_AXIS_HW
