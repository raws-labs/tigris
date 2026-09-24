"""Tests for 2D (height + width) receptive field computation and tile solving."""

import math

from onnx import TensorProto

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW, TILE_AXIS_NONE
from tigris.analysis.partition_spatial import (
    _op_supports_axis,
    _stage_2d_eligible,
    _stage_tile_axis,
    compute_receptive_field,
    partition_spatial,
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
# The runtime runs a (tile_height, tile_width) grid over the stage OUTPUT and
# checks each tile against the fast arena: one packed input tile, whose
# rectangle it back-computes as (t - 1) * stride + eff_k, plus every op's
# packed output tile. The compiler has to size the tile the same way.


def _strided_depthwise_stages(tmp_path, budget):
    import numpy as np
    import onnx
    from onnx import helper

    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.analysis.partition_temporal import partition_temporal
    from tigris.loaders import load_model

    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [96, 1, 3, 3],
        np.ones((96, 1, 3, 3), dtype=np.float32).ravel().tolist())
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w"], ["output"], group=96,
                          kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])],
        "dw",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 96, 34, 34])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 96, 17, 17])],
        [weight])
    path = str(tmp_path / "dw.onnx")
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    ag = compute_memory_timeline(compute_lifetimes(load_model(path)))
    return partition_spatial(partition_temporal(ag, budget))


def _runtime_2d_bytes(th, tw):
    """stage_2d_fast_bytes for the 96-channel float stride-2 depthwise stage."""
    def aligned(n):
        return (n + 31) // 32 * 32
    in_h, in_w = min((th - 1) * 2 + 3, 34), min((tw - 1) * 2 + 3, 34)
    return aligned(in_h * in_w * 96 * 4) + aligned(th * tw * 96 * 4)


def test_strided_2d_tile_is_an_output_tile_that_fits(tmp_path):
    budget = 16 * 1024
    ag = _strided_depthwise_stages(tmp_path, budget)
    tp = ag.stages[0].tile_plan
    assert tp.tileable and tp.axis == TILE_AXIS_HW
    assert tp.num_tiles == math.ceil(17 / tp.tile_height) * math.ceil(17 / tp.tile_width)
    assert _runtime_2d_bytes(tp.tile_height, tp.tile_width) <= budget
    assert tp.tiled_peak_bytes == _runtime_2d_bytes(tp.tile_height, tp.tile_width)
    # The tile is the largest area the budget allows.
    assert all(
        _runtime_2d_bytes(th, tw) > budget
        for th in range(1, 18) for tw in range(1, 18)
        if th * tw > tp.tile_height * tp.tile_width
    )


def test_strided_2d_tile_infeasible_is_marked(tmp_path):
    # A 1x1 output tile still reads a 3x3x96 float input window, 3.5K with
    # its output; a 2K budget fits no 2D tile.
    ag = _strided_depthwise_stages(tmp_path, 2 * 1024)
    tp = ag.stages[0].tile_plan
    assert tp.axis != TILE_AXIS_HW
    assert tp.min_2d_tile_infeasible


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


def build_high_res_conv_concat_graph(h: int, w: int, c: int, mem_budget: int) -> AnalyzedGraph:
    """Single-stage graph: Conv followed by a channel-axis Concat against a
    separate full-resolution operand. Concat takes an independent second
    operand the 2D executor's shared-rectangle load does not co-tile, so the
    stage must not be 2D eligible (same hazard as the binary-op case).
    """
    conv = OpNode(
        name="conv",
        op_type="Conv",
        inputs=["input"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]},
    )
    concat = OpNode(
        name="concat",
        op_type="Concat",
        inputs=["conv_out", "other"],
        outputs=["output"],
        attrs={"axis": 1},
    )
    stage = Stage(
        stage_id=0,
        op_indices=[0, 1],
        input_tensors=["input", "other"],
        output_tensors=["output"],
        peak_bytes=c * h * w,
    )
    return AnalyzedGraph(
        ops=[conv, concat],
        stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "conv_out": TensorInfo("conv_out", (1, c, h, w), TensorProto.INT8),
            "other": TensorInfo("other", (1, c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, 2 * c, h, w), TensorProto.INT8),
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


def test_stage_2d_eligible_excludes_concat():
    # A Conv+Concat stage carries the same non-co-tiled-operand hazard as a
    # binary op, so it must not be 2D eligible either.
    ag = build_high_res_conv_concat_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
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


# Pre-spatial co-tiled skip connections (Add/Concat consumed BEFORE the
# spatial op) are a different hazard shape than the post-spatial cases
# above: the skip operand sits at the spatial op's INPUT resolution, which
# is exactly the halo rectangle exec_stage_tiled_2d loads every stage input
# at. When every stage-external operand is same-H/W, the shared-rectangle
# load co-tiles it correctly, so these stages must be admitted. Different
# resolution operands keep the fail-closed rejection.


def build_high_res_add_conv_graph(h, w, c, mem_budget):
    """Pre-spatial: Add(input, skip) -> Conv. The Add is consumed by the Conv,
    both operands same [1,c,h,w]. Co-tileable skip -> must be 2D eligible."""
    add = OpNode(name="add", op_type="Add",
                 inputs=["input", "skip"], outputs=["added"], attrs={})
    conv = OpNode(name="conv", op_type="Conv",
                  inputs=["added"], outputs=["output"],
                  attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]})
    stage = Stage(stage_id=0, op_indices=[0, 1],
                  input_tensors=["input", "skip"], output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=[add, conv], stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, c, h, w), TensorProto.INT8),
            "added": TensorInfo("added", (1, c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget))


def build_high_res_concat_conv_graph(h, w, c, mem_budget):
    """Pre-spatial: Concat(up, skip) -> Conv. Concat on channel axis, both
    operands same H/W. Co-tileable skip -> must be 2D eligible."""
    concat = OpNode(name="concat", op_type="Concat",
                    inputs=["up", "skip"], outputs=["cat"], attrs={"axis": 1})
    conv = OpNode(name="conv", op_type="Conv",
                  inputs=["cat"], outputs=["output"],
                  attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]})
    stage = Stage(stage_id=0, op_indices=[0, 1],
                  input_tensors=["skip", "up"], output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=[concat, conv], stages=[stage],
        tensors={
            "up": TensorInfo("up", (1, c, h, w), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, c, h, w), TensorProto.INT8),
            "cat": TensorInfo("cat", (1, 2 * c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget))


def build_high_res_concat_conv_diffres_graph(h, w, c, mem_budget):
    """Pre-spatial Concat but the skip is a DIFFERENT resolution (h//2 x w//2).
    Not co-tileable -> must stay rejected (clause 3 fail-closed)."""
    concat = OpNode(name="concat", op_type="Concat",
                    inputs=["up", "skip"], outputs=["cat"], attrs={"axis": 1})
    conv = OpNode(name="conv", op_type="Conv",
                  inputs=["cat"], outputs=["output"],
                  attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]})
    stage = Stage(stage_id=0, op_indices=[0, 1],
                  input_tensors=["skip", "up"], output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=[concat, conv], stages=[stage],
        tensors={
            "up": TensorInfo("up", (1, c, h, w), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, c, h // 2, w // 2), TensorProto.INT8),
            "cat": TensorInfo("cat", (1, 2 * c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget))


def build_high_res_concat_const_skip_graph(h, w, c, mem_budget):
    """Pre-spatial Concat whose skip operand is a rank-4 CONSTANT (initializer),
    same H/W as `up`. A constant is never a stage input, so it escapes the
    external same-H/W co-tile check in _cotileable_skip_operands, yet the 2D
    executor cannot tile-offset a full-size constant against the spatial op's
    input-halo rectangle. Must fail closed."""
    concat = OpNode(name="concat", op_type="Concat",
                    inputs=["up", "const_skip"], outputs=["cat"], attrs={"axis": 1})
    conv = OpNode(name="conv", op_type="Conv",
                  inputs=["cat"], outputs=["output"],
                  attrs={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1]})
    stage = Stage(stage_id=0, op_indices=[0, 1],
                  input_tensors=["up"], output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=[concat, conv], stages=[stage],
        tensors={
            "up": TensorInfo("up", (1, c, h, w), TensorProto.INT8),
            "const_skip": TensorInfo(
                "const_skip", (1, c, h, w), TensorProto.INT8, is_constant=True
            ),
            "cat": TensorInfo("cat", (1, 2 * c, h, w), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h, w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget))


def build_post_spatial_add_skip_graph(h, w, c, mem_budget):
    """Post-spatial: Conv(stride 2) -> Add(conv_out, skip). The Add is consumed
    AFTER the strided Conv, and `skip` is a stage-external tensor at the Conv's
    OUTPUT resolution. The tiled executors load every stage input at the Conv's
    INPUT-halo rectangle, so a strided Conv would read `skip` at stride*out_start
    rows (and past its height) on interior tiles. Both the 1D-height and the 2D
    tile paths must fail closed on this shape (-> exec_stage_normal)."""
    conv = OpNode(name="conv", op_type="Conv",
                  inputs=["input"], outputs=["conv_out"],
                  attrs={"kernel_shape": [1, 1], "strides": [2, 2], "dilations": [1, 1]})
    add = OpNode(name="add", op_type="Add",
                 inputs=["conv_out", "skip"], outputs=["output"], attrs={})
    stage = Stage(stage_id=0, op_indices=[0, 1],
                  input_tensors=["input", "skip"], output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=[conv, add], stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, c, h, w), TensorProto.INT8),
            "conv_out": TensorInfo("conv_out", (1, c, h // 2, w // 2), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, c, h // 2, w // 2), TensorProto.INT8),
            "output": TensorInfo("output", (1, c, h // 2, w // 2), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=mem_budget))


def test_stage_tile_axis_rejects_post_spatial_external_skip():
    # Conv(stride 2) -> Add(conv_out, external skip): the 1D-height executor loads
    # the skip at the Conv's input rows, mis-tiling every interior tile (and
    # reading past the skip's height). The axis selection must fail closed to
    # TILE_AXIS_NONE so the stage runs untiled, matching the 2D path's rejection.
    ag = build_post_spatial_add_skip_graph(h=256, w=256, c=64, mem_budget=24 * 1024)
    assert _stage_tile_axis(ag, ag.stages[0], ag.ops) == TILE_AXIS_NONE


def test_stage_2d_eligible_rejects_post_spatial_external_skip():
    # The 2D path already rejects this shape; keep both paths in lockstep.
    ag = build_post_spatial_add_skip_graph(h=256, w=256, c=64, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is False


def test_stage_tile_axis_admits_pre_spatial_add_skip():
    # A pre-spatial Add(input, skip) -> Conv is co-tileable and loaded correctly
    # at the input resolution, so the 1D-height axis stays admitted.
    ag = build_high_res_add_conv_graph(h=256, w=256, c=64, mem_budget=24 * 1024)
    assert _stage_tile_axis(ag, ag.stages[0], ag.ops) == TILE_AXIS_HEIGHT_OR_LENGTH


def test_stage_2d_eligible_admits_pre_spatial_add_skip():
    ag = build_high_res_add_conv_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is True


def test_stage_2d_eligible_admits_pre_spatial_concat_skip():
    ag = build_high_res_concat_conv_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is True


def test_stage_2d_eligible_rejects_constant_concat_skip():
    # A rank-4 constant Concat operand is not a stage input, so it escapes the
    # external same-H/W co-tile check, but it cannot be tile-offset for the 2D
    # executor's shared input-halo rectangle load. The stage must fail closed.
    ag = build_high_res_concat_const_skip_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is False


def test_stage_2d_eligible_rejects_diffres_skip():
    ag = build_high_res_concat_conv_diffres_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is False


def test_pre_spatial_concat_skip_goes_hw():
    # End to end: the admitted pre-spatial concat-skip stage tiles on TILE_AXIS_HW.
    ag = build_high_res_concat_conv_graph(h=256, w=256, c=256, mem_budget=24 * 1024)
    ag = partition_spatial(ag)
    assert plan_for_single_stage(ag).axis == TILE_AXIS_HW


# ConvTranspose 2D tiling over the OUTPUT extent.
#
# ConvTranspose stays UNTILEABLE in _OP_CATEGORY (so it is auto-excluded from
# chains, the 1D height solve, and receptive-field composition). Its 2D tiling
# is handled by a dedicated isolated branch in partition_spatial that grids the
# OUTPUT extent (the expanded, post-upsample shape) with zero halo. These tests
# exercise that branch end to end.


def build_convtranspose_graph(
    in_hw: tuple[int, int],
    stride: int,
    kernel: int,
    in_ch: int,
    out_ch: int,
) -> AnalyzedGraph:
    """Single-stage AnalyzedGraph for one ConvTranspose on an int8 activation.

    group == 1 and unit dilation (the runtime-supported subset). The output
    tensor shape is the ONNX-inferred expanded extent for pads == 0:
        out = (in - 1) * stride + kernel

    peak_bytes is the whole OUTPUT activation working set (channels included),
    matching the c*h*w convention of build_high_res_conv_graph above, so a
    single output pixel costs exactly out_ch bytes and the proportional
    tiled-peak model tiles the expanded output area.
    """
    in_h, in_w = in_hw
    out_h = (in_h - 1) * stride + kernel
    out_w = (in_w - 1) * stride + kernel
    op = OpNode(
        name="conv_transpose",
        op_type="ConvTranspose",
        inputs=["input"],
        outputs=["output"],
        attrs={
            "kernel_shape": [kernel, kernel],
            "strides": [stride, stride],
            "pads": [0, 0, 0, 0],
            "dilations": [1, 1],
            "group": 1,
        },
    )
    stage = Stage(
        stage_id=0,
        op_indices=[0],
        input_tensors=["input"],
        output_tensors=["output"],
        peak_bytes=out_ch * out_h * out_w,
    )
    return AnalyzedGraph(
        ops=[op],
        stages=[stage],
        tensors={
            "input": TensorInfo("input", (1, in_ch, in_h, in_w), TensorProto.INT8),
            "output": TensorInfo("output", (1, out_ch, out_h, out_w), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=0),
    )


def _ct_stage(ag):
    """The single ConvTranspose stage."""
    return next(
        s
        for s in ag.stages
        if any(ag.ops[i].op_type == "ConvTranspose" for i in s.op_indices)
    )


def test_convtranspose_over_budget_tiles_2d():
    ag = build_convtranspose_graph(
        in_hw=(32, 32), stride=2, kernel=2, in_ch=8, out_ch=8
    )  # out 64x64
    stage = _ct_stage(ag)
    ag.budget = MemoryBudget(fast=stage.peak_bytes // 4)  # multiple tiles needed
    ag = partition_spatial(ag)
    tp = _ct_stage(ag).tile_plan
    assert tp.tileable and tp.axis == TILE_AXIS_HW
    assert tp.original_height == 64  # OUTPUT extent, not the 32-row input
    assert tp.halo == 0 and tp.receptive_field == 1
    tiles_h = math.ceil(tp.original_height / tp.tile_height)
    tiles_w = tp.num_tiles // tiles_h
    assert tiles_h > 1 and tiles_w > 1


def test_convtranspose_infeasible_budget_fails_closed():
    ag = build_convtranspose_graph(
        in_hw=(32, 32), stride=2, kernel=2, in_ch=8, out_ch=8
    )
    # 4 bytes is smaller than even a single output pixel's working set (out_ch
    # == 8 int8 bytes), so no output tile fits: the solve must fail closed.
    # (The brief's illustrative literal 256 assumes a many-channel stage; at
    # out_ch == 8 the honest per-pixel threshold is 8 bytes, matching the
    # brief's own "smaller than a 1x1 output tile working set" comment.)
    ag.budget = MemoryBudget(fast=4)
    ag = partition_spatial(ag)
    assert not _ct_stage(ag).tile_plan.tileable


def test_convtranspose_under_budget_untiled():
    ag = build_convtranspose_graph(
        in_hw=(32, 32), stride=2, kernel=2, in_ch=8, out_ch=8
    )
    stage = _ct_stage(ag)
    ag.budget = MemoryBudget(fast=stage.peak_bytes * 2)  # fits whole, no tiling
    ag = partition_spatial(ag)
    assert _ct_stage(ag).tile_plan is None


def _runtime_ct_working_set(
    th, tw, *, in_ch, out_ch, full_in_h, full_in_w,
    eff_kh, eff_kw, stride_h, stride_w, in_elem, out_elem, align=32,
):
    """Independent reimplementation of the runtime's stage_2d_fast_bytes
    (tigris-runtime/src/tigris_executor.c) for a single-input, single-output
    ConvTranspose: one resident input tile plus the output tile, each aligned.

    Kept deliberately separate from the compiler's own model so the test is a
    real property check, not a tautology: a proportional peak model (the BUG B
    cost model) emits a 32x32 tile here whose working set is 11104 bytes and
    exceeds the 8192-byte budget, which this assertion catches.
    """
    def align_up(n):
        return (n + align - 1) & ~(align - 1)

    in_tile_h = min((th + eff_kh + stride_h - 1) // stride_h + 2, full_in_h)
    in_tile_w = min((tw + eff_kw + stride_w - 1) // stride_w + 2, full_in_w)
    total = align_up(1 * in_tile_h * in_tile_w * in_ch * in_elem)
    total += align_up(1 * th * tw * out_ch * out_elem)
    return total


def test_convtranspose_emitted_tile_fits_runtime_working_set():
    # Pins the BUG B fix: the tile the solver emits for the over-budget case
    # must fit the runtime's real working-set model, not just a proportional
    # activation-area estimate. Recompute the runtime formula independently and
    # assert working_set <= budget (the invariant the proportional model broke).
    ag = build_convtranspose_graph(
        in_hw=(32, 32), stride=2, kernel=2, in_ch=8, out_ch=8
    )
    stage = _ct_stage(ag)
    budget = stage.peak_bytes // 4  # 8192
    ag.budget = MemoryBudget(fast=budget)
    ag = partition_spatial(ag)
    tp = _ct_stage(ag).tile_plan
    assert tp.tileable and tp.axis == TILE_AXIS_HW

    ws = _runtime_ct_working_set(
        tp.tile_height, tp.tile_width,
        in_ch=8, out_ch=8, full_in_h=32, full_in_w=32,
        eff_kh=2, eff_kw=2, stride_h=2, stride_w=2,
        in_elem=1, out_elem=1,
    )
    assert ws <= budget, (
        f"emitted tile {tp.tile_height}x{tp.tile_width} needs {ws} bytes "
        f"> budget {budget}"
    )
    # The solver also reports that working set as the tiled peak.
    assert tp.tiled_peak_bytes == ws

    # Sanity: the old proportional model would have emitted a 32x32 tile whose
    # working set overflows the budget, proving this is a non-trivial check.
    overflow = _runtime_ct_working_set(
        32, 32, in_ch=8, out_ch=8, full_in_h=32, full_in_w=32,
        eff_kh=2, eff_kw=2, stride_h=2, stride_w=2, in_elem=1, out_elem=1,
    )
    assert overflow > budget
