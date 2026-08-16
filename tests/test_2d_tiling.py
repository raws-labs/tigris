"""Tests for 2D (height + width) receptive field computation and tile solving."""

import math

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
