"""Tests for 2D (height + width) tile axis selection and tile solving."""

import math

import pytest
from onnx import TensorProto

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW, TILE_AXIS_NONE
from tigris.analysis.partition_spatial import (
    _op_supports_axis,
    _stage_2d_eligible,
    _stage_tile_axis,
    partition_spatial,
)
from tigris.graph.ir import AnalyzedGraph, MemoryBudget, OpNode, Stage, TensorInfo
from tigris.analysis.validation import validate_memory_plan

from conftest import save_onnx


# 2D tile solver
#
# The runtime runs a (tile_height, tile_width) grid over the stage OUTPUT and
# checks each tile against the fast arena: one packed input tile, whose
# rectangle it back-computes as (t - 1) * stride + eff_k, plus every op's
# packed output tile. The compiler has to size the tile the same way.


def _strided_depthwise_stages(tmp_path, budget):
    import numpy as np
    from onnx import helper

    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.analysis.partition_temporal import partition_temporal
    from tigris.loaders import load_model

    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [96, 1, 3, 3],
        np.ones((96, 1, 3, 3), dtype=np.float32).ravel().tolist())
    path = save_onnx(tmp_path, "dw",
                     [helper.make_node("Conv", ["input", "w"], ["output"], group=96,
                                       kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])],
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 96, 34, 34])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 96, 17, 17])],
                     [weight])
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


def test_infeasible_2d_stage_reports_the_smallest_2d_tile(tmp_path):
    # What the stage needs is the 1x1 output tile, not the single-row band of
    # the height-only solve, which is larger.
    ag = _strided_depthwise_stages(tmp_path, 2 * 1024)
    issue = next(issue for issue in validate_memory_plan(ag).issues if issue.stage_id == 0)
    assert issue.reason == "smallest 2D tile"
    assert issue.required_bytes == _runtime_2d_bytes(1, 1)
    assert issue.required_bytes < ag.stages[0].tile_plan.tiled_peak_bytes


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


def _conv(inp, out, kernel=3, stride=1):
    return OpNode(
        name="conv", op_type="Conv", inputs=[inp], outputs=[out],
        attrs={"kernel_shape": [kernel, kernel], "strides": [stride, stride],
               "dilations": [1, 1]},
    )


def single_stage(ops, shapes, inputs, c, h, w, mem_budget, constants=()):
    """One stage over all ops on int8 tensors, with the whole-tensor working
    set c*h*w as peak_bytes."""
    stage = Stage(stage_id=0, op_indices=list(range(len(ops))),
                  input_tensors=inputs, output_tensors=["output"],
                  peak_bytes=c * h * w)
    return AnalyzedGraph(
        ops=ops, stages=[stage],
        tensors={name: TensorInfo(name, shape, TensorProto.INT8,
                                  is_constant=name in constants)
                 for name, shape in shapes.items()},
        budget=MemoryBudget(fast=mem_budget))


def build_high_res_conv_graph(h, w, c, mem_budget):
    """One 3x3 stride-1 Conv on an int8 [1,c,h,w] activation."""
    full = (1, c, h, w)
    return single_stage([_conv("input", "output")],
                        {"input": full, "output": full}, ["input"], c, h, w, mem_budget)


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


def build_high_res_conv_add_graph(h, w, c, mem_budget):
    """Conv -> Add(conv_out, skip) against a separate full-resolution skip."""
    full = (1, c, h, w)
    add = OpNode(name="add", op_type="Add",
                 inputs=["conv_out", "skip"], outputs=["output"], attrs={})
    return single_stage(
        [_conv("input", "conv_out"), add],
        {"input": full, "conv_out": full, "skip": full, "output": full},
        ["input", "skip"], c, h, w, mem_budget)


def build_high_res_conv_concat_graph(h, w, c, mem_budget):
    """Conv -> channel Concat against a separate full-resolution operand,
    which the 2D executor's shared-rectangle load does not co-tile."""
    full = (1, c, h, w)
    concat = OpNode(name="concat", op_type="Concat",
                    inputs=["conv_out", "other"], outputs=["output"], attrs={"axis": 1})
    return single_stage(
        [_conv("input", "conv_out"), concat],
        {"input": full, "conv_out": full, "other": full, "output": (1, 2 * c, h, w)},
        ["input", "other"], c, h, w, mem_budget)


def build_high_res_conv_sigmoid_graph(h, w, c, mem_budget):
    """Conv -> unary Sigmoid: the control the binary-op guard must still admit."""
    full = (1, c, h, w)
    sigmoid = OpNode(name="sigmoid", op_type="Sigmoid",
                     inputs=["conv_out"], outputs=["output"], attrs={})
    return single_stage(
        [_conv("input", "conv_out"), sigmoid],
        {"input": full, "conv_out": full, "output": full},
        ["input"], c, h, w, mem_budget)


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
    """Pre-spatial: Add(input, skip) -> Conv, both operands same [1,c,h,w]."""
    full = (1, c, h, w)
    add = OpNode(name="add", op_type="Add",
                 inputs=["input", "skip"], outputs=["added"], attrs={})
    return single_stage(
        [add, _conv("added", "output")],
        {"input": full, "skip": full, "added": full, "output": full},
        ["input", "skip"], c, h, w, mem_budget)


def _concat_conv_graph(h, w, c, mem_budget, skip, skip_shape, inputs, constants=()):
    """Pre-spatial: Concat(up, skip) on the channel axis -> Conv."""
    full = (1, c, h, w)
    concat = OpNode(name="concat", op_type="Concat",
                    inputs=["up", skip], outputs=["cat"], attrs={"axis": 1})
    return single_stage(
        [concat, _conv("cat", "output")],
        {"up": full, skip: skip_shape, "cat": (1, 2 * c, h, w), "output": full},
        inputs, c, h, w, mem_budget, constants)


def build_high_res_concat_conv_graph(h, w, c, mem_budget):
    """Same-H/W skip: co-tileable, so 2D eligible."""
    return _concat_conv_graph(h, w, c, mem_budget, "skip", (1, c, h, w), ["skip", "up"])


def build_high_res_concat_conv_diffres_graph(h, w, c, mem_budget):
    """Skip at h//2 x w//2: not co-tileable, so it stays rejected."""
    return _concat_conv_graph(
        h, w, c, mem_budget, "skip", (1, c, h // 2, w // 2), ["skip", "up"])


def build_high_res_concat_const_skip_graph(h, w, c, mem_budget):
    """Skip is a same-H/W rank-4 constant. A constant is never a stage input,
    so it escapes the external same-H/W co-tile check in
    _cotileable_skip_operands, yet the 2D executor cannot tile-offset a
    full-size constant against the spatial op's input-halo rectangle."""
    return _concat_conv_graph(
        h, w, c, mem_budget, "const_skip", (1, c, h, w), ["up"], constants=("const_skip",))


def build_post_spatial_add_skip_graph(h, w, c, mem_budget):
    """Post-spatial: Conv(stride 2) -> Add(conv_out, skip). The Add is consumed
    AFTER the strided Conv, and `skip` is a stage-external tensor at the Conv's
    OUTPUT resolution. The tiled executors load every stage input at the Conv's
    INPUT-halo rectangle, so a strided Conv would read `skip` at stride*out_start
    rows (and past its height) on interior tiles. Both the 1D-height and the 2D
    tile paths must fail closed on this shape (-> exec_stage_normal)."""
    half = (1, c, h // 2, w // 2)
    add = OpNode(name="add", op_type="Add",
                 inputs=["conv_out", "skip"], outputs=["output"], attrs={})
    return single_stage(
        [_conv("input", "conv_out", kernel=1, stride=2), add],
        {"input": (1, c, h, w), "conv_out": half, "skip": half, "output": half},
        ["input", "skip"], c, h, w, mem_budget)


def test_stage_tile_axis_rejects_post_spatial_external_skip():
    # Conv(stride 2) -> Add(conv_out, external skip): the 1D-height executor loads
    # the skip at the Conv's input rows, mis-tiling every interior tile (and
    # reading past the skip's height). The axis selection must fail closed to
    # TILE_AXIS_NONE so the stage runs untiled, matching the 2D path's rejection.
    ag = build_post_spatial_add_skip_graph(h=256, w=256, c=64, mem_budget=24 * 1024)
    assert _stage_tile_axis(ag, ag.stages[0], ag.ops) == TILE_AXIS_NONE


def test_stage_tile_axis_admits_pre_spatial_add_skip():
    # A pre-spatial Add(input, skip) -> Conv is co-tileable and loaded correctly
    # at the input resolution, so the 1D-height axis stays admitted.
    ag = build_high_res_add_conv_graph(h=256, w=256, c=64, mem_budget=24 * 1024)
    assert _stage_tile_axis(ag, ag.stages[0], ag.ops) == TILE_AXIS_HEIGHT_OR_LENGTH


@pytest.mark.parametrize("build, c, eligible", [
    (build_high_res_conv_add_graph, 256, False),
    (build_high_res_conv_concat_graph, 256, False),
    (build_post_spatial_add_skip_graph, 64, False),
    (build_high_res_add_conv_graph, 256, True),
    (build_high_res_concat_const_skip_graph, 256, False),
    (build_high_res_concat_conv_diffres_graph, 256, False),
], ids=["binary-op", "concat", "post-spatial-skip", "pre-spatial-add-skip",
        "constant-concat-skip", "diffres-skip"])
def test_stage_2d_eligible(build, c, eligible):
    ag = build(h=256, w=256, c=c, mem_budget=24 * 1024)
    assert _stage_2d_eligible(ag, ag.stages[0], ag.ops) is eligible


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
    assert tp.original_height == 64  # OUTPUT extent, not the 32-row input
    assert tp.halo == 0 and tp.receptive_field == 1
    tiles_h = math.ceil(tp.original_height / tp.tile_height)
    tiles_w = tp.num_tiles // tiles_h
    assert tiles_h > 1 and tiles_w > 1

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
