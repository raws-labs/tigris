"""Spatial partitioning - receptive field computation and tile solving.

For stages whose peak memory exceeds the SRAM budget, determines whether ops
are spatially tileable, computes the receptive field (halo overlap), and solves
for tile dimensions that fit within budget.

Also detects *streamable chains*: consecutive tiled stages whose intermediate
tensors can stay in fast memory as tiles, avoiding full-size slow allocation.
"""

import math
from enum import Enum

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW, TILE_AXIS_NONE
from tigris.graph.ir import (
    AnalyzedGraph,
    OpNode,
    Stage,
    TilePlan,
)


# Op classification


class TileCategory(Enum):
    CONV = "conv"
    POOL = "pool"
    POINTWISE = "pointwise"
    UNTILEABLE = "untileable"


# Spatial tiling is an execution contract, not a purely mathematical
# classification.  Keep this list aligned with exec_stage_tiled in the runtime:
# every listed op is only accepted on a stage axis audited below. Unknown and
# unaudited ops fail closed as UNTILEABLE.
_OP_CATEGORY: dict[str, TileCategory] = {
    # Spatial ops whose height geometry the runtime propagates.
    "Conv": TileCategory.CONV,
    "Conv1D": TileCategory.CONV,
    "DepthwiseConv": TileCategory.CONV,
    "MaxPool": TileCategory.POOL,
    "AveragePool": TileCategory.POOL,
    # Audited, shape-preserving runtime kernels.
    "Relu": TileCategory.POINTWISE,
    "Relu6": TileCategory.POINTWISE,
    "Sigmoid": TileCategory.POINTWISE,
    "Tanh": TileCategory.POINTWISE,
    "Add": TileCategory.POINTWISE,
    "Mul": TileCategory.POINTWISE,
    "Concat": TileCategory.POINTWISE,
}


# Dynamic binary ops (Add, Mul) whose second operand is an independent
# tensor. Neither the rank-3 axis-1 executor nor the rank-4 HW (2D) executor
# guarantees that operand is co-tiled with a spatial op's output: both load
# every stage input using the spatial op's own tile geometry (length stripe
# or HW halo rectangle), so a binary op combined with a spatial op can read
# the wrong region or size from its second operand. Safe only in stages with
# no spatial op. Shared between the rank-3 and rank-4 eligibility checks
# below since the hazard is the same in both.
_BINARY_OPS = frozenset({"Add", "Mul"})

# Rank-3 NLC stages have a deliberately narrower axis-1 contract than rank-4
# NHWC stages.  Unary pointwise operators preserve the current length and may
# surround one Conv1D.  Dynamic binary operators are safe only in a
# pointwise-only stage: combining them with a strided Conv1D could expose
# external operands at different length resolutions.
_RANK3_AXIS1_UNARY_OPS = frozenset({
    "Relu",
    "Relu6",
    "Sigmoid",
    "Tanh",
})
_RANK3_AXIS1_OPS = _RANK3_AXIS1_UNARY_OPS | _BINARY_OPS | {"Conv1D"}

# Conservative per-tile allocation alignment for the backend-agnostic tiled
# working-set model. The runtime rounds every fast-arena tile allocation up to
# its target's TIGRIS_TENSOR_ALIGN (Xtensa 8, aarch64 16, x86_64 32, default 4);
# 32 is the maximum of that standard set, so aligning the compiler's estimate to
# it never under-counts against any of them (align_up is monotonic in the
# alignment). Under-counting would let the solver emit a tile the runtime's
# stage_2d_fast_bytes check rejects, which is exactly the bug this guards.
_CONSERVATIVE_TENSOR_ALIGN = 32


def classify_op(op_type: str) -> TileCategory:
    """Classify an op type into a tile category. Unknown ops are UNTILEABLE."""
    return _OP_CATEGORY.get(op_type, TileCategory.UNTILEABLE)


# Receptive field computation


def compute_receptive_field(ops: list[OpNode]) -> tuple[int, int]:
    """Compute the height and width receptive fields for a sequence of ops.

    Walks the ops in reverse once, accumulating RF and jump (cumulative
    stride) independently for the height and width axes.
    Returns (rf_h, rf_w).

    For pointwise ops, RF and jump are unchanged on both axes.
    For conv/pool ops, RF grows based on effective kernel size.
    """
    rf_h = 1
    jump_h = 1
    rf_w = 1
    jump_w = 1

    for op in reversed(ops):
        cat = classify_op(op.op_type)
        if cat in (TileCategory.CONV, TileCategory.POOL):
            kernel_h = _get_kernel_h(op)
            stride_h = _get_stride_h(op)
            dilation_h = _get_dilation_h(op)

            effective_kh = dilation_h * (kernel_h - 1) + 1
            rf_h = rf_h + (effective_kh - 1) * jump_h
            jump_h = jump_h * stride_h

            kernel_w = _get_kernel_w(op)
            stride_w = _get_stride_w(op)
            dilation_w = _get_dilation_w(op)

            effective_kw = dilation_w * (kernel_w - 1) + 1
            rf_w = rf_w + (effective_kw - 1) * jump_w
            jump_w = jump_w * stride_w

    return rf_h, rf_w


def _get_kernel_h(op: OpNode) -> int:
    """Get the height dimension of the kernel (first element of kernel_shape)."""
    ks = op.attrs.get("kernel_shape")
    if ks and len(ks) >= 1:
        return int(ks[0])
    return 1


def _get_stride_h(op: OpNode) -> int:
    """Get the height dimension of the stride."""
    strides = op.attrs.get("strides")
    if strides and len(strides) >= 1:
        return int(strides[0])
    return 1


def _get_dilation_h(op: OpNode) -> int:
    """Get the height dimension of the dilation."""
    dilations = op.attrs.get("dilations")
    if dilations and len(dilations) >= 1:
        return int(dilations[0])
    return 1


def _get_kernel_w(op: OpNode) -> int:
    """Get the width dimension of the kernel (second element of kernel_shape)."""
    ks = op.attrs.get("kernel_shape")
    if ks and len(ks) >= 2:
        return int(ks[1])
    return 1


def _get_stride_w(op: OpNode) -> int:
    """Get the width dimension of the stride."""
    strides = op.attrs.get("strides")
    if strides and len(strides) >= 2:
        return int(strides[1])
    return 1


def _get_dilation_w(op: OpNode) -> int:
    """Get the width dimension of the dilation."""
    dilations = op.attrs.get("dilations")
    if dilations and len(dilations) >= 2:
        return int(dilations[1])
    return 1


# Tile solver


def solve_2d_tile(
    budget: int,
    peak: int,
    input_h: int,
    input_w: int,
    halo_h: int,
    halo_w: int,
) -> tuple[int, int] | None:
    """Largest square-ish (tile_h, tile_w) whose proportional working set fits budget.

    Mirrors the 1D proportional model already used for HEIGHT_OR_LENGTH in
    partition_spatial(): ``tile_h = floor(budget*input_h/peak) - halo`` and
    ``tiled_peak = int(peak * (tile_h + halo) / input_h)``. ``peak`` is the
    stage's activation-only peak_bytes (compute_lifetimes skips constant
    tensors, so weights/bias are never counted there); it is scaled by the
    fraction of the haloed input area the tile covers:

        tiled_peak(th, tw) = int(peak * (th + halo_h) * (tw + halo_w)
                                  / (input_h * input_w))

    Neither this solver nor the 1D one models resident weight/scratch bytes
    against the tiled budget; that is a known pre-existing gap, and the
    runtime backstops it by validating the emitted tile shape against the
    real fast arena and failing closed.

    Returns None if even a 1x1 core tile does not fit.
    """
    if input_h <= 0 or input_w <= 0:
        return None

    def tiled_peak(th: int, tw: int) -> int:
        return int(peak * (th + halo_h) * (tw + halo_w) / (input_h * input_w))

    if tiled_peak(1, 1) > budget:
        return None

    th = tw = max(min(input_h, input_w), 1)

    # Shrink the larger side first, keeping the core roughly square, until it fits.
    while tiled_peak(th, tw) > budget:
        if th >= tw and th > 1:
            th -= 1
        elif tw > 1:
            tw -= 1
        else:
            th = tw = 1
            break

    th = min(th, input_h)
    tw = min(tw, input_w)
    return (max(th, 1), max(tw, 1))


def _stage_2d_eligible(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode]
) -> bool:
    """A stage may attempt HW tiling only if it is a standalone rank-4 stage
    with at most one spatial op, no binary op, and all of whose ops implement
    the HW tile contract. Multi-spatial-op stages are excluded: the runtime
    executor's 2D contract is audited only for a single composed spatial op
    per stage.

    Binary ops (Add, Mul) are excluded even though they pass the per-op HW
    contract check below: exec_stage_tiled_2d loads every stage input using
    the same conv input-halo rectangle, so a binary op's second operand
    (e.g. a residual skip tensor) is not guaranteed to be co-tiled with the
    spatial op's output at that rectangle. Admitting a stage like
    [Conv, Add(conv_out, skip)] as 2D would load the skip operand with the
    wrong region and size and silently produce a wrong result. This is the
    conservative, fail-closed choice: such a stage falls back to the 1D path
    (or fails closed if 1D is also infeasible). Co-tiled-binary 2D tiling is
    a future refinement, not attempted here.
    """
    if _stage_io_ranks(ag, stage) != {4}:
        return False
    spatial_count = sum(
        1
        for op in stage_ops
        if classify_op(op.op_type) in (TileCategory.CONV, TileCategory.POOL)
    )
    if spatial_count > 1:
        return False
    if any(op.op_type in _BINARY_OPS for op in stage_ops):
        return False
    # Concat carries the same hazard as the binary ops above: it takes an
    # independent second operand, and exec_stage_tiled_2d loads every stage
    # input with the spatial op's own input-halo rectangle, which is not
    # guaranteed to be co-tiled with a distinct Concat operand at output
    # resolution. Exclude it from HW eligibility for symmetry with _BINARY_OPS;
    # such a stage falls back to the 1D path or fails closed. A Concat that is
    # a stage head/fan-in is unaffected (it is not a single-spatial-op stage).
    if any(op.op_type == "Concat" for op in stage_ops):
        return False
    return all(_op_supports_axis(op, TILE_AXIS_HW) for op in stage_ops)


def _stage_is_convtranspose_2d(stage_ops: list[OpNode]) -> bool:
    """A stage qualifies for the dedicated ConvTranspose 2D solve iff exactly
    one op is a ConvTranspose (group == 1, unit dilation) and every other op is
    an audited unary pointwise wrapper.

    ConvTranspose is deliberately kept UNTILEABLE in _OP_CATEGORY, so it never
    reaches the height (1D), HW-conv, or chain paths; this predicate gates the
    isolated 2D-output-extent branch that replaces them for it. The group and
    dilation checks are defense in depth: validate_operator_support already
    rejects group != 1 or non-unit dilation, but a not-yet-rejected op must
    never fall into the 2D solve.

    "Audited pointwise" reuses the same set _stage_2d_eligible admits: the
    POINTWISE category minus _BINARY_OPS and Concat. Those take an independent
    second operand that the shared input-halo rectangle load does not co-tile,
    so they are excluded here for the same reason. This bounds Phase 1.3c to a
    ConvTranspose plus optional unary pointwise.
    """
    convtranspose = [op for op in stage_ops if op.op_type == "ConvTranspose"]
    if len(convtranspose) != 1:
        return False
    ct = convtranspose[0]
    if int(ct.attrs.get("group", 1)) != 1:
        return False
    if _get_dilation_h(ct) != 1 or _get_dilation_w(ct) != 1:
        return False
    for op in stage_ops:
        if op is ct:
            continue
        if op.op_type in _BINARY_OPS or op.op_type == "Concat":
            return False
        if classify_op(op.op_type) != TileCategory.POINTWISE:
            return False
    return True


def _stage_rank4_input_infos(ag: AnalyzedGraph, stage: Stage) -> list:
    """The stage's external activation inputs that are rank-4 tensors.

    Mirrors the runtime's stage_inputs (tigris_stage_inputs): only the declared
    external activation inputs, never an op's weight/bias operands. No fallback
    to op inputs, which would wrongly pull in the ConvTranspose weight tensor.
    """
    infos = []
    for name in stage.input_tensors:
        info = ag.tensors.get(name)
        if info and len(info.shape) == 4:
            infos.append(info)
    return infos


def _solve_convtranspose_2d(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode], budget: int
) -> TilePlan:
    """Emit a 2D (HW) tile plan for an over-budget ConvTranspose stage.

    The tile grid is normalized over the expanded OUTPUT extent, but the
    per-tile working set is sized the SAME way the runtime does in
    stage_2d_fast_bytes (tigris_executor.c): a resident packed INPUT tile plus
    every op's packed output tile, all live at once. This matters because a
    ConvTranspose's input tile does NOT shrink by the output-area ratio - it
    inverts to a fixed-halo rectangle
    ``in_tile = (out_tile + eff_k + stride - 1)//stride + 2`` (clamped to the
    full input), which a proportional peak model under-counts. Under-counting
    made the runtime reject every emitted tile (ERR_TILE); this models the real
    working set so every emitted tile fits.

    Fails closed (a non-tileable TilePlan) when the output/input extent cannot
    be determined, or when no output tile - not even a 1x1 core - fits budget.
    """
    ct = next((op for op in stage_ops if op.op_type == "ConvTranspose"), None)
    if ct is None:  # guarded by _stage_is_convtranspose_2d; defensive
        return TilePlan(
            tileable=False,
            warnings=[f"Stage {stage.stage_id}: no ConvTranspose op in stage"],
        )

    out_h = _find_output_extent(ag, stage)
    out_w = _find_output_extent_width(ag, stage)
    if out_h <= 0 or out_w <= 0:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id}: cannot determine ConvTranspose "
                f"output extent"
            ],
        )

    in_infos = _stage_rank4_input_infos(ag, stage)
    if not in_infos:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id}: cannot determine ConvTranspose "
                f"input extent"
            ],
        )
    full_in_h = int(in_infos[0].shape[2])  # NCHW: H at dim 2
    full_in_w = int(in_infos[0].shape[3])  # NCHW: W at dim 3

    # group == 1 and unit dilation are enforced by _stage_is_convtranspose_2d;
    # compute eff_k with dilation folded in anyway to match the runtime exactly.
    eff_kh = _get_dilation_h(ct) * (_get_kernel_h(ct) - 1) + 1
    eff_kw = _get_dilation_w(ct) * (_get_kernel_w(ct) - 1) + 1
    stride_h = _get_stride_h(ct)
    stride_w = _get_stride_w(ct)
    if full_in_h <= 0 or full_in_w <= 0 or stride_h <= 0 or stride_w <= 0:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id}: malformed ConvTranspose geometry"
            ],
        )

    align = max(ag.tensor_alignment, _CONSERVATIVE_TENSOR_ALIGN)

    def working_set(th: int, tw: int) -> int:
        """Runtime stage_2d_fast_bytes for one (th, tw) output tile."""
        # ConvTranspose input tile: inverts to a smaller fixed-halo rectangle,
        # clamped to the full input, exactly as the runtime computes it.
        in_tile_h = min((th + eff_kh + stride_h - 1) // stride_h + 2, full_in_h)
        in_tile_w = min((tw + eff_kw + stride_w - 1) // stride_w + 2, full_in_w)
        total = 0
        for info in in_infos:
            total += _align_up(
                int(info.shape[0]) * in_tile_h * in_tile_w
                * int(info.shape[1]) * info.elem_size,
                align,
            )
        # Walk the op sequence tracking the running tile extent: it starts at
        # the input tile, the single spatial op resizes it to the output tile,
        # pointwise ops preserve it. Matches the runtime's cur_h/cur_w walk.
        cur_h, cur_w = in_tile_h, in_tile_w
        for op in stage_ops:
            is_spatial = op is ct
            ah = th if is_spatial else cur_h
            aw = tw if is_spatial else cur_w
            for name in op.outputs:
                info = ag.tensors.get(name)
                if info and len(info.shape) == 4:
                    total += _align_up(
                        int(info.shape[0]) * ah * aw
                        * int(info.shape[1]) * info.elem_size,
                        align,
                    )
            if is_spatial:
                cur_h, cur_w = th, tw
        return total

    # Fail closed if even a 1x1 output tile overflows the budget.
    if working_set(1, 1) > budget:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id} minimum 2D ConvTranspose tile still "
                f"exceeds budget ({budget:,} bytes)"
            ],
        )

    # Largest output tile whose runtime working set fits, maximizing tile area
    # (fewest tiles). working_set is monotonic non-decreasing in both th and tw,
    # so per th the largest feasible tw is a binary search, and once th at tw==1
    # overflows no larger th can fit at any width.
    best_th, best_tw, best_area = 1, 1, 1
    for th in range(1, out_h + 1):
        if working_set(th, 1) > budget:
            break
        lo, hi, tw_for_th = 1, out_w, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if working_set(th, mid) <= budget:
                tw_for_th = mid
                lo = mid + 1
            else:
                hi = mid - 1
        area = th * tw_for_th
        if area > best_area:
            best_area, best_th, best_tw = area, th, tw_for_th

    th, tw = best_th, best_tw
    return TilePlan(
        tileable=True,
        axis=TILE_AXIS_HW,
        tile_height=th,
        tile_width=tw,
        num_tiles=math.ceil(out_h / th) * math.ceil(out_w / tw),
        halo=0,
        receptive_field=1,
        original_height=out_h,
        tiled_peak_bytes=working_set(th, tw),
        overhead_bytes=0,
        warnings=[],
    )


def partition_spatial(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Analyze each stage and attach a TilePlan where needed.

    Only stages whose peak_bytes exceed mem_budget are analyzed.
    Stages that fit within budget get no tile_plan (None).
    """
    if not ag.stages or ag.mem_budget <= 0:
        return ag

    budget = ag.mem_budget

    for stage in ag.stages:
        if stage.peak_bytes <= budget:
            continue  # fits, no tiling needed

        stage_ops = [ag.ops[i] for i in stage.op_indices]

        # ConvTranspose stays UNTILEABLE in _OP_CATEGORY on purpose (so it is
        # auto-excluded from chains, the 1D height solve, and receptive-field
        # composition). Its 2D tiling is handled here by a dedicated isolated
        # branch that grids the expanded OUTPUT extent. Every other stage falls
        # through to the existing byte-identical path below.
        if _stage_is_convtranspose_2d(stage_ops):
            stage.tile_plan = _solve_convtranspose_2d(ag, stage, stage_ops, budget)
            continue

        tile_axis = _stage_tile_axis(ag, stage, stage_ops)

        # Check if all ops are tileable
        untileable: list[str] = []
        for op in stage_ops:
            cat = classify_op(op.op_type)
            if cat == TileCategory.UNTILEABLE or not _op_supports_axis(
                op, tile_axis
            ):
                untileable.append(f"{op.name} ({op.op_type})")

        if tile_axis == TILE_AXIS_NONE or untileable:
            stage.tile_plan = TilePlan(
                tileable=False,
                untileable_ops=untileable,
                warnings=[
                    f"Stage {stage.stage_id} has no audited common tile axis"
                    if tile_axis == TILE_AXIS_NONE
                    else f"Stage {stage.stage_id} contains untileable ops"
                ],
            )
            continue

        # Compute receptive field
        rf_h, rf_w = compute_receptive_field(stage_ops)
        halo = rf_h - 1

        # Axis 1 in the serialized NHWC/NLC layout maps to H/L at source dim 2.
        input_h = _find_input_extent(ag, stage, tile_axis)
        if input_h <= 0:
            stage.tile_plan = TilePlan(
                tileable=False,
                warnings=[f"Stage {stage.stage_id}: cannot determine spatial height"],
            )
            continue

        # Solve for tile height
        peak = stage.peak_bytes
        tile_h = math.floor(budget * input_h / peak) - halo
        tile_h = max(tile_h, 1)

        num_tiles = math.ceil(input_h / tile_h)

        # Estimate tiled peak memory
        tiled_peak = int(peak * (tile_h + halo) / input_h)

        # A single-row height tile that still overflows the budget cannot be
        # rescued by any smaller height-only tile: height is already at its
        # floor. If the stage is eligible for 2D (HW) tiling, try shrinking
        # both axes together before falling back to the 1D infeasible
        # warning below.
        min_2d_tile_infeasible = False
        if (
            tile_h == 1
            and tiled_peak > budget
            and _stage_2d_eligible(ag, stage, stage_ops)
        ):
            input_w = _find_input_extent_width(ag, stage)
            if input_w > 0:
                halo_w = rf_w - 1
                shape = solve_2d_tile(
                    budget=budget,
                    peak=peak,
                    input_h=input_h,
                    input_w=input_w,
                    halo_h=halo,
                    halo_w=halo_w,
                )
                if shape is not None:
                    tile_h_2d, tile_w_2d = shape
                    tiled_peak_2d = int(
                        peak
                        * (tile_h_2d + halo)
                        * (tile_w_2d + halo_w)
                        / (input_h * input_w)
                    )
                    stage.tile_plan = TilePlan(
                        tileable=True,
                        axis=TILE_AXIS_HW,
                        tile_height=tile_h_2d,
                        tile_width=tile_w_2d,
                        num_tiles=math.ceil(input_h / tile_h_2d)
                        * math.ceil(input_w / tile_w_2d),
                        halo=halo,
                        receptive_field=rf_h,
                        original_height=input_h,
                        tiled_peak_bytes=tiled_peak_2d,
                        overhead_bytes=0,
                        warnings=[],
                    )
                    continue

                # solve_2d_tile was attempted and even a 1x1 core tile does
                # not fit the budget. Mark this stage distinctly so the
                # surfaced diagnostic names the 2D tile instead of falling
                # back to the generic 1D minimum-tile message below.
                min_2d_tile_infeasible = True

        # Overhead: extra halo reads per tile boundary
        # Each internal tile boundary reads halo rows extra from the input
        halo_tensor_bytes = _estimate_halo_bytes(ag, stage, halo, input_h)
        overhead = halo_tensor_bytes * max(num_tiles - 1, 0)

        warnings: list[str] = []
        if min_2d_tile_infeasible:
            warnings.append(
                f"Stage {stage.stage_id} minimum 2D tile still exceeds "
                f"budget ({budget:,} bytes)"
            )
        elif tiled_peak > budget:
            warnings.append(
                f"Stage {stage.stage_id} tiled peak ({tiled_peak:,} bytes) "
                f"still exceeds budget ({budget:,} bytes)"
            )

        stage.tile_plan = TilePlan(
            tileable=True,
            axis=tile_axis,
            tile_height=tile_h,
            num_tiles=num_tiles,
            halo=halo,
            receptive_field=rf_h,
            original_height=input_h,
            tiled_peak_bytes=tiled_peak,
            overhead_bytes=overhead,
            warnings=warnings,
            min_2d_tile_infeasible=min_2d_tile_infeasible,
        )

    return ag


def _stage_io_ranks(ag: AnalyzedGraph, stage: Stage) -> set[int]:
    """Return concrete ranks for a stage's external activation tensors."""
    names = [*stage.input_tensors, *stage.output_tensors]
    if not names:
        return set()
    ranks: set[int] = set()
    for name in names:
        info = ag.tensors.get(name)
        if info is None:
            return set()
        ranks.add(len(info.shape))
    return ranks


def _stage_tile_axis(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode]
) -> int:
    """Select an audited serialized activation axis for a standalone stage."""
    ranks = _stage_io_ranks(ag, stage)
    if ranks == {4} and all(op.op_type != "Conv1D" for op in stage_ops):
        return TILE_AXIS_HEIGHT_OR_LENGTH
    if ranks == {3} and stage_ops:
        op_types = [op.op_type for op in stage_ops]
        conv_count = op_types.count("Conv1D")
        if (
            all(op_type in _RANK3_AXIS1_OPS for op_type in op_types)
            and conv_count <= 1
            and not (
                conv_count == 1
                and any(
                    op_type in _BINARY_OPS
                    for op_type in op_types
                )
            )
        ):
            return TILE_AXIS_HEIGHT_OR_LENGTH
    return TILE_AXIS_NONE


def _op_supports_axis(op: OpNode, axis: int) -> bool:
    """Fail closed unless an operator implements the selected tile contract."""
    if axis == TILE_AXIS_HEIGHT_OR_LENGTH:
        if op.op_type == "Conv1D":
            return True
        return op.op_type in _OP_CATEGORY
    if axis == TILE_AXIS_HW:
        # Rank-4 spatial/pointwise/channel-Concat set only; Conv1D is rank-3
        # and has no width axis to tile, even though it shares the CONV
        # category with Conv in _OP_CATEGORY, so it must be excluded here
        # explicitly rather than relying on the membership check alone.
        if op.op_type == "Conv1D":
            return False
        return op.op_type in _OP_CATEGORY
    return False


def _find_input_extent(ag: AnalyzedGraph, stage: Stage, axis: int) -> int:
    """Find the H/L extent that serializes as axis 1 (source NCHW/NCL dim 2)."""
    if axis not in (TILE_AXIS_HEIGHT_OR_LENGTH, TILE_AXIS_HW):
        return 0
    return _find_source_dim_extent(ag, stage, dim=2, ranks={3, 4})


def _find_input_extent_width(ag: AnalyzedGraph, stage: Stage) -> int:
    """Find the W extent that serializes as axis 2 (source NCHW dim 3).

    Only rank-4 tensors carry a width dimension; HW tiling never applies to
    the rank-3 NCL layout.
    """
    return _find_source_dim_extent(ag, stage, dim=3, ranks={4})


def _find_source_dim_extent(
    ag: AnalyzedGraph, stage: Stage, dim: int, ranks: set[int]
) -> int:
    """Find a stage's source-shape extent at ``dim`` among candidate tensors."""
    # Check stage input tensors first, then look at first op's inputs
    candidates = stage.input_tensors.copy()
    if not candidates:
        first_op = ag.ops[stage.op_indices[0]]
        candidates = [n for n in first_op.inputs if n in ag.tensors]

    for name in candidates:
        info = ag.tensors.get(name)
        if info and len(info.shape) in ranks:
            return int(info.shape[dim])

    return 0


def _find_output_extent(ag: AnalyzedGraph, stage: Stage) -> int:
    """Find the H extent of the stage OUTPUT (source NCHW/NCL dim 2)."""
    return _find_output_dim_extent(ag, stage, dim=2, ranks={3, 4})


def _find_output_extent_width(ag: AnalyzedGraph, stage: Stage) -> int:
    """Find the W extent of the stage OUTPUT (source NCHW dim 3).

    Only rank-4 tensors carry a width dimension, so this mirrors the rank
    restriction of _find_input_extent_width.
    """
    return _find_output_dim_extent(ag, stage, dim=3, ranks={4})


def _find_output_dim_extent(
    ag: AnalyzedGraph, stage: Stage, dim: int, ranks: set[int]
) -> int:
    """Find a stage's OUTPUT-shape extent at ``dim`` among candidate tensors.

    Mirrors _find_source_dim_extent but reads the stage's output tensors
    (falling back to the last op's outputs), so ConvTranspose tiling grids over
    the expanded output extent rather than the pre-upsample input.
    """
    candidates = stage.output_tensors.copy()
    if not candidates:
        last_op = ag.ops[stage.op_indices[-1]]
        candidates = [n for n in last_op.outputs if n in ag.tensors]

    for name in candidates:
        info = ag.tensors.get(name)
        if info and len(info.shape) in ranks:
            return int(info.shape[dim])

    return 0


def _estimate_halo_bytes(ag: AnalyzedGraph, stage, halo: int, input_h: int) -> int:
    """Estimate bytes for one halo region of the stage's input tensor."""
    candidates = stage.input_tensors.copy()
    if not candidates:
        first_op = ag.ops[stage.op_indices[0]]
        candidates = [n for n in first_op.inputs if n in ag.tensors]

    for name in candidates:
        info = ag.tensors.get(name)
        if info and len(info.shape) in {3, 4}:
            # bytes per row = total_bytes / H
            if input_h > 0:
                return int(info.size_bytes * halo / input_h)

    return 0


# Chain detection and solving


def _is_stage_tileable(ag: AnalyzedGraph, stage: Stage) -> bool:
    """Check if a stage may be part of a streamable CHAIN (used only by
    detect_chains): all ops implement the schema-v4 height-stripe contract and
    all stage I/O is 4D.

    The runtime composes Conv, DepthwiseConv, MaxPool, and AveragePool geometry
    while pointwise operators preserve the current stripe height.
    """
    for op_i in stage.op_indices:
        cat = classify_op(ag.ops[op_i].op_type)
        if cat == TileCategory.UNTILEABLE:
            return False
    # All inputs and outputs must be 4D
    for name in stage.input_tensors:
        info = ag.tensors.get(name)
        if not info or len(info.shape) != 4:
            return False
    for name in stage.output_tensors:
        info = ag.tensors.get(name)
        if not info or len(info.shape) != 4:
            return False
    return True


def _tensor_consumer_count(ag: AnalyzedGraph) -> dict[str, int]:
    """Count how many stages consume each tensor as an input."""
    counts: dict[str, int] = {}
    for stage in ag.stages:
        for name in stage.input_tensors:
            counts[name] = counts.get(name, 0) + 1
    return counts


def detect_chains(ag: AnalyzedGraph) -> list[list[int]]:
    """Detect streamable chains: maximal runs of consecutive tiled stages.

    A chain [s_i, s_{i+1}, ...] requires for each adjacent pair (s_k, s_{k+1}):
      1. Both stages are spatially tileable (all ops tileable, 4D I/O)
      2. s_k has exactly one output tensor
      3. s_{k+1} has exactly one input tensor
      4. s_k's output IS s_{k+1}'s input (same tensor)
      5. No other stage consumes that intermediate tensor (no fan-out)

    Returns a list of chain groups, each a sorted list of stage indices (length >= 2).
    """
    if not ag.stages or len(ag.stages) < 2:
        return []

    consumer_counts = _tensor_consumer_count(ag)
    chains: list[list[int]] = []
    current_chain: list[int] = []

    for i, stage in enumerate(ag.stages):
        if not current_chain:
            # Try to start a chain at this stage
            if _is_stage_tileable(ag, stage):
                current_chain = [i]
            continue

        prev_stage = ag.stages[current_chain[-1]]

        # Check chaining conditions between prev and current
        can_chain = (
            _is_stage_tileable(ag, stage)
            and len(prev_stage.output_tensors) == 1
            and len(stage.input_tensors) == 1
            and prev_stage.output_tensors[0] == stage.input_tensors[0]
            and consumer_counts.get(prev_stage.output_tensors[0], 0) == 1
            # The chain intermediate is streamed tile-by-tile and never
            # materialized to slow memory; if it is also a model output, chaining
            # would leave that output unwritten. Keep it out of the chain.
            and prev_stage.output_tensors[0] not in ag.model_outputs
        )

        if can_chain:
            current_chain.append(i)
        else:
            # Flush current chain if length >= 2
            if len(current_chain) >= 2:
                chains.append(current_chain)
            # Try starting a new chain from this stage
            if _is_stage_tileable(ag, stage):
                current_chain = [i]
            else:
                current_chain = []

    # Flush last chain
    if len(current_chain) >= 2:
        chains.append(current_chain)

    return chains


def _get_stage_spatial_params(ag: AnalyzedGraph, stage: Stage) -> tuple[int, int, int]:
    """Compose (eff_kh, stride_h, 1) across ALL spatial ops in a stage.

    The runtime executor composes receptive fields from Conv, DepthwiseConv,
    MaxPool, and AveragePool ops in a stage when validating chain tile heights,
    so the compiler must match by composing here too.

    Returns (1, 1, 1) for pointwise-only stages.
    """
    comp_stride = 1
    comp_eff_kh = 1
    found = False
    for op_i in stage.op_indices:
        op = ag.ops[op_i]
        cat = classify_op(op.op_type)
        if cat in (TileCategory.CONV, TileCategory.POOL):
            kh = _get_kernel_h(op)
            sh = _get_stride_h(op)
            dh = _get_dilation_h(op)
            ekh = dh * (kh - 1) + 1
            # Compose: eff_kh_new = eff_kh + (ekh - 1) * stride
            comp_eff_kh = comp_eff_kh + (ekh - 1) * comp_stride
            comp_stride = comp_stride * sh
            found = True
    if not found:
        return 1, 1, 1
    # Return composed params as (eff_kh, stride, 1) - dilation already folded in
    return comp_eff_kh, comp_stride, 1


def _back_propagate_tile_heights(
    chain_params: list[tuple[int, int, int]],
    last_out_h: int,
) -> list[tuple[int, int]]:
    """Back-propagate tile heights through a chain.

    Given the output tile height of the last stage, compute (in_h, out_h) for
    each stage from last to first.

    Args:
        chain_params: [(kernel_h, stride_h, dilation_h)] per stage.
        last_out_h: output tile height for the last stage.

    Returns:
        [(in_h, out_h)] per stage, indexed from first to last.
    """
    heights: list[tuple[int, int]] = []
    out_h = last_out_h

    for kh, sh, dh in reversed(chain_params):
        eff_kh = dh * (kh - 1) + 1
        in_h = out_h * sh + max(eff_kh - sh, 0)
        heights.append((in_h, out_h))
        out_h = in_h  # this stage's input = previous stage's output

    heights.reverse()
    return heights


def _align_up(x: int, align: int) -> int:
    """Round up to the deployment memory model's tensor alignment."""
    return (x + align - 1) & ~(align - 1)


def _chain_fast_bytes(
    ag: AnalyzedGraph,
    chain_stages: list[Stage],
    heights: list[tuple[int, int]],
) -> int:
    """Compute total fast memory needed for one tile iteration of a chain.

    With bump allocation (no mid-chain reset), all buffers accumulate:
      - First stage's input tile (loaded from slow)
      - All op output tiles across all stages

    Each allocation is rounded up to the graph's conservative deployment
    alignment to match the runtime bump allocator's alignment overhead.

    Within each stage, forward-computes the intermediate height through
    spatial ops to match the runtime executor's memory calculation.
    """
    total = 0

    # First stage input tile
    first_stage = chain_stages[0]
    in_h = heights[0][0]
    for name in first_stage.input_tensors:
        info = ag.tensors.get(name)
        if info and len(info.shape) == 4:
            N, C, H, W = info.shape  # NCHW in Python IR
            total += _align_up(N * in_h * W * C * info.elem_size,
                               ag.tensor_alignment)

    # All op output tiles - forward-compute height through spatial ops
    for s_idx, stage in enumerate(chain_stages):
        cur_h = heights[s_idx][0]  # start with stage input height
        for op_i in stage.op_indices:
            op = ag.ops[op_i]
            cat = classify_op(op.op_type)
            # Spatial ops reduce height
            if cat in (TileCategory.CONV, TileCategory.POOL):
                kh = _get_kernel_h(op)
                sh = _get_stride_h(op)
                dh = _get_dilation_h(op)
                ekh = dh * (kh - 1) + 1
                pt = op.attrs.get("pads", [0])[0] if "pads" in op.attrs else 0
                cur_h = (cur_h + pt - ekh) // sh + 1
            for out_name in op.outputs:
                info = ag.tensors.get(out_name)
                if info and len(info.shape) == 4:
                    N, C, H, W = info.shape
                    total += _align_up(N * cur_h * W * C * info.elem_size,
                                       ag.tensor_alignment)

    return total


def solve_chain_tile_height(
    ag: AnalyzedGraph,
    chain_stage_indices: list[int],
) -> int:
    """Solve for the maximum last-stage output tile height that fits in fast.

    Binary searches for the largest out_tile_h such that all tile buffers
    (first stage input + all op outputs across chain stages) fit in the
    fast memory budget.

    Returns 0 if the chain cannot be tiled (budget too small for even 1 row).
    """
    budget = ag.mem_budget
    if budget <= 0:
        return 0

    chain_stages = [ag.stages[i] for i in chain_stage_indices]
    chain_params = [_get_stage_spatial_params(ag, s) for s in chain_stages]

    # Find the output height of the last stage
    last_stage = chain_stages[-1]
    last_out_h = 0
    for name in last_stage.output_tensors:
        info = ag.tensors.get(name)
        if info and len(info.shape) == 4:
            last_out_h = int(info.shape[2])  # NCHW: H is dim 2
            break
    if last_out_h <= 0:
        return 0

    # Binary search for maximum out_tile_h
    lo, hi = 1, last_out_h
    best = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        heights = _back_propagate_tile_heights(chain_params, mid)
        needed = _chain_fast_bytes(ag, chain_stages, heights)
        if needed <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def detect_and_solve_chains(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Detect chains and solve tile heights. Sets chain fields on stages.

    Should be called after partition_spatial() has determined individual tiling.
    """
    chains = detect_chains(ag)

    for chain in chains:
        tile_h = solve_chain_tile_height(ag, chain)
        if tile_h <= 0:
            continue  # chain doesn't fit, leave stages as standalone tiled

        head_id = chain[0]
        chain_len = len(chain)

        for s_idx in chain:
            stage = ag.stages[s_idx]
            stage.chain_id = head_id
            stage.chain_len = chain_len
            # Clear individual tile plan - chain executor handles tiling
            stage.tile_plan = None

        # Store tile height on the head stage
        ag.stages[head_id].chain_tile_h = tile_h

        # A chain recomputes iff any member stage has a positive composed
        # halo (eff_kh - stride > 0). Mark the head so the runtime can later
        # keep the shared boundary rows in a line buffer instead of
        # redundantly recomputing them per tile. Memory-neutral, no threshold.
        recomputes = any(
            (eff_kh - stride) > 0
            for eff_kh, stride, _dh in (
                _get_stage_spatial_params(ag, ag.stages[s_idx]) for s_idx in chain
            )
        )
        if recomputes:
            ag.stages[head_id].line_buffered = True

    return ag
