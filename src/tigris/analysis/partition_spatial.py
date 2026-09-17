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
from tigris.analysis.partition_temporal import partition_temporal
from tigris.graph.ir import (
    AnalyzedGraph,
    Layout,
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
    "Sub": TileCategory.POINTWISE,
    "Mul": TileCategory.POINTWISE,
    "Concat": TileCategory.POINTWISE,
    # Softmax normalizes along the final stored dimension, which is the channel
    # axis in both NHWC and NLC. Neither tile axis cuts it: the height/length
    # axis and the width axis are both ahead of it, so a tile always holds
    # whole normalization rows and the kernel needs nothing from its
    # neighbours. Shape-preserving and halo-free, which is what POINTWISE means
    # here, even though the operator itself is a reduction.
    "Softmax": TileCategory.POINTWISE,
}


# Dynamic binary ops (Add, Mul) whose second operand is an independent
# tensor. Neither the rank-3 axis-1 executor nor the rank-4 HW (2D) executor
# guarantees that operand is co-tiled with a spatial op's output: both load
# every stage input using the spatial op's own tile geometry (length stripe
# or HW halo rectangle), so a binary op combined with a spatial op can read
# the wrong region or size from its second operand. Safe only in stages with
# no spatial op. Shared between the rank-3 and rank-4 eligibility checks
# below since the hazard is the same in both.
_BINARY_OPS = frozenset({"Add", "Sub", "Mul"})

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
    "Softmax",
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


def compute_receptive_field(
    ops: list[OpNode],
    weight_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[int, int]:
    """Compute the height and width receptive fields for a sequence of ops.

    Walks the ops in reverse once, accumulating RF and jump (cumulative
    stride) independently for the height and width axes.
    Returns (rf_h, rf_w).

    For pointwise ops, RF and jump are unchanged on both axes.
    For conv/pool ops, RF grows based on effective kernel size.

    ``weight_shapes`` (constant tensor name -> shape) lets a Conv that omits
    the ONNX ``kernel_shape`` attribute recover its kernel from the weight,
    matching the emitter; without it such ops fall back to a 1x1 kernel.
    """
    rf_h = 1
    jump_h = 1
    rf_w = 1
    jump_w = 1

    for op in reversed(ops):
        cat = classify_op(op.op_type)
        if cat in (TileCategory.CONV, TileCategory.POOL):
            kernel_h = _get_kernel_h(op, weight_shapes)
            stride_h = _get_stride_h(op)
            dilation_h = _get_dilation_h(op)

            effective_kh = dilation_h * (kernel_h - 1) + 1
            rf_h = rf_h + (effective_kh - 1) * jump_h
            jump_h = jump_h * stride_h

            kernel_w = _get_kernel_w(op, weight_shapes)
            stride_w = _get_stride_w(op)
            dilation_w = _get_dilation_w(op)

            effective_kw = dilation_w * (kernel_w - 1) + 1
            rf_w = rf_w + (effective_kw - 1) * jump_w
            jump_w = jump_w * stride_w

    return rf_h, rf_w


# Ops whose kernel extent the emitter infers from the weight tensor when the
# optional ONNX ``kernel_shape`` attribute is absent (emitters/binary/writer.py).
_KERNEL_INFERRABLE_OPS = {"Conv", "Conv1D", "DepthwiseConv", "ConvTranspose"}


def _infer_kernel_shape(
    op: OpNode, weight_shapes: dict[str, tuple[int, ...]] | None
) -> tuple[int, ...]:
    """kernel_shape inferred from the first constant weight operand.

    Mirrors the emitter (emitters/binary/writer.py): a Conv may omit the
    optional ONNX ``kernel_shape``, in which case the spatial extent is the
    last one or two dims of its constant weight tensor ([kH, kW] for a 2D
    weight, [kH] for a 1D-as-height weight). Returns () when nothing can be
    inferred, so callers keep their prior fallback.
    """
    if weight_shapes is None or op.op_type not in _KERNEL_INFERRABLE_OPS:
        return ()
    for input_name in op.inputs:
        shape = weight_shapes.get(input_name)
        if shape is None:
            continue
        if len(shape) >= 4:
            return (int(shape[-2]), int(shape[-1]))
        if len(shape) == 3:
            return (int(shape[-1]),)
        return ()
    return ()


def _get_kernel_h(
    op: OpNode, weight_shapes: dict[str, tuple[int, ...]] | None = None
) -> int:
    """Get the height dimension of the kernel (first element of kernel_shape).

    Falls back to the constant weight shape when ``kernel_shape`` is absent,
    matching the emitter, so a recomputing chain that omits the attribute is
    not silently treated as a 1x1 kernel.
    """
    ks = op.attrs.get("kernel_shape")
    if ks:
        return int(ks[0])
    inferred = _infer_kernel_shape(op, weight_shapes)
    if len(inferred) >= 1:
        return int(inferred[0])
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


def _get_kernel_w(
    op: OpNode, weight_shapes: dict[str, tuple[int, ...]] | None = None
) -> int:
    """Get the width dimension of the kernel (second element of kernel_shape).

    Falls back to the constant weight shape when ``kernel_shape`` is absent,
    matching the emitter. A present-but-1D ``kernel_shape`` keeps width 1.
    """
    ks = op.attrs.get("kernel_shape")
    if ks:
        return int(ks[1]) if len(ks) >= 2 else 1
    inferred = _infer_kernel_shape(op, weight_shapes)
    if len(inferred) >= 2:
        return int(inferred[1])
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


def _ag_weight_shapes(ag: AnalyzedGraph) -> dict[str, tuple[int, ...]]:
    """Constant weight tensor shapes keyed by name, for kernel_shape inference."""
    return {name: tuple(arr.shape) for name, arr in ag.weight_data.items()}


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


def _cotileable_skip_operands(
    ag: AnalyzedGraph, stage: Stage, op: OpNode
) -> bool:
    """Every stage-external operand of a pre-spatial Concat/Add/Mul must be a
    rank-4 tensor at the same H/W as the op output, so the executor's shared
    input-halo rectangle load co-tiles it correctly. Intra-stage operands
    (produced by an earlier op in the stage) are fine - they are not loaded.

    A rank-4 CONSTANT operand (an initializer, e.g. a Concat against a baked
    tensor) is never a stage input, so it escapes the external same-H/W check;
    but the 2D executor would still have to co-tile it against the spatial
    op's input-halo rectangle and cannot tile-offset a full-size constant.
    Fail closed on any such operand rather than silently emit a wrong result.
    """
    out = ag.tensors.get(op.outputs[0])
    if out is None or len(out.shape) != 4:
        return False
    out_hw = tuple(out.shape[2:4])
    external = set(stage.input_tensors)
    for name in op.inputs:
        info = ag.tensors.get(name)
        if info is not None and info.is_constant and len(info.shape) == 4:
            return False
        if name in external:
            if info is None or len(info.shape) != 4 or tuple(info.shape[2:4]) != out_hw:
                return False
    return True


def _has_post_spatial_binary(stage_ops: list[OpNode]) -> bool:
    """True if a Concat/Add/Mul is consumed at or after the stage's spatial op.

    Both tiled executors (the 1D-height exec_stage_tiled and the 2D
    exec_stage_tiled_2d) load every stage input at the spatial op's INPUT-halo
    rectangle. A post-spatial binary/Concat operand lives at the spatial op's
    OUTPUT resolution, so a strided spatial op would make the executor read that
    operand at stride*out_start rows -- the wrong rows, and past the operand's
    height -- on interior tiles. Both tile paths must fail closed on this shape
    and run the stage untiled. Shared by _stage_2d_eligible and _stage_tile_axis
    so the height and HW paths reject in lockstep.
    """
    spatial_idx = next(
        (i for i, o in enumerate(stage_ops)
         if classify_op(o.op_type) in (TileCategory.CONV, TileCategory.POOL)),
        None,
    )
    if spatial_idx is None:
        return False
    return any(
        (op.op_type in _BINARY_OPS or op.op_type == "Concat") and i >= spatial_idx
        for i, op in enumerate(stage_ops)
    )


def _stage_2d_eligible(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode]
) -> bool:
    """A stage may attempt HW tiling only if it is a standalone rank-4 stage
    with at most one spatial op, and all of whose ops implement the HW tile
    contract. Multi-spatial-op stages are excluded: the runtime executor's
    2D contract is audited only for a single composed spatial op per stage.

    Add/Mul/Concat are admitted only when they are consumed strictly BEFORE
    the spatial op (their operands then sit at the spatial op's input
    resolution, which is exactly the halo rectangle exec_stage_tiled_2d
    loads every stage input at) and every stage-external operand is a
    same-H/W co-tileable skip (see _cotileable_skip_operands). A stage like
    [Add(input, skip), Conv] is admitted this way. Post-spatial occurrences
    (e.g. [Conv, Add(conv_out, skip)]) and different-resolution operands
    keep the fail-closed rejection: exec_stage_tiled_2d loads every stage
    input using the spatial op's own input-halo rectangle, and a post-spatial
    or different-resolution operand is not guaranteed to be co-tiled with
    that rectangle. Admitting such a stage as 2D would load the operand with
    the wrong region and size and silently produce a wrong result.
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
    # Concat/Add/Mul are admitted only when consumed strictly BEFORE the spatial
    # op (so their operands sit at the spatial op's input resolution = the halo
    # rectangle the executor loads every stage input at) and every stage-external
    # operand is a same-resolution skip. Post-spatial operands fail closed via the
    # shared _has_post_spatial_binary check; different-resolution or constant
    # pre-spatial operands fail closed via _cotileable_skip_operands.
    if _has_post_spatial_binary(stage_ops):
        return False
    for op in stage_ops:
        if op.op_type in _BINARY_OPS or op.op_type == "Concat":
            if not _cotileable_skip_operands(ag, stage, op):
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
    so they are excluded here for the same reason. A 2D-tiled ConvTranspose stage
    is therefore the ConvTranspose plus optional unary pointwise, nothing else.
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


def _serialized_shape(info) -> tuple[int, ...]:
    """The extents in the order the runtime stores them.

    The IR keeps ONNX NCHW/NCL. A SPATIAL tensor serializes with its channel
    axis last; a LINEAR one is already in storage order. Tile geometry for a
    conversion stage has to be read in that storage order, because the
    conversion is a permutation of exactly those axes.
    """
    shape = tuple(int(dim) for dim in info.shape)
    if info.layout is Layout.LINEAR:
        return shape
    if len(shape) == 4:
        return (shape[0], shape[2], shape[3], shape[1])
    if len(shape) == 3:
        return (shape[0], shape[2], shape[1])
    return shape


def _conversion_extents(info) -> tuple[int, int, int] | None:
    """The batch and the two extents a conversion of this tensor transposes.

    A conversion moves the channel axis past the spatial ones and leaves those
    in their relative order, so once the axes that travel together are read as
    one it is a plain matrix transpose. In storage order a SPATIAL tensor has
    its channels last, a LINEAR one has them at position 1.
    """
    stored = _serialized_shape(info)
    if len(stored) not in (3, 4):
        return None
    batch = stored[0]
    if info.layout is Layout.SPATIAL:
        rows = math.prod(stored[1:-1])
        cols = stored[-1]
    else:
        rows = stored[1]
        cols = math.prod(stored[2:])
    if batch <= 0 or rows <= 0 or cols <= 0:
        return None
    return batch, rows, cols


def _stage_is_global_reduction(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode]
) -> bool:
    """A stage that reduces a whole rank-4 plane and nothing else.

    GlobalAveragePool stays UNTILEABLE in _OP_CATEGORY because it collapses
    the height the stripe contract propagates. The runtime instead walks such
    a stage along its input, a band of rows at a time, carrying one partial
    per channel between bands. That only works when the reduction is the
    whole stage: with another op present there would be a height to carry.
    Mirrors stage_is_input_driven_reduction in the executor.
    """
    if len(stage_ops) != 1 or stage_ops[0].op_type != "GlobalAveragePool":
        return False
    op = stage_ops[0]
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    if stage.input_tensors != op.inputs or stage.output_tensors != op.outputs:
        return False

    source = ag.tensors.get(op.inputs[0])
    result = ag.tensors.get(op.outputs[0])
    if source is None or result is None:
        return False
    if len(source.shape) != 4 or len(result.shape) != 4:
        return False
    # NCHW: the reduction collapses H and W, keeping N and C.
    return (
        int(source.shape[2]) > 1
        and int(source.shape[3]) > 0
        and int(result.shape[0]) == int(source.shape[0])
        and int(result.shape[1]) == int(source.shape[1])
        and int(result.shape[2]) == 1
        and int(result.shape[3]) == 1
    )


def _solve_global_reduction(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode], budget: int
) -> TilePlan:
    """Size the band of input rows a global reduction reads at a time.

    Mirrors the runtime's arena accounting: one accumulator of four bytes per
    (batch, channel) held for the whole stage, plus one band of input rows and
    the collapsed output, all resident together.
    """
    op = stage_ops[0]
    source = ag.tensors[op.inputs[0]]
    result = ag.tensors[op.outputs[0]]

    batch = int(source.shape[0])
    channels = int(source.shape[1])
    full_h = int(source.shape[2])
    full_w = int(source.shape[3])

    align = max(ag.tensor_alignment, _CONSERVATIVE_TENSOR_ALIGN)
    # The accumulator is int32 for the int8 kernels and float for the float32
    # ones, four bytes either way.
    acc_bytes = _align_up(batch * channels * 4, align)
    out_bytes = _align_up(
        batch * channels * result.elem_size, align
    )
    row_elems = batch * full_w * channels

    def working_set(rows: int) -> int:
        return (
            acc_bytes
            + _align_up(rows * row_elems * source.elem_size, align)
            + out_bytes
        )

    if working_set(1) > budget:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id} minimum reduction band still "
                f"exceeds budget ({budget:,} bytes)"
            ],
        )

    low, high, rows = 1, full_h, 1
    while low <= high:
        candidate = low + (high - low) // 2
        if working_set(candidate) <= budget:
            rows = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    return TilePlan(
        tileable=True,
        axis=TILE_AXIS_HEIGHT_OR_LENGTH,
        tile_height=rows,
        num_tiles=math.ceil(full_h / rows),
        halo=0,
        receptive_field=1,
        original_height=full_h,
        tiled_peak_bytes=working_set(rows),
        overhead_bytes=0,
        warnings=[],
    )

def _stage_is_layout_conversion(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode]
) -> bool:
    """A stage that is one conversion between the two storage orders.

    Transpose stays UNTILEABLE in _OP_CATEGORY because it swaps the two axes
    the stripe contract propagates. The runtime instead walks the conversion's
    output rows, gathering the input columns each band transposes, which only
    works when the conversion is the whole stage.
    """
    if len(stage_ops) != 1 or stage_ops[0].op_type != "Transpose":
        return False
    op = stage_ops[0]
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    if stage.input_tensors != op.inputs or stage.output_tensors != op.outputs:
        return False

    source = ag.tensors.get(op.inputs[0])
    result = ag.tensors.get(op.outputs[0])
    if source is None or result is None:
        return False
    if source.layout is result.layout:
        return False
    if len(source.shape) != len(result.shape):
        return False

    perm = op.attrs.get("perm")
    if perm is None or list(perm) != list(range(len(source.shape))):
        return False

    from_side = _conversion_extents(source)
    to_side = _conversion_extents(result)
    if from_side is None or to_side is None:
        return False
    # The conversion transposes the pair, so the other side reads reversed.
    return (
        from_side[0] == to_side[0]
        and from_side[1] == to_side[2]
        and from_side[2] == to_side[1]
    )


def _solve_layout_conversion(
    ag: AnalyzedGraph, stage: Stage, stage_ops: list[OpNode], budget: int
) -> TilePlan:
    """Size the band a conversion transposes at a time.

    Mirrors the runtime: the band runs along the longer of the two permuted
    axes, and the two slices hold the same element count but are separate
    allocations, so each is aligned on its own.
    """
    source = ag.tensors[stage_ops[0].inputs[0]]
    extents = _conversion_extents(source)
    if extents is None:  # guarded by _stage_is_layout_conversion; defensive
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id}: cannot determine conversion extents"
            ],
        )
    batch, rows, cols = extents
    banded = max(rows, cols)
    other = min(rows, cols)

    align = max(ag.tensor_alignment, _CONSERVATIVE_TENSOR_ALIGN)

    def working_set(band: int) -> int:
        return 2 * _align_up(batch * other * band * source.elem_size, align)

    if working_set(1) > budget:
        return TilePlan(
            tileable=False,
            warnings=[
                f"Stage {stage.stage_id} minimum conversion band still "
                f"exceeds budget ({budget:,} bytes)"
            ],
        )

    low, high, band = 1, banded, 1
    while low <= high:
        candidate = low + (high - low) // 2
        if working_set(candidate) <= budget:
            band = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    return TilePlan(
        tileable=True,
        axis=TILE_AXIS_HEIGHT_OR_LENGTH,
        tile_height=band,
        num_tiles=math.ceil(banded / band),
        halo=0,
        receptive_field=1,
        original_height=banded,
        tiled_peak_bytes=working_set(band),
        overhead_bytes=0,
        warnings=[],
    )


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
    weight_shapes = _ag_weight_shapes(ag)
    eff_kh = _get_dilation_h(ct) * (_get_kernel_h(ct, weight_shapes) - 1) + 1
    eff_kw = _get_dilation_w(ct) * (_get_kernel_w(ct, weight_shapes) - 1) + 1
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



def _is_layout_conversion(ag: AnalyzedGraph, op: OpNode) -> bool:
    """A Transpose that only restates a tensor's axis order.

    The normalizer inserts one where a producer and a consumer disagree about
    layout, with an identity permutation: the tensors differ in layout, not in
    the order their axes are named.
    """
    if op.op_type != "Transpose" or len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    perm = op.attrs.get("perm")
    if perm is None or list(perm) != list(range(len(perm))):
        return False
    source = ag.tensors.get(op.inputs[0])
    result = ag.tensors.get(op.outputs[0])
    return (
        source is not None
        and result is not None
        and source.layout is not result.layout
    )


def conversion_cut_points(ag: AnalyzedGraph) -> frozenset[int]:
    """Op indices that must start a stage for an oversized stage to tile.

    A layout conversion swaps the two axes on either side of it, so a stage
    holding one wants different tile axes for its input and its interior and
    cannot tile on either. Isolating the conversion lets each side tile on its
    own axis. Only stages that are both oversized and untileable are split:
    a conversion that fits keeps its intermediates in the fast arena, which
    splitting would force out to slow.
    """
    if not ag.stages or ag.mem_budget <= 0:
        return frozenset()

    cuts: set[int] = set()
    for stage in ag.stages:
        if stage.peak_bytes <= ag.mem_budget:
            continue
        if stage.tile_plan is not None and stage.tile_plan.tileable:
            continue
        for position, op_index in enumerate(stage.op_indices):
            if not _is_layout_conversion(ag, ag.ops[op_index]):
                continue
            # Isolate it: cut before it, and after it when it is not last.
            cuts.add(op_index)
            if position + 1 < len(stage.op_indices):
                cuts.add(stage.op_indices[position + 1])
    return frozenset(cuts)


def partition_spatial(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Analyze each stage and attach a TilePlan where needed.

    Only stages whose peak_bytes exceed mem_budget are analyzed.
    Stages that fit within budget get no tile_plan (None).

    A stage that stays untileable because it fuses a layout conversion is
    given a second chance: the graph is re-partitioned with the conversion
    on its own stage, then re-analyzed. Stages that tiled the first time are
    unaffected, so a graph without such a stage takes the single pass.
    """
    _assign_tile_plans(ag)

    cuts = conversion_cut_points(ag)
    if cuts:
        partition_temporal(ag, ag.mem_budget, forced_cuts=cuts)
        _assign_tile_plans(ag)
    return ag


def _assign_tile_plans(ag: AnalyzedGraph) -> AnalyzedGraph:
    """One pass of tile analysis over the current stage list."""
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

        # A global reduction has no output axis to tile, so the runtime walks
        # its input instead. Like ConvTranspose it stays UNTILEABLE in
        # _OP_CATEGORY and reaches its execution path only through this branch.
        if _stage_is_global_reduction(ag, stage, stage_ops):
            stage.tile_plan = _solve_global_reduction(
                ag, stage, stage_ops, budget)
            continue

        # A conversion permutes the two axes the stripe contract propagates,
        # so it reaches its own execution path the same way.
        if _stage_is_layout_conversion(ag, stage, stage_ops):
            stage.tile_plan = _solve_layout_conversion(
                ag, stage, stage_ops, budget)
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
        rf_h, rf_w = compute_receptive_field(stage_ops, _ag_weight_shapes(ag))
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
        # A post-spatial binary/Concat with a stage-external operand mis-tiles on
        # the height path exactly as on the 2D path (the skip is loaded at the
        # spatial op's input rows, not its output rows). Fail closed so the stage
        # runs untiled via exec_stage_normal, matching _stage_2d_eligible.
        if _has_post_spatial_binary(stage_ops):
            return TILE_AXIS_NONE
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
