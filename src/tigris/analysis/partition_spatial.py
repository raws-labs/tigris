"""Spatial partitioning - receptive field computation and tile solving.

For stages whose peak memory exceeds the SRAM budget, determines whether ops
are spatially tileable, computes the receptive field (halo overlap), and solves
for tile dimensions that fit within budget.

Also detects *streamable chains*: consecutive tiled stages whose intermediate
tensors can stay in fast memory as tiles, avoiding full-size slow allocation.
"""

import math
from enum import Enum

from tigris.graph.ir import AnalyzedGraph, OpNode, Stage, TilePlan


# Op classification


class TileCategory(Enum):
    CONV = "conv"
    POOL = "pool"
    POINTWISE = "pointwise"
    UNTILEABLE = "untileable"


# Op types that are spatially tileable (operate on spatial dims independently)
_OP_CATEGORY: dict[str, TileCategory] = {
    # Convolutions
    "Conv": TileCategory.CONV,
    "Conv1D": TileCategory.CONV,
    "ConvTranspose": TileCategory.CONV,
    "DepthwiseConv": TileCategory.CONV,
    # Pooling
    "MaxPool": TileCategory.POOL,
    "AveragePool": TileCategory.POOL,
    "GlobalAveragePool": TileCategory.POOL,
    "GlobalMaxPool": TileCategory.POOL,
    # Pointwise / element-wise (pass through spatial dims unchanged)
    "Relu": TileCategory.POINTWISE,
    "Relu6": TileCategory.POINTWISE,
    "LeakyRelu": TileCategory.POINTWISE,
    "Sigmoid": TileCategory.POINTWISE,
    "Tanh": TileCategory.POINTWISE,
    "HardSigmoid": TileCategory.POINTWISE,
    "HardSwish": TileCategory.POINTWISE,
    "Clip": TileCategory.POINTWISE,
    "Add": TileCategory.POINTWISE,
    "Sub": TileCategory.POINTWISE,
    "Mul": TileCategory.POINTWISE,
    "Div": TileCategory.POINTWISE,
    "BatchNormalization": TileCategory.POINTWISE,
    "InstanceNormalization": TileCategory.POINTWISE,
    "Concat": TileCategory.POINTWISE,
    "Resize": TileCategory.POINTWISE,
    "Pad": TileCategory.POINTWISE,
    # Untileable - these collapse or reshape spatial dims
    "Flatten": TileCategory.UNTILEABLE,
    "Reshape": TileCategory.UNTILEABLE,
    "Gemm": TileCategory.UNTILEABLE,
    "MatMul": TileCategory.UNTILEABLE,
    "Softmax": TileCategory.UNTILEABLE,
    "ReduceMean": TileCategory.UNTILEABLE,
    "Squeeze": TileCategory.UNTILEABLE,
    "Unsqueeze": TileCategory.UNTILEABLE,
    "Transpose": TileCategory.UNTILEABLE,
}


def classify_op(op_type: str) -> TileCategory:
    """Classify an op type into a tile category. Unknown ops are UNTILEABLE."""
    return _OP_CATEGORY.get(op_type, TileCategory.UNTILEABLE)


# Receptive field computation


def compute_receptive_field(ops: list[OpNode]) -> tuple[int, int]:
    """Compute the receptive field and total stride for a sequence of ops.

    Walks the ops in reverse, accumulating RF and jump (cumulative stride).
    Returns (receptive_field, total_jump).

    For pointwise ops, RF and jump are unchanged.
    For conv/pool ops, RF grows based on effective kernel size.
    """
    rf = 1
    jump = 1

    for op in reversed(ops):
        cat = classify_op(op.op_type)
        if cat in (TileCategory.CONV, TileCategory.POOL):
            kernel = _get_kernel_h(op)
            stride = _get_stride_h(op)
            dilation = _get_dilation_h(op)

            effective_k = dilation * (kernel - 1) + 1
            rf = rf + (effective_k - 1) * jump
            jump = jump * stride

    return rf, jump


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


# Tile solver


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

        # Check if all ops are tileable
        untileable: list[str] = []
        for op in stage_ops:
            cat = classify_op(op.op_type)
            if cat == TileCategory.UNTILEABLE:
                untileable.append(f"{op.name} ({op.op_type})")

        if untileable:
            stage.tile_plan = TilePlan(
                tileable=False,
                untileable_ops=untileable,
                warnings=[f"Stage {stage.stage_id} contains untileable ops"],
            )
            continue

        # Compute receptive field
        rf, _jump = compute_receptive_field(stage_ops)
        halo = rf - 1

        # Find input height (NCHW layout, index 2)
        input_h = _find_input_height(ag, stage)
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

        # Overhead: extra halo reads per tile boundary
        # Each internal tile boundary reads halo rows extra from the input
        halo_tensor_bytes = _estimate_halo_bytes(ag, stage, halo, input_h)
        overhead = halo_tensor_bytes * max(num_tiles - 1, 0)

        warnings: list[str] = []
        if tiled_peak > budget:
            warnings.append(
                f"Stage {stage.stage_id} tiled peak ({tiled_peak:,} bytes) "
                f"still exceeds budget ({budget:,} bytes)"
            )

        stage.tile_plan = TilePlan(
            tileable=True,
            tile_height=tile_h,
            num_tiles=num_tiles,
            halo=halo,
            receptive_field=rf,
            original_height=input_h,
            tiled_peak_bytes=tiled_peak,
            overhead_bytes=overhead,
            warnings=warnings,
        )

    return ag


def _find_input_height(ag: AnalyzedGraph, stage) -> int:
    """Find the spatial height of the stage's primary input tensor (NCHW dim 2)."""
    # Check stage input tensors first, then look at first op's inputs
    candidates = stage.input_tensors.copy()
    if not candidates:
        first_op = ag.ops[stage.op_indices[0]]
        candidates = [n for n in first_op.inputs if n in ag.tensors]

    for name in candidates:
        info = ag.tensors.get(name)
        if info and len(info.shape) >= 4:
            return int(info.shape[2])  # NCHW -> H is dim 2

    return 0


def _estimate_halo_bytes(ag: AnalyzedGraph, stage, halo: int, input_h: int) -> int:
    """Estimate bytes for one halo region of the stage's input tensor."""
    candidates = stage.input_tensors.copy()
    if not candidates:
        first_op = ag.ops[stage.op_indices[0]]
        candidates = [n for n in first_op.inputs if n in ag.tensors]

    for name in candidates:
        info = ag.tensors.get(name)
        if info and len(info.shape) >= 4:
            # bytes per row = total_bytes / H
            if input_h > 0:
                return int(info.size_bytes * halo / input_h)

    return 0


# Chain detection and solving


def _is_stage_tileable(ag: AnalyzedGraph, stage: Stage) -> bool:
    """Check if a stage may be part of a streamable CHAIN (used only by
    detect_chains): all ops spatially tileable, no POOL op, and 4D I/O.

    POOL is excluded because the runtime chain executor (exec_chain_tiled)
    composes/back-propagates receptive fields for Conv/DepthwiseConv only; a POOL
    op in a chained stage would be treated as height-preserving (pointwise),
    giving wrong tile row counts and OOB reads. Pool stages still tile standalone
    via exec_stage_tiled. Remove the POOL exclusion once the chain executor
    handles pool spatial ops.
    """
    for op_i in stage.op_indices:
        cat = classify_op(ag.ops[op_i].op_type)
        if cat == TileCategory.UNTILEABLE:
            return False
        if cat == TileCategory.POOL:
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

    The runtime executor composes receptive fields from all Conv/DW ops
    in a stage when validating chain tile heights, so the compiler must
    match by composing here too.

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


def _align_up(x: int, align: int = 8) -> int:
    """Round up to alignment boundary (matches TIGRIS_TENSOR_ALIGN on Xtensa)."""
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

    Each allocation is rounded up to TIGRIS_TENSOR_ALIGN (8 bytes) to match
    the runtime bump allocator's alignment overhead.

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
            total += _align_up(N * in_h * W * C * info.elem_size)

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
                    total += _align_up(N * cur_h * W * C * info.elem_size)

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

    return ag
