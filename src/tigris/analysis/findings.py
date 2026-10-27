"""Auto-generate actionable findings from an AnalyzedGraph.

Pure data, no formatting. Used by the compile and analyze commands."""

from dataclasses import dataclass, field

import numpy as np

from tigris.analysis.memory import align_up
from tigris.analysis.validation import PlanGate, plan_gate
from tigris.graph.ir import AnalyzedGraph
from tigris.utils import source_shape


@dataclass
class BlockingStage:
    """An execution unit that does not fit the budget, with what it needs."""
    stage_id: int
    required_bytes: int
    reason: str


@dataclass
class Findings:
    # verdict
    verdict: str = ""  # "ok", "partitioned", "tiled", "needs_work"

    # basics
    scheduled_peak_bytes: int = 0  # actual max after partitioning + tiling
    total_stages: int = 0
    stages_needing_tiling: int = 0

    # largest tensor
    largest_tensor_shape: str = ""
    largest_tensor_bytes: int = 0

    # tiling
    stages_tileable: int = 0
    stages_untileable: int = 0
    untileable_op_types: list[str] = field(default_factory=list)
    # Execution units over budget in the partition chosen for this budget, and
    # the smallest fast arena that partition needs. Another budget can
    # partition differently, so this is not a floor across budgets.
    blocking_stages: list[BlockingStage] = field(default_factory=list)
    min_fast_for_partition: int = 0
    feasibility_errors: list[str] = field(default_factory=list)
    unsupported_operators: list[str] = field(default_factory=list)
    output_errors: list[str] = field(default_factory=list)
    dtype_errors: list[str] = field(default_factory=list)

    # quantization
    is_float32: bool = False
    is_quantized: bool = False

    # deployment / plan size
    total_weight_bytes: int = 0
    plan_overhead_bytes: int = 0
    plan_size_bytes: int = 0
    # Runtime build limits the plan needs above their defaults, as inspect
    # reports them: {"name", "needed", "default"}.
    build_limits: list = field(default_factory=list)
    int8_plan_size_bytes: int = 0
    lz4_plan_size_bytes: int = 0
    plan_fits_flash: bool = False

    # slow memory (PSRAM) residency
    slow_peak_bytes: int = 0  # peak slow-pool residency over all execution groups
    slow_overflow_stages: list[int] = field(default_factory=list)
    slow_fits: bool = True

    # The gate compile applies, and the plan as compile writes it without
    # compression or XIP; empty when there is no plan to measure.
    gate: PlanGate | None = None
    plan: bytes = field(default=b"", repr=False)


def _weight_sizes(ag: AnalyzedGraph) -> tuple[list[int], list[int], int]:
    """Per-weight stored sizes, as written and with float weights as int8, and
    the LZ4 size of the stored weights."""
    from tigris.emitters.binary.writer import serialized_weight_bytes

    raws = [serialized_weight_bytes(arr) for arr in ag.weight_data.values()]
    int8_sizes = [arr.size if np.issubdtype(arr.dtype, np.floating) else len(raw)
                  for arr, raw in zip(ag.weight_data.values(), raws)]

    # LZ4 on the actual weight data (meaningful for the current dtype only)
    import lz4.block

    lz4_total = 0
    blob = b"".join(raws)
    if blob:
        lz4_total = len(lz4.block.compress(blob, store_size=False))

    return [len(raw) for raw in raws], int8_sizes, lz4_total


def _aligned_weight_blob_size(sizes: list[int], alignment: int) -> int:
    """Mirror the per-weight padding used by the uncompressed writer."""
    size = 0
    for weight_size in sizes:
        size = align_up(size, alignment)
        size += weight_size
    return size


def _compressed_weight_block_count(ag: AnalyzedGraph) -> int | None:
    """The stage weight blocks a compressed plan holds, or None when the
    writer cannot group the weights by stage."""
    from tigris.emitters.binary.writer import _group_weights_by_stage

    try:
        return len(_group_weights_by_stage(ag, {name: idx for idx, name in enumerate(ag.weight_data)}))
    except ValueError:
        return None


def _measure_plan(ag: AnalyzedGraph, names: bool) -> bytes:
    """The serialized plan, from the writer itself; empty when the plan does
    not serialize (an unsupported operator or dtype)."""
    from tigris.emitters.binary.writer import emit_binary_bytes

    try:
        return emit_binary_bytes(ag, names=names)
    except ValueError:
        return b""


def compute_findings(ag: AnalyzedGraph, names: bool = False) -> Findings:
    """Analyze the graph and produce structured findings. The plan is sized as
    compile writes it, with `names` as --keep-names."""
    f = Findings()
    peak = ag.peak_memory_bytes
    budget = ag.mem_budget

    gate = f.gate = plan_gate(ag)
    f.unsupported_operators = [issue.describe() for issue in gate.operators.issues]
    f.output_errors = list(gate.outputs.issues)
    f.dtype_errors = list(gate.dtype.issues)
    if not gate.usable:
        f.verdict = "needs_work"

    if not ag.lifetimes:
        return f

    largest = max(ag.lifetimes.values(), key=lambda lt: lt.size_bytes)
    largest_info = ag.tensors.get(largest.tensor_name)
    f.largest_tensor_bytes = largest.size_bytes
    if largest_info:
        f.largest_tensor_shape = "x".join(str(d) for d in source_shape(ag, largest_info.shape))

    # Analysis and compile share one gate, so analyze cannot report a fit that compile refuses.
    validation = gate.budget.fast
    blocked_units = {issue.stage_id for issue in validation.issues if not issue.subgraph}

    # Tileable: a tile plan the validator accepts. Untileable: an operator that cannot tile.
    untileable_types: set[str] = set()
    for s in ag.stages:
        if s.tile_plan is not None:
            if s.tile_plan.tileable and s.stage_id not in blocked_units:
                f.stages_tileable += 1
            elif s.tile_plan.untileable_ops:
                f.stages_untileable += 1
                untileable_types.update(op_type for _, op_type in s.tile_plan.untileable_ops)
    f.untileable_op_types = sorted(untileable_types)

    f.scheduled_peak_bytes = validation.scheduled_peak_bytes
    f.feasibility_errors = [issue.describe() for issue in validation.issues]
    # The same requirements compile checks, so the report cannot name a
    # smaller or larger need than the error compile would raise.
    f.blocking_stages = sorted(
        (BlockingStage(issue.stage_id, issue.required_bytes, issue.reason)
         for issue in validation.issues),
        key=lambda stage: -stage.required_bytes,
    )
    if f.blocking_stages:
        f.min_fast_for_partition = f.blocking_stages[0].required_bytes

    if budget > 0:
        f.stages_needing_tiling = sum(1 for s in ag.stages if s.warnings)
        f.total_stages = len(ag.stages)

    # Verdict, when the gate has not already refused the graph
    if gate.usable and budget > 0:
        if not validation.feasible:
            f.verdict = "needs_work"
        elif peak / budget <= 1.0:
            f.verdict = "ok"
        elif f.stages_needing_tiling == 0:
            f.verdict = "partitioned"
        elif f.stages_untileable == 0:
            f.verdict = "tiled"
        else:
            f.verdict = "needs_work"

    f.is_float32 = any(
        ag.tensors.get(lt.tensor_name) and ag.tensors[lt.tensor_name].dtype == 1
        for lt in ag.lifetimes.values()
    )
    f.is_quantized = ag.is_quantized

    # Deployment size: the plan as the writer serializes it, its subgraphs'
    # weights included; the int8 and LZ4 variants swap the weight blob.
    from tigris.emitters.binary.defs import (
        PLAN_SECTION_ALIGNMENT,
        WEIGHT_BLOCK_SECTION_HEADER_STRUCT,
        WEIGHT_BLOCK_SIZE,
    )
    from tigris.graph.subgraphs import flatten_subgraphs

    whole = flatten_subgraphs(ag)[0] if ag.subgraphs else ag
    sizes, int8_sizes, lz4_weight_bytes = _weight_sizes(whole)
    f.total_weight_bytes = sum(sizes)
    current_blob = _aligned_weight_blob_size(sizes, PLAN_SECTION_ALIGNMENT)
    plan = f.plan = _measure_plan(ag, names)
    f.plan_size_bytes = len(plan)
    if plan:
        from tigris.inspection import _plan, _plan_requirements

        f.build_limits = [limit for limit in _plan_requirements(_plan(plan))["build_limits"]
                          if limit["needed"] > limit["default"]]
        f.plan_overhead_bytes = f.plan_size_bytes - f.total_weight_bytes
        f.int8_plan_size_bytes = f.plan_size_bytes - current_blob + _aligned_weight_blob_size(
            int8_sizes, PLAN_SECTION_ALIGNMENT)
        blocks = _compressed_weight_block_count(whole) if lz4_weight_bytes > 0 else None
        if blocks is not None:
            f.lz4_plan_size_bytes = (f.plan_size_bytes - current_blob
                                     + WEIGHT_BLOCK_SECTION_HEADER_STRUCT.size
                                     + blocks * WEIGHT_BLOCK_SIZE
                                     + lz4_weight_bytes)
    if ag.budget.flash > 0 and f.plan_size_bytes:
        f.plan_fits_flash = f.plan_size_bytes <= ag.budget.flash

    # Slow memory check (PSRAM)
    # Slow residency is measured over every execution group; a group whose
    # resident tensors exceed the slow budget overflows.
    usage = gate.budget.slow
    f.slow_peak_bytes = usage.slow_peak_bytes
    f.slow_overflow_stages = list(usage.overflow_stage_ids)
    f.slow_fits = usage.fits
    if not f.slow_fits and f.verdict == "tiled":
        f.verdict = "needs_work"

    return f
