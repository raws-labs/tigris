"""Auto-generate actionable findings from an AnalyzedGraph.

Pure data - no HTML, no formatting. Used by both CLI and report."""

import copy
from dataclasses import dataclass, field

import numpy as np

from tigris.graph.ir import AnalyzedGraph
from tigris.analysis.partition_temporal import partition_temporal
from tigris.utils import fmt_bytes, source_shape


@dataclass
class BudgetRow:
    budget: int
    budget_str: str
    stages: int
    need_tiling: int
    ok: bool


@dataclass
class BlockingStage:
    """An execution unit that does not fit the budget, with what it needs."""
    stage_id: int
    required_bytes: int
    reason: str


@dataclass
class Findings:
    # verdict
    verdict: str = ""  # "ok", "partitioned", "needs_tiling"
    verdict_text: str = ""

    # basics
    peak_bytes: int = 0
    scheduled_peak_bytes: int = 0  # actual max after partitioning + tiling
    budget: int = 0
    ratio: float = 0.0
    total_stages: int = 0
    stages_needing_tiling: int = 0

    # largest tensor
    largest_tensor_name: str = ""
    largest_tensor_shape: str = ""
    largest_tensor_bytes: int = 0

    # minimum budget to avoid tiling
    min_budget_no_tiling: int = 0

    # spill/reload
    total_spill_bytes: int = 0
    total_reload_bytes: int = 0
    stage_transitions: int = 0

    # tiling
    stages_tileable: int = 0
    stages_untileable: int = 0
    total_tiles: int = 0
    max_halo: int = 0
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
    int8_estimate_bytes: int = 0
    int8_fits_budget: bool = False

    # deployment / plan size
    total_weight_bytes: int = 0
    int8_weight_bytes: int = 0
    plan_overhead_bytes: int = 0
    plan_size_bytes: int = 0
    # Runtime build limits the plan needs above their defaults, as inspect
    # reports them: {"name", "needed", "default"}.
    build_limits: list = field(default_factory=list)
    int8_plan_size_bytes: int = 0
    lz4_weight_bytes: int = 0
    lz4_plan_size_bytes: int = 0
    flash_budget: int = 0
    plan_fits_flash: bool = False
    int8_fits_flash: bool = False

    # budget sweep
    budget_sweep: list[BudgetRow] = field(default_factory=list)

    # slow memory (PSRAM) constraints for tiled execution
    slow_peak_bytes: int = 0  # max(input + output) across tiled stages
    slow_overflow_stages: list[int] = field(default_factory=list)
    slow_fits: bool = True


def _weight_sizes(ag: AnalyzedGraph) -> tuple[list[int], list[int], int]:
    """Per-weight stored sizes, as written and with float weights as int8, and
    the LZ4 size of the stored weights."""
    from tigris.emitters.binary.writer import serialized_weight_bytes

    raws = [serialized_weight_bytes(arr) for arr in ag.weight_data.values()]
    int8_sizes = [arr.size if np.issubdtype(arr.dtype, np.floating) else len(raw)
                  for arr, raw in zip(ag.weight_data.values(), raws)]

    # LZ4 on the actual weight data (meaningful for the current dtype only)
    lz4_total = 0
    try:
        import lz4.block
        blob = b"".join(raws)
        if blob:
            lz4_total = len(lz4.block.compress(blob, store_size=False))
    except ImportError:
        pass

    return [len(raw) for raw in raws], int8_sizes, lz4_total


def _aligned_weight_blob_size(sizes: list[int], alignment: int) -> int:
    """Mirror the per-weight padding used by the uncompressed writer."""
    size = 0
    for weight_size in sizes:
        size = (size + alignment - 1) // alignment * alignment
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


def compute_findings(ag: AnalyzedGraph, flash_budget: int = 0, names: bool = False) -> Findings:
    """Analyze the graph and produce structured findings. The plan is sized as
    compile writes it, with `names` as --keep-names."""
    f = Findings()
    peak = ag.peak_memory_bytes
    budget = ag.mem_budget
    f.peak_bytes = peak
    f.budget = budget
    f.flash_budget = flash_budget

    from tigris.analysis.validation import (
        validate_execution_dtype,
        validate_model_outputs,
        validate_operator_support,
    )

    operator_validation = validate_operator_support(ag)
    f.unsupported_operators = [
        issue.describe() for issue in operator_validation.issues
    ]
    f.output_errors = list(validate_model_outputs(ag).issues)
    dtype_validation = validate_execution_dtype(ag)
    f.dtype_errors = list(dtype_validation.issues)

    if not ag.lifetimes:
        if f.unsupported_operators or f.output_errors or f.dtype_errors:
            f.verdict = "needs_work"
            if f.unsupported_operators:
                f.verdict_text = (
                    "Unsupported operators: " + ", ".join(f.unsupported_operators)
                )
            else:
                f.verdict_text = (f.output_errors + f.dtype_errors)[0]
        return f

    # Largest tensor
    sorted_lt = sorted(ag.lifetimes.values(), key=lambda lt: -lt.size_bytes)
    largest = sorted_lt[0]
    largest_info = ag.tensors.get(largest.tensor_name)
    f.largest_tensor_name = largest.tensor_name
    f.largest_tensor_bytes = largest.size_bytes
    if largest_info:
        f.largest_tensor_shape = "x".join(str(d) for d in source_shape(ag, largest_info.shape))

    # Min budget to avoid tiling
    f.min_budget_no_tiling = max(
        (s.live_bytes for s in ag.timeline), default=peak
    )

    # Tiling results (computed before verdict)
    untileable_ops: set[str] = set()
    for s in ag.stages:
        if s.tile_plan is not None:
            if s.tile_plan.tileable:
                f.stages_tileable += 1
                f.total_tiles += s.tile_plan.num_tiles
                if s.tile_plan.halo > f.max_halo:
                    f.max_halo = s.tile_plan.halo
            elif s.tile_plan.untileable_ops:
                # Only an operator that cannot tile makes a stage untileable;
                # a stage whose smallest tile is too large is still tileable.
                f.stages_untileable += 1
                for op_desc in s.tile_plan.untileable_ops:
                    # op_desc is "name (OpType)" - extract the type
                    if "(" in op_desc:
                        op_type = op_desc.split("(")[-1].rstrip(")")
                    else:
                        op_type = op_desc
                    untileable_ops.add(op_type)
            else:
                f.stages_tileable += 1
    f.untileable_op_types = sorted(untileable_ops)

    # Analysis and deployment share one execution-unit validator, so a CLI
    # PASS cannot disagree with `tigris compile` about memory feasibility.
    from tigris.analysis.validation import validate_memory_plan

    validation = validate_memory_plan(ag)
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

    # Verdict
    if budget > 0:
        f.ratio = peak / budget
        f.stages_needing_tiling = sum(1 for s in ag.stages if s.warnings)
        f.total_stages = len(ag.stages)

        if not validation.feasible:
            f.verdict = "needs_work"
            f.verdict_text = f.feasibility_errors[0]
        elif f.ratio <= 1.0:
            f.verdict = "ok"
            f.verdict_text = (
                f"Peak memory ({fmt_bytes(peak)}) fits within budget "
                f"({fmt_bytes(budget)}). Depth-axis partitioning into "
                f"{f.total_stages} stage(s) is sufficient."
            )
        elif f.stages_needing_tiling == 0:
            f.verdict = "partitioned"
            f.verdict_text = (
                f"Peak memory is {f.ratio:.1f}x the budget, but depth-axis "
                f"partitioning into {f.total_stages} stages handles it "
                f"- no tiled streaming needed."
            )
        elif f.stages_untileable == 0:
            f.verdict = "tiled"
            f.verdict_text = (
                f"Peak memory ({fmt_bytes(peak)}) is {f.ratio:.1f}x the "
                f"budget ({fmt_bytes(budget)}). {f.stages_needing_tiling} of "
                f"{f.total_stages} stages exceed budget - tiled streaming "
                f"resolves all ({f.total_tiles} tiles, max halo "
                f"{f.max_halo})."
            )
        else:
            f.verdict = "needs_work"
            f.verdict_text = (
                f"Peak memory ({fmt_bytes(peak)}) is {f.ratio:.1f}x the "
                f"budget ({fmt_bytes(budget)}). {f.stages_needing_tiling} of "
                f"{f.total_stages} stages exceed budget. "
                f"{f.stages_untileable} stage(s) contain untileable ops."
            )

    if f.unsupported_operators:
        f.verdict = "needs_work"
        f.verdict_text = (
            "Unsupported operators: " + ", ".join(f.unsupported_operators)
        )
    elif f.output_errors or f.dtype_errors:
        f.verdict = "needs_work"
        f.verdict_text = (f.output_errors + f.dtype_errors)[0]

    # Spill/reload cost
    if ag.stages and len(ag.stages) > 1:
        f.stage_transitions = len(ag.stages) - 1
        for s in ag.stages:
            for tname in s.output_tensors:
                info = ag.tensors.get(tname)
                if info:
                    f.total_spill_bytes += info.size_bytes
            for tname in s.input_tensors:
                info = ag.tensors.get(tname)
                if info:
                    f.total_reload_bytes += info.size_bytes

    # Quantization estimate (activations)
    float_tensors = [
        lt for lt in ag.lifetimes.values()
        if ag.tensors.get(lt.tensor_name)
        and ag.tensors[lt.tensor_name].dtype == 1
    ]
    if float_tensors:
        f.is_float32 = True
        f.int8_estimate_bytes = peak // 4
        f.int8_fits_budget = budget > 0 and f.int8_estimate_bytes < budget

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
    sizes, int8_sizes, f.lz4_weight_bytes = _weight_sizes(whole)
    f.total_weight_bytes, f.int8_weight_bytes = sum(sizes), sum(int8_sizes)
    current_blob = _aligned_weight_blob_size(sizes, PLAN_SECTION_ALIGNMENT)
    plan = _measure_plan(ag, names)
    f.plan_size_bytes = len(plan)
    if plan:
        from tigris.inspection import _plan, _plan_requirements

        f.build_limits = [limit for limit in _plan_requirements(_plan(plan))["build_limits"]
                          if limit["needed"] > limit["default"]]
        f.plan_overhead_bytes = f.plan_size_bytes - f.total_weight_bytes
        f.int8_plan_size_bytes = f.plan_size_bytes - current_blob + _aligned_weight_blob_size(
            int8_sizes, PLAN_SECTION_ALIGNMENT)
        blocks = _compressed_weight_block_count(whole) if f.lz4_weight_bytes > 0 else None
        if blocks is not None:
            f.lz4_plan_size_bytes = (f.plan_size_bytes - current_blob
                                     + WEIGHT_BLOCK_SECTION_HEADER_STRUCT.size
                                     + blocks * WEIGHT_BLOCK_SIZE
                                     + f.lz4_weight_bytes)
    if flash_budget > 0 and f.plan_size_bytes:
        f.plan_fits_flash = f.plan_size_bytes <= flash_budget
        f.int8_fits_flash = f.int8_plan_size_bytes <= flash_budget

    # Slow memory check (PSRAM) for tiled execution
    # Tiled execution pre-allocates full output in slow while input is
    # still in slow. If input + output > slow_budget, allocation fails.
    from tigris.analysis.validation import slow_pool_usage

    usage = slow_pool_usage(ag)
    f.slow_peak_bytes = usage.slow_peak_bytes
    f.slow_overflow_stages = list(usage.overflow_stage_ids)
    f.slow_fits = usage.fits
    if not f.slow_fits and f.verdict == "tiled":
        f.verdict = "needs_work"
        f.verdict_text = (
            f"Tiling resolves fast memory, but {len(f.slow_overflow_stages)} "
            f"stage(s) overflow slow memory ({fmt_bytes(f.slow_peak_bytes)} "
            f"needed, {fmt_bytes(ag.budget.slow)} available). "
            f"Need more PSRAM or smaller intermediate tensors."
        )

    # Budget sweep
    f.budget_sweep = _budget_sweep(ag)

    return f


def _budget_sweep(ag: AnalyzedGraph) -> list[BudgetRow]:
    if not ag.lifetimes or not ag.timeline:
        return []

    peak = ag.peak_memory_bytes
    user_budget = ag.mem_budget

    # Compute the minimum viable budget: partition maximally (one op per
    # stage), then find the largest peak among untileable ops. Tileable ops
    # can be brought down by spatial tiling, but untileable ones (Flatten,
    # Reshape, Gemm, etc.) need their full peak - that's the hard floor.
    from tigris.analysis.partition_spatial import op_category, TileCategory
    ag_max = copy.deepcopy(ag)
    for op in ag_max.ops:
        op.stage = -1
    ag_max.stages = []
    ag_max = partition_temporal(ag_max, 1)
    untileable_peaks = []
    for s in ag_max.stages:
        ops = [ag_max.ops[i] for i in s.op_indices]
        if any(op_category(op) == TileCategory.UNTILEABLE for op in ops):
            untileable_peaks.append(s.peak_bytes)
    floor = max(untileable_peaks) if untileable_peaks else 0

    candidates = [32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024,
                  512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024]
    budgets = sorted(set(
        [b for b in candidates if floor <= b <= peak * 2]
        + ([user_budget] if user_budget > 0 else [])
    ))
    if not budgets:
        return []

    ag_clean = copy.deepcopy(ag)
    for op in ag_clean.ops:
        op.stage = -1
    ag_clean.stages = []

    rows: list[BudgetRow] = []
    for b in budgets:
        ag_copy = copy.deepcopy(ag_clean)
        ag_copy = partition_temporal(ag_copy, b)
        n_stages = len(ag_copy.stages)
        n_tiling = sum(1 for s in ag_copy.stages if s.warnings)
        rows.append(BudgetRow(
            budget=b,
            budget_str=fmt_bytes(b),
            stages=n_stages,
            need_tiling=n_tiling,
            ok=n_tiling == 0,
        ))
    return rows
