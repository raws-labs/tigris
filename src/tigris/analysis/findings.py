"""Auto-generate actionable findings from an AnalyzedGraph.

Pure data - no HTML, no formatting. Used by both CLI and report."""

import copy
from dataclasses import dataclass, field

import numpy as np

from tigris.graph.ir import AnalyzedGraph
from tigris.analysis.partition_temporal import partition_temporal
from tigris.utils import fmt_bytes


@dataclass
class BudgetRow:
    budget: int
    budget_str: str
    stages: int
    need_tiling: int
    ok: bool


@dataclass
class UntileableStage:
    stage_id: int
    peak_bytes: int
    op_types: list[str]


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
    untileable_stages: list[UntileableStage] = field(default_factory=list)
    min_untileable_peak: int = 0
    feasibility_errors: list[str] = field(default_factory=list)
    unsupported_operators: list[str] = field(default_factory=list)
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
    int8_plan_size_bytes: int = 0
    lz4_weight_bytes: int = 0
    lz4_plan_size_bytes: int = 0
    flash_budget: int = 0
    plan_fits_flash: bool = False
    int8_fits_flash: bool = False

    # budget sweep
    budget_sweep: list[BudgetRow] = field(default_factory=list)

    # slow memory (PSRAM) constraints for tiled execution
    slow_budget: int = 0
    slow_peak_bytes: int = 0  # max(input + output) across tiled stages
    slow_overflow_stages: list[int] = field(default_factory=list)
    slow_fits: bool = True


def _estimate_weight_sizes(ag: AnalyzedGraph) -> tuple[int, int, int]:
    """Return serialized (current, hypothetical int8, LZ4) weight bytes."""
    raw_parts: list[bytes] = []
    int8_total = 0
    for arr in ag.weight_data.values():
        if arr.dtype in (np.int8, np.int32):
            raw = arr.tobytes()
        else:
            # The writer normalizes every other initializer to float32.
            raw = arr.astype(np.float32).tobytes()
        raw_parts.append(raw)
        if np.issubdtype(arr.dtype, np.floating):
            int8_total += arr.size  # 1 byte per element
        else:
            int8_total += len(raw)

    # LZ4 on the actual weight data (meaningful for the current dtype only)
    lz4_total = 0
    try:
        import lz4.block
        blob = b"".join(raw_parts)
        if blob:
            lz4_total = len(lz4.block.compress(blob, store_size=False))
    except ImportError:
        pass

    return sum(map(len, raw_parts)), int8_total, lz4_total


def _serialized_weight_sizes(ag: AnalyzedGraph, *, int8: bool) -> list[int]:
    """Return per-entry blob sizes for one plan-size profile."""
    sizes = []
    for arr in ag.weight_data.values():
        if int8 and np.issubdtype(arr.dtype, np.floating):
            sizes.append(int(arr.size))
        elif arr.dtype in (np.int8, np.int32):
            sizes.append(int(arr.nbytes))
        else:
            sizes.append(int(arr.size) * np.dtype(np.float32).itemsize)
    return sizes


def _aligned_weight_blob_size(sizes: list[int], alignment: int) -> int:
    """Mirror the per-weight padding used by the uncompressed writer."""
    size = 0
    for weight_size in sizes:
        size = (size + alignment - 1) // alignment * alignment
        size += weight_size
    return size


def _index_pool_item_count(ag: AnalyzedGraph, runtime_names: set[str]) -> int:
    """Count the uint16 elements appended by the writer's index pool."""
    count = sum(name in runtime_names for name in ag.model_inputs)
    count += sum(name in runtime_names for name in ag.model_outputs)
    for op in ag.ops:
        count += sum(name in runtime_names for name in op.inputs)
        count += sum(name in runtime_names for name in op.outputs)
    for stage in ag.stages:
        count += len(stage.op_indices)
        count += sum(name in runtime_names for name in stage.input_tensors)
        count += sum(name in runtime_names for name in stage.output_tensors)
    return count


def _string_pool_size(ag: AnalyzedGraph, runtime_names: set[str]) -> int:
    """Measure the writer's deduplicated, NUL-terminated UTF-8 strings."""
    names = [ag.model_name]
    names.extend(ag.weight_data)
    names.extend(name for name in ag.tensors if name in runtime_names)
    names.extend(op.name for op in ag.ops)
    unique_names = dict.fromkeys(names)
    if not unique_names:
        return 1
    return sum(len(name.encode("utf-8")) + 1 for name in unique_names)


def _op_attribute_section_size(ag: AnalyzedGraph) -> int:
    """Measure the optional typed operator-attribute section."""
    from tigris.emitters.binary.defs import (
        OP_ATTRIBUTE_SECTION_HEADER_STRUCT,
        OP_ATTRIBUTE_SIZE,
    )

    payload_bytes = 0
    count = 0
    for op in ag.ops:
        if op.op_type != "Transpose":
            continue
        count += 1
        payload_bytes += len(ag.tensors[op.inputs[0]].shape)
    if not count:
        return 0
    return OP_ATTRIBUTE_SECTION_HEADER_STRUCT.size + count * OP_ATTRIBUTE_SIZE + payload_bytes


def _compressed_weight_block_count(ag: AnalyzedGraph) -> int:
    """Count stage-local blocks created by compressed serialization."""
    weight_names = set(ag.weight_data)
    stages_with_weights: set[int] = set()
    for stage in ag.stages:
        for op_index in stage.op_indices:
            if any(name in weight_names for name in ag.ops[op_index].inputs):
                stages_with_weights.add(stage.stage_id)
                break
    return len(stages_with_weights)


def _estimate_serialized_plan_size(
    ag: AnalyzedGraph,
    weight_sizes: list[int],
    *,
    compressed_weight_bytes: int = 0,
) -> int:
    """Estimate one complete plan using the canonical schema layouts."""
    from tigris.emitters.binary.defs import (
        HEADER_SIZE,
        INDEX_STRUCT,
        OP_SIZE,
        PLAN_SECTION_ALIGNMENT,
        SEC_INDEX_POOL,
        SEC_OP_ATTRIBUTES,
        SEC_OPS,
        SEC_QUANT_PARAMS,
        SEC_SHAPE_POOL,
        SEC_STAGES,
        SEC_STRINGS,
        SEC_TENSORS,
        SEC_TILE_PLANS,
        SEC_WEIGHT_BLOCKS,
        SEC_WEIGHTS,
        SECTION_ENTRY_SIZE,
        SHAPE_DIM_STRUCT,
        STAGE_SIZE,
        TENSOR_SIZE,
        TILE_PLAN_SIZE,
        WEIGHT_BLOCK_SECTION_HEADER_STRUCT,
        WEIGHT_BLOCK_SIZE,
        WEIGHT_ENTRY_SIZE,
    )
    from tigris.emitters.binary.writer import _build_quant_params

    runtime_tensors = [
        (name, info) for name, info in ag.tensors.items() if not info.is_constant
    ]
    runtime_names = {name for name, _ in runtime_tensors}
    quant_size = len(_build_quant_params(ag)[0])
    attribute_size = _op_attribute_section_size(ag)

    section_sizes = [
        (SEC_TENSORS, len(runtime_tensors) * TENSOR_SIZE),
        (SEC_OPS, len(ag.ops) * OP_SIZE),
        (SEC_STAGES, len(ag.stages) * STAGE_SIZE),
        (
            SEC_TILE_PLANS,
            sum(stage.tile_plan is not None for stage in ag.stages) * TILE_PLAN_SIZE,
        ),
        (SEC_INDEX_POOL, _index_pool_item_count(ag, runtime_names) * INDEX_STRUCT.size),
        (
            SEC_SHAPE_POOL,
            sum(len(info.shape) for _, info in runtime_tensors) * SHAPE_DIM_STRUCT.size,
        ),
        (SEC_STRINGS, _string_pool_size(ag, runtime_names)),
    ]

    num_weights = len(weight_sizes)
    if num_weights:
        if compressed_weight_bytes:
            weight_section_size = num_weights * WEIGHT_ENTRY_SIZE
        else:
            weight_section_size = (
                num_weights * WEIGHT_ENTRY_SIZE
                + _aligned_weight_blob_size(weight_sizes, PLAN_SECTION_ALIGNMENT)
            )
        section_sizes.append((SEC_WEIGHTS, weight_section_size))
    if quant_size:
        section_sizes.append((SEC_QUANT_PARAMS, quant_size))
    if num_weights and compressed_weight_bytes:
        block_count = _compressed_weight_block_count(ag)
        section_sizes.append((
            SEC_WEIGHT_BLOCKS,
            WEIGHT_BLOCK_SECTION_HEADER_STRUCT.size
            + block_count * WEIGHT_BLOCK_SIZE
            + compressed_weight_bytes,
        ))
    if attribute_size:
        section_sizes.append((SEC_OP_ATTRIBUTES, attribute_size))

    current = HEADER_SIZE + (len(section_sizes) + 1) * SECTION_ENTRY_SIZE
    for section_type, section_size in section_sizes:
        padding = (-current) % PLAN_SECTION_ALIGNMENT
        if section_type == SEC_WEIGHTS:
            blob_pos = current + padding + num_weights * WEIGHT_ENTRY_SIZE
            padding += (-blob_pos) % PLAN_SECTION_ALIGNMENT
        current += padding + section_size
    return current


def _estimate_plan_overhead(ag: AnalyzedGraph) -> int:
    """Estimate current-plan bytes other than serialized weight payloads."""
    weight_sizes = _serialized_weight_sizes(ag, int8=False)
    return _estimate_serialized_plan_size(ag, weight_sizes) - sum(weight_sizes)


def compute_findings(ag: AnalyzedGraph, flash_budget: int = 0, slow_budget: int = 0) -> Findings:
    """Analyze the graph and produce structured findings."""
    f = Findings()
    peak = ag.peak_memory_bytes
    budget = ag.mem_budget
    f.peak_bytes = peak
    f.budget = budget
    f.flash_budget = flash_budget
    f.slow_budget = slow_budget

    from tigris.analysis.validation import (
        validate_execution_dtype,
        validate_operator_support,
    )

    operator_validation = validate_operator_support(ag)
    f.unsupported_operators = [
        issue.describe() for issue in operator_validation.issues
    ]
    dtype_validation = validate_execution_dtype(ag)
    f.dtype_errors = list(dtype_validation.issues)

    if not ag.lifetimes:
        if f.unsupported_operators or f.dtype_errors:
            f.verdict = "needs_work"
            if f.unsupported_operators:
                f.verdict_text = (
                    "Unsupported operators: " + ", ".join(f.unsupported_operators)
                )
            else:
                f.verdict_text = f.dtype_errors[0]
        return f

    # Largest tensor
    sorted_lt = sorted(ag.lifetimes.values(), key=lambda lt: -lt.size_bytes)
    largest = sorted_lt[0]
    largest_info = ag.tensors.get(largest.tensor_name)
    f.largest_tensor_name = largest.tensor_name
    f.largest_tensor_bytes = largest.size_bytes
    if largest_info:
        f.largest_tensor_shape = "x".join(str(d) for d in largest_info.shape)

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
            else:
                f.stages_untileable += 1
                stage_op_types = []
                for op_desc in s.tile_plan.untileable_ops:
                    # op_desc is "name (OpType)" - extract the type
                    if "(" in op_desc:
                        op_type = op_desc.split("(")[-1].rstrip(")")
                    else:
                        op_type = op_desc
                    untileable_ops.add(op_type)
                    stage_op_types.append(op_type)
                f.untileable_stages.append(UntileableStage(
                    stage_id=s.stage_id,
                    peak_bytes=s.peak_bytes,
                    op_types=stage_op_types,
                ))
    f.untileable_op_types = sorted(untileable_ops)
    if f.untileable_stages:
        f.untileable_stages.sort(key=lambda u: -u.peak_bytes)
        f.min_untileable_peak = f.untileable_stages[0].peak_bytes

    # Analysis and deployment share one execution-unit validator, so a CLI
    # PASS cannot disagree with `tigris compile` about memory feasibility.
    from tigris.analysis.validation import validate_memory_plan

    validation = validate_memory_plan(ag)
    f.scheduled_peak_bytes = validation.scheduled_peak_bytes
    f.feasibility_errors = [issue.describe() for issue in validation.issues]

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
    elif f.dtype_errors:
        f.verdict = "needs_work"
        f.verdict_text = f.dtype_errors[0]

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

    # Deployment size (weights + plan)
    f.total_weight_bytes, f.int8_weight_bytes, f.lz4_weight_bytes = _estimate_weight_sizes(ag)
    current_weight_sizes = _serialized_weight_sizes(ag, int8=False)
    int8_weight_sizes = _serialized_weight_sizes(ag, int8=True)
    f.plan_size_bytes = _estimate_serialized_plan_size(ag, current_weight_sizes)
    f.plan_overhead_bytes = f.plan_size_bytes - f.total_weight_bytes
    f.int8_plan_size_bytes = _estimate_serialized_plan_size(ag, int8_weight_sizes)
    if f.lz4_weight_bytes > 0:
        f.lz4_plan_size_bytes = _estimate_serialized_plan_size(
            ag,
            current_weight_sizes,
            compressed_weight_bytes=f.lz4_weight_bytes,
        )
    if flash_budget > 0:
        f.plan_fits_flash = f.plan_size_bytes <= flash_budget
        f.int8_fits_flash = f.int8_plan_size_bytes <= flash_budget

    # Slow memory check (PSRAM) for tiled execution
    # Tiled execution pre-allocates full output in slow while input is
    # still in slow. If input + output > slow_budget, allocation fails.
    if slow_budget > 0 and ag.stages:
        for s in ag.stages:
            # Only check stages that need tiling
            if s.peak_bytes > budget:
                in_size = sum(
                    ag.tensors[n].size_bytes for n in s.input_tensors
                    if n in ag.tensors
                )
                out_size = sum(
                    ag.tensors[n].size_bytes for n in s.output_tensors
                    if n in ag.tensors
                )
                stage_slow = in_size + out_size
                if stage_slow > f.slow_peak_bytes:
                    f.slow_peak_bytes = stage_slow
                if stage_slow > slow_budget:
                    f.slow_overflow_stages.append(s.stage_id)
                    f.slow_fits = False

        # Update verdict if slow memory overflows
        if not f.slow_fits and f.verdict == "tiled":
            f.verdict = "needs_work"
            f.verdict_text = (
                f"Tiling resolves fast memory, but {len(f.slow_overflow_stages)} "
                f"stage(s) overflow slow memory ({fmt_bytes(f.slow_peak_bytes)} "
                f"needed, {fmt_bytes(slow_budget)} available). "
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
    from tigris.analysis.partition_spatial import classify_op, TileCategory
    ag_max = copy.deepcopy(ag)
    for op in ag_max.ops:
        op.stage = -1
    ag_max.stages = []
    ag_max = partition_temporal(ag_max, 1)
    untileable_peaks = []
    for s in ag_max.stages:
        ops = [ag_max.ops[i] for i in s.op_indices]
        if any(classify_op(op.op_type) == TileCategory.UNTILEABLE for op in ops):
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
