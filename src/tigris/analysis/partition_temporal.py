"""Temporal partitioning - greedy depth-axis graph partitioning for a given memory budget."""

from tigris.graph.ir import AnalyzedGraph, Stage


def partition_temporal(ag: AnalyzedGraph, budget: int) -> AnalyzedGraph:
    """Partition the execution graph into sequential stages.

    Greedy forward walk: accumulate ops into the current stage. When adding
    the next op would push the stage's peak live memory above *budget*,
    cut before it and start a new stage.

    Stage inputs = tensors produced outside the stage but consumed inside.
    Stage outputs = tensors produced inside the stage but consumed later (or model outputs).
    """
    ag.mem_budget = budget
    num_ops = len(ag.ops)
    if num_ops == 0:
        return ag

    # Pre-compute which tensors are alive at each step (from the timeline)
    step_live: dict[int, list[str]] = {}
    step_bytes: dict[int, int] = {}
    for snap in ag.timeline:
        step_live[snap.step] = snap.live_tensors
        step_bytes[snap.step] = snap.live_bytes

    # Which step each tensor is produced at
    tensor_birth: dict[str, int] = {}
    tensor_death: dict[str, int] = {}
    for lt in ag.lifetimes.values():
        tensor_birth[lt.tensor_name] = lt.birth_step
        tensor_death[lt.tensor_name] = lt.death_step

    stages: list[Stage] = []
    current_start = 0

    while current_start < num_ops:
        # Try extending the stage one op at a time
        best_end = current_start  # inclusive end

        for candidate_end in range(current_start, num_ops):
            # Compute peak memory for the candidate stage [current_start..candidate_end]
            # We need to account for stage inputs: tensors born outside this range
            # but alive inside it. These must be loaded into SRAM.
            stage_peak = _stage_peak_memory(
                ag, current_start, candidate_end, tensor_birth, tensor_death
            )

            if stage_peak <= budget:
                best_end = candidate_end
            else:
                # This op doesn't fit - cut before it
                if candidate_end == current_start:
                    # Single op exceeds budget - include it with a warning
                    best_end = candidate_end
                break
        else:
            # All remaining ops fit
            best_end = num_ops - 1

        stage = _build_stage(
            ag, len(stages), current_start, best_end,
            tensor_birth, tensor_death, budget,
        )
        stages.append(stage)

        # Mark ops with their stage assignment
        for step in range(current_start, best_end + 1):
            ag.ops[step].stage = stage.stage_id

        current_start = best_end + 1

    ag.stages = stages
    return ag


def _stage_peak_memory(
    ag: AnalyzedGraph,
    start: int,
    end: int,
    tensor_birth: dict[str, int],
    tensor_death: dict[str, int],
) -> int:
    """Compute peak live activation memory for a stage spanning [start..end].

    A tensor is live within the stage if:
      - It was born within the stage (birth_step in [start..end]), or
      - It was born before the stage but is consumed within it (a stage input).
    And it hasn't died before the step we're examining.
    """
    peak = 0
    for step in range(start, end + 1):
        live = 0
        for lt in ag.lifetimes.values():
            # Tensor is alive from its producing step (output co-resides with the
            # inputs that die there - the runtime allocs the output before freeing
            # inputs) through its last consumer. Same convention as memory.py.
            alive_from = lt.birth_step
            freed_at = lt.death_step + 1

            # Within this stage, the tensor is live if:
            # it's alive at this step AND (born in stage OR consumed in stage)
            if alive_from <= step and step < freed_at:
                live += lt.size_bytes
        if live > peak:
            peak = live
    return peak


def _build_stage(
    ag: AnalyzedGraph,
    stage_id: int,
    start: int,
    end: int,
    tensor_birth: dict[str, int],
    tensor_death: dict[str, int],
    budget: int,
) -> Stage:
    """Build a Stage object with input/output tensor accounting."""
    op_indices = list(range(start, end + 1))

    # Tensors produced inside this stage
    produced_in_stage: set[str] = set()
    for step in op_indices:
        for out in ag.ops[step].outputs:
            if out and out in ag.lifetimes:
                produced_in_stage.add(out)

    # Tensors consumed inside this stage
    consumed_in_stage: set[str] = set()
    for step in op_indices:
        for inp in ag.ops[step].inputs:
            if inp and inp in ag.lifetimes:
                consumed_in_stage.add(inp)

    # Stage inputs: consumed but not produced here (must be loaded from slow mem)
    input_tensors = sorted(consumed_in_stage - produced_in_stage)

    # Stage outputs: produced here but consumed after this stage ends (must be spilled)
    output_tensors: list[str] = []
    for name in sorted(produced_in_stage):
        death = tensor_death.get(name, -1)
        if death > end:
            output_tensors.append(name)

    # Peak memory for this stage
    peak = _stage_peak_memory(ag, start, end, tensor_birth, tensor_death)

    warnings: list[str] = []
    if peak > budget:
        warnings.append(
            f"Stage {stage_id} peak ({peak:,} bytes) exceeds budget "
            f"({budget:,} bytes). Needs tiled streaming (Phase 2)."
        )

    return Stage(
        stage_id=stage_id,
        op_indices=op_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        peak_bytes=peak,
        warnings=warnings,
    )
