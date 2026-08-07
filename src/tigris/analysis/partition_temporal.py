"""Temporal partitioning - greedy depth-axis graph partitioning for a given memory budget."""

from tigris.graph.ir import AnalyzedGraph, Stage


def _aligned_size(size_bytes: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("tensor_alignment must be a positive power of two")
    return (size_bytes + alignment - 1) & ~(alignment - 1)


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

    # Compute live bytes once.  The old implementation rescanned every tensor
    # at every step for every candidate stage, making an all-fit graph cubic in
    # its op count.  A candidate's peak is simply the running maximum of this
    # graph-wide timeline.
    step_bytes = _live_bytes_by_step(ag)

    tensor_death: dict[str, int] = {}
    for lt in ag.lifetimes.values():
        tensor_death[lt.tensor_name] = lt.death_step

    stages: list[Stage] = []
    current_start = 0

    while current_start < num_ops:
        # Try extending the stage one op at a time
        best_end = current_start  # inclusive end
        best_peak = 0
        running_peak = 0

        for candidate_end in range(current_start, num_ops):
            running_peak = max(running_peak, step_bytes[candidate_end])

            if running_peak <= budget:
                best_end = candidate_end
                best_peak = running_peak
            else:
                # This op doesn't fit - cut before it
                if candidate_end == current_start:
                    # Single op exceeds budget - include it with a warning
                    best_end = candidate_end
                    best_peak = running_peak
                break
        else:
            # All remaining ops fit
            best_end = num_ops - 1
            best_peak = running_peak

        stage = _build_stage(
            ag,
            len(stages),
            current_start,
            best_end,
            tensor_death,
            budget,
            best_peak,
        )
        stages.append(stage)

        # Mark ops with their stage assignment
        for step in range(current_start, best_end + 1):
            ag.ops[step].stage = stage.stage_id

        current_start = best_end + 1

    ag.stages = stages
    return ag


def _live_bytes_by_step(ag: AnalyzedGraph) -> list[int]:
    """Return aligned live activation bytes for each execution step.

    Lifetimes use inclusive birth/death steps.  Clipping each lifetime to the
    execution range and accumulating deltas preserves the memory model while
    taking O(ops + tensors) time.
    """
    num_ops = len(ag.ops)
    deltas = [0] * (num_ops + 1)

    for lt in ag.lifetimes.values():
        alive_from = max(0, lt.birth_step)
        freed_at = min(num_ops, lt.death_step + 1)
        if alive_from >= freed_at:
            continue

        size = _aligned_size(lt.size_bytes, ag.tensor_alignment)
        deltas[alive_from] += size
        deltas[freed_at] -= size

    live_bytes = 0
    result: list[int] = []
    for step in range(num_ops):
        live_bytes += deltas[step]
        result.append(live_bytes)
    return result


def _build_stage(
    ag: AnalyzedGraph,
    stage_id: int,
    start: int,
    end: int,
    tensor_death: dict[str, int],
    budget: int,
    peak: int,
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
