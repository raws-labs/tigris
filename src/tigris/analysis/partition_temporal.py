"""Temporal partitioning - greedy depth-axis graph partitioning for a given memory budget."""

from dataclasses import replace

from tigris.graph.ir import AnalyzedGraph, Stage


def _aligned_size(size_bytes: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("tensor_alignment must be a positive power of two")
    return (size_bytes + alignment - 1) & ~(alignment - 1)




class _StageCost:
    """The fast bytes a candidate stage holds, grown one op at a time.

    Mirrors the executor's stage loop, which loads the stage inputs from the
    slow pool, allocates each op output as it runs, frees a tensor after its
    last read inside the stage, spills the stage outputs and then resets the
    fast arena. Nothing survives that reset, so a tensor produced before the
    stage and consumed after it costs the stage nothing: it sits in the slow
    pool throughout. Summing the graph-wide live set instead charges the stage
    for those bytes and cuts long before it has to.

    A tensor produced inside the stage is held from its own step until its
    last read, or until the end of the stage when something later reads it. A
    stage input is held from the start of the stage until its last read.
    """

    def __init__(self, ag: AnalyzedGraph, start: int) -> None:
        self._ag = ag
        self._start = start
        self._align = ag.tensor_alignment
        self._bytes_at: dict[int, int] = {}
        self._produced_open = 0
        self._dying: dict[int, int] = {}
        self._input_covered: dict[str, int] = {}
        self.peak = 0

    def _size(self, name: str) -> int | None:
        lt = self._ag.lifetimes.get(name)
        if lt is None:
            return None
        return _aligned_size(lt.size_bytes, self._align)

    def _charge(self, step: int, size: int) -> None:
        total = self._bytes_at.get(step, 0) + size
        self._bytes_at[step] = total
        if total > self.peak:
            self.peak = total

    def extend(self, step: int) -> int:
        """Add one op and return the candidate stage's peak."""
        op = self._ag.ops[step]

        # Tensors produced earlier in the stage stop costing once nothing
        # reads them again; the rest are held to the end of the stage.
        self._produced_open -= self._dying.pop(step - 1, 0)
        for name in op.outputs:
            size = self._size(name)
            if size is None:
                continue
            self._produced_open += size
            death = self._ag.lifetimes[name].death_step
            self._dying[death] = self._dying.get(death, 0) + size
        self._charge(step, self._produced_open)

        # A stage input is loaded at the start of the stage, so a later read
        # widens what it has already cost, back to the first op.
        for name in op.inputs:
            size = self._size(name)
            if size is None:
                continue
            if self._ag.lifetimes[name].birth_step >= self._start:
                continue
            covered = self._input_covered.get(name)
            first = self._start if covered is None else covered + 1
            for earlier in range(first, step + 1):
                self._charge(earlier, size)
            self._input_covered[name] = step

        return self.peak


def partition_temporal(
    ag: AnalyzedGraph,
    budget: int,
    forced_cuts: frozenset[int] | None = None,
) -> AnalyzedGraph:
    """Partition the execution graph into sequential stages.

    Greedy forward walk: accumulate ops into the current stage. When adding
    the next op would push the stage's peak live memory above *budget*,
    cut before it and start a new stage.

    *forced_cuts* names op indices that must begin a stage whatever the
    memory model says. Spatial partitioning supplies them for a stage it
    could not tile, where a different set of stage boundaries can be.

    Stage inputs = tensors produced outside the stage but consumed inside.
    Stage outputs = tensors produced inside the stage but consumed later (or model outputs).
    """
    cuts = forced_cuts or frozenset()
    ag.budget = replace(ag.budget, fast=budget)
    num_ops = len(ag.ops)
    if num_ops == 0:
        return ag

    tensor_death: dict[str, int] = {}
    for lt in ag.lifetimes.values():
        tensor_death[lt.tensor_name] = lt.death_step

    stages: list[Stage] = []
    current_start = 0

    while current_start < num_ops:
        # Try extending the stage one op at a time
        best_end = current_start  # inclusive end
        best_peak = 0
        cost = _StageCost(ag, current_start)

        for candidate_end in range(current_start, num_ops):
            if candidate_end > current_start and candidate_end in cuts:
                # A caller-required boundary: stop before it.
                break

            running_peak = cost.extend(candidate_end)

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
            best_peak = cost.peak

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
            f"({budget:,} bytes). Needs tiled streaming."
        )

    return Stage(
        stage_id=stage_id,
        op_indices=op_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        peak_bytes=peak,
        warnings=warnings,
    )
