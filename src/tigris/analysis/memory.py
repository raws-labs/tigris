"""Event-sweep memory timeline and peak calculation."""

from tigris.graph.ir import AnalyzedGraph, MemorySnapshot


def compute_memory_timeline(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Compute the memory footprint at each execution step using an event-sweep.

    For each step, the live memory is the sum of sizes of all activation tensors
    whose lifetime spans that step: birth_step < step <= death_step.

    Produces a MemorySnapshot per step and records the overall peak.
    """
    num_ops = len(ag.ops)
    if num_ops == 0:
        return ag

    # Build sorted event list: (step, +/- size, tensor_name)
    # +size at birth_step: the runtime (exec_stage_normal) ALLOCATES an op's
    #   output before running the kernel and frees the consumed inputs only after
    #   it returns, so the output is live AT its producing step, co-resident with
    #   the inputs that die there. Counting it at birth_step+1 instead misses that
    #   overlap (undercounts peak ~output-size) and drops the final op's output
    #   entirely (its birth_step+1 falls past the end of the sweep).
    # -size at death_step+1 (tensor freed after its last consumer finishes)
    events: list[tuple[int, int, str]] = []
    for lt in ag.lifetimes.values():
        alive_from = lt.birth_step      # live starting AT the producing step
        freed_at = lt.death_step + 1    # freed *after* the last consumer
        # For model inputs (birth=-1): alive_from=0
        # For model outputs (death=num_ops): freed_at=num_ops+1 (never freed during exec)
        events.append((alive_from, lt.size_bytes, lt.tensor_name))
        events.append((freed_at, -lt.size_bytes, lt.tensor_name))

    # Sort by step, then frees before allocs at the same step
    events.sort(key=lambda e: (e[0], e[1]))

    # Sweep through steps 0..num_ops-1
    live_bytes = 0
    live_set: set[str] = set()
    ev_idx = 0

    timeline: list[MemorySnapshot] = []
    peak = 0

    for step in range(num_ops):
        # Apply all events at this step
        while ev_idx < len(events) and events[ev_idx][0] <= step:
            _, delta, tname = events[ev_idx]
            live_bytes += delta
            if delta > 0:
                live_set.add(tname)
            else:
                live_set.discard(tname)
            ev_idx += 1

        snapshot = MemorySnapshot(
            step=step,
            live_bytes=live_bytes,
            live_tensors=sorted(live_set),
        )
        timeline.append(snapshot)
        if live_bytes > peak:
            peak = live_bytes

    ag.timeline = timeline
    ag.peak_memory_bytes = peak
    return ag
