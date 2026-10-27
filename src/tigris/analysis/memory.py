"""Event-sweep peak activation memory."""

from tigris.graph.ir import AnalyzedGraph


# The runtime rounds every allocation up to TIGRIS_TENSOR_ALIGN (tigris_mem.h).
# 32 covers every value that macro auto-detects except SVE (64), whose build stops.
TENSOR_ALIGN = 32


def align_up(size_bytes: int, alignment: int = TENSOR_ALIGN) -> int:
    """Round a size up to a multiple of the allocation alignment."""
    return (size_bytes + alignment - 1) // alignment * alignment


def compute_memory_timeline(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Record the peak live activation memory over the execution steps, using
    an event-sweep.

    For each step, the live memory is the sum of sizes of all activation tensors
    whose lifetime spans that step: birth_step <= step <= death_step.
    """
    num_ops = len(ag.ops)
    if num_ops == 0:
        return ag

    # Build sorted event list: (step, +/- physically allocated size)
    # +size at birth_step: the runtime (exec_stage_normal) ALLOCATES an op's
    #   output before running the kernel and frees the consumed inputs only after
    #   it returns, so the output is live AT its producing step, co-resident with
    #   the inputs that die there. Counting it at birth_step+1 instead misses that
    #   overlap (undercounts peak ~output-size) and drops the final op's output
    #   entirely (its birth_step+1 falls past the end of the sweep).
    # -size at death_step+1 (tensor freed after its last consumer finishes)
    events: list[tuple[int, int]] = []
    for lt in ag.lifetimes.values():
        alive_from = lt.birth_step      # live starting AT the producing step
        freed_at = lt.death_step + 1    # freed *after* the last consumer
        # For model inputs (birth=-1): alive_from=0
        # For model outputs (death=num_ops): freed_at=num_ops+1 (never freed during exec)
        size = align_up(lt.size_bytes)
        events.append((alive_from, size))
        events.append((freed_at, -size))

    # Sort by step, then frees before allocs at the same step
    events.sort(key=lambda e: (e[0], e[1]))

    # Sweep through steps 0..num_ops-1
    live_bytes = 0
    ev_idx = 0
    peak = 0

    for step in range(num_ops):
        # Apply all events at this step
        while ev_idx < len(events) and events[ev_idx][0] <= step:
            live_bytes += events[ev_idx][1]
            ev_idx += 1
        if live_bytes > peak:
            peak = live_bytes

    ag.peak_memory_bytes = peak
    return ag
