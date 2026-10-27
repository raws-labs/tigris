"""Planning a model for a memory budget, as ``tigris compile`` plans it.

``plan_model`` loads a model, partitions it into stages and tiles for a fast
budget, and with ``compress="lz4"`` reserves fast memory for the stages'
decompressed weights. Errors are raised as ``ValueError``.
"""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

# A budget is a sequence of pool sizes, fast first then slow, each a byte
# count or text such as "256K".
Budgets = Sequence[int | str]

_UINT32_MAX = 0xFFFFFFFF
# Each larger reservation comes from a new stage layout; the bound turns a
# planner oscillation into an error instead of an endless loop.
_RESERVE_ATTEMPTS = 64


def parse_size(s: str) -> int:
    """Bytes in ``s``: a byte count, or a number with a K, KB, M or MB suffix."""
    text = s.strip().upper()
    multipliers = {"K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2}
    try:
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if text.endswith(suffix):
                size = int(float(text[: -len(suffix)]) * mult)
                break
        else:
            size = int(text)
    except (ValueError, OverflowError):
        raise ValueError(
            f"invalid memory size {s!r}: expected bytes or a number with a K, KB, M "
            "or MB suffix, for example 256K"
        ) from None
    if size <= 0:
        raise ValueError(f"invalid memory size {s!r}: must be greater than zero")
    return size


def _pools(mem: Budgets) -> tuple[int, int]:
    """The fast and slow budgets in bytes; 0 for a pool not given."""
    pools = [parse_size(str(size)) for size in mem]
    return (pools[0] if pools else 0), (pools[1] if len(pools) > 1 else 0)


def load(model: str | Path, input_shapes: dict[str, tuple[int, ...]] | None = None):
    """Load a model and compute its tensor lifetimes and unscheduled memory."""
    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.loaders import load_model

    return compute_memory_timeline(compute_lifetimes(load_model(Path(model), input_shapes)))


def plan(ag, mem: Budgets = (), *, chains: bool = True):
    """Partition a loaded graph for the fast budget ``mem[0]`` and record the
    slow budget ``mem[1]``. Without a fast budget nothing is partitioned.
    ``chains=False`` leaves out the chain pass, so tiled stages stay apart."""
    return _planned(ag, *_pools(mem), 0, chains)


def _planned(ag, total: int, slow: int, reserve: int, chains: bool):
    """Partition ``ag`` for ``total`` fast bytes, ``reserve`` of them held
    outside the activation arena. Each subgraph a control-flow operator runs
    is planned as a graph of its own under the same budget, those it runs in
    turn first."""
    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.analysis.partition_spatial import detect_and_solve_chains, partition_spatial
    from tigris.analysis.partition_temporal import partition_temporal

    if not 0 <= ag.peak_memory_bytes <= _UINT32_MAX:
        raise ValueError("Peak activation memory exceeds the uint32 plan-format limit")
    if total > _UINT32_MAX:
        raise ValueError("Fast-memory budget exceeds the uint32 plan-format limit")
    _check_reserve(reserve, total)
    budget = total - reserve

    def planned(graph):
        if budget > 0:
            graph = partition_temporal(graph, budget)
            graph = partition_spatial(graph)
            if chains:
                graph = detect_and_solve_chains(graph)
        graph.budget = replace(graph.budget, fast_reserve=reserve)
        return graph

    def planned_below(graph):
        graph.subgraphs = [planned(planned_below(compute_memory_timeline(compute_lifetimes(sub))))
                           for sub in graph.subgraphs]
        return graph

    ag = planned(planned_below(ag))
    for graph in (*ag.subgraphs, ag):
        graph.budget = replace(graph.budget, slow=slow)
    return ag


def _check_reserve(reserve: int, total: int) -> None:
    """Refuse a weight reservation that leaves no fast memory for activations."""
    if reserve <= 0 or reserve < total:
        return
    from tigris.utils import fmt_bytes
    if reserve == total:
        # An activation budget of 0 would read as no budget, and nothing would be planned.
        raise ValueError(
            f"no plan written: decompressed weights need all {fmt_bytes(total)} "
            "of fast memory, none is left for activations"
        )
    raise ValueError(
        f"no plan written: decompressed weights need {fmt_bytes(reserve)} "
        f"of fast memory, the budget is {fmt_bytes(total)}"
    )


def _weight_reserve(ag) -> int:
    from tigris.emitters.binary.writer import compressed_weight_reserve_bytes

    try:
        return compressed_weight_reserve_bytes(ag)
    except ValueError as exc:
        raise ValueError(f"Cannot compress the plan: {exc}") from exc


def plan_model(
    model: str | Path,
    mem: Budgets = (),
    *,
    compress: str | None = None,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    chains: bool = True,
):
    """Load and plan a model the way ``tigris compile`` does.

    ``mem`` holds the fast pool, then optionally the slow pool. With
    ``compress="lz4"`` each stage's weights are decompressed into fast memory,
    so the activations are planned against what is left: weight blocks follow
    the stage layout and the layout follows the memory left, so the model is
    re-planned until the reservation it needs fits the one it was planned
    with. The graph's ``fast_memory_reserve_bytes`` is then that reservation.
    """
    if compress not in (None, "lz4"):
        raise ValueError(f"unknown weight compression {compress!r}: expected None or 'lz4'")
    total, slow = _pools(mem)
    if compress is None:
        return _planned(load(model, input_shapes), total, slow, 0, chains)
    reserve = 0
    for _ in range(_RESERVE_ATTEMPTS):
        ag = _planned(load(model, input_shapes), total, slow, reserve, chains)
        required = _weight_reserve(ag)
        if required <= reserve:
            # Keeping the larger reservation on a changing layout is
            # conservative: the plan never needs more fast memory than given.
            ag.budget = replace(ag.budget, fast_reserve=required)
            return ag
        reserve = required
    raise ValueError(
        f"Compressed-weight reservation did not converge after {_RESERVE_ATTEMPTS} "
        "planning attempts"
    )


def reserve_weights(ag) -> None:
    """Reserve fast memory for the decompressed weights of ``ag``'s stages,
    out of its fast budget, for a graph planned without compression whose
    stages were replaced afterwards."""
    if ag.fast_memory_reserve_bytes:
        raise ValueError("the graph already reserves fast memory for its weights")
    reserve = _weight_reserve(ag)
    _check_reserve(reserve, ag.mem_budget)
    ag.budget = replace(ag.budget, fast=ag.mem_budget - reserve, fast_reserve=reserve)
