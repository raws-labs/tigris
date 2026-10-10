"""CLI entry point: ``tigris analyze model.onnx --mem 256K``."""

from dataclasses import replace
import os
from pathlib import Path

import click


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    multipliers = {"K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def _expand_mem(ctx, param, value: tuple[str, ...]) -> tuple[str, ...]:
    """Expand ``-m 256K+4M`` into separate pools. Sugar for repeated ``-m``."""
    expanded: list[str] = []
    for token in value:
        parts = token.split("+")
        for part in parts:
            if not part.strip():
                raise click.BadParameter(
                    f"invalid memory budget {token!r}: empty memory pool"
                )
            expanded.append(part.strip())
    if len(expanded) > 2:
        raise click.BadParameter(
            f"{len(expanded)} memory pools given; the planner supports a fast "
            "and a slow pool"
        )
    return tuple(expanded)


def _parse_input_shape(ctx, param, value: tuple[str, ...]):
    """Parse ``--input-shape name:1x3x224x224`` into a name to shape mapping."""
    shapes: dict[str, tuple[int, ...]] = {}
    for token in value:
        name, sep, extents = token.rpartition(":")
        if not sep or not name:
            raise click.BadParameter(
                f"invalid input shape {token!r}: expected name:1x3x224x224"
            )
        try:
            dims = tuple(int(part) for part in extents.split("x"))
        except ValueError:
            raise click.BadParameter(
                f"invalid input shape {token!r}: dimensions must be integers"
            ) from None
        if not dims or any(dim <= 0 for dim in dims):
            raise click.BadParameter(
                f"invalid input shape {token!r}: dimensions must be positive"
            )
        shapes[name] = dims
    return shapes


def _report_shape_bindings(ag) -> None:
    """Warn about a dimension the compiler picked itself.

    The plan is sized for a guess, which the interface rows cannot show: they
    state the shape without saying where it came from. A shape the caller named
    needs no warning, and the rows already report it.
    """
    for binding in ag.shape_bindings:
        from tigris.cli import text
        text.echo(text.warn("warning: ") + str(binding), err=True)


def _run_pipeline(
    model: str,
    mem: tuple[str, ...],
    *,
    fast_reserve_bytes: int = 0,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    report_bindings: bool = True,
):
    """Run the shared planning pipeline.

    ``--mem`` is the total fast arena supplied by the embedding application.
    A caller that needs bytes outside the activation allocator (for example,
    decompressed weights) supplies ``fast_reserve_bytes``; partitioning then
    sees only the remaining activation capacity.  The returned budget remains
    the caller's total, preserving the CLI's public accounting.
    """
    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.analysis.partition_spatial import (
        detect_and_solve_chains,
        partition_spatial,
    )
    from tigris.analysis.partition_temporal import partition_temporal
    from tigris.loaders import load_model

    model_path = Path(model)
    mem_pools = [_parse_size(m) for m in mem]
    if fast_reserve_bytes < 0:
        raise click.ClickException("Fast-memory reservation must not be negative")

    try:
        ag = load_model(model_path, input_shapes)
        ag = compute_lifetimes(ag)
        ag = compute_memory_timeline(ag, capture_live_tensors=False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if report_bindings:
        _report_shape_bindings(ag)

    if not 0 <= ag.peak_memory_bytes <= 0xFFFFFFFF:
        raise click.ClickException(
            "Peak activation memory exceeds the uint32 plan-format limit"
        )

    total_budget = mem_pools[0] if mem_pools else 0
    if total_budget > 0xFFFFFFFF:
        raise click.ClickException(
            "Fast-memory budget exceeds the uint32 plan-format limit"
        )
    if total_budget >= 0 and fast_reserve_bytes > total_budget:
        # Only compressed plans reserve fast memory, for decompressed weights.
        from tigris.utils import fmt_bytes
        raise click.ClickException(
            f"no plan written: decompressed weights need {fmt_bytes(fast_reserve_bytes)} "
            f"of fast memory, the budget is {fmt_bytes(total_budget)}"
        )
    budget = total_budget - fast_reserve_bytes

    def planned(graph):
        if budget > 0:
            graph = partition_temporal(graph, budget)
            graph = partition_spatial(graph)
            graph = detect_and_solve_chains(graph)
        graph.budget = replace(graph.budget, fast_reserve=fast_reserve_bytes)
        return graph

    # Each subgraph a control-flow operator runs is planned as a graph of its
    # own under the same budget, those it runs in turn first.
    def planned_below(graph):
        graph.subgraphs = [planned(planned_below(compute_memory_timeline(
            compute_lifetimes(sub), capture_live_tensors=False))) for sub in graph.subgraphs]
        return graph

    ag = planned(planned_below(ag))

    slow_budget = mem_pools[1] if len(mem_pools) > 1 else 0
    if len(mem) > 1 and slow_budget <= 0:
        raise click.ClickException("Slow-memory budget must be greater than zero")
    for graph in (*ag.subgraphs, ag):
        graph.budget = replace(graph.budget, slow=slow_budget)

    return ag, total_budget


def _show_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    from importlib.metadata import version
    from tigris.runtime import runtime_info

    try:
        info = runtime_info()
        runtime = f"{info['version']}; {info['source']}"
    except ValueError as exc:
        if "TIGRIS_HOST_LIBRARY" in os.environ:
            raise click.ClickException(str(exc)) from exc
        runtime = "unavailable"
    click.echo(f"tigris, version {version('tigris-ml')} (runtime {runtime})")
    ctx.exit()


@click.group()
@click.option("--version", is_flag=True, is_eager=True, expose_value=False,
              callback=_show_version, help="Show compiler and host runtime versions and runtime origin.")
def cli():
    """TiGrIS - Tiled Graph Inference Scheduler

    Fits ONNX and TFLite models into the memory of embedded devices: analyze a
    model against a budget, compile it into a tiled execution plan, generate
    the C code that runs it.
    """


def main():
    cli()


# Register subcommands
from tigris.cli.analyze import analyze  # noqa: E402, F401
from tigris.cli.codegen import codegen  # noqa: E402, F401
from tigris.cli.compile import compile  # noqa: E402, F401
from tigris.cli.inspect import inspect  # noqa: E402, F401
from tigris.cli.plan import plan  # noqa: E402, F401
from tigris.cli.run import run  # noqa: E402, F401
from tigris.cli.simulate import simulate  # noqa: E402, F401
from tigris.cli.zoo import zoo  # noqa: E402, F401
