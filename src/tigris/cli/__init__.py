"""CLI entry point: ``tigris analyze model.onnx --mem 256K``."""

from dataclasses import replace
from pathlib import Path

import click
from rich.console import Console

console = Console()


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


def _report_shape_bindings(
    ag, input_shapes: dict[str, tuple[int, ...]] | None = None
) -> None:
    """Say which shape the plan was built for.

    A dimension the compiler picked itself is a warning, since the plan is
    sized for a guess. A shape the caller named is echoed, not warned about.
    """
    for name, shape in sorted((input_shapes or {}).items()):
        extents = "x".join(str(dim) for dim in shape)
        console.print(f"{name} compiled for {extents}", style="dim")
    for note in ag.normalization_notes:
        console.print(f"[yellow]note:[/] {note}")
    for binding in ag.shape_bindings:
        console.print(f"[yellow]warning:[/] {binding}")
    if ag.shape_bindings:
        console.print(
            "  pass --input-shape NAME:1x3x224x224 to compile for another shape",
            style="dim",
        )


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
        with console.status("Loading model..."):
            ag = load_model(model_path, input_shapes)
            ag = compute_lifetimes(ag)
            ag = compute_memory_timeline(ag, capture_live_tensors=False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if report_bindings:
        _report_shape_bindings(ag, input_shapes)

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
        raise click.ClickException(
            "Fast-memory reservation "
            f"({fast_reserve_bytes:,} bytes) exceeds the total budget "
            f"({total_budget:,} bytes)"
        )
    budget = total_budget - fast_reserve_bytes
    if budget > 0:
        with console.status("Partitioning..."):
            ag = partition_temporal(ag, budget)
        with console.status("Computing spatial partitioning..."):
            ag = partition_spatial(ag)
            ag = detect_and_solve_chains(ag)

    ag.budget = replace(ag.budget, fast_reserve=fast_reserve_bytes)

    slow_budget = mem_pools[1] if len(mem_pools) > 1 else 0
    if len(mem) > 1 and slow_budget <= 0:
        raise click.ClickException("Slow-memory budget must be greater than zero")
    ag.budget = replace(ag.budget, slow=slow_budget)

    return ag, total_budget


@click.group()
@click.version_option(package_name="tigris-ml")
def cli():
    """TiGrIS - Tiled Graph Inference Scheduler"""


def main():
    cli()


# Register subcommands
from tigris.cli.analyze import analyze  # noqa: E402, F401
from tigris.cli.codegen import codegen  # noqa: E402, F401
from tigris.cli.compile import compile  # noqa: E402, F401
from tigris.cli.plan import plan  # noqa: E402, F401
from tigris.cli.simulate import simulate  # noqa: E402, F401
