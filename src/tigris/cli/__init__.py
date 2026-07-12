"""CLI entry point: ``tigris analyze model.onnx --mem 256K``."""

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


def _run_pipeline(model: str, mem: tuple[str, ...]):
    """Shared pipeline: load -> lifetimes -> memory -> partition -> tiling. Returns (ag, budget)."""
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

    try:
        with console.status("Loading model..."):
            ag = load_model(model_path)
            ag = compute_lifetimes(ag)
            ag = compute_memory_timeline(ag)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not 0 <= ag.peak_memory_bytes <= 0xFFFFFFFF:
        raise click.ClickException(
            "Peak activation memory exceeds the uint32 plan-format limit"
        )

    budget = mem_pools[0] if mem_pools else 0
    if budget > 0xFFFFFFFF:
        raise click.ClickException(
            "Fast-memory budget exceeds the uint32 plan-format limit"
        )
    if budget > 0:
        with console.status("Partitioning..."):
            ag = partition_temporal(ag, budget)
        with console.status("Computing spatial partitioning..."):
            ag = partition_spatial(ag)
            ag = detect_and_solve_chains(ag)

    return ag, budget


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
