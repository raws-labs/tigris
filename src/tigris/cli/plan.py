"""``tigris plan`` command."""

from pathlib import Path

import click

from tigris.cli import cli, console, _expand_mem, _run_pipeline
from tigris.utils import fmt_bytes


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, required=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--output", "-o", default=None, help="Output path (default: <model>.plan.yaml)")
def plan(model: str, mem: tuple[str, ...], output: str | None):
    """Generate a YAML execution plan for memory-constrained deployment."""
    from tigris.emitters.yaml import emit_yaml

    ag, budget = _run_pipeline(model, mem)

    out = Path(output) if output else Path(model).with_suffix(".plan.yaml")
    with console.status("Writing plan..."):
        emit_yaml(ag, out)

    console.print(f"[bold green]Plan written to {out}[/]")
    console.print(f"  {len(ag.ops)} ops, {len(ag.stages)} stages @ {fmt_bytes(budget)} budget", style="dim")
