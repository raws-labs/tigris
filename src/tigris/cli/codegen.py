"""``tigris codegen`` command."""

from pathlib import Path

import click

from tigris.capabilities import CODEGEN_BACKENDS
from tigris.cli import cli, console


@cli.command()
@click.argument("plan", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(CODEGEN_BACKENDS),
              default="reference",
              help="Target/kernel backend; acceleration is int8-only (default: reference)")
@click.option("--output", "-o", default=None, help="Output C file path")
def codegen(plan: str, backend: str, output: str | None):
    """Generate a C inference harness from a compiled .tgrs plan."""
    from tigris.emitters.codegen import generate_c

    plan_path = Path(plan)
    plan_data = plan_path.read_bytes()

    try:
        with console.status("Generating C source..."):
            c_source = generate_c(plan_data, backend)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out = Path(output) if output else plan_path.with_suffix(".c")
    out.write_text(c_source)

    console.print(f"[bold green]C source written to {out}[/]")
    console.print(f"  backend: {backend}", style="dim")
    console.print(f"  size: {len(c_source)} bytes", style="dim")
