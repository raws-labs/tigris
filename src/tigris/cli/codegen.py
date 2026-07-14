"""``tigris codegen`` command."""

from pathlib import Path

import click

from tigris.capabilities import CODEGEN_BACKENDS
from tigris.cli import cli, console


@cli.command()
@click.argument("plan", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(CODEGEN_BACKENDS),
              default="reference",
              help="Kernel backend; acceleration is int8-only (default: reference)")
@click.option("--output", "-o", default=None, help="Output C file path")
@click.option("--format", "output_format", type=click.Choice(["app", "core"]),
              default="app", show_default=True,
              help="app is standalone; core is embedded deployment glue")
@click.option("--header", "header", default=None,
              help="Header path for --format core (defaults next to --output)")
@click.option("--name", "core_name", default="tigris_codegen",
              help="C symbol prefix for --format core (default: tigris_codegen)")
def codegen(plan: str, backend: str, output: str | None,
            output_format: str, header: str | None, core_name: str):
    """Generate C deployment code from a compiled .tgrs plan."""
    from tigris.emitters.codegen import generate_c

    plan_path = Path(plan)
    plan_data = plan_path.read_bytes()

    if output_format != "core":
        if header is not None:
            raise click.UsageError("--header requires --format core")
        if core_name != "tigris_codegen":
            raise click.UsageError("--name requires --format core")

    out = Path(output) if output else plan_path.with_suffix(".c")
    header_path = Path(header) if header else out.with_suffix(".h")
    try:
        with console.status("Generating C source..."):
            c_source = generate_c(
                plan_data, backend, output_format,
                header_path.name, core_name,
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out.write_text(c_source)
    if output_format == "core":
        from tigris.emitters.codegen import generate_core_header
        header_path.write_text(generate_core_header(plan_data, core_name))

    console.print(f"[bold green]C source written to {out}[/]")
    console.print(f"  backend: {backend}", style="dim")
    console.print(f"  format: {output_format}", style="dim")
    if output_format == "core":
        console.print(f"  header: {header_path}", style="dim")
        console.print(f"  name: {core_name}", style="dim")
    console.print(f"  size: {len(c_source)} bytes", style="dim")
