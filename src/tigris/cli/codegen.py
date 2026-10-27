"""``tigris codegen`` command."""

import re
from pathlib import Path

import click

from tigris.capabilities import CODEGEN_BACKENDS
from tigris.cli import cli, text
from tigris.utils import fmt_bytes


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
    """Generate C deployment code for a plan."""
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.codegen import generate_c

    plan_path = Path(plan)

    if output_format != "core":
        if header is not None:
            raise click.UsageError("--header requires --format core")
        if core_name != "tigris_codegen":
            raise click.UsageError("--name requires --format core")

    out = Path(output) if output else plan_path.with_suffix(".c")
    header_path = Path(header) if header else out.with_suffix(".h")
    try:
        parsed = read_binary_plan(plan_path.read_bytes(), decompress_weights=False)
        c_source = generate_c(parsed, backend, output_format, header_path.name, core_name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    from tigris.emitters.codegen import generate_kernel_config
    kernels_path = out.with_name(out.stem + "_kernels.h")
    guard = re.sub(r"[^A-Za-z0-9_]", "_", kernels_path.name).upper()
    if guard[0].isdigit():
        guard = "_" + guard
    out.write_text(c_source)
    kernels_path.write_text(generate_kernel_config(parsed, guard))
    if output_format == "core":
        from tigris.emitters.codegen import generate_core_header
        header_path.write_text(generate_core_header(parsed, core_name))

    details = f"{output_format}, {backend} backend, {fmt_bytes(len(c_source))}, kernels {kernels_path}"
    if output_format == "core":
        details += f", header {header_path}, symbols {core_name}_*"
    text.echo(f"wrote {out}   {details}")
