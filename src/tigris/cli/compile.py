"""``tigris compile`` command."""

from pathlib import Path

import click

from tigris.cli import cli, console, _parse_size, _run_pipeline
from tigris.utils import fmt_bytes


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, required=True, help="Memory pool size, fast to slow (e.g. 256K)")
@click.option("--output", "-o", default=None, help="Output path (default: <model>.tgrs)")
@click.option("--flash", "-f", default=None, help="Flash size - warn if plan exceeds (e.g. 4M)")
@click.option("--compress", "-c", type=click.Choice(["none", "lz4"]), default="none",
              help="Weight compression (default: none)")
@click.option("--xip", is_flag=True, default=False,
              help="Execute-in-place: weights read directly from flash at runtime")
def compile(model: str, mem: tuple[str, ...], output: str | None, flash: str | None, compress: str, xip: bool):
    """Compile an ONNX model to binary deployment format."""
    from tigris.analysis.validation import (
        validate_execution_dtype,
        validate_memory_plan,
        validate_operator_support,
    )
    from tigris.emitters.binary.writer import emit_binary

    ag, budget = _run_pipeline(model, mem)
    if budget <= 0:
        raise click.ClickException("Fast-memory budget must be greater than zero")
    if budget > 0xFFFFFFFF:
        raise click.ClickException(
            "Fast-memory budget exceeds the uint32 plan-format limit"
        )

    dtype_validation = validate_execution_dtype(ag)
    if not dtype_validation.supported:
        raise click.ClickException(
            "Cannot compile a plan with unsupported tensor dtypes: "
            + dtype_validation.describe()
        )

    operator_validation = validate_operator_support(ag)
    if not operator_validation.supported:
        raise click.ClickException(
            "Cannot compile a plan with unsupported operators: "
            + operator_validation.describe()
        )

    validation = validate_memory_plan(ag)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise click.ClickException(f"Cannot compile an infeasible memory plan: {details}")

    compress_arg = compress if compress != "none" else None
    out = Path(output) if output else Path(model).with_suffix(".tgrs")
    with console.status("Writing binary plan..."):
        emit_binary(ag, out, compress=compress_arg, xip=xip)

    plan_bytes = out.stat().st_size

    if compress_arg:
        from tigris.emitters.binary.writer import emit_binary_bytes
        uncompressed_size = len(emit_binary_bytes(ag))
        ratio = plan_bytes / uncompressed_size if uncompressed_size > 0 else 1.0
        console.print(f"[bold green]Binary plan written to {out}[/] (LZ4 compressed)")
        console.print(f"  {len(ag.ops)} ops, {len(ag.stages)} stages @ {fmt_bytes(budget)} budget", style="dim")
        console.print(f"  plan size: {fmt_bytes(plan_bytes)} (uncompressed: {fmt_bytes(uncompressed_size)}, ratio: {ratio:.2f}x)", style="dim")
    else:
        console.print(f"[bold green]Binary plan written to {out}[/]")
        console.print(f"  {len(ag.ops)} ops, {len(ag.stages)} stages @ {fmt_bytes(budget)} budget", style="dim")
        console.print(f"  plan size: {fmt_bytes(plan_bytes)}", style="dim")

    if flash:
        flash_bytes = _parse_size(flash)
        if plan_bytes <= flash_bytes:
            console.print(f"  flash {fmt_bytes(flash_bytes)}: [green]fits[/]", style="dim")
        else:
            ratio = plan_bytes / flash_bytes
            console.print(
                f"  flash {fmt_bytes(flash_bytes)}: [red]does not fit[/] ({ratio:.1f}x)",
                style="dim",
            )
