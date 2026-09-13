"""``tigris compile`` command."""

from dataclasses import replace
from pathlib import Path

import click

from tigris.cli import (
    cli,
    console,
    _expand_mem,
    _parse_input_shape,
    _parse_size,
    _report_shape_bindings,
    _run_pipeline,
)
from tigris.utils import fmt_bytes


def _run_compressed_pipeline(
    model: str,
    mem: tuple[str, ...],
    input_shapes: dict[str, tuple[int, ...]] | None = None,
):
    """Plan a compressed model against its actual activation capacity.

    Weight blocks depend on stage boundaries, while stage boundaries depend on
    the memory left after reserving those blocks.  Re-plan until the computed
    reservation fits the reservation used for partitioning.  Keeping the
    larger value on a changing layout is conservative: the emitted plan can
    never require more fast memory than the caller supplied.
    """
    from tigris.emitters.binary.writer import compressed_weight_reserve_bytes

    reserve = 0
    # Each additional reservation can only be introduced by a stage layout;
    # this bound also turns an accidental planner oscillation into a clear
    # compiler error instead of an unbounded command.
    max_attempts = 64
    for _ in range(max_attempts):
        # The loop re-plans the same model, so only the final pass reports
        # the bound input dimensions.
        ag, total_budget = _run_pipeline(
            model,
            mem,
            fast_reserve_bytes=reserve,
            input_shapes=input_shapes,
            report_bindings=False,
        )
        required = compressed_weight_reserve_bytes(ag)
        if required <= reserve:
            # ``mem_budget`` is already the reduced activation capacity.  The
            # writer uses this marker to avoid subtracting the same reserve a
            # second time from the serialized plan budget.
            ag.budget = replace(ag.budget, fast_reserve=required)
            return ag, total_budget, reserve, required
        reserve = required

    raise click.ClickException(
        "Compressed-weight reservation did not converge after "
        f"{max_attempts} planning attempts"
    )


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, required=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--output", "-o", default=None, help="Output path (default: <model>.tgrs)")
@click.option("--flash", "-f", default=None, help="Flash size budget - fail if plan exceeds (e.g. 4M)")
@click.option("--compress", "-c", type=click.Choice(["none", "lz4"]), default="none",
              help="Weight compression (default: none)")
@click.option("--xip", is_flag=True, default=False,
              help="Execute-in-place: weights read directly from flash at runtime")
@click.option("--input-shape", "input_shape", multiple=True,
              callback=_parse_input_shape,
              help="Shape to compile an input for (e.g. --input-shape input:1x3x224x224)")
def compile(model: str, mem: tuple[str, ...], output: str | None, flash: str | None,
            compress: str, xip: bool, input_shape: dict[str, tuple[int, ...]]):
    """Compile an ONNX model to binary deployment format."""
    from tigris.analysis.validation import (
        validate_budget,
        validate_execution_dtype,
        validate_operator_support,
    )
    from tigris.emitters.binary.writer import emit_binary_bytes

    compress_arg = compress if compress != "none" else None
    if compress_arg:
        ag, budget, reserved_budget, weight_reserve = _run_compressed_pipeline(
            model, mem, input_shape
        )
        _report_shape_bindings(ag, input_shape)
    else:
        ag, budget = _run_pipeline(model, mem, input_shapes=input_shape)
        reserved_budget = 0
        weight_reserve = 0

    ag.budget = replace(ag.budget, flash=_parse_size(flash) if flash else 0)

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

    result = validate_budget(ag)
    if not result.fast.feasible:
        details = "; ".join(issue.describe() for issue in result.fast.issues)
        raise click.ClickException(
            f"Cannot compile an infeasible memory plan: {details}"
        )
    if not result.slow.fits:
        raise click.ClickException(
            f"Cannot compile this plan: {result.slow.describe()}"
        )

    out = Path(output) if output else Path(model).with_suffix(".tgrs")
    with console.status("Writing binary plan..."):
        plan_data = emit_binary_bytes(ag, compress=compress_arg, xip=xip)
        if ag.budget.flash > 0 and len(plan_data) > ag.budget.flash:
            raise click.ClickException(
                f"Cannot compile a plan that exceeds the flash budget: "
                f"plan is {fmt_bytes(len(plan_data))} but the flash budget is "
                f"{fmt_bytes(ag.budget.flash)}"
            )
        out.write_bytes(plan_data)

    plan_bytes = len(plan_data)

    if compress_arg:
        uncompressed_size = len(emit_binary_bytes(ag))
        ratio = plan_bytes / uncompressed_size if uncompressed_size > 0 else 1.0
        console.print(f"[bold green]Binary plan written to {out}[/] (LZ4 compressed)")
        console.print(
            f"  {len(ag.ops)} ops, {len(ag.stages)} stages @ "
            f"{fmt_bytes(budget)} total fast memory "
            f"({fmt_bytes(ag.mem_budget)} activations + "
            f"{fmt_bytes(weight_reserve)} compressed weights)",
            style="dim",
        )
        if reserved_budget != weight_reserve:
            console.print(
                f"  conservative planning reserve: {fmt_bytes(reserved_budget)}",
                style="dim",
            )
        console.print(f"  plan size: {fmt_bytes(plan_bytes)} (uncompressed: {fmt_bytes(uncompressed_size)}, ratio: {ratio:.2f}x)", style="dim")
    else:
        console.print(f"[bold green]Binary plan written to {out}[/]")
        console.print(f"  {len(ag.ops)} ops, {len(ag.stages)} stages @ {fmt_bytes(budget)} budget", style="dim")
        console.print(f"  plan size: {fmt_bytes(plan_bytes)}", style="dim")
