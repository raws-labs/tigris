"""``tigris compile`` command."""

from dataclasses import replace
from pathlib import Path

import click

from tigris.emitters.binary.writer import RUNTIME_MAX_TENSORS

from tigris.cli import (
    cli,
    text,
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
        try:
            required = compressed_weight_reserve_bytes(ag)
        except ValueError as exc:
            raise click.ClickException(f"Cannot compress the plan: {exc}") from exc
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
    """Compile a model into an execution plan."""
    from tigris.analysis.validation import (
        validate_budget,
        validate_execution_dtype,
        validate_operator_support,
    )
    from tigris.emitters.binary.writer import emit_binary_bytes

    compress_arg = compress if compress != "none" else None
    if compress_arg:
        ag, budget, _, weight_reserve = _run_compressed_pipeline(model, mem, input_shape)
        _report_shape_bindings(ag)
    else:
        ag, budget = _run_pipeline(model, mem, input_shapes=input_shape)
        weight_reserve = 0

    ag.budget = replace(ag.budget, flash=_parse_size(flash) if flash else 0)

    if budget <= 0:
        raise click.ClickException("Fast-memory budget must be greater than zero")
    if budget > 0xFFFFFFFF:
        raise click.ClickException(
            "Fast-memory budget exceeds the uint32 plan-format limit"
        )

    from tigris.analysis.findings import compute_findings
    from tigris.cli.analyze import summary
    from tigris.frontends.tflite import is_tflite, tflm_tensor_arena

    # The same summary analyze prints: what was decided, and why not when it
    # was not. A plan is written only when everything holds.
    data = Path(model).read_bytes()
    findings = compute_findings(ag, flash_budget=ag.budget.flash)
    summary(model, ag, findings, budget, ag.budget.slow, ag.budget.flash,
            tflm_tensor_arena(data) if is_tflite(data) else None)
    usable = (validate_execution_dtype(ag).supported and validate_operator_support(ag).supported)
    result = validate_budget(ag) if usable else None
    if result is None or not result.fast.feasible or not result.slow.fits:
        raise click.ClickException("no plan written")

    out = Path(output) if output else Path(model).with_suffix(".tgrs")
    try:
        plan_data = emit_binary_bytes(ag, compress=compress_arg, xip=xip)
    except ValueError as exc:
        raise click.ClickException(f"no plan written: {exc}") from exc
    if ag.budget.flash > 0 and len(plan_data) > ag.budget.flash:
        raise click.ClickException(
            f"no plan written: the plan is {fmt_bytes(len(plan_data))}, "
            f"the flash budget is {fmt_bytes(ag.budget.flash)}")
    out.write_bytes(plan_data)

    # The count follows from the model, so it is not something to set; what it
    # decides is how the target has to be built. A target built for fewer
    # refuses the plan and says so, which is why this is a note and not a
    # failure.
    runtime_tensors = sum(
        1 for info in ag.tensors.values() if not info.is_constant)
    text.echo()
    if runtime_tensors > RUNTIME_MAX_TENSORS:
        text.echo(text.warn("warning: ") +
                  f"the plan holds {runtime_tensors} tensors, more than the "
                  f"{RUNTIME_MAX_TENSORS} a runtime carries by default; build the "
                  f"runtime with -DTIGRIS_MAX_TENSORS={runtime_tensors}")
    note = ""
    if compress_arg:
        note = (f", LZ4, {fmt_bytes(weight_reserve)} of the {fmt_bytes(budget)} fast memory "
                f"holds decompressed weights")
    text.echo(f"wrote {out}   {fmt_bytes(len(plan_data))}{note}")
