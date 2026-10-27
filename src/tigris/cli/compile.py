"""``tigris compile`` command."""

from dataclasses import replace
from pathlib import Path

import click

from tigris.cli import _memory_sizes, _parse_input_shape, _planned, cli, text
from tigris.pipeline import parse_size
from tigris.utils import fmt_bytes


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, required=True, callback=_memory_sizes,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--output", "-o", default=None, help="Output path (default: <model>.tgrs)")
@click.option("--flash", "-f", default=None, callback=_memory_sizes,
              help="Flash size budget; no plan is written when the plan is larger (e.g. 4M)")
@click.option("--compress", "-c", type=click.Choice(["none", "lz4"]), default="none",
              help="Weight compression (default: none)")
@click.option("--xip", is_flag=True, default=False,
              help="Execute-in-place: weights read directly from flash at runtime")
@click.option("--input-shape", "input_shape", multiple=True,
              callback=_parse_input_shape,
              help="Shape to compile an input for (e.g. --input-shape input:1x3x224x224)")
@click.option("--keep-names", is_flag=True,
              help="Keep operator and tensor names in the plan for inspect")
def compile(model: str, mem: tuple[str, ...], output: str | None, flash: str | None,
            compress: str, xip: bool, input_shape: dict[str, tuple[int, ...]], keep_names: bool):
    """Compile a model into an execution plan."""
    from tigris.emitters.binary.writer import emit_binary_bytes

    compress_arg = compress if compress != "none" else None
    ag = _planned(model, mem, compress=compress_arg, input_shapes=input_shape)
    budget = parse_size(mem[0])

    ag.budget = replace(ag.budget, flash=parse_size(flash) if flash else 0)

    from tigris.analysis.findings import compute_findings
    from tigris.cli.analyze import summary
    from tigris.frontends.tflite_export import tflm_tensor_arena

    # The same summary analyze prints: what was decided, and why not when it
    # was not. A plan is written only when everything holds.
    data = Path(model).read_bytes()
    findings = compute_findings(ag, names=keep_names)
    summary(model, ag, findings, budget, tflm_tensor_arena(data))
    if not findings.gate.passes:
        raise click.ClickException("no plan written")

    out = Path(output) if output else Path(model).with_suffix(".tgrs")
    # Findings measured the plan without compression or XIP; an empty
    # measurement leaves the writer to give its error.
    plan_data = findings.plan if compress_arg is None and not xip else b""
    if not plan_data:
        try:
            plan_data = emit_binary_bytes(ag, compress=compress_arg, xip=xip, names=keep_names)
        except ValueError as exc:
            raise click.ClickException(f"no plan written: {exc}") from exc
    if ag.budget.flash > 0 and len(plan_data) > ag.budget.flash:
        raise click.ClickException(
            f"no plan written: the plan is {fmt_bytes(len(plan_data))}, "
            f"the flash budget is {fmt_bytes(ag.budget.flash)}")
    from tigris.runtime import RuntimeError as HostRuntimeError, check_plan

    try:
        check_plan(plan_data)
    except HostRuntimeError as exc:
        raise click.ClickException(f"no plan written: {exc}") from exc
    out.write_bytes(plan_data)

    text.echo()
    note = ""
    if compress_arg:
        note = (f", LZ4, {fmt_bytes(ag.fast_memory_reserve_bytes)} of the {fmt_bytes(budget)} "
                "fast memory holds decompressed weights")
    text.echo(f"wrote {out}   {fmt_bytes(len(plan_data))}{note}")
