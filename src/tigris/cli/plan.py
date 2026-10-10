"""``tigris plan`` command."""

from pathlib import Path

import click

from tigris.cli import _expand_mem, _parse_input_shape, _run_pipeline, cli, text
from tigris.utils import fmt_bytes


@cli.command(hidden=True)
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, required=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--output", "-o", default=None, help="Output path (default: <model>.plan.yaml)")
@click.option("--input-shape", "input_shape", multiple=True,
              callback=_parse_input_shape,
              help="Shape to compile an input for (e.g. --input-shape input:1x3x224x224)")
def plan(model: str, mem: tuple[str, ...], output: str | None,
         input_shape: dict[str, tuple[int, ...]]):
    """Deprecated: use 'tigris compile' and 'tigris inspect PLAN --json'. Removed in the next release."""
    click.echo("tigris plan is deprecated: use 'tigris compile' and 'tigris inspect PLAN --json'. "
               "It is removed in the next release. It now writes the plan as JSON, which YAML "
               "parsers read.", err=True)
    from tigris.emitters.plan_json import emit_plan_json

    ag, budget = _run_pipeline(model, mem, input_shapes=input_shape)

    out = Path(output) if output else Path(model).with_suffix(".plan.yaml")
    emit_plan_json(ag, out)
    text.echo(f"wrote {out}   {len(ag.ops)} operators, {len(ag.stages)} stages, "
              f"budget {fmt_bytes(budget)}")
