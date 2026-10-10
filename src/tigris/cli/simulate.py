"""``tigris simulate``, deprecated in favor of ``tigris analyze --trace``."""

import click

from tigris.cli import _expand_mem, _parse_input_shape, _run_pipeline, cli


@cli.command(hidden=True)
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--input-shape", "input_shape", multiple=True,
              callback=_parse_input_shape,
              help="Shape to compile an input for (e.g. --input-shape input:1x3x224x224)")
def simulate(model: str, mem: tuple[str, ...],
             input_shape: dict[str, tuple[int, ...]]):
    """Deprecated: use 'tigris analyze MODEL --trace'. Removed in the next release."""
    from tigris.cli.trace import trace_model

    click.echo("tigris simulate is deprecated: use 'tigris analyze MODEL --trace'. "
               "It is removed in the next release.", err=True)
    ag, _ = _run_pipeline(model, mem[:1], input_shapes=input_shape)
    trace_model(model, ag, (), verbose=False)
