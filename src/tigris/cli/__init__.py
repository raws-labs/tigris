"""CLI entry point: ``tigris analyze model.onnx --mem 256K``."""

import os

import click

from tigris.pipeline import parse_size


def _memory_sizes(ctx, param, value):
    """Check the sizes of ``-m`` and ``-f``, which stay text.

    ``-m 256K+4M`` is sugar for repeated ``-m`` and expands into separate pools.
    """
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [value]
    else:
        tokens = []
        for token in value:
            for part in token.split("+"):
                if not part.strip():
                    raise click.BadParameter(
                        f"invalid memory budget {token!r}: empty memory pool"
                    )
                tokens.append(part.strip())
        if len(tokens) > 2:
            raise click.BadParameter(
                f"{len(tokens)} memory pools given; the planner supports a fast "
                "and a slow pool"
            )
    for token in tokens:
        try:
            parse_size(token)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from None
    return value if isinstance(value, str) else tuple(tokens)


def _parse_input_shape(ctx, param, value: tuple[str, ...]):
    """Parse ``--input-shape name:1x3x224x224`` into a name to shape mapping."""
    shapes: dict[str, tuple[int, ...]] = {}
    for token in value:
        name, sep, extents = token.rpartition(":")
        if not sep or not name:
            raise click.BadParameter(
                f"invalid input shape {token!r}: expected name:1x3x224x224"
            )
        try:
            dims = tuple(int(part) for part in extents.split("x"))
        except ValueError:
            raise click.BadParameter(
                f"invalid input shape {token!r}: dimensions must be integers"
            ) from None
        if not dims or any(dim <= 0 for dim in dims):
            raise click.BadParameter(
                f"invalid input shape {token!r}: dimensions must be positive"
            )
        shapes[name] = dims
    return shapes


def _report_shape_bindings(ag) -> None:
    """Warn about a dimension the compiler picked itself.

    The plan is sized for a guess, which the interface rows cannot show: they
    state the shape without saying where it came from. A shape the caller named
    needs no warning, and the rows already report it.
    """
    for binding in ag.shape_bindings:
        from tigris.cli import text
        text.echo(text.warn("warning: ") + str(binding), err=True)


def _planned(model: str, mem: tuple[str, ...], *, compress: str | None = None,
             input_shapes: dict[str, tuple[int, ...]] | None = None,
             report_bindings: bool = True):
    """Plan a model as ``pipeline.plan_model`` does, for a command: dimensions
    the compiler bound are reported once, an error is a ClickException."""
    from tigris import pipeline

    try:
        if compress is None:
            ag = pipeline.load(model, input_shapes)
            if report_bindings:
                _report_shape_bindings(ag)
            return pipeline.plan(ag, mem)
        ag = pipeline.plan_model(model, mem, compress=compress, input_shapes=input_shapes)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if report_bindings:
        _report_shape_bindings(ag)
    return ag


def _show_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    from importlib.metadata import version
    from tigris.runtime import runtime_info

    try:
        info = runtime_info()
        runtime = f"{info['version']}; {info['source']}"
    except ValueError as exc:
        if "TIGRIS_HOST_LIBRARY" in os.environ:
            raise click.ClickException(str(exc)) from exc
        runtime = "unavailable"
    click.echo(f"tigris, version {version('tigris-ml')} (runtime {runtime})")
    ctx.exit()


@click.group()
@click.option("--version", is_flag=True, is_eager=True, expose_value=False,
              callback=_show_version, help="Show compiler and host runtime versions and runtime origin.")
def cli():
    """TiGrIS: Tiled Graph Inference Scheduler

    Fits ONNX and TFLite models into the memory of embedded devices: analyze a
    model against a budget, compile it into a tiled execution plan, generate
    the C code that runs it.
    """


def main():
    cli()


# Register subcommands
from tigris.cli.analyze import analyze  # noqa: E402, F401
from tigris.cli.codegen import codegen  # noqa: E402, F401
from tigris.cli.compile import compile  # noqa: E402, F401
from tigris.cli.export import export  # noqa: E402, F401
from tigris.cli.inspect import inspect  # noqa: E402, F401
from tigris.cli.run import run  # noqa: E402, F401
from tigris.cli.zoo import zoo  # noqa: E402, F401
