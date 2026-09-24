"""List and download precompiled models."""

import json
from pathlib import Path

import click

from tigris.cli import _expand_mem, _parse_size, cli
from tigris.zoo import REPOSITORY, Zoo, select, version


def _size(value):
    try:
        result = _parse_size(value)
        if result < 0:
            raise ValueError()
        return result
    except (ValueError, OverflowError):
        raise click.BadParameter(f"invalid nonnegative memory size: {value}") from None


def _filters(function):
    options = [
        click.option("--category"), click.option("--runtime", help="Compatible runtime release, e.g. 0.9.1."),
        click.option("--backend"), click.option("--quantization", "--quant"),
        click.option("-m", "--mem", multiple=True, callback=_expand_mem,
                     help="Maximum fast arena; repeat for slow arena. Omitted pools are unconstrained."),
        click.option("-f", "--flash", help="Maximum compiled plan bytes."),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _matches(config, *, mem, flash, **filters):
    if len(mem) > 2:
        raise click.UsageError("At most two memory pools are supported")
    if filters.get("runtime") is not None:
        version(filters["runtime"])
    pools = [_size(value) for value in mem]
    zoo = Zoo(**config)
    return zoo, select(zoo.artifacts, fast=pools[0] if pools else None,
                       slow=pools[1] if len(pools) > 1 else None,
                       flash=_size(flash) if flash is not None else None, **filters)


@cli.group()
@click.option("--catalog", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Use a local catalog and its adjacent model directories.")
@click.option("--repository", default=REPOSITORY, show_default=True)
@click.option("--revision", default="main", show_default=True, help="HF catalog revision.")
@click.option("--offline", is_flag=True, help="Use cached files without network requests.")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path))
@click.pass_context
def zoo(ctx, **config):
    """Find and download precompiled .tgrs models without an account."""
    ctx.obj = config


@zoo.command("list")
@click.option("--model")
@click.option("--json", "as_json", is_flag=True, help="Print matching catalog entries as JSON.")
@_filters
@click.pass_obj
def list_models(config, as_json, **filters):
    """List matching artifacts, newest first; omit withdrawn builds."""
    try:
        _, matches = _matches(config, **filters)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(matches, indent=2))
    elif not matches:
        click.echo("No matching artifacts.")
    else:
        for item in matches:
            _report(item)


def _report(item):
    memory = item["memory"]
    maximum = item["runtime"]["max"]
    upper = f", <= {maximum}" if maximum is not None else " (no known upper bound)"
    click.echo(f"{item['id']}  model={item['model']}  category={item['category']}")
    click.echo(f"  runtime >= {item['runtime']['min']}{upper}"
               f"  schema={item['schema']}  backends={','.join(item['backends'])}")
    click.echo(f"  tested runtimes: {', '.join(item['tested_runtime_versions']) or 'none recorded'}")
    click.echo(f"  quantization={item['quantization']}  fast={memory['fast_bytes']} B"
               f"  slow={memory['slow_bytes']} B  plan={memory['flash_bytes']} B"
               f"  published={item['published_at']}")
    if item.get("withdrawn"):
        click.echo(f"  WARNING: withdrawn: {item['withdrawn']}", err=True)


@zoo.command()
@click.argument("model", required=False)
@click.option("--artifact", "artifact_id", help="Fetch an exact artifact ID, including withdrawn builds.")
@click.option("-o", "--output", type=click.Path(file_okay=False, path_type=Path),
              help="New destination directory; defaults to the artifact ID.")
@_filters
@click.pass_obj
def fetch(config, model, artifact_id, output, **filters):
    """Download the newest matching artifact and its runtime requirements."""
    if bool(model) == bool(artifact_id):
        raise click.UsageError("Supply either MODEL or --artifact, exclusively")
    try:
        source, matches = _matches(config, model=model, artifact_id=artifact_id, **filters)
        if not matches:
            raise ValueError("No compatible artifact matches the supplied filters")
        chosen = matches[0]
        plan = source.fetch(chosen, output or Path(chosen["id"]))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    _report(chosen)
    click.echo(f"Plan: {plan}")
    click.echo(f"Runtime requirements: {plan.parent / 'download.json'}")
