"""``tigris simulate``, deprecated, and the trace ``tigris analyze --trace`` prints."""

import click

from tigris.cli import _expand_mem, _parse_input_shape, _run_pipeline, cli, text
from tigris.utils import fmt_bytes, source_shape


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
    click.echo("tigris simulate is deprecated: use 'tigris analyze MODEL --trace'. "
               "It is removed in the next release.", err=True)
    ag, budget = _run_pipeline(model, mem, input_shapes=input_shape)
    print_trace(ag, budget)


def _shape(ag, name: str) -> str:
    info = ag.tensors.get(name)
    return "x".join(str(d) for d in source_shape(ag, info.shape)) if info else ""


def print_trace(ag, budget: int) -> None:
    """Each stage's reloads, operators with their live memory, and spills."""
    parts = [f"{len(ag.ops)} operator{'s' if len(ag.ops) != 1 else ''}",
             f"{len(ag.stages)} stage{'s' if len(ag.stages) != 1 else ''}"]
    if budget > 0:
        parts.append(f"budget {fmt_bytes(budget)}")
    parts.append(f"unscheduled peak {fmt_bytes(ag.peak_memory_bytes)}")
    text.gap()
    text.echo(text.bold(ag.model_name) + "   " + ", ".join(parts))
    for stage in ag.stages:
        _print_stage(ag, stage, budget, len(ag.stages) > 1)


def _print_stage(ag, stage, budget: int, multi: bool) -> None:
    if multi:
        first, last = stage.op_indices[0], stage.op_indices[-1]
        notes = [f"peak {fmt_bytes(stage.peak_bytes)}"]
        tp = stage.tile_plan
        if tp and tp.tileable:
            notes.append(f"{tp.num_tiles} tiles, axis {tp.axis}, extent {tp.tile_height} "
                         f"+ {tp.halo} halo")
        elif budget > 0 and stage.peak_bytes <= budget:
            notes.append("fits the budget")
        text.gap()
        text.echo(text.bold(f"stage {stage.stage_id}") + f"   operators {first}-{last}, " +
                  ", ".join(notes))
        for name in stage.input_tensors:
            info = ag.tensors.get(name)
            if info:
                text.echo(f"  reload {name}  {_shape(ag, name)}  {fmt_bytes(info.size_bytes)}")
    rows = [[text.dim(cell) for cell in ("step", "operator", "type", "in", "out", "live")]]
    for index in stage.op_indices:
        op = ag.ops[index]
        source = next((n for n in op.inputs if n in ag.tensors and not ag.tensors[n].is_constant), "")
        live = fmt_bytes(ag.timeline[op.step].live_bytes) if op.step < len(ag.timeline) else ""
        rows.append([str(op.step), op.name, op.op_type, _shape(ag, source) if source else "",
                     _shape(ag, op.outputs[0]) if op.outputs else "", live])
    for line in text.columns(rows, ">    >"):
        text.echo(line)
    if multi:
        for name in stage.output_tensors:
            info = ag.tensors.get(name)
            if info:
                text.echo(f"  spill  {name}  {_shape(ag, name)}  {fmt_bytes(info.size_bytes)}")
