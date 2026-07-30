"""``tigris simulate`` command."""

import click
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from tigris.cli import cli, console, _run_pipeline
from tigris.utils import fmt_bytes


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K -m 8M)")
def simulate(model: str, mem: tuple[str, ...]):
    """Print a step-by-step execution trace for an ONNX model."""
    ag, budget = _run_pipeline(model, mem)
    _print_simulate(ag, budget)


def _print_simulate(ag, budget: int):
    """Render the execution trace for an analyzed graph."""
    num_stages = len(ag.stages)
    multi = num_stages > 1

    # Header
    parts = [f"{len(ag.ops)} ops"]
    if multi:
        parts.append(f"{num_stages} stages")
    if budget > 0:
        parts.append(f"{fmt_bytes(budget)} budget")
    parts.append(f"{fmt_bytes(ag.peak_memory_bytes)} peak")

    console.print(Panel(
        ", ".join(parts),
        title=f"[bold]TiGrIS Simulate - {ag.model_name}[/]",
        border_style="blue",
    ))

    if not ag.stages:
        return

    for stage in ag.stages:
        _print_stage(ag, stage, budget, multi)


def _print_stage(ag, stage, budget: int, multi: bool):
    """Render one stage block."""
    stage_ops = [ag.ops[i] for i in stage.op_indices]
    first, last = stage.op_indices[0], stage.op_indices[-1]

    # Stage header
    if multi:
        header = f" Stage {stage.stage_id} (ops {first}-{last}) "
        console.print()
        console.print(Rule(header, style="bold"))

        # Peak + tiling/budget status
        info_parts = [f"Peak: {fmt_bytes(stage.peak_bytes)}"]
        tp = stage.tile_plan
        if tp and tp.tileable:
            info_parts.append(
                f"Tiled: {tp.num_tiles} tiles, "
                f"axis {tp.axis}, extent {tp.tile_height} + {tp.halo} halo "
                f"(RF {tp.receptive_field})"
            )
        elif budget > 0 and stage.peak_bytes <= budget:
            info_parts.append("Fits budget")
        console.print("  " + " | ".join(info_parts))

        # Reload inputs (stages > 0)
        if stage.input_tensors:
            console.print()
            console.print("  [bold]Reload inputs:[/]")
            for tname in stage.input_tensors:
                ti = ag.tensors.get(tname)
                if ti:
                    console.print(
                        f"    {tname}  {list(ti.shape)}  {fmt_bytes(ti.size_bytes)}  [dim]<- slow memory[/]"
                    )

    # Op table
    console.print()
    table = Table(border_style="dim", pad_edge=False, box=None)
    table.add_column("Step", justify="right", style="dim", width=5)
    table.add_column("Op", min_width=20, max_width=30, no_wrap=True)
    table.add_column("Type", min_width=10)
    table.add_column("In shape", min_width=14)
    table.add_column("Out shape", min_width=14)
    table.add_column("Live mem", justify="right", min_width=10)

    for op in stage_ops:
        # Primary input shape: first non-constant input
        in_shape = ""
        for inp in op.inputs:
            ti = ag.tensors.get(inp)
            if ti and not ti.is_constant:
                in_shape = str(list(ti.shape))
                break

        # Primary output shape
        out_shape = ""
        if op.outputs:
            ti = ag.tensors.get(op.outputs[0])
            if ti:
                out_shape = str(list(ti.shape))

        # Live memory at this step
        live = ""
        if op.step < len(ag.timeline):
            live = fmt_bytes(ag.timeline[op.step].live_bytes)

        table.add_row(str(op.step), op.name, op.op_type, in_shape, out_shape, live)

    console.print(table)

    # Spill outputs
    if multi and stage.output_tensors:
        console.print()
        console.print("  [bold]Spill outputs:[/]")
        for tname in stage.output_tensors:
            ti = ag.tensors.get(tname)
            if ti:
                console.print(
                    f"    {tname}  {list(ti.shape)}  {fmt_bytes(ti.size_bytes)}  [dim]-> slow memory[/]"
                )
