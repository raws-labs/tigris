"""``tigris export``: a TFLite model rewritten for a smaller TFLite Micro arena."""

from pathlib import Path

import click

from tigris.cli import cli, text
from tigris.frontends.tflite import TFLM_TARGETS
from tigris.frontends.tflite_export import DEFAULT_TARGET
from tigris.utils import fmt_bytes


def _signed(value: int) -> str:
    return ("+" if value > 0 else "-" if value < 0 else "") + fmt_bytes(abs(value))


def _macs(result) -> str:
    return f", {100 * result.extra_macs / result.macs:+.2f}% MACs" if result.macs else ""


def export_line(result) -> list[str]:
    """The memory row analyze prints for a TFLite model: the change of TFLite
    Micro's total arena, its arena head as is and after export, the MACs the
    row bands add, then the kernels and target the figures are for. A declined
    model gets the reason, and the kernels and target when its figures decided."""
    kernels = f", reference kernels on {result.target}"
    if result.data is None:
        decided = kernels if result.arena_before is not None else ""
        return ["TFLite Micro export", "none", text.dim((result.reason or "") + decided)]
    return ["TFLite Micro export", _signed(result.change),
            text.dim(f"head {fmt_bytes(result.arena_before)} -> "
                     f"{fmt_bytes(result.arena_after)}, persistent "
                     f"{_signed(result.persistent_growth)}"
                     + (_macs(result) if result.region is not None else "") + kernels)]


def export_report(result) -> dict:
    """What export found, as analyze --json reports it; byte and MAC counts
    are integers, the head before and the MACs null where the model was not
    planned, the other figures null where it was not written."""
    written, planned = result.data is not None, result.arena_before is not None
    return {
        "arena_head_before_bytes": result.arena_before,
        "arena_head_after_bytes": result.arena_after if written else None,
        "persistent_growth_bytes": result.persistent_growth if written else None,
        "change_bytes": result.change if written else None,
        "region": list(result.region) if written and result.region is not None else None,
        "bands": result.bands if written and result.region is not None else None,
        "pads_folded": result.folded if written else None,
        "macs": result.macs if planned else None,
        "extra_macs": result.extra_macs if written else None,
        "target": result.target,
        "reason": result.reason,
    }


@cli.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None,
              help="Output path (default: <model>_export.tflite)")
@click.option("--target", type=click.Choice(list(TFLM_TARGETS)), default=DEFAULT_TARGET,
              show_default=True,
              help="C ABI of the TFLite Micro build with reference kernels: arm-none-eabi has "
                   "4-byte pointers and enums sized to their values, ilp32 4-byte pointers and "
                   "enums, lp64 8-byte pointers")
def export(model: str, output: str | None, target: str):
    """Rewrite a TFLite model for a smaller TFLite Micro arena.

    Export expects a model that TFLite Micro already runs and does not check
    that TFLite Micro's kernels accept its tensor types. The arena figures are
    for TFLite Micro's reference kernels built for the chosen target.
    """
    from tigris.frontends.tflite import is_tflite
    from tigris.frontends.tflite_export import export as export_model

    path = Path(model)
    data = path.read_bytes()
    if not is_tflite(data):
        raise click.ClickException(f"{path.name} is not a TFLite model")
    result = export_model(data, target)
    count = f"   {result.operators_before} operators" if result.operators_before else ""
    text.echo(text.bold(path.name) + count)
    if result.data is None:
        raise click.ClickException(f"no model written: {result.reason}")
    done = []
    if result.region is not None:
        first, last = result.region
        span = f"operator {first}" if first == last else f"operators {first}-{last}"
        done.append(f"{span} computed in {result.bands} row bands" + _macs(result))
    if result.folded:
        done.append(f"{result.folded} PAD operator{'s' if result.folded != 1 else ''} folded "
                    "into convolutions")
    text.echo("; ".join(done) + f"; {result.operators_after} operators after export")
    rows = [["head", fmt_bytes(result.arena_after),
             text.dim(f"{fmt_bytes(result.arena_before)} before")],
            ["persistent", _signed(result.persistent_growth), ""],
            ["total change", _signed(result.change), ""]]
    text.section(f"TFLite Micro arena, reference kernels on {target}", rows, "<>")
    out = Path(output) if output else path.with_name(path.stem + "_export.tflite")
    out.write_bytes(result.data)
    text.echo()
    text.echo(f"wrote {out}   {fmt_bytes(len(result.data))}")
