"""``tigris export``: a TFLite model or an ExecuTorch program rewritten to run in less RAM."""

import hashlib
import json
import math
from pathlib import Path

import click
from click.core import ParameterSource

from tigris.cli import cli, text
from tigris.frontends.tflite_export import DEFAULT_TARGET
from tigris.frontends.tflm import TFLM_TARGETS
from tigris.pipeline import parse_size
from tigris.utils import fmt_bytes, plural

# Exit status when a stated limit is not met: --arena, --method, --require-gain.
LIMIT_NOT_MET = 3

CPU_HELP = "ExecuTorch: the firmware's Cortex-M core, M0 to M85; read from the program by default"
OVERWRITE_HELP = "ExecuTorch: reuse the model input's memory once it has been read"
MACS_HELP = "ExecuTorch: limit on recomputed MACs, e.g. 5%"


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


def pte_no_gain(max_extra_macs: float | None, arena: str | None = None,
                method: str | None = None) -> str:
    """Why export keeps the program as it is, naming the limits that were given."""
    limits = [f"{option} {size}" for option, size in (("--arena", arena), ("--method", method))
              if size]
    if max_extra_macs is not None:
        limits.append(f"--max-extra-macs {100 * max_extra_macs:g}%")
    return "no smaller total found" + (" within " + " and ".join(limits) if limits else "")


def pte_reason(result, arena: str | None, method: str | None,
               max_extra_macs: float | None) -> str | None:
    """Why export writes no rewrite, as the command prints it: the refusal,
    each limit no program meets, or that no smaller total was found; None
    for a written program."""
    if result.refused:
        return result.reason
    if result.written:
        return None
    if result.unmet:
        return "; ".join(_unmet_lines(result, arena, method))
    return pte_no_gain(max_extra_macs, arena, method)


def pte_change(before: int, after: int) -> str:
    """The ratio when `after` is smaller, else the signed change."""
    return f"{before / after:.2f}x less" if 0 < after < before else _signed(after - before)


def pte_percent(part: int, whole: int | None) -> str:
    return f"{100 * part / whole:+.2f}%" if whole else "+0.00%"


def _after(result) -> dict:
    """The figures of the program export writes: the rewrite, or the input
    unchanged."""
    if result.written:
        return {"planned": result.planned_after, "method": result.method_after,
                "temp": result.temp_after, "total": result.total_after}
    return {"planned": result.planned_before, "method": result.method_before,
            "temp": result.temp_before, "total": result.total_before}


def pte_title(name: str, result, cpu: str | None) -> str:
    """The first line for an ExecuTorch program: its instructions and the
    CMSIS-NN kernels it was lowered for."""
    from tigris.frontends import pte_cortex_m as cm

    checked = f", checked against --cpu {cpu}" if cpu else ""
    return (text.bold(name) + f"   ExecuTorch program, "
            f"{plural(result.instructions_before, 'instruction')}, "
            f"CMSIS-NN {cm.describe(result.classes)}{checked}")


def pte_pools(planned: int, method: int, temp: int) -> str:
    return f"model memory {fmt_bytes(planned)}, method {fmt_bytes(method)}, temp {fmt_bytes(temp)}"


def pte_report(result, cpu: str | None, data: bytes, reason: str | None) -> dict:
    """What export found for the ExecuTorch program `data`, as --json writes it,
    with `reason` as pte_reason gives it; the after figures are those of the
    program export writes, null for a refused program."""
    from tigris import __version__

    refused = result.refused
    after = None if refused else _after(result)

    def pair(before, key: str) -> dict:
        return {"before": before, "after": None if after is None else after[key]}

    planned = None if result.planned_before is None else sum(result.planned_before)
    return {
        "runtime": "executorch",
        "source": "program",
        "tigris_version": __version__,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "output_sha256": None if refused else hashlib.sha256(result.data).hexdigest(),
        "written": result.written,
        "reason": reason,
        "warnings": list(result.warnings),
        "cmsis_class": list(result.classes) or None,
        "cpu": cpu,
        "model_memory_bytes": {"before": planned,
                               "after": None if after is None else sum(after["planned"])},
        "method_allocator_bytes": pair(result.method_before, "method"),
        "temp_allocator_bytes": pair(result.temp_before, "temp"),
        "total_bytes": pair(result.total_before, "total"),
        "macs": result.macs,
        "extra_macs": None if refused else result.extra_macs,
        "pte_bytes": {"before": result.pte_bytes_before,
                      "after": None if refused else len(result.data)},
        "regions": [{"first": r.first, "last": r.last, "bands": r.bands, "in_place": r.in_place}
                    for r in result.regions],
        "input_overwritten": {
            "before": result.input_overwritten_before,
            "after": None if refused else (result.input_overwritten_after if result.written
                                           else result.input_overwritten_before)},
        "instruction_map": None if refused else [{"original": j, "band": band}
                                                 for j, band in result.origin],
    }


def pte_header(result, out: Path) -> str:
    """The C header --header writes: the pool sizes of the program at `out`."""
    from tigris import __version__

    after = _after(result)
    digest = hashlib.sha256(result.data).hexdigest()[:16]
    lines = [f"/* Pool sizes for {out.name}, from tigris export {__version__} "
             f"(sha256 {digest}). */"]
    lines += [f"#define TIGRIS_ET_PLANNED_BUFFER_{i}_BYTES {size}u"
              for i, size in enumerate(after["planned"]) if i >= 1]
    lines.append(f"#define TIGRIS_ET_METHOD_ALLOCATOR_MIN_BYTES {after['method']}u")
    lines.append(f"#define TIGRIS_ET_TEMP_ALLOCATOR_MIN_BYTES {after['temp']}u")
    lines += [f"#define TIGRIS_ET_INPUT_{k}_OFFSET {offset}u"
              for k, offset in enumerate(result.input_offsets)]
    lines += [f"#define TIGRIS_ET_OUTPUT_{k}_OFFSET {offset}u"
              for k, offset in enumerate(result.output_offsets)]
    return "\n".join(lines) + "\n"


def pte_warnings(result) -> None:
    for warning in result.warnings:
        text.echo(text.warn("warning: ") + warning, err=True)


def _ram(result) -> None:
    """The ExecuTorch RAM section: each pool after export and before."""
    planned_before, planned_after = sum(result.planned_before), sum(result.planned_after)

    def row(name: str, after: int, before: int, note: str) -> list[str]:
        return [name, fmt_bytes(after), text.dim(f"{fmt_bytes(before)} before"), text.dim(note)]

    rows = [
        row("model memory", planned_after, planned_before,
            pte_change(planned_before, planned_after) + "; activations and CMSIS-NN scratch"),
        row("method allocator", result.method_after, result.method_before,
            "ExecuTorch's records for the method"),
        row("temp allocator", result.temp_after, result.temp_before, "needed to load the method"),
        row("total", result.total_after, result.total_before,
            pte_change(result.total_before, result.total_after)),
    ]
    text.section("ExecuTorch RAM on Cortex-M, from the program", rows, "<>>")
    text.echo(text.dim("  plus the runner's own input and output buffers, stack and statics"))


def _input_line(result) -> str:
    if not result.input_overwritten_after:
        return "model input untouched"
    if result.input_overwritten_before:
        return "model input memory reused once read, as in the original"
    return "model input memory reused once read (--allow-input-overwrite)"


def _unmet_lines(result, arena: str | None, method: str | None) -> list[str]:
    lines = []
    for limit, smallest, total in result.unmet:
        if limit == "arena":
            lines.append(f"smallest model memory found is {fmt_bytes(smallest)}, above --arena "
                         f"{arena} (total {fmt_bytes(total)})")
        elif limit == "method":
            lines.append(f"smallest method allocator found is {fmt_bytes(smallest)}, above "
                         f"--method {method} (total {fmt_bytes(total)})")
        else:
            lines.append(f"no program found meets --arena {arena} and --method {method} together")
    return lines


def _write_json(path: str | None, report: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
                              + "\n")


def _size(ctx, param, value):
    if value is None:
        return None
    try:
        parse_size(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from None
    return value


def _cpu(ctx, param, value):
    if value is None:
        return None
    from tigris.frontends.pte_cortex_m import cpu_class

    try:
        cpu_class(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from None
    return value


def _share(ctx, param, value):
    """A share from "5%" or "5"."""
    if value is None:
        return None
    try:
        share = float(value.strip().removesuffix("%")) / 100
    except ValueError:
        share = -1.0
    if not (share >= 0 and math.isfinite(share)):
        raise click.BadParameter(f"invalid percentage {value!r}: expected a number of 0 or more "
                                 f"with an optional %, for example 5%")
    return share


cpu_option = click.option("--cpu", default=None, callback=_cpu, help=CPU_HELP)
overwrite_option = click.option("--allow-input-overwrite", is_flag=True, help=OVERWRITE_HELP)
macs_option = click.option("--max-extra-macs", "max_extra_macs", metavar="PERCENT",
                           default=None, callback=_share, help=MACS_HELP)


def refuse_options(ctx: click.Context, options: dict, applies_to: str) -> None:
    """A usage error for the first of `options` (parameter name -> option)
    given on the command line."""
    for name, option in options.items():
        if ctx.get_parameter_source(name) not in (ParameterSource.DEFAULT, None):
            raise click.UsageError(f"{option} applies to {applies_to}")


_TFLITE_OPTIONS = {"target": "--target"}
_PTE_OPTIONS = {"cpu": "--cpu", "allow_input_overwrite": "--allow-input-overwrite",
                "arena": "--arena", "method": "--method", "max_extra_macs": "--max-extra-macs",
                "header": "--header"}


@cli.command(short_help="Rewrite a TFLite model or an ExecuTorch program to run in less RAM.")
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None,
              help="Output path (default: <model>_export.tflite or <model>_export.pte)")
@click.option("--target", type=click.Choice(list(TFLM_TARGETS)), default=DEFAULT_TARGET,
              show_default=True,
              help="TFLite: C ABI of the TFLite Micro build with reference kernels: "
                   "arm-none-eabi has 4-byte pointers and enums sized to their values, ilp32 "
                   "4-byte pointers and enums, lp64 8-byte pointers")
@cpu_option
@overwrite_option
@click.option("--arena", metavar="SIZE", default=None, callback=_size,
              help="ExecuTorch: planned buffer limit, e.g. 64K")
@click.option("--method", metavar="SIZE", default=None, callback=_size,
              help="ExecuTorch: method allocator limit, e.g. 32K")
@macs_option
@click.option("--require-gain", is_flag=True, help="Exit 3 when the rewrite saves nothing")
@click.option("--json", "json_path", metavar="PATH", default=None,
              help="Write the report as JSON")
@click.option("--header", metavar="PATH", default=None,
              help="ExecuTorch: write the pool sizes as a C header")
def export(model: str, output: str | None, target: str, cpu: str | None,
           allow_input_overwrite: bool, arena: str | None, method: str | None,
           max_extra_macs: float | None, require_gain: bool, json_path: str | None,
           header: str | None):
    """Rewrite a TFLite model or an ExecuTorch program to run in less RAM.

    Runs of operators are split into row bands, so fewer activations are held
    at once. A TFLite model keeps running on TFLite Micro; an ExecuTorch
    program lowered for Cortex-M keeps running on the same ExecuTorch build,
    with smaller memory pools.
    """
    from tigris.frontends.pte_schema import is_pte
    from tigris.frontends.tflite import is_tflite

    ctx = click.get_current_context()
    path = Path(model)
    data = path.read_bytes()
    if is_pte(data):
        refuse_options(ctx, _TFLITE_OPTIONS, "TFLite models")
        status = _export_pte(path, data, output, cpu, allow_input_overwrite, arena, method,
                             max_extra_macs, require_gain, json_path, header)
    elif is_tflite(data):
        refuse_options(ctx, _PTE_OPTIONS, "ExecuTorch programs")
        status = _export_tflite(path, data, output, target, require_gain, json_path)
    else:
        raise click.ClickException(f"{path.name} is neither a TFLite model nor an ExecuTorch "
                                   f"program")
    if status:
        ctx.exit(status)


def _export_tflite(path: Path, data: bytes, output: str | None, target: str, require_gain: bool,
                   json_path: str | None) -> int:
    from tigris.frontends.tflite_export import export as export_model

    result = export_model(data, target)
    count = f"   {result.operators_before} operators" if result.operators_before else ""
    text.echo(text.bold(path.name) + count)
    if result.arena_before is None:
        raise click.ClickException(f"no model written: {result.reason}")
    _write_json(json_path, {"runtime": "tflite-micro", "source": "model",
                            **export_report(result)})
    out = Path(output) if output else path.with_name(path.stem + "_export.tflite")
    if result.data is None:
        out.write_bytes(data)
        text.echo(f"no smaller arena found; {out} is an unchanged copy")
        return LIMIT_NOT_MET if require_gain else 0
    done = []
    if result.region is not None:
        first, last = result.region
        span = f"operator {first}" if first == last else f"operators {first}-{last}"
        done.append(f"{span} computed in {result.bands} row bands" + _macs(result))
    if result.folded:
        done.append(f"{plural(result.folded, 'PAD operator')} folded "
                    "into convolutions")
    text.echo("; ".join(done) + f"; {result.operators_after} operators after export")
    rows = [["head", fmt_bytes(result.arena_after),
             text.dim(f"{fmt_bytes(result.arena_before)} before")],
            ["persistent", _signed(result.persistent_growth), ""],
            ["total change", _signed(result.change), ""]]
    text.section(f"TFLite Micro arena, reference kernels on {target}", rows, "<>")
    out.write_bytes(result.data)
    text.echo()
    text.echo(f"wrote {out}   {fmt_bytes(len(result.data))}")
    return 0


def _export_pte(path: Path, data: bytes, output: str | None, cpu: str | None,
                allow_input_overwrite: bool, arena: str | None, method: str | None,
                max_extra_macs: float | None, require_gain: bool, json_path: str | None,
                header: str | None) -> int:
    from tigris.frontends.pte_export import export as export_program

    result = export_program(data, cpu=cpu, allow_input_overwrite=allow_input_overwrite,
                            arena=parse_size(arena) if arena else None,
                            method=parse_size(method) if method else None,
                            max_extra_macs=max_extra_macs)
    pte_warnings(result)
    if result.refused:
        raise click.ClickException(f"{path.name} {result.reason}")
    reason = pte_reason(result, arena, method, max_extra_macs)
    _write_json(json_path, pte_report(result, cpu, data, reason))
    out = Path(output) if output else path.with_name(path.stem + "_export.pte")
    out.write_bytes(result.data)
    if header:
        Path(header).write_text(pte_header(result, out))
    text.echo(pte_title(path.name, result, cpu))
    if not result.written:
        if result.unmet:
            for line in _unmet_lines(result, arena, method):
                text.echo(line)
            text.echo(f"{out} is an unchanged copy")
            return LIMIT_NOT_MET
        text.echo(f"{reason}; {out} is an unchanged copy")
        return LIMIT_NOT_MET if require_gain else 0
    rows = []
    for region in result.regions:
        span = (f"instruction {region.first}" if region.first == region.last
                else f"instructions {region.first}-{region.last}")
        rows.append([span, f"{region.bands} row bands"
                     + (", output in place" if region.in_place else "")])
    rows.append(["extra MACs", pte_percent(result.extra_macs, result.macs)])
    rows.append(["instructions after", str(result.instructions_after)])
    text.section("rewrite", rows)
    _ram(result)
    text.gap()
    text.echo(_input_line(result))
    text.gap()
    text.echo(f"wrote {out}   {fmt_bytes(len(result.data))}   "
              f"{_signed(len(result.data) - len(data))}")
    return 0
