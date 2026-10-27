"""``tigris analyze``: does a model fit a budget, and if not, why."""

import json
from dataclasses import replace
from pathlib import Path

import click

from tigris import TILE_AXIS_HW
from tigris.cli import _memory_sizes, _parse_input_shape, _planned, cli, text
from tigris.pipeline import parse_size
from tigris.utils import describe_interface, fmt_bytes, plural, source_shape

# Budgets tried beyond the one given, halving below a fit or doubling above a failure.
_BUDGETS_TRIED = 3


def _report(ag, findings, budget: int, source: str, tflm_arena: int | None, tried: list,
            traffic, tflm_export=None) -> dict:
    """The analysis as versioned JSON; byte counts are integers."""
    from tigris.cli.export import export_report
    f = findings
    return {
        "report": "tigris-analysis",
        "version": 1,
        "model": {
            "name": ag.model_name,
            "format": source,
            "dtype": "int8" if f.is_quantized else "float32" if f.is_float32 else None,
            "operators": len(ag.ops),
            "tensors": len(ag.tensors),
            "activations": len(ag.lifetimes),
            "unscheduled_peak_bytes": ag.peak_memory_bytes,
            "largest_tensor": {"shape": f.largest_tensor_shape, "bytes": f.largest_tensor_bytes},
            # Shapes in the caller's axis order, as the interface takes them.
            "inputs": [{"name": n, "shape": list(source_shape(ag, ag.tensors[n].shape))}
                       for n in ag.model_inputs],
            "outputs": [{"name": n, "shape": list(source_shape(ag, ag.tensors[n].shape))}
                        for n in ag.model_outputs],
            "unsupported_operators": list(f.unsupported_operators),
            "output_errors": list(f.output_errors),
            "dtype_errors": list(f.dtype_errors),
        },
        "tflite_micro": None if tflm_arena is None else {
            "tensor_arena_bytes": tflm_arena,
            "excludes": ["kernel scratch buffers other than those of MEAN, SUM, REDUCE_MAX, "
                         "REDUCE_MIN and REDUCE_ALL", "persistent allocations"],
            "export": None if tflm_export is None else export_report(tflm_export),
        },
        "fast": {
            "budget_bytes": budget,
            "verdict": f.verdict or None,
            "scheduled_peak_bytes": f.scheduled_peak_bytes,
            "stages": f.total_stages,
            "stages_needing_tiling": f.stages_needing_tiling,
            "stages_tileable": f.stages_tileable,
            "stages_untileable": f.stages_untileable,
            "untileable_operators": list(f.untileable_op_types),
            "minimum_for_partition_bytes": f.min_fast_for_partition,
            "blocking_stages": [{"stage": b.stage_id, "required_bytes": b.required_bytes,
                                 "reason": b.reason} for b in f.blocking_stages],
            "infeasible": list(f.feasibility_errors),
        },
        "slow": {
            "budget_bytes": ag.budget.slow,
            "peak_bytes": f.slow_peak_bytes,
            "fits": f.slow_fits,
            "overflow_stages": list(f.slow_overflow_stages),
            # Measured on the host runtime; null when not measured.
            "traffic": traffic if isinstance(traffic, dict) else None,
        },
        "flash": {
            "budget_bytes": ag.budget.flash,
            "weight_bytes": f.total_weight_bytes,
            "plan_bytes": f.plan_size_bytes or None,
            "plan_fits": f.plan_fits_flash if ag.budget.flash and f.plan_size_bytes else None,
            "estimates": {"int8_plan_bytes": f.int8_plan_size_bytes or None,
                          "lz4_plan_bytes": f.lz4_plan_size_bytes or None},
        },
        # Runtime build limits the plan needs above their defaults.
        "runtime_build": list(f.build_limits),
        "budgets_tried": [{"budget_bytes": b, "fits": fits, "stages": stages, "tiled": tiled}
                          for b, fits, stages, tiled in tried],
        "stages": [{
            "stage": s.stage_id,
            "operators": len(s.op_indices),
            "peak_bytes": s.peak_bytes,
            "inputs": len(s.input_tensors),
            "outputs": len(s.output_tensors),
            "needs_tiling": bool(s.warnings),
            "tiles": (None if s.tile_plan is None or not s.tile_plan.tileable else {
                "axis": s.tile_plan.axis, "tile": s.tile_plan.tile_height,
                "count": s.tile_plan.num_tiles, "halo": s.tile_plan.halo,
                "tiled_peak_bytes": s.tile_plan.tiled_peak_bytes}),
        } for s in ag.stages],
    }




def _tried(model: str, budget: int, fits: bool, input_shape) -> list[tuple[int, bool, int, int]]:
    """Other budgets compiled for comparison, as (budget, fits, stages,
    tiled stages): halving from a budget that fits, doubling from one that
    does not, until the answer changes. Only budgets actually compiled are
    reported; a budget that fits does not imply every larger one does."""
    from tigris.analysis.validation import validate_memory_plan

    tried = []
    candidate = budget
    for _ in range(_BUDGETS_TRIED):
        candidate = candidate // 2 if fits else candidate * 2
        if candidate <= 0 or candidate > 0xFFFFFFFF:
            break
        try:
            graph = _planned(model, (str(candidate),), input_shapes=input_shape,
                             report_bindings=False)
        except click.ClickException:
            break
        ok = validate_memory_plan(graph).feasible
        tiled = sum(runs_tiled(s) for s in graph.stages)
        tried.append((candidate, ok, len(graph.stages), tiled))
        if ok != fits:
            break
    return tried


def _lz4_compiles(model: str, mem: tuple[str, ...], input_shape) -> bool:
    """Whether compile -c lz4 writes a plan at these budgets: the fast arena
    also holds each stage's decompressed weights, which can leave too little."""
    from tigris.analysis.validation import plan_gate

    try:
        graph = _planned(model, mem, compress="lz4", input_shapes=input_shape,
                         report_bindings=False)
    except click.ClickException:
        return False
    return plan_gate(graph).passes


def interface_rows(ag) -> list[list[str]]:
    """One row per input and output: label, name, then shape and dtype."""
    rows = []
    for label, value in describe_interface(ag):
        name, _, rest = value.partition(" ")
        rows.append([label.lower(), name, rest] if label != "State" else [label.lower(), value, ""])
    return rows


def runs_tiled(stage) -> bool:
    """A stage that executes in tiles: over the budget with a usable tile
    plan, or a place in a chain."""
    return ((bool(stage.warnings) and stage.tile_plan is not None and stage.tile_plan.tileable)
            or stage.chain_id != 0xFFFF)


def _stage_count(n: int) -> str:
    return plural(n, "stage")


def verdict_lines(findings, budget: int, ag) -> list[str]:
    """The verdict, then one line per cause, worst first."""
    f = findings
    if f.unsupported_operators:
        count = len(f.unsupported_operators)
        lines = [text.bad("cannot compile") +
                 f": {plural(count, 'unsupported operator')}"]
        return lines + text.columns([[issue] for issue in f.unsupported_operators[:text.CAUSES_SHOWN]])
    if f.output_errors or f.dtype_errors:
        return [text.bad("cannot compile") + f": {(f.output_errors + f.dtype_errors)[0]}"]
    if budget <= 0:
        return [text.dim("no budget given; memory figures only")]
    tiled = sum(runs_tiled(s) for s in ag.stages)
    if f.blocking_stages:
        count = len(f.blocking_stages)
        lines = [text.bad("does not fit") + f" {fmt_bytes(budget)} fast memory: "
                 f"{_stage_count(count)} exceed{'s' if count == 1 else ''} it"]
        causes = sorted(f.blocking_stages, key=lambda b: (-b.required_bytes, b.stage_id))
        rows = [[f"stage {b.stage_id}", text.bad(fmt_bytes(b.required_bytes)), b.reason]
                for b in causes[:text.CAUSES_SHOWN]]
        lines += text.columns(rows, "<>")
        if count > text.CAUSES_SHOWN:
            lines.append(text.dim(f"  and {count - text.CAUSES_SHOWN} more; --json lists all"))
        return lines
    fit = f" {fmt_bytes(budget)} fast memory, {_stage_count(f.total_stages)}"
    if tiled:
        fit += f", {tiled} tiled"
    if not f.slow_fits:
        return [text.warn(f"fits{fit}; slow memory {fmt_bytes(f.slow_peak_bytes)} needed, "
                          f"{fmt_bytes(ag.budget.slow)} given")]
    return [text.good("fits") + fit]


def summary(model: str, ag, findings, budget: int, tflm_arena: int | None,
            tried: list | None = None, traffic=None, lz4_tried: bool = False,
            tflm_export=None) -> None:
    """The header, verdict, memory and flash, shared by analyze and compile.
    `traffic` is what trace.traffic measured; None leaves the row out. The LZ4
    row needs `lz4_tried`: the estimate stands only where -c lz4 compiles.
    `tflm_export` is what tflite_export.export found; None leaves its row out."""
    f = findings
    dtype = "int8" if f.is_quantized else "float32" if f.is_float32 else "mixed"
    graphs = len(getattr(ag, "subgraphs", []) or [])
    title = f"{dtype}, {plural(len(ag.ops), 'operator')}" + (
        f", {plural(graphs, 'subgraph')}" if graphs else "")
    text.gap()
    text.echo(text.bold(Path(model).name) + "   " + title)
    for line in text.columns(interface_rows(ag)):
        text.echo(line)
    for line in verdict_lines(f, budget, ag):
        text.echo(line)

    memory = []
    if tflm_arena is not None:
        memory.append(["TFLite Micro arena", fmt_bytes(tflm_arena),
                       text.dim("its planner and reduction scratch; persistent allocations and "
                                "other kernel scratch excluded")])
    if tflm_export is not None:
        from tigris.cli.export import export_line
        memory.append(export_line(tflm_export))
    memory.append(["unscheduled", fmt_bytes(ag.peak_memory_bytes), ""])
    if f.largest_tensor_bytes:
        memory.append(["largest tensor", fmt_bytes(f.largest_tensor_bytes),
                       text.dim(f.largest_tensor_shape)])
    if budget > 0 and f.gate.usable and f.scheduled_peak_bytes > 0 and not f.blocking_stages:
        memory.append(["this plan", fmt_bytes(f.scheduled_peak_bytes),
                       text.dim(f"{fmt_bytes(budget - f.scheduled_peak_bytes)} headroom")])
    if f.slow_peak_bytes > 0:
        memory.append(["slow memory", fmt_bytes(f.slow_peak_bytes),
                       text.dim(f"budget {fmt_bytes(ag.budget.slow)}") if ag.budget.slow else ""])
    if isinstance(traffic, dict):
        memory.append(["slow traffic", fmt_bytes(traffic["written"] + traffic["read"]),
                       text.dim(f"per inference: {fmt_bytes(traffic['written'])} written, "
                                f"{fmt_bytes(traffic['read'])} read")])
    elif traffic is not None:
        memory.append(["slow traffic", "", text.dim(f"not measured: {traffic}")])
    fits_here = not f.blocking_stages
    for b, fits, stages, tiled in tried or []:
        label = ("also fits at" if fits_here else "fits at") if fits else "does not fit at"
        detail = _stage_count(stages) + (f", {tiled} tiled" if tiled else "") if fits else ""
        memory.append([label, fmt_bytes(b), text.dim(detail)])
    text.section("memory", memory, "<>")

    if f.total_weight_bytes > 0 or f.plan_size_bytes > 0:
        flash = []
        if not f.plan_size_bytes:
            flash.append(["plan", "", text.dim("none at this budget")])
        else:
            flash.append(["plan", fmt_bytes(f.plan_size_bytes),
                          text.dim(f"weights {fmt_bytes(f.total_weight_bytes)}, "
                                   f"overhead {fmt_bytes(f.plan_overhead_bytes)}")])
            if lz4_tried and 0 < f.lz4_plan_size_bytes < f.plan_size_bytes * 95 // 100:
                flash.append(["with -c lz4", fmt_bytes(f.lz4_plan_size_bytes), text.dim("estimate")])
            if f.is_float32 and not f.is_quantized:
                flash.append(["as int8", fmt_bytes(f.int8_plan_size_bytes), text.dim("estimate")])
            if ag.budget.flash > 0:
                over = f.plan_size_bytes - ag.budget.flash
                flash.append(["flash budget", fmt_bytes(ag.budget.flash),
                              text.good("fits") if over <= 0 else
                              text.bad(f"exceeds it by {fmt_bytes(over)}")])
        text.section("flash", flash, "<>")
    if f.build_limits:
        # As inspect states it: the -D options a runtime build needs for this plan.
        text.block([["runtime build", ", ".join(f"-D{limit['name']}={limit['needed']}"
                                                for limit in f.build_limits)]])


def _stages(ag) -> None:
    """One row per stage of the planned graph: its operators, untiled peak, input and
    output tensor counts, and how it is tiled or chained."""
    rows = [[text.dim(cell) for cell in ("stage", "ops", "untiled peak", "in", "out", "tiling")]]
    for s in ag.stages:
        tp = s.tile_plan
        tiling = "untiled"
        if runs_tiled(s) and tp is not None and tp.tileable:
            tiling = (f"{tp.num_tiles} tiles of {tp.tile_height} x {tp.tile_width}, halo {tp.halo}"
                      if tp.axis == TILE_AXIS_HW else f"{tp.num_tiles} tiles, axis {tp.axis}, halo {tp.halo}")
        if s.chain_id != 0xFFFF:
            tiling = (f"chain of {s.chain_len}, tile height {s.chain_tile_h}"
                      if s.chain_id == s.stage_id else f"in chain {s.chain_id}")
        first, last = s.op_indices[0], s.op_indices[-1]
        rows.append([str(s.stage_id), str(first) if first == last else f"{first}-{last}",
                     fmt_bytes(s.peak_bytes), str(len(s.input_tensors)),
                     str(len(s.output_tensors)), tiling])
    text.section("stages", rows, ">>>>>")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, callback=_memory_sizes,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--flash", "-f", default=None, callback=_memory_sizes,
              help="Flash size for plan fit check (e.g. 4M)")
@click.option("--verbose", "-v", is_flag=True, help="Add the per-stage table")
@click.option("--input-shape", "input_shape", multiple=True,
              callback=_parse_input_shape,
              help="Shape to compile an input for (e.g. --input-shape input:1x3x224x224)")
@click.option("--json", "as_json", is_flag=True, help="Emit the analysis as versioned JSON")
@click.option("--trace", is_flag=True,
              help="Run the plan on the host runtime and print what it did instead")
@click.option("--input", "input_files", multiple=True,
              help="Input for --trace as .bin or .npy; NAME=FILE for several (default: zeros)")
def analyze(model: str, mem: tuple[str, ...], flash: str | None, verbose: bool,
            input_shape: dict[str, tuple[int, ...]], as_json: bool, trace: bool,
            input_files: tuple[str, ...]):
    """Check whether a model fits a memory budget, and why not."""
    from tigris.analysis.findings import compute_findings
    from tigris.frontends.tflite import is_tflite
    from tigris.frontends.tflite_export import export, tflm_tensor_arena

    if as_json and trace:
        raise click.UsageError("--json and --trace are two different outputs; choose one")
    if input_files and not trace:
        raise click.UsageError("--input is for --trace")
    mem_pools = [parse_size(m) for m in mem]
    flash_budget = parse_size(flash) if flash else 0
    budget = mem_pools[0] if mem_pools else 0
    slow_budget = mem_pools[1] if len(mem_pools) > 1 else 0
    # Only the fast tier is planned: analyze interprets the slow tier itself
    # and stays display-only.
    ag = _planned(model, mem[:1], input_shapes=input_shape, report_bindings=not as_json)
    ag.budget = replace(ag.budget, slow=slow_budget, flash=flash_budget)
    if trace:
        from tigris.cli.trace import trace_model
        trace_model(model, ag, input_files, verbose)
        return

    data = Path(model).read_bytes()
    tflm_arena = tflm_tensor_arena(data)
    findings = compute_findings(ag)
    supported = findings.gate.usable
    tried = (_tried(model, budget, not findings.blocking_stages, input_shape)
             if budget > 0 and supported else [])
    if findings.lz4_plan_size_bytes and not (
            supported and not findings.blocking_stages and _lz4_compiles(model, mem, input_shape)):
        findings.lz4_plan_size_bytes = 0
    from tigris.cli.trace import traffic as measure_traffic
    traffic = measure_traffic(ag) if supported else None
    tflm_export = None if tflm_arena is None else export(data)
    if as_json:
        click.echo(json.dumps(_report(ag, findings, budget, "tflite" if is_tflite(data) else "onnx",
                                      tflm_arena, tried, traffic, tflm_export),
                              indent=2, ensure_ascii=True, allow_nan=False))
        return
    summary(model, ag, findings, budget, tflm_arena, tried, traffic, lz4_tried=True,
            tflm_export=tflm_export)
    if verbose and ag.stages:
        _stages(ag)
