"""``tigris analyze``: does a model fit a budget, and if not, why."""

import json
from dataclasses import replace
from pathlib import Path

import click

from tigris.cli import _expand_mem, _parse_input_shape, _parse_size, _run_pipeline, cli, text
from tigris.utils import describe_interface, fmt_bytes, source_shape

# Causes listed under a verdict; --json lists all.
_CAUSES_SHOWN = 5
# Budgets tried beyond the one given, halving below a fit or doubling above a failure.
_BUDGETS_TRIED = 3


def _report(ag, findings, budget: int, slow_budget: int, flash_budget: int, source: str,
            tflm_arena: int | None, tried: list) -> dict:
    """The analysis as versioned JSON; byte counts are integers."""
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
            "dtype_errors": list(f.dtype_errors),
        },
        "tflite_micro": None if tflm_arena is None else {
            "tensor_arena_bytes": tflm_arena,
            "excludes": ["kernel scratch buffers", "persistent allocations"],
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
            "budget_bytes": slow_budget,
            "peak_bytes": f.slow_peak_bytes,
            "fits": f.slow_fits,
            "overflow_stages": list(f.slow_overflow_stages),
        },
        "flash": {
            "budget_bytes": flash_budget,
            "weight_bytes": f.total_weight_bytes,
            "plan_bytes": f.plan_size_bytes or None,
            "plan_fits": f.plan_fits_flash if flash_budget and f.plan_size_bytes else None,
            "estimates": {"int8_plan_bytes": f.int8_plan_size_bytes or None,
                          "lz4_plan_bytes": f.lz4_plan_size_bytes or None},
        },
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
            graph, _ = _run_pipeline(model, (str(candidate),), input_shapes=input_shape,
                                     report_bindings=False)
        except click.ClickException:
            break
        ok = validate_memory_plan(graph).feasible
        tiled = sum(runs_tiled(s) for s in graph.stages)
        tried.append((candidate, ok, len(graph.stages), tiled))
        if ok != fits:
            break
    return tried


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


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _stage_count(n: int) -> str:
    return _count(n, "stage")


def verdict_lines(findings, budget: int, slow_budget: int, stages=()) -> list[str]:
    """The verdict, then one line per cause, worst first."""
    f = findings
    if f.unsupported_operators:
        count = len(f.unsupported_operators)
        lines = [text.bad("cannot compile") +
                 f": {count} unsupported operator{'s' if count != 1 else ''}"]
        return lines + text.columns([[issue] for issue in f.unsupported_operators[:_CAUSES_SHOWN]])
    if f.dtype_errors:
        return [text.bad("cannot compile") + f": {f.dtype_errors[0]}"]
    if budget <= 0:
        return [text.dim("no budget given; memory figures only")]
    tiled = sum(runs_tiled(s) for s in stages)
    if f.blocking_stages:
        count = len(f.blocking_stages)
        lines = [text.bad("does not fit") + f" {fmt_bytes(budget)} fast memory: "
                 f"{_stage_count(count)} exceed{'s' if count == 1 else ''} it"]
        causes = sorted(f.blocking_stages, key=lambda b: (-b.required_bytes, b.stage_id))
        rows = [[f"stage {b.stage_id}", text.bad(fmt_bytes(b.required_bytes)), b.reason]
                for b in causes[:_CAUSES_SHOWN]]
        lines += text.columns(rows, "<>")
        if count > _CAUSES_SHOWN:
            lines.append(text.dim(f"  and {count - _CAUSES_SHOWN} more; --json lists all"))
        return lines
    fit = f" {fmt_bytes(budget)} fast memory, {_stage_count(f.total_stages)}"
    if tiled:
        fit += f", {tiled} tiled"
    if not f.slow_fits:
        return [text.warn(f"fits{fit}; slow memory {fmt_bytes(f.slow_peak_bytes)} needed, "
                          f"{fmt_bytes(slow_budget)} given")]
    return [text.good("fits") + fit]


def summary(model: str, ag, findings, budget: int, slow_budget: int, flash_budget: int,
            tflm_arena: int | None, tried: list | None = None) -> None:
    """The header, verdict, memory and flash, shared by analyze and compile."""
    f = findings
    dtype = "int8" if f.is_quantized else "float32" if f.is_float32 else "mixed"
    graphs = len(getattr(ag, "subgraphs", []) or [])
    title = f"{dtype}, {_count(len(ag.ops), 'operator')}" + (
        f", {_count(graphs, 'subgraph')}" if graphs else "")
    text.gap()
    text.echo(text.bold(Path(model).name) + "   " + title)
    for line in text.columns(interface_rows(ag)):
        text.echo(line)
    for line in verdict_lines(f, budget, slow_budget, ag.stages):
        text.echo(line)

    memory = []
    if tflm_arena is not None:
        memory.append(["TFLite Micro tensors", fmt_bytes(tflm_arena),
                       text.dim("its planner; kernel scratch excluded")])
    memory.append(["unscheduled", fmt_bytes(ag.peak_memory_bytes), ""])
    if f.largest_tensor_bytes:
        memory.append(["largest tensor", fmt_bytes(f.largest_tensor_bytes),
                       text.dim(f.largest_tensor_shape)])
    supported = not (f.unsupported_operators or f.dtype_errors)
    if budget > 0 and supported and f.scheduled_peak_bytes > 0 and not f.blocking_stages:
        memory.append(["this plan", fmt_bytes(f.scheduled_peak_bytes),
                       text.dim(f"{fmt_bytes(budget - f.scheduled_peak_bytes)} headroom")])
    if f.slow_peak_bytes > 0:
        memory.append(["slow memory", fmt_bytes(f.slow_peak_bytes),
                       text.dim(f"budget {fmt_bytes(slow_budget)}") if slow_budget else ""])
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
            if 0 < f.lz4_plan_size_bytes < f.plan_size_bytes * 95 // 100:
                flash.append(["with -c lz4", fmt_bytes(f.lz4_plan_size_bytes), text.dim("estimate")])
            if f.is_float32 and not f.is_quantized:
                flash.append(["as int8", fmt_bytes(f.int8_plan_size_bytes), text.dim("estimate")])
            if flash_budget > 0:
                over = f.plan_size_bytes - flash_budget
                flash.append(["flash budget", fmt_bytes(flash_budget),
                              text.good("fits") if over <= 0 else
                              text.bad(f"exceeds it by {fmt_bytes(over)}")])
        text.section("flash", flash, "<>")


def _stages(ag) -> None:
    """One row per stage, as inspect prints a plan's stages."""
    rows = [[text.dim(cell) for cell in ("stage", "ops", "untiled peak", "in", "out", "tiling")]]
    for s in ag.stages:
        tp = s.tile_plan
        tiling = (f"{tp.num_tiles} tiles, axis {tp.axis}, halo {tp.halo}"
                  if runs_tiled(s) and tp is not None and tp.tileable else "untiled")
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
@click.option("--mem", "-m", multiple=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+4M)")
@click.option("--flash", "-f", default=None, help="Flash size for plan fit check (e.g. 4M)")
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
    from tigris.frontends.tflite import is_tflite, tflm_tensor_arena

    if as_json and trace:
        raise click.UsageError("--json and --trace are two different outputs; choose one")
    if input_files and not trace:
        raise click.UsageError("--input is for --trace")
    mem_pools = [_parse_size(m) for m in mem]
    flash_budget = _parse_size(flash) if flash else 0
    slow_budget = mem_pools[1] if len(mem_pools) > 1 else 0
    # Only forward the fast tier to _run_pipeline: analyze interprets the slow
    # tier itself and stays display-only, so a non-positive slow tier is
    # reported as unconstrained rather than raised.
    ag, budget = _run_pipeline(model, mem[:1], input_shapes=input_shape, report_bindings=not as_json)
    ag.budget = replace(ag.budget, slow=slow_budget, flash=flash_budget)
    if trace:
        from tigris.cli.trace import trace_model
        trace_model(model, ag, input_files, verbose)
        return

    data = Path(model).read_bytes()
    tflite = is_tflite(data)
    tflm_arena = tflm_tensor_arena(data) if tflite else None
    findings = compute_findings(ag, flash_budget=flash_budget)
    supported = not (findings.unsupported_operators or findings.dtype_errors)
    tried = (_tried(model, budget, not findings.blocking_stages, input_shape)
             if budget > 0 and supported else [])
    if as_json:
        click.echo(json.dumps(_report(ag, findings, budget, slow_budget, flash_budget,
                                      "tflite" if tflite else "onnx", tflm_arena, tried),
                              indent=2, ensure_ascii=True, allow_nan=False))
        return
    summary(model, ag, findings, budget, slow_budget, flash_budget, tflm_arena, tried)
    if verbose and ag.stages:
        _stages(ag)
