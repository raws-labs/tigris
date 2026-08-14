"""``tigris analyze`` command."""

from dataclasses import replace

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tigris.cli import cli, console, _expand_mem, _parse_size, _run_pipeline
from tigris.utils import fmt_bytes


def _side_by_side(*panels):
    """Render panels side by side if the terminal is wide enough, else stack."""
    panels = [p for p in panels if p is not None]
    if not panels:
        return
    if len(panels) == 1:
        console.print(panels[0])
        return

    # Need roughly 40 chars per panel + 1 gap
    if console.width >= 40 * len(panels) + len(panels) - 1:
        grid = Table.grid(padding=(0, 1), expand=True)
        for _ in panels:
            grid.add_column(ratio=1)
        grid.add_row(*panels)
        console.print(grid)
    else:
        for p in panels:
            console.print(p)


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--mem", "-m", multiple=True, callback=_expand_mem,
              help="Memory pool size, fast to slow (e.g. -m 256K or -m 256K+8M)")
@click.option("--flash", "-f", default=None, help="Flash size for plan fit check (e.g. 4M)")
@click.option("--verbose", "-v", is_flag=True, help="Show per-stage and tiling tables")
def analyze(model: str, mem: tuple[str, ...], flash: str | None, verbose: bool):
    """Analyze an ONNX model for memory-constrained deployment."""
    from tigris.analysis.findings import compute_findings

    mem_pools = [_parse_size(m) for m in mem]
    flash_budget = _parse_size(flash) if flash else 0
    slow_budget = mem_pools[1] if len(mem_pools) > 1 else 0
    # Only forward the fast tier to _run_pipeline: analyze interprets the slow
    # tier itself below and stays display-only, so a non-positive slow tier
    # must be reported as unconstrained rather than raised.
    ag, budget = _run_pipeline(model, mem[:1])
    ag.budget = replace(ag.budget, slow=slow_budget, flash=flash_budget)

    with console.status("Computing findings..."):
        findings = compute_findings(ag, flash_budget=flash_budget)

    # Model
    model_grid = Table.grid(padding=(0, 2))
    model_grid.add_column(style="bold")
    model_grid.add_column()
    model_grid.add_row("Operators", str(len(ag.ops)))
    model_grid.add_row("Tensors", f"{len(ag.tensors)} ({len(ag.lifetimes)} activations)")
    model_grid.add_row("Peak memory (naive)", fmt_bytes(ag.peak_memory_bytes))
    if findings.largest_tensor_shape:
        model_grid.add_row("Largest tensor", f"{findings.largest_tensor_shape} ({fmt_bytes(findings.largest_tensor_bytes)})")
    if findings.is_quantized:
        model_grid.add_row("Quantization", "INT8 (QDQ)")
    elif findings.is_float32:
        model_grid.add_row("Dtype", "float32")
    if findings.unsupported_operators:
        model_grid.add_row(
            "[red]Unsupported operators[/]",
            "[red]" + ", ".join(findings.unsupported_operators) + "[/]",
        )
    if findings.dtype_errors:
        model_grid.add_row(
            "[red]Unsupported dtype[/]",
            "[red]" + "; ".join(findings.dtype_errors) + "[/]",
        )

    model_failed = bool(findings.unsupported_operators or findings.dtype_errors)
    model_panel = Panel(
        model_grid,
        title=f"[bold]TiGrIS - {ag.model_name}[/]",
        subtitle=(
            "[bold red] FAIL - unsupported deployment contract [/]"
            if model_failed
            else None
        ),
        border_style="red" if model_failed else "blue",
    )
    console.print(model_panel)

    # SRAM
    sram_panel = None
    if budget > 0:
        sram_style_map = {"ok": "green", "partitioned": "green", "tiled": "yellow", "needs_work": "red"}
        sram_label_map = {"ok": "PASS", "partitioned": "PASS", "tiled": "PASS", "needs_work": "FAIL"}
        sram_verdict_map = {
            "ok": "fits in budget",
            "partitioned": "partitioned, no tiling needed",
            "tiled": "tiling resolves all stages",
            "needs_work": "untileable stages remain",
        }
        vs = sram_style_map.get(findings.verdict, "dim")
        vlabel = sram_label_map.get(findings.verdict, "?")
        vtext = sram_verdict_map.get(findings.verdict, "")
        if findings.unsupported_operators or findings.dtype_errors:
            vtext = "unsupported deployment contract"

        sram = Table.grid(padding=(0, 2))
        sram.add_column(style="bold")
        sram.add_column()
        sram.add_row("Budget", fmt_bytes(budget))
        if len(mem_pools) > 1:
            for i, pool in enumerate(mem_pools[1:], 1):
                sram.add_row(f"  pool {i+1} (slow)", fmt_bytes(pool))
        if findings.scheduled_peak_bytes > 0 and findings.ratio > 1.0:
            pct = findings.scheduled_peak_bytes / findings.peak_bytes * 100
            sram.add_row(
                "Scheduled peak",
                f"{fmt_bytes(findings.scheduled_peak_bytes)} ({pct:.1f}% of naive peak)",
            )
        sram.add_row("Stages", str(findings.total_stages))

        if findings.stage_transitions > 0:
            sram.add_row("Spill / reload I/O", f"{fmt_bytes(findings.total_spill_bytes)} / {fmt_bytes(findings.total_reload_bytes)}")

        # Tiling breakdown
        if findings.stages_needing_tiling > 0:
            sram.add_row("", "")
            sram.add_row("Need tiling", f"{findings.stages_needing_tiling} of {findings.total_stages} stages")
            if findings.stages_tileable > 0:
                tile_style = "red" if findings.feasibility_errors else "green"
                sram.add_row(
                    "  tileable",
                    f"[{tile_style}]{findings.stages_tileable}[/] "
                    f"({findings.total_tiles} tiles, max halo {findings.max_halo})",
                )
            if findings.stages_untileable > 0:
                ops = ", ".join(findings.untileable_op_types)
                sram.add_row(
                    "  untileable",
                    f"[red]{findings.stages_untileable}[/] - blocked by: {ops}",
                )
                sram.add_row(
                    "  min SRAM (hard floor)",
                    fmt_bytes(findings.min_untileable_peak),
                )
                for us in findings.untileable_stages:
                    ops_str = ", ".join(us.op_types)
                    sram.add_row(
                        f"    stage {us.stage_id}",
                        f"{fmt_bytes(us.peak_bytes)} ({ops_str})",
                    )

        if findings.feasibility_errors:
            sram.add_row("", "")
            for i, error in enumerate(findings.feasibility_errors[:3]):
                sram.add_row("Infeasible" if i == 0 else "", f"[red]{error}[/]")
            if len(findings.feasibility_errors) > 3:
                sram.add_row("", f"... and {len(findings.feasibility_errors) - 3} more")
            vtext = "minimum execution unit exceeds budget"

        # Slow memory (PSRAM) overflow warning
        if slow_budget > 0 and not findings.slow_fits:
            sram.add_row("", "")
            sram.add_row(
                "[red]Slow memory overflow[/]",
                f"{len(findings.slow_overflow_stages)} stage(s)",
            )
            sram.add_row(
                "  peak (in+out)",
                f"[red]{fmt_bytes(findings.slow_peak_bytes)}[/] > {fmt_bytes(slow_budget)}",
            )
            for sid in findings.slow_overflow_stages[:3]:
                sram.add_row(f"    stage {sid}", "[red]overflow[/]")
            if len(findings.slow_overflow_stages) > 3:
                sram.add_row("", f"... and {len(findings.slow_overflow_stages) - 3} more")
            vs = "red"
            vlabel = "FAIL"
            vtext = "slow memory overflow"

        subtitle = f"[bold {vs}] {vlabel} - {vtext} [/]" if findings.verdict else None
        sram_panel = Panel(sram, title="[bold]SRAM[/]", subtitle=subtitle, border_style=vs)

    # Flash
    flash_panel = None
    if findings.total_weight_bytes > 0:
        fl = Table.grid(padding=(0, 2))
        fl.add_column(style="bold")
        fl.add_column(justify="right")

        # Use plan size as unit reference so all flash values share the same unit
        _ur = findings.plan_size_bytes

        # Format a plan size row, colored by flash fit
        def _flash_row(b: int) -> str:
            v = fmt_bytes(b, unit_ref=_ur)
            if flash_budget <= 0:
                return v
            return f"[green]{v}[/]" if b <= flash_budget else f"[red]{v}[/]"

        fs = "magenta"
        flash_subtitle = None
        if flash_budget > 0:
            fl.add_row("Budget", fmt_bytes(flash_budget, unit_ref=_ur))
            if findings.plan_fits_flash:
                fs = "green"
                flash_subtitle = "[bold green] PASS - plan fits [/]"
            else:
                fs = "red"
                flash_subtitle = "[bold red] FAIL - plan does not fit [/]"

        fl.add_row("Weight data", fmt_bytes(findings.total_weight_bytes, unit_ref=_ur))
        fl.add_row("Plan overhead", fmt_bytes(findings.plan_overhead_bytes, unit_ref=_ur))
        fl.add_row("Plan (est.)", _flash_row(findings.plan_size_bytes))
        if findings.lz4_plan_size_bytes > 0 and findings.lz4_plan_size_bytes < findings.plan_size_bytes * 95 // 100:
            fl.add_row("Plan LZ4 (est.)", _flash_row(findings.lz4_plan_size_bytes))
        if findings.is_float32 and not findings.is_quantized:
            fl.add_row("Plan INT8 (est.)", _flash_row(findings.int8_plan_size_bytes))

        flash_panel = Panel(fl, title="[bold]Flash[/]", subtitle=flash_subtitle, border_style=fs)

    _side_by_side(sram_panel, flash_panel)

    # Budget Sweep
    if verbose and findings.budget_sweep:
        table = Table(title="Budget Comparison", border_style="dim")
        table.add_column("Budget", justify="right")
        table.add_column("Stages", justify="right")
        table.add_column("Need Tiling", justify="right")
        table.add_column("Status")

        for row in findings.budget_sweep:
            is_current = row.budget == budget
            status = Text("OK", style="green") if row.ok else Text(
                f"{row.need_tiling} need tiling", style="red"
            )
            style = "bold" if is_current else ""
            marker = " <--" if is_current else ""
            table.add_row(
                row.budget_str + marker,
                str(row.stages),
                str(row.need_tiling),
                status,
                style=style,
            )

        console.print(table)

    # Stages
    if verbose and ag.stages:
        table = Table(title=f"Stages ({len(ag.stages)})", border_style="dim")
        table.add_column("#", justify="right")
        table.add_column("Ops", justify="right")
        table.add_column("Peak", justify="right")
        table.add_column("In", justify="right")
        table.add_column("Out", justify="right")
        table.add_column("Note")

        for s in ag.stages:
            note = Text("NEEDS TILING", style="red") if s.warnings else Text("")
            table.add_row(
                str(s.stage_id),
                str(len(s.op_indices)),
                fmt_bytes(s.peak_bytes),
                str(len(s.input_tensors)),
                str(len(s.output_tensors)),
                note,
            )

        console.print(table)

    # Tiling Analysis
    tiled_stages = [s for s in ag.stages if s.tile_plan is not None]
    if verbose and tiled_stages:
        table = Table(title="Tiling Analysis", border_style="dim")
        table.add_column("Stage", justify="right")
        table.add_column("Tileable?")
        table.add_column("Axis", justify="right")
        table.add_column("Tile", justify="right")
        table.add_column("Tiles", justify="right")
        table.add_column("Halo", justify="right")
        table.add_column("RF", justify="right")
        table.add_column("Tiled Peak", justify="right")

        for s in tiled_stages:
            tp = s.tile_plan
            if tp is None:
                continue
            tileable = Text("Yes", style="green") if tp.tileable else Text("No", style="red")
            tile_h = str(tp.tile_height) if tp.tileable else "-"
            axis = str(tp.axis) if tp.tileable else "-"
            tiles = str(tp.num_tiles) if tp.tileable else "-"
            halo_str = str(tp.halo) if tp.tileable else "-"
            rf_str = str(tp.receptive_field) if tp.tileable else "-"
            tiled_peak = fmt_bytes(tp.tiled_peak_bytes) if tp.tileable else "-"

            table.add_row(
                str(s.stage_id), tileable, axis, tile_h, tiles,
                halo_str, rf_str, tiled_peak,
            )

        console.print(table)
