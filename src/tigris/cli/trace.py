"""The execution trace: what the runtime did when it ran a plan, event by event."""

from tigris.cli import text
from tigris.utils import fmt_bytes

_PATHS = {"normal": "untiled", "tiled": "1D tile", "tiled_2d": "2D tile", "by_input": "reduce by rows",
          "transpose": "transpose", "rows": "row bands", "reshape": "reshape bands", "control": "control"}
# Paths whose event ranges are rows and columns of the tensor; the others
# report ranges in their own iteration order.
_SPATIAL = {"tiled", "tiled_2d", "chain"}
_NAME_WIDTH = 32


def units(plan: dict, events: list[dict]) -> list[dict]:
    """Group events into execution units, one per stage or chain, in the order
    they ran, with what each read, wrote and held."""
    result, current = [], None
    for event in events:
        if event["kind"] == "stage_begin":
            stage = plan["stages"][event["stage"]] if event["stage"] is not None else None
            kind = (f"chain {stage['chain_len']}" if event["path"] == "chain" and stage
                    else _PATHS.get(event["path"], event["path"]))
            current = {"stage": event["stage"], "path": event["path"], "kind": kind, "tiles": 0, "read": 0, "written": 0,
                       "weights": 0, "copied": 0, "fast_peak": event["fast_used"],
                       "slow_used": event["slow_used"], "events": []}
            result.append(current)
            continue
        if current is None:
            continue
        current["events"].append(event)
        current["fast_peak"] = max(current["fast_peak"], event["fast_used"])
        current["slow_used"] = max(current["slow_used"], event["slow_used"])
        kind = event["kind"]
        if kind == "tile_begin":
            current["tiles"] += 1
        elif kind == "load":
            current["read"] += event["bytes"]
        elif kind == "spill":
            current["written"] += event["bytes"]
        elif kind == "weights":
            current["weights"] += event["bytes"]
        elif kind == "copy":
            current["copied"] += event["bytes"]
        elif kind == "stage_end":
            current = None
    return result


def check(events: list[dict], counters: dict[str, int]) -> list[str]:
    """Differences between the event stream and the runtime's own counters."""
    sums = {"load_bytes": "load", "spill_bytes": "spill", "weight_bytes": "weights", "copy_bytes": "copy"}
    problems = []
    for counter, kind in sums.items():
        total = sum(event["bytes"] for event in events if event["kind"] == kind)
        if total != counters[counter]:
            problems.append(f"{kind} events add up to {total} bytes, the runtime counted {counters[counter]}")
    tiles = sum(event["kind"] == "tile_begin" for event in events)
    if tiles != counters["tiles"]:
        problems.append(f"{tiles} tile events, the runtime counted {counters['tiles']} tiles")
    return problems


def _name(plan: dict, index) -> str:
    if index is None:
        return ""
    name = plan["tensors"][index]["name"]
    return name if len(name) <= _NAME_WIDTH else name[:_NAME_WIDTH - 3] + "..."


def _rows(event) -> str:
    if event["row0"] is None:
        return ""
    rows = f"rows {event['row0']}-{event['row1'] - 1}"
    if event["col0"] is not None:
        rows += f", cols {event['col0']}-{event['col1'] - 1}"
    return rows


def _event_row(plan: dict, event: dict, spatial: bool) -> list[str] | None:
    kind = event["kind"]
    rows = _rows(event) if spatial else ""
    where = f"{event['pool']}+{event['offset']}" if event["pool"] else ""
    if kind == "load":
        return [kind, _name(plan, event["tensor"]), f"slow+{event['src_offset']}", f"fast+{event['offset']}",
                fmt_bytes(event["bytes"]), rows]
    if kind == "spill":
        return [kind, _name(plan, event["tensor"]), f"fast+{event['src_offset']}", f"slow+{event['offset']}",
                fmt_bytes(event["bytes"]), rows]
    if kind == "alloc":
        return [kind, _name(plan, event["tensor"]), "", where, fmt_bytes(event["bytes"]), ""]
    if kind == "move":
        return [kind, _name(plan, event["tensor"]), f"{event['pool']}+{event['src_offset']}", where,
                fmt_bytes(event["bytes"]), ""]
    if kind in ("weights", "copy"):
        return [kind, _name(plan, event["tensor"]), "", where, fmt_bytes(event["bytes"]), ""]
    if kind == "op":
        op = plan["ops"][event["op"]]
        return [kind, op["name"] if len(op["name"]) <= _NAME_WIDTH else op["name"][:_NAME_WIDTH - 3] + "...",
                "", "", "", ""]
    if kind == "reset":
        return [kind, "", "", f"fast+{event['fast_used']}", "", ""]
    return None


def render(model: str, plan: dict, events: list[dict], counters: dict[str, int], memory: dict[str, int],
           info: dict[str, str], verbose: bool) -> None:
    """Print the trace: totals, one row per execution unit, and with
    `verbose` every event of every tile."""
    done = units(plan, events)
    read = sum(unit["read"] for unit in done)
    written = sum(unit["written"] for unit in done)
    fast_peak = max((event["fast_used"] for event in events), default=0)
    text.echo(text.bold(model) + f"   traced on runtime {info['version']}, host reference backend, "
              f"{info['input']}")
    rows = [["moved", f"{fmt_bytes(written)} written, {fmt_bytes(read)} read"],
            ["fast peak", f"{fmt_bytes(fast_peak)} of {fmt_bytes(memory['fast_capacity_bytes'])}"]]
    weights = sum(unit["weights"] for unit in done)
    if weights:
        rows.append(["weights", f"{fmt_bytes(weights)} decompressed into fast memory"])
    for line in text.columns(rows):
        text.echo(line)

    header = ["stage", "kind", "tiles", "read", "written", "fast peak", "slow used"]
    table = [[text.dim(cell) for cell in header]]
    for unit in done:
        table.append([str(unit["stage"]) if unit["stage"] is not None else "", unit["kind"],
                      str(unit["tiles"]) if unit["tiles"] else "",
                      fmt_bytes(unit["read"]) if unit["read"] else "",
                      fmt_bytes(unit["written"]) if unit["written"] else "",
                      fmt_bytes(unit["fast_peak"]), fmt_bytes(unit["slow_used"])])
    text.gap()
    for line in text.columns(table, "><>>>>>", indent=0):
        text.echo(line)

    if verbose:
        for unit in done:
            text.section(f"stage {unit['stage']}   {unit['kind']}", [])
            rows = []
            for event in unit["events"]:
                if event["kind"] == "tile_begin":
                    ranges = _rows(event) if unit["path"] in _SPATIAL else ""
                    rows.append([text.dim(f"tile {event['op']}"), ranges, "", "", "", ""])
                    continue
                row = _event_row(plan, event, unit["path"] in _SPATIAL)
                if row is not None:
                    # Events sit under the tile they belong to.
                    rows.append(["  " + row[0], *row[1:]] if unit["tiles"] else row)
            for line in text.columns(rows, "<<<<>"):
                text.echo(line)

    problems = check(events, counters)
    text.gap()
    if problems:
        for problem in problems:
            text.echo(text.bad("mismatch: ") + problem)
    else:
        text.echo(text.dim(f"{len(events)} events; their bytes equal the runtime's own counters. "
                           f"Host alignment is {info['align']} bytes; a target with less uses at most "
                           "these bytes."))


def trace_model(model: str, ag, input_files: tuple[str, ...], verbose: bool) -> None:
    """Compile the analyzed graph in memory, run it once on the host runtime
    and print what the runtime did."""
    import tempfile
    from pathlib import Path

    import click
    import numpy as np

    from tigris.analysis.validation import (
        validate_budget, validate_execution_dtype, validate_operator_support)
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.binary.writer import emit_binary_bytes
    from tigris.runtime import RuntimeError as HostError
    from tigris.runtime import Session

    if ag.mem_budget <= 0:
        raise click.UsageError("--trace runs the plan, so it needs a fast-memory budget (-m)")
    usable = validate_execution_dtype(ag).supported and validate_operator_support(ag).supported
    result = validate_budget(ag) if usable else None
    if result is None or not result.fast.feasible:
        raise click.ClickException("no plan to trace: run analyze without --trace to see why")
    try:
        data = emit_binary_bytes(ag)
    except ValueError as exc:
        raise click.ClickException(f"no plan to trace: {exc}") from exc
    plan = read_binary_plan(data, decompress_weights=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "plan.tgrs"
        path.write_bytes(data)
        try:
            with Session(path) as session:
                if input_files:
                    from tigris.cli.run import read_inputs
                    inputs = read_inputs(session, input_files)
                    described = "given input"
                else:
                    inputs = {info["name"]: np.zeros(info["shape"], dtype=info["dtype"])
                              for info in session.inputs}
                    described = "zero input"
                _, events, counters = session.trace(inputs)
                memory = session.memory
                info = {"version": session.runtime_version, "input": described,
                        "align": counters.pop("tensor_align")}
        except (HostError, OSError, ValueError) as exc:
            raise click.ClickException(f"cannot trace: {exc}") from exc
    render(Path(model).name, plan, events, counters, memory, info, verbose)
