"""The execution trace: what the runtime did when it ran a plan, event by event."""

from tigris.cli import text
from tigris.runtime import check_trace, trace_units
from tigris.utils import fmt_bytes

# Paths whose event ranges are rows and columns of the tensor; the others
# report ranges in their own iteration order.
_SPATIAL = {"tiled", "tiled_2d", "chain"}
_NAME_WIDTH = 32
# Zero inputs can keep such a loop running forever.
_ZERO_INPUT_LOOP = "the number of While iterations depends on the input values"


def _shorten(name: str) -> str:
    return name if len(name) <= _NAME_WIDTH else name[:_NAME_WIDTH - 3] + "..."


def _name(plan: dict, index) -> str:
    return "" if index is None else _shorten(plan["tensors"][index]["name"])


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
        return [kind, _shorten(op["name"]), "", "", "", ""]
    if kind == "reset":
        return [kind, "", "", f"fast+{event['fast_used']}", "", ""]
    return None


def render(model: str, plan: dict, events: list[dict], counters: dict[str, int], memory: dict[str, int],
           info: dict[str, str], verbose: bool) -> None:
    """Print the trace: totals, one row per execution unit, and with
    `verbose` every event of every tile."""
    done = trace_units(plan, events)
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

    problems = check_trace(events, counters)
    text.gap()
    if problems:
        for problem in problems:
            text.echo(text.bad("mismatch: ") + problem)
    else:
        text.echo(text.dim(f"{len(events)} events; their bytes equal the runtime's own counters. "
                           f"Host alignment is {info['align']} bytes; a target with less uses at most "
                           "these bytes."))


def _plan_bytes(ag) -> bytes | None:
    """The plan with names, or None when it is unusable or does not fit the fast and slow budgets."""
    from tigris.analysis.validation import plan_gate
    from tigris.emitters.binary.writer import emit_binary_bytes

    if ag.mem_budget <= 0 or not plan_gate(ag).passes:
        return None
    try:
        # The trace names every operator and tensor it reports.
        return emit_binary_bytes(ag, names=True)
    except ValueError:
        return None


def _run(data: bytes, input_files: tuple[str, ...], traced: bool):
    """Run a plan once on the host runtime with zeros or the given inputs.
    Returns the events (empty unless traced), counters, memory and run facts."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from tigris.runtime import Session

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "plan.tgrs"
        path.write_bytes(data)
        with Session(path) as session:
            if input_files:
                from tigris.cli.run import read_inputs
                inputs = read_inputs(session, input_files)
                described = "given input"
            else:
                inputs = {info["name"]: np.zeros(info["shape"], dtype=info["dtype"])
                          for info in session.inputs}
                described = "zero input"
            if traced:
                _, events, counters = session.trace(inputs)
            else:
                session.run(inputs)
                events, counters = [], session.counters
            info = {"version": session.runtime_version, "input": described,
                    "align": counters.pop("tensor_align")}
            return events, counters, session.memory, info


def traffic(ag) -> dict[str, int] | str | None:
    """Bytes one inference writes to and reads from slow memory, measured on
    the host runtime: {"written", "read"}, a reason it was not measured, or
    None when there is no plan at this budget."""
    from tigris.graph.subgraphs import input_dependent_loop
    from tigris.runtime import RuntimeError as HostError
    from tigris.runtime import runtime_info

    data = _plan_bytes(ag)
    if data is None:
        return None
    if input_dependent_loop(ag):
        return _ZERO_INPUT_LOOP
    try:
        runtime_info()
    except HostError:
        return "no host runtime in this install"
    try:
        _, counters, _, _ = _run(data, (), traced=False)
    except (HostError, OSError, ValueError) as exc:
        return str(exc)
    return {"written": counters["spill_bytes"], "read": counters["load_bytes"]}


def trace_model(model: str, ag, input_files: tuple[str, ...], verbose: bool) -> None:
    """Compile the analyzed graph in memory, run it once on the host runtime
    and print what the runtime did."""
    from pathlib import Path

    import click

    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.graph.subgraphs import input_dependent_loop
    from tigris.runtime import RuntimeError as HostError

    if ag.mem_budget <= 0:
        raise click.UsageError("--trace runs the plan, so it needs a fast-memory budget (-m)")
    data = _plan_bytes(ag)
    if data is None:
        raise click.ClickException("no plan to trace: run analyze without --trace to see why")
    if not input_files and input_dependent_loop(ag):
        raise click.UsageError(f"--trace needs --input here: {_ZERO_INPUT_LOOP}")
    plan = read_binary_plan(data, decompress_weights=False)
    try:
        events, counters, memory, info = _run(data, input_files, traced=True)
    except (HostError, OSError, ValueError) as exc:
        raise click.ClickException(f"cannot trace: {exc}") from exc
    render(Path(model).name, plan, events, counters, memory, info, verbose)
