"""Inspect original ONNX graphs and compiled TiGrIS plans."""

import json
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tigris.cli import cli, console
from tigris.emitters.binary.defs import FLAG_XIP, STAGE_FLAG_LINE_BUFFERED
from tigris.inspection import inspect_file
from tigris.utils import fmt_bytes


def _panel(title, rows):
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    for label, value in rows:
        grid.add_row(Text(label), Text(str(value)))
    console.print(Panel(grid, title=Text(title, style="bold"), border_style="blue"))


def _shape(shape):
    return "unknown shape" if shape is None else "[" + ", ".join(
        "?" if dim is None else str(dim) for dim in shape) + "]"


def _interface(tensor, *, plan=False):
    dtype = tensor.get("interface_dtype", tensor.get("dtype", tensor.get("kind", "unknown")))
    value = f"{tensor['name']}  {dtype}"
    if plan:
        value += f"  (stored: {tensor['dtype']} {_shape(tensor['shape'])}, {tensor['layout']})"
    else:
        value += f" {_shape(tensor.get('shape'))}"
        if tensor.get("initializer_default"):
            value += "  (initializer default)"
    return value


def _names(indices, tensors):
    return ", ".join(tensors[index]["name"] for index in indices) or "none"


def _operators(ops, tensors=None):
    table = Table(box=None, padding=(0, 1), expand=True)
    for name in ("Step", "Operator", "Type", "Inputs", "Outputs"):
        table.add_column(name, overflow="fold")
    if tensors is not None:
        table.add_column("Fused activation")
    for index, op in ops:
        op_type = f"{op['domain']}::{op['type']}" if op.get("domain") else op["type"]
        values = [str(index), op["name"] or "(unnamed)", op_type,
                  _names(op["inputs"], tensors) if tensors is not None else ", ".join(op["inputs"]),
                  _names(op["outputs"], tensors) if tensors is not None else ", ".join(op["outputs"])]
        if tensors is not None:
            values.append(op["fused_activation"])
        table.add_row(*(Text(value, style="cyan" if column == 1 else "")
                        for column, value in enumerate(values)))
    console.print(table)


def _value(value):
    if value is None:
        return "none"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_value(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return str(value)


def _details(title, value):
    if isinstance(value, dict):
        _panel(title, [(key.replace("_", " ").capitalize(), _value(item))
                       for key, item in value.items()])
    else:
        _panel(title, [(str(index), _value(item)) for index, item in enumerate(value)])


def _tensors(title, tensors, *, plan=False):
    table = Table(box=None, padding=(0, 1), expand=True)
    columns = ["ID", "Tensor", "Type", "Stored shape" if plan else "Declared shape"]
    columns += ["Bytes", "Layout", "Quant"] if plan else ["Bytes", "Storage"]
    for label in columns:
        table.add_column(label, overflow="fold")
    for index, tensor in enumerate(tensors):
        values = [str(index), tensor["name"], tensor.get("dtype", tensor.get("kind", "unknown")),
                  _shape(tensor.get("shape")),
                  fmt_bytes(tensor["size_bytes"]) if tensor.get("size_bytes") is not None else "unknown"]
        if plan:
            values += [tensor["layout"], str(tensor["quant_param_idx"]) if tensor["quant_param_idx"] is not None else "none"]
        else:
            values.append(tensor.get("storage", "not declared"))
        table.add_row(*(Text(value, style="cyan" if column == 1 else "")
                        for column, value in enumerate(values)))
    console.print(Panel(table, title=Text(title, style="bold"), border_style="blue"))


def _render(report, verbose):
    is_plan = report["format"] == "tgrs"
    rows = [("Format", "TiGrIS execution plan" if is_plan else "ONNX graph"),
            ("File size", fmt_bytes(report["file_size_bytes"]))]
    if is_plan:
        plan = report["plan"]
        tensors = plan["tensors"]
        rows.append(("Schema", plan["version"]))
        inputs = [tensors[index] for index in plan["model_inputs"]]
        outputs = [tensors[index] for index in plan["model_outputs"]]
    else:
        graph = report["graph"]
        defaults = {item["name"] for item in graph["initializers"] + graph["sparse_initializers"]}
        inputs = [{**item, "initializer_default": item["name"] in defaults} for item in graph["inputs"]]
        outputs = graph["outputs"]
        rows.extend([("IR version", report["ir_version"]),
                     ("Opsets", ", ".join(f"{item['domain'] or 'ai.onnx'}: {item['version']}"
                                          for item in report["opsets"]))])
    for label, items in (("Input", inputs), ("Output", outputs)):
        for item in items:
            rows.append((label, _interface(item, plan=is_plan)))
    counts = report["operator_counts"]
    rows.append(("Operators", f"{sum(counts.values())}: " + ", ".join(
        f"{name} x {count}" if count > 1 else name for name, count in counts.items())))
    if is_plan:
        tiled = sum(stage["tile_plan_idx"] is not None and
                    plan["tile_plans"][stage["tile_plan_idx"]]["tileable"] for stage in plan["stages"])
        chains = sum(stage["chain_len"] >= 2 and stage["chain_id"] == index
                     for index, stage in enumerate(plan["stages"]))
        count = len(plan["stages"])
        rows.append(("Schedule", f"{count} {'stage' if count == 1 else 'stages'}, {tiled} tiled, {chains} chains"))
    else:
        initializers = graph["initializers"]
        external = sum(item["storage"] == "external" for item in initializers)
        rows.append(("Initializers", f"{len(initializers)} dense, {len(graph['sparse_initializers'])} sparse"
                     f", {external} external dense"))
    _panel(f"TiGrIS Inspect - {report['name'] or '(unnamed)'}", rows)
    if is_plan:
        compression = "LZ4" if plan["weight_blocks_compression"] == 1 else "uncompressed"
        storage = "Read in place (XIP)" if plan["flags"] & FLAG_XIP else "Loaded by stage"
        _panel("Memory", [
            ("Fast budget", fmt_bytes(plan["budget"])),
            ("Recorded graph peak", fmt_bytes(plan["peak"])),
            ("Weights (including bias)", fmt_bytes(sum(weight["size_bytes"] for weight in plan["weights"]))),
            ("Weight storage", f"{storage}, {compression}"),
        ])
        console.print(Text("Memory values are compiler records, not measured runtime peak or total RAM.", style="dim"))
    else:
        console.print(Text("Declared graph metadata. Shapes are not inferred; external tensor data is not loaded.", style="dim"))
    if not verbose:
        return
    if not is_plan:
        _operators(enumerate(graph["operators"]))
        for index, op in enumerate(graph["operators"]):
            if op["attributes"]:
                _details(f"Operator {index} attributes", op["attributes"])
        for label, key in (("Declared intermediate tensors", "value_info"), ("Initializers", "initializers")):
            if graph[key]:
                _tensors(label, graph[key])
        for tensor in graph["initializers"]:
            if tensor["external_data"]:
                _details(f"External data - {tensor['name']}", tensor["external_data"])
        if graph["sparse_initializers"]:
            _details("Sparse initializers", graph["sparse_initializers"])
        if report["functions"]:
            _details("Local functions", report["functions"])
        return
    for index, stage in enumerate(plan["stages"]):
        rows = [("Inputs", _names(stage["inputs"], tensors)),
                ("Outputs", _names(stage["outputs"], tensors)),
                ("Recorded peak", fmt_bytes(stage["peak_bytes"]))]
        tile_index = stage["tile_plan_idx"]
        if tile_index is not None:
            tile = plan["tile_plans"][tile_index]
            if tile["tileable"]:
                rows.append(("Tiling", f"{tile['num_tiles']} tiles; axis {tile['axis']}; "
                             f"height {tile['tile_height']}, width {tile['tile_width']}; halo {tile['halo']}"))
        if stage["chain_id"] is not None and stage["chain_len"] >= 2:
            head = plan["stages"][stage["chain_id"]]
            mode = "line buffered" if head["flags"] & STAGE_FLAG_LINE_BUFFERED else "tile buffered"
            rows.append(("Chain", f"head {stage['chain_id']}, {stage['chain_len']} stages, "
                         f"{mode}, tile height {head['chain_tile_h']}"))
        _panel(f"Stage {index}", rows)
        _operators(((op_index, plan["ops"][op_index]) for op_index in stage["ops"]), tensors)
    if not plan["stages"]:
        _operators(enumerate(plan["ops"]), tensors)
    for index, op in enumerate(plan["ops"]):
        details = {}
        if op["spatial"]["kernel_h"]:
            details["spatial"] = op["spatial"]
        if op["attributes"]:
            details["attributes"] = op["attributes"]
        if any(tensors[item]["dtype"] == "int8" for item in op["outputs"]):
            details["int8_activation_clamp"] = [op["act_min"], op["act_max"]]
        for key in ("weight_idx", "bias_idx"):
            if op[key] is not None:
                details[key.removesuffix("_idx")] = plan["weights"][op[key]]["name"]
        if details:
            _details(f"Operator {index} parameters", details)
    _tensors("Stored tensors", tensors, plan=True)
    for label, key in (("Weights", "weights"),
                       ("Quantization", "quant_params"), ("Tile plans", "tile_plans"),
                       ("Weight blocks", "weight_blocks")):
        if plan[key]:
            _details(label, plan[key])


@cli.command("inspect")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Show operators, tensors, and execution-plan details.")
@click.option("--json", "as_json", is_flag=True, help="Emit complete, versioned metadata as JSON.")
def inspect(model: Path, verbose: bool, as_json: bool):
    """Inspect an ONNX graph or .tgrs plan without compiling or executing it.

    Format is detected from file contents. Inspection is offline and does not
    load external ONNX tensor data. Memory values are compiler records.
    Inspection does not execute kernels; run checks the portable reference
    path, not ESP-NN or CMSIS-NN
    numerics or on-device latency.
    """
    try:
        report = inspect_file(model)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    else:
        _render(report, verbose)
