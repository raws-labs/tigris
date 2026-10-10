"""Inspect original ONNX graphs and compiled TiGrIS plans."""

import json
from pathlib import Path

import click
from tigris.cli import cli, text
from tigris.emitters.binary.defs import FLAG_XIP, STAGE_FLAG_LINE_BUFFERED
from tigris.inspection import inspect_file
from tigris.utils import fmt_bytes


def _panel(title, rows):
    text.section(title, [[label[:1].lower() + label[1:], str(value)] for label, value in rows])


def _shape(shape):
    if shape is None:
        return "unknown shape"
    return "x".join("?" if dim is None else str(dim) for dim in shape) or "scalar"


def _interface(tensor, *, plan=False):
    """Name, then shape and dtype, then a dim note on storage."""
    dtype = tensor.get("interface_dtype", tensor.get("dtype", tensor.get("kind", "unknown")))
    note = ""
    if plan:
        note = f"stored {tensor['dtype']}, {tensor['layout']}"
    elif tensor.get("initializer_default"):
        note = "initializer default"
    return [tensor["name"], f"{_shape(tensor.get('shape'))} {dtype}", text.dim(note)]


def _names(indices, tensors):
    return ", ".join(tensors[index]["name"] for index in indices) or "none"


def _operators(ops, tensors=None, title="steps"):
    header = ["step", "operator", "type", "inputs", "outputs"]
    if tensors is not None:
        header.append("fused activation")
    rows = [[text.dim(cell) for cell in header]]
    for index, op in ops:
        op_type = f"{op['domain']}::{op['type']}" if op.get("domain") else op["type"]
        values = [str(index), op["name"] or "(unnamed)", op_type,
                  _names(op["inputs"], tensors) if tensors is not None else ", ".join(op["inputs"]),
                  _names(op["outputs"], tensors) if tensors is not None else ", ".join(op["outputs"])]
        if tensors is not None:
            values.append(op["fused_activation"])
        rows.append(values)
    text.section(title, rows, ">")


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
    columns = ["id", "tensor", "type", "stored shape" if plan else "declared shape"]
    columns += ["bytes", "layout", "quant"] if plan else ["bytes", "storage"]
    rows = [[text.dim(cell) for cell in columns]]
    for index, tensor in enumerate(tensors):
        values = [str(index), tensor["name"], tensor.get("dtype", tensor.get("kind", "unknown")),
                  _shape(tensor.get("shape")),
                  fmt_bytes(tensor["size_bytes"]) if tensor.get("size_bytes") is not None else "unknown"]
        if plan:
            values += [tensor["layout"], str(tensor["quant_param_idx"]) if tensor["quant_param_idx"] is not None else "none"]
        else:
            values.append(tensor.get("storage", "not declared"))
        rows.append(values)
    text.section(title, rows, ">   >")


_FORMATS = {"tgrs": "plan", "onnx": "ONNX", "tflite": "TFLite"}


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _render(report, verbose, path: Path):
    is_plan = report["format"] == "tgrs"
    is_tflite = report["format"] == "tflite"
    counts = report["operator_counts"]
    summary = [_FORMATS[report["format"]]]
    if is_plan:
        plan = report["plan"]
        tensors = plan["tensors"]
        summary.append(f"schema {plan['version']}")
        inputs = [tensors[index] for index in plan["model_inputs"]]
        outputs = [tensors[index] for index in plan["model_outputs"]]
    else:
        graph = report["graph"]
        defaults = {item["name"] for item in graph["initializers"] + graph["sparse_initializers"]}
        inputs = [{**item, "initializer_default": item["name"] in defaults} for item in graph["inputs"]]
        outputs = graph["outputs"]
        if is_tflite:
            summary.append(f"schema {report['tflite_version']}")
        else:
            summary.append(f"IR {report['ir_version']}, opsets " + ", ".join(
                f"{item['domain'] or 'ai.onnx'} {item['version']}" for item in report["opsets"]))
    summary.append(_count(sum(counts.values()), "operator"))
    if is_tflite and report["subgraphs"] > 1:
        summary.append(_count(report["subgraphs"], "subgraph"))
    if is_plan:
        stages = plan["stages"]
        # A stage runs tiled when it has a tile plan or belongs to a chain.
        tiled = sum((stage["tile_plan_idx"] is not None and
                     plan["tile_plans"][stage["tile_plan_idx"]]["tileable"]) or
                    (stage["chain_id"] is not None and stage["chain_len"] >= 2) for stage in stages)
        summary.append(_count(len(stages), "stage") + (f", {tiled} tiled" if tiled else ""))
    summary.append(fmt_bytes(report["file_size_bytes"]))
    # The name stored inside the file, where it is not the file's own.
    if report.get("name") and report["name"] != path.stem:
        summary.insert(1, f"model {report['name']}")
    text.echo(text.bold(path.name) + "   " + ", ".join(summary))
    rows = [[label, *_interface(item, plan=is_plan)]
            for label, items in (("input", inputs), ("output", outputs)) for item in items]
    for line in text.columns(rows):
        text.echo(line)

    if is_plan:
        _plan_summary(plan, report["requirements"])
    else:
        _model_summary(report, is_tflite)
    if not verbose:
        return
    if is_plan:
        _plan_details(plan)
        return
    _steps(graph, report["costs"])
    if graph["initializers"]:
        _tensors("constants", graph["initializers"])
    if is_tflite:
        return
    for index, op in enumerate(graph["operators"]):
        if op["attributes"]:
            _details(f"operator {index} attributes", op["attributes"])
    if graph["value_info"]:
        _tensors("declared intermediate tensors", graph["value_info"])
    for tensor in graph["initializers"]:
        if tensor["external_data"]:
            _details(f"external data {tensor['name']}", tensor["external_data"])
    if graph["sparse_initializers"]:
        _details("sparse initializers", graph["sparse_initializers"])
    if report["functions"]:
        _details("local functions", report["functions"])


def _macs(n):
    if n is None:
        return "?"
    for limit, unit in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if n >= limit:
            return f"{n / limit:.2f} {unit}"
    return str(n)


def _share(part, whole):
    if not part or not whole:
        return ""
    share = 100 * part / whole
    return "<1%" if share < 0.5 else f"{share:.0f}%"


def _cost_rows(kinds, weight_total, macs_total, *, with_macs):
    """One row per operator type, heaviest first, and a total."""
    header = ["", "count", "weights", ""] + (["MACs", ""] if with_macs else [])
    rows = [[text.dim(cell) for cell in header]]
    for kind in kinds:
        row = [kind["type"], str(kind["count"]),
               fmt_bytes(kind["weight_bytes"]) if kind["weight_bytes"] else "",
               _share(kind["weight_bytes"], weight_total)]
        if with_macs:
            row += [_macs(kind["macs"]) if kind["macs"] != 0 else "", _share(kind["macs"], macs_total)]
        rows.append(row)
    total = [text.dim("total"), str(sum(kind["count"] for kind in kinds)),
             fmt_bytes(weight_total), ""]
    if with_macs:
        total += [_macs(macs_total), ""]
    rows.append(total)
    return rows


def _model_summary(report, is_tflite):
    costs = report["costs"]
    text.section("operators", _cost_rows(costs["operators"], costs["weight_bytes"], costs["macs"],
                                         with_macs=True), "<>>>>>")
    largest = costs["largest_activations"]
    if largest:
        text.section("largest activations", [
            [f"{_shape(item['shape'])} {item['dtype']}", fmt_bytes(item["bytes"]),
             "model input" if item["step"] is None else f"step {item['step']}, {item['type']}"]
            for item in largest], "<>")
    rows = []
    if is_tflite:
        rows.append(["quantization", _quantization(report["quantization"])])
        reasons = report["unsupported"]
        rows.append(["converts", "yes" if not reasons else
                     text.bad("no") + f", {_count(len(reasons), 'reason')}"])
        rows += [["", reason] for reason in reasons[:5]]
        if len(reasons) > 5:
            rows.append(["", text.dim(f"and {len(reasons) - 5} more; --json lists all")])
    else:
        graph = report["graph"]
        external = sum(item["storage"] == "external" for item in graph["initializers"])
        rows.append(["initializers", f"{len(graph['initializers'])} dense, "
                     f"{len(graph['sparse_initializers'])} sparse, {external} external"])
    text.block(rows)
    if not is_tflite:
        text.gap()
        text.echo(text.dim("Declared metadata: shapes are not inferred, external data is not loaded; "
                           "? marks a figure an undeclared shape hides."))


def _quantization(quant):
    parts = [", ".join(quant["activation_dtypes"]) + " activations"]
    weights = quant["int8_weights"]
    if weights:
        per_channel = quant["per_channel"]
        if per_channel == weights:
            layout = "per channel"
        elif per_channel == 0:
            layout = "per tensor"
        else:
            layout = f"{per_channel} of {weights} per channel"
        symmetry = ("symmetric" if not quant["asymmetric"]
                    else f"{quant['asymmetric']} asymmetric")
        parts.append(f"int8 weights {layout}, {symmetry}")
    if quant["float_islands"]:
        parts.append(_count(quant["float_islands"], "float32 region"))
    return "; ".join(parts)


def _plan_kinds(plan):
    kinds = {}
    for op in plan["ops"]:
        entry = kinds.setdefault(op["type"], {"type": op["type"], "count": 0, "weight_bytes": 0})
        entry["count"] += 1
        entry["weight_bytes"] += sum(plan["weights"][op[key]]["size_bytes"]
                                     for key in ("weight_idx", "bias_idx") if op[key] is not None)
    return sorted(kinds.values(), key=lambda item: (-item["weight_bytes"], -item["count"], item["type"]))


def _plan_summary(plan, needs):
    weights = sum(weight["size_bytes"] for weight in plan["weights"])
    text.section("operators", _cost_rows(_plan_kinds(plan), weights, None, with_macs=False), "<>>>")
    compression = "LZ4" if plan["weight_blocks_compression"] == 1 else "uncompressed"
    storage = "read in place (XIP)" if plan["flags"] & FLAG_XIP else "loaded by stage"
    arena_note = ""
    if needs["decompression_bytes"]:
        arena_note = (f"{fmt_bytes(plan['budget'])} activations, "
                      f"{fmt_bytes(needs['decompression_bytes'])} decompressed weights")
    rows = [["fast arena", fmt_bytes(needs["fast_arena_bytes"]), text.dim(arena_note)],
            ["unscheduled", fmt_bytes(plan["peak"]), ""],
            ["weights", fmt_bytes(weights), text.dim(f"{storage}, {compression}")]]
    if needs["state_bytes"]:
        rows.append(["state", fmt_bytes(needs["state_bytes"]), text.dim("kept between runs")])
    text.section("memory", rows, "<>")
    over = [limit for limit in needs["build_limits"] if limit["needed"] > limit["default"]]
    text.block([["runtime build", "default limits suffice" if not over else
                 ", ".join(f"-D{limit['name']}={limit['needed']}" for limit in over)]])


def _tiling(plan, index, stage):
    if stage["chain_id"] is not None and stage["chain_len"] >= 2:
        head = plan["stages"][stage["chain_id"]]
        if stage["chain_id"] != index:
            return f"in chain {stage['chain_id']}"
        mode = "line buffered" if head["flags"] & STAGE_FLAG_LINE_BUFFERED else "tile buffered"
        return f"chain of {stage['chain_len']}, tile height {head['chain_tile_h']}, {mode}"
    if stage["tile_plan_idx"] is not None:
        tile = plan["tile_plans"][stage["tile_plan_idx"]]
        if tile["tileable"]:
            return f"{tile['num_tiles']} tiles, axis {tile['axis']}, halo {tile['halo']}"
    return "untiled"


def _plan_details(plan):
    tensors = plan["tensors"]
    rows = [[text.dim(cell) for cell in ("stage", "ops", "untiled peak", "weights", "tiling")]]
    for index, stage in enumerate(plan["stages"]):
        weights = sum(plan["weights"][op[key]]["size_bytes"] for op in (plan["ops"][i] for i in stage["ops"])
                      for key in ("weight_idx", "bias_idx") if op[key] is not None)
        first, last = stage["ops"][0], stage["ops"][-1]
        rows.append([str(index), str(first) if first == last else f"{first}-{last}",
                     fmt_bytes(stage["peak_bytes"]), fmt_bytes(weights) if weights else "",
                     _tiling(plan, index, stage)])
    if plan["stages"]:
        text.section("stages", rows, ">>>>")
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
            _details(f"operator {index} parameters", details)
    _tensors("stored tensors", tensors, plan=True)
    for label, key in (("weights", "weights"), ("quantization", "quant_params"),
                       ("tile plans", "tile_plans"), ("weight blocks", "weight_blocks")):
        if plan[key]:
            _details(label, plan[key])


def _steps(graph, costs):
    """Every operator with its output, the weights it reads first, and its MACs."""
    shapes = {item["name"]: item for item in graph.get("activations", []) + graph.get("value_info", [])
              + graph["inputs"] + graph["outputs"]}
    rows = [[text.dim(cell) for cell in ("step", "type", "output", "bytes", "weights", "MACs", "name")]]
    for op, step in zip(graph["operators"], costs["steps"]):
        out = shapes.get(op["outputs"][0]) if op["outputs"] else None
        shape = f"{_shape(out.get('shape'))} {out.get('dtype', '')}".strip() if out else "?"
        size = (text.dim("constant") if step["constant"] else
                fmt_bytes(step["output_bytes"]) if step["output_bytes"] is not None else "?")
        rows.append([str(step["step"]), op["type"], shape, size,
                     fmt_bytes(step["weight_bytes"]) if step["weight_bytes"] else "",
                     _macs(step["macs"]) if step["macs"] != 0 else "", op["name"] or ""])
    text.section("steps", rows, "><<>>>")


@cli.command("inspect")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Show operators, tensors, and execution-plan details.")
@click.option("--json", "as_json", is_flag=True, help="Emit complete, versioned metadata as JSON.")
def inspect(model: Path, verbose: bool, as_json: bool):
    """Show what a model or plan contains.

    Offline: external ONNX tensor data is not loaded. Memory values are the
    compiler's records, not measurements.
    """
    try:
        report = inspect_file(model)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    else:
        _render(report, verbose, model)
