"""YAML plan emitter - human-readable execution plan."""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from tigris import SCHEMA_VERSION
from tigris.graph.ir import AnalyzedGraph

DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}


# Custom YAML formatting
# Short lists (shapes, tensor names, op indices) render inline:
#   shape: [1, 3, 224, 224]
# Everything else stays block style for readability.


class _Inline(list):
    """Marker: serialize this list in YAML flow style."""


class _PlanDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


_PlanDumper.add_representer(
    _Inline,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def _inline(seq) -> _Inline:
    return _Inline(seq)


# Public API


def _dtype_name(dtype: int) -> str:
    return DTYPE_NAMES.get(dtype, f"unknown({dtype})")


def emit_yaml(ag: AnalyzedGraph, path: Path) -> None:
    """Write an execution plan as YAML to *path*."""
    with open(path, "w") as f:
        f.write(_to_str(ag))


def emit_yaml_str(ag: AnalyzedGraph) -> str:
    """Return the execution plan as a YAML string."""
    return _to_str(ag)


def _to_str(ag: AnalyzedGraph) -> str:
    plan = _build_plan(ag)
    header = (
        f"# TiGrIS Execution Plan v{SCHEMA_VERSION}\n"
        f"# Model: {ag.model_name}\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
    )
    return header + yaml.dump(
        plan, Dumper=_PlanDumper, default_flow_style=False, sort_keys=False
    )


def _build_plan(ag: AnalyzedGraph) -> dict:
    plan: dict = {
        "version": SCHEMA_VERSION,
        "model": {"name": ag.model_name},
        "memory": {
            "peak_bytes": ag.peak_memory_bytes,
            "budget": ag.mem_budget,
        },
        "inputs": [_tensor_entry(ag, n) for n in ag.model_inputs],
        "outputs": [_tensor_entry(ag, n) for n in ag.model_outputs],
        "ops": [_op_entry(op, has_stages=bool(ag.stages)) for op in ag.ops],
    }

    if ag.stages:
        plan["stages"] = [_stage_entry(s) for s in ag.stages]

    return plan


def _stage_entry(s) -> dict:
    entry = {
        "id": s.stage_id,
        "ops": _inline(s.op_indices),
        "peak_bytes": s.peak_bytes,
        "inputs": _inline(s.input_tensors),
        "outputs": _inline(s.output_tensors),
        "warnings": _inline(s.warnings),
    }
    if s.tile_plan is not None:
        tp = s.tile_plan
        entry["tile_plan"] = {
            "tileable": tp.tileable,
            "axis": tp.axis,
            "tile_height": tp.tile_height,
            "num_tiles": tp.num_tiles,
            "halo": tp.halo,
            "receptive_field": tp.receptive_field,
            "original_height": tp.original_height,
            "tiled_peak_bytes": tp.tiled_peak_bytes,
            "overhead_bytes": tp.overhead_bytes,
        }
        if tp.untileable_ops:
            entry["tile_plan"]["untileable_ops"] = _inline(tp.untileable_ops)
        if tp.warnings:
            entry["tile_plan"]["warnings"] = _inline(tp.warnings)
    return entry


def _op_entry(op, *, has_stages: bool) -> dict:
    entry = {
        "name": op.name,
        "type": op.op_type,
        "inputs": _inline(op.inputs),
        "outputs": _inline(op.outputs),
    }
    if has_stages:
        entry["stage"] = op.stage
    # Spatial attributes (when present)
    for key in ("kernel_shape", "strides", "pads", "dilations"):
        val = op.attrs.get(key)
        if val:
            entry[key] = _inline(int(v) for v in val)
    if "group" in op.attrs and op.attrs["group"] != 1:
        entry["group"] = int(op.attrs["group"])
    return entry


def _tensor_entry(ag: AnalyzedGraph, name: str) -> dict:
    info = ag.tensors.get(name)
    if info is None:
        return {"name": name}
    return {
        "name": name,
        "shape": _inline(int(d) for d in info.shape),
        "dtype": _dtype_name(info.dtype),
        "size_bytes": info.size_bytes,
    }
