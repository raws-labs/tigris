"""Plan emitter for the deprecated ``tigris plan``.

It writes JSON, which every YAML parser reads, so the command keeps working
without a YAML dependency until it is removed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

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


def _inline(seq) -> list:
    return list(seq)


def _dtype_name(dtype: int) -> str:
    return DTYPE_NAMES.get(dtype, f"unknown({dtype})")


def emit_plan_json(ag: AnalyzedGraph, path: Path) -> None:
    """Write an execution plan to *path*."""
    with open(path, "w") as f:
        f.write(_to_str(ag))


def plan_json_str(ag: AnalyzedGraph) -> str:
    """Return the execution plan as JSON text."""
    return _to_str(ag)


def _to_str(ag: AnalyzedGraph) -> str:
    plan = {"generated": datetime.now(timezone.utc).isoformat(), **_build_plan(ag)}
    return json.dumps(plan, indent=2) + "\n"


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
