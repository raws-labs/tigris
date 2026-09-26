"""Read model and execution-plan metadata without compiling or executing it."""

import math
import re
import struct
from collections import Counter
from pathlib import Path

import onnx
from google.protobuf.message import DecodeError

from tigris import SCHEMA_VERSION_INTERFACE_DTYPE, SCHEMA_VERSION_TENSOR_LAYOUT
from tigris.emitters.binary import defs
from tigris.emitters.binary.reader import read_binary_plan


def _dtype(code: int) -> str:
    try:
        name = onnx.TensorProto.DataType.Name(code).lower()
        return {"float": "float32", "double": "float64"}.get(name, name)
    except ValueError:
        return f"unknown({code})"


def _type(proto) -> dict:
    kind = proto.WhichOneof("value")
    if kind in ("tensor_type", "sparse_tensor_type"):
        tensor = getattr(proto, kind)
        shape = None
        if tensor.HasField("shape"):
            shape = [
                dim.dim_value if dim.HasField("dim_value") else
                dim.dim_param if dim.HasField("dim_param") else None
                for dim in tensor.shape.dim
            ]
        return {"kind": kind, "dtype": _dtype(tensor.elem_type), "shape": shape}
    if kind in ("sequence_type", "optional_type"):
        return {"kind": kind, "element": _type(getattr(proto, kind).elem_type)}
    if kind == "map_type":
        return {"kind": kind, "key": _dtype(proto.map_type.key_type),
                "value": _type(proto.map_type.value_type)}
    return {"kind": kind or "unknown"}


def _tensor(proto) -> dict:
    dtype = _dtype(proto.data_type)
    match = re.match(r"(?:u?int|float|bfloat|complex)(\d+)", dtype)
    bits = int(match[1]) if match else 8 if dtype == "bool" else None
    return {
        "name": proto.name, "dtype": dtype, "shape": list(proto.dims),
        "size_bytes": (math.prod(proto.dims) * bits + 7) // 8 if bits else None,
        "storage": "external" if proto.data_location == onnx.TensorProto.EXTERNAL else "embedded",
        "external_data": {item.key: item.value for item in proto.external_data},
    }


def _sparse_tensor(proto) -> dict:
    return {"name": proto.values.name, "shape": list(proto.dims),
            "values": _tensor(proto.values), "indices": _tensor(proto.indices)}


def _attribute(proto) -> dict:
    kind = onnx.AttributeProto.AttributeType.Name(proto.type)
    result = {"name": proto.name, "type": kind, "reference": proto.ref_attr_name or None}
    fields = {"FLOAT": "f", "INT": "i", "STRING": "s", "TENSOR": "t",
              "GRAPH": "g", "SPARSE_TENSOR": "sparse_tensor", "TYPE_PROTO": "tp",
              "FLOATS": "floats", "INTS": "ints", "STRINGS": "strings",
              "TENSORS": "tensors", "GRAPHS": "graphs",
              "SPARSE_TENSORS": "sparse_tensors", "TYPE_PROTOS": "type_protos"}
    converters = {"TENSOR": _tensor, "GRAPH": _graph, "SPARSE_TENSOR": _sparse_tensor,
                  "TYPE_PROTO": _type, "STRING": lambda value: value.decode("utf-8", "replace")}
    if kind in fields and not proto.ref_attr_name:
        convert = converters.get(kind.rstrip("S"), lambda value: value)
        value = getattr(proto, fields[kind])
        result["value"] = [convert(item) for item in value] if kind.endswith("S") else convert(value)
    return result


def _node(proto) -> dict:
    return {"name": proto.name, "type": proto.op_type, "domain": proto.domain,
            "inputs": list(proto.input), "outputs": list(proto.output),
            "attributes": [_attribute(attr) for attr in proto.attribute]}


def _graph(proto) -> dict:
    def value(info):
        return {"name": info.name, **_type(info.type)}

    return {
        "name": proto.name, "inputs": [value(info) for info in proto.input],
        "outputs": [value(info) for info in proto.output],
        "value_info": [value(info) for info in proto.value_info],
        "initializers": [_tensor(tensor) for tensor in proto.initializer],
        "sparse_initializers": [_sparse_tensor(tensor) for tensor in proto.sparse_initializer],
        "operators": [_node(node) for node in proto.node],
    }


def _plan_attribute(attr: dict) -> dict:
    kinds = {
        defs.OP_ATTR_TRANSPOSE_PERM: ("transpose_perm", "B"),
        defs.OP_ATTR_EPSILON: ("epsilon", "f"),
        defs.OP_ATTR_ALPHA: ("alpha", "f"),
        defs.OP_ATTR_CLIP_BOUNDS: ("clip_bounds", "f"),
        defs.OP_ATTR_PADS: ("pads", "i"),
        defs.OP_ATTR_AXES: ("axes", "B"),
        defs.OP_ATTR_BINARY_REQUANT: ("binary_requant", "i"),
        defs.OP_ATTR_POOL_ROUNDING: ("pool_rounding", "B"),
    }
    name, code = kinds.get(attr["type"], (f"unknown({attr['type']})", "B"))
    size = struct.calcsize(code)
    if len(attr["data"]) % size:
        raise ValueError(f"Invalid {name} attribute size")
    values = list(struct.unpack(f"<{len(attr['data']) // size}{code}", attr["data"]))
    return {"type": name, "values": values}


def _plan(data: bytes) -> dict:
    plan = read_binary_plan(data, decompress_weights=False)

    def reference(index, key, *, optional=False):
        if optional and index == 65535:
            return None
        if index >= len(plan[key]):
            raise ValueError(f"Invalid {key} reference: {index}")
        return index

    for key in ("model_inputs", "model_outputs"):
        for index in plan[key]:
            reference(index, "tensors")
    for tensor in plan["tensors"]:
        tensor["quant_param_idx"] = reference(tensor["quant_param_idx"], "quant_params", optional=True)
        tensor["dtype"] = _dtype(tensor["dtype"])
        tensor["interface_dtype"] = (
            _dtype(tensor["iface_dtype"]) if tensor["iface_dtype"] and
            plan["version"] >= SCHEMA_VERSION_INTERFACE_DTYPE else tensor["dtype"]
        )
        tensor["layout"] = "unspecified"
        if plan["version"] >= SCHEMA_VERSION_TENSOR_LAYOUT:
            tensor["layout"] = (
                "model order" if tensor["flags"] & defs.TENSOR_FLAG_LINEAR else
                {4: "NHWC", 3: "NLC"}.get(tensor["ndim"], "model order")
            )
    op_names = {code: name for name, code in defs.OP_TYPE_MAP.items()}
    for op in plan["ops"]:
        op["type"] = op_names.get(op["op_type"], f"unknown({op['op_type']})")
        op["fused_activation"] = {0: "none", 1: "Relu", 2: "Relu6"}.get(
            op["fused_act"], f"unknown({op['fused_act']})")
        for index in op["inputs"] + op["outputs"]:
            reference(index, "tensors")
        for key in ("weight_idx", "bias_idx"):
            op[key] = reference(op[key], "weights", optional=True)
        op["attributes"] = []
    for attr in plan["op_attributes"]:
        index = reference(attr["op_index"], "ops")
        plan["ops"][index]["attributes"].append(_plan_attribute(attr))
    assigned = set()
    for stage_index, stage in enumerate(plan["stages"]):
        for index in stage["ops"]:
            reference(index, "ops")
            if index in assigned:
                raise ValueError(f"Operator {index} belongs to multiple stages")
            assigned.add(index)
        for index in stage["inputs"] + stage["outputs"]:
            reference(index, "tensors")
        stage["tile_plan_idx"] = reference(stage["tile_plan_idx"], "tile_plans", optional=True)
        stage["chain_id"] = reference(stage["chain_id"], "stages", optional=True)
        if stage["chain_id"] is not None and stage["chain_id"] + stage["chain_len"] > len(plan["stages"]):
            raise ValueError("Chain extends beyond stage table")
        if stage["chain_len"]:
            head = stage["chain_id"]
            if head is None or not head <= stage_index < head + stage["chain_len"]:
                raise ValueError("Invalid chain membership")
            if plan["stages"][head]["chain_len"] != stage["chain_len"]:
                raise ValueError("Inconsistent chain length")
    if plan["stages"] and len(assigned) != len(plan["ops"]):
        raise ValueError("Stage table does not cover all operators")
    for block in plan["weight_blocks"]:
        reference(block["stage_idx"], "stages")
        if block["first_weight_idx"] + block["num_weights"] > len(plan["weights"]):
            raise ValueError("Weight block extends beyond weight table")
    plan.pop("magic")
    plan.pop("op_attributes")
    return plan


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def inspect_file(path: str | Path) -> dict:
    """Return JSON-safe metadata; never load external ONNX tensor data."""
    data = Path(path).read_bytes()
    report = {"inspection_version": 1, "file_size_bytes": len(data)}
    if data.startswith(defs.MAGIC):
        try:
            plan = _plan(data)
        except (ValueError, KeyError, IndexError, struct.error) as exc:
            raise ValueError(f"Invalid TiGrIS plan: {exc}") from exc
        report.update(format="tgrs", name=plan["model_name"], plan=plan)
        operators = plan["ops"]
    else:
        try:
            model = onnx.load_model_from_string(data)
        except DecodeError as exc:
            raise ValueError("Not a TiGrIS plan or a readable ONNX model") from exc
        if not model.HasField("graph") or model.ir_version <= 0:
            raise ValueError("Not a TiGrIS plan or a readable ONNX model")
        graph = _graph(model.graph)
        report.update(
            format="onnx", name=graph["name"], ir_version=model.ir_version,
            producer=model.producer_name, producer_version=model.producer_version,
            model_version=model.model_version,
            opsets=[{"domain": item.domain, "version": item.version} for item in model.opset_import],
            metadata={item.key: item.value for item in model.metadata_props}, graph=graph,
            functions=[{"name": fn.name, "domain": fn.domain, "inputs": list(fn.input),
                        "outputs": list(fn.output), "operators": [_node(node) for node in fn.node]}
                       for fn in model.functions],
        )
        operators = graph["operators"]
    report["operator_counts"] = dict(sorted(Counter(
        f"{op['domain']}::{op['type']}" if op.get("domain") else op["type"]
        for op in operators
    ).items()))
    return _finite(report)
