"""What a model's operators cost, from the shapes and constants its file states."""

import math

_BYTES = {"float32": 4, "int32": 4, "uint32": 4, "float16": 2, "bfloat16": 2, "int16": 2,
          "uint16": 2, "int8": 1, "uint8": 1, "bool": 1, "int64": 8, "uint64": 8, "float64": 8}

# Multiply-accumulates per output element, from the weight's shape: TFLite and
# ONNX convolutions both put the output channels first.
_PER_OUTPUT = {
    "CONV_2D": lambda w: math.prod(w[1:]),
    "Conv": lambda w: math.prod(w[1:]),
    "DEPTHWISE_CONV_2D": lambda w: math.prod(w[1:3]),
    "FULLY_CONNECTED": lambda w: math.prod(w[1:]),
}
# Operators whose cost follows the first input's last axis.
_MATRIX = {"BATCH_MATMUL", "MatMul", "Gemm"}


def _elements(shape) -> int | None:
    if shape is None or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        return None
    return math.prod(shape)


def _bytes(tensor) -> int | None:
    count = _elements(tensor.get("shape"))
    size = _BYTES.get(tensor.get("dtype"))
    return None if count is None or size is None else count * size


def _macs(op, tensors) -> int | None:
    """The operator's multiply-accumulates, 0 for an operator without any,
    None where a shape it needs is not stated."""
    kind = op["type"]
    inputs = [tensors.get(name) for name in op["inputs"]]
    outputs = [tensors.get(name) for name in op["outputs"]]
    out = _elements(outputs[0]["shape"]) if outputs and outputs[0] else None
    weight = next((t for t in inputs[1:] if t and t.get("constant")), None)
    if kind in _PER_OUTPUT:
        if out is None or weight is None or _elements(weight["shape"]) is None:
            return None
        return out * _PER_OUTPUT[kind](weight["shape"])
    if kind in _MATRIX:
        first = inputs[0] if inputs else None
        if out is None or not first or not first.get("shape"):
            return None
        depth = first["shape"][-1]
        return out * depth if isinstance(depth, int) and depth > 0 else None
    if kind in ("TRANSPOSE_CONV", "ConvTranspose"):
        # Each input element scatters into kernel x output-channel positions.
        data = inputs[-1] if kind == "TRANSPOSE_CONV" else inputs[0]
        count = _elements(data["shape"]) if data else None
        if count is None or weight is None or _elements(weight["shape"]) is None:
            return None
        w = weight["shape"]
        per_input = math.prod(w[:3]) if kind == "TRANSPOSE_CONV" else math.prod(w[1:])
        return count * per_input
    return 0


def _from_constants(op, sources) -> bool:
    inputs = [name for name in op["inputs"] if name]
    return op["type"] == "Constant" or (bool(inputs) and all(name in sources for name in inputs))


def _fold_constants(operators, tensors):
    """Mark every tensor computed from constants alone as a constant, such as
    a dequantized weight, and map each constant to the stored constants it
    comes from. An undeclared shape follows the operator's first input."""
    tensors = {name: dict(tensor) for name, tensor in tensors.items()}
    sources = {name: {name} for name, tensor in tensors.items() if tensor.get("constant")}
    for op in operators:
        if not _from_constants(op, sources):
            continue
        inputs = [name for name in op["inputs"] if name]
        derived = set().union(*(sources[name] for name in inputs)) if inputs else set()
        first = tensors.get(inputs[0], {}) if inputs else {}
        for name in op["outputs"]:
            tensor = tensors.setdefault(name, {"shape": None, "dtype": None})
            if tensor.get("shape") is None:
                tensor["shape"] = first.get("shape")
            tensor["constant"] = True
            sources[name] = derived
    return tensors, sources


def model_costs(operators: list[dict], tensors: dict[str, dict], inputs: list[str], *,
                top: int = 3) -> dict:
    """Per operator type: count, weight bytes, multiply-accumulates; the largest
    activations with the operator that writes them. `tensors` maps a name to
    shape, dtype, and whether it is a constant with its byte size."""
    tensors, sources = _fold_constants(operators, tensors)
    by_type: dict[str, dict] = {}
    counted: set[str] = set()
    steps = []
    for index, op in enumerate(operators):
        entry = by_type.setdefault(op["type"], {"type": op["type"], "count": 0, "weight_bytes": 0,
                                                "macs": 0, "macs_known": True})
        entry["count"] += 1
        weight = 0
        # An operator computed from constants alone is part of a weight, so
        # the weight counts where a computed tensor reads it.
        if not _from_constants(op, sources):
            for name in op["inputs"]:
                for source in sorted(sources.get(name, ())):
                    if source not in counted:
                        counted.add(source)
                        weight += tensors[source].get("size_bytes") or 0
        macs = _macs(op, tensors)
        entry["weight_bytes"] += weight
        if macs is None:
            entry["macs_known"] = False
        else:
            entry["macs"] += macs
        steps.append({"step": index, "type": op["type"], "weight_bytes": weight, "macs": macs,
                      "constant": _from_constants(op, sources)})
    produced = []
    for name in inputs:
        tensor = tensors.get(name)
        size = _bytes(tensor) if tensor and not tensor.get("constant") else None
        if size is not None:
            produced.append({"name": name, "shape": tensor["shape"], "dtype": tensor["dtype"],
                             "bytes": size, "step": None, "type": None})
    for index, op in enumerate(operators):
        for name in op["outputs"]:
            tensor = tensors.get(name)
            size = _bytes(tensor) if tensor and not tensor.get("constant") else None
            if size is not None:
                produced.append({"name": name, "shape": tensor["shape"], "dtype": tensor["dtype"],
                                 "bytes": size, "step": index, "type": op["type"]})
    by_step = {}
    for item in produced:
        if item["step"] is not None:
            by_step.setdefault(item["step"], item)
    for step in steps:
        out = by_step.get(step["step"])
        step["output_bytes"] = out["bytes"] if out else None
    produced.sort(key=lambda item: (-item["bytes"], -1 if item["step"] is None else item["step"]))
    kinds = sorted(by_type.values(), key=lambda item: (-item["macs"], -item["weight_bytes"],
                                                       -item["count"], item["type"]))
    return {
        "operators": [{"type": e["type"], "count": e["count"], "weight_bytes": e["weight_bytes"],
                       "macs": e["macs"] if e["macs_known"] else None} for e in kinds],
        "steps": steps,
        "weight_bytes": sum(e["weight_bytes"] for e in kinds),
        "macs": sum(e["macs"] for e in kinds) if all(e["macs_known"] for e in kinds) else None,
        "largest_activations": produced[:top],
        "activations": produced,
        "constants": sorted(sources),
    }


def quantization(activations: list[dict], constants: list[dict], operators: list[dict],
                 outputs: list[str], folded: set[str]) -> dict:
    """Activation dtypes, how int8 weights are quantized, and float32 islands:
    DEQUANTIZE operators on activations whose result is not a model output.
    `folded` names the tensors computed from constants alone."""
    weights = [t for t in constants if t.get("dtype") == "int8"]
    return {
        "activation_dtypes": sorted({t["dtype"] for t in activations}),
        "int8_weights": len(weights),
        "per_channel": sum(t.get("scales", 0) > 1 for t in weights),
        "asymmetric": sum(not t.get("symmetric", True) for t in weights),
        "float_islands": sum(op["type"] == "DEQUANTIZE" and not set(op["outputs"]) & set(outputs)
                             and not set(op["inputs"]) <= folded for op in operators),
    }
