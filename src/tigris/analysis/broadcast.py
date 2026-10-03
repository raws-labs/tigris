"""How a binary operator's operands reach its output's shape.

The runtime reads each operand in the output's stored axis order in one of
three ways: dense, with the output's shape; repeating every n elements, when
every axis ahead of the ones matching the output is 1; or by output
coordinate, for any other broadcast. The last is never tiled.
"""

from tigris.graph.ir import AnalyzedGraph, OpNode, serialized_shape

BINARY_OPS = frozenset({"Add", "Sub", "Mul", "Div", "SquaredDifference", "Max", "Min",
                        "FloorDiv", "FloorMod", "PRelu",
                        "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual",
                        "And", "Or", "Where"})

DENSE, PERIODIC, GENERAL = "dense", "periodic", "general"


def padded(shape, rank: int) -> tuple[int, ...]:
    """A shape aligned to `rank` axes from the innermost, as broadcasting does."""
    shape = tuple(int(dim) for dim in shape)
    return (1,) * (rank - len(shape)) + shape


def broadcasts_to(shape, target) -> bool:
    shape, target = tuple(shape), tuple(target)
    if len(shape) > len(target):
        return False
    return all(dim in (1, extent) for dim, extent in zip(padded(shape, len(target)), target))


def access(stored, stored_output) -> str | None:
    """DENSE, PERIODIC or GENERAL for an operand in stored axis order, or None
    when it does not broadcast to the output."""
    stored, stored_output = tuple(stored), tuple(stored_output)
    if len(stored) != len(stored_output) or not broadcasts_to(stored, stored_output):
        return None
    if stored == stored_output:
        return DENSE
    start = len(stored)
    while start > 0 and stored[start - 1] == stored_output[start - 1]:
        start -= 1
    return PERIODIC if all(dim == 1 for dim in stored[:start]) else GENERAL


def is_constant(ag: AnalyzedGraph, name: str) -> bool:
    info = ag.tensors.get(name)
    return name in ag.weight_data or (info is not None and info.is_constant)


def stored_operands(ag: AnalyzedGraph, op: OpNode):
    """(name, constant, stored shape, access) per operand of a binary operator,
    or None when an operand's shape or the output is unknown. A constant is
    aligned to the output's rank and held in the output's layout; a tensor
    operand is taken as stored."""
    if op.op_type not in BINARY_OPS or len(op.inputs) != (3 if op.op_type == "Where" else 2) or len(op.outputs) != 1:
        return None
    output = ag.tensors.get(op.outputs[0])
    if output is None:
        return None
    stored_output = serialized_shape(output.shape, output.layout)
    result = []
    for name in op.inputs:
        info = ag.tensors.get(name)
        constant = is_constant(ag, name)
        if constant:
            data = ag.weight_data.get(name)
            shape = data.shape if data is not None else (info.shape if info else None)
            if shape is None or len(shape) > len(output.shape):
                return None
            stored = serialized_shape(padded(shape, len(output.shape)), output.layout)
        else:
            if info is None or len(info.shape) > len(output.shape):
                return None
            if len(info.shape) != len(output.shape) and op.op_type not in {
                    "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Where"}:
                return None
            if len(info.shape) != len(output.shape) and ((len(info.shape) >= 3 and info.layout.value != "linear") or (len(output.shape) >= 3 and output.layout.value != "linear")):
                return None
            stored = padded(serialized_shape(info.shape, info.layout), len(output.shape))
        result.append((name, constant, stored, access(stored, stored_output)))
    return result
