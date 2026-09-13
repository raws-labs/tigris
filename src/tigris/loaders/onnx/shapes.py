"""Resolve ONNX deployment shapes before the graph enters the IR.

Exporters emit a free batch dimension and compute the classifier reshape from
``Shape`` at runtime, so a stock export carries no concrete shape for memory
planning. Binding the free dimensions is not enough on its own: the shape
subgraph keeps downstream tensors at unknown rank until it is folded away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class DimBinding:
    """A free dimension the compiler gave a concrete extent of its own choice.

    A dimension the caller named is not a binding: the shape is what was asked
    for, so there is nothing to report about it.
    """

    tensor: str
    axis: int
    symbol: str
    value: int

    def describe(self) -> str:
        name = self.symbol or "unknown"
        return (
            f"{self.tensor} axis {self.axis} ({name}) "
            f"has no fixed size; using {self.value}"
        )


def bind_free_dims(
    model: onnx.ModelProto,
    overrides: dict[str, tuple[int, ...]] | None = None,
) -> list[DimBinding]:
    """Give every free graph-input dimension a concrete extent.

    An input named in ``overrides`` takes the shape given there, whatever the
    model declared. Every other free dimension becomes 1, the deployment case
    for embedded inference. Inferred shapes downstream are dropped so they are
    recomputed from the bound inputs rather than kept at their symbolic extents.
    Only the dimensions this function chose itself are returned.
    """
    overrides = overrides or {}
    initializers = {init.name for init in model.graph.initializer}
    bindings: list[DimBinding] = []
    changed = False

    for value_info in model.graph.input:
        if value_info.name in initializers:
            continue
        dims = value_info.type.tensor_type.shape.dim
        override = overrides.get(value_info.name)
        if override is not None:
            if len(override) != len(dims):
                raise ValueError(
                    f"Input {value_info.name!r} has rank {len(dims)}, "
                    f"but the given shape has rank {len(override)}"
                )
            for dim, extent in zip(dims, override):
                changed = changed or dim.dim_value != extent
                dim.ClearField("dim_param")
                dim.dim_value = extent
            continue
        for axis, dim in enumerate(dims):
            if dim.dim_value > 0:
                continue
            bindings.append(DimBinding(value_info.name, axis, dim.dim_param, 1))
            dim.ClearField("dim_param")
            dim.dim_value = 1
            changed = True

    unknown = set(overrides) - {vi.name for vi in model.graph.input}
    if unknown:
        raise ValueError(
            "No such model input: " + ", ".join(sorted(unknown))
        )

    if changed:
        del model.graph.value_info[:]
        for value_info in model.graph.output:
            value_info.type.tensor_type.ClearField("shape")
    return bindings


def _axes_of(node: onnx.NodeProto, known: dict[str, np.ndarray]) -> np.ndarray | None:
    """Axes live in an attribute before opset 13 and in an input after it."""
    for attr in node.attribute:
        if attr.name == "axes":
            return np.asarray(attr.ints, dtype=np.int64)
    if len(node.input) > 1:
        return known.get(node.input[1])
    return None


def _fold_node(
    node: onnx.NodeProto,
    known: dict[str, np.ndarray],
    shapes: dict[str, tuple[int, ...]],
) -> np.ndarray | None:
    """Evaluate one shape-computing node, or return None if it is not foldable."""
    attrs = {a.name: a for a in node.attribute}

    if node.op_type == "Constant":
        if "value" not in attrs:
            return None
        return numpy_helper.to_array(attrs["value"].t)

    if node.op_type == "Shape":
        shape = shapes.get(node.input[0])
        if shape is None:
            return None
        extents = np.asarray(shape, dtype=np.int64)
        start = attrs["start"].i if "start" in attrs else 0
        end = attrs["end"].i if "end" in attrs else len(extents)
        return extents[start:end]

    operands = [known.get(name) for name in node.input]
    if any(operand is None for operand in operands):
        return None

    if node.op_type == "Gather":
        axis = attrs["axis"].i if "axis" in attrs else 0
        return np.take(operands[0], operands[1], axis=axis)
    if node.op_type == "Concat":
        return np.concatenate([np.atleast_1d(o) for o in operands], axis=attrs["axis"].i)
    if node.op_type == "Unsqueeze":
        axes = _axes_of(node, known)
        if axes is None:
            return None
        result = operands[0]
        for axis in sorted(int(a) for a in np.atleast_1d(axes)):
            result = np.expand_dims(result, axis)
        return result
    if node.op_type == "Squeeze":
        axes = _axes_of(node, known)
        if axes is None:
            return np.squeeze(operands[0])
        return np.squeeze(operands[0], axis=tuple(int(a) for a in np.atleast_1d(axes)))
    if node.op_type == "Cast":
        return operands[0].astype(onnx.helper.tensor_dtype_to_np_dtype(attrs["to"].i))
    if node.op_type == "Slice":
        data, starts, ends = operands[0], operands[1], operands[2]
        axes = operands[3] if len(operands) > 3 else np.arange(len(np.atleast_1d(starts)))
        steps = operands[4] if len(operands) > 4 else np.ones_like(np.atleast_1d(starts))
        result = data
        for axis, start, end, step in zip(axes, starts, ends, steps):
            index = [slice(None)] * result.ndim
            index[int(axis)] = slice(int(start), int(end), int(step))
            result = result[tuple(index)]
        return result
    if node.op_type in ("Add", "Sub", "Mul", "Div"):
        left, right = operands[0], operands[1]
        if node.op_type == "Add":
            return left + right
        if node.op_type == "Sub":
            return left - right
        if node.op_type == "Mul":
            return left * right
        return left // right if np.issubdtype(left.dtype, np.integer) else left / right
    return None


# Shape vectors are a handful of elements. Capping the materialized result
# keeps a constant-operand arithmetic node from copying a whole weight tensor
# into a second initializer just to resolve a shape.
_MAX_FOLDED_ELEMENTS = 4096


def fold_shape_subgraph(model: onnx.ModelProto) -> int:
    """Replace runtime shape arithmetic with the constants it evaluates to.

    Returns the number of nodes folded away. Node values are computed only
    where every operand is already known, so activations are never touched.
    """
    graph = model.graph
    known: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        if all(d.dim_value > 0 for d in tensor_type.shape.dim):
            shapes[value_info.name] = tuple(d.dim_value for d in tensor_type.shape.dim)

    folded: set[int] = set()
    for index, node in enumerate(graph.node):
        value = _fold_node(node, known, shapes)
        if value is None:
            continue
        if node.op_type != "Constant" and value.size > _MAX_FOLDED_ELEMENTS:
            continue
        known[node.output[0]] = value
        folded.add(index)

    if not folded:
        return 0

    # Keep the values that a surviving node or a graph output still reads.
    survivors = [n for i, n in enumerate(graph.node) if i not in folded]
    needed = {name for node in survivors for name in node.input}
    needed |= {value_info.name for value_info in graph.output}
    existing = {init.name for init in graph.initializer}
    for index in sorted(folded):
        for name in graph.node[index].output:
            if name in needed and name not in existing and name in known:
                graph.initializer.append(numpy_helper.from_array(known[name], name))

    del graph.node[:]
    graph.node.extend(survivors)
    del graph.value_info[:]
    return len(folded)
