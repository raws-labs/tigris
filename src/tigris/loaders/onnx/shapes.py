"""Resolve ONNX deployment shapes before the graph enters the IR.

Exporters emit a free batch dimension and compute the classifier reshape from
``Shape`` at runtime, so a stock export carries no concrete shape for memory
planning. Binding the free dimensions is not enough on its own: the shape
subgraph keeps downstream tensors at unknown rank until it is folded away.
"""

from __future__ import annotations

from collections.abc import Callable
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
            f"{self.tensor} axis {self.axis} ({name}) is unset; "
            f"using {self.value} (--input-shape overrides)"
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


# Shape vectors are a handful of elements. Capping a computed value keeps a
# constant-operand arithmetic node from copying a whole weight tensor into a
# second constant just to resolve a shape.
MAX_FOLDED_ELEMENTS = 4096

_ARITHMETIC = frozenset({"Add", "Sub", "Mul", "Div", "Pow"})
_COMPUTED = _ARITHMETIC | {"Cast", "Sqrt", "Reciprocal", "Neg", "Min", "Max", "Concat", "Gather",
                           "Unsqueeze", "Squeeze", "Reshape", "Slice"}


def constant_value(attrs: dict) -> np.ndarray | None:
    """The tensor a Constant node gives; None for ``sparse_value`` and the string forms."""
    if "value" in attrs:
        return numpy_helper.to_array(attrs["value"])
    if "value_float" in attrs or "value_floats" in attrs:
        return np.asarray(attrs.get("value_float", attrs.get("value_floats")), np.float32)
    if "value_int" in attrs or "value_ints" in attrs:
        return np.asarray(attrs.get("value_int", attrs.get("value_ints")), np.int64)
    return None


def static_value(
    kind: str,
    attrs: dict,
    inputs: list[str],
    value_of: Callable[[str], np.ndarray | None],
    shape_of: Callable[[str], tuple[int, ...] | None],
) -> np.ndarray | None:
    """The value an ONNX operator gives when the graph settles its operands.

    ``attrs`` holds the attribute values, ``value_of`` gives a constant's value
    and ``shape_of`` a tensor's concrete shape, both None when unknown. Returns
    None for an operator outside this set, an unsettled operand, an input ONNX
    leaves undefined, or a computed value larger than a shape vector.
    """
    if kind == "Constant":
        return constant_value(attrs)
    if kind == "Shape":
        shape = shape_of(inputs[0])
        if shape is None:
            return None
        extents = np.asarray(shape, dtype=np.int64)
        return extents[attrs.get("start", 0):attrs.get("end", len(extents))]
    if kind not in _COMPUTED:
        return None
    values = [value_of(name) if name else None for name in inputs]
    if not values or values[0] is None or any(
            value is None for name, value in zip(inputs, values) if name):
        return None
    try:
        result = _compute(kind, attrs, values)
    except (ValueError, IndexError, TypeError):
        return None
    if result is None:
        return None
    result = np.asarray(result, order="C")
    return None if result.size > MAX_FOLDED_ELEMENTS else result


def _ints(attrs: dict, values: list, name: str, index: int) -> list[int] | None:
    """Integers an older opset gives as attribute ``name`` and a newer one as input ``index``."""
    given = attrs[name] if name in attrs else (values[index] if index < len(values) else None)
    return None if given is None else [int(v) for v in np.atleast_1d(given)]


def _compute(kind: str, attrs: dict, values: list) -> np.ndarray | None:
    first = values[0]
    if kind == "Cast":
        target = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(int(attrs["to"])))
        return first.astype(target) if target.kind in "biuf" else None
    if kind == "Sqrt":
        return np.sqrt(first)
    if kind == "Reciprocal":
        return np.reciprocal(first)
    if kind == "Neg":
        return np.negative(first)
    if kind in ("Min", "Max"):
        combine = np.minimum if kind == "Min" else np.maximum
        result = first
        for value in values[1:]:
            result = combine(result, value)
        return result
    if kind in _ARITHMETIC:
        left, right = first, values[1]
        if kind == "Add":
            return left + right
        if kind == "Sub":
            return left - right
        if kind == "Mul":
            return left * right
        if kind == "Pow":
            return np.power(left, right).astype(left.dtype)
        if np.any(right == 0):
            return None
        if np.issubdtype(left.dtype, np.integer):
            # ONNX integer division truncates toward zero.
            quotient = np.abs(left) // np.abs(right)
            return np.where((left < 0) != (right < 0), -quotient, quotient).astype(left.dtype)
        return left / right
    if kind == "Concat":
        return np.concatenate([np.atleast_1d(v) for v in values], axis=int(attrs.get("axis", 0)))
    if kind == "Gather":
        return np.take(first, values[1], axis=int(attrs.get("axis", 0)))
    if kind == "Unsqueeze":
        axes = _ints(attrs, values, "axes", 1)
        return None if axes is None else np.expand_dims(first, tuple(axes))
    if kind == "Squeeze":
        axes = _ints(attrs, values, "axes", 1)
        return np.squeeze(first) if axes is None else np.squeeze(first, tuple(axes))
    if kind == "Reshape":
        dims = [int(d) for d in np.atleast_1d(values[1])]
        if not attrs.get("allowzero", 0):
            dims = [first.shape[axis] if extent == 0 else extent for axis, extent in enumerate(dims)]
        return first.reshape(dims)
    starts, ends = _ints(attrs, values, "starts", 1), _ints(attrs, values, "ends", 2)
    if starts is None or ends is None:
        return None
    axes = _ints(attrs, values, "axes", 3)
    axes = list(range(len(starts))) if axes is None else axes
    steps = _ints(attrs, values, "steps", 4)
    steps = [1] * len(starts) if steps is None else steps
    if not len(starts) == len(ends) == len(axes) == len(steps):
        return None
    index: list[slice] = [slice(None)] * first.ndim
    for axis, start, end, step in zip(axes, starts, ends, steps):
        if step == 0 or not -first.ndim <= axis < first.ndim:
            return None
        index[axis] = slice(start, end, step)
    return first[tuple(index)]


def bodies(node: onnx.NodeProto) -> list[onnx.GraphProto]:
    return [body for attr in node.attribute
            for body in ([attr.g] if attr.type == onnx.AttributeProto.GRAPH else attr.graphs)]


def outer_reads(body: onnx.GraphProto) -> list[str]:
    """Values from enclosing scopes that the body reads."""
    defined = (
        {value.name for value in body.input}
        | {tensor.name for tensor in body.initializer}
        | {tensor.values.name for tensor in body.sparse_initializer}
        | {name for inner in body.node for name in inner.output}
    )
    used = [name for inner in body.node for name in (*inner.input, *body_reads(inner))]
    return [name for name in (*used, *(value.name for value in body.output))
            if name and name not in defined]


def body_reads(node: onnx.NodeProto) -> list[str]:
    """Values from enclosing scopes that the node's subgraphs read."""
    return [name for body in bodies(node) for name in outer_reads(body)]


def fold_shape_subgraph(model: onnx.ModelProto) -> int:
    """Replace runtime shape arithmetic with the constants it evaluates to.

    Returns the number of nodes folded away. Node values are computed only
    where every operand is already known, so activations are never touched.
    """
    graph = model.graph
    known: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    shapes: dict[str, tuple[int, ...]] = {init.name: tuple(init.dims) for init in graph.initializer}
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        if all(d.dim_value > 0 for d in tensor_type.shape.dim):
            shapes[value_info.name] = tuple(d.dim_value for d in tensor_type.shape.dim)

    folded: set[int] = set()
    for index, node in enumerate(graph.node):
        if node.domain not in ("", "ai.onnx"):
            continue
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        value = static_value(node.op_type, attrs, list(node.input), known.get, shapes.get)
        if value is None:
            continue
        known[node.output[0]] = value
        folded.add(index)

    if not folded:
        return 0

    # Keep the values that a surviving node, its subgraphs or a graph output still read.
    survivors = [n for i, n in enumerate(graph.node) if i not in folded]
    needed = {name for node in survivors for name in (*node.input, *body_reads(node))}
    needed |= {value_info.name for value_info in graph.output}
    existing = {init.name for init in graph.initializer}
    for index in sorted(folded):
        for name in graph.node[index].output:
            if name in needed and name not in existing and name in known:
                graph.initializer.append(numpy_helper.from_array(known[name], name))
                if model.ir_version < 4:
                    # Before IR version 4 an initializer is also a graph input, or inference skips it.
                    graph.input.append(onnx.helper.make_tensor_value_info(
                        name, graph.initializer[-1].data_type, known[name].shape))

    del graph.node[:]
    graph.node.extend(survivors)
    del graph.value_info[:]
    return len(folded)
