"""ONNX model loading, shape inference, and topological sorting."""

from pathlib import Path

import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper, shape_inference

from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders.onnx.shapes import bind_free_dims, fold_shape_subgraph

# One binding pass plus one fold usually resolves a stock export.  A fold can
# expose shapes that let the next fold proceed, so iterate a few times and stop
# as soon as a round changes nothing.
_MAX_RESOLVE_ROUNDS = 4


def _extract_shape(type_proto: onnx.TypeProto, tensor_name: str) -> tuple[int, ...]:
    """Extract a fully concrete deployment shape from an ONNX TypeProto.

    Memory planning cannot safely guess symbolic or otherwise unresolved
    dimensions.  ``resolve_shapes`` binds the free input dimensions first, so
    anything still unresolved here is a shape the compiler could not derive;
    reject it rather than understate the required arena size.
    """
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(
            f"Tensor {tensor_name!r} has unknown rank; TiGrIS requires "
            "concrete deployment shapes"
        )
    dims: list[int] = []
    for axis, d in enumerate(tensor_type.shape.dim):
        if d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            detail = f"symbolic value {d.dim_param!r}" if d.dim_param else "unknown value"
            raise ValueError(
                f"Tensor {tensor_name!r} dimension {axis} has {detail}; "
                "TiGrIS requires concrete deployment shapes"
            )
    return tuple(dims)


def _extract_dtype(type_proto: onnx.TypeProto) -> int:
    return type_proto.tensor_type.elem_type


def _infer(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        # Some models fail full data propagation; try without
        return shape_inference.infer_shapes(model)


def resolve_shapes(
    model: onnx.ModelProto,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> list[str]:
    """Give the model concrete deployment shapes, in place.

    A stock export leaves the batch dimension free and computes the classifier
    reshape from ``Shape`` at runtime.  Binding the free input dimensions is
    not enough on its own: the shape subgraph keeps everything downstream of
    it at unknown rank until it is folded to the constants it evaluates to.
    Returns one description per bound dimension.
    """
    bindings = bind_free_dims(model, input_shapes)
    model.CopyFrom(_infer(model))
    for _ in range(_MAX_RESOLVE_ROUNDS):
        if fold_shape_subgraph(model) == 0:
            break
        model.CopyFrom(_infer(model))
    return [binding.describe() for binding in bindings]


def load_model(
    path: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> AnalyzedGraph:
    """Load an ONNX model and return a partially populated AnalyzedGraph.

    Resolves deployment shapes, extracts operators and tensor metadata,
    performs a DFS topological sort favouring early tensor consumption.
    ``input_shapes`` maps a model input name to the full shape to compile for.
    """
    model = onnx.load(str(path))
    bindings = resolve_shapes(model, input_shapes)

    graph = model.graph
    ag = AnalyzedGraph()
    ag.model_name = Path(path).stem
    ag.shape_bindings = bindings

    # --- Collect initializers (constants / weights) -----------------------
    initializer_names: set[str] = set()
    for init in graph.initializer:
        shape = tuple(init.dims)
        ag.tensors[init.name] = TensorInfo(
            name=init.name,
            shape=shape,
            dtype=init.data_type,
            is_constant=True,
        )
        initializer_names.add(init.name)
        ag.weight_data[init.name] = numpy_helper.to_array(init)

    # --- Collect value_info (intermediate tensors with inferred shapes) ---
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name in ag.tensors:
            continue
        ag.tensors[vi.name] = TensorInfo(
            name=vi.name,
            shape=_extract_shape(vi.type, vi.name),
            dtype=_extract_dtype(vi.type),
            is_constant=vi.name in initializer_names,
        )

    # --- Model inputs / outputs -------------------------------------------
    ag.model_inputs = [
        inp.name for inp in graph.input if inp.name not in initializer_names
    ]
    ag.model_outputs = [out.name for out in graph.output]

    # --- Build OpNodes ----------------------------------------------------
    nodes_by_output: dict[str, int] = {}  # tensor_name -> node index
    raw_nodes: list[OpNode] = []
    for i, node in enumerate(graph.node):
        name = node.name or f"{node.op_type}_{i}"
        attrs: dict = {}
        for attr in node.attribute:
            val = onnx_helper.get_attribute_value(attr)
            # Convert bytes to str for cleaner downstream usage
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            # Convert repeated ints/floats to plain lists
            elif hasattr(val, "__len__") and not isinstance(val, str):
                val = list(val)
            attrs[attr.name] = val
        op = OpNode(
            name=name,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        )
        raw_nodes.append(op)
        for out in node.output:
            nodes_by_output[out] = i

    # --- DFS topological sort (prefer early consumption) ------------------
    ag.ops = _topo_sort(raw_nodes, nodes_by_output, initializer_names, ag.model_inputs)

    return ag


def _topo_sort(
    nodes: list[OpNode],
    tensor_to_producer: dict[str, int],
    constants: set[str],
    model_inputs: list[str],
) -> list[OpNode]:
    """DFS-based topological sort.

    Visits children in reverse order so that the first consumer of a tensor
    appears earliest - this tends to reduce peak live memory by freeing
    tensors sooner.
    """
    n = len(nodes)
    # Build adjacency: producer_idx -> list[consumer_idx]
    children: list[list[int]] = [[] for _ in range(n)]
    for idx, node in enumerate(nodes):
        for inp in node.inputs:
            if inp in constants or inp in model_inputs or inp == "":
                continue
            producer = tensor_to_producer.get(inp)
            if producer is not None and producer != idx:
                children[producer].append(idx)

    # Also build in-degree for cycle-safety
    in_degree = [0] * n
    for idx, node in enumerate(nodes):
        for inp in node.inputs:
            if inp in constants or inp in model_inputs or inp == "":
                continue
            producer = tensor_to_producer.get(inp)
            if producer is not None and producer != idx:
                in_degree[idx] += 1

    # DFS post-order, then reverse
    visited = [False] * n
    order: list[int] = []

    def dfs(idx: int) -> None:
        visited[idx] = True
        # Visit children in reverse so first child ends up earliest in final order
        for child in reversed(children[idx]):
            if not visited[child]:
                dfs(child)
        order.append(idx)

    # Start from roots (nodes with in_degree 0)
    roots = [i for i in range(n) if in_degree[i] == 0]
    for r in roots:
        if not visited[r]:
            dfs(r)

    # Any unvisited (e.g. cycles or disconnected) - append in original order
    for i in range(n):
        if not visited[i]:
            order.append(i)

    order.reverse()

    # Assign step indices and return
    sorted_ops: list[OpNode] = []
    for step, idx in enumerate(order):
        node = nodes[idx]
        node.step = step
        sorted_ops.append(node)
    return sorted_ops
