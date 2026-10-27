"""ONNX model loading, shape inference, and topological sorting."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper, shape_inference

from tigris.frontends import load_onnx
from tigris.frontends.tflite import BOUNDARY_LAYOUT_KEY, STATE_KEY
from tigris.graph.ir import AnalyzedGraph, OpNode, StatePort, TensorInfo
from tigris.graph.subgraphs import CONTROL_FLOW
from tigris.loaders.onnx.shapes import (
    bind_free_dims, bodies, body_reads, constant_value, fold_shape_subgraph, outer_reads)

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
    return shape_inference.infer_shapes(model, data_prop=True)


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


def _file_shapes_only(
    model: onnx.ModelProto, input_shapes: dict[str, tuple[int, ...]]
) -> dict[str, tuple[int, ...]]:
    """Check shapes given for a channels-last file against the file.

    The conversion fixed every tensor shape from the file, so a different input
    shape would not describe the same model. A shape is given in the file's own
    order; the graph carries it channels-first.
    """
    declared = {vi.name: tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
                for vi in model.graph.input}
    result = {}
    for name, shape in input_shapes.items():
        graph_shape = (shape[0], shape[-1], *shape[1:-1]) if len(shape) >= 3 else tuple(shape)
        if name in declared and graph_shape != declared[name]:
            file_shape = declared[name]
            if len(file_shape) >= 3:
                file_shape = (file_shape[0], *file_shape[2:], file_shape[1])
            raise ValueError(
                f"Input {name!r} is {'x'.join(map(str, file_shape))} in the model file, "
                "which fixes every tensor shape; it cannot be compiled for another shape")
        result[name] = graph_shape
    return result


def load_model(
    path: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> AnalyzedGraph:
    """Load an ONNX model and return a partially populated AnalyzedGraph.

    Resolves deployment shapes, extracts operators and tensor metadata,
    performs a DFS topological sort favouring early tensor consumption.
    ``input_shapes`` maps a model input name to the full shape to compile for.
    """
    model = load_onnx(path)
    channels_last = any(
        prop.key == BOUNDARY_LAYOUT_KEY and prop.value == "channels_last"
        for prop in model.metadata_props)
    if channels_last and input_shapes:
        input_shapes = _file_shapes_only(model, input_shapes)
    # Checked before shape resolution, which can fail on a cycle or fold one of its edges away.
    _refuse_unordered(model.graph)
    bindings = resolve_shapes(model, input_shapes)
    _operands_from_outer_reads(model)

    ag = _graph_to_ag(model.graph, model.opset_import)
    ag.model_name = Path(path).stem
    ag.channels_last_source = channels_last
    ag.shape_bindings = bindings
    state = next((json.loads(prop.value) for prop in model.metadata_props
                  if prop.key == STATE_KEY), [])
    for entry in state:
        if entry["input"] not in ag.model_inputs or entry["initial"] not in ag.weight_data or (
                entry["output"] is not None and entry["output"] not in ag.model_outputs):
            raise ValueError(f"state entry {entry!r} does not name the model's own values")
        ag.state_ports.append(StatePort(
            input=ag.model_inputs.index(entry["input"]),
            output=None if entry["output"] is None else ag.model_outputs.index(entry["output"]),
            initial=np.asarray(ag.weight_data[entry["initial"]])))
    return ag


def _graph_to_ag(graph: onnx.GraphProto, opset_import) -> AnalyzedGraph:
    """One graph's operators and tensors; a graph-valued attribute becomes a
    subgraph of its own, referenced by index."""
    ag = AnalyzedGraph()
    ag.opset = next(
        (entry.version for entry in opset_import if entry.domain in ("", "ai.onnx")),
        0,
    )

    # --- Collect initializers (constants / weights) -----------------------
    initializer_names: set[str] = set()
    for init in graph.initializer:
        shape = tuple(init.dims)
        data, dtype = numpy_helper.to_array(init), init.data_type
        if dtype == onnx.TensorProto.INT4:
            # 4-bit values are int8 values; the int8 kernels run them exactly.
            data, dtype = data.astype(np.int8), onnx.TensorProto.INT8
        ag.tensors[init.name] = TensorInfo(
            name=init.name,
            shape=shape,
            dtype=dtype,
            is_constant=True,
        )
        initializer_names.add(init.name)
        ag.weight_data[init.name] = data

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
    ag.model_input_dtypes = [
        _extract_dtype(inp.type)
        for inp in graph.input
        if inp.name not in initializer_names
    ]
    ag.model_output_dtypes = [_extract_dtype(out.type) for out in graph.output]

    # --- Build OpNodes ----------------------------------------------------
    raw_nodes: list[OpNode] = []
    for i, node in enumerate(graph.node):
        name = _op_name(node, i)
        attrs: dict = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                branch = _infer(onnx_helper.make_model(attr.g, opset_imports=list(opset_import)))
                ag.subgraphs.append(_graph_to_ag(branch.graph, opset_import))
                attrs[attr.name] = len(ag.subgraphs) - 1
                continue
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
            op_type=_op_type(node),
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        )
        raw_nodes.append(op)

    # --- DFS topological sort (prefer early consumption) ------------------
    # A Constant becomes a constant after the sort, which it takes part in, so
    # the other operators keep their order.
    ag.ops = []
    for op in (raw_nodes[i] for i in _topo_sort(graph)):
        value = constant_value(op.attrs) if op.op_type == "Constant" else None
        if value is None:
            ag.ops.append(op)
            continue
        ag.weight_data[op.outputs[0]] = value
        info = ag.tensors.get(op.outputs[0])
        if info is None:
            ag.tensors[op.outputs[0]] = TensorInfo(
                name=op.outputs[0], shape=tuple(value.shape),
                dtype=onnx_helper.np_dtype_to_tensor_dtype(value.dtype), is_constant=True)
        else:
            info.is_constant = True
    for step, op in enumerate(ag.ops):
        op.step = step

    # A graph optimizer can leave value_info behind for a tensor its fusion
    # removed. Those entries describe nothing the graph computes, and carrying
    # them into the IR makes a fully quantized model look mixed-dtype.
    referenced = {name for op in ag.ops for name in op.inputs + op.outputs if name}
    boundary = set(ag.model_inputs) | set(ag.model_outputs)
    for name in [
        n
        for n, info in ag.tensors.items()
        if not info.is_constant and n not in referenced and n not in boundary
    ]:
        del ag.tensors[name]

    return ag


def _op_name(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"{node.op_type}_{index}"


def _op_type(node: onnx.NodeProto) -> str:
    # An operator from another domain is a different operator that may
    # share a standard name; qualifying it keeps every pass and the
    # support check from reading it as the standard one.
    return node.op_type if node.domain in ("", "ai.onnx") else f"{node.domain}::{node.op_type}"


def _scope(graph: onnx.GraphProto, typed: onnx.GraphProto, enclosing: dict) -> dict:
    """Every name the graph sees: a constant's tensor, or a value's type as
    ``typed``, the graph after inference, gives it (None when it gives none)."""
    stated = {value.name: value.type for value in (*typed.input, *typed.value_info, *typed.output)}
    scope = dict(enclosing)
    for node in graph.node:
        for name in node.output:
            scope[name] = stated.get(name)
        value = next((attr.t for attr in node.attribute if attr.name == "value"), None)
        if _op_type(node) == "Constant" and value is not None:
            scope[node.output[0]] = onnx.TensorProto()
            scope[node.output[0]].CopyFrom(value)
            scope[node.output[0]].name = node.output[0]
    scope.update((tensor.values.name, None) for tensor in graph.sparse_initializer)
    scope.update((value.name, value.type) for value in graph.input)
    scope.update((tensor.name, tensor) for tensor in graph.initializer)
    return scope


def _operands_from_outer_reads(model: onnx.ModelProto) -> None:
    """Make each If take what its branches read from enclosing graphs, in place:
    a value becomes an operand and an input of both branches, a constant is
    copied into the branch that reads it. ONNX's If becomes the compiler's
    own, whose branches take its operands after the condition. A While is
    refused instead, as is a read of a name no graph defines."""

    def rewrite(graph: onnx.GraphProto, enclosing: dict | None) -> None:
        # A subgraph's values are typed by inferring it on its own, as _graph_to_ag does.
        typed = graph if enclosing is None else _infer(
            onnx_helper.make_model(graph, opset_imports=list(model.opset_import))).graph
        scope = _scope(graph, typed, enclosing or {})
        for index, node in enumerate(graph.node):
            kind = _op_type(node).removeprefix("tigris::")
            graphs = {attr.name: attr.g for attr in node.attribute
                      if attr.type == onnx.AttributeProto.GRAPH}
            roles = [role for role in CONTROL_FLOW.get(kind, ()) if role in graphs]
            reads = {role: list(dict.fromkeys(outer_reads(graphs[role]))) for role in roles}
            for role in roles:
                for name in reads[role]:
                    if name not in scope:
                        raise ValueError(f"operator {_op_name(node, index)} reads {name}, "
                                         "which no graph defines")
                    if kind == "While":
                        raise ValueError(
                            f"operator {_op_name(node, index)} reads {name} from an enclosing "
                            f"graph in its {role}; a While condition and body can read only "
                            "its loop variables and the values they define")
            if kind == "If":
                if node.domain != "tigris" and not any(
                        entry.domain == "tigris" for entry in model.opset_import):
                    model.opset_import.append(onnx_helper.make_opsetid("tigris", 1))
                node.domain = "tigris"
                values = [name for name in dict.fromkeys(n for r in roles for n in reads[r])
                          if not isinstance(scope[name], onnx.TensorProto)]
                node.input.extend(values)
                for role in roles:
                    graphs[role].initializer.extend(
                        scope[name] for name in reads[role] if isinstance(scope[name], onnx.TensorProto))
                    for name in values:
                        entering = graphs[role].input.add()
                        entering.name = name
                        if scope[name] is not None:
                            entering.type.CopyFrom(scope[name])
            for role in roles:
                rewrite(graphs[role], scope)

    rewrite(model.graph, None)


def _refuse_unordered(graph: onnx.GraphProto) -> None:
    """Refuse the graph if it or any subgraph in it has no execution order."""
    _topo_sort(graph)
    for node in graph.node:
        for body in bodies(node):
            _refuse_unordered(body)


def _topo_sort(graph: onnx.GraphProto) -> list[int]:
    """DFS-based topological sort, as indices into ``graph.node``.

    A node reads its inputs and every value its subgraphs read from this
    graph.  Visits children in reverse order so that the first consumer of a
    tensor appears earliest, which tends to reduce peak live memory by freeing
    tensors sooner.  A graph whose nodes cannot all be ordered is refused.
    """
    nodes = graph.node
    n = len(nodes)
    sources = {tensor.name for tensor in graph.initializer} | {value.name for value in graph.input}
    producer_of = {name: idx for idx, node in enumerate(nodes) for name in node.output}
    # Build adjacency: producer_idx -> list[consumer_idx]
    children: list[list[int]] = [[] for _ in range(n)]
    in_degree = [0] * n
    for idx, node in enumerate(nodes):
        for inp in (*node.input, *body_reads(node)):
            if inp in sources or inp == "":
                continue
            producer = producer_of.get(inp)
            if producer is not None:
                children[producer].append(idx)
                in_degree[idx] += 1

    # Kahn's algorithm: a node is placed once all its producers are.
    waiting = list(in_degree)
    ready = [i for i in range(n) if waiting[i] == 0]
    placed = [False] * n
    while ready:
        idx = ready.pop()
        placed[idx] = True
        for child in children[idx]:
            waiting[child] -= 1
            if waiting[child] == 0:
                ready.append(child)
    unplaced = next((i for i in range(n) if not placed[i]), None)
    if unplaced is not None:
        raise ValueError(
            f"operator {_op_name(nodes[unplaced], unplaced)} is on a cycle or depends on one, "
            "so the graph has no execution order"
        )

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

    # Start from roots (nodes with in_degree 0); in an acyclic graph they reach every node.
    for r in range(n):
        if in_degree[r] == 0 and not visited[r]:
            dfs(r)

    order.reverse()
    return order
