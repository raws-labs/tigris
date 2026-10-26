"""Values the graph settles before it runs, folded into constants, and what computes or holds nothing at inference."""

import numpy as np
from onnx.helper import np_dtype_to_tensor_dtype

from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders.onnx.normalize.common import _Uses
from tigris.loaders.onnx.shapes import static_value


def _drop_unreferenced_weights(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Forget constants no remaining operator consumes.

    Folded subgraphs and absorbed activations leave their operands behind, and
    the emitter writes every entry of ``weight_data`` into the plan blob.
    """
    uses = _Uses(ag)
    for name in [n for n in ag.weight_data if not uses.reader_count(n)]:
        del ag.weight_data[name]
    return ag


_INFERENCE_IDENTITY_OPS = frozenset({"Dropout", "Identity"})


def _drop_inference_identities(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Remove operators that are the identity at inference.

    Dropout scales nothing once training_mode is false, and Identity never did.
    Exporters leave both in place, and neither has a plan opcode, so a stock
    export is rejected for operators that would compute nothing. Dropout's
    optional second output is a mask that only training consumes; an op whose
    mask is actually read is left alone rather than silently dropped.
    """
    uses = _Uses(ag)
    outputs_kept = set(ag.model_outputs)

    surviving: list[OpNode] = []
    rename: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type not in _INFERENCE_IDENTITY_OPS:
            surviving.append(op)
            continue
        if len(op.outputs) > 1 and any(
            uses.reader_count(out) or out in outputs_kept for out in op.outputs[1:]
        ):
            surviving.append(op)
            continue
        source = op.inputs[0]
        rename[op.outputs[0]] = rename.get(source, source)
        for stale in op.outputs[1:]:
            ag.tensors.pop(stale, None)

    if not rename:
        return ag

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in rename and name not in seen:
            seen.add(name)
            name = rename[name]
        return name

    for op in surviving:
        op.inputs = [resolve(n) if n else n for n in op.inputs]
    ag.model_outputs = [resolve(n) for n in ag.model_outputs]

    # A removed op's output tensor keeps the graph's declared name when it is a
    # model output, so only drop tensors nothing refers to any more.
    live = {n for op in surviving for n in op.inputs + op.outputs if n}
    live |= set(ag.model_inputs) | set(ag.model_outputs)
    for name in [n for n in rename if n not in live]:
        ag.tensors.pop(name, None)

    ag.ops = surviving
    return ag


def _fold_static_subgraphs(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Compute the parts of the graph that do not depend on the input data.

    An exporter that traces a model rather than folding it leaves the
    arithmetic behind: an attention scale becomes a Shape of the query, a
    Slice of that shape, a Cast, a Sqrt and a division. Every value in the
    chain is settled by the shapes the plan already states, so it is computed
    here and the operators disappear, which also turns the multiplication that
    reads the result from a broadcast the runtime lacks into a scale it has.

    A Shape counts as settled because the plan states every extent; a model
    with a dimension left open is bound before normalization runs.
    """
    known: dict[str, np.ndarray] = {}
    produced_outputs = set(ag.model_outputs)
    removed: set[int] = set()

    def value_of(name: str) -> np.ndarray | None:
        return known[name] if name in known else ag.weight_data.get(name)

    def shape_of(name: str) -> tuple[int, ...] | None:
        info = ag.tensors.get(name)
        if info is None or any(int(extent) <= 0 for extent in info.shape):
            return None
        return tuple(int(extent) for extent in info.shape)

    progressed = True
    while progressed:
        progressed = False
        for index, op in enumerate(ag.ops):
            if index in removed or len(op.outputs) != 1:
                continue
            result_name = op.outputs[0]
            if result_name in produced_outputs or result_name in known:
                continue
            value = static_value(op.op_type, op.attrs, op.inputs, value_of, shape_of)
            if value is None:
                continue
            known[result_name] = value
            removed.add(index)
            progressed = True

    if not known:
        return ag

    for name, value in known.items():
        ag.weight_data[name] = value
        info = ag.tensors.get(name)
        if info is None:
            ag.tensors[name] = TensorInfo(
                name=name, shape=tuple(value.shape),
                dtype=np_dtype_to_tensor_dtype(value.dtype), is_constant=True)
        else:
            info.shape = tuple(value.shape)
            info.is_constant = True

    ag.ops = [op for index, op in enumerate(ag.ops) if index not in removed]
    return ag


_SHAPE_OP_TYPES = frozenset({"Shape", "Gather", "Constant", "Unsqueeze", "Concat"})


def _fold_shape_ops(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Remove shape-computation op chains that feed Reshape's shape input.

    In models like MobileNetV2, a chain of Constant->Shape->Gather->Unsqueeze->
    Concat computes the target shape for Reshape at runtime.  Since all shapes
    are static at compile time, these ops can be removed.  The intermediate
    tensors are marked ``is_constant=True`` so the binary emitter skips them.
    """
    uses = _Uses(ag)
    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "Reshape":
            continue
        if len(op.inputs) < 2:
            continue

        # Reshape's second input is the target shape tensor
        shape_tensor = op.inputs[1]

        # BFS backward to collect all producer ops
        visited_ops: set[int] = set()
        queue = [shape_tensor]
        seen_tensors: set[str] = set()
        all_shape_ops = True

        while queue:
            tname = queue.pop()
            if tname in seen_tensors:
                continue
            seen_tensors.add(tname)

            if tname not in uses.producers:
                continue  # model input or initializer - not an op output

            prod_idx = uses.producers[tname]
            if prod_idx in visited_ops:
                continue
            visited_ops.add(prod_idx)

            prod_op = ag.ops[prod_idx]
            if prod_op.op_type not in _SHAPE_OP_TYPES:
                all_shape_ops = False
                break

            # Shape and Constant ops don't need to follow inputs:
            # Shape reads shape metadata (not data), Constant has none.
            if prod_op.op_type in ("Shape", "Constant"):
                continue

            # Continue BFS through this op's inputs
            for inp in prod_op.inputs:
                queue.append(inp)

        if all_shape_ops and visited_ops:
            removed |= visited_ops
            # Mark all intermediate tensors as constant
            for tname in seen_tensors:
                if tname in ag.tensors:
                    ag.tensors[tname].is_constant = True

    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]

    # Fix Reshape output shapes that ONNX shape inference couldn't resolve
    _fix_reshape_shapes(ag)

    return ag


# Edits ag in place and returns None; the passes return ag.
def _fix_reshape_shapes(ag: AnalyzedGraph) -> None:
    """Compute correct output shapes for Reshape ops with unknown outputs.

    ONNX shape inference often can't resolve Reshape output shapes when the
    target shape comes from a dynamic computation chain (Shape->Gather->Concat).
    After folding those chains, we can compute the shape statically from the
    data input shape.
    """
    for op in ag.ops:
        if op.op_type != "Reshape":
            continue
        if len(op.inputs) < 2 or len(op.outputs) < 1:
            continue

        out_name = op.outputs[0]
        out_info = ag.tensors.get(out_name)
        if out_info is None or out_info.shape:
            continue  # shape already known

        data_name = op.inputs[0]
        data_info = ag.tensors.get(data_name)
        if data_info is None or not data_info.shape:
            continue

        # Try to get target shape from weight_data (constant initializer)
        shape_name = op.inputs[1]
        if shape_name in ag.weight_data:
            target = ag.weight_data[shape_name].flatten().tolist()
        else:
            # Infer from data input: flatten to [batch, -1]
            target = [data_info.shape[0], -1]

        # Resolve -1 dimension
        total = 1
        for d in data_info.shape:
            total *= d
        neg_idx = None
        known_product = 1
        resolved = []
        for i, d in enumerate(target):
            d = int(d)
            if d == -1:
                neg_idx = i
                resolved.append(-1)
            elif d == 0:
                # 0 means "copy from input"
                resolved.append(data_info.shape[i] if i < len(data_info.shape) else 1)
                known_product *= resolved[-1]
            else:
                resolved.append(d)
                known_product *= d

        if neg_idx is not None:
            resolved[neg_idx] = total // known_product

        out_info.shape = tuple(resolved)
