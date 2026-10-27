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
