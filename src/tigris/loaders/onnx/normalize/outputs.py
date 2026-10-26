"""The model outputs under the names the graph declares."""

from dataclasses import replace

from tigris.graph.ir import AnalyzedGraph, OpNode


def _restore_output_names(ag: AnalyzedGraph, declared: list[str]) -> AnalyzedGraph:
    """Give each model output back the name the ONNX graph declares.

    Removing an Identity, or the Q/DQ pair behind an output, makes the tensor
    that produced it the output under its own name. The interface keeps the
    declared name instead. One tensor cannot carry two names, so where that
    tensor is also a model input or an earlier output, the declared output is
    lowered to a same-shape copy: a Reshape that writes the tensor into a new
    one under the declared name. The copy runs after every other operator. It
    shares the last stage when that stage can take it within the fast budget,
    and runs in a stage of its own otherwise, as when the last stage tiles. In
    a plan whose outputs go to slow memory, the copy has a slow buffer of its
    own, and its stage loads the source from slow memory unless it computes
    the source or already reads it. A variable whose next value is its own
    input gets no next value at all.
    """
    if len(declared) != len(ag.model_outputs):
        return ag
    declared = _drop_unchanged_state_outputs(ag, declared)
    taken = set(ag.model_inputs)
    for position, name in enumerate(declared):
        current = ag.model_outputs[position]
        if current == name or name in ag.tensors or current not in ag.tensors:
            taken.add(current)
            continue
        if current in taken:
            ag.tensors[name] = replace(ag.tensors[current], name=name)
            ag.ops.append(OpNode(name=f"{name}_copy", op_type="Reshape", inputs=[current],
                                 outputs=[name], step=len(ag.ops)))
            ag.model_outputs[position] = name
            taken.add(name)
            continue
        info = ag.tensors.pop(current)
        info.name = name
        ag.tensors[name] = info
        for op in ag.ops:
            op.inputs = [name if n == current else n for n in op.inputs]
            op.outputs = [name if n == current else n for n in op.outputs]
        ag.model_outputs = [name if n == current else n for n in ag.model_outputs]
        taken.add(name)
    return ag


def _drop_unchanged_state_outputs(ag: AnalyzedGraph, declared: list[str]) -> list[str]:
    """A variable whose next value is its own input keeps its value.

    A plan states that by giving the variable no next value, since one tensor
    cannot hold the variable both on entry and on exit. Returns the declared
    names of the outputs that remain, in order.
    """
    unchanged = {port.output for port in ag.state_ports if port.output is not None
                 and ag.model_outputs[port.output] == ag.model_inputs[port.input]}
    if not unchanged:
        return declared
    kept = [position for position in range(len(ag.model_outputs)) if position not in unchanged]
    moved = {old: new for new, old in enumerate(kept)}
    for port in ag.state_ports:
        port.output = moved.get(port.output)
    if len(ag.model_output_dtypes) == len(ag.model_outputs):
        ag.model_output_dtypes = [ag.model_output_dtypes[position] for position in kept]
    ag.model_outputs = [ag.model_outputs[position] for position in kept]
    return [declared[position] for position in kept]
