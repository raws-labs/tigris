"""One plan from a graph and the subgraphs its control-flow operators run."""

from dataclasses import dataclass, replace

from tigris.graph.ir import AnalyzedGraph

# Operators that run subgraphs, by the attributes naming them.
CONTROL_FLOW = {"If": ("then_branch", "else_branch"), "While": ("cond_branch", "body_branch")}


@dataclass(frozen=True)
class SubgraphRange:
    """A subgraph's stages and its input and output tensors in the merged plan."""

    first_stage: int
    num_stages: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def flatten_subgraphs(ag: AnalyzedGraph) -> tuple[AnalyzedGraph, list[SubgraphRange]]:
    """The main graph with every subgraph's tensors, weights, operators and
    stages appended after its own, names prefixed so they cannot collide, and
    the stage range of each graph; range 0 is the main graph's. Each graph's
    tensors stay one block, which the runtime requires of the tensor table."""
    ops = list(ag.ops)
    stages = list(ag.stages)
    tensors = dict(ag.tensors)
    weights = dict(ag.weight_data)
    ranges = [SubgraphRange(0, len(ag.stages), tuple(ag.model_inputs), tuple(ag.model_outputs))]
    for position, sub in enumerate(ag.subgraphs, start=1):
        if sub.subgraphs:
            raise ValueError("control flow inside a subgraph is not supported")
        prefix = f"sg{position}/"
        names = {name: prefix + name for name in sub.tensors}
        for name, info in sub.tensors.items():
            tensors[names[name]] = replace(info, name=names[name])
        for name, array in sub.weight_data.items():
            weights[names.get(name, prefix + name)] = array
        op_offset, stage_offset = len(ops), len(stages)
        for op in sub.ops:
            ops.append(replace(op, inputs=[names.get(n, n) for n in op.inputs],
                               outputs=[names.get(n, n) for n in op.outputs],
                               stage=op.stage + stage_offset if op.stage >= 0 else op.stage))
        for stage in sub.stages:
            stages.append(replace(
                stage,
                stage_id=stage.stage_id + stage_offset,
                op_indices=[i + op_offset for i in stage.op_indices],
                input_tensors=[names.get(n, n) for n in stage.input_tensors],
                output_tensors=[names.get(n, n) for n in stage.output_tensors],
                chain_id=stage.chain_id + stage_offset if stage.chain_id != 0xFFFF else 0xFFFF))
        ranges.append(SubgraphRange(stage_offset, len(sub.stages),
                                    tuple(names[n] for n in sub.model_inputs),
                                    tuple(names[n] for n in sub.model_outputs)))
    merged = replace(ag, ops=ops, stages=stages, tensors=tensors, weight_data=weights,
                     subgraphs=[])
    return merged, ranges


def control_flow_cuts(ag: AnalyzedGraph) -> frozenset[int]:
    """Operator indices that start a stage so each control-flow operator has
    its stage to itself."""
    cuts = set()
    for index, op in enumerate(ag.ops):
        if op.op_type in CONTROL_FLOW:
            cuts.update((index, index + 1))
    return frozenset(cuts)

