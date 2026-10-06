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
    """The main graph and every subgraph below it in one graph: each graph's
    tensors, weights, operators and stages appended after those of the graphs
    placed before it, names prefixed so they cannot collide, and the stage range
    of each graph, the main graph first. A graph is placed before the subgraphs
    it runs, and a control-flow operator names them by their place. Each
    graph's tensors stay one block, which the runtime requires of the tensor
    table."""
    ops: list = []
    stages: list = []
    tensors: dict = {}
    weights: dict = {}
    ranges: list = []

    def place(graph: AnalyzedGraph) -> int:
        position = len(ranges)
        ranges.append(None)
        prefix = f"sg{position}/" if position else ""
        names = {name: prefix + name for name in graph.tensors}
        for name, info in graph.tensors.items():
            tensors[names[name]] = replace(info, name=names[name])
        for name, array in graph.weight_data.items():
            weights[names.get(name, prefix + name)] = array
        op_offset, stage_offset = len(ops), len(stages)
        placed = [replace(op, inputs=[names.get(n, n) for n in op.inputs],
                          outputs=[names.get(n, n) for n in op.outputs], attrs=dict(op.attrs),
                          stage=op.stage + stage_offset if op.stage >= 0 else op.stage)
                  for op in graph.ops]
        ops.extend(placed)
        for stage in graph.stages:
            stages.append(replace(
                stage,
                stage_id=stage.stage_id + stage_offset,
                op_indices=[i + op_offset for i in stage.op_indices],
                input_tensors=[names.get(n, n) for n in stage.input_tensors],
                output_tensors=[names.get(n, n) for n in stage.output_tensors],
                chain_id=stage.chain_id + stage_offset if stage.chain_id != 0xFFFF else 0xFFFF))
        ranges[position] = SubgraphRange(stage_offset, len(graph.stages),
                                         tuple(names[n] for n in graph.model_inputs),
                                         tuple(names[n] for n in graph.model_outputs))
        for op in placed:
            for key in CONTROL_FLOW.get(op.op_type, ()):
                op.attrs[key] = place(graph.subgraphs[op.attrs[key]])
        return position

    place(ag)
    merged = replace(ag, ops=ops, stages=stages, tensors=tensors, weight_data=weights,
                     subgraphs=[])
    return merged, ranges


def all_graphs(ag: AnalyzedGraph) -> list[AnalyzedGraph]:
    """The graph and every subgraph below it."""
    return [ag, *(graph for sub in ag.subgraphs for graph in all_graphs(sub))]


def control_flow_cuts(ag: AnalyzedGraph) -> frozenset[int]:
    """Operator indices that start a stage so each control-flow operator has
    its stage to itself."""
    cuts = set()
    for index, op in enumerate(ag.ops):
        if op.op_type in CONTROL_FLOW:
            cuts.update((index, index + 1))
    return frozenset(cuts)

