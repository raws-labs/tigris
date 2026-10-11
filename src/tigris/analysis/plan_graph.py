"""A graph rebuilt from a decoded plan, enough for the memory analysis.

The analysis reads op types, tensor sizes, shapes, layouts and quantization,
the stage table with its chains and tile plans, and the graphs control-flow
operators run. A plan carries all of that, so a requirement the compiler
derives from them can be derived again from the plan alone. Constant operands
are not plan tensors; the analysis skips them.
"""

import struct

import numpy as np

from tigris.emitters.binary.defs import (
    OP_ATTR_SUBGRAPHS,
    OP_TYPE_MAP,
    TENSOR_FLAG_CONSTANT,
    TENSOR_FLAG_LINEAR,
)
from tigris.graph.ir import (
    AnalyzedGraph,
    Layout,
    MemoryBudget,
    OpNode,
    QuantParam,
    Stage,
    TensorInfo,
    TilePlan,
    serialized_axis_map,
)
from tigris.graph.subgraphs import CONTROL_FLOW

_OP_NAMES = {code: name for name, code in OP_TYPE_MAP.items()}
_NO_TILE_PLAN = 0xFFFF
_NO_QUANT_PARAM = 0xFFFF


def _tensor(index: int, record: dict, quant_params: list[dict]) -> TensorInfo:
    layout = Layout.LINEAR if record["flags"] & TENSOR_FLAG_LINEAR else Layout.SPATIAL
    stored = record["shape"]
    axes = serialized_axis_map(len(stored), layout)
    quant = None
    if record["quant_param_idx"] != _NO_QUANT_PARAM:
        entry = quant_params[record["quant_param_idx"]]
        quant = QuantParam(scale=np.array([entry["scale"]], np.float32),
                           zero_point=np.array([entry["zero_point"]], np.int32))
    return TensorInfo(name=f"t{index}", shape=tuple(stored[axis] for axis in axes),
                      dtype=record["dtype"],
                      is_constant=bool(record["flags"] & TENSOR_FLAG_CONSTANT),
                      quant=quant, layout=layout)


def graph_from_plan(plan: dict, fast_reserve: int = 0) -> AnalyzedGraph:
    """The plan's main graph, with every graph a control-flow operator runs in
    ``subgraphs`` at the index the plan names it by. ``fast_reserve`` is the
    fast memory held outside the activation arena, the decompression buffer of
    a compressed plan."""
    tensors = {f"t{i}": _tensor(i, record, plan["quant_params"])
               for i, record in enumerate(plan["tensors"])}
    names = list(tensors)
    attrs: dict[int, dict] = {}
    for record in plan["op_attributes"]:
        if record["type"] == OP_ATTR_SUBGRAPHS:
            attrs[record["op_index"]] = record["data"]
    budget = MemoryBudget(fast=plan["budget"], fast_reserve=fast_reserve)
    ranges = plan["subgraphs"] or [{
        "first_stage": 0, "num_stages": len(plan["stages"]),
        "inputs": plan["model_inputs"], "outputs": plan["model_outputs"]}]

    graphs: list[AnalyzedGraph] = []
    for graph_range in ranges:
        records = plan["stages"][graph_range["first_stage"]:
                                 graph_range["first_stage"] + graph_range["num_stages"]]
        op_order = sorted({i for record in records for i in record["ops"]})
        local = {index: step for step, index in enumerate(op_order)}
        ops = []
        for step, index in enumerate(op_order):
            record = plan["ops"][index]
            op = OpNode(name=f"op{index}", op_type=_OP_NAMES[record["op_type"]],
                        inputs=[names[i] for i in record["inputs"]],
                        outputs=[names[i] for i in record["outputs"]], step=step)
            if op.op_type in CONTROL_FLOW and index in attrs:
                for key, graph in zip(CONTROL_FLOW[op.op_type],
                                      struct.unpack("<2H", attrs[index])):
                    op.attrs[key] = graph
            ops.append(op)
        stages = []
        first = graph_range["first_stage"]
        for position, record in enumerate(records):
            tile = None
            if record["tile_plan_idx"] != _NO_TILE_PLAN:
                entry = plan["tile_plans"][record["tile_plan_idx"]]
                tile = TilePlan(
                    tileable=entry["tileable"], axis=entry["axis"],
                    tile_height=entry["tile_height"], tile_width=entry["tile_width"],
                    num_tiles=entry["num_tiles"], halo=entry["halo"],
                    receptive_field=entry["receptive_field"],
                    original_height=entry["original_height"],
                    tiled_peak_bytes=entry["tiled_peak_bytes"],
                    overhead_bytes=entry["overhead_bytes"],
                    band_on_columns=entry["band_on_columns"])
            stage = Stage(
                stage_id=position, op_indices=[local[i] for i in record["ops"]],
                input_tensors=[names[i] for i in record["inputs"]],
                output_tensors=[names[i] for i in record["outputs"]],
                peak_bytes=record["peak_bytes"], tile_plan=tile,
                chain_id=(record["chain_id"] - first if record["chain_id"] != 0xFFFF
                          else 0xFFFF),
                chain_len=record["chain_len"], chain_tile_h=record["chain_tile_h"])
            for i in stage.op_indices:
                ops[i].stage = position
            stages.append(stage)
        inputs = [names[i] for i in graph_range["inputs"]]
        outputs = [names[i] for i in graph_range["outputs"]]
        if not graphs:
            # Variables are model inputs and outputs to the analysis; the plan
            # lists them in its state section instead.
            for entry in plan["state"]["entries"]:
                inputs.append(names[entry["input"]])
                if entry["output"] is not None:
                    outputs.append(names[entry["output"]])
        graphs.append(AnalyzedGraph(
            model_name=plan["model_name"], ops=ops, tensors=tensors,
            model_inputs=inputs, model_outputs=outputs, stages=stages, budget=budget))
    # Every graph names the graphs it runs by their place in the plan.
    for graph in graphs:
        graph.subgraphs = graphs
    return graphs[0]


def slow_arena_required(plan: dict) -> int:
    """Slow memory the plan needs: the compiler's slow-pool requirement,
    derived again from the plan at 32-byte allocations, which covers every
    runtime whose TIGRIS_TENSOR_ALIGN is at most that."""
    from tigris.analysis.validation import slow_pool_usage
    from tigris.emitters.codegen import _PLAN_TENSOR_ALIGN, _weight_decompression_overhead

    reserve = _weight_decompression_overhead(plan, _PLAN_TENSOR_ALIGN)
    return slow_pool_usage(graph_from_plan(plan, reserve)).slow_peak_bytes
