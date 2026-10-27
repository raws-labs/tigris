"""A graph rebuilt from a decoded plan, enough for the memory analysis.

The analysis reads op types, tensor sizes, shapes, layouts and quantization,
the stage table with its chains and tile plans, and the graphs control-flow
operators run. A plan carries all of that, so a requirement the compiler
derives from them can be derived again from the plan alone. Constant operands
are not plan tensors; the analysis skips them.
"""

import functools
import struct

import numpy as np

from tigris import TILE_AXIS_HW
from tigris.analysis.memory import TENSOR_ALIGN, align_up
from tigris.emitters.binary.defs import (
    NO_CHAIN,
    NO_QUANT_PARAM,
    NO_TILE_PLAN,
    OP_ATTR_SUBGRAPHS,
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
    StatePort,
    TensorInfo,
    TilePlan,
    serialized_axis_map,
)
from tigris.graph.subgraphs import CONTROL_FLOW
from tigris.operator_contract import OP_TYPE_BY_CODE, operators_with

# The defaults of tigris_config.h, in the order build_limits_needed states them.
BUILD_LIMIT_DEFAULTS = {
    "TIGRIS_MAX_TENSORS": 512,
    "TIGRIS_MAX_STAGE_INPUTS": 16,
    "TIGRIS_MAX_STAGE_OUTPUTS": 16,
    "TIGRIS_MAX_CHAIN_STAGES": 16,
    "TIGRIS_MAX_SPATIAL_OPS_PER_STAGE": 8,
}

# The operators the executor reserves per-op tiling state for in a chain
# (tigris_height_spatial_op in the runtime).
_HEIGHT_SPATIAL_OPS = operators_with("spatial")


def _tensor(index: int, record: dict, quant_params: list[dict]) -> TensorInfo:
    layout = Layout.LINEAR if record["flags"] & TENSOR_FLAG_LINEAR else Layout.SPATIAL
    stored = record["shape"]
    axes = serialized_axis_map(len(stored), layout)
    quant = None
    if record["quant_param_idx"] != NO_QUANT_PARAM:
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
            op = OpNode(name=f"op{index}", op_type=OP_TYPE_BY_CODE[record["op_type"]],
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
            if record["tile_plan_idx"] != NO_TILE_PLAN:
                entry = plan["tile_plans"][record["tile_plan_idx"]]
                tile = TilePlan(
                    tileable=entry["tileable"], axis=entry["axis"],
                    tiled_peak_bytes=entry["tiled_peak_bytes"],
                    band_on_columns=entry["band_on_columns"], **_tile_fields(entry))
            stage = Stage(
                stage_id=position, op_indices=[local[i] for i in record["ops"]],
                input_tensors=[names[i] for i in record["inputs"]],
                output_tensors=[names[i] for i in record["outputs"]],
                peak_bytes=record["peak_bytes"], tile_plan=tile,
                chain_id=(record["chain_id"] - first if record["chain_id"] != NO_CHAIN
                          else NO_CHAIN),
                chain_len=record["chain_len"], chain_tile_h=record["chain_tile_h"])
            for i in stage.op_indices:
                ops[i].stage = position
            stages.append(stage)
        inputs = [names[i] for i in graph_range["inputs"]]
        outputs = [names[i] for i in graph_range["outputs"]]
        ports = []
        if not graphs:
            # Variables are model I/O to the analysis; the plan lists them, with initial values, as state.
            for entry in plan["state"]["entries"]:
                inputs.append(names[entry["input"]])
                output = None
                if entry["output"] is not None:
                    outputs.append(names[entry["output"]])
                    output = len(outputs) - 1
                ports.append(StatePort(input=len(inputs) - 1, output=output,
                                       initial=np.frombuffer(entry["initial"], np.uint8)))
        graphs.append(AnalyzedGraph(
            model_name=plan["model_name"], ops=ops, tensors=tensors,
            model_inputs=inputs, model_outputs=outputs, stages=stages, budget=budget,
            state_ports=ports, schema_version=plan["version"]))
    # Every graph names the graphs it runs by their place in the plan.
    for graph in graphs:
        graph.subgraphs = graphs
    return graphs[0]


def _tile_fields(entry: dict) -> dict:
    """The TilePlan fields a tile record states; any other keeps its default."""
    if "input_band_extent" not in entry:
        return {key: entry[key] for key in (
            "tile_height", "tile_width", "num_tiles", "halo", "receptive_field",
            "original_height", "overhead_bytes")}
    grouped = entry["input_channel_group"] != 0
    fields = {"tile_width": entry["output_width_extent"],
              "input_channel_group": entry["input_channel_group"],
              "input_spatial_interval": entry["input_spatial_interval"],
              "band_in_output_rows": entry["output_band_extent"] != 0 and entry["axis"] != TILE_AXIS_HW}
    if entry["tile_extent"] is None:
        # A schema 10 record states its scheduling choices only.
        fields["tile_height"] = 0 if grouped else (
            entry["input_band_extent"] or entry["output_band_extent"])
        return fields
    # A grouped reduction's tile_extent is its interval, which TilePlan keeps apart from tile_height.
    fields.update(tile_height=0 if grouped else entry["tile_extent"], num_tiles=entry["num_tiles"],
                  original_height=entry["original_extent"], halo=entry["halo"],
                  band_axis=entry["band_axis"], band_kind=entry["band_kind"], band_mode=entry["band_mode"])
    return fields


def fast_activation_bytes(plan: dict) -> int:
    """Fast-arena bytes the plan's activations need: the scheduled peak the
    plan states from schema 11 on, its budget before."""
    return plan["budget"] if plan["scheduled_peak"] is None else plan["scheduled_peak"]


def decompression_reserve(blocks, stages, alignment: int = TENSOR_ALIGN) -> int:
    """The fast bytes tigris_weight_decompression_overhead() holds for
    decompressed weights: the largest aligned block, or for a chain the
    aligned first block of each member stage together. ``blocks`` are
    (stage index, uncompressed size, compressed size) records; one whose
    compressed size is 0 is skipped, as the runtime skips it. ``stages`` are
    (chain_id, chain_len) records in stage-table order."""
    reserve = 0
    first: dict[int, int] = {}
    for stage_idx, uncompressed, compressed in blocks:
        if compressed == 0:
            continue
        size = align_up(uncompressed, alignment)
        reserve = max(reserve, size)
        first.setdefault(stage_idx, size)
    for head, (chain_id, chain_len) in enumerate(stages):
        if chain_len < 2 or chain_id != head or chain_len > len(stages) - head:
            continue
        reserve = max(reserve, sum(first.get(member, 0)
                                   for member in range(head, head + chain_len)))
    return reserve


def plan_decompression_reserve(plan: dict, alignment: int = TENSOR_ALIGN) -> int:
    """decompression_reserve for a decoded plan."""
    return decompression_reserve(
        [(block["stage_idx"], block["uncompressed_size"], block["compressed_size"])
         for block in plan.get("weight_blocks", [])],
        [(stage["chain_id"], stage["chain_len"]) for stage in plan.get("stages", [])],
        alignment)


def slow_arena_required(plan: dict) -> int:
    """Slow memory the plan needs: the compiler's slow-pool requirement,
    derived again from the plan at TENSOR_ALIGN allocations."""
    from tigris.analysis.validation import slow_pool_usage

    reserve = plan_decompression_reserve(plan)
    return slow_pool_usage(graph_from_plan(plan, reserve)).slow_peak_bytes


def _chains(plan: dict) -> list[list[dict]]:
    """The stages of each chain, from its head."""
    stages = plan.get("stages", [])
    chains = []
    for stage_idx, stage in enumerate(stages):
        chain_len = stage["chain_len"]
        if chain_len < 2 or stage["chain_id"] != stage_idx:
            continue
        if stage_idx + chain_len > len(stages):
            raise ValueError("Plan contains a chain beyond the stage table")
        chains.append(stages[stage_idx : stage_idx + chain_len])
    return chains


def _spatial_ops(plan: dict, stage: dict) -> int:
    ops = plan.get("ops", [])
    return sum(OP_TYPE_BY_CODE.get(ops[op_idx]["op_type"]) in _HEIGHT_SPATIAL_OPS
               for op_idx in stage["ops"])


def _subgraph_depth(plan: dict) -> int:
    """How many control-flow operators run one inside another at most."""
    graphs = plan.get("subgraphs", [])
    if not graphs:
        return 0
    runs = {attr["op_index"]: attr["data"] for attr in plan.get("op_attributes", [])
            if attr["type"] == OP_ATTR_SUBGRAPHS}
    stages = plan["stages"]

    @functools.cache
    def height(graph: int) -> int:
        first, count = graphs[graph]["first_stage"], graphs[graph]["num_stages"]
        children = [int.from_bytes(runs[op][slot:slot + 2], "little")
                    for stage in stages[first:first + count] for op in stage["ops"]
                    if op in runs for slot in (0, 2)]
        return max((height(child) + 1 for child in children), default=0)

    return height(0)


def build_limits_needed(plan: dict) -> tuple[int, int, int, int, int]:
    """The build limits the plan asks for, in tigris_config.h's order."""
    stages = plan.get("stages", [])
    chains = _chains(plan)
    return (
        plan["num_tensors"],
        max((len(stage["inputs"]) for stage in stages), default=0),
        max((len(stage["outputs"]) for stage in stages), default=0),
        max((len(chain) for chain in chains), default=0),
        max((_spatial_ops(plan, stage) for chain in chains for stage in chain), default=0),
    )


def executor_workspace_limits(plan: dict) -> tuple[int, int, int, int, int, int]:
    """The arguments of TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS for the
    plan: its tensors, the most inputs and outputs of a stage, the longest
    chain, the most spatial operators in one chain and the nesting depth."""
    tensors, inputs, outputs, chain, _ = build_limits_needed(plan)
    chain_spatial = max((sum(_spatial_ops(plan, stage) for stage in chain)
                         for chain in _chains(plan)), default=0)
    return tensors, inputs, outputs, chain, chain_spatial, _subgraph_depth(plan)
