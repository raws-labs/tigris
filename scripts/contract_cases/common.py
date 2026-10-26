"""The contract case record and the model helpers every family uses."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from tigris.frontends.tflite import STATE_KEY


Array = NDArray[np.generic]


@dataclass(frozen=True)
class ContractCase:
    """A compile model, independent ORT reference model, and model inputs."""

    name: str
    compile_model: onnx.ModelProto
    reference_model: onnx.ModelProto
    inputs: dict[str, Array]
    expected_operators: tuple[str, ...]
    mem_budget: str = "4K"
    slow_budget: str | None = None
    compression: str | None = None
    xip: bool = False
    expect_tiled: bool = False
    expect_chain: bool = False
    expect_line_buffered: bool = False
    expect_2d: bool = False
    recompute_metric: bool = False
    force_one_op_stages: bool = False
    # Evaluate the reference operator by operator. ONNX Runtime's graph
    # optimizer can replace a quantized graph with a fused integer kernel
    # whose result depends on the host CPU.
    reference_unoptimized: bool = False
    exact_untiled: bool = False
    # Compare outputs byte for byte with the same model compiled without compression.
    exact_uncompressed: bool = False
    tiled_operators: tuple[str, ...] = ()
    expect_grouped: bool = False
    # Compare outputs byte for byte instead of within one quantization step.
    exact: bool = False
    # The plan reads its inputs in place, so the gate also runs it without a slow arena.
    expect_inputs_in_fast: bool = False
    # The compiler's slow figure is the runtime's slow peak, so the gate also runs the plan in it.
    exact_slow: bool = False


def _model(
    name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] = [],
    opset: int = 13,
) -> onnx.ModelProto:
    """Build a checked model. `opset` rises only for an operator that needs it."""
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 8 if opset < 17 else 9
    onnx.checker.check_model(model)
    return model


def _state_tile_case(case: ContractCase) -> ContractCase:
    """Exercise the same tiled schedule with a schema-10 state section."""
    model = copy.deepcopy(case.compile_model)
    model.graph.input.append(helper.make_tensor_value_info("saved", TensorProto.FLOAT, [1]))
    model.graph.initializer.append(numpy_helper.from_array(np.zeros(1, np.float32), "saved_initial"))
    model.graph.output.append(helper.make_tensor_value_info("saved_out", TensorProto.FLOAT, [1]))
    model.graph.node.insert(0, helper.make_node("Relu", ["saved"], ["saved_out"]))
    helper.set_model_props(model, {STATE_KEY: json.dumps([
        {"input": "saved", "output": "saved_out", "initial": "saved_initial"}])})
    return replace(case, name=case.name + "_state", compile_model=model,
                   expected_operators=("Relu", *case.expected_operators),
                   tiled_operators=case.tiled_operators or tuple(
                       kind for kind in case.expected_operators if kind not in {"Relu", "Transpose"}))


def _unread_input_case(case: ContractCase, state: bool) -> ContractCase:
    """Add an input no operator reads, as a model input or as a variable with
    no next value; it takes slow memory until the first stage ends."""
    model = copy.deepcopy(case.compile_model)
    model.graph.input.append(helper.make_tensor_value_info("unread", TensorProto.FLOAT, [1, 3]))
    inputs = dict(case.inputs)
    if state:
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros((1, 3), np.float32), "unread_initial"))
        helper.set_model_props(model, {STATE_KEY: json.dumps(
            [{"input": "unread", "output": None, "initial": "unread_initial"}])})
    else:
        inputs["unread"] = np.ones((1, 3), np.float32)
    return replace(case, name=case.name + ("_unread_state" if state else "_unread_input"),
                   compile_model=model, reference_model=case.reference_model if state else model,
                   inputs=inputs)


def _qdq(tensor: str, scale: str, zero: str, out: str) -> list[onnx.NodeProto]:
    """A QuantizeLinear/DequantizeLinear pair, the way a QDQ export marks a tensor int8."""
    return [
        helper.make_node("QuantizeLinear", [tensor, scale, zero], [f"{out}_q"]),
        helper.make_node("DequantizeLinear", [f"{out}_q", scale, zero], [out]),
    ]


def _scalars(**values: tuple[float, int]) -> list[onnx.TensorProto]:
    """Scale and int8 zero-point initializers, named <key>_s and <key>_z."""
    tensors = []
    for key, (scale, zero) in values.items():
        tensors.append(numpy_helper.from_array(
            np.array(scale, dtype=np.float32), f"{key}_s"))
        tensors.append(numpy_helper.from_array(
            np.array(zero, dtype=np.int8), f"{key}_z"))
    return tensors
