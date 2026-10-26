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


def _vi(name: str, shape: list[int], dtype: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    """A graph input or output, float unless `dtype` says otherwise."""
    return helper.make_tensor_value_info(name, dtype, shape)


def _qdq(tensor: str, scale: str, zero: str, out: str, quantized: str | None = None,
         **attrs) -> list[onnx.NodeProto]:
    """A QuantizeLinear/DequantizeLinear pair, the way a QDQ export marks a tensor int8.

    The int8 tensor is named `quantized`, <out>_q unless given; `attrs` go on both nodes.
    """
    quantized = quantized or f"{out}_q"
    return [
        helper.make_node("QuantizeLinear", [tensor, scale, zero], [quantized], **attrs),
        helper.make_node("DequantizeLinear", [quantized, scale, zero], [out], **attrs),
    ]


def _qparams(scale_name: str, zero_name: str, scale: float | list[float], zero: int | list[int],
             dtype: type = np.int8) -> list[onnx.TensorProto]:
    """A scale and a zero-point initializer; a list makes a 1-element or per-channel tensor."""
    return [
        numpy_helper.from_array(np.array(scale, dtype=np.float32), scale_name),
        numpy_helper.from_array(np.array(zero, dtype=dtype), zero_name),
    ]


def _scalars(**values: tuple[float, int]) -> list[onnx.TensorProto]:
    """Scale and int8 zero-point initializers, named <key>_s and <key>_z."""
    return [tensor for key, (scale, zero) in values.items()
            for tensor in _qparams(f"{key}_s", f"{key}_z", scale, zero)]


def _qdq_named(prefix: str, tensor: str | None = None, out: str | None = None,
               **attrs) -> list[onnx.NodeProto]:
    """`_qdq` over <prefix>_scale and <prefix>_zero_point through <prefix>_q, from `tensor`
    (default `prefix`) to `out` (default <prefix>_dq)."""
    return _qdq(tensor or prefix, f"{prefix}_scale", f"{prefix}_zero_point",
                out or f"{prefix}_dq", f"{prefix}_q", **attrs)


def _qdq_operator_model(
    graph_name: str, op_type: str, input_shape: list[int], output_shape: list[int],
    input_q: tuple, output_q: tuple, weight: Array | None = None, weight_q: tuple = (),
    weight_axis: int | None = None, **attrs,
) -> onnx.ModelProto:
    """input -> Q/DQ -> `op_type` (with a Q/DQ `weight` operand when given) -> raw -> Q/DQ ->
    output, with the names of `_qdq_named`. `attrs` go on the operator node, `name` included."""
    initializers = (_qparams("input_scale", "input_zero_point", *input_q)
                    + _qparams("output_scale", "output_zero_point", *output_q))
    nodes, operands = _qdq_named("input"), ["input_dq"]
    if weight is not None:
        initializers += [numpy_helper.from_array(weight, "weight"),
                         *_qparams("weight_scale", "weight_zero_point", *weight_q)]
        nodes += _qdq_named("weight", axis=weight_axis)
        operands.append("weight_dq")
    nodes += [helper.make_node(op_type, operands, ["raw"], **attrs),
              *_qdq_named("output", "raw", "output")]
    return _model(graph_name, nodes, [_vi("input", input_shape)], [_vi("output", output_shape)],
                  initializers)
