"""The int8 interface: quantized model boundaries handed over in the plan's encoding."""

import onnx
import pytest
from click.testing import CliRunner

from tigris.cli import cli
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.pipeline import plan_model
from tigris.utils import describe_interface


def _boundaries(graph, **emit):
    """(stored dtype, interface dtype) of each model input, then each output."""
    plan = read_binary_plan(emit_binary_bytes(graph, **emit))
    return [(plan["tensors"][index]["dtype"], plan["tensors"][index]["iface_dtype"])
            for index in plan["model_inputs"] + plan["model_outputs"]]


def _saved(tmp_path, model):
    path = tmp_path / f"{model.graph.name}.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("compress", [None, "lz4"])
def test_quantized_boundaries_are_handed_over_as_int8(qdq_conv_path, compress):
    float_io = plan_model(qdq_conv_path, ("4K",), compress=compress)
    int8_io = plan_model(qdq_conv_path, ("4K",), compress=compress, int8_io=True)
    assert _boundaries(float_io, compress=compress) == [(3, 1), (3, 1)]
    assert _boundaries(int8_io, compress=compress) == [(3, 0), (3, 0)]
    # The encoding is the one the plan stores: same scale and zero point.
    assert (read_binary_plan(emit_binary_bytes(int8_io))["quant_params"]
            == read_binary_plan(emit_binary_bytes(float_io))["quant_params"])
    assert describe_interface(int8_io) == [
        ("Input", "input 1x1x4x4 int8 at scale 0.05, zero point 0"),
        ("Output", "output 1x2x2x2 int8 at scale 0.1, zero point 0")]


def test_an_index_output_keeps_its_declared_dtype(tmp_path):
    from scripts.contract_cases.reduction import _arg_case

    graph = plan_model(_saved(tmp_path, _arg_case("ArgMax", 1, True, False).compile_model),
                       ("4K",), int8_io=True)
    assert _boundaries(graph) == [(3, 0), (6, 7)]


def test_a_float_output_keeps_its_declared_dtype(tmp_path):
    from scripts.contract_cases.qdq import _qdq_float_section_case

    graph = plan_model(_saved(tmp_path, _qdq_float_section_case("conv", True).compile_model),
                       ("128K",), int8_io=True)
    assert _boundaries(graph) == [(3, 0), (1, 0)]


def test_a_float_model_is_refused(linear_3op_path):
    with pytest.raises(ValueError) as excinfo:
        plan_model(linear_3op_path, ("64K",), int8_io=True)
    assert str(excinfo.value) == (
        "int8 interface changes nothing: no model input or output is stored as quantized int8")


def test_a_model_declaring_int8_is_refused(tmp_path):
    from scripts.contract_cases.qdq import _int8_interface_case

    path = _saved(tmp_path, _int8_interface_case().reference_model)
    assert _boundaries(plan_model(path, ("4K",))) == [(3, 0), (3, 0)]
    with pytest.raises(ValueError) as excinfo:
        plan_model(path, ("4K",), int8_io=True)
    assert str(excinfo.value) == (
        "int8 interface changes nothing: every quantized model input and output "
        "is already declared int8")


def test_analyze_reports_the_int8_interface(qdq_conv_path, linear_3op_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(qdq_conv_path), "-m", "4K", "--int8-io"])
    assert result.exit_code == 0, result.output
    assert "input    input    1x1x4x4 int8 at scale 0.05, zero point 0\n" in result.output
    assert "output   output   1x2x2x2 int8 at scale 0.1, zero point 0\n" in result.output
    result = runner.invoke(cli, ["compile", str(linear_3op_path), "-m", "64K", "--int8-io"])
    assert result.exit_code == 1
    assert result.output == ("Error: int8 interface changes nothing: no model input or output "
                             "is stored as quantized int8\n")


def _stateful_qdq(tmp_path):
    """x + state, both quantized at the boundary, kept as state and returned as y."""
    import json

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    from tigris.frontends.tflite import STATE_KEY

    def qdq(source, name):
        return [helper.make_node("QuantizeLinear", [source, "s", "z"], [name + "_q"]),
                helper.make_node("DequantizeLinear", [name + "_q", "s", "z"], [name])]

    nodes = [*qdq("x", "xd"), *qdq("state_in", "sd"),
             helper.make_node("Add", ["xd", "sd"], ["sum"]),
             helper.make_node("QuantizeLinear", ["sum", "s", "z"], ["sum_q"]),
             helper.make_node("DequantizeLinear", ["sum_q", "s", "z"], ["y"]),
             helper.make_node("DequantizeLinear", ["sum_q", "s", "z"], ["state_out"])]
    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        nodes, "stateful_qdq", [value("x"), value("state_in")], [value("y"), value("state_out")],
        [numpy_helper.from_array(np.float32(0.125), "s"), numpy_helper.from_array(np.int8(0), "z"),
         numpy_helper.from_array(np.zeros((1, 4), np.float32), "state_initial")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    helper.set_model_props(model, {STATE_KEY: json.dumps(
        [{"input": "state_in", "output": "state_out", "initial": "state_initial"}])})
    return _saved(tmp_path, model)


def test_a_state_variable_keeps_its_declared_dtype(tmp_path):
    path = _stateful_qdq(tmp_path)

    def interfaces(**options):
        plan = read_binary_plan(emit_binary_bytes(plan_model(path, ("4K",), **options)))
        return {tensor["name"]: (tensor["dtype"], tensor["iface_dtype"]) for tensor in plan["tensors"]}

    assert interfaces() == {name: (3, 1) for name in ("x", "state_in", "y", "state_out")}
    assert interfaces(int8_io=True) == {"x": (3, 0), "state_in": (3, 1), "y": (3, 0),
                                        "state_out": (3, 1)}
