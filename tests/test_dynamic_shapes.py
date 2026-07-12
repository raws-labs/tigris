"""Fail-closed handling for unresolved ONNX deployment shapes."""

import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from tigris.cli import cli
from tigris.loaders import load_model


def _save_relu_model(tmp_path, input_shape, output_shape):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    relu = helper.make_node("Relu", ["input"], ["output"], name="relu")
    graph = helper.make_graph([relu], "dynamic_relu", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "dynamic_relu.onnx"
    onnx.save(model, path)
    return path


def test_symbolic_input_dimension_is_rejected(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    with pytest.raises(
        ValueError,
        match=r"Tensor 'input' dimension 0 has symbolic value 'batch'",
    ):
        load_model(path)


def test_unknown_input_dimension_is_rejected(tmp_path):
    path = _save_relu_model(tmp_path, [None, 64], [None, 64])

    with pytest.raises(
        ValueError,
        match=r"Tensor 'input' dimension 0 has unknown value",
    ):
        load_model(path)


def test_concrete_shape_still_loads(tmp_path):
    path = _save_relu_model(tmp_path, [1, 64], [1, 64])

    graph = load_model(path)

    assert graph.tensors["input"].shape == (1, 64)
    assert graph.tensors["output"].shape == (1, 64)


def test_cli_reports_dynamic_shape_without_traceback(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    result = CliRunner().invoke(cli, ["analyze", str(path), "-m", "4K"])

    assert result.exit_code != 0
    assert "Error: Tensor 'input' dimension 0 has symbolic value 'batch'" in result.output
    assert "Traceback" not in result.output
