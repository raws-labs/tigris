"""Resolution of unresolved ONNX deployment shapes."""

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.loaders import load_model
from tigris.loaders.onnx.loader import resolve_shapes
from tigris.loaders.onnx.shapes import bind_free_dims


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


def _save_shape_subgraph_model(tmp_path):
    """A Relu whose output is flattened through a runtime-computed shape.

    ``Shape -> Gather -> Unsqueeze -> Concat -> Reshape`` is what exporters
    emit for a classifier head.  This is the opset-12 spelling, where
    ``Unsqueeze`` carries its axes as an attribute; ONNX data propagation
    does not carry values through it, so the Reshape output stays at unknown
    rank until the shape arithmetic is folded.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 4, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    initializers = [numpy_helper.from_array(np.array([-1], dtype=np.int64), "rest")]
    nodes = [
        helper.make_node("Relu", ["input"], ["act"], name="relu"),
        helper.make_node(
            "Constant", [], ["axis"], name="const_axis",
            value=numpy_helper.from_array(np.array(0, dtype=np.int64), "axis_value"),
        ),
        helper.make_node("Shape", ["act"], ["dims"], name="shape"),
        helper.make_node("Gather", ["dims", "axis"], ["batch"], name="gather", axis=0),
        helper.make_node("Unsqueeze", ["batch"], ["batch_1d"], name="unsqueeze", axes=[0]),
        helper.make_node("Concat", ["batch_1d", "rest"], ["target"], name="concat", axis=0),
        helper.make_node("Reshape", ["act", "target"], ["output"], name="reshape"),
    ]
    graph = helper.make_graph(nodes, "shape_subgraph", [inp], [out], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7
    path = tmp_path / "shape_subgraph.onnx"
    onnx.save(model, path)
    return path


def test_free_batch_dimension_is_bound_to_one(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    graph = load_model(path)

    assert graph.tensors["input"].shape == (1, 64)
    assert graph.tensors["output"].shape == (1, 64)
    assert graph.shape_bindings == [
        "input axis 0 (batch) has no fixed size; using 1"
    ]


def test_unknown_dimension_is_bound_to_one(tmp_path):
    path = _save_relu_model(tmp_path, [None, 64], [None, 64])

    graph = load_model(path)

    assert graph.tensors["input"].shape == (1, 64)
    assert graph.shape_bindings == [
        "input axis 0 (unknown) has no fixed size; using 1"
    ]


def test_input_shape_is_not_reported_as_a_binding(tmp_path):
    """A shape the caller named is not a dimension the compiler guessed."""
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    graph = load_model(path, {"input": (4, 64)})

    assert graph.tensors["input"].shape == (4, 64)
    assert graph.tensors["output"].shape == (4, 64)
    assert graph.shape_bindings == []


def test_input_shape_overrides_a_concrete_dimension(tmp_path):
    path = _save_relu_model(tmp_path, [1, 64], [1, 64])

    graph = load_model(path, {"input": (8, 64)})

    assert graph.tensors["input"].shape == (8, 64)
    assert graph.tensors["output"].shape == (8, 64)
    assert graph.shape_bindings == []


def test_binding_is_reported_for_an_input_no_override_names(tmp_path):
    """One input named on the command line does not silence another."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 64])
    other = helper.make_tensor_value_info("other", TensorProto.FLOAT, ["batch", 64])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    nodes = [helper.make_node("Add", ["input", "other"], ["output"], name="add")]
    graph = helper.make_graph(nodes, "two_inputs", [inp, other], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "two_inputs.onnx"
    onnx.save(model, path)

    ag = load_model(path, {"input": (4, 64)})

    assert ag.tensors["input"].shape == (4, 64)
    assert ag.shape_bindings == [
        "other axis 0 (batch) has no fixed size; using 1"
    ]


def test_input_shape_of_wrong_rank_is_rejected(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    with pytest.raises(ValueError, match=r"has rank 2, but the given shape has rank 3"):
        load_model(path, {"input": (1, 1, 64)})


def test_input_shape_for_unknown_input_is_rejected(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    with pytest.raises(ValueError, match=r"No such model input: images"):
        load_model(path, {"images": (1, 64)})


def test_concrete_shape_loads_without_a_binding(tmp_path):
    path = _save_relu_model(tmp_path, [1, 64], [1, 64])

    graph = load_model(path)

    assert graph.tensors["input"].shape == (1, 64)
    assert graph.shape_bindings == []


def test_unknown_rank_input_is_still_rejected(tmp_path):
    path = _save_relu_model(tmp_path, None, None)

    with pytest.raises(ValueError, match=r"Tensor 'input' has unknown rank"):
        load_model(path)


def test_shape_subgraph_is_folded(tmp_path):
    path = _save_shape_subgraph_model(tmp_path)
    model = onnx.load(str(path))

    bindings = resolve_shapes(model)

    assert bindings == ["input axis 0 (batch) has no fixed size; using 1"]
    remaining = {node.op_type for node in model.graph.node}
    assert remaining == {"Relu", "Reshape"}
    shapes = {vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)}
    dims = shapes["output"].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 16]


def test_binding_alone_leaves_the_shape_subgraph_unresolved(tmp_path):
    """The fold is required: binding the input dimensions does not suffice."""
    path = _save_shape_subgraph_model(tmp_path)
    model = onnx.load(str(path))

    bind_free_dims(model)
    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)

    shapes = {vi.name: vi for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
    assert not shapes["output"].type.tensor_type.HasField("shape")


def test_folded_model_loads_end_to_end(tmp_path):
    path = _save_shape_subgraph_model(tmp_path)

    graph = load_model(path)

    assert graph.tensors["output"].shape == (1, 16)


def test_cli_warns_about_a_bound_dimension(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    result = CliRunner().invoke(cli, ["analyze", str(path), "-m", "4K"])

    assert result.exit_code == 0, result.output
    assert "input axis 0 (batch) has no fixed size; using 1" in result.output
    assert "--input-shape" in result.output


def test_cli_echoes_an_input_shape_override_without_warning(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    result = CliRunner().invoke(
        cli, ["analyze", str(path), "-m", "4K", "--input-shape", "input:4x64"]
    )

    assert result.exit_code == 0, result.output
    assert "input compiled for 4x64" in result.output
    assert "warning" not in result.output
    assert "--input-shape" not in result.output


def test_cli_does_not_warn_when_an_override_pins_a_concrete_shape(tmp_path):
    path = _save_relu_model(tmp_path, [1, 64], [1, 64])

    result = CliRunner().invoke(
        cli, ["analyze", str(path), "-m", "4K", "--input-shape", "input:8x64"]
    )

    assert result.exit_code == 0, result.output
    assert "input compiled for 8x64" in result.output
    assert "has no fixed size" not in result.output


def test_cli_rejects_a_malformed_input_shape(tmp_path):
    path = _save_relu_model(tmp_path, ["batch", 64], ["batch", 64])

    result = CliRunner().invoke(
        cli, ["analyze", str(path), "-m", "4K", "--input-shape", "input:1xNx64"]
    )

    assert result.exit_code != 0
    assert "dimensions must be integers" in result.output
    assert "Traceback" not in result.output


def test_cli_reports_an_unresolvable_shape_without_traceback(tmp_path):
    path = _save_relu_model(tmp_path, None, None)

    result = CliRunner().invoke(cli, ["analyze", str(path), "-m", "4K"])

    assert result.exit_code != 0
    assert "Error: Tensor 'input' has unknown rank" in result.output
    assert "Traceback" not in result.output
