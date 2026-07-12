"""Observable model-output contract validation."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from tigris.cli import cli
from tigris.loaders import load_model
from tigris.loaders.onnx.loader import load_model as load_raw_model


@pytest.fixture
def trailing_transpose_path(tmp_path):
    """A model whose final permutation is not an internal layout conversion."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [3, 1, 2]
    )
    nodes = [
        helper.make_node("Relu", ["input"], ["pretranspose"], name="relu"),
        helper.make_node(
            "Transpose",
            ["pretranspose"],
            ["output"],
            name="output_transpose",
            perm=[2, 0, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes, "trailing_transpose", [model_input], [model_output]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = tmp_path / "trailing_transpose.onnx"
    onnx.save(model, path)
    return path


def test_trailing_transpose_is_observable_and_rejected(trailing_transpose_path):
    model_input = np.array(
        [[[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]], dtype=np.float32
    )
    pretranspose = np.maximum(model_input, 0.0)
    expected = ort.InferenceSession(
        str(trailing_transpose_path), providers=["CPUExecutionProvider"]
    ).run(None, {"input": model_input})[0]

    assert pretranspose.shape == (1, 2, 3)
    assert expected.shape == (3, 1, 2)
    np.testing.assert_array_equal(expected, np.transpose(pretranspose, (2, 0, 1)))

    raw_graph = load_raw_model(trailing_transpose_path)
    assert raw_graph.model_outputs == ["output"]
    assert raw_graph.tensors["output"].shape == expected.shape
    assert raw_graph.ops[-1].op_type == "Transpose"

    with pytest.raises(ValueError, match="Cannot preserve model output 'output'"):
        load_model(trailing_transpose_path)


def test_compile_refuses_trailing_transpose_without_output(
    trailing_transpose_path, tmp_path
):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(
        cli,
        [
            "compile",
            str(trailing_transpose_path),
            "-m",
            "4K",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "Cannot preserve model output 'output'" in result.output
    assert "perm=[2, 0, 1]" in result.output
    assert "shape (1, 2, 3) to (3, 1, 2)" in result.output
    assert not output.exists()
