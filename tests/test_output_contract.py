"""Observable model-output contract validation."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.loaders import load_model
from tigris.loaders.onnx.loader import load_model as load_raw_model

from conftest import save_onnx


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
    return save_onnx(tmp_path, "trailing_transpose", nodes, [model_input], [model_output],
                     ir_version=8, check=True)


def test_trailing_transpose_is_observable_and_preserved(trailing_transpose_path):
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

    graph = load_model(trailing_transpose_path)
    graph = compute_lifetimes(graph)
    graph = compute_memory_timeline(graph)
    graph = partition_temporal(graph, 4096)
    graph = partition_spatial(graph)
    plan = read_binary_plan(emit_binary_bytes(graph))

    output_idx = plan["model_outputs"][0]
    assert plan["tensors"][output_idx]["shape"] == [3, 1, 2]
    assert plan["op_attributes"] == [
        {"op_index": 1, "type": 1, "data": b"\x01\x00\x02"}
    ]


def test_terminal_rank4_transpose_remaps_internal_layout(tmp_path):
    """The permutation payload must bridge NCHW internals to raw output I/O."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, 2, 3]
    )
    path = save_onnx(tmp_path, "rank4_trailing_transpose",
                     [helper.make_node("Relu", ["input"], ["pretranspose"], name="relu"),
                      helper.make_node("Transpose", ["pretranspose"], ["output"],
                                       name="output_transpose", perm=[0, 3, 1, 2])], [model_input],
                     [model_output], ir_version=8)

    compiled = load_model(path)
    compiled = compute_lifetimes(compiled)
    compiled = compute_memory_timeline(compiled)
    compiled = partition_temporal(compiled, 4096)
    compiled = partition_spatial(compiled)
    plan = read_binary_plan(emit_binary_bytes(compiled))

    output_idx = plan["model_outputs"][0]
    assert plan["tensors"][output_idx]["shape"] == [1, 4, 2, 3]
    assert plan["op_attributes"] == [
        {"op_index": 1, "type": 1, "data": b"\x00\x02\x03\x01"}
    ]
