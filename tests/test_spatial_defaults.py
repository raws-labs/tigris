"""Regression tests for ONNX spatial attributes omitted at their default."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.loaders import load_model


def _plan(path):
    graph = load_model(path)
    graph = compute_lifetimes(graph)
    graph = compute_memory_timeline(graph)
    graph = partition_temporal(graph, 4096)
    graph = partition_spatial(graph)
    return read_binary_plan(emit_binary_bytes(graph))


def test_onnx_default_conv_stride_and_dilation_emit_as_one(tmp_path):
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 5, 5]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    weights = numpy_helper.from_array(
        np.ones((1, 1, 2, 2), dtype=np.float32), "weights"
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Conv", ["input", "weights"], ["output"])],
            "default_spatial_attrs",
            [model_input],
            [model_output],
            [weights],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    path = tmp_path / "default_spatial_attrs.onnx"
    onnx.save(model, path)

    plan = _plan(path)

    assert plan["ops"][0]["spatial"]["stride_h"] == 1
    assert plan["ops"][0]["spatial"]["stride_w"] == 1
    assert plan["ops"][0]["spatial"]["dilation_h"] == 1
    assert plan["ops"][0]["spatial"]["dilation_w"] == 1
    assert plan["ops"][0]["spatial"]["kernel_h"] == 2
    assert plan["ops"][0]["spatial"]["kernel_w"] == 2
