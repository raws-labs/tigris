"""Regression tests for ONNX spatial attributes omitted at their default."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.pipeline import plan_model

from conftest import save_onnx


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
    path = save_onnx(tmp_path, "default_spatial_attrs",
                     [helper.make_node("Conv", ["input", "weights"], ["output"])], [model_input],
                     [model_output], [weights], ir_version=8)

    plan = read_binary_plan(emit_binary_bytes(plan_model(path, (4096,))))

    assert plan["ops"][0]["spatial"]["stride_h"] == 1
    assert plan["ops"][0]["spatial"]["stride_w"] == 1
    assert plan["ops"][0]["spatial"]["dilation_h"] == 1
    assert plan["ops"][0]["spatial"]["dilation_w"] == 1
    assert plan["ops"][0]["spatial"]["kernel_h"] == 2
    assert plan["ops"][0]["spatial"]["kernel_w"] == 2
