"""A Clip that states an activation the runtime already has."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.validation import validate_operator_support
from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize


def _normalized(tmp_path, inputs, init):
    node = helper.make_node("Clip", inputs, ["y"], name="clip1")
    graph = helper.make_graph(
        [node], "clip",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        list(init),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "clip.onnx"
    onnx.save(model, str(path))
    return normalize(load_model(str(path)))


def _scalar(value, name):
    return numpy_helper.from_array(np.float32(value), name)


@pytest.mark.parametrize("ceiling", [None, np.inf, np.finfo(np.float32).max])
def test_zero_floor_without_a_ceiling_is_relu(tmp_path, ceiling):
    """Absent, infinite, and float-max ceilings all mean unbounded above."""
    init = [_scalar(0.0, "lo")]
    inputs = ["x", "lo"]
    if ceiling is not None:
        init.append(_scalar(ceiling, "hi"))
        inputs.append("hi")

    ag = _normalized(tmp_path, inputs, init)
    assert [op.op_type for op in ag.ops] == ["Relu"]
    assert validate_operator_support(ag).supported


def test_zero_to_six_is_relu6(tmp_path):
    ag = _normalized(
        tmp_path, ["x", "lo", "hi"],
        [_scalar(0.0, "lo"), _scalar(6.0, "hi")])
    assert [op.op_type for op in ag.ops] == ["Relu6"]


def test_other_bounds_have_no_kernel(tmp_path):
    """Clip(0, 3) names no activation the runtime carries."""
    ag = _normalized(
        tmp_path, ["x", "lo", "hi"],
        [_scalar(0.0, "lo"), _scalar(3.0, "hi")])
    assert [op.op_type for op in ag.ops] == ["Clip"]
    assert not validate_operator_support(ag).supported


def test_nonzero_floor_is_left_alone(tmp_path):
    """A floor other than zero is not a Relu whatever the ceiling is."""
    ag = _normalized(
        tmp_path, ["x", "lo"], [_scalar(-1.0, "lo")])
    assert [op.op_type for op in ag.ops] == ["Clip"]
