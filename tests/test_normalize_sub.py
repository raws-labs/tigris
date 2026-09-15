"""A constant subtrahend becomes an added negation."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.validation import validate_operator_support
from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize


def _normalized(tmp_path, nodes, inputs, outputs, init=()):
    model = helper.make_model(
        helper.make_graph(nodes, "sub", inputs, outputs, list(init)),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / "sub.onnx"
    onnx.save(model, str(path))
    return normalize(load_model(str(path)))


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _constant(values, name="constant"):
    return numpy_helper.from_array(np.array([values], np.float32), name)


def test_constant_subtrahend_becomes_an_added_negation(tmp_path):
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["x", "constant"], ["y"], name="sub1")],
        [_vi("x", [1, 4])], [_vi("y", [1, 4])],
        [_constant([0.5, -1.0, 2.0, 0.25])])

    assert [op.op_type for op in ag.ops] == ["Add"]
    assert np.allclose(
        ag.weight_data["constant"].reshape(-1), [-0.5, 1.0, -2.0, -0.25])


def test_two_activations_stay_a_subtraction(tmp_path):
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["a", "b"], ["y"], name="sub1")],
        [_vi("a", [1, 4]), _vi("b", [1, 4])], [_vi("y", [1, 4])])

    assert [op.op_type for op in ag.ops] == ["Sub"]
    assert validate_operator_support(ag).supported


def test_constant_minuend_is_refused(tmp_path):
    """c - x has no commutative equivalent the plan can express."""
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["constant", "x"], ["y"], name="sub1")],
        [_vi("x", [1, 4])], [_vi("y", [1, 4])],
        [_constant([0.5, -1.0, 2.0, 0.25])])

    assert [op.op_type for op in ag.ops] == ["Sub"]
    support = validate_operator_support(ag)
    assert not support.supported
    assert "does not commute" in support.describe()
