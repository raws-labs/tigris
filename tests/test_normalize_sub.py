"""A constant subtrahend becomes an added negation."""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from tigris.analysis.validation import validate_operator_support

from conftest import normalized, vi


def _normalized(tmp_path, nodes, inputs, outputs, init=()):
    return normalized(tmp_path, "sub", nodes, inputs, outputs, init)


def _constant(values, name="constant"):
    return numpy_helper.from_array(np.array([values], np.float32), name)


def test_constant_subtrahend_becomes_an_added_negation(tmp_path):
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["x", "constant"], ["y"], name="sub1")],
        [vi("x", [1, 4])], [vi("y", [1, 4])],
        [_constant([0.5, -1.0, 2.0, 0.25])])

    assert [op.op_type for op in ag.ops] == ["Add"]
    assert np.allclose(
        ag.weight_data["constant"].reshape(-1), [-0.5, 1.0, -2.0, -0.25])


def test_two_activations_stay_a_subtraction(tmp_path):
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["a", "b"], ["y"], name="sub1")],
        [vi("a", [1, 4]), vi("b", [1, 4])], [vi("y", [1, 4])])

    assert [op.op_type for op in ag.ops] == ["Sub"]
    assert validate_operator_support(ag).supported


def test_constant_minuend_stays_a_subtraction(tmp_path):
    """c - x has no commutative equivalent; the plan states the constant's side."""
    ag = _normalized(
        tmp_path,
        [helper.make_node("Sub", ["constant", "x"], ["y"], name="sub1")],
        [vi("x", [1, 4])], [vi("y", [1, 4])],
        [_constant([0.5, -1.0, 2.0, 0.25])])

    assert [op.op_type for op in ag.ops] == ["Sub"]
    assert ag.ops[0].inputs[0] == "constant"
    assert validate_operator_support(ag).supported


def test_neg_uses_its_unary_kernel(tmp_path):
    ag = _normalized(
        tmp_path,
        [helper.make_node("Neg", ["x"], ["y"], name="neg1")],
        [vi("x", [1, 4])], [vi("y", [1, 4])])

    assert [op.op_type for op in ag.ops] == ["Neg"]
    assert ag.ops[0].inputs == ["x"]
    assert not ag.weight_data
    assert validate_operator_support(ag).supported


@pytest.mark.parametrize("expose_difference", [False, True])
def test_squared_difference_preserves_shared_intermediate(tmp_path, expose_difference):
    nodes = [helper.make_node("Sub", ["a", "b"], ["difference"]),
             helper.make_node("Mul", ["difference", "difference"], ["y"])]
    side_output = "difference" if expose_difference else "side"
    if not expose_difference:
        nodes.append(helper.make_node("Relu", ["difference"], ["side"]))
    ag = _normalized(tmp_path, nodes, [vi("a", [1, 4]), vi("b", [1, 4])],
                     [vi("y", [1, 4]), vi(side_output, [1, 4])])
    assert ag.ops[0].op_type == "Sub"
    assert "SquaredDifference" not in [op.op_type for op in ag.ops]
    assert side_output in ag.model_outputs
    assert "difference" in ag.tensors


def test_constant_squared_difference_folds_with_its_constant(tmp_path):
    ag = _normalized(tmp_path,
                     [helper.make_node("Sub", ["a", "constant"], ["difference"]),
                      helper.make_node("Mul", ["difference", "difference"], ["y"])],
                     [vi("a", [1, 4])], [vi("y", [1, 4])],
                     [_constant([0.5, -1, 2, 0.25])])
    assert [op.op_type for op in ag.ops] == ["SquaredDifference"]
    assert validate_operator_support(ag).supported


def test_squared_difference_does_not_cross_quantization(tmp_path):
    ag = _normalized(tmp_path,
                     [helper.make_node("Sub", ["a", "b"], ["difference"]),
                      helper.make_node("QuantizeLinear", ["difference", "s", "z"], ["q"]),
                      helper.make_node("DequantizeLinear", ["q", "s", "z"], ["dq"]),
                      helper.make_node("Mul", ["dq", "dq"], ["product"]),
                      helper.make_node("QuantizeLinear", ["product", "s", "z"], ["yq"]),
                      helper.make_node("DequantizeLinear", ["yq", "s", "z"], ["y"])],
                     [vi("a", [1, 4]), vi("b", [1, 4])], [vi("y", [1, 4])],
                     [numpy_helper.from_array(np.array(0.125, np.float32), "s"),
                      numpy_helper.from_array(np.array(0, np.int8), "z")])
    assert [op.op_type for op in ag.ops] == ["Sub", "Mul"]


def test_floor_mod_requires_sign_correction(tmp_path):
    ag = _normalized(tmp_path,
                     [helper.make_node("Mod", ["a", "b"], ["r"], fmod=1),
                      helper.make_node("Add", ["r", "b"], ["sum"]),
                      helper.make_node("Less", ["r", "zero"], ["negative"]),
                      helper.make_node("Where", ["negative", "sum", "r"], ["y"])],
                     [vi("a", [1, 4]), vi("b", [1, 4])], [vi("y", [1, 4])],
                     [numpy_helper.from_array(np.array(0, np.float32), "zero")])
    assert "FloorMod" not in [op.op_type for op in ag.ops]
    assert not validate_operator_support(ag).supported
