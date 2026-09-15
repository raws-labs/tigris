"""Gemm's alpha and beta, which the plan has no field for."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.validation import validate_operator_support
from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize


def _normalized(tmp_path, with_bias=True, **attrs):
    weight = numpy_helper.from_array(np.ones((4, 3), np.float32), "weight")
    inputs = ["x", "weight"]
    init = [weight]
    if with_bias:
        inputs.append("bias")
        init.append(
            numpy_helper.from_array(
                np.array([1.0, 2.0, 3.0, 4.0], np.float32), "bias"))
    node = helper.make_node("Gemm", inputs, ["y"], name="gemm1", **attrs)
    graph = helper.make_graph(
        [node], "gemm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        init,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "gemm.onnx"
    onnx.save(model, str(path))
    return normalize(load_model(str(path)))


def test_alpha_scales_the_weight(tmp_path):
    ag = _normalized(tmp_path, transB=1, alpha=2.0)
    gemm = next(op for op in ag.ops if op.op_type == "Gemm")
    assert np.allclose(ag.weight_data[gemm.inputs[1]], 2.0)
    assert validate_operator_support(ag).supported


def test_beta_scales_the_bias(tmp_path):
    ag = _normalized(tmp_path, transB=1, beta=3.0)
    gemm = next(op for op in ag.ops if op.op_type == "Gemm")
    assert np.allclose(
        ag.weight_data[gemm.inputs[2]], [3.0, 6.0, 9.0, 12.0])
    assert validate_operator_support(ag).supported


def test_beta_without_a_bias_has_nothing_to_scale(tmp_path):
    ag = _normalized(tmp_path, with_bias=False, transB=1, beta=3.0)
    assert validate_operator_support(ag).supported


def test_transposed_activation_is_refused(tmp_path):
    """transA transposes an activation, which no constant can absorb."""
    ag = _normalized(tmp_path, transB=1, transA=1)
    support = validate_operator_support(ag)
    assert not support.supported
    assert "transA" in support.describe()


@pytest.mark.parametrize("attr", ["alpha", "beta"])
def test_an_unfoldable_scalar_is_refused(tmp_path, attr):
    """A dynamic weight leaves alpha with nothing constant to fold into."""
    node = helper.make_node(
        "Gemm", ["x", "w"], ["y"], name="gemm1", transB=1, **{attr: 2.0})
    graph = helper.make_graph(
        [node], "gemm",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "gemm_dynamic.onnx"
    onnx.save(model, str(path))
    ag = normalize(load_model(str(path)))

    if attr == "beta":
        # beta scales C, and this graph has none.
        assert validate_operator_support(ag).supported
    else:
        support = validate_operator_support(ag)
        assert not support.supported
        assert "alpha" in support.describe()
