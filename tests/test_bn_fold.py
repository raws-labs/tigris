"""Tests for BN folding."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize as bn_fold

from conftest import save_onnx, vi


# Fixtures


@pytest.fixture
def conv_bn_path(tmp_path):
    """Conv -> BatchNormalization on [1, 8, 16, 16]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 14, 14])

    np.random.seed(42)
    W = np.random.randn(8, 3, 3, 3).astype(np.float32)
    B = np.random.randn(8).astype(np.float32)
    gamma = np.random.randn(8).astype(np.float32)
    beta = np.random.randn(8).astype(np.float32)
    mean = np.random.randn(8).astype(np.float32)
    var = np.abs(np.random.randn(8)).astype(np.float32) + 0.01

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3], W.flatten().tolist())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [8], B.tolist())
    gamma_init = helper.make_tensor("gamma", TensorProto.FLOAT, [8], gamma.tolist())
    beta_init = helper.make_tensor("beta", TensorProto.FLOAT, [8], beta.tolist())
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, [8], mean.tolist())
    var_init = helper.make_tensor("var", TensorProto.FLOAT, [8], var.tolist())

    conv = helper.make_node("Conv", ["input", "W", "B"], ["conv_out"], name="conv",
                            kernel_shape=[3, 3])
    bn = helper.make_node("BatchNormalization", ["conv_out", "gamma", "beta", "mean", "var"],
                          ["output"], name="bn", epsilon=1e-5)

    return save_onnx(
        tmp_path, "conv_bn", [conv, bn], [X], [Y],
        [w_init, b_init, gamma_init, beta_init, mean_init, var_init],
        opset=17, ir_version=8,
    )


@pytest.fixture
def conv_bn_no_bias_path(tmp_path):
    """Conv (no bias) -> BatchNormalization."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    np.random.seed(99)
    W = np.random.randn(4, 3, 3, 3).astype(np.float32)
    gamma = np.ones(4, dtype=np.float32)
    beta = np.zeros(4, dtype=np.float32)
    mean = np.random.randn(4).astype(np.float32)
    var = np.abs(np.random.randn(4)).astype(np.float32) + 0.1

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 3, 3, 3], W.flatten().tolist())
    gamma_init = helper.make_tensor("gamma", TensorProto.FLOAT, [4], gamma.tolist())
    beta_init = helper.make_tensor("beta", TensorProto.FLOAT, [4], beta.tolist())
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, [4], mean.tolist())
    var_init = helper.make_tensor("var", TensorProto.FLOAT, [4], var.tolist())

    conv = helper.make_node("Conv", ["input", "W"], ["conv_out"], name="conv",
                            kernel_shape=[3, 3])
    bn = helper.make_node("BatchNormalization", ["conv_out", "gamma", "beta", "mean", "var"],
                          ["output"], name="bn")

    return save_onnx(
        tmp_path, "conv_bn_nobias", [conv, bn], [X], [Y],
        [w_init, gamma_init, beta_init, mean_init, var_init],
        opset=17, ir_version=8,
    )


def _conv3x3(x, w, b):
    """Valid 3x3 convolution of NCHW x (batch 1) with OIHW w plus bias b."""
    windows = np.lib.stride_tricks.sliding_window_view(x[0], (3, 3), axis=(1, 2))
    return (np.einsum("chwij,ocij->ohw", windows, w) + b[:, None, None])[None]


# BN fold correctness


def test_bn_folds_into_the_conv(conv_bn_path):
    """BN and its parameters disappear; the plain Conv takes over the BN output name."""
    ag = load_model(conv_bn_path)
    assert any(op.op_type == "BatchNormalization" for op in ag.ops)

    ag = bn_fold(ag)
    assert [op.op_type for op in ag.ops] == ["Conv"]
    assert ag.ops[0].outputs[0] == "output"
    for name in ["gamma", "beta", "mean", "var"]:
        assert name not in ag.weight_data
    for i, op in enumerate(ag.ops):
        assert op.step == i


def test_bn_fold_numerical_correctness(conv_bn_path):
    """Folded conv should produce same output as conv+BN on random input."""
    ag = load_model(conv_bn_path)

    # Get original params
    W_orig = ag.weight_data["W"].copy()
    B_orig = ag.weight_data["B"].copy()
    gamma = ag.weight_data["gamma"].copy()
    beta = ag.weight_data["beta"].copy()
    mean = ag.weight_data["mean"].copy()
    var = ag.weight_data["var"].copy()
    eps = 1e-5

    # Compute conv + BN reference on random input
    np.random.seed(123)
    x = np.random.randn(1, 3, 16, 16).astype(np.float32)

    conv_out = _conv3x3(x, W_orig, B_orig)
    inv_std = 1.0 / np.sqrt(var + eps)
    bn_out = (conv_out - mean[:, None, None]) * (inv_std * gamma)[:, None, None] \
        + beta[:, None, None]

    # Now fold and compute
    ag2 = load_model(conv_bn_path)
    ag2 = bn_fold(ag2)
    folded_out = _conv3x3(x, ag2.weight_data["W"], ag2.weight_data["B"])

    assert np.allclose(bn_out, folded_out, atol=1e-5), \
        f"Max diff: {np.max(np.abs(bn_out - folded_out))}"


def test_bn_fold_no_bias(conv_bn_no_bias_path):
    """Conv without bias + BN fold should create a bias."""
    ag = load_model(conv_bn_no_bias_path)
    ag = bn_fold(ag)

    conv_op = ag.ops[0]
    assert len(conv_op.inputs) >= 3, "Folded conv should have a bias input"

    bias_name = conv_op.inputs[2]
    assert bias_name in ag.weight_data, "Bias should be in weight_data"
    assert ag.weight_data[bias_name].shape == (4,)


def _quantized_conv_bn(tmp_path, weight, gamma):
    """QDQ Conv (int8 weights, scale 0.01) followed by a float BatchNormalization."""
    channels = weight.shape[0]
    scalars = {"x_s": 0.1, "w_s": 0.01, "y_s": 0.1}
    initializers = [onnx.numpy_helper.from_array(weight.astype(np.float32), "w")]
    for name, value in scalars.items():
        initializers.append(onnx.numpy_helper.from_array(np.array(value, np.float32), name))
    initializers += [
        onnx.numpy_helper.from_array(np.array(0, np.int8), "zero"),
        onnx.numpy_helper.from_array(gamma.astype(np.float32), "gamma"),
        onnx.numpy_helper.from_array(np.full(channels, 0.5, np.float32), "beta"),
        onnx.numpy_helper.from_array(np.zeros(channels, np.float32), "mean"),
        onnx.numpy_helper.from_array(np.ones(channels, np.float32), "var"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "x_s", "zero"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "x_s", "zero"], ["x"]),
        helper.make_node("QuantizeLinear", ["w", "w_s", "zero"], ["wq"]),
        helper.make_node("DequantizeLinear", ["wq", "w_s", "zero"], ["wdq"]),
        helper.make_node("Conv", ["x", "wdq"], ["raw"], kernel_shape=[1, 1]),
        helper.make_node("BatchNormalization", ["raw", "gamma", "beta", "mean", "var"], ["normed"]),
        helper.make_node("QuantizeLinear", ["normed", "y_s", "zero"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "y_s", "zero"], ["output"]),
    ]
    path = save_onnx(
        tmp_path, "qconv_bn", nodes, [vi("input", [1, weight.shape[1], 2, 2])],
        [vi("output", [1, channels, 2, 2])], initializers,
    )
    return bn_fold(load_model(str(path)))


def test_quantized_bn_moves_into_weight_scale_and_bias(tmp_path):
    """A negative gamma negates the channel's int8 weights; |a| scales its weight scale."""
    weight = np.array([[[[0.5]], [[-0.25]]], [[[0.3]], [[0.1]]]])
    ag = _quantized_conv_bn(tmp_path, weight, np.array([2.0, -0.5]))
    assert [op.op_type for op in ag.ops] == ["Conv"]
    conv = ag.ops[0]
    w_q = ag.weight_data[conv.inputs[1]]
    a = np.array([2.0, -0.5]) / np.sqrt(1.0 + 1e-5)
    assert w_q[:, :, 0, 0].tolist() == [[50, -25], [-30, -10]]
    np.testing.assert_allclose(ag.tensors[conv.inputs[1]].quant.scale, 0.01 * np.abs(a), rtol=1e-6)
    bias = ag.weight_data[conv.inputs[2]]
    np.testing.assert_array_equal(bias, np.round(0.5 / (0.1 * 0.01 * np.abs(a))).astype(np.int32))


def test_quantized_bn_that_cannot_fold_exactly_is_rejected(tmp_path):
    """-128 has no int8 negation, so a negative gamma on it leaves the BN, which validation refuses."""
    from tigris.analysis.validation import validate_operator_support

    weight = np.array([[[[-1.28]], [[0.1]]], [[[0.3]], [[0.1]]]])
    ag = _quantized_conv_bn(tmp_path, weight, np.array([-1.0, 1.0]))
    assert "BatchNormalization" in [op.op_type for op in ag.ops]
    issues = validate_operator_support(ag).issues
    assert any(issue.op_type == "BatchNormalization" for issue in issues)
