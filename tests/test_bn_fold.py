"""Tests for BN folding, DepthwiseConv relabeling, Clip->Relu6, and shape op folding."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize as bn_fold


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

    graph = helper.make_graph(
        [conv, bn], "conv_bn", [X], [Y],
        initializer=[w_init, b_init, gamma_init, beta_init, mean_init, var_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "conv_bn.onnx"
    onnx.save(model, str(path))
    return path


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

    graph = helper.make_graph(
        [conv, bn], "conv_bn_nobias", [X], [Y],
        initializer=[w_init, gamma_init, beta_init, mean_init, var_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "conv_bn_nobias.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def depthwise_conv_path(tmp_path):
    """Depthwise Conv (group == C_in == 8)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 14, 14])

    W = np.random.randn(8, 1, 3, 3).astype(np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, [8, 1, 3, 3], W.flatten().tolist())

    conv = helper.make_node("Conv", ["input", "W"], ["output"], name="dw_conv",
                            kernel_shape=[3, 3], group=8)

    graph = helper.make_graph([conv], "depthwise", [X], [Y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "depthwise.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def clip_relu6_path(tmp_path):
    """Clip(min=0, max=6) -> should become Relu6."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

    # Opset >= 11: min/max are inputs, not attributes
    min_val = helper.make_tensor("min_val", TensorProto.FLOAT, [], [0.0])
    max_val = helper.make_tensor("max_val", TensorProto.FLOAT, [], [6.0])

    clip = helper.make_node("Clip", ["input", "min_val", "max_val"], ["output"], name="clip0")

    graph = helper.make_graph([clip], "clip_relu6", [X], [Y],
                              initializer=[min_val, max_val])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "clip_relu6.onnx"
    onnx.save(model, str(path))
    return path


# BN fold correctness


def test_bn_removed(conv_bn_path):
    """After folding, the BN op should be removed."""
    ag = load_model(conv_bn_path)
    assert any(op.op_type == "BatchNormalization" for op in ag.ops)

    ag = bn_fold(ag)
    assert not any(op.op_type == "BatchNormalization" for op in ag.ops)
    assert len(ag.ops) == 1  # only Conv remains


def test_bn_fold_preserves_output_name(conv_bn_path):
    """The conv's output should be rewired to the BN's original output name."""
    ag = load_model(conv_bn_path)
    ag = bn_fold(ag)

    conv_op = ag.ops[0]
    assert conv_op.outputs[0] == "output"


def test_bn_fold_updates_weights(conv_bn_path):
    """Folded weights should differ from original."""
    ag_before = load_model(conv_bn_path)
    W_before = ag_before.weight_data["W"].copy()

    ag = bn_fold(ag_before)
    W_after = ag.weight_data["W"]

    assert not np.allclose(W_before, W_after), "Weights should have been modified by BN fold"


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

    # Simple conv (no padding)
    conv_out = np.zeros((1, 8, 14, 14), dtype=np.float32)
    for oc in range(8):
        for oh in range(14):
            for ow in range(14):
                patch = x[0, :, oh:oh+3, ow:ow+3]
                conv_out[0, oc, oh, ow] = np.sum(patch * W_orig[oc]) + B_orig[oc]

    # BN
    inv_std = 1.0 / np.sqrt(var + eps)
    bn_out = np.zeros_like(conv_out)
    for oc in range(8):
        bn_out[0, oc] = (conv_out[0, oc] - mean[oc]) * inv_std[oc] * gamma[oc] + beta[oc]

    # Now fold and compute
    ag2 = load_model(conv_bn_path)
    ag2 = bn_fold(ag2)
    W_folded = ag2.weight_data["W"]
    B_folded = ag2.weight_data["B"]

    folded_out = np.zeros((1, 8, 14, 14), dtype=np.float32)
    for oc in range(8):
        for oh in range(14):
            for ow in range(14):
                patch = x[0, :, oh:oh+3, ow:ow+3]
                folded_out[0, oc, oh, ow] = np.sum(patch * W_folded[oc]) + B_folded[oc]

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


def test_bn_params_removed(conv_bn_path):
    """BN parameters (gamma, beta, mean, var) should be removed from weight_data."""
    ag = load_model(conv_bn_path)
    ag = bn_fold(ag)

    for name in ["gamma", "beta", "mean", "var"]:
        assert name not in ag.weight_data


def test_step_indices_reassigned(conv_bn_path):
    """After BN removal, step indices should be contiguous."""
    ag = load_model(conv_bn_path)
    ag = bn_fold(ag)

    for i, op in enumerate(ag.ops):
        assert op.step == i


# DepthwiseConv relabeling


def test_depthwise_relabel(depthwise_conv_path):
    ag = load_model(depthwise_conv_path)
    assert ag.ops[0].op_type == "Conv"

    ag = bn_fold(ag)
    assert ag.ops[0].op_type == "DepthwiseConv"


def test_normal_conv_not_relabeled(conv_bn_path):
    """A normal conv (group=1) should stay as Conv."""
    ag = load_model(conv_bn_path)
    ag = bn_fold(ag)
    assert ag.ops[0].op_type == "Conv"


# Clip -> Relu6


def test_clip_to_relu6(clip_relu6_path):
    ag = load_model(clip_relu6_path)
    assert ag.ops[0].op_type == "Clip"

    ag = bn_fold(ag)
    assert ag.ops[0].op_type == "Relu6"


def test_clip_relu6_single_input(clip_relu6_path):
    """After conversion, Relu6 should have only the data input."""
    ag = load_model(clip_relu6_path)
    ag = bn_fold(ag)
    assert len(ag.ops[0].inputs) == 1


# Shape op folding

MOBILENETV2_PATH = Path(__file__).resolve().parent.parent / "mobilenetv2-12.onnx"
_has_mobilenetv2 = MOBILENETV2_PATH.exists()


@pytest.mark.skipif(not _has_mobilenetv2, reason="mobilenetv2-12.onnx not present")
def test_shape_ops_folded():
    """After bn_fold, shape-computation ops should be removed from MobileNetV2."""
    ag = load_model(MOBILENETV2_PATH)
    ag = bn_fold(ag)

    remaining_types = {op.op_type for op in ag.ops}
    for forbidden in ("Shape", "Gather", "Constant", "Unsqueeze"):
        assert forbidden not in remaining_types, f"{forbidden} ops should be folded away"

    assert len(ag.ops) == 65, f"Expected 65 ops after folding + activation fusion, got {len(ag.ops)}"

    # Step indices should be contiguous
    for i, op in enumerate(ag.ops):
        assert op.step == i, f"Step mismatch at index {i}: {op.step}"


@pytest.mark.skipif(not _has_mobilenetv2, reason="mobilenetv2-12.onnx not present")
def test_reshape_has_one_nonconst_input():
    """After folding, Reshape ops should have exactly 1 non-constant input."""
    ag = load_model(MOBILENETV2_PATH)
    ag = bn_fold(ag)

    reshape_ops = [op for op in ag.ops if op.op_type == "Reshape"]
    assert len(reshape_ops) >= 1, "Should have at least one Reshape op"

    for op in reshape_ops:
        non_const_inputs = [
            inp for inp in op.inputs
            if inp in ag.tensors and not ag.tensors[inp].is_constant
        ]
        assert len(non_const_inputs) == 1, (
            f"Reshape '{op.name}' has {len(non_const_inputs)} non-constant inputs, expected 1"
        )


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
    model = helper.make_model(helper.make_graph(
        nodes, "qconv_bn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, weight.shape[1], 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 2, 2])],
        initializers), opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "qconv_bn.onnx"
    onnx.save(model, path)
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
