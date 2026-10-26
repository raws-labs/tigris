"""Convolution and ConvTranspose cases."""

from __future__ import annotations

import copy

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars, _vi


def _depthwise_multiplier_case(*, quantized: bool) -> ContractCase:
    """A depthwise convolution writing two output channels per input channel,
    banded by height. ONNX Runtime fuses the quantized form into an integer
    grouped convolution whose result varies with the host CPU, so the
    reference runs unfused."""
    channels, multiplier, side = 4, 2, 16
    rng = np.random.default_rng(22)
    weight = (rng.normal(size=(channels * multiplier, 1, 3, 3)) * 0.4).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    conv = dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=channels)
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), conv=(0.02, 3))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "conv_s", "conv_z", "output"),
        ]
    else:
        nodes = [helper.make_node("Conv", ["input", "weight"], ["output"], **conv)]
    label = f"{'int8' if quantized else 'float'}_depthwise_multiplier"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, channels * multiplier, side, side])],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("DepthwiseConv",), mem_budget="2K", expect_tiled=True,
                        reference_unoptimized=quantized)


def _inferred_kernel_chain_case(*, quantized: bool) -> ContractCase:
    """A 5x5 stride-2 depthwise that omits kernel_shape, behind a 1x1 conv.

    ONNX lets a convolution take its kernel size from the weight. Read as
    1x1, the pair looks like a chain that streams at one row per tile; with
    the real 5x5 window one row needs five input rows, which does not fit
    the budget, so the stages have to run tiled on their own.
    """
    cin, cmid, side = 24, 96, 32
    rng = np.random.default_rng(37)
    w1 = (rng.normal(size=(cmid, cin, 1, 1)) * 0.2).astype(np.float32)
    w2 = (rng.normal(size=(cmid, 1, 5, 5)) * 0.2).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, cin, side, side)).astype(np.float32)
    initializers = [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2")]
    dw = dict(group=cmid, strides=[2, 2], pads=[1, 1, 2, 2])
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w1q=(0.01, 0), mid=(0.02, 0),
                                 w2q=(0.01, 0), out=(0.03, 0))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("w1", "w1q_s", "w1q_z", "w1dq"),
            helper.make_node("Conv", ["x", "w1dq"], ["a"]),
            helper.make_node("Relu", ["a"], ["ar"]),
            *_qdq("ar", "mid_s", "mid_z", "b"),
            *_qdq("w2", "w2q_s", "w2q_z", "w2dq"),
            helper.make_node("Conv", ["b", "w2dq"], ["c"], **dw),
            *_qdq("c", "out_s", "out_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("Conv", ["input", "w1"], ["a"]),
            helper.make_node("Relu", ["a"], ["b"]),
            helper.make_node("Conv", ["b", "w2"], ["output"], **dw),
        ]
    label = f"{'int8' if quantized else 'float'}_inferred_kernel_chain"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, cin, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, cmid, side // 2, side // 2])],
        initializers,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Conv", "DepthwiseConv"), mem_budget="16K" if quantized else "64K",
                        expect_tiled=True)


def _dilated_conv_case() -> ContractCase:
    weights = numpy_helper.from_array(
        np.array([[[[1.0, -0.5], [0.25, 2.0]]]], dtype=np.float32), "weights")
    model = _model(
        "dilated_conv",
        [helper.make_node("Conv", ["input", "weights"], ["output"],
                          dilations=[2, 2], kernel_shape=[2, 2])],
        [_vi("input", [1, 1, 5, 5])], [_vi("output", [1, 1, 3, 3])], [weights],
    )
    return ContractCase("float_dilated_conv", model, model,
                        {"input": np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)}, ("Conv",))


def _depthwise_conv_case() -> ContractCase:
    """Float depthwise Conv exercises its distinct weight layout and route."""
    weights = numpy_helper.from_array(np.array([
        [[[1.0, 0.0, -1.0], [0.5, 0.25, -0.5], [0.0, 1.0, 0.0]]],
        [[[-0.5, 0.25, 0.5], [1.0, -1.0, 0.0], [0.25, 0.0, -0.25]]],
    ], dtype=np.float32), "weights")
    model = _model(
        "depthwise_conv",
        [helper.make_node("Conv", ["input", "weights"], ["output"], pads=[1, 1, 1, 1], group=2)],
        [_vi("input", [1, 2, 8, 8])], [_vi("output", [1, 2, 8, 8])], [weights],
    )
    return ContractCase(
        "float_depthwise_conv", model, model,
        {"input": np.linspace(-2.0, 2.0, 128, dtype=np.float32).reshape(1, 2, 8, 8)},
        ("DepthwiseConv",),
    )


def _conv1d_case() -> ContractCase:
    """ONNX Conv1D relabeling and fused activation execution."""
    length = 256
    weights = numpy_helper.from_array(np.array([
        [[0.5, 1.0, -0.5], [0.25, 0.0, 0.75]],
        [[-1.0, 0.5, 1.0], [0.5, -0.25, 0.25]],
        [[0.25, 0.25, 0.25], [-0.5, 1.0, -0.5]],
    ], dtype=np.float32), "weights")
    bias = numpy_helper.from_array(np.array([0.1, -0.2, 0.3], dtype=np.float32), "bias")
    model = _model(
        "conv1d",
        [
            helper.make_node("Conv", ["input", "weights", "bias"], ["convolved"],
                             kernel_shape=[3], pads=[1, 1]),
            helper.make_node("Relu", ["convolved"], ["output"]),
        ],
        [_vi("input", [1, 2, length])], [_vi("output", [1, 3, length])], [weights, bias],
    )
    return ContractCase(
        "float_conv1d", model, model,
        {"input": np.linspace(-1.5, 1.5, 2 * length, dtype=np.float32).reshape(1, 2, length)},
        ("Conv1D",), mem_budget="1K", expect_tiled=True,
    )


def _channel_bias_add_case(*, producer_has_bias: bool) -> ContractCase:
    """A per-channel constant Add must reach the same values as its own graph.

    The reference model keeps the standalone Add, so ONNX Runtime evaluates the
    broadcast exactly as an exporter wrote it; TiGrIS folds it into the Conv
    bias and must land on the same numbers either way.
    """
    initializers = [
        numpy_helper.from_array(
            np.linspace(-0.4, 0.4, 3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3), "weight"),
        numpy_helper.from_array(np.array([[[[0.5]], [[-1.25]], [[2.0]]]], dtype=np.float32), "channel"),
    ]
    conv_inputs = ["input", "weight"]
    if producer_has_bias:
        initializers.append(
            numpy_helper.from_array(np.array([0.125, -0.25, 0.75], dtype=np.float32), "bias"))
        conv_inputs.append("bias")
    model = _model(
        "channel_bias_add",
        [
            helper.make_node("Conv", conv_inputs, ["product"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Add", ["product", "channel"], ["output"]),
        ],
        [_vi("input", [1, 2, 3, 3])], [_vi("output", [1, 3, 3, 3])], initializers,
    )
    suffix = "onto_bias" if producer_has_bias else "as_bias"
    return ContractCase(
        f"float_channel_bias_add_{suffix}", model, model,
        {"input": np.linspace(-1.0, 1.0, 18, dtype=np.float32).reshape(1, 2, 3, 3)},
        ("Conv",),
    )


def _convtranspose_case(name: str, seed: int, *, kernel: int, pads: list[int],
                        output_padding: list[int] | None = None) -> ContractCase:
    """A standalone float ConvTranspose upsampling 2 channels at 4x4 to 3 at 8x8, stride 2.

    The ONNX weight is [C_in, C_out, kH, kW]; the compiler transposes it to the
    runtime's OHWI layout. The output side is stride * (in - 1) + output_padding +
    kernel - pad_begin - pad_end. The gate's variants:
    - kernel 2, pad 0: one tap per output pixel.
    - kernel 4, pad 1: taps overlap and sum into each output pixel (the U-Net
      upsampler), which the multi-tap accumulation of the gather kernel has to get right.
    - kernel 3, pad 1, output_padding [1, 1]: 7x7 without the extra trailing row and
      column. The compiler takes the output shape from ONNX shape inference and the
      runtime bounds itself by the allocated output, so both must agree with it.
    - kernel 3, pads [0, 1, 1, 0] (ONNX order h_begin, w_begin, h_end, w_end): the
      begin and end pads differ on each axis, so a swapped pad shows as a mismatch.
    The compile and ORT reference models are one model.
    """
    rng = np.random.default_rng(seed)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 3, kernel, kernel)).astype(np.float32), "weights")
    bias = numpy_helper.from_array(rng.normal(0.0, 0.5, size=(3,)).astype(np.float32), "bias")
    model = _model(
        name,
        [helper.make_node("ConvTranspose", ["input", "weights", "bias"], ["output"],
                          kernel_shape=[kernel, kernel], strides=[2, 2], pads=pads,
                          output_padding=output_padding, group=1)],
        [_vi("input", [1, 2, 4, 4])], [_vi("output", [1, 3, 8, 8])], [weights, bias],
    )
    return ContractCase(
        f"float_{name}", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4)).astype(np.float32)},
        ("ConvTranspose",),
    )


def _conv_then_convtranspose_case() -> ContractCase:
    """A strided Conv downsampler feeding a ConvTranspose upsampler.

    Conv (stride 2, kernel 2, pad 0) halves an 8x8 input to 4x4, then
    ConvTranspose (stride 2, kernel 2, pad 0) restores 8x8. Both stages keep
    group 1 so the Conv is not relabeled DepthwiseConv.
    """
    rng = np.random.default_rng(23)
    conv_weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 1, 2, 2)).astype(np.float32), "conv_weights")
    convt_weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 1, 2, 2)).astype(np.float32), "convt_weights")
    halve = dict(kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], group=1)
    model = _model(
        "conv_then_convtranspose",
        [
            helper.make_node("Conv", ["input", "conv_weights"], ["mid"], **halve),
            helper.make_node("ConvTranspose", ["mid", "convt_weights"], ["output"], **halve),
        ],
        [_vi("input", [1, 1, 8, 8])], [_vi("output", [1, 1, 8, 8])], [conv_weights, convt_weights],
    )
    return ContractCase(
        "float_conv_then_convtranspose", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 1, 8, 8)).astype(np.float32)},
        ("Conv", "ConvTranspose"),
    )
