"""Convolution and ConvTranspose cases."""

from __future__ import annotations

import copy

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars


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
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 5, 5]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 3, 3]
    )
    weights = numpy_helper.from_array(
        np.array([[[[1.0, -0.5], [0.25, 2.0]]]], dtype=np.float32),
        "weights",
    )
    model = _model(
        "dilated_conv",
        [
            helper.make_node(
                "Conv",
                ["input", "weights"],
                ["output"],
                dilations=[2, 2],
                kernel_shape=[2, 2],
            )
        ],
        [model_input],
        [model_output],
        [weights],
    )
    return ContractCase(
        "float_dilated_conv",
        model,
        model,
        {
            "input": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        },
        ("Conv",),
    )


def _depthwise_conv_case() -> ContractCase:
    """Float depthwise Conv exercises its distinct weight layout and route."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 8, 8]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 8, 8]
    )
    weights = numpy_helper.from_array(
        np.array(
            [
                [[[1.0, 0.0, -1.0], [0.5, 0.25, -0.5], [0.0, 1.0, 0.0]]],
                [[[-0.5, 0.25, 0.5], [1.0, -1.0, 0.0], [0.25, 0.0, -0.25]]],
            ],
            dtype=np.float32,
        ),
        "weights",
    )
    model = _model(
        "depthwise_conv",
        [
            helper.make_node(
                "Conv",
                ["input", "weights"],
                ["output"],
                pads=[1, 1, 1, 1],
                group=2,
            )
        ],
        [model_input],
        [model_output],
        [weights],
    )
    return ContractCase(
        "float_depthwise_conv",
        model,
        model,
        {"input": np.linspace(-2.0, 2.0, 128, dtype=np.float32).reshape(1, 2, 8, 8)},
        ("DepthwiseConv",),
    )


def _conv1d_case() -> ContractCase:
    """ONNX Conv1D relabeling and fused activation execution."""
    length = 256
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, length]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, length]
    )
    weights = numpy_helper.from_array(
        np.array(
            [
                [[0.5, 1.0, -0.5], [0.25, 0.0, 0.75]],
                [[-1.0, 0.5, 1.0], [0.5, -0.25, 0.25]],
                [[0.25, 0.25, 0.25], [-0.5, 1.0, -0.5]],
            ],
            dtype=np.float32,
        ),
        "weights",
    )
    bias = numpy_helper.from_array(
        np.array([0.1, -0.2, 0.3], dtype=np.float32), "bias"
    )
    model = _model(
        "conv1d",
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["convolved"],
                kernel_shape=[3],
                pads=[1, 1],
            ),
            helper.make_node("Relu", ["convolved"], ["output"]),
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    return ContractCase(
        "float_conv1d",
        model,
        model,
        {
            "input": np.linspace(
                -1.5, 1.5, 2 * length, dtype=np.float32
            ).reshape(1, 2, length)
        },
        ("Conv1D",),
        mem_budget="1K",
        expect_tiled=True,
    )


def _channel_bias_add_case(*, producer_has_bias: bool) -> ContractCase:
    """A per-channel constant Add must reach the same values as its own graph.

    The reference model keeps the standalone Add, so ONNX Runtime evaluates the
    broadcast exactly as an exporter wrote it; TiGrIS folds it into the Conv
    bias and must land on the same numbers either way.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 3, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 3, 3]
    )
    weight = numpy_helper.from_array(
        np.linspace(-0.4, 0.4, 3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3),
        "weight",
    )
    channel = numpy_helper.from_array(
        np.array([[[[0.5]], [[-1.25]], [[2.0]]]], dtype=np.float32), "channel"
    )
    conv_inputs = ["input", "weight"]
    initializers = [weight, channel]
    if producer_has_bias:
        initializers.append(
            numpy_helper.from_array(
                np.array([0.125, -0.25, 0.75], dtype=np.float32), "bias"
            )
        )
        conv_inputs.append("bias")
    model = _model(
        "channel_bias_add",
        [
            helper.make_node(
                "Conv", conv_inputs, ["product"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1],
            ),
            helper.make_node("Add", ["product", "channel"], ["output"]),
        ],
        [model_input],
        [model_output],
        initializers,
    )
    suffix = "onto_bias" if producer_has_bias else "as_bias"
    return ContractCase(
        f"float_channel_bias_add_{suffix}",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 18, dtype=np.float32
            ).reshape(1, 2, 3, 3)
        },
        ("Conv",),
    )


def _convtranspose_case() -> ContractCase:
    """A standalone float ConvTranspose upsampler (stride 2, kernel 2, pad 0).

    The ONNX ConvTranspose weight is [C_in, C_out, kH, kW]; the compiler
    transposes it to the runtime's OHWI layout. Output height/width follow the
    standard relation stride * (in - 1) + kernel - pad_begin - pad_end, so a
    4x4 input upsamples to 8x8. The compile and ORT reference models are
    identical, as in the other single-operator float cases.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    rng = np.random.default_rng(19)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 3, 2, 2)).astype(np.float32), "weights"
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(3,)).astype(np.float32), "bias"
    )
    model = _model(
        "convtranspose",
        [
            helper.make_node(
                "ConvTranspose",
                ["input", "weights", "bias"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                pads=[0, 0, 0, 0],
                group=1,
            )
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    return ContractCase(
        "float_convtranspose",
        model,
        model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4)).astype(np.float32)},
        ("ConvTranspose",),
    )


def _conv_then_convtranspose_case() -> ContractCase:
    """A strided Conv downsampler feeding a ConvTranspose upsampler.

    Conv (stride 2, kernel 2, pad 0) halves an 8x8 input to 4x4, then
    ConvTranspose (stride 2, kernel 2, pad 0) restores 8x8. Both stages keep
    group 1 so the Conv is not relabeled DepthwiseConv.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 8, 8]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 8, 8]
    )
    rng = np.random.default_rng(23)
    conv_weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 1, 2, 2)).astype(np.float32), "conv_weights"
    )
    convt_weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 1, 2, 2)).astype(np.float32), "convt_weights"
    )
    model = _model(
        "conv_then_convtranspose",
        [
            helper.make_node(
                "Conv",
                ["input", "conv_weights"],
                ["mid"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                pads=[0, 0, 0, 0],
                group=1,
            ),
            helper.make_node(
                "ConvTranspose",
                ["mid", "convt_weights"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                pads=[0, 0, 0, 0],
                group=1,
            ),
        ],
        [model_input],
        [model_output],
        [conv_weights, convt_weights],
    )
    return ContractCase(
        "float_conv_then_convtranspose",
        model,
        model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 1, 8, 8)).astype(np.float32)},
        ("Conv", "ConvTranspose"),
    )


def _convtranspose_overlap_case() -> ContractCase:
    """A float ConvTranspose whose kernel exceeds its stride, so multiple taps
    overlap-and-sum into each output pixel.

    A 4x4 kernel with stride 2 and pad 1 is the classic U-Net upsampler:
    output = stride * (in - 1) + kernel - pad_begin - pad_end = 2 * in, so a
    4x4 input doubles to 8x8. Because kernel (4) exceeds stride (2), up to two
    taps per axis (four total) accumulate into one output pixel, exercising the
    gather kernel's multi-tap accumulation against the ORT oracle rather than
    the single-tap stride==kernel path the other ConvTranspose cases cover.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    rng = np.random.default_rng(29)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 3, 4, 4)).astype(np.float32), "weights"
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(3,)).astype(np.float32), "bias"
    )
    model = _model(
        "convtranspose_overlap",
        [
            helper.make_node(
                "ConvTranspose",
                ["input", "weights", "bias"],
                ["output"],
                kernel_shape=[4, 4],
                strides=[2, 2],
                pads=[1, 1, 1, 1],
                group=1,
            )
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    return ContractCase(
        "float_convtranspose_overlap",
        model,
        model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4)).astype(np.float32)},
        ("ConvTranspose",),
    )


def _convtranspose_output_padding_case() -> ContractCase:
    """A float ConvTranspose with a nonzero output_padding attribute.

    Kernel 3, stride 2, symmetric pad 1: per the ONNX formula output = stride
    * (in - 1) + output_padding + kernel - pad_begin - pad_end, a 4x4 input
    with output_padding=0 would upsample to 7x7. Setting output_padding=[1,1]
    adds the extra trailing row/column to reach 8x8. The compiler reads that
    output shape from ONNX's own shape inference rather than re-deriving it
    (the runtime's gather kernel bounds itself against the allocated output
    extent, not a shrink formula), so this exercises the case the other
    ConvTranspose cases leave untested: the compiled output shape must match
    what output_padding actually produces, not what it would be without it.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    rng = np.random.default_rng(37)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 3, 3, 3)).astype(np.float32), "weights"
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(3,)).astype(np.float32), "bias"
    )
    model = _model(
        "convtranspose_output_padding",
        [
            helper.make_node(
                "ConvTranspose",
                ["input", "weights", "bias"],
                ["output"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                pads=[1, 1, 1, 1],
                output_padding=[1, 1],
                group=1,
            )
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    return ContractCase(
        "float_convtranspose_output_padding",
        model,
        model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4)).astype(np.float32)},
        ("ConvTranspose",),
    )


def _convtranspose_asymmetric_pad_case() -> ContractCase:
    """A float ConvTranspose with asymmetric pads (top != bottom, left != right).

    Kernel 3, stride 2, pads=[0, 1, 1, 0] (ONNX order [h_begin, w_begin,
    h_end, w_end]): pad_top=0/pad_bottom=1 on height, pad_left=1/pad_right=0
    on width. Both other ConvTranspose cases in this file use symmetric pads,
    so this is the only case where a pad_top/pad_bottom or pad_left/pad_right
    mixup in the compiler or the gather kernel's per-axis pad indexing would
    surface as a mismatch against the ORT oracle.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    rng = np.random.default_rng(41)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(2, 3, 3, 3)).astype(np.float32), "weights"
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.5, size=(3,)).astype(np.float32), "bias"
    )
    model = _model(
        "convtranspose_asymmetric_pad",
        [
            helper.make_node(
                "ConvTranspose",
                ["input", "weights", "bias"],
                ["output"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                pads=[0, 1, 1, 0],
                group=1,
            )
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    return ContractCase(
        "float_convtranspose_asymmetric_pad",
        model,
        model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4)).astype(np.float32)},
        ("ConvTranspose",),
    )
