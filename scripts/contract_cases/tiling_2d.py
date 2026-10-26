"""Cases tiled along both spatial axes."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import Array, ContractCase, _model, _qdq, _scalars


def build_conv(
    *,
    n: int,
    c_in: int,
    c_out: int,
    h: int,
    w: int,
    kernel: int,
    stride: int,
    pad: int,
    seed: int = 0,
) -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Array]]:
    """A single float32 Conv on an NCHW activation.

    The compile model and the ORT reference model are identical (as in
    _tiled_pool_case and _dilated_conv_case); only random weights and a
    uniform input distinguish instances at different resolutions.
    """
    out_h = (h + 2 * pad - kernel) // stride + 1
    out_w = (w + 2 * pad - kernel) // stride + 1
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [n, c_in, h, w]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [n, c_out, out_h, out_w]
    )
    rng = np.random.default_rng(seed)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c_out, c_in, kernel, kernel)).astype(
            np.float32
        ),
        "weights",
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c_out,)).astype(np.float32), "bias"
    )
    model = _model(
        "conv2d",
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["output"],
                name="conv0",
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
            )
        ],
        [model_input],
        [model_output],
        [weights, bias],
    )
    inputs = {
        "input": rng.uniform(-1.0, 1.0, size=(n, c_in, h, w)).astype(
            np.float32
        )
    }
    return model, model, inputs


def _2d_tiled_conv_case() -> ContractCase:
    """A high-res Conv whose 1D height-only tile is infeasible at the budget
    but a 2D (H and W) tile fits.

    Input NCHW [1, 64, 66, 66], a 3x3 stride-1 pad-1 Conv to [1, 64, 66, 66],
    at a 24K fast budget. 64 float32 channels give the same 256 bytes per
    pixel as the int8 sibling's 256 channels, so the compiler solves the same
    4x8 output tile. 66 is not divisible by 4 or 8, so the last row, the last
    column, and the bottom-right corner tile are all partial.

    Height-only tiling is infeasible first: partition_spatial only attempts
    the 2D solve after its 1D tile_h == 1 candidate still exceeds budget, so
    an emitted axis == TILE_AXIS_HW plan is itself proof the 1D path failed
    closed at this budget.
    """
    compile_model, reference_model, inputs = build_conv(
        n=1, c_in=64, c_out=64, h=66, w=66, kernel=3, stride=1, pad=1
    )
    return ContractCase(
        "float_2d_tiled_conv",
        compile_model,
        reference_model,
        inputs,
        expected_operators=("Conv",),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _strided_depthwise_2d_case(*, quantized: bool) -> ContractCase:
    """A stride-2 depthwise convolution tiled along height and width.

    The runtime reads the 2D tile shape as an output tile and back-computes
    its input rectangle, (t - 1) * 2 + 3 rows and columns, so a strided tile
    reads roughly twice the rows it writes. 34x34 in and 17x17 out leave a
    partial last row and column of tiles.
    """
    channels, side = 96, 34
    rng = np.random.default_rng(31)
    weight = (rng.normal(size=(channels, 1, 3, 3)) * 0.3).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    conv = dict(group=channels, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), out=(0.02, 0))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "out_s", "out_z", "output"),
        ]
    else:
        nodes = [helper.make_node("Conv", ["input", "weight"], ["output"], **conv)]
    label = f"{'int8' if quantized else 'float'}_strided_depthwise_2d"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side // 2, side // 2])],
        initializers,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("DepthwiseConv",), mem_budget="4K" if quantized else "16K",
                        expect_tiled=True, expect_2d=True)


def _qdq_2d_tiled_conv_case() -> ContractCase:
    """The int8 sibling of _2d_tiled_conv_case, built via the _qdq_case QDQ
    pattern: input NCHW [1, 256, 66, 66], a 3x3 stride-1 pad-1 Conv, 24K
    budget. 256 int8 channels give the same per-pixel byte footprint as the
    float case's 64 float32 channels, so the compiler solves the same 4x8
    2D output tile with the same partial last row, column, and corner.
    """
    h = w = 66
    c = 256
    kernel, stride, pad = 3, 1, 1
    out_h = (h + 2 * pad - kernel) // stride + 1
    out_w = (w + 2 * pad - kernel) // stride + 1
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, c, h, w]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c, out_h, out_w]
    )

    rng = np.random.default_rng(3)
    input_scale = numpy_helper.from_array(
        np.array([0.02], dtype=np.float32), "input_scale"
    )
    input_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "input_zero_point"
    )
    output_scale = numpy_helper.from_array(
        np.array([0.05], dtype=np.float32), "output_scale"
    )
    output_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "output_zero_point"
    )
    weight = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c, c, kernel, kernel)).astype(
            np.float32
        ),
        "weight",
    )
    weight_scale = numpy_helper.from_array(
        np.array([0.01], dtype=np.float32), "weight_scale"
    )
    weight_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "weight_zero_point"
    )
    initializers = [
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        weight,
        weight_scale,
        weight_zero_point,
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input_scale", "input_zero_point"],
            ["input_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "input_zero_point"],
            ["input_dq"],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zero_point"],
            ["weight_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zero_point"],
            ["weight_dq"],
        ),
        helper.make_node(
            "Conv",
            ["input_dq", "weight_dq"],
            ["raw"],
            name="conv0",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["raw", "output_scale", "output_zero_point"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zero_point"],
            ["output"],
        ),
    ]
    compile_model = _model(
        "qdq_2d_tiled_conv", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)

    input_data = rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32)
    return ContractCase(
        "int8_2d_tiled_conv",
        compile_model,
        reference_model,
        {"input": input_data},
        ("Conv",),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _2d_tiled_conv_sigmoid_case() -> ContractCase:
    """Conv followed by a non-fused pointwise Sigmoid, both forced 2D at the
    same 24K/66x66/64-channel geometry as _2d_tiled_conv_case.

    Relu/Relu6 fuse into the Conv at compile time, so this uses Sigmoid to
    keep the pointwise op a standalone stage. At this budget the row-based
    (full-width) chain streamer cannot fit even one row, so Conv and Sigmoid
    stay as two independent stages, each solving its own 4x5 HW tile; the
    Sigmoid stage exercises the 2D executor running a pointwise op on a
    packed (non-full-width) tile, which no other contract case covers.
    """
    h = w = 66
    c = 64
    kernel, stride, pad = 3, 1, 1
    rng = np.random.default_rng(5)
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, c, h, w]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c, h, w]
    )
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c, c, kernel, kernel)).astype(
            np.float32
        ),
        "weights",
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c,)).astype(np.float32), "bias"
    )
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "weights", "bias"],
            ["conv_out"],
            name="conv0",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
        ),
        helper.make_node("Sigmoid", ["conv_out"], ["output"]),
    ]
    model = _model(
        "conv_sigmoid_2d", nodes, [model_input], [model_output], [weights, bias]
    )
    inputs = {
        "input": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32)
    }
    return ContractCase(
        "float_2d_tiled_conv_sigmoid",
        model,
        model,
        inputs,
        expected_operators=("Conv", "Sigmoid"),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _cotiled_concat_2d_case() -> ContractCase:
    """A pre-spatial channel Concat of two same-resolution skips feeding a
    Conv, forced to 2D (HW) tiling at a 24K budget.

    up and skip are both NCHW [1, 32, 66, 66]. A channel-axis Concat builds
    cat [1, 64, 66, 66], which a 3x3 stride-1 pad-1 Conv maps to output
    [1, 32, 66, 66]. At this budget the greedy temporal partition keeps the
    Concat and the Conv as separate over-budget stages. The 2D eligibility
    rules let the multi-input Concat stage tile on both axes; both the Concat
    and the Conv stage solve a TILE_AXIS_HW tile with tiles_h > 1 and
    tiles_w > 1, so the plan's first tile-plan record proves 2D. The runtime
    loads up and skip at the same tile rectangle and must reproduce the
    whole-op result bit-exact.
    """
    h = w = 66
    c = 32
    kernel, stride, pad = 3, 1, 1
    rng = np.random.default_rng(17)
    up = helper.make_tensor_value_info("up", TensorProto.FLOAT, [1, c, h, w])
    skip = helper.make_tensor_value_info(
        "skip", TensorProto.FLOAT, [1, c, h, w]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c, h, w]
    )
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c, 2 * c, kernel, kernel)).astype(
            np.float32
        ),
        "weights",
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c,)).astype(np.float32), "bias"
    )
    model = _model(
        "cotiled_concat_2d",
        [
            helper.make_node("Concat", ["up", "skip"], ["cat"], axis=1),
            helper.make_node(
                "Conv",
                ["cat", "weights", "bias"],
                ["output"],
                name="conv0",
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
            ),
        ],
        [up, skip],
        [output],
        [weights, bias],
    )
    inputs = {
        "up": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
        "skip": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
    }
    return ContractCase(
        "float_cotiled_concat_2d",
        model,
        model,
        inputs,
        expected_operators=("Concat", "Conv"),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _qdq_cotiled_concat_2d_case() -> ContractCase:
    """The int8 sibling of _cotiled_concat_2d_case, built via the _qdq
    QDQ pattern.

    up and skip are NCHW [1, 128, 66, 66]; a channel-axis Concat builds
    cat [1, 256, 66, 66] which a 3x3 stride-1 pad-1 Conv maps to int8 output
    [1, 128, 66, 66]. up, skip, and cat share one scale/zero-point so the
    Concat is a lossless channel copy (the co-tiled multi-input LOAD path,
    not the requant path, is what this gate exercises); the single int8
    rounding boundary is the Conv output, exactly as in _qdq_2d_tiled_conv.
    Both the Concat and the Conv stage tile on TILE_AXIS_HW with tiles_h > 1
    and tiles_w > 1.
    """
    h = w = 66
    c = 128
    kernel, stride, pad = 3, 1, 1
    rng = np.random.default_rng(23)
    up = helper.make_tensor_value_info("up", TensorProto.FLOAT, [1, c, h, w])
    skip = helper.make_tensor_value_info(
        "skip", TensorProto.FLOAT, [1, c, h, w]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c, h, w]
    )

    def _scalar(value: float, name: str, dtype=np.float32) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array([value], dtype=dtype), name)

    # up, skip, and cat share one scale so the Concat rescales nothing.
    skip_scale = _scalar(0.02, "skip_scale")
    skip_zero_point = _scalar(0, "skip_zero_point", np.int8)
    up_scale = _scalar(0.02, "up_scale")
    up_zero_point = _scalar(0, "up_zero_point", np.int8)
    cat_scale = _scalar(0.02, "cat_scale")
    cat_zero_point = _scalar(0, "cat_zero_point", np.int8)
    output_scale = _scalar(0.05, "output_scale")
    output_zero_point = _scalar(0, "output_zero_point", np.int8)
    weight = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c, 2 * c, kernel, kernel)).astype(
            np.float32
        ),
        "weight",
    )
    weight_scale = _scalar(0.01, "weight_scale")
    weight_zero_point = _scalar(0, "weight_zero_point", np.int8)
    initializers = [
        up_scale,
        up_zero_point,
        skip_scale,
        skip_zero_point,
        cat_scale,
        cat_zero_point,
        output_scale,
        output_zero_point,
        weight,
        weight_scale,
        weight_zero_point,
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["up", "up_scale", "up_zero_point"], ["up_q"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["up_q", "up_scale", "up_zero_point"],
            ["up_dq"],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["skip", "skip_scale", "skip_zero_point"],
            ["skip_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["skip_q", "skip_scale", "skip_zero_point"],
            ["skip_dq"],
        ),
        helper.make_node("Concat", ["up_dq", "skip_dq"], ["cat"], axis=1),
        helper.make_node(
            "QuantizeLinear", ["cat", "cat_scale", "cat_zero_point"], ["cat_q"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["cat_q", "cat_scale", "cat_zero_point"],
            ["cat_dq"],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zero_point"],
            ["weight_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zero_point"],
            ["weight_dq"],
        ),
        helper.make_node(
            "Conv",
            ["cat_dq", "weight_dq"],
            ["raw"],
            name="conv0",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["raw", "output_scale", "output_zero_point"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zero_point"],
            ["output"],
        ),
    ]
    compile_model = _model(
        "qdq_cotiled_concat_2d",
        nodes,
        [up, skip],
        [model_output],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)

    inputs = {
        "up": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
        "skip": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
    }
    return ContractCase(
        "int8_cotiled_concat_2d",
        compile_model,
        reference_model,
        inputs,
        ("Concat", "Conv"),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _cotiled_add_2d_case() -> ContractCase:
    """A pre-spatial residual Add of two same-resolution operands feeding a
    Conv, forced to 2D (HW) tiling at a 24K budget.

    x and skip are both NCHW [1, 64, 66, 66]; Add produces added
    [1, 64, 66, 66], which a 3x3 stride-1 pad-1 Conv maps to output
    [1, 64, 66, 66]. Like the Concat sibling, the greedy temporal partition
    keeps the Add and the Conv as separate over-budget stages; the 2D
    eligibility rules let the multi-input Add stage tile on both axes. Both
    stages solve a TILE_AXIS_HW tile with tiles_h > 1 and tiles_w > 1, and
    the runtime loads x and skip at the same tile rectangle.
    """
    h = w = 66
    c = 64
    kernel, stride, pad = 3, 1, 1
    rng = np.random.default_rng(19)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, h, w])
    skip = helper.make_tensor_value_info(
        "skip", TensorProto.FLOAT, [1, c, h, w]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c, h, w]
    )
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c, c, kernel, kernel)).astype(np.float32),
        "weights",
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c,)).astype(np.float32), "bias"
    )
    model = _model(
        "cotiled_add_2d",
        [
            helper.make_node("Add", ["x", "skip"], ["added"]),
            helper.make_node(
                "Conv",
                ["added", "weights", "bias"],
                ["output"],
                name="conv0",
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
            ),
        ],
        [x, skip],
        [output],
        [weights, bias],
    )
    inputs = {
        "x": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
        "skip": rng.uniform(-1.0, 1.0, size=(1, c, h, w)).astype(np.float32),
    }
    return ContractCase(
        "float_cotiled_add_2d",
        model,
        model,
        inputs,
        expected_operators=("Add", "Conv"),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )


def _build_convtranspose_2d(
    c_in: int, c_out: int, h_in: int, w_in: int, seed: int
) -> tuple[onnx.ModelProto, dict[str, Array]]:
    """Build a stride-2 kernel-2 pad-0 float ConvTranspose upsampler.

    Mirrors _convtranspose_case's node and weight construction (ONNX weight
    layout [C_in, C_out, kH, kW], transposed to OHWI by the compiler) but
    parameterizes the geometry so a caller can drive the stage over a tight
    budget. Output height and width follow stride * (in - 1) + kernel = 2 * in,
    so the tensor doubles on each spatial axis; the compiler grids the 2D tile
    over that expanded OUTPUT extent, not the pre-upsample input.
    """
    out_h, out_w = 2 * h_in, 2 * w_in
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, c_in, h_in, w_in]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c_out, out_h, out_w]
    )
    rng = np.random.default_rng(seed)
    weights = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c_in, c_out, 2, 2)).astype(np.float32),
        "weights",
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c_out,)).astype(np.float32), "bias"
    )
    model = _model(
        "convtranspose2d",
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
    inputs = {
        "input": rng.uniform(-1.0, 1.0, size=(1, c_in, h_in, w_in)).astype(
            np.float32
        )
    }
    return model, inputs


def _convtranspose_2d_tiled_case() -> ContractCase:
    """A stride-2 ConvTranspose whose expanded output overflows an 8K budget,
    forcing a 2D (HW) tile.

    Input NCHW [1, 24, 16, 16] upsamples to output [1, 24, 32, 32]. The stage's
    ~120 KB activation peak far exceeds the 8K fast budget, and ConvTranspose
    is kept untileable on the 1D height and chain paths, so it routes only
    through the compiler's dedicated output-extent 2D solve, which splits both
    the height and width of the 32x32 output. The solved core tile does not evenly divide the output, so the last
    tile row, the last tile column, and the bottom-right corner tile are all
    partial. An emitted axis == TILE_AXIS_HW plan is itself proof the isolated
    ConvTranspose 2D branch fired, and the runtime must reproduce ORT's float
    upsample bit-exact across every tile.
    """
    model, inputs = _build_convtranspose_2d(
        c_in=24, c_out=24, h_in=16, w_in=16, seed=19
    )
    return ContractCase(
        "float_convtranspose_2d",
        model,
        model,
        inputs,
        expected_operators=("ConvTranspose",),
        mem_budget="8K",
        expect_tiled=True,
        expect_2d=True,
    )


def _qdq_convtranspose_2d_tiled_case() -> ContractCase:
    """The int8 sibling of _convtranspose_2d_tiled_case, built with the same QDQ
    pattern as _qdq_2d_tiled_conv_case but wrapping a stride-2 ConvTranspose.

    Input NCHW [1, 48, 16, 16] upsamples to [1, 48, 32, 32]. int8 activations
    are half the per-pixel footprint of the float case (48 int8 channels vs 24
    float32), and the 4K budget is half the float case's 8K, so the compiler
    splits both the height and width of the 32x32 output into a 2D tile grid.
    The solved core tile does not evenly divide the output, so the last row,
    column, and corner tiles are partial. The runtime executes the s8 reference
    ConvTranspose kernel under the 2D tile context and must match ORT's int8 QDQ
    reference to one LSB.
    """
    c_in = c_out = 48
    h_in = w_in = 16
    out_h, out_w = 2 * h_in, 2 * w_in
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, c_in, h_in, w_in]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, c_out, out_h, out_w]
    )
    rng = np.random.default_rng(7)
    input_scale = numpy_helper.from_array(
        np.array([0.02], dtype=np.float32), "input_scale"
    )
    input_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "input_zero_point"
    )
    output_scale = numpy_helper.from_array(
        np.array([0.05], dtype=np.float32), "output_scale"
    )
    output_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "output_zero_point"
    )
    weight = numpy_helper.from_array(
        rng.normal(0.0, 0.05, size=(c_in, c_out, 2, 2)).astype(np.float32),
        "weight",
    )
    weight_scale = numpy_helper.from_array(
        np.array([0.01], dtype=np.float32), "weight_scale"
    )
    weight_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "weight_zero_point"
    )
    initializers = [
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        weight,
        weight_scale,
        weight_zero_point,
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input_scale", "input_zero_point"],
            ["input_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "input_zero_point"],
            ["input_dq"],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zero_point"],
            ["weight_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zero_point"],
            ["weight_dq"],
        ),
        helper.make_node(
            "ConvTranspose",
            ["input_dq", "weight_dq"],
            ["raw"],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            group=1,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["raw", "output_scale", "output_zero_point"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zero_point"],
            ["output"],
        ),
    ]
    compile_model = _model(
        "qdq_convtranspose_2d",
        nodes,
        [model_input],
        [model_output],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    input_data = rng.uniform(-1.0, 1.0, size=(1, c_in, h_in, w_in)).astype(
        np.float32
    )
    return ContractCase(
        "int8_convtranspose_2d",
        compile_model,
        reference_model,
        {"input": input_data},
        ("ConvTranspose",),
        mem_budget="4K",
        expect_tiled=True,
        expect_2d=True,
    )


def _convtranspose_2d_partial_edge_case() -> ContractCase:
    """A stride-2 ConvTranspose with a non-square input, whose 2D tile does NOT
    evenly divide the output, exercising the partial edge and corner tiles.

    Input NCHW [1, 32, 15, 15] upsamples to output [1, 24, 30, 30]. At a 24K
    budget the ConvTranspose solve splits both the height and width of the 30x30
    output into a non-square core tile that divides neither axis evenly, so the
    last tile row, the last tile column, and the bottom-right corner tile are all
    partial (and generally differently sized). The runtime must place every
    partial edge and corner tile at the correct output offset and still match ORT
    bit-exact, which exercises the inverted input rectangle and effective
    pads of each tile.
    """
    model, inputs = _build_convtranspose_2d(
        c_in=32, c_out=24, h_in=15, w_in=15, seed=23
    )
    return ContractCase(
        "float_convtranspose_2d_partial",
        model,
        model,
        inputs,
        expected_operators=("ConvTranspose",),
        mem_budget="24K",
        expect_tiled=True,
        expect_2d=True,
    )
