"""Cases tiled along both spatial axes."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import (
    Array, ContractCase, _model, _qdq, _qdq_named, _qdq_operator_model, _qparams, _scalars, _vi,
)


# The Conv of the 66x66 cases, float and int8.
_CONV0 = dict(name="conv0", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])


def _conv0(rng: np.random.Generator, source: str, out: str, c_in: int,
           c_out: int) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """A float 3x3 stride-1 pad-1 Conv named conv0, with weights and bias drawn from `rng`."""
    initializers = [
        numpy_helper.from_array(
            rng.normal(0.0, 0.05, size=(c_out, c_in, 3, 3)).astype(np.float32), "weights"),
        numpy_helper.from_array(rng.normal(0.0, 0.05, size=(c_out,)).astype(np.float32), "bias"),
    ]
    return helper.make_node("Conv", [source, "weights", "bias"], [out], **_CONV0), initializers


def _2d_tiled_conv_case() -> ContractCase:
    """A high-res Conv whose 1D height-only tile is infeasible at the budget
    but a 2D (H and W) tile fits.

    Input NCHW [1, 64, 66, 66], a 3x3 stride-1 pad-1 Conv to [1, 64, 66, 66],
    at a 24K fast budget. 64 float32 channels give the same 256 bytes per
    pixel as the int8 sibling's 256 channels, so the compiler solves the same
    4x8 output tile. 66 is not divisible by 4 or 8, so the last row, the last
    column, and the bottom-right corner tile are all partial.

    Height-only tiling is infeasible at this budget: a one-row stripe holds
    three input rows and one output row of 16,896 bytes each (67,584 bytes),
    more than 24K, so the stage takes the 2D tile.
    """
    rng = np.random.default_rng(0)
    conv, initializers = _conv0(rng, "input", "output", 64, 64)
    model = _model("conv2d", [conv], [_vi("input", [1, 64, 66, 66])],
                   [_vi("output", [1, 64, 66, 66])], initializers)
    return ContractCase(
        "float_2d_tiled_conv", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, 64, 66, 66)).astype(np.float32)},
        expected_operators=("Conv",), mem_budget="24K", expect_tiled=True, expect_2d=True,
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
    """The int8 sibling of _2d_tiled_conv_case: input NCHW [1, 256, 66, 66], a
    3x3 stride-1 pad-1 Conv, 24K budget. 256 int8 channels give the same
    per-pixel byte footprint as the float case's 64 float32 channels, so the
    compiler solves the same 4x8 2D output tile with the same partial last row,
    column, and corner.
    """
    shape = [1, 256, 66, 66]
    rng = np.random.default_rng(3)
    weight = rng.normal(0.0, 0.05, size=(256, 256, 3, 3)).astype(np.float32)
    model = _qdq_operator_model("qdq_2d_tiled_conv", "Conv", shape, shape, ([0.02], [0]),
                                ([0.05], [0]), weight, ([0.01], [0]), **_CONV0)
    return ContractCase(
        "int8_2d_tiled_conv", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)},
        ("Conv",), mem_budget="24K", expect_tiled=True, expect_2d=True,
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
    shape = [1, 64, 66, 66]
    rng = np.random.default_rng(5)
    conv, initializers = _conv0(rng, "input", "conv_out", 64, 64)
    model = _model("conv_sigmoid_2d", [conv, helper.make_node("Sigmoid", ["conv_out"], ["output"])],
                   [_vi("input", shape)], [_vi("output", shape)], initializers)
    return ContractCase(
        "float_2d_tiled_conv_sigmoid", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)},
        expected_operators=("Conv", "Sigmoid"), mem_budget="24K", expect_tiled=True, expect_2d=True,
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
    shape = [1, 32, 66, 66]
    rng = np.random.default_rng(17)
    conv, initializers = _conv0(rng, "cat", "output", 64, 32)
    model = _model(
        "cotiled_concat_2d", [helper.make_node("Concat", ["up", "skip"], ["cat"], axis=1), conv],
        [_vi("up", shape), _vi("skip", shape)], [_vi("output", shape)], initializers,
    )
    inputs = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for name in ("up", "skip")}
    return ContractCase(
        "float_cotiled_concat_2d", model, model, inputs,
        expected_operators=("Concat", "Conv"), mem_budget="24K", expect_tiled=True, expect_2d=True,
    )


def _qdq_cotiled_concat_2d_case() -> ContractCase:
    """The int8 sibling of _cotiled_concat_2d_case.

    up and skip are NCHW [1, 128, 66, 66]; a channel-axis Concat builds
    cat [1, 256, 66, 66] which a 3x3 stride-1 pad-1 Conv maps to int8 output
    [1, 128, 66, 66]. up, skip, and cat share one scale/zero-point so the
    Concat is a lossless channel copy (the co-tiled multi-input LOAD path,
    not the requant path, is what this gate exercises); the single int8
    rounding boundary is the Conv output, exactly as in _qdq_2d_tiled_conv.
    Both the Concat and the Conv stage tile on TILE_AXIS_HW with tiles_h > 1
    and tiles_w > 1.
    """
    c = 128
    shape = [1, c, 66, 66]
    rng = np.random.default_rng(23)
    initializers = [
        tensor for prefix, scale in (("up", 0.02), ("skip", 0.02), ("cat", 0.02), ("output", 0.05))
        for tensor in _qparams(f"{prefix}_scale", f"{prefix}_zero_point", [scale], [0])
    ] + [
        numpy_helper.from_array(
            rng.normal(0.0, 0.05, size=(c, 2 * c, 3, 3)).astype(np.float32), "weight"),
        *_qparams("weight_scale", "weight_zero_point", [0.01], [0]),
    ]
    nodes = [
        *_qdq_named("up"),
        *_qdq_named("skip"),
        helper.make_node("Concat", ["up_dq", "skip_dq"], ["cat"], axis=1),
        *_qdq_named("cat"),
        *_qdq_named("weight"),
        helper.make_node("Conv", ["cat_dq", "weight_dq"], ["raw"], **_CONV0),
        *_qdq_named("output", "raw", "output"),
    ]
    model = _model("qdq_cotiled_concat_2d", nodes, [_vi("up", shape), _vi("skip", shape)],
                   [_vi("output", shape)], initializers)
    inputs = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for name in ("up", "skip")}
    return ContractCase(
        "int8_cotiled_concat_2d", model, model, inputs,
        ("Concat", "Conv"), mem_budget="24K", expect_tiled=True, expect_2d=True,
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
    shape = [1, 64, 66, 66]
    rng = np.random.default_rng(19)
    conv, initializers = _conv0(rng, "added", "output", 64, 64)
    model = _model(
        "cotiled_add_2d", [helper.make_node("Add", ["x", "skip"], ["added"]), conv],
        [_vi("x", shape), _vi("skip", shape)], [_vi("output", shape)], initializers,
    )
    inputs = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for name in ("x", "skip")}
    return ContractCase(
        "float_cotiled_add_2d", model, model, inputs,
        expected_operators=("Add", "Conv"), mem_budget="24K", expect_tiled=True, expect_2d=True,
    )


def _build_convtranspose_2d(
    c_in: int, c_out: int, h_in: int, w_in: int, seed: int
) -> tuple[onnx.ModelProto, dict[str, Array]]:
    """Build a stride-2 kernel-2 pad-0 float ConvTranspose upsampler.

    The node and weight layout of _convtranspose_case (ONNX weight [C_in, C_out, kH, kW],
    transposed to OHWI by the compiler) with the geometry as parameters, so a caller can
    drive the stage over a tight budget. Height and width double: stride * (in - 1) +
    kernel = 2 * in, and the compiler grids the 2D tile over that expanded OUTPUT
    extent, not the pre-upsample input.
    """
    rng = np.random.default_rng(seed)
    initializers = [
        numpy_helper.from_array(
            rng.normal(0.0, 0.05, size=(c_in, c_out, 2, 2)).astype(np.float32), "weights"),
        numpy_helper.from_array(rng.normal(0.0, 0.05, size=(c_out,)).astype(np.float32), "bias"),
    ]
    model = _model(
        "convtranspose2d",
        [helper.make_node("ConvTranspose", ["input", "weights", "bias"], ["output"],
                          kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], group=1)],
        [_vi("input", [1, c_in, h_in, w_in])], [_vi("output", [1, c_out, 2 * h_in, 2 * w_in])],
        initializers,
    )
    return model, {"input": rng.uniform(-1.0, 1.0, size=(1, c_in, h_in, w_in)).astype(np.float32)}


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
    c, side = 48, 16
    rng = np.random.default_rng(7)
    weight = rng.normal(0.0, 0.05, size=(c, c, 2, 2)).astype(np.float32)
    model = _qdq_operator_model(
        "qdq_convtranspose_2d", "ConvTranspose", [1, c, side, side], [1, c, 2 * side, 2 * side],
        ([0.02], [0]), ([0.05], [0]), weight, ([0.01], [0]),
        kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], group=1,
    )
    return ContractCase(
        "int8_convtranspose_2d", model, model,
        {"input": rng.uniform(-1.0, 1.0, size=(1, c, side, side)).astype(np.float32)},
        ("ConvTranspose",), mem_budget="4K", expect_tiled=True, expect_2d=True,
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
