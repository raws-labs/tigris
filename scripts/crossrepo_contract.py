#!/usr/bin/env python3
"""Prove a deterministic compiler -> runtime contract against ONNX Runtime.

The gate intentionally uses only the float32 and s8 reference dispatchers.
Accelerated backends retain their own target-specific parity tests; this checks
that plans emitted by the compiler execute with the sibling runtime's public
loader, arena, executor, and raw I/O contract.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from click import ClickException
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from tigris import TILE_AXIS_HW
from tigris.analysis.validation import validate_memory_plan
from tigris.capabilities import KERNEL_CAPABILITIES, OP_TYPE_BY_CODE
from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import (
    COMPRESS_LZ4,
    FLAG_XIP,
    STAGE_FLAG_LINE_BUFFERED,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary
from tigris.fixtures import build_tcn
from tigris.graph.ir import Stage

# The byte-level line-buffer flag decoder already exists in the compiler's
# plan test-suite; reuse it rather than duplicating the stage-record parser.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_linebuffer_plan import _stage_reserved1  # noqa: E402
from test_2d_tiling_plan import decode_first_tile_plan  # noqa: E402


Array = NDArray[np.generic]
_DTYPE_BY_ONNX_CODE = {
    TensorProto.FLOAT: np.dtype("<f4"),
    TensorProto.INT8: np.dtype("i1"),
}
_INT8_LSB_TOLERANCE = 1


@dataclass(frozen=True)
class ContractCase:
    """A compile model, independent ORT reference model, and model inputs."""

    name: str
    compile_model: onnx.ModelProto
    reference_model: onnx.ModelProto
    inputs: dict[str, Array]
    expected_operators: tuple[str, ...]
    mem_budget: str = "4K"
    compression: str | None = None
    xip: bool = False
    expect_tiled: bool = False
    expect_chain: bool = False
    expect_line_buffered: bool = False
    expect_2d: bool = False
    recompute_metric: bool = False
    force_one_op_stages: bool = False


def _model(
    name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] = [],
) -> onnx.ModelProto:
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _constant_add_case() -> ContractCase:
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    constant = numpy_helper.from_array(
        np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32), "constant"
    )
    model = _model(
        "constant_add",
        [
            helper.make_node("Add", ["input", "constant"], ["shifted"]),
            helper.make_node("Relu", ["shifted"], ["output"]),
        ],
        [model_input],
        [model_output],
        [constant],
    )
    return ContractCase(
        "float_constant_add",
        model,
        model,
        {
            "input": np.array(
                [[0.25, -1.0, 5.0, -4.0]], dtype=np.float32
            )
        },
        ("Add", "Relu"),
    )


def _residual_case() -> ContractCase:
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    model = _model(
        "residual",
        [
            helper.make_node("Relu", ["input"], ["left"]),
            helper.make_node("Sigmoid", ["input"], ["right"]),
            helper.make_node("Add", ["left", "right"], ["output"]),
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_residual",
        model,
        model,
        {
            "input": np.array(
                [[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32
            )
        },
        ("Relu", "Sigmoid", "Add"),
    )


def _output_transpose_case() -> ContractCase:
    """A public Transpose must retain its ONNX shape and element ordering."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [3, 1, 2]
    )
    model = _model(
        "output_transpose",
        [
            helper.make_node(
                "Transpose", ["input"], ["output"], perm=[2, 0, 1]
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_output_transpose",
        model,
        model,
        {"input": np.array([[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]], dtype=np.float32)},
        ("Transpose",),
    )


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


def _math_normalization_case() -> ContractCase:
    """Constant folding, Clip->Relu6, unary math, Mul, and Softmax."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["clip_min"],
            value=numpy_helper.from_array(np.array(0.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["clip_max"],
            value=numpy_helper.from_array(np.array(6.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Clip", ["input", "clip_min", "clip_max"], ["clipped"]
        ),
        helper.make_node("Tanh", ["clipped"], ["tanh"]),
        helper.make_node("Sigmoid", ["clipped"], ["sigmoid"]),
        helper.make_node("Mul", ["tanh", "sigmoid"], ["product"]),
        helper.make_node("Softmax", ["product"], ["output"], axis=1),
    ]
    model = _model(
        "math_normalization", nodes, [model_input], [model_output]
    )
    return ContractCase(
        "float_math_normalization",
        model,
        model,
        {
            "input": np.array(
                [[-3.0, 0.25, 2.0, 9.0]], dtype=np.float32
            )
        },
        ("Relu6", "Tanh", "Sigmoid", "Mul", "Softmax"),
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


def _rank3_pointwise_case() -> ContractCase:
    """Rank-3 unary and exact-shape binary ops tile along NLC length."""
    length = 256
    left = helper.make_tensor_value_info(
        "left", TensorProto.FLOAT, [1, 4, length]
    )
    right = helper.make_tensor_value_info(
        "right", TensorProto.FLOAT, [1, 4, length]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, length]
    )
    model = _model(
        "rank3_pointwise",
        [
            helper.make_node("Tanh", ["left"], ["left_tanh"]),
            helper.make_node("Sigmoid", ["right"], ["right_sigmoid"]),
            helper.make_node(
                "Mul", ["left_tanh", "right_sigmoid"], ["gated"]
            ),
            helper.make_node("Add", ["gated", "left"], ["output"]),
        ],
        [left, right],
        [output],
    )
    return ContractCase(
        "float_rank3_pointwise",
        model,
        model,
        {
            "left": np.linspace(
                -2.0, 2.0, 4 * length, dtype=np.float32
            ).reshape(1, 4, length),
            "right": np.linspace(
                1.5, -1.5, 4 * length, dtype=np.float32
            ).reshape(1, 4, length),
        },
        ("Tanh", "Sigmoid", "Mul", "Add"),
        mem_budget="1K",
        expect_tiled=True,
    )


def _many_stage_case() -> ContractCase:
    """Schema v5 derives full stage IDs from the uint16 stage table."""
    stage_count = 300
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    nodes = []
    previous = "input"
    for index in range(stage_count):
        output = "output" if index == stage_count - 1 else f"value_{index}"
        nodes.append(helper.make_node("Relu", [previous], [output]))
        previous = output
    model = _model(
        "many_stage_relu", nodes, [model_input], [model_output]
    )
    return ContractCase(
        "float_schema_v5_many_stage",
        model,
        model,
        {"input": np.array([[-2.0, -0.5, 0.25, 3.0]], dtype=np.float32)},
        ("Relu",) * stage_count,
        mem_budget="64",
        force_one_op_stages=True,
    )


def _tcn_16k_case() -> ContractCase:
    """The project TCN becomes deployable once its gated pointwise path tiles."""
    model = build_tcn()
    return ContractCase(
        "float_tcn_16k",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 3 * 128, dtype=np.float32
            ).reshape(1, 3, 128)
        },
        (
            "Conv1D",
            "Conv1D",
            "Conv1D",
            "Tanh",
            "Conv1D",
            "Sigmoid",
            "Mul",
            "Conv1D",
            "Flatten",
            "Gemm",
        ),
        mem_budget="16K",
        expect_tiled=True,
    )


def _reduce_mean_case() -> ContractCase:
    """ReduceMean over spatial axes must execute as GlobalAveragePool."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 1, 1]
    )
    model = _model(
        "reduce_mean_to_gap",
        [
            helper.make_node(
                "ReduceMean",
                ["input"],
                ["output"],
                axes=[2, 3],
                keepdims=1,
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_reduce_mean_to_gap",
        model,
        model,
        {
            "input": np.arange(
                18, dtype=np.float32
            ).reshape(1, 3, 2, 3)
        },
        ("GlobalAveragePool",),
    )


def _normalized_classifier_case() -> ContractCase:
    """BN, Relu6 fusion, shape folding, pooling, reshape, FC, flatten."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([[[[0.75]]], [[[-0.5]]]], dtype=np.float32),
            "conv_weights",
        ),
        numpy_helper.from_array(
            np.array([0.1, -0.2], dtype=np.float32), "conv_bias"
        ),
        numpy_helper.from_array(
            np.array([1.2, 0.8], dtype=np.float32), "bn_scale"
        ),
        numpy_helper.from_array(
            np.array([0.1, -0.1], dtype=np.float32), "bn_bias"
        ),
        numpy_helper.from_array(
            np.array([0.2, -0.3], dtype=np.float32), "bn_mean"
        ),
        numpy_helper.from_array(
            np.array([0.5, 0.25], dtype=np.float32), "bn_var"
        ),
        numpy_helper.from_array(
            np.linspace(-0.4, 0.5, 32, dtype=np.float32).reshape(4, 8),
            "fc_weights",
        ),
        numpy_helper.from_array(
            np.array([0.05, -0.1, 0.15, 0.2], dtype=np.float32),
            "fc_bias",
        ),
    ]
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["clip_min"],
            value=numpy_helper.from_array(np.array(0.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["clip_max"],
            value=numpy_helper.from_array(np.array(6.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["shape_index"],
            value=numpy_helper.from_array(np.array(0, dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["unsqueeze_axes"],
            value=numpy_helper.from_array(np.array([0], dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["flat_tail"],
            value=numpy_helper.from_array(np.array([-1], dtype=np.int64)),
        ),
        helper.make_node(
            "Conv",
            ["input", "conv_weights", "conv_bias"],
            ["conv"],
            kernel_shape=[1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["normalized"],
        ),
        helper.make_node(
            "Clip",
            ["normalized", "clip_min", "clip_max"],
            ["activated"],
        ),
        helper.make_node(
            "MaxPool",
            ["activated"],
            ["pooled"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node("Shape", ["pooled"], ["pool_shape"]),
        helper.make_node(
            "Gather", ["pool_shape", "shape_index"], ["batch"], axis=0
        ),
        helper.make_node(
            "Unsqueeze", ["batch", "unsqueeze_axes"], ["batch_vector"]
        ),
        helper.make_node(
            "Concat",
            ["batch_vector", "flat_tail"],
            ["target_shape"],
            axis=0,
        ),
        helper.make_node(
            "Reshape", ["pooled", "target_shape"], ["reshaped"]
        ),
        helper.make_node(
            "Gemm",
            ["reshaped", "fc_weights", "fc_bias"],
            ["classified"],
            transB=1,
        ),
        helper.make_node("Flatten", ["classified"], ["output"], axis=1),
    ]
    model = _model(
        "normalized_classifier",
        nodes,
        [model_input],
        [model_output],
        initializers,
    )
    # The deployment shape is static even though the ONNX graph computes its
    # Reshape input through Shape/Gather/Unsqueeze/Concat. Preserve the
    # compiler's fail-closed dynamic-shape rule by declaring that known value.
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "reshaped", TensorProto.FLOAT, [1, 8]
            )
        ]
    )
    onnx.checker.check_model(model)
    return ContractCase(
        "float_normalized_classifier",
        model,
        model,
        {
            "input": np.array(
                [
                    [
                        [
                            [-1.0, 0.0, 1.0, 2.0],
                            [3.0, 4.0, 5.0, 6.0],
                            [2.5, 1.5, 0.5, -0.5],
                            [4.5, 3.5, 2.5, 1.5],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        },
        ("Conv", "MaxPool", "Reshape", "Gemm", "Flatten"),
        compression="lz4",
    )


def _resize_concat_case() -> ContractCase:
    """Resize scale extraction and NCHW-to-NHWC Concat-axis mapping."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            "left_scales",
        ),
        numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            "right_scales",
        ),
        numpy_helper.from_array(
            np.array(1.5, dtype=np.float32), "multiplier"
        ),
    ]
    resize_attrs = {
        "coordinate_transformation_mode": "asymmetric",
        "mode": "nearest",
        "nearest_mode": "floor",
    }
    model = _model(
        "resize_concat",
        [
            helper.make_node(
                "Resize",
                ["input", "", "left_scales"],
                ["left"],
                **resize_attrs,
            ),
            helper.make_node(
                "Mul", ["input", "multiplier"], ["scaled"]
            ),
            helper.make_node(
                "Resize",
                ["scaled", "", "right_scales"],
                ["right"],
                **resize_attrs,
            ),
            helper.make_node(
                "Concat", ["left", "right"], ["output"], axis=1
            ),
        ],
        [model_input],
        [model_output],
        initializers,
    )
    return ContractCase(
        "float_resize_concat",
        model,
        model,
        {
            "input": np.array(
                [[[[1.0, -2.0], [3.0, 0.5]]]], dtype=np.float32
            )
        },
        ("Resize", "Mul", "Resize", "Concat"),
    )


def _tiled_pool_case() -> ContractCase:
    """A pool case large enough to require standalone tiled execution."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 32, 32]
    )
    model = _model(
        "tiled_pool",
        [
            helper.make_node(
                "AveragePool",
                ["input"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_tiled_averagepool",
        model,
        model,
        {"input": np.arange(4096, dtype=np.float32).reshape(1, 1, 64, 64) / 64.0},
        ("AveragePool",),
        mem_budget="4K",
        expect_tiled=True,
    )


def _tiled_pool_chain_case() -> ContractCase:
    """AveragePool -> MaxPool must stream with both spatial ranges composed."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 16, 16]
    )
    model = _model(
        "tiled_pool_chain",
        [
            helper.make_node(
                "AveragePool",
                ["input"],
                ["average"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
            helper.make_node(
                "MaxPool",
                ["average"],
                ["output"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_tiled_pool_chain",
        model,
        model,
        {
            "input": np.linspace(
                -2.0, 3.0, 4096, dtype=np.float32
            ).reshape(1, 1, 64, 64)
        },
        ("AveragePool", "MaxPool"),
        mem_budget="4K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
    )


def _tiled_chain_case(*, compression: str | None = None, xip: bool = False) -> ContractCase:
    """Three padded Conv stages force the streamable-chain executor."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 64, 64]
    )
    rng = np.random.default_rng(7)
    weights = [
        numpy_helper.from_array(
            rng.normal(0.0, 0.05, size=shape).astype(np.float32), name
        )
        for name, shape in (
            ("w0", (4, 3, 3, 3)),
            ("w1", (4, 4, 3, 3)),
            ("w2", (8, 4, 3, 3)),
        )
    ]
    nodes = [
        helper.make_node(
            "Conv", ["input", "w0"], ["mid0"], pads=[1, 1, 1, 1]
        ),
        helper.make_node(
            "Conv", ["mid0", "w1"], ["mid1"], pads=[1, 1, 1, 1]
        ),
        helper.make_node(
            "Conv", ["mid1", "w2"], ["output"], pads=[1, 1, 1, 1]
        ),
    ]
    model = _model("tiled_chain", nodes, [model_input], [model_output], weights)
    suffix = "lz4" if compression else "xip"
    return ContractCase(
        f"float_tiled_chain_{suffix}",
        model,
        model,
        {"input": np.linspace(-1.0, 1.0, 12288, dtype=np.float32).reshape(1, 3, 64, 64)},
        ("Conv", "Conv", "Conv"),
        mem_budget="32K",
        compression=compression,
        xip=xip,
        expect_tiled=True,
        expect_chain=True,
    )


def _qdq_case(operator: str) -> ContractCase:
    """Build a QDQ Conv or AveragePool model with an int8 ORT reference."""
    output_shape = [1, 1, 4, 4] if operator == "Conv" else [1, 1, 2, 2]
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )
    int8_output = helper.make_tensor_value_info(
        "output_q", TensorProto.INT8, output_shape
    )

    input_scale = numpy_helper.from_array(
        np.array([0.25], dtype=np.float32), "input_scale"
    )
    input_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "input_zero_point"
    )
    output_scale = numpy_helper.from_array(
        np.array([0.25], dtype=np.float32), "output_scale"
    )
    output_zero_point = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "output_zero_point"
    )
    initializers = [
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
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
    ]
    if operator == "Conv":
        weight = numpy_helper.from_array(
            np.array([[[[0.5]]]], dtype=np.float32), "weight"
        )
        weight_scale = numpy_helper.from_array(
            np.array([0.25], dtype=np.float32), "weight_scale"
        )
        weight_zero_point = numpy_helper.from_array(
            np.array([0], dtype=np.int8), "weight_zero_point"
        )
        initializers.extend([weight, weight_scale, weight_zero_point])
        nodes.extend(
            [
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
                helper.make_node("Conv", ["input_dq", "weight_dq"], ["raw"]),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "AveragePool",
                ["input_dq"],
                ["raw"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        )
    nodes.extend(
        [
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
    )
    compile_model = _model(
        f"qdq_{operator.lower()}", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    del reference_model.graph.output[:]
    reference_model.graph.output.extend([int8_output])
    onnx.checker.check_model(reference_model)
    input_data = np.array(
        [
            [
                [
                    [-1.0, -0.75, -0.5, -0.25],
                    [0.0, 0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5, 1.75],
                    [2.0, 2.25, 2.5, 2.75],
                ]
            ]
        ],
        dtype=np.float32,
    )
    return ContractCase(
        f"int8_{operator.lower()}",
        compile_model,
        reference_model,
        {"input": input_data},
        (operator,),
    )


def _linebuffer_conv_chain_case() -> ContractCase:
    """A padded Conv chain compiled tight enough to be line-buffered.

    Three 3x3 stride-1 padded Conv stages each compose a halo of 2 rows, so at
    a 32K budget the compiler forms a recomputing chain and marks the head
    line-buffered. The runtime re-derives a tile height of 3 against that
    budget (22 tiles over an output height of 64, with a partial last tile of
    one row), so a single execution exercises tile 0, interior tiles, and the
    partial last tile. The Conv nodes carry an explicit kernel_shape so the
    compiler's halo analysis sees the real 3x3 receptive field. Drives both the
    differential parity check and the recompute-reduction metric.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 64, 64]
    )
    rng = np.random.default_rng(7)
    weights = [
        numpy_helper.from_array(
            rng.normal(0.0, 0.05, size=shape).astype(np.float32), name
        )
        for name, shape in (
            ("w0", (4, 3, 3, 3)),
            ("w1", (4, 4, 3, 3)),
            ("w2", (8, 4, 3, 3)),
        )
    ]
    nodes = [
        helper.make_node(
            "Conv", ["input", "w0"], ["mid0"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]
        ),
        helper.make_node(
            "Conv", ["mid0", "w1"], ["mid1"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]
        ),
        helper.make_node(
            "Conv", ["mid1", "w2"], ["output"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]
        ),
    ]
    model = _model(
        "linebuffer_conv_chain", nodes, [model_input], [model_output], weights
    )
    return ContractCase(
        "float_linebuffer_conv_chain",
        model,
        model,
        {"input": np.linspace(-1.0, 1.0, 12288, dtype=np.float32).reshape(1, 3, 64, 64)},
        ("Conv", "Conv", "Conv"),
        mem_budget="32K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
        recompute_metric=True,
    )


def _qdq_conv_chain_case() -> ContractCase:
    """The int8 twin of the line-buffered Conv chain.

    Three QDQ 3x3 stride-1 padded Conv stages, wrapped exactly as ``_qdq_case``
    wraps a single Conv, fold to an int8 Conv chain. At an 8K budget the
    compiler forms the recomputing chain and marks the head line-buffered; the
    runtime re-derives a tile height of 3 (22 tiles over output height 64, a
    partial last tile of one row). The int8 ORT reference is the quantized
    (post-final-QuantizeLinear) tensor, matched to one LSB as ``_qdq_case``
    does.
    """
    input_shape = [1, 3, 64, 64]
    output_shape = [1, 8, 64, 64]
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )
    int8_output = helper.make_tensor_value_info(
        "act_q2", TensorProto.INT8, output_shape
    )

    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "act_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "act_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "act_scale", "act_zero_point"],
            ["input_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "act_scale", "act_zero_point"],
            ["act_0"],
        ),
    ]
    rng = np.random.default_rng(11)
    channels = ((4, 3), (4, 4), (8, 4))
    previous = "act_0"
    for index, (out_channels, in_channels) in enumerate(channels):
        weight_name = f"w{index}"
        weight_scale = f"w_scale{index}"
        weight_zero = f"w_zero{index}"
        initializers.extend(
            [
                numpy_helper.from_array(
                    rng.normal(
                        0.0, 0.05, size=(out_channels, in_channels, 3, 3)
                    ).astype(np.float32),
                    weight_name,
                ),
                numpy_helper.from_array(
                    np.array([0.02], dtype=np.float32), weight_scale
                ),
                numpy_helper.from_array(np.array([0], dtype=np.int8), weight_zero),
            ]
        )
        activation = "output" if index == len(channels) - 1 else f"act_{index + 1}"
        int8_activation = f"act_q{index}"
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [weight_name, weight_scale, weight_zero],
                    [f"w_q{index}"],
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [f"w_q{index}", weight_scale, weight_zero],
                    [f"w_dq{index}"],
                ),
                helper.make_node(
                    "Conv",
                    [previous, f"w_dq{index}"],
                    [f"conv{index}"],
                    pads=[1, 1, 1, 1],
                    kernel_shape=[3, 3],
                ),
                helper.make_node(
                    "QuantizeLinear",
                    [f"conv{index}", "act_scale", "act_zero_point"],
                    [int8_activation],
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [int8_activation, "act_scale", "act_zero_point"],
                    [activation],
                ),
            ]
        )
        previous = activation

    compile_model = _model(
        "qdq_conv_chain", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    del reference_model.graph.output[:]
    reference_model.graph.output.extend([int8_output])
    onnx.checker.check_model(reference_model)
    return ContractCase(
        "int8_linebuffer_conv_chain",
        compile_model,
        reference_model,
        {
            "input": np.linspace(
                -1.0, 1.0, 3 * 64 * 64, dtype=np.float32
            ).reshape(1, 3, 64, 64)
        },
        ("Conv", "Conv", "Conv"),
        mem_budget="8K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
        recompute_metric=True,
    )


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
    4x5 core tile. 66 is not divisible by 4 or 5, so the last row, the last
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


def _qdq_2d_tiled_conv_case() -> ContractCase:
    """The int8 sibling of _2d_tiled_conv_case, built via the _qdq_case QDQ
    pattern: input NCHW [1, 256, 66, 66], a 3x3 stride-1 pad-1 Conv, 24K
    budget. 256 int8 channels give the same per-pixel byte footprint as the
    float case's 64 float32 channels, so the compiler solves the same 4x5
    2D core tile with the same partial last row, column, and corner.
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
    int8_output = helper.make_tensor_value_info(
        "output_q", TensorProto.INT8, [1, c, out_h, out_w]
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
    del reference_model.graph.output[:]
    reference_model.graph.output.extend([int8_output])
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
    packed (non-full-width) tile, closing the Task 7 coverage gap.
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


def _to_runtime_layout(value: Array) -> Array:
    if value.ndim == 4:
        return np.ascontiguousarray(value.transpose(0, 2, 3, 1))
    if value.ndim == 3:
        return np.ascontiguousarray(value.transpose(0, 2, 1))
    return np.ascontiguousarray(value)


def _from_runtime_layout(value: Array, original_ndim: int) -> Array:
    if original_ndim == 4:
        return np.ascontiguousarray(value.transpose(0, 3, 1, 2))
    if original_ndim == 3:
        return np.ascontiguousarray(value.transpose(0, 2, 1))
    return value


def _compile_plan(
    model_path: Path,
    plan_path: Path,
    *,
    mem_budget: str,
    compression: str | None,
    xip: bool,
    force_one_op_stages: bool = False,
) -> dict:
    graph, _ = _run_pipeline(str(model_path), (mem_budget,))
    if force_one_op_stages:
        stages: list[Stage] = []
        for index, op in enumerate(graph.ops):
            op.stage = index
            inputs = [
                name
                for name in op.inputs
                if name in graph.tensors and not graph.tensors[name].is_constant
            ]
            outputs = [
                name
                for name in op.outputs
                if name in graph.tensors and not graph.tensors[name].is_constant
            ]
            peak = sum(
                (graph.tensors[name].size_bytes + 31) & ~31
                for name in dict.fromkeys(inputs + outputs)
            )
            stages.append(
                Stage(
                    stage_id=index,
                    op_indices=[index],
                    input_tensors=inputs,
                    output_tensors=outputs,
                    peak_bytes=peak,
                )
            )
        graph.stages = stages
    validation = validate_memory_plan(graph)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise AssertionError(f"compiler produced an infeasible graph: {details}")
    emit_binary(graph, plan_path, compress=compression, xip=xip)
    plan_bytes = plan_path.read_bytes()
    plan = read_binary_plan(plan_bytes)
    plan["_compiler_scheduled_peak"] = validation.scheduled_peak_bytes
    plan["_plan_bytes"] = plan_bytes
    return plan


def _quantize_input(value: Array, plan: dict, tensor: dict) -> Array:
    quant_index = tensor["quant_param_idx"]
    if quant_index == 0xFFFF:
        raise AssertionError(f"int8 input {tensor['name']!r} has no quant params")
    quant = plan["quant_params"][quant_index]
    scale = float(quant["scale"])
    if scale <= 0:
        raise AssertionError(f"int8 input {tensor['name']!r} has invalid scale")
    quantized = np.rint(value / scale) + int(quant["zero_point"])
    return np.clip(quantized, -128, 127).astype(np.int8)


def _pack_inputs(plan: dict, inputs: dict[str, Array]) -> bytes:
    chunks: list[bytes] = []
    for tensor_index in plan["model_inputs"]:
        tensor = plan["tensors"][tensor_index]
        source = inputs[tensor["name"]]
        if tensor["dtype"] == TensorProto.FLOAT:
            encoded = source.astype(np.float32, copy=False)
        elif tensor["dtype"] == TensorProto.INT8:
            encoded = _quantize_input(source, plan, tensor)
        else:
            raise AssertionError(
                f"unsupported contract input dtype {tensor['dtype']}"
            )
        encoded = _to_runtime_layout(encoded)
        if encoded.nbytes != tensor["size_bytes"]:
            raise AssertionError(
                f"input {tensor['name']!r} has {encoded.nbytes} bytes, "
                f"plan expects {tensor['size_bytes']}"
            )
        chunks.append(encoded.tobytes())
    return b"".join(chunks)


def _decode_outputs(
    plan: dict, raw: bytes, reference_outputs: list[Array]
) -> list[Array]:
    decoded: list[Array] = []
    offset = 0
    if len(plan["model_outputs"]) != len(reference_outputs):
        raise AssertionError("plan and ONNX Runtime output counts differ")
    terminal_transpose_outputs = {
        plan["ops"][attr["op_index"]]["outputs"][0]
        for attr in plan["op_attributes"]
        if attr["type"] == 1
        and plan["ops"][attr["op_index"]]["op_type"] == 29
        and len(plan["ops"][attr["op_index"]]["outputs"]) == 1
    }
    for tensor_index, reference in zip(plan["model_outputs"], reference_outputs):
        tensor = plan["tensors"][tensor_index]
        dtype = _DTYPE_BY_ONNX_CODE.get(tensor["dtype"])
        if dtype is None:
            raise AssertionError(
                f"unsupported contract output dtype {tensor['dtype']}"
            )
        end = offset + tensor["size_bytes"]
        if end > len(raw):
            raise AssertionError("runtime output file is truncated")
        value = np.frombuffer(raw[offset:end], dtype=dtype).copy()
        value = value.reshape(tensor["shape"])
        if tensor_index not in terminal_transpose_outputs:
            value = _from_runtime_layout(value, reference.ndim)
        decoded.append(value)
        offset = end
    if offset != len(raw):
        raise AssertionError("runtime output file has trailing bytes")
    return decoded


def _run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


_MEMORY_REPORT = re.compile(
    r"^TIGRIS_CONTRACT_MEMORY budget=(\d+) activation_limit=(\d+) "
    r"reserve=(\d+) required=(\d+) allocated=(\d+) peak=(\d+)$",
    re.MULTILINE,
)


def _assert_memory_contract(
    case: ContractCase, plan: dict, runtime_stdout: str
) -> None:
    match = _MEMORY_REPORT.search(runtime_stdout)
    if match is None:
        raise AssertionError(f"{case.name}: runtime emitted no memory report")

    budget, activation_limit, reserve, required, allocated, measured_peak = map(
        int, match.groups()
    )
    scheduled_peak = int(plan["_compiler_scheduled_peak"])
    if budget != plan["budget"]:
        raise AssertionError(
            f"{case.name}: runtime budget {budget} != plan budget {plan['budget']}"
        )
    if required != budget + reserve:
        raise AssertionError(
            f"{case.name}: required arena {required} != budget + reserve "
            f"({budget} + {reserve})"
        )
    if activation_limit != scheduled_peak:
        raise AssertionError(
            f"{case.name}: runner activation limit {activation_limit} != "
            f"compiler scheduled peak {scheduled_peak}"
        )
    if scheduled_peak > budget:
        raise AssertionError(
            f"{case.name}: compiler scheduled peak {scheduled_peak} exceeds "
            f"activation budget {budget}"
        )
    if allocated != scheduled_peak + reserve:
        raise AssertionError(
            f"{case.name}: allocated arena {allocated} != compiler core "
            f"estimate {scheduled_peak + reserve}"
        )
    if measured_peak > allocated:
        raise AssertionError(
            f"{case.name}: runtime peak {measured_peak} exceeds compiler "
            f"core estimate {allocated} "
            f"({scheduled_peak} activations + {reserve} reserve)"
        )


def _build_runner(runtime: Path, build_dir: Path) -> tuple[Path, Path]:
    """Build the default and rows-instrumented contract runners.

    The rows-instrumented runner compiles the runtime sources with
    TIGRIS_COUNT_KERNEL_ROWS so the gate can read kernel output-row counts. The
    default runner and the shipping library stay free of that test-only counter.
    """
    _run(
        [
            "cmake",
            "-S",
            str(runtime),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "runtime configure",
    )
    _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "tigris_contract_runner",
            "tigris_contract_runner_rows",
            "--parallel",
        ],
        "runtime contract-runner build",
    )
    return (
        build_dir / "tigris_contract_runner",
        build_dir / "tigris_contract_runner_rows",
    )


_ROWS_REPORT = re.compile(
    r"^TIGRIS_CONTRACT_ROWS kernel_rows=(\d+) chain_tiles=(\d+) "
    r"chain_tile_h=(\d+) interior_clamps=(\d+)$",
    re.MULTILINE,
)


def _parse_rows(case: ContractCase, runtime_stdout: str) -> dict[str, int]:
    match = _ROWS_REPORT.search(runtime_stdout)
    if match is None:
        raise AssertionError(f"{case.name}: rows runner emitted no row report")
    keys = ("kernel_rows", "chain_tiles", "chain_tile_h", "interior_clamps")
    return {key: int(value) for key, value in zip(keys, match.groups())}


def _run_metric_case(
    case: ContractCase, rows_runner: Path, work_dir: Path
) -> None:
    """Prove the line-buffered roll matches ORT and computes fewer kernel rows.

    Compiles the (asserted line-buffered) plan once, then executes it through
    the rows-instrumented runner twice: normally (roll on) and with
    ``--no-linebuffer`` (the recompute path). Both runs must match ONNX Runtime,
    and the roll must reduce total kernel output-rows, approaching the unique
    output-row count. The reduction ratio is the headline metric.
    """
    case_dir = work_dir / case.name
    case_dir.mkdir()
    compile_path = case_dir / "compile.onnx"
    reference_path = case_dir / "reference.onnx"
    plan_path = case_dir / "model.tgrs"
    inputs_path = case_dir / "inputs.bin"
    onnx.save(case.compile_model, compile_path)
    onnx.save(case.reference_model, reference_path)

    plan = _compile_plan(
        compile_path,
        plan_path,
        mem_budget=case.mem_budget,
        compression=case.compression,
        xip=case.xip,
    )
    _assert_plan_mode(case, plan)
    session = ort.InferenceSession(
        str(reference_path), providers=["CPUExecutionProvider"]
    )
    reference_outputs = session.run(None, case.inputs)
    inputs_path.write_bytes(_pack_inputs(plan, case.inputs))
    limit = str(plan["_compiler_scheduled_peak"])

    def _execute(extra_args: list[str], label: str) -> dict[str, int]:
        outputs_path = case_dir / f"outputs_{label}.bin"
        completed = _run(
            [str(rows_runner), str(plan_path), str(inputs_path), str(outputs_path), limit]
            + extra_args,
            f"{case.name} rows-runner ({label})",
        )
        if label == "linebuffered":
            _assert_memory_contract(case, plan, completed.stdout)
        actual_outputs = _decode_outputs(
            plan, outputs_path.read_bytes(), reference_outputs
        )
        _assert_output_parity(actual_outputs, reference_outputs)
        return _parse_rows(case, completed.stdout)

    rows_on = _execute([], "linebuffered")
    rows_off = _execute(["--no-linebuffer"], "recompute")

    # Model output height (ONNX NCHW reference); the runtime tiles the height.
    out_h = int(reference_outputs[0].shape[2])
    tiles = rows_on["chain_tiles"]
    tile_h = rows_on["chain_tile_h"]
    if tiles < 3:
        raise AssertionError(
            f"{case.name}: chain produced {tiles} tiles, need >= 3 so tile 0, "
            f"an interior tile, and a last tile all run"
        )
    if tile_h <= 0 or out_h % tile_h == 0:
        raise AssertionError(
            f"{case.name}: output height {out_h} is not partial against tile "
            f"height {tile_h}; the last tile must be partial"
        )
    if rows_off["chain_tiles"] != tiles or rows_off["chain_tile_h"] != tile_h:
        raise AssertionError(
            f"{case.name}: recompute run tiled differently "
            f"({rows_off['chain_tiles']}x{rows_off['chain_tile_h']}) than the "
            f"line-buffered run ({tiles}x{tile_h})"
        )
    if not (rows_on["kernel_rows"] > 0 and rows_off["kernel_rows"] > 0):
        raise AssertionError(f"{case.name}: row counter recorded no work")
    if rows_on["kernel_rows"] >= rows_off["kernel_rows"]:
        raise AssertionError(
            f"{case.name}: line-buffered kernel rows "
            f"{rows_on['kernel_rows']} did not drop below recompute "
            f"{rows_off['kernel_rows']}"
        )

    unique_rows = len(case.expected_operators) * out_h
    ratio = rows_off["kernel_rows"] / rows_on["kernel_rows"]
    print(
        f"PASS {case.name} recompute-reduction "
        f"rows_on={rows_on['kernel_rows']} rows_off={rows_off['kernel_rows']} "
        f"ratio={ratio:.3f}x unique_rows={unique_rows} "
        f"tiles={tiles} tile_h={tile_h} last_tile_h={out_h - (tiles - 1) * tile_h}"
    )


def _run_case(
    case: ContractCase, runner: Path, work_dir: Path
) -> Path:
    case_dir = work_dir / case.name
    case_dir.mkdir()
    compile_path = case_dir / "compile.onnx"
    reference_path = case_dir / "reference.onnx"
    plan_path = case_dir / "model.tgrs"
    inputs_path = case_dir / "inputs.bin"
    outputs_path = case_dir / "outputs.bin"
    onnx.save(case.compile_model, compile_path)
    onnx.save(case.reference_model, reference_path)

    plan = _compile_plan(
        compile_path,
        plan_path,
        mem_budget=case.mem_budget,
        compression=case.compression,
        xip=case.xip,
        force_one_op_stages=case.force_one_op_stages,
    )
    _assert_plan_mode(case, plan)
    session = ort.InferenceSession(
        str(reference_path), providers=["CPUExecutionProvider"]
    )
    reference_outputs = session.run(None, case.inputs)
    inputs_path.write_bytes(_pack_inputs(plan, case.inputs))
    completed = _run(
        [
            str(runner),
            str(plan_path),
            str(inputs_path),
            str(outputs_path),
            str(plan["_compiler_scheduled_peak"]),
        ],
        f"{case.name} runtime execution",
    )
    _assert_memory_contract(case, plan, completed.stdout)
    actual_outputs = _decode_outputs(
        plan, outputs_path.read_bytes(), reference_outputs
    )
    _assert_output_parity(actual_outputs, reference_outputs)
    print(f"PASS {case.name} memory-contract")
    return plan_path


def _assert_output_parity(
    actual_outputs: list[Array], reference_outputs: list[Array]
) -> None:
    for actual, expected in zip(actual_outputs, reference_outputs):
        if np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        else:
            # The runtime uses integer half-away-from-zero quantization while
            # this ONNX Runtime QDQ reference follows a different half-tie
            # rule. Keep the same one-LSB acceptance bound used by benchmark
            # validation, while rejecting any larger contract drift.
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=_INT8_LSB_TOLERANCE
            )


def _assert_plan_mode(case: ContractCase, plan: dict) -> None:
    """Ensure each corpus case actually exercises the intended plan mode."""
    actual_operators = tuple(
        OP_TYPE_BY_CODE[op["op_type"]] for op in plan["ops"]
    )
    if Counter(actual_operators) != Counter(case.expected_operators):
        raise AssertionError(
            f"{case.name}: plan operators {actual_operators}, expected "
            f"{case.expected_operators}"
        )

    standalone_tiled = any(
        tile["tileable"] and tile["num_tiles"] > 1
        for tile in plan["tile_plans"]
    )
    chained = any(stage["chain_len"] >= 2 for stage in plan["stages"])
    # Streamable chains encode their tile height on the chain head rather than
    # creating standalone tile-plan records.
    tiled = standalone_tiled or chained
    if tiled != case.expect_tiled:
        raise AssertionError(f"{case.name}: tiled={tiled}, expected {case.expect_tiled}")
    if chained != case.expect_chain:
        raise AssertionError(f"{case.name}: chained={chained}, expected {case.expect_chain}")

    # Decode the head-stage line-buffer flag straight from the emitted plan
    # bytes (a recomputing chain marks only its head, where chain_id equals the
    # stage's own index). This makes "is actually line-buffered" a checked
    # precondition of the differential rather than an assumption.
    plan_bytes = plan["_plan_bytes"]
    line_buffered = any(
        _stage_reserved1(plan_bytes, index) & STAGE_FLAG_LINE_BUFFERED
        for index, stage in enumerate(plan["stages"])
        if stage["chain_len"] >= 2 and stage["chain_id"] == index
    )
    if line_buffered != case.expect_line_buffered:
        raise AssertionError(
            f"{case.name}: line_buffered={line_buffered}, expected "
            f"{case.expect_line_buffered}"
        )

    # A 2D case decodes the plan's first tile-plan record straight from the
    # emitted bytes (reusing test_2d_tiling_plan.py's decoder) and confirms
    # both axes actually split into more than one tile. num_tiles is the
    # solver's ceil(H/tile_h) * ceil(W/tile_w) product, so dividing it by the
    # H-axis tile count derived from the decoded original_height/tile_height
    # recovers the W-axis tile count without needing a separate width field
    # in the plan format.
    if case.expect_2d:
        tile_plan = decode_first_tile_plan(plan_bytes)
        if tile_plan.axis != TILE_AXIS_HW:
            raise AssertionError(
                f"{case.name}: tile plan axis {tile_plan.axis}, expected "
                f"TILE_AXIS_HW ({TILE_AXIS_HW})"
            )
        tiles_h = math.ceil(tile_plan.original_height / tile_plan.tile_height)
        tiles_w = tile_plan.num_tiles // tiles_h
        if not (tiles_h > 1 and tiles_w > 1):
            raise AssertionError(
                f"{case.name}: expected multi-tile on both axes, got "
                f"tiles_h={tiles_h} tiles_w={tiles_w} "
                f"(num_tiles={tile_plan.num_tiles}, "
                f"tile_h={tile_plan.tile_height}, tile_w={tile_plan.tile_width})"
            )

    if case.compression == "lz4":
        if plan["weight_blocks_compression"] != COMPRESS_LZ4:
            raise AssertionError(f"{case.name}: expected LZ4 weight blocks")
    elif plan["weight_blocks"]:
        raise AssertionError(f"{case.name}: unexpected compressed weight blocks")
    has_xip = bool(plan["flags"] & FLAG_XIP)
    if has_xip != case.xip:
        raise AssertionError(f"{case.name}: XIP flag={has_xip}, expected {case.xip}")


def _assert_compile_rejected(work_dir: Path) -> None:
    resize_scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"
    )
    cases = [
        _model(
            "unsupported_sin",
            [helper.make_node("Sin", ["input"], ["output"])],
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        ),
        _model(
            "unsupported_softmax_axis",
            [
                helper.make_node(
                    "Softmax", ["input"], ["output"], axis=0
                )
            ],
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [2, 4]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [2, 4]
                )
            ],
        ),
        _model(
            "unsupported_concat_axis",
            [
                helper.make_node(
                    "Concat", ["left", "right"], ["output"], axis=2
                )
            ],
            [
                helper.make_tensor_value_info(
                    "left", TensorProto.FLOAT, [1, 2, 3, 4]
                ),
                helper.make_tensor_value_info(
                    "right", TensorProto.FLOAT, [1, 2, 3, 4]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, 2, 6, 4]
                )
            ],
        ),
        _model(
            "unsupported_resize_mode",
            [
                helper.make_node(
                    "Resize",
                    ["input", "", "scales"],
                    ["output"],
                    coordinate_transformation_mode="asymmetric",
                    mode="linear",
                )
            ],
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, 1, 2, 2]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, 1, 4, 4]
                )
            ],
            [resize_scales],
        ),
        _model(
            "unsupported_pool_ceil",
            [
                helper.make_node(
                    "MaxPool",
                    ["input"],
                    ["output"],
                    ceil_mode=1,
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            ],
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, 1, 3, 3]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, 1, 2, 2]
                )
            ],
        ),
    ]
    for index, model in enumerate(cases):
        path = work_dir / f"rejected-{index}.onnx"
        onnx.save(model, path)
        try:
            _compile_plan(
                path,
                work_dir / f"rejected-{index}.tgrs",
                mem_budget="4K",
                compression=None,
                xip=False,
            )
        except (ClickException, ValueError):
            continue
        raise AssertionError(f"{model.graph.name} unexpectedly compiled")
    print("PASS compiler_rejections")


def _assert_runtime_rejects_incompatible_plan(
    runner: Path, plan_path: Path, work_dir: Path
) -> None:
    data = bytearray(plan_path.read_bytes())
    data[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    incompatible = work_dir / "incompatible.tgrs"
    input_path = work_dir / "empty-input.bin"
    output_path = work_dir / "unexpected-output.bin"
    incompatible.write_bytes(data)
    input_path.write_bytes(b"")
    completed = subprocess.run(
        [str(runner), str(incompatible), str(input_path), str(output_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError("runtime accepted an incompatible plan version")
    if output_path.exists():
        raise AssertionError("runtime wrote output for an incompatible plan")
    print("PASS incompatible_plan_rejection")


def _run_gate(runtime: Path, work_dir: Path) -> None:
    cases = [
        _constant_add_case(),
        _residual_case(),
        _output_transpose_case(),
        _dilated_conv_case(),
        _depthwise_conv_case(),
        _math_normalization_case(),
        _conv1d_case(),
        _rank3_pointwise_case(),
        _many_stage_case(),
        _tcn_16k_case(),
        _reduce_mean_case(),
        _normalized_classifier_case(),
        _resize_concat_case(),
        _tiled_pool_case(),
        _tiled_pool_chain_case(),
        _tiled_chain_case(compression="lz4"),
        _tiled_chain_case(xip=True),
        _qdq_case("Conv"),
        _qdq_case("AveragePool"),
        _linebuffer_conv_chain_case(),
        _qdq_conv_chain_case(),
        _2d_tiled_conv_case(),
        _qdq_2d_tiled_conv_case(),
        _2d_tiled_conv_sigmoid_case(),
    ]
    covered_operators = {
        operator for case in cases for operator in case.expected_operators
    }
    supported_operators = KERNEL_CAPABILITIES["reference"].native_operators
    if covered_operators != supported_operators:
        raise AssertionError(
            "reference corpus coverage drift: "
            f"missing={sorted(supported_operators - covered_operators)}, "
            f"unexpected={sorted(covered_operators - supported_operators)}"
        )
    print(
        f"PASS reference_operator_coverage "
        f"{len(covered_operators)}/{len(supported_operators)}"
    )

    runner, rows_runner = _build_runner(runtime, work_dir / "runtime-build")
    first_plan: Path | None = None
    for case in cases:
        if case.recompute_metric:
            _run_metric_case(case, rows_runner, work_dir)
        else:
            plan_path = _run_case(case, runner, work_dir)
            if first_plan is None:
                first_plan = plan_path
    assert first_plan is not None, "gate needs at least one non-metric case"
    _assert_compile_rejected(work_dir)
    _assert_runtime_rejects_incompatible_plan(runner, first_plan, work_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("../tigris-runtime"),
        help="Path to the sibling tigris-runtime checkout",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Retain generated ONNX, plan, input, and output files in this empty directory",
    )
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if not (runtime / "CMakeLists.txt").is_file():
        raise SystemExit(f"runtime checkout not found: {runtime}")

    if args.artifacts:
        work_dir = args.artifacts.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if any(work_dir.iterdir()):
            raise SystemExit(f"artifact directory must be empty: {work_dir}")
        _run_gate(runtime, work_dir)
        print(f"Retained contract artifacts in {work_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="tigris-crossrepo-") as temp:
            _run_gate(runtime, Path(temp))
    print("Cross-repository compiler/runtime contract gate passed.")


if __name__ == "__main__":
    main()
