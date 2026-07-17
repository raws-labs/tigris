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
import re
import subprocess
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

from tigris.analysis.validation import validate_memory_plan
from tigris.capabilities import KERNEL_CAPABILITIES, OP_TYPE_BY_CODE
from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import COMPRESS_LZ4, FLAG_XIP
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary


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
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 8]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8]
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
                -1.5, 1.5, 16, dtype=np.float32
            ).reshape(1, 2, 8)
        },
        ("Conv1D",),
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
) -> dict:
    graph, _ = _run_pipeline(str(model_path), (mem_budget,))
    validation = validate_memory_plan(graph)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise AssertionError(f"compiler produced an infeasible graph: {details}")
    emit_binary(graph, plan_path, compress=compression, xip=xip)
    plan = read_binary_plan(plan_path.read_bytes())
    plan["_compiler_scheduled_peak"] = validation.scheduled_peak_bytes
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


def _build_runner(runtime: Path, build_dir: Path) -> Path:
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
            "--parallel",
        ],
        "runtime contract-runner build",
    )
    return build_dir / "tigris_contract_runner"


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
    print(f"PASS {case.name} memory-contract")
    return plan_path


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
        _reduce_mean_case(),
        _normalized_classifier_case(),
        _resize_concat_case(),
        _tiled_pool_case(),
        _tiled_chain_case(compression="lz4"),
        _tiled_chain_case(xip=True),
        _qdq_case("Conv"),
        _qdq_case("AveragePool"),
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

    runner = _build_runner(runtime, work_dir / "runtime-build")
    first_plan = _run_case(cases[0], runner, work_dir)
    for case in cases[1:]:
        _run_case(case, runner, work_dir)
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
