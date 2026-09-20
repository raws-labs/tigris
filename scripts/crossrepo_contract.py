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
    opset: int = 13,
) -> onnx.ModelProto:
    """Build a checked model. `opset` rises only for an operator that needs it."""
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 8 if opset < 17 else 9
    onnx.checker.check_model(model)
    return model


def _constant_add_case() -> ContractCase:
    """A constant-operand Add, with the Relu ahead of it so it stays its own op.

    An activation that follows a fusable producer is absorbed into that
    producer, so leading with the Relu is what keeps a standalone Relu in the
    reference corpus. ``_add_relu_fusion_case`` covers the absorbed form.
    """
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
            helper.make_node("Relu", ["input"], ["gated"]),
            helper.make_node("Add", ["gated", "constant"], ["output"]),
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
        ("Relu", "Add"),
    )


def _add_relu_fusion_case() -> ContractCase:
    """A float Add whose trailing Relu is absorbed into the Add itself.

    The plan carries one operator, so the runtime's elementwise kernel has to
    apply the fused activation. The constant shifts half the lanes negative,
    which is what makes the clamp observable in the output.
    """
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
        "add_relu_fusion",
        [
            helper.make_node("Add", ["input", "constant"], ["shifted"]),
            helper.make_node("Relu", ["shifted"], ["output"]),
        ],
        [model_input],
        [model_output],
        [constant],
    )
    return ContractCase(
        "float_add_relu_fusion",
        model,
        model,
        {
            "input": np.array(
                [[0.25, -1.0, 5.0, -4.0]], dtype=np.float32
            )
        },
        ("Add",),
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


def _reshape_alias_case(*, spatial: bool) -> ContractCase:
    """A reshape that moves no byte, so the two tensors share a buffer.

    The linear case regroups a matrix's shape; the spatial one regroups a
    feature map's spatial axes into patches, which is what a vision
    transformer's unfold is. Both leave the stored bytes untouched, so the
    executor gives the output the input's buffer and the compiler counts one
    allocation. A wrong answer here means the two do not agree on when that is
    safe.
    """
    if spatial:
        channels, side = 8, 4
        nodes = [
            helper.make_node(
                "Conv", ["input", "w"], ["features"], kernel_shape=[1, 1]),
            helper.make_node("Reshape", ["features", "shape"], ["patches"]),
            helper.make_node("Relu", ["patches"], ["output"]),
        ]
        initializers = [
            numpy_helper.from_array(
                (np.random.default_rng(0).normal(
                    size=(channels, channels, 1, 1)) * 0.3
                 ).astype(np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, channels, side * side, 1], np.int64), "shape"),
        ]
        in_shape = [1, channels, side, side]
        out_shape = [1, channels, side * side, 1]
    else:
        tokens, width = 8, 8
        nodes = [
            helper.make_node("MatMul", ["input", "w"], ["projected"]),
            helper.make_node("Reshape", ["projected", "shape"], ["heads"]),
            helper.make_node("Erf", ["heads"], ["output"]),
        ]
        initializers = [
            numpy_helper.from_array(
                (np.random.default_rng(0).normal(size=(width, width)) * 0.3
                 ).astype(np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, tokens, 2, width // 2], np.int64), "shape"),
        ]
        in_shape = [1, tokens, width]
        out_shape = [1, tokens, 2, width // 2]
    kind = "spatial" if spatial else "linear"
    model = _model(
        f"reshape_alias_{kind}",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    data = np.linspace(
        -1.5, 1.5, int(np.prod(in_shape)), dtype=np.float32
    ).reshape(in_shape)
    ops = (("Conv", "Reshape", "Relu") if spatial
           else ("Transpose", "Reshape", "Gemm", "Reshape", "Reshape", "Erf",
                 "Transpose"))
    return ContractCase(
        f"float_reshape_alias_{kind}",
        model,
        model,
        {"input": data},
        ops,
    )


def _token_bias_case() -> ContractCase:
    """A bias added onto a batched matrix product over a token sequence.

    An exporter that does not fuse the bias writes it as a rank-1 constant Add
    on the product, in either operand order. The Add kernel takes two operands
    of one shape, so the graph is refused for a broadcast it cannot do, while
    the product it follows is lowered to the fully-connected kernel, which
    reads a bias per output feature. Which axis that bias addresses is a
    property of the producer: a convolution biases the channel the model
    states second, a matrix product the last one.
    """
    tokens, width, hidden = 12, 16, 24
    rng = np.random.default_rng(7)
    first = (rng.normal(size=(width, hidden)) * 0.3).astype(np.float32)
    first_bias = (rng.normal(size=(hidden,)) * 0.5).astype(np.float32)
    second = (rng.normal(size=(hidden, width)) * 0.3).astype(np.float32)
    second_bias = (rng.normal(size=(width,)) * 0.5).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["input", "w1"], ["p1"]),
        helper.make_node("Add", ["b1", "p1"], ["h"]),
        helper.make_node("Erf", ["h"], ["he"]),
        helper.make_node("MatMul", ["he", "w2"], ["p2"]),
        helper.make_node("Add", ["p2", "b2"], ["output"]),
    ]
    model = _model(
        "token_bias",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array(first, "w1"),
         numpy_helper.from_array(first_bias, "b1"),
         numpy_helper.from_array(second, "w2"),
         numpy_helper.from_array(second_bias, "b2")],
    )
    data = np.linspace(
        -1.5, 1.5, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        "float_token_bias",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Erf", "Reshape", "Gemm",
         "Reshape", "Transpose"),
    )


def _flattened_token_head_case() -> ContractCase:
    """A sequence flattened whole and fed to a matrix product.

    A classifier or forecast head flattens every token into one vector. When
    the tensor is one the runtime stores channels-last, the flat order differs
    from the model's and the weight's columns are permuted to compensate. A
    tensor that states its own axis order is already stored the way the model
    states it, so permuting it there is a silent wrong answer: the rank is the
    same, only the layout differs.
    """
    tokens, width, outputs = 9, 16, 5
    rng = np.random.default_rng(13)
    project = (rng.normal(size=(width, width)) * 0.3).astype(np.float32)
    head = (rng.normal(size=(outputs, tokens * width)) * 0.05).astype(np.float32)
    bias = (rng.normal(size=(outputs,)) * 0.1).astype(np.float32)
    nodes = [
        # The product is what makes the sequence state its own axis order.
        helper.make_node("MatMul", ["input", "w"], ["tokens"]),
        helper.make_node("Reshape", ["tokens", "flat"], ["vector"]),
        helper.make_node(
            "Gemm", ["vector", "head", "bias"], ["output"],
            alpha=1.0, beta=1.0, transB=1),
    ]
    model = _model(
        "flattened_token_head",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, outputs])],
        [numpy_helper.from_array(project, "w"),
         numpy_helper.from_array(
             np.array([1, tokens * width], np.int64), "flat"),
         numpy_helper.from_array(head, "head"),
         numpy_helper.from_array(bias, "bias")],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        "float_flattened_token_head",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Reshape", "Gemm"),
    )


def _constant_divisor_case() -> ContractCase:
    """A division by a constant, which is how an exporter writes a GELU.

    ``gelu`` leaves ``Erf(x / sqrt(2))`` behind and an attention scale leaves a
    division by the head width. Div has no opcode, but a constant divisor is
    the Mul the kernels already carry, so the graph is rewritten rather than
    refused. Both shapes a Mul takes are here: a scalar and a whole tensor.
    """
    channels, side = 4, 8
    shape = [1, channels, side, side]
    nodes = [
        helper.make_node("Div", ["input", "root_two"], ["scaled"]),
        helper.make_node("Erf", ["scaled"], ["shaped"]),
        helper.make_node("Div", ["shaped", "whole"], ["output"]),
    ]
    model = _model(
        "constant_divisor",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(
            np.array(np.sqrt(2.0), dtype=np.float32), "root_two"),
         numpy_helper.from_array(
             np.linspace(0.5, 2.0, int(np.prod(shape)), dtype=np.float32
                         ).reshape(shape), "whole")],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_constant_divisor",
        model,
        model,
        {"input": data},
        ("Mul", "Erf", "Mul"),
    )


def _traced_shape_scale_case() -> ContractCase:
    """A scale the exporter traced out of the tensor's own shape.

    An exporter that traces a model rather than folding it writes a scale
    stated in code as ``1 / sqrt(x.shape[-1])`` as a Shape of the tensor, a
    Slice of that shape, a Cast, a Sqrt and a division, then multiplies by the
    result. Every value in the chain is settled by extents the plan already
    states, so none of it survives into the plan, and the multiplication that
    reads it is a scale rather than a broadcast against a rank-1 operand.
    """
    tokens, width = 6, 8
    shape = [1, tokens, width]
    nodes = [
        helper.make_node("Shape", ["input"], ["dims"]),
        helper.make_node("Slice", ["dims", "last", "stop", "axis"], ["width"]),
        helper.make_node("Cast", ["width"], ["as_float"], to=TensorProto.FLOAT),
        helper.make_node("Sqrt", ["as_float"], ["root"]),
        helper.make_node("Div", ["one", "root"], ["scale"]),
        helper.make_node("Mul", ["input", "scale"], ["scaled"]),
        helper.make_node("Erf", ["scaled"], ["output"]),
    ]
    model = _model(
        "traced_shape_scale",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(np.array([-1], np.int64), "last"),
         numpy_helper.from_array(
             np.array([np.iinfo(np.int64).max], np.int64), "stop"),
         numpy_helper.from_array(np.array([0], np.int64), "axis"),
         numpy_helper.from_array(np.array(1.0, np.float32), "one")],
    )
    data = np.linspace(
        -2.0, 2.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_traced_shape_scale",
        model,
        model,
        {"input": data},
        ("Mul", "Erf"),
    )


def _unfolded_feature_map_case() -> ContractCase:
    """A feature map regrouped into patches, which is an unfold.

    A reshape renames axes over one sequence and the kernel copies straight
    through, which states the model's regrouping only when the runtime holds
    the elements in that order. This one folds the channel axis into the
    leading one, which a tensor held channels-last does not, so the plan used
    to compile and come back wrong. Holding the operand in the model's own
    order first is what makes it expressible.
    """
    channels, side, patch = 6, 8, 2
    cells = side // patch
    nodes = [
        helper.make_node(
            "Conv", ["input", "w"], ["features"], kernel_shape=[1, 1]),
        # [1, C, 8, 8] -> [C * cells, patch, cells, patch]: the channel axis
        # joins the leading one, which the stored order does not hold next to
        # it.
        helper.make_node("Reshape", ["features", "patches"], ["grouped"]),
        helper.make_node("Erf", ["grouped"], ["output"]),
    ]
    model = _model(
        "unfolded_feature_map",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT,
            [channels * cells, patch, cells, patch])],
        [numpy_helper.from_array(
            (np.random.default_rng(3).normal(size=(channels, channels, 1, 1))
             * 0.4).astype(np.float32), "w"),
         numpy_helper.from_array(
             np.array([channels * cells, patch, cells, patch], np.int64),
             "patches")],
    )
    data = np.linspace(
        -1.0, 1.0, channels * side * side, dtype=np.float32
    ).reshape(1, channels, side, side)
    return ContractCase(
        "float_unfolded_feature_map",
        model,
        model,
        {"input": data},
        ("Conv", "Transpose", "Reshape", "Erf", "Transpose"),
    )


def _split_case() -> ContractCase:
    """A tensor cut into contiguous parts, which is how a fused qkv unbinds.

    One projection produces query, key and value together and the model cuts
    them apart along the axis they were stacked on. The parts are runs of the
    input, so each is that many bytes taken in order; a cut anywhere else
    interleaves and is refused rather than copied wrongly.
    """
    parts, tokens, width = 3, 5, 8
    nodes = [
        helper.make_node(
            "Split", ["input", "parts"], ["first", "second", "third"], axis=0),
        helper.make_node("Add", ["first", "second"], ["pair"]),
        helper.make_node("Mul", ["pair", "third"], ["output"]),
    ]
    model = _model(
        "split_parts",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [parts, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array(np.array([1, 1, 1], np.int64), "parts")],
    )
    data = np.linspace(
        -2.0, 2.0, parts * tokens * width, dtype=np.float32
    ).reshape(parts, tokens, width)
    return ContractCase(
        "float_split_parts",
        model,
        model,
        {"input": data},
        ("Split", "Add", "Mul"),
    )


def _reduce_mean_case(*, quantized: bool, keepdims: bool) -> ContractCase:
    """A mean over the token axis, which is what a pooled sequence head is.

    The encoder leaves one vector per token and the head wants one vector for
    the sequence, so the mean collapses the rows of a rank-3 tensor. Both the
    kept and the dropped row axis are here because the two write the same
    bytes and only the declared rank differs, which is exactly the kind of
    difference the loader has to police rather than guess at.
    """
    tokens, width = 12, 8
    in_shape = [1, tokens, width]
    out_shape = [1, 1, width] if keepdims else [1, width]
    kd = 1 if keepdims else 0
    name = f"reduce_mean_{'keepdims' if keepdims else 'flat'}"
    if quantized:
        in_scale, out_scale = 0.05, 0.03
        initializers = [
            numpy_helper.from_array(
                np.array(in_scale, dtype=np.float32), "in_s"),
            numpy_helper.from_array(np.array(2, dtype=np.int8), "in_z"),
            numpy_helper.from_array(
                np.array(out_scale, dtype=np.float32), "out_s"),
            numpy_helper.from_array(np.array(-5, dtype=np.int8), "out_z"),
        ]
        nodes = [
            helper.make_node(
                "QuantizeLinear", ["input", "in_s", "in_z"], ["iq"]),
            helper.make_node(
                "DequantizeLinear", ["iq", "in_s", "in_z"], ["idq"]),
            helper.make_node(
                "ReduceMean", ["idq"], ["raw"], axes=[1], keepdims=kd),
            helper.make_node(
                "QuantizeLinear", ["raw", "out_s", "out_z"], ["oq"]),
            helper.make_node(
                "DequantizeLinear", ["oq", "out_s", "out_z"], ["output"]),
        ]
        rng = np.random.default_rng(11)
        data = (rng.integers(-100, 100, size=in_shape).astype(np.float32)
                * in_scale)
        label = f"int8_{name}"
    else:
        initializers = []
        nodes = [
            helper.make_node(
                "ReduceMean", ["input"], ["output"], axes=[1], keepdims=kd),
        ]
        data = np.linspace(
            -1.5, 1.5, int(np.prod(in_shape)), dtype=np.float32
        ).reshape(in_shape)
        label = f"float_{name}"
    model = _model(
        label,
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    reference_model = copy.deepcopy(model)
    onnx.checker.check_model(reference_model)
    return ContractCase(
        label,
        model,
        reference_model,
        {"input": data},
        ("ReduceMean",),
    )


def _head_permutation_case() -> ContractCase:
    """The permutation that moves an attention block into its head layout.

    Stored (0, 2, 1, 3): a token by head transpose carrying the head width at
    each position. It is the first permutation the band path accepts with a
    suffix, and the three it named before were the cases with none.
    """
    tokens, heads, width = 64, 4, 16
    model = _model(
        "head_permutation",
        [helper.make_node(
            "Transpose", ["input"], ["output"], perm=[0, 2, 1, 3])],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, heads, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, heads, tokens, width])],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * heads * width, dtype=np.float32
    ).reshape(1, tokens, heads, width)
    return ContractCase(
        "float_head_permutation",
        model,
        model,
        {"input": data},
        ("Transpose",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _banded_attention_case() -> ContractCase:
    """An attention region banded over its query axis.

    The band cuts the second to last axis of a rank-4 tensor, so the head axis
    is batch that the band spans rather than cuts, and the matrix product's
    second operand is read whole the way a weight is. Sized so the budget
    forces the band: the same graph at a roomy budget runs whole and exercises
    none of it.
    """
    heads, tokens, width = 4, 64, 16
    shape = [1, heads, tokens, width]
    model = _model(
        "banded_attention",
        [
            helper.make_node(
                "Transpose", ["input"], ["keys"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["input", "keys"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["output"], axis=-1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, heads, tokens, tokens])],
    )
    data = np.linspace(
        -1.0, 1.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_banded_attention",
        model,
        model,
        {"input": data},
        ("Transpose", "Transpose", "Transpose", "MatMul", "Softmax",
         "Transpose"),
        mem_budget="64K",
        expect_tiled=True,
    )


def _layout_mixing_residual_case(*, square: bool) -> ContractCase:
    """A residual connection around a matrix product.

    The matrix product needs the model's own axis order and the skip arrives
    in storage order, so the Add's two operands disagree about layout and the
    normalizer has to unify them. Two shapes, because the failure wore two
    faces: an oblong block was refused at load on the Add shape check, and a
    square one passed that check and added transposed data. The square case is
    the one that matters, and it is the ordinary shape for a small transformer
    whose sequence length equals its width.
    """
    tokens, width = (8, 8) if square else (8, 4)
    rng = np.random.default_rng(0)
    weight = (rng.normal(size=(width, width)) * 0.3).astype(np.float32)
    model = _model(
        f"layout_mixing_residual_{'square' if square else 'oblong'}",
        [
            helper.make_node("MatMul", ["input", "w"], ["projected"]),
            helper.make_node("Add", ["input", "projected"], ["output"]),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array(weight, "w")],
    )
    data = np.linspace(
        -1.5, 1.5, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        f"float_residual_over_matmul_{'square' if square else 'oblong'}",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Add", "Transpose"),
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


def _qdq_conv1d_case() -> ContractCase:
    """An int8 Conv1D, whose kernel requantizes its int32 accumulator.

    The scales are chosen so the effective scale (0.03125) and the raw output
    scale (0.5) differ by a factor of sixteen: an output requantized with the
    wrong one of the two is off by tens of quantization steps, not by rounding.
    """
    shape = [1, 1, 8]
    initializers = [
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "in_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "in_zero_point"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "out_zero_point"),
        numpy_helper.from_array(
            np.full((1, 1, 3), 0.5, dtype=np.float32), "weight"),
        numpy_helper.from_array(
            np.array(0.0625, dtype=np.float32), "weight_scale"),
        numpy_helper.from_array(
            np.array(0, dtype=np.int8), "weight_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "in_scale", "in_zero_point"], ["iq"]),
        helper.make_node(
            "DequantizeLinear", ["iq", "in_scale", "in_zero_point"], ["idq"]),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zero_point"], ["wq"]),
        helper.make_node(
            "DequantizeLinear",
            ["wq", "weight_scale", "weight_zero_point"], ["wdq"]),
        helper.make_node(
            "Conv", ["idq", "wdq"], ["raw"], kernel_shape=[3], pads=[1, 1]),
        helper.make_node(
            "QuantizeLinear", ["raw", "out_scale", "out_zero_point"], ["oq"]),
        helper.make_node(
            "DequantizeLinear",
            ["oq", "out_scale", "out_zero_point"], ["output"]),
    ]
    compile_model = _model(
        "qdq_conv1d",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    return ContractCase(
        "int8_conv1d",
        compile_model,
        reference_model,
        {"input": np.array(
            [[[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]]],
            dtype=np.float32)},
        ("Conv1D",),
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


def _reduce_case(*, maximum: bool) -> ContractCase:
    """A spatial ReduceMean or ReduceMax must execute as its global pool."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 1, 1]
    )
    operator = "ReduceMax" if maximum else "ReduceMean"
    pool = "GlobalMaxPool" if maximum else "GlobalAveragePool"
    model = _model(
        f"{operator.lower()}_to_pool",
        [
            helper.make_node(
                operator,
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
        f"float_{operator.lower()}_to_pool",
        model,
        model,
        {
            "input": np.arange(
                18, dtype=np.float32
            ).reshape(1, 3, 2, 3)
        },
        (pool,),
    )


def _inference_identity_case() -> ContractCase:
    """Dropout and Identity must vanish, and Squeeze must execute as Reshape.

    The reference model keeps all three so ONNX Runtime evaluates the graph an
    exporter actually emits; TiGrIS must reach the same values with them gone.
    The Squeeze here drops the two trailing unit axes of a pooled [1, C, 1, 1]
    tensor, which leaves the runtime's element order untouched.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes")
    model = _model(
        "inference_identity",
        [
            helper.make_node("Identity", ["input"], ["same"]),
            helper.make_node("Dropout", ["same"], ["kept"]),
            helper.make_node("GlobalAveragePool", ["kept"], ["pooled"]),
            helper.make_node("Squeeze", ["pooled", "axes"], ["output"]),
        ],
        [model_input],
        [model_output],
        [axes],
    )
    return ContractCase(
        "float_inference_identity",
        model,
        model,
        {
            "input": np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
        },
        ("GlobalAveragePool", "Reshape"),
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


def _float_gemm_bias_add_case() -> ContractCase:
    """A float Gemm whose bias arrives as a separate Add, the rank-2 form.

    The operand is per-channel against a [1, C] product, so the Add kernel
    cannot take it as written; the bias slot can. ONNX Runtime evaluates the
    two-op graph and TiGrIS must match it with one op.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([[0.5, -0.25, 0.75, 0.25], [-0.5, 0.25, 0.5, -0.75]],
                     dtype=np.float32),
            "weight",
        ),
        numpy_helper.from_array(
            np.array([0.5, -0.25], dtype=np.float32), "bias"),
    ]
    model = _model(
        "float_gemm_bias_add",
        [
            helper.make_node("Gemm", ["input", "weight"], ["product"], transB=1),
            helper.make_node("Add", ["product", "bias"], ["output"]),
        ],
        [model_input],
        [model_output],
        initializers,
    )
    return ContractCase(
        "float_gemm_bias_add",
        model,
        model,
        {"input": np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)},
        ("Gemm",),
    )


def _batched_matmul_case() -> ContractCase:
    """A rank-3 constant-weight MatMul, the per-position linear layer.

    ONNX Runtime evaluates the batched product directly. TiGrIS makes the
    operand linear, collapses its leading axes and runs the fully-connected
    kernel, so this checks the layout conversion and the lowering together.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 5, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 5, 3]
    )
    weight = numpy_helper.from_array(
        np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(4, 3), "weight"
    )
    model = _model(
        "batched_matmul",
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        [model_input],
        [model_output],
        [weight],
    )
    return ContractCase(
        "float_batched_matmul",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 20, dtype=np.float32
            ).reshape(1, 5, 4)
        },
        # Model boundaries keep the channels-last convention callers rely on,
        # so reaching a linear operand costs a conversion at each end. Those
        # Transposes are the layout change made explicit.
        ("Transpose", "Reshape", "Gemm", "Reshape", "Transpose"),
    )


def _dynamic_matmul_case(*, batched: bool) -> ContractCase:
    """A matrix product of two activations, which has no constant to fold.

    Neither operand is a weight, so this is the form the fully-connected kernel
    cannot express and the MatMul kernel exists for. The batched variant also
    exercises the layout conversion: rank-3 model boundaries keep the
    channels-last convention, so reaching the model's own axis order costs a
    Transpose on each operand and one on the result.
    """
    lhs_shape = [2, 3, 4] if batched else [3, 4]
    rhs_shape = [2, 4, 2] if batched else [4, 2]
    out_shape = [2, 3, 2] if batched else [3, 2]
    model = _model(
        "dynamic_matmul",
        [helper.make_node("MatMul", ["lhs", "rhs"], ["output"])],
        [
            helper.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs_shape),
            helper.make_tensor_value_info("rhs", TensorProto.FLOAT, rhs_shape),
        ],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
    )
    lhs = np.linspace(
        -1.0, 1.0, int(np.prod(lhs_shape)), dtype=np.float32
    ).reshape(lhs_shape)
    rhs = np.linspace(
        0.5, -0.5, int(np.prod(rhs_shape)), dtype=np.float32
    ).reshape(rhs_shape)
    operators = (
        ("Transpose", "Transpose", "MatMul", "Transpose")
        if batched else ("MatMul",)
    )
    return ContractCase(
        f"float_dynamic_matmul{'_batched' if batched else ''}",
        model,
        model,
        {"lhs": lhs, "rhs": rhs},
        operators,
    )


def _softmax_axis_case(*, rank: int, last_axis: bool) -> ContractCase:
    """Softmax over the axis the layout puts where the kernel reduces.

    The kernel normalizes along the final stored dimension. Spatial storage puts
    the channel axis there and the model's own order puts the last ONNX axis
    there, so both are expressible and the layout is what selects between them.
    ONNX Runtime evaluates the axis as written either way.
    """
    shape = [1, 4, 3, 5] if rank == 4 else [1, 4, 6]
    axis = -1 if last_axis else 1
    model = _model(
        "softmax_axis",
        [helper.make_node("Softmax", ["input"], ["output"], axis=axis)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    count = int(np.prod(shape))
    data = np.linspace(-2.0, 2.0, count, dtype=np.float32).reshape(shape)
    # Reaching the model's own order costs a conversion at each end, because
    # boundaries keep the channels-last convention.
    operators = (
        ("Transpose", "Softmax", "Transpose") if last_axis else ("Softmax",)
    )
    return ContractCase(
        f"float_softmax_rank{rank}_{'last' if last_axis else 'channel'}_axis",
        model,
        model,
        {"input": data},
        operators,
    )


def _tiled_softmax_case() -> ContractCase:
    """Softmax on a stage the solver has to tile.

    Normalization runs along the final stored dimension and the tile cuts
    stored axis 1, so each tile holds whole rows. The budget is set well below
    the tensor so the plan cannot be a single tile: if the kernel ignored its
    tile geometry it would normalize the wrong span and the values would not
    match ONNX Runtime.
    """
    shape = [1, 6, 512]
    model = _model(
        "tiled_softmax",
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_softmax",
        model,
        model,
        {"input": data},
        ("Softmax",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _subtract_case() -> ContractCase:
    """Sub of two activations, the form whose operand order is unambiguous."""
    shape = [1, 3, 4]
    model = _model(
        "subtract",
        [helper.make_node("Sub", ["left", "right"], ["output"])],
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    count = int(np.prod(shape))
    return ContractCase(
        "float_subtract",
        model,
        model,
        {
            "left": np.linspace(-2.0, 2.0, count, dtype=np.float32).reshape(shape),
            "right": np.linspace(1.0, -1.0, count, dtype=np.float32).reshape(shape),
        },
        ("Sub",),
    )


def _qdq_subtract_case() -> ContractCase:
    """A QDQ Sub of two quantized activations.

    Both operands arrive from DequantizeLinear, so the difference is taken in
    the integer domain and requantized the way TFLite does it. The reference
    evaluates the same graph, so a sign or scale error in the shared Add/Sub
    path shows up here rather than in the float case.
    """
    shape = [1, 1, 4, 4]
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, shape)
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, shape)
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zero_point"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array([-8], dtype=np.int8), "out_zero_point"),
        numpy_helper.from_array(np.array([[[[0.5]]]], dtype=np.float32), "weight"),
        numpy_helper.from_array(
            np.array([0.25], dtype=np.float32), "weight_scale"),
        numpy_helper.from_array(
            np.array([0], dtype=np.int8), "weight_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "io_scale", "io_zero_point"], ["input_q"]),
        helper.make_node(
            "DequantizeLinear", ["input_q", "io_scale", "io_zero_point"],
            ["input_dq"]),
        helper.make_node(
            "QuantizeLinear", ["weight", "weight_scale", "weight_zero_point"],
            ["weight_q"]),
        helper.make_node(
            "DequantizeLinear", ["weight_q", "weight_scale", "weight_zero_point"],
            ["weight_dq"]),
        helper.make_node("Conv", ["input_dq", "weight_dq"], ["branch"]),
        helper.make_node(
            "QuantizeLinear", ["branch", "io_scale", "io_zero_point"],
            ["branch_q"]),
        helper.make_node(
            "DequantizeLinear", ["branch_q", "io_scale", "io_zero_point"],
            ["branch_dq"]),
        helper.make_node("Sub", ["input_dq", "branch_dq"], ["difference"]),
        helper.make_node(
            "QuantizeLinear", ["difference", "out_scale", "out_zero_point"],
            ["output_q"]),
        helper.make_node(
            "DequantizeLinear", ["output_q", "out_scale", "out_zero_point"],
            ["output"]),
    ]
    compile_model = _model(
        "qdq_subtract", nodes, [model_input], [model_output], initializers)
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    input_data = np.linspace(
        -2.0, 1.75, 16, dtype=np.float32).reshape(shape)
    return ContractCase(
        "int8_subtract",
        compile_model,
        reference_model,
        {"input": input_data},
        ("Conv", "Sub"),
    )


def _global_max_pool_case() -> ContractCase:
    """GlobalMaxPool, the counterpart of GlobalAveragePool."""
    model = _model(
        "global_max_pool",
        [helper.make_node("GlobalMaxPool", ["input"], ["output"])],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 3, 2, 4])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    data = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(1, 3, 2, 4)
    return ContractCase(
        "float_global_max_pool",
        model,
        model,
        {"input": data},
        ("GlobalMaxPool",),
    )


def _sub_constant_case() -> ContractCase:
    """Sub against a constant, which compiles as an added negation.

    ONNX Runtime evaluates the subtraction as written, so a sign error in the
    rewrite shows up here.
    """
    shape = [1, 4]
    constant = numpy_helper.from_array(
        np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32), "constant")
    model = _model(
        "sub_constant",
        [
            helper.make_node("Relu", ["input"], ["gated"]),
            helper.make_node("Sub", ["gated", "constant"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [constant],
    )
    return ContractCase(
        "float_sub_constant",
        model,
        model,
        {"input": np.array([[0.25, -1.0, 5.0, -4.0]], dtype=np.float32)},
        ("Relu", "Add"),
    )


def _clip_as_relu_case() -> ContractCase:
    """Clip with a zero floor and no ceiling, which an exporter writes for Relu."""
    shape = [1, 6]
    lower = numpy_helper.from_array(np.float32(0.0), "lower")
    model = _model(
        "clip_as_relu",
        [helper.make_node("Clip", ["input", "lower"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [lower],
    )
    return ContractCase(
        "float_clip_as_relu",
        model,
        model,
        {"input": np.array([[-3.0, -0.5, 0.0, 0.5, 2.0, 9.0]],
                           dtype=np.float32)},
        ("Relu",),
    )


def _negate_case() -> ContractCase:
    """Neg has no opcode; it compiles as a multiplication by minus one."""
    shape = [1, 5]
    model = _model(
        "negate",
        [
            helper.make_node("Relu", ["input"], ["gated"]),
            helper.make_node("Neg", ["gated"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    return ContractCase(
        "float_negate",
        model,
        model,
        {"input": np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]], dtype=np.float32)},
        ("Relu", "Mul"),
    )


def _gemm_scaled_case() -> ContractCase:
    """A Gemm carrying alpha and beta, folded into the constants they scale.

    The plan has no field for either and the kernel computes Y = X * W^T + B,
    so before the fold this compiled and returned a silently wrong answer.
    ONNX Runtime applies both, which is what makes this case decisive.
    """
    weight = numpy_helper.from_array(
        np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(4, 3), "weight")
    bias = numpy_helper.from_array(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "bias")
    model = _model(
        "gemm_scaled",
        [
            helper.make_node(
                "Gemm", ["input", "weight", "bias"], ["output"],
                alpha=2.0, beta=3.0, transB=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [weight, bias],
    )
    return ContractCase(
        "float_gemm_scaled",
        model,
        model,
        {"input": np.array([[1.0, -2.0, 0.5]], dtype=np.float32)},
        ("Gemm",),
    )


def _chain_pointwise_before_spatial_case() -> ContractCase:
    """A chain stage whose first op comes before its convolution.

    exec_chain_tiled set the tile context once per stage, at the stage's
    OUTPUT height, and corrected it only when it reached the spatial op. An op
    ahead of the convolution therefore computed only the rows the stage
    finally emits and left the halo rows of its own output untouched, which
    the convolution then read as data: a silent wrong answer on interior
    rows.

    The Clip is what puts a pointwise op ahead of the convolution, and the two
    Concats are what make the stage large enough that the partitioner cuts it
    into two chained stages at this budget rather than one stage or four.
    """
    c, s = 8, 13
    rng = np.random.default_rng(0)
    weight = (rng.normal(size=(c, c, 3, 3)) * 0.2).astype(np.float32)
    model = _model(
        "chain_pointwise_before_spatial",
        [
            helper.make_node("Clip", ["input", "lo", "hi"], ["clipped"]),
            helper.make_node(
                "Conv", ["clipped", "w"], ["conv"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Concat", ["conv", "conv"], ["wide"], axis=1),
            helper.make_node("Concat", ["wide", "wide"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, s, s])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 4 * c, s, s])],
        initializers=[
            numpy_helper.from_array(weight, "w"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "lo"),
            numpy_helper.from_array(np.array(6.0, dtype=np.float32), "hi"),
        ],
    )
    data = (rng.normal(size=(1, c, s, s)) * 0.7).astype(np.float32)
    return ContractCase(
        "float_chain_pointwise_before_spatial",
        model,
        model,
        {"input": data},
        ("Relu6", "Conv", "Concat", "Concat"),
        mem_budget="16K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
    )


def _chain_skip_out_of_a_stage_case() -> ContractCase:
    """A chained stage that emits a tensor produced before its last convolution.

    A stage may hand more than one tensor to later stages, and one of them can
    be written by an op that runs before the stage's last spatial op. That
    tensor spans the spatial op's INPUT rows: more rows than the stage finally
    emits, starting on a different row. Both tiled executors spilled every
    stage output at the stage's output range, so the tensor landed shifted in
    slow memory from the second tile onward, and the later Concat read it.

    The SiLU ahead of the two convolutions is what overflows the budget and
    makes the partitioner chain, and the Concat is what keeps the 1x1
    convolution's output alive past the stage that wrote it.
    """
    c, s = 4, 16
    rng = np.random.default_rng(0)
    first = (rng.normal(size=(c, c, 3, 3)) * 0.2).astype(np.float32)
    point = (rng.normal(size=(c, c, 1, 1)) * 0.3).astype(np.float32)
    deep = (rng.normal(size=(c, c, 3, 3)) * 0.2).astype(np.float32)
    model = _model(
        "chain_skip_out_of_a_stage",
        [
            helper.make_node(
                "Conv", ["input", "first"], ["a"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Sigmoid", ["a"], ["gate"]),
            helper.make_node("Mul", ["a", "gate"], ["silu"]),
            helper.make_node(
                "Conv", ["silu", "point"], ["skip"], kernel_shape=[1, 1]),
            helper.make_node(
                "Conv", ["skip", "deep"], ["wide"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Concat", ["skip", "wide"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, s, s])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 2 * c, s, s])],
        initializers=[
            numpy_helper.from_array(first, "first"),
            numpy_helper.from_array(point, "point"),
            numpy_helper.from_array(deep, "deep"),
        ],
    )
    data = (rng.normal(size=(1, c, s, s)) * 0.7).astype(np.float32)
    return ContractCase(
        "float_chain_skip_out_of_a_stage",
        model,
        model,
        {"input": data},
        ("Conv", "Sigmoid", "Mul", "Conv", "Conv", "Concat"),
        mem_budget="8K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
    )


def _chained_normalization_case() -> ContractCase:
    """A chain whose second stage normalizes.

    The chain operator list in the loader is a fifth copy of the height-stripe
    set and had drifted by four operators. A chain is a run of stripe-tileable
    stages, so anything the stripe contract admits has to survive being
    chained. The budget is what makes the compiler cut and chain: the same
    graph at a roomy budget is one stage and exercises none of this.
    """
    shape = [1, 4, 26, 26]
    model = _model(
        "chained_normalization",
        [
            helper.make_node("Add", ["input", "input"], ["sum"]),
            helper.make_node("Softmax", ["sum"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -2.0, 2.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_chained_normalization",
        model,
        model,
        {"input": data},
        ("Add", "Softmax"),
        mem_budget="16K",
        expect_tiled=True,
        expect_chain=True,
    )


def _tiled_rank4_binary_case(*, op_type: str) -> ContractCase:
    """A rank-4 binary pointwise stage the solver has to tile.

    The height-stripe contract is stated in four places: the compiler's op
    category table, the loader's rank-4 operator list, the executor's
    is_height_tiling_op, and the accelerator routing policy. Sub was in three
    of them, so a model with one compiled and then failed to load. One case
    per binary operator keeps each of the four honest.
    """
    shape = [1, 4, 32, 32]
    model = _model(
        f"tiled_rank4_{op_type.lower()}",
        [helper.make_node(op_type, ["left", "right"], ["output"])],
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    count = int(np.prod(shape))
    return ContractCase(
        f"float_tiled_rank4_{op_type.lower()}",
        model,
        model,
        {
            "left": np.linspace(-1.0, 1.0, count, dtype=np.float32).reshape(shape),
            "right": np.linspace(0.5, -0.5, count, dtype=np.float32).reshape(shape),
        },
        (op_type,),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_softmax_rank4_case() -> ContractCase:
    """Softmax on a rank-4 stage the solver has to tile.

    The rank-3 twin above covers the NLC contract. This covers the NHWC one,
    which the compiler, the loader and the executor each state separately.
    """
    shape = [1, 4, 32, 32]
    model = _model(
        "tiled_softmax_rank4",
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_softmax_rank4",
        model,
        model,
        {"input": data},
        ("Softmax",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_last_axis_softmax_case(*, rank: int) -> ContractCase:
    """Softmax over the last axis on a stage the solver has to tile.

    Normalization runs along the model's own final axis, which is not the axis
    the runtime stores last, so the graph compiles to a conversion, the
    Softmax, and a conversion back. At this budget all three have to tile, and
    the two conversions tile along opposite transfers.

    The rank-4 shape is the one an attention block produces: the conversion
    there moves the channel axis past two spatial axes rather than one, which
    collapses to the same transpose only because those two keep their order.
    """
    shape = [1, 512, 6] if rank == 3 else [1, 6, 16, 16]
    name = f"tiled_last_axis_softmax_rank{rank}"
    model = _model(
        name,
        [helper.make_node("Softmax", ["input"], ["output"], axis=-1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        f"float_{name}",
        model,
        model,
        {"input": data},
        ("Transpose", "Softmax", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _layer_norm_case() -> ContractCase:
    """LayerNormalization over the model's own last axis.

    The kernel normalizes along the final stored dimension, so the graph
    compiles to a conversion, the normalization, and a conversion back, the
    same shape a last-axis Softmax takes.
    """
    shape = [1, 12, 16]
    initializers = [
        numpy_helper.from_array(
            np.linspace(0.75, 1.25, shape[-1], dtype=np.float32), "gamma"),
        numpy_helper.from_array(
            np.linspace(-0.2, 0.2, shape[-1], dtype=np.float32), "beta"),
    ]
    model = _model(
        "layer_norm",
        [helper.make_node(
            "LayerNormalization", ["input", "gamma", "beta"], ["output"],
            axis=-1, epsilon=1e-5)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
        opset=17,
    )
    # Each row spans its own range about zero. A row whose spread is small
    # against its mean makes the centering cancel and the division by a small
    # deviation amplify what is left, which measures float32 summation order
    # against ONNX Runtime rather than the kernel.
    row = np.linspace(-3.0, 3.0, shape[-1], dtype=np.float32)
    gains = (1.0 + 0.1 * np.arange(shape[1], dtype=np.float32))[:, None]
    data = (row[None, None, :] * gains[None, :, :]).astype(np.float32)
    return ContractCase(
        "float_layer_norm",
        model,
        model,
        {"input": data},
        ("Transpose", "LayerNormalization", "Transpose"),
    )


def _erf_case() -> ContractCase:
    """Erf, the exact form GELU is written in."""
    shape = [1, 3, 8]
    model = _model(
        "erf",
        [helper.make_node("Erf", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase("float_erf", model, model, {"input": data}, ("Erf",))


def _qdq_rescale_passthrough_case(*, op_type: str) -> ContractCase:
    """A value-preserving int8 operator whose output rescales.

    Relu, Relu6, Reshape and nearest Resize all carry a value through
    unchanged, and all four used to write the input's encoding straight out.
    One case each, with an output scale and zero point deliberately unlike the
    input's, which is what a quantizer that assigns per-tensor scales
    independently produces.
    """
    c, s = 4, 8
    in_scale, out_scale = 0.05, 0.08
    initializers = [
        numpy_helper.from_array(np.array(in_scale, dtype=np.float32), "in_s"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "in_z"),
        numpy_helper.from_array(np.array(out_scale, dtype=np.float32), "out_s"),
        numpy_helper.from_array(np.array(-3, dtype=np.int8), "out_z"),
    ]
    out_shape = [1, c, s, s]
    if op_type == "Relu":
        body = [helper.make_node("Relu", ["idq"], ["raw"])]
    elif op_type == "Relu6":
        initializers += [
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "lo"),
            numpy_helper.from_array(np.array(6.0, dtype=np.float32), "hi"),
        ]
        body = [helper.make_node("Clip", ["idq", "lo", "hi"], ["raw"])]
    elif op_type == "Reshape":
        initializers.append(numpy_helper.from_array(
            np.array([1, c, s * s], dtype=np.int64), "newshape"))
        body = [helper.make_node("Reshape", ["idq", "newshape"], ["raw"])]
        out_shape = [1, c, s * s]
    else:
        initializers.append(numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"))
        body = [helper.make_node(
            "Resize", ["idq", "", "scales"], ["raw"], mode="nearest",
            coordinate_transformation_mode="asymmetric", nearest_mode="floor")]
        out_shape = [1, c, 2 * s, 2 * s]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_s", "in_z"], ["iq"]),
        helper.make_node("DequantizeLinear", ["iq", "in_s", "in_z"], ["idq"]),
    ] + body + [
        helper.make_node("QuantizeLinear", ["raw", "out_s", "out_z"], ["oq"]),
        helper.make_node(
            "DequantizeLinear", ["oq", "out_s", "out_z"], ["output"]),
    ]
    compile_model = _model(
        f"qdq_rescale_{op_type.lower()}",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, s, s])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    rng = np.random.default_rng(5)
    data = (rng.integers(-100, 100, size=(1, c, s, s)).astype(np.float32)
            * in_scale)
    return ContractCase(
        f"int8_rescale_{op_type.lower()}",
        compile_model,
        reference_model,
        {"input": data},
        (op_type,),
    )


def _qdq_max_pool_rescale_case() -> ContractCase:
    """An int8 MaxPool whose output declares a different quantization.

    A maximum preserves the value, not its encoding. Both max kernels wrote the
    input's encoding straight out, so a model whose quantizer assigned the pool
    a different output scale came back wrong by tens of steps with no error.
    kern_avg_pool_s8 has always defined this case; this is its MaxPool twin.
    """
    c, s = 4, 9
    in_scale, out_scale = 0.05, 0.08
    initializers = [
        numpy_helper.from_array(np.array(in_scale, dtype=np.float32), "in_s"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "in_z"),
        numpy_helper.from_array(np.array(out_scale, dtype=np.float32), "out_s"),
        numpy_helper.from_array(np.array(-3, dtype=np.int8), "out_z"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_s", "in_z"], ["iq"]),
        helper.make_node("DequantizeLinear", ["iq", "in_s", "in_z"], ["idq"]),
        helper.make_node(
            "MaxPool", ["idq"], ["raw"], kernel_shape=[3, 3], strides=[1, 1],
            pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["raw", "out_s", "out_z"], ["oq"]),
        helper.make_node(
            "DequantizeLinear", ["oq", "out_s", "out_z"], ["output"]),
    ]
    compile_model = _model(
        "qdq_max_pool_rescale",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, s, s])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, c, s, s])],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    rng = np.random.default_rng(5)
    data = (rng.integers(-100, 100, size=(1, c, s, s)).astype(np.float32)
            * in_scale)
    return ContractCase(
        "int8_max_pool_rescale",
        compile_model,
        reference_model,
        {"input": data},
        ("MaxPool",),
    )


def _qdq_erf_case() -> ContractCase:
    """The int8 sibling of _erf_case, through a lookup table."""
    shape = [1, 3, 8]
    initializers = [
        numpy_helper.from_array(np.array(0.03, dtype=np.float32), "in_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "in_zero_point"),
        numpy_helper.from_array(np.array(0.01, dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array(-5, dtype=np.int8), "out_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "in_scale", "in_zero_point"], ["iq"]),
        helper.make_node(
            "DequantizeLinear", ["iq", "in_scale", "in_zero_point"], ["idq"]),
        helper.make_node("Erf", ["idq"], ["raw"]),
        helper.make_node(
            "QuantizeLinear", ["raw", "out_scale", "out_zero_point"], ["oq"]),
        helper.make_node(
            "DequantizeLinear",
            ["oq", "out_scale", "out_zero_point"], ["output"]),
    ]
    compile_model = _model(
        "qdq_erf",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    data = np.linspace(
        -2.0, 2.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "int8_erf", compile_model, reference_model, {"input": data}, ("Erf",))


def _qdq_layer_norm_case() -> ContractCase:
    """The int8 sibling of _layer_norm_case.

    A normalization divides by a per-row standard deviation that no fixed
    multiplier stands in for, so the kernel takes its statistics in float from
    the dequantized row and requantizes only the result.
    """
    shape = [1, 12, 16]
    initializers = [
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), "in_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "in_zero_point"),
        numpy_helper.from_array(np.array(0.03, dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array(2, dtype=np.int8), "out_zero_point"),
        numpy_helper.from_array(
            np.linspace(0.75, 1.25, shape[-1], dtype=np.float32), "gamma"),
        numpy_helper.from_array(
            np.linspace(-0.2, 0.2, shape[-1], dtype=np.float32), "beta"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "in_scale", "in_zero_point"], ["iq"]),
        helper.make_node(
            "DequantizeLinear", ["iq", "in_scale", "in_zero_point"], ["idq"]),
        helper.make_node(
            "LayerNormalization", ["idq", "gamma", "beta"], ["raw"],
            axis=-1, epsilon=1e-5),
        helper.make_node(
            "QuantizeLinear", ["raw", "out_scale", "out_zero_point"], ["oq"]),
        helper.make_node(
            "DequantizeLinear",
            ["oq", "out_scale", "out_zero_point"], ["output"]),
    ]
    compile_model = _model(
        "qdq_layer_norm",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
        opset=17,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    row = np.linspace(-3.0, 3.0, shape[-1], dtype=np.float32)
    gains = (1.0 + 0.1 * np.arange(shape[1], dtype=np.float32))[:, None]
    data = (row[None, None, :] * gains[None, :, :]).astype(np.float32)
    return ContractCase(
        "int8_layer_norm",
        compile_model,
        reference_model,
        {"input": data},
        ("Transpose", "LayerNormalization", "Transpose"),
    )


def _tiled_layer_norm_case() -> ContractCase:
    """LayerNormalization on a stage the solver has to tile.

    A tile cuts an axis ahead of the one the kernel normalizes, so each tile
    holds whole normalization rows. If the kernel took its statistics over the
    wrong span the values would not match ONNX Runtime.
    """
    shape = [1, 256, 16]
    initializers = [
        numpy_helper.from_array(
            np.linspace(0.75, 1.25, shape[-1], dtype=np.float32), "gamma"),
        numpy_helper.from_array(
            np.linspace(-0.2, 0.2, shape[-1], dtype=np.float32), "beta"),
    ]
    model = _model(
        "tiled_layer_norm",
        [helper.make_node(
            "LayerNormalization", ["input", "gamma", "beta"], ["output"],
            axis=-1, epsilon=1e-5)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
        opset=17,
    )
    row = np.linspace(-3.0, 3.0, shape[-1], dtype=np.float32)
    gains = (1.0 + 0.01 * np.arange(shape[1], dtype=np.float32))[:, None]
    data = (row[None, None, :] * gains[None, :, :]).astype(np.float32)
    return ContractCase(
        "float_tiled_layer_norm",
        model,
        model,
        {"input": data},
        ("Transpose", "LayerNormalization", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_erf_case() -> ContractCase:
    """Erf on a stage the solver has to tile."""
    shape = [1, 8, 512]
    model = _model(
        "tiled_erf",
        [helper.make_node("Erf", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_erf",
        model,
        model,
        {"input": data},
        ("Erf",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_matrix_rows_case() -> ContractCase:
    """A projection pipeline the solver has to band along its rows.

    A matrix product against a constant reads one row to write one row, so a
    band of rows computes exactly the rows it holds. Rank 2 has no axis the
    height contract names, which is why this needs its own path: the
    projections of a transformer lower to exactly this shape.
    """
    rows, width, hidden = 256, 32, 64
    initializers = [
        numpy_helper.from_array(
            (np.arange(width * hidden, dtype=np.float32).reshape(width, hidden)
             / (width * hidden) - 0.5), "w1"),
        numpy_helper.from_array(
            (np.arange(hidden * width, dtype=np.float32).reshape(hidden, width)
             / (hidden * width) - 0.5), "w2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["input", "w1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["act"]),
        helper.make_node("MatMul", ["act", "w2"], ["output"]),
    ]
    model = _model(
        "tiled_matrix_rows",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, rows, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, rows, width])],
        initializers,
    )
    data = np.linspace(
        -1.0, 1.0, rows * width, dtype=np.float32
    ).reshape(1, rows, width)
    return ContractCase(
        "float_tiled_matrix_rows",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Relu",
         "Reshape", "Gemm", "Reshape", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_attention_transpose_case() -> ContractCase:
    """A transpose the model itself asks for, on a stage that has to tile.

    An attention block transposes its keys between two operands that are both
    already in the model's own order, so no layout conversion is involved.
    The permutation of stored axes is the same one a conversion emits, which
    is what lets the same band serve it.
    """
    tokens, width = 256, 16
    weight = numpy_helper.from_array(
        (np.arange(width * width, dtype=np.float32).reshape(width, width)
         / (width * width) - 0.5), "wk")
    nodes = [
        helper.make_node("MatMul", ["input", "wk"], ["keys"]),
        helper.make_node("Transpose", ["keys"], ["output"], perm=[0, 2, 1]),
    ]
    model = _model(
        "tiled_attention_transpose",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, width, tokens])],
        [weight],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        "float_tiled_attention_transpose",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
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


def _tiled_global_reduction_case() -> ContractCase:
    """A GlobalAveragePool too large for the budget, so it reduces in bands."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 8, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 1, 1]
    )
    model = _model(
        "tiled_global_reduction",
        [
            helper.make_node(
                "GlobalAveragePool", ["input"], ["output"], name="gap"
            )
        ],
        [model_input],
        [model_output],
    )
    # 4096 samples per channel, kept inside [-1, 1] so the running sum stays
    # small: a mean over values of magnitude 30 accumulates past 1e5, where
    # float32 summation order alone moves the result by more than the 1e-5
    # bound and the case would measure numpy against ONNX Runtime rather than
    # tiled against untiled.
    values = np.arange(1 * 8 * 64 * 64, dtype=np.float32).reshape(1, 8, 64, 64)
    return ContractCase(
        "float_tiled_global_reduction",
        model,
        model,
        {"input": (values % 17.0) * 0.125 - 1.0},
        ("GlobalAveragePool",),
        mem_budget="16K",
        expect_tiled=True,
    )


def _qdq_tiled_global_reduction_case() -> ContractCase:
    """The int8 sibling: the banded sum must requantize exactly once."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 8, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 1, 1]
    )
    initializers = [
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "input_scale"),
        numpy_helper.from_array(np.array(-3, dtype=np.int8), "input_zero_point"),
        numpy_helper.from_array(np.array(0.125, dtype=np.float32), "output_scale"),
        numpy_helper.from_array(np.array(7, dtype=np.int8), "output_zero_point"),
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
            "GlobalAveragePool", ["input_dq"], ["raw"], name="gap"
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
        "qdq_tiled_global_reduction",
        nodes,
        [model_input],
        [model_output],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    values = np.arange(1 * 8 * 64 * 64, dtype=np.float32).reshape(1, 8, 64, 64)
    return ContractCase(
        "int8_tiled_global_reduction",
        compile_model,
        reference_model,
        {"input": (values % 61) * 0.25 - 7.5},
        ("GlobalAveragePool",),
        mem_budget="8K",
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
    """Build a QDQ Conv, ConvTranspose, or AveragePool model with an int8 ORT reference."""
    if operator == "Conv":
        output_shape = [1, 1, 4, 4]
    elif operator == "ConvTranspose":
        # stride 2, kernel 2, pad 0: 2 * (4 - 1) + 2 = 8 on each spatial axis.
        output_shape = [1, 1, 8, 8]
    else:
        output_shape = [1, 1, 2, 2]
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    input_scale = numpy_helper.from_array(
        np.array(0.25, dtype=np.float32), "input_scale"
    )
    input_zero_point = numpy_helper.from_array(
        np.array(0, dtype=np.int8), "input_zero_point"
    )
    output_scale = numpy_helper.from_array(
        np.array(0.25, dtype=np.float32), "output_scale"
    )
    output_zero_point = numpy_helper.from_array(
        np.array(0, dtype=np.int8), "output_zero_point"
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
            np.array(0.25, dtype=np.float32), "weight_scale"
        )
        weight_zero_point = numpy_helper.from_array(
            np.array(0, dtype=np.int8), "weight_zero_point"
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
    elif operator == "ConvTranspose":
        # ONNX ConvTranspose weight is [C_in, C_out, kH, kW]; here 1 -> 1 with a
        # 2x2 kernel, stride 2, pad 0, group 1. Weight values are exact
        # multiples of the weight scale so the fake-quant is lossless.
        weight = numpy_helper.from_array(
            np.array([[[[0.5, -0.25], [0.25, 0.75]]]], dtype=np.float32), "weight"
        )
        weight_scale = numpy_helper.from_array(
            np.array(0.25, dtype=np.float32), "weight_scale"
        )
        weight_zero_point = numpy_helper.from_array(
            np.array(0, dtype=np.int8), "weight_zero_point"
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
                helper.make_node(
                    "ConvTranspose",
                    ["input_dq", "weight_dq"],
                    ["raw"],
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    pads=[0, 0, 0, 0],
                    group=1,
                ),
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


def _qdq_add_relu_case() -> ContractCase:
    """A QDQ residual Add whose Relu sits between the Add and its QuantizeLinear.

    This is how an ONNX quantizer writes a ResNet residual block: both Add
    operands arrive from DequantizeLinear, the sum stays on an unquantized edge,
    and the only QuantizeLinear comes after the Relu. The activation therefore
    belongs to the Add's output requantization, and the compiler has to fuse it
    to give the sum a dtype at all. The output zero point is non-zero so the
    fused lower bound is the zero point rather than the natural int8 floor, and
    the input reaches negative sums so the clamp is observable.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zero_point"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array([-8], dtype=np.int8), "out_zero_point"),
        numpy_helper.from_array(np.array([[[[0.5]]]], dtype=np.float32), "weight"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "weight_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "weight_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "io_scale", "io_zero_point"], ["input_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["input_q", "io_scale", "io_zero_point"], ["input_dq"]
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
        helper.make_node("Conv", ["input_dq", "weight_dq"], ["branch"]),
        helper.make_node(
            "QuantizeLinear", ["branch", "io_scale", "io_zero_point"], ["branch_q"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["branch_q", "io_scale", "io_zero_point"],
            ["branch_dq"],
        ),
        helper.make_node("Add", ["input_dq", "branch_dq"], ["sum"]),
        helper.make_node("Relu", ["sum"], ["activated"]),
        helper.make_node(
            "QuantizeLinear",
            ["activated", "out_scale", "out_zero_point"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "out_scale", "out_zero_point"],
            ["output"],
        ),
    ]
    compile_model = _model(
        "qdq_add_relu", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    input_data = np.array(
        [
            [
                [
                    [-2.0, -1.75, -1.5, -1.25],
                    [-1.0, -0.75, -0.5, -0.25],
                    [0.0, 0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5, 1.75],
                ]
            ]
        ],
        dtype=np.float32,
    )
    return ContractCase(
        "int8_add_relu",
        compile_model,
        reference_model,
        {"input": input_data},
        ("Conv", "Add"),
    )


def _qdq_gemm_bias_add_case() -> ContractCase:
    """A quantized Gemm whose bias arrives as an unfused float Add.

    A quantizer that does not fuse the classifier bias writes it as an Add on
    the dequantized product, which the runtime cannot execute: a constant
    operand carries no scale or zero point. The compiler requantizes the
    constant into the product's int32 accumulator domain and hands it to the
    operator as a bias, so the plan ends at the product's own int8 encoding
    rather than the float the ONNX graph declares. The reference model
    quantizes its float result with the same scale to compare on that footing.
    Bias values are exact multiples of the bias scale, so the fold itself
    introduces no rounding.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2]
    )
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zero_point"),
        numpy_helper.from_array(
            np.array([[0.5, -0.25, 0.75, 0.25], [-0.5, 0.25, 0.5, -0.75]],
                     dtype=np.float32),
            "weight",
        ),
        # 0.25 * 0.25 = 0.0625 is the bias scale; both values are multiples.
        numpy_helper.from_array(np.array([0.5, -0.25], dtype=np.float32), "bias"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "io_scale", "io_zero_point"], ["input_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["input_q", "io_scale", "io_zero_point"], ["input_dq"]
        ),
        helper.make_node(
            "QuantizeLinear", ["weight", "io_scale", "io_zero_point"], ["weight_q"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "io_scale", "io_zero_point"],
            ["weight_dq"],
        ),
        helper.make_node(
            "Gemm", ["input_dq", "weight_dq"], ["product"], transB=1
        ),
        helper.make_node(
            "QuantizeLinear", ["product", "io_scale", "io_zero_point"], ["product_q"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["product_q", "io_scale", "io_zero_point"],
            ["product_dq"],
        ),
        helper.make_node("Add", ["product_dq", "bias"], ["output"]),
    ]
    compile_model = _model(
        "qdq_gemm_bias_add", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    reference_model.graph.node.append(
        helper.make_node(
            "QuantizeLinear", ["output", "io_scale", "io_zero_point"], ["output_q"]
        )
    )
    onnx.checker.check_model(reference_model)
    return ContractCase(
        "int8_gemm_bias_add",
        compile_model,
        reference_model,
        {"input": np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)},
        ("Gemm",),
    )


def _matmul_case() -> ContractCase:
    """A rank-2 MatMul against a constant weight.

    ONNX states the product with the weight as [K, N] while the kernels index
    it as [OC, IC], so the compiler transposes the constant and relabels the
    operator. A deliberately asymmetric weight makes a missed transpose change
    the result rather than hide in a symmetric matrix.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    weight = numpy_helper.from_array(
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ),
        "weight",
    )
    model = _model(
        "matmul",
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        [model_input],
        [model_output],
        [weight],
    )
    return ContractCase(
        "float_matmul",
        model,
        model,
        {"input": np.array([[0.5, -1.0, 2.0, -0.25]], dtype=np.float32)},
        ("Gemm",),
    )


def _gemm_no_transpose_case() -> ContractCase:
    """A Gemm that leaves transB at its default.

    The weight is then [K, N] like MatMul's, so it needs the same transpose
    before it means what the kernels compute.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    initializers = [
        numpy_helper.from_array(
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0]],
                dtype=np.float32,
            ),
            "weight",
        ),
        numpy_helper.from_array(
            np.array([0.5, -1.5, 2.0], dtype=np.float32), "bias"
        ),
    ]
    model = _model(
        "gemm_no_transpose",
        [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
        [model_input],
        [model_output],
        initializers,
    )
    return ContractCase(
        "float_gemm_no_transpose",
        model,
        model,
        {"input": np.array([[0.5, -1.0, 2.0, -0.25]], dtype=np.float32)},
        ("Gemm",),
    )


def _qdq_matmul_case() -> ContractCase:
    """A quantized rank-2 MatMul, the shape an int8 classifier head takes."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zp"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "w_zp"),
        numpy_helper.from_array(
            np.array(
                [[0.5, 1.0, 1.5], [2.0, -0.5, 1.0], [-1.0, 0.5, 2.0],
                 [1.5, -2.0, 0.5]],
                dtype=np.float32,
            ),
            "weight",
        ),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "io_scale", "io_zp"], ["in_q"]),
        helper.make_node(
            "DequantizeLinear", ["in_q", "io_scale", "io_zp"], ["in_dq"]
        ),
        helper.make_node("QuantizeLinear", ["weight", "w_scale", "w_zp"], ["w_q"]),
        helper.make_node(
            "DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"]
        ),
        helper.make_node("MatMul", ["in_dq", "w_dq"], ["product"]),
        helper.make_node(
            "QuantizeLinear", ["product", "io_scale", "io_zp"], ["output_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["output_q", "io_scale", "io_zp"], ["output"]
        ),
    ]
    compile_model = _model(
        "qdq_matmul", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    return ContractCase(
        "int8_matmul",
        compile_model,
        reference_model,
        {"input": np.array([[0.5, -1.0, 2.0, -0.25]], dtype=np.float32)},
        ("Gemm",),
    )


def _quint8_activation_case() -> ContractCase:
    """A QDQ Conv whose activations are quantized as uint8.

    That is what the ONNX Runtime quantizer emits by default: uint8
    activations with int8 weights. uint8 value v and int8 value v - 128 denote
    the same real number under zero points that differ by the same 128, so the
    compiler restates the activation in the signed domain the kernels work in.
    The final QuantizeLinear stays int8 so the plan and the reference compare
    on the same footing; every interior activation exercises the shift.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "u8_scale"),
        numpy_helper.from_array(np.array([128], dtype=np.uint8), "u8_zp"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "s8_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "s8_zp"),
        numpy_helper.from_array(np.array([[[[0.5]]]], dtype=np.float32), "weight"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "w_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "u8_scale", "u8_zp"], ["in_q"]),
        helper.make_node(
            "DequantizeLinear", ["in_q", "u8_scale", "u8_zp"], ["in_dq"]
        ),
        helper.make_node("QuantizeLinear", ["weight", "w_scale", "w_zp"], ["w_q"]),
        helper.make_node(
            "DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"]
        ),
        helper.make_node("Conv", ["in_dq", "w_dq"], ["conv"]),
        # An interior uint8 activation between the two operators.
        helper.make_node("QuantizeLinear", ["conv", "u8_scale", "u8_zp"], ["conv_q"]),
        helper.make_node(
            "DequantizeLinear", ["conv_q", "u8_scale", "u8_zp"], ["conv_dq"]
        ),
        helper.make_node("Relu", ["conv_dq"], ["activated"]),
        helper.make_node(
            "QuantizeLinear", ["activated", "s8_scale", "s8_zp"], ["output_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["output_q", "s8_scale", "s8_zp"], ["output"]
        ),
    ]
    compile_model = _model(
        "quint8_activation", nodes, [model_input], [model_output], initializers
    )
    reference_model = copy.deepcopy(compile_model)
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
        "quint8_activation",
        compile_model,
        reference_model,
        {"input": input_data},
        ("Conv", "Relu"),
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
    ConvTranspose (stride 2, kernel 2, pad 0) restores 8x8: the encoder-then-
    upsample shape that motivates ConvTranspose support. Both stages keep
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


def _qdq_convtranspose_per_channel_case() -> ContractCase:
    """Per-channel int8 QDQ ConvTranspose with multiple output channels.

    ONNX ConvTranspose weight is [C_in, C_out, kH, kW], so per-output-channel
    weight quantization uses axis=1 with a length-C_out scale vector, NOT
    axis=0 as for Conv (whose weight is [C_out, C_in, kH, kW]). This case uses
    C_in=2, C_out=3 with a distinct per-channel weight scale so a wrong
    output-channel axis anywhere in the int8 requant path (the effective-scale
    indexing) would diverge from the ORT reference beyond one LSB. Matched to
    one LSB like the other int8 cases.
    """
    c_in, c_out = 2, 3
    input_shape = [1, c_in, 4, 4]
    output_shape = [1, c_out, 8, 8]
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    rng = np.random.default_rng(31)
    input_scale = numpy_helper.from_array(
        np.array([0.25], dtype=np.float32), "input_scale"
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
        rng.normal(0.0, 0.2, size=(c_in, c_out, 2, 2)).astype(np.float32), "weight"
    )
    # Per output channel (axis=1): one scale and zero-point per C_out. The
    # scales differ per channel so a swapped axis mismatches every channel.
    weight_scale = numpy_helper.from_array(
        np.array([0.02, 0.03, 0.015], dtype=np.float32), "weight_scale"
    )
    weight_zero_point = numpy_helper.from_array(
        np.zeros(c_out, dtype=np.int8), "weight_zero_point"
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
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zero_point"],
            ["weight_dq"],
            axis=1,
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
        "qdq_convtranspose_per_channel",
        nodes,
        [model_input],
        [model_output],
        initializers,
    )
    reference_model = copy.deepcopy(compile_model)
    onnx.checker.check_model(reference_model)
    return ContractCase(
        "int8_convtranspose_per_channel",
        compile_model,
        reference_model,
        {"input": rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)},
        ("ConvTranspose",),
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
    rules are what let the multi-input Concat stage tile on both axes at all
    (any Concat stage was previously excluded from HW tiling); both the
    Concat and the Conv stage now solve a TILE_AXIS_HW tile with tiles_h > 1
    and tiles_w > 1, so the plan's first tile-plan record proves 2D. This is
    the co-tiled skip contract: the runtime loads up and skip at the same
    tile rectangle and must reproduce the whole-op result bit-exact.
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
    eligibility rules let the multi-input Add stage tile on both axes (Add was
    previously excluded from HW tiling). Both stages solve a TILE_AXIS_HW
    tile with tiles_h > 1 and tiles_w > 1, and the runtime loads x and skip
    at the same tile rectangle.
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
    ~120 KB activation peak far exceeds the 8K fast budget - a single output row
    does not fit on its own - and ConvTranspose is kept untileable on the 1D
    height and chain paths, so it routes only through the compiler's dedicated
    output-extent 2D solve, which splits both the height and width of the 32x32
    output. The solved core tile does not evenly divide the output, so the last
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
    bit-exact, which is the geometry (inverted rect plus effective pads) the
    ConvTranspose tiling support introduced.
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


def _declared_dtype(tensor: dict) -> int:
    """The dtype the model states for this boundary, which the runner converts."""
    return tensor["iface_dtype"] or tensor["dtype"]


def _pack_inputs(plan: dict, inputs: dict[str, Array]) -> bytes:
    chunks: list[bytes] = []
    for tensor_index in plan["model_inputs"]:
        tensor = plan["tensors"][tensor_index]
        source = inputs[tensor["name"]]
        # The runner is handed the dtype the model declares and converts it,
        # so the plan is exercised through the interface the model states.
        declared = _declared_dtype(tensor)
        if declared == TensorProto.FLOAT:
            encoded = source.astype(np.float32, copy=False)
        elif declared == TensorProto.INT8:
            encoded = _quantize_input(source, plan, tensor)
        else:
            raise AssertionError(
                f"unsupported contract input dtype {declared}"
            )
        encoded = _to_runtime_layout(encoded)
        expected = tensor["size_bytes"]
        if declared != tensor["dtype"]:
            expected = (
                tensor["size_bytes"] // np.dtype(_DTYPE_BY_ONNX_CODE[tensor["dtype"]]).itemsize
            ) * np.dtype(_DTYPE_BY_ONNX_CODE[declared]).itemsize
        if encoded.nbytes != expected:
            raise AssertionError(
                f"input {tensor['name']!r} has {encoded.nbytes} bytes, "
                f"the declared interface expects {expected}"
            )
        chunks.append(encoded.tobytes())
    return b"".join(chunks)


_TENSOR_FLAG_LINEAR = 0x08
"""Axes are stored in the order the model states them, not channels-last."""


def _decode_outputs(
    plan: dict, raw: bytes, reference_outputs: list[Array]
) -> list[Array]:
    decoded: list[Array] = []
    offset = 0
    if len(plan["model_outputs"]) != len(reference_outputs):
        raise AssertionError("plan and ONNX Runtime output counts differ")
    for tensor_index, reference in zip(plan["model_outputs"], reference_outputs):
        tensor = plan["tensors"][tensor_index]
        declared = _declared_dtype(tensor)
        dtype = _DTYPE_BY_ONNX_CODE.get(declared)
        if dtype is None:
            raise AssertionError(
                f"unsupported contract output dtype {declared}"
            )
        stored = _DTYPE_BY_ONNX_CODE[tensor["dtype"]]
        size_bytes = (
            tensor["size_bytes"] // np.dtype(stored).itemsize
        ) * np.dtype(dtype).itemsize
        end = offset + size_bytes
        if end > len(raw):
            raise AssertionError("runtime output file is truncated")
        value = np.frombuffer(raw[offset:end], dtype=dtype).copy()
        value = value.reshape(tensor["shape"])
        # Schema 7 records the order a tensor is stored in, so the harness reads
        # it instead of inferring it from the shape of the producing graph. That
        # guess could not tell a model's own terminal Transpose from a layout
        # conversion the compiler inserted, since both are a Transpose writing a
        # model output.
        if not tensor["flags"] & _TENSOR_FLAG_LINEAR:
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
        _assert_output_parity(
        actual_outputs, reference_outputs, _output_scales(plan)
    )
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
    _assert_output_parity(
        actual_outputs, reference_outputs, _output_scales(plan)
    )
    print(f"PASS {case.name} memory-contract")
    return plan_path


def _output_scales(plan: dict) -> list[float]:
    """Quantization step of each model output, or 0 where the plan is not quantized.

    A float interface over a quantized plan is compared after dequantization.
    Both sides are then multiples of this step, so the comparison stays in
    whole steps rather than in floats that cannot represent the step exactly.
    """
    scales: list[float] = []
    for tensor_index in plan["model_outputs"]:
        tensor = plan["tensors"][tensor_index]
        if _declared_dtype(tensor) == tensor["dtype"]:
            scales.append(0.0)
            continue
        quant = plan["quant_params"][tensor["quant_param_idx"]]
        scales.append(float(quant["scale"]))
    return scales


def _assert_output_parity(
    actual_outputs: list[Array],
    reference_outputs: list[Array],
    output_scales: list[float] | None = None,
) -> None:
    output_scales = output_scales or [0.0] * len(actual_outputs)
    for actual, expected, scale in zip(
        actual_outputs, reference_outputs, output_scales
    ):
        if scale > 0.0:
            # The runtime uses integer half-away-from-zero quantization while
            # this ONNX Runtime QDQ reference follows a different half-tie
            # rule. Keep the same one-LSB acceptance bound used by benchmark
            # validation, while rejecting any larger contract drift.
            steps = np.abs(np.rint((actual - expected) / scale))
            worst = float(steps.max()) if steps.size else 0.0
            if worst > _INT8_LSB_TOLERANCE:
                raise AssertionError(
                    f"dequantized output differs by {worst:.0f} quantization "
                    f"steps, bound is {_INT8_LSB_TOLERANCE}"
                )
        elif np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        else:
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
        _add_relu_fusion_case(),
        _residual_case(),
        _layout_mixing_residual_case(square=True),
        _layout_mixing_residual_case(square=False),
        _banded_attention_case(),
        _head_permutation_case(),
        _reshape_alias_case(spatial=False),
        _reshape_alias_case(spatial=True),
        _token_bias_case(),
        _flattened_token_head_case(),
        _constant_divisor_case(),
        _traced_shape_scale_case(),
        _unfolded_feature_map_case(),
        _split_case(),
        _reduce_mean_case(quantized=False, keepdims=True),
        _reduce_mean_case(quantized=False, keepdims=False),
        _reduce_mean_case(quantized=True, keepdims=True),
        _reduce_mean_case(quantized=True, keepdims=False),
        _output_transpose_case(),
        _dilated_conv_case(),
        _depthwise_conv_case(),
        _math_normalization_case(),
        _conv1d_case(),
        _qdq_conv1d_case(),
        _rank3_pointwise_case(),
        _many_stage_case(),
        _tcn_16k_case(),
        _reduce_case(maximum=False),
        _reduce_case(maximum=True),
        _inference_identity_case(),
        _channel_bias_add_case(producer_has_bias=False),
        _channel_bias_add_case(producer_has_bias=True),
        _float_gemm_bias_add_case(),
        _batched_matmul_case(),
        _dynamic_matmul_case(batched=False),
        _dynamic_matmul_case(batched=True),
        _softmax_axis_case(rank=3, last_axis=True),
        _softmax_axis_case(rank=3, last_axis=False),
        _softmax_axis_case(rank=4, last_axis=True),
        _softmax_axis_case(rank=4, last_axis=False),
        _tiled_softmax_case(),
        _subtract_case(),
        _qdq_subtract_case(),
        _global_max_pool_case(),
        _sub_constant_case(),
        _clip_as_relu_case(),
        _negate_case(),
        _gemm_scaled_case(),
        _tiled_softmax_rank4_case(),
        _tiled_rank4_binary_case(op_type="Add"),
        _tiled_rank4_binary_case(op_type="Sub"),
        _tiled_rank4_binary_case(op_type="Mul"),
        _chained_normalization_case(),
        _chain_pointwise_before_spatial_case(),
        _chain_skip_out_of_a_stage_case(),
        _qdq_max_pool_rescale_case(),
        _qdq_rescale_passthrough_case(op_type="Relu"),
        _qdq_rescale_passthrough_case(op_type="Relu6"),
        _qdq_rescale_passthrough_case(op_type="Reshape"),
        _qdq_rescale_passthrough_case(op_type="Resize"),
        _tiled_last_axis_softmax_case(rank=3),
        _tiled_last_axis_softmax_case(rank=4),
        _layer_norm_case(),
        _erf_case(),
        _qdq_erf_case(),
        _qdq_layer_norm_case(),
        _tiled_layer_norm_case(),
        _tiled_erf_case(),
        _tiled_matrix_rows_case(),
        _tiled_attention_transpose_case(),
        _normalized_classifier_case(),
        _resize_concat_case(),
        _tiled_pool_case(),
        _tiled_pool_chain_case(),
        _tiled_global_reduction_case(),
        _qdq_tiled_global_reduction_case(),
        _tiled_chain_case(compression="lz4"),
        _tiled_chain_case(xip=True),
        _qdq_case("Conv"),
        _qdq_case("AveragePool"),
        _qdq_add_relu_case(),
        _qdq_gemm_bias_add_case(),
        _matmul_case(),
        _gemm_no_transpose_case(),
        _qdq_matmul_case(),
        _quint8_activation_case(),
        _convtranspose_case(),
        _conv_then_convtranspose_case(),
        _convtranspose_overlap_case(),
        _convtranspose_output_padding_case(),
        _convtranspose_asymmetric_pad_case(),
        _qdq_case("ConvTranspose"),
        _qdq_convtranspose_per_channel_case(),
        _linebuffer_conv_chain_case(),
        _qdq_conv_chain_case(),
        _2d_tiled_conv_case(),
        _qdq_2d_tiled_conv_case(),
        _2d_tiled_conv_sigmoid_case(),
        _cotiled_concat_2d_case(),
        _qdq_cotiled_concat_2d_case(),
        _cotiled_add_2d_case(),
        _convtranspose_2d_tiled_case(),
        _qdq_convtranspose_2d_tiled_case(),
        _convtranspose_2d_partial_edge_case(),
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
