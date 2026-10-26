"""Cases that stream through stage chains."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _qparams, _scalars, _vi


def _split_chain_case(*, quantized: bool) -> ContractCase:
    """A strided stem feeding wide layers, too big to stream as one chain.

    Three stride-2 convolutions shrink the map to 8x8 and three wide layers
    follow. At the budget the whole run has no tile height that fits, so it
    streams as consecutive chains, each writing only its last output.
    """
    rng = np.random.default_rng(29)
    layers = [(3, 8, 3, 2), (8, 16, 3, 2), (16, 16, 3, 2), (16, 64, 1, 1), (64, 64, 3, 1), (64, 8, 1, 1)]
    data = rng.uniform(0.0, 1.0, size=(1, 3, 64, 64)).astype(np.float32)
    initializers, nodes = [], []
    previous = "input"
    if quantized:
        initializers += _scalars(inp=(1 / 255, -128))
        nodes += _qdq("input", "inp_s", "inp_z", "x")
        previous = "x"
    for index, (cin, cout, kernel, stride) in enumerate(layers):
        weight = (rng.normal(size=(cout, cin, kernel, kernel)) / np.sqrt(cin * kernel * kernel)).astype(np.float32)
        initializers.append(numpy_helper.from_array(weight, f"w{index}"))
        weight_name = f"w{index}"
        if quantized:
            initializers += _scalars(**{f"w{index}q": (float(np.abs(weight).max()) / 127, 0),
                                        f"a{index}": (0.02, -128)})
            nodes += _qdq(f"w{index}", f"w{index}q_s", f"w{index}q_z", f"w{index}dq")
            weight_name = f"w{index}dq"
        last = index == len(layers) - 1
        nodes += [
            helper.make_node("Conv", [previous, weight_name], [f"c{index}"], kernel_shape=[kernel, kernel],
                             strides=[stride, stride], pads=[kernel // 2] * 4),
            helper.make_node("Relu", [f"c{index}"], [f"r{index}"]),
        ]
        previous = f"r{index}"
        if quantized:
            out = "output" if last else f"q{index}"
            nodes += _qdq(previous, f"a{index}_s", f"a{index}_z", out)
            previous = out
    nodes[-1].output[0] = "output"
    label = f"{'int8' if quantized else 'float'}_split_chain"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8, 8])],
        initializers,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Conv",) * len(layers), mem_budget="3K" if quantized else "12K",
                        expect_tiled=True, expect_chain=True, expect_line_buffered=True)


def _chain_pointwise_before_spatial_case() -> ContractCase:
    """A chain stage whose first op comes before its convolution.

    An op ahead of the convolution has to compute the halo rows of its own
    output, since the convolution reads them as data. A tile context set at
    the stage's output height leaves those rows unwritten: a silent wrong
    answer on interior rows.

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
    emits, starting on a different row. Spilled at the stage's output range,
    the tensor lands shifted in slow memory from the second tile onward, and
    the later Concat reads it.

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


def _cotiled_gate_chain_case() -> ContractCase:
    """A chain that carries one operator's output past the next operator.

    A gate reads the tensor it gates, so the Mul takes both the convolution's
    output and the Sigmoid of it. A chain that stops at an operator with a
    second operand leaves both of those tensors in slow memory.
    Carried instead, they are streamed a tile at a time and never reach it:
    the slow budget here is what the pair costs when the chain carries them,
    and a compiler that stops chaining at the Mul needs half again as much.
    """
    c, s = 8, 24
    rng = np.random.default_rng(0)
    first = (rng.normal(size=(c, c, 3, 3)) * 0.2).astype(np.float32)
    second = (rng.normal(size=(c, c, 3, 3)) * 0.2).astype(np.float32)
    model = _model(
        "cotiled_gate_chain",
        [
            helper.make_node(
                "Conv", ["input", "first"], ["gated"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Sigmoid", ["gated"], ["gate"]),
            helper.make_node("Mul", ["gated", "gate"], ["silu"]),
            helper.make_node(
                "Conv", ["silu", "second"], ["output"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, s, s])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, c, s, s])],
        initializers=[
            numpy_helper.from_array(first, "first"),
            numpy_helper.from_array(second, "second"),
        ],
    )
    data = (rng.normal(size=(1, c, s, s)) * 0.7).astype(np.float32)
    return ContractCase(
        "float_cotiled_gate_chain",
        model,
        model,
        {"input": data},
        ("Conv", "Sigmoid", "Mul", "Conv"),
        mem_budget="16K",
        slow_budget="36K",
        expect_tiled=True,
        expect_chain=True,
        expect_line_buffered=True,
    )


def _chained_normalization_case() -> ContractCase:
    """A chain whose second stage normalizes.

    A chain is a run of stripe-tileable stages, so anything the stripe
    contract admits has to survive being chained. The budget is what makes
    the compiler cut and chain: the same graph at a roomy budget is one
    stage and exercises none of this.
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


def _three_conv_chain(graph_name: str, height: int, **conv) -> onnx.ModelProto:
    """Three padded 3x3 Convs, 3 -> 4 -> 4 -> 8 channels, on a [1, 3, height, 64] input."""
    rng = np.random.default_rng(7)
    weights = [
        numpy_helper.from_array(rng.normal(0.0, 0.05, size=shape).astype(np.float32), name)
        for name, shape in (("w0", (4, 3, 3, 3)), ("w1", (4, 4, 3, 3)), ("w2", (8, 4, 3, 3)))
    ]
    nodes = [
        helper.make_node("Conv", [source, weight], [out], pads=[1, 1, 1, 1], **conv)
        for source, weight, out in (("input", "w0", "mid0"), ("mid0", "w1", "mid1"),
                                    ("mid1", "w2", "output"))
    ]
    return _model(graph_name, nodes, [_vi("input", [1, 3, height, 64])],
                  [_vi("output", [1, 8, height, 64])], weights)


def _tiled_chain_case(*, compression: str | None = None, xip: bool = False) -> ContractCase:
    """Three padded Conv stages force the streamable-chain executor.

    The convolutions take their 3x3 kernel from the weights, so the chain
    recomputes a halo and runs line-buffered.
    """
    model = _three_conv_chain("tiled_chain", 64)
    suffix = "lz4" if compression else "xip"
    return ContractCase(
        f"float_tiled_chain_{suffix}", model, model,
        {"input": np.linspace(-1.0, 1.0, 12288, dtype=np.float32).reshape(1, 3, 64, 64)},
        ("Conv", "Conv", "Conv"), mem_budget="32K", compression=compression, xip=xip,
        expect_tiled=True, expect_chain=True, expect_line_buffered=True,
    )


def _linebuffer_conv_chain_case() -> ContractCase:
    """Padded Conv chain with a partial last band under either buffer bound."""
    model = _three_conv_chain("linebuffer_conv_chain", 67, kernel_shape=[3, 3])
    return ContractCase(
        "float_linebuffer_conv_chain", model, model,
        {"input": np.linspace(-1.0, 1.0, 12864, dtype=np.float32).reshape(1, 3, 67, 64)},
        ("Conv", "Conv", "Conv"), mem_budget="32K", expect_tiled=True, expect_chain=True,
        expect_line_buffered=True, recompute_metric=True,
    )


def _qdq_conv_chain_case() -> ContractCase:
    """Int8 padded chain with a partial last band and recomputation checks."""
    initializers = _qparams("act_scale", "act_zero_point", [0.25], [0])
    nodes = _qdq("input", "act_scale", "act_zero_point", "act_0", "input_q")
    rng = np.random.default_rng(11)
    channels = ((4, 3), (4, 4), (8, 4))
    for index, (out_channels, in_channels) in enumerate(channels):
        scale, zero = f"w_scale{index}", f"w_zero{index}"
        initializers += [
            numpy_helper.from_array(
                rng.normal(0.0, 0.05, size=(out_channels, in_channels, 3, 3)).astype(np.float32),
                f"w{index}"),
            *_qparams(scale, zero, [0.02], [0]),
        ]
        activation = "output" if index == len(channels) - 1 else f"act_{index + 1}"
        nodes += [
            *_qdq(f"w{index}", scale, zero, f"w_dq{index}", f"w_q{index}"),
            helper.make_node("Conv", [f"act_{index}", f"w_dq{index}"], [f"conv{index}"],
                             pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            *_qdq(f"conv{index}", "act_scale", "act_zero_point", activation, f"act_q{index}"),
        ]
    model = _model("qdq_conv_chain", nodes, [_vi("input", [1, 3, 67, 64])],
                   [_vi("output", [1, 8, 67, 64])], initializers)
    return ContractCase(
        "int8_linebuffer_conv_chain", model, model,
        {"input": np.linspace(-1.0, 1.0, 3 * 67 * 64, dtype=np.float32).reshape(1, 3, 67, 64)},
        ("Conv", "Conv", "Conv"), mem_budget="8K", expect_tiled=True, expect_chain=True,
        expect_line_buffered=True, recompute_metric=True,
    )
