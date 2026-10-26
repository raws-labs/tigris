"""Unary, binary, activation, comparison and normalization cases."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars, _vi


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


def _constant_divisor_case() -> ContractCase:
    """A division by a constant, which is how an exporter writes a GELU.

    ``gelu`` leaves ``Erf(x / sqrt(2))`` behind and an attention scale leaves a
    division by the head width. Div has no opcode, but a constant divisor is
    the Mul the kernels already carry, so the graph is rewritten rather than
    refused. Both shapes a Mul takes are here: a scalar and a whole tensor.

    The scalar is read by two divisions, which is what an exporter leaves
    behind once a deduplicating pass gives every GELU in a stack of blocks the
    same initializer. Inverting that constant in place inverts it once per
    reader, so the second division would multiply by sqrt(2) instead of
    dividing by it: a factor of two off.
    """
    channels, side = 4, 8
    shape = [1, channels, side, side]
    nodes = [
        helper.make_node("Div", ["input", "root_two"], ["scaled"]),
        helper.make_node("Erf", ["scaled"], ["shaped"]),
        helper.make_node("Div", ["shaped", "whole"], ["widened"]),
        helper.make_node("Div", ["widened", "root_two"], ["output"]),
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
        ("Mul", "Erf", "Mul", "Mul"),
    )


def _per_channel_constant_case() -> ContractCase:
    """An input normalization, which is one constant value per channel.

    An exporter writes the mean and the standard deviation into the graph as
    (1, C, 1, 1) constants. The wire format gives a constant operand a byte
    size and nothing else, so the runtime reads that length as one value per
    channel and repeats it: channels are stored innermost, which is what makes
    the pattern hold wherever a tile starts.
    """
    channels, side = 3, 12
    rng = np.random.default_rng(0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    deviation = np.array(
        [0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    weight = (rng.normal(size=(4, channels, 3, 3)) * 0.2).astype(np.float32)
    model = _model(
        "per_channel_constant",
        [
            helper.make_node("Sub", ["input", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "deviation"], ["scaled"]),
            helper.make_node(
                "Conv", ["scaled", "weight"], ["output"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 4, side, side])],
        initializers=[
            numpy_helper.from_array(mean, "mean"),
            numpy_helper.from_array(deviation, "deviation"),
            numpy_helper.from_array(weight, "weight"),
        ],
    )
    data = rng.random((1, channels, side, side)).astype(np.float32)
    return ContractCase(
        "float_per_channel_constant",
        model,
        model,
        {"input": data},
        ("Add", "Mul", "Conv"),
    )


def _elementwise_nodes(kind: str, a: str, b: str | None,
                       output: str) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """The ONNX nodes, and their initializers, that the compiler reads as one `kind` operator."""
    if kind == "Rsqrt":
        return [helper.make_node("Sqrt", [a], ["root"]),
                helper.make_node("Reciprocal", ["root"], [output])], []
    if kind == "Square":
        return [helper.make_node("Mul", [a, a], [output])], []
    if kind == "SquaredDifference":
        return [helper.make_node("Sub", [a, b], ["difference"]),
                helper.make_node("Mul", ["difference", "difference"], [output])], []
    if kind == "FloorDiv":
        return [helper.make_node("Div", [a, b], ["quotient"]),
                helper.make_node("Floor", ["quotient"], [output])], []
    if kind == "FloorMod":
        nodes = [helper.make_node("Mod", [a, b], ["remainder"], fmod=1),
                 helper.make_node("Equal", ["remainder", "zero"], ["is_zero"]),
                 helper.make_node("Not", ["is_zero"], ["nonzero"]),
                 helper.make_node("Less", [b, "zero"], ["negative_b"]),
                 helper.make_node("Less", ["remainder", "zero"], ["negative_r"]),
                 helper.make_node("Xor", ["negative_b", "negative_r"], ["different"]),
                 helper.make_node("And", ["nonzero", "different"], ["adjust"]),
                 helper.make_node("Add", ["remainder", b], ["shifted"]),
                 helper.make_node("Where", ["adjust", "shifted", "remainder"], [output])]
        return nodes, [numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")]
    return [helper.make_node(kind, [a] if b is None else [a, b], [output])], []


def _elementwise_case(kind: str, *, quantized: bool = False) -> ContractCase:
    shape = [1, 4, 16, 16]
    binary = kind in {"Div", "SquaredDifference", "Max", "Min", "FloorDiv", "FloorMod"}
    rng = np.random.default_rng(917)
    data = rng.uniform(-3, 3, size=shape).astype(np.float32)
    if kind in {"Rsqrt", "Sqrt", "Log"}:
        data = np.abs(data) + np.float32(0.5)
    other = rng.uniform(0.5, 3, size=shape).astype(np.float32)
    other.reshape(-1)[::2] *= -1
    inputs = {"input": data}
    if binary:
        inputs["second"] = other
    nodes = []
    initializers = []
    a, b = "input", "second"
    if quantized:
        if kind == "Div":
            initializers += _scalars(a=(0.125, 0), b=(0.25, 0), y=(0.125, 0))
            data.reshape(-1)[0] = 0.0
            other.reshape(-1)[0] = 19 * 0.25
        else:
            initializers += _scalars(a=(0.03125, -17), b=(0.03125, -17), y=(0.03125, -17))
        nodes += _qdq(a, "a_s", "a_z", "a_float")
        a = "a_float"
        if binary:
            nodes += _qdq(b, "b_s", "b_z", "b_float")
            b = "b_float"
    output = "raw" if quantized else "output"
    more_nodes, more_initializers = _elementwise_nodes(kind, a, b if binary else None, output)
    nodes += more_nodes
    initializers += more_initializers
    if quantized:
        nodes += _qdq("raw", "y_s", "y_z", "output")
    label = f"{'int8' if quantized else 'float'}_elementwise_{kind.lower()}"
    model = _model(label, nodes,
                   [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in inputs],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)], initializers)
    return ContractCase(label, model, copy.deepcopy(model), inputs, (kind,),
                        mem_budget="256", expect_tiled=True)


def _bool_case(kind: str, quantized: bool = False, tiled: bool = False, variant: int = 0) -> ContractCase:
    shape = [] if variant == 4 else [99, 7] if variant == 5 else [1, 3, 231] if variant == 6 else [1, 3, 33, 7] if tiled else [1, 2, 3, 4]
    comparisons = {"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"}
    arity = 3 if kind in {"Where", "Sum"} else 2 if kind in comparisons | {"And", "Or"} else 1
    shapes = [shape] * arity
    if variant == 1:
        if kind == "Where":
            shape = [2, 2, 3, 4, 2]
            shapes = [[2, 1, 3, 1, 2], [1, 2, 1, 4, 1], [1, 1, 1, 1, 2]]
        else:
            shapes = [[1, 2, 1, 4], [1, 1, 3, 1]]
    elif variant == 2:
        shapes[-1] = [4]
    elif variant >= 10:
        shapes[variant - 10] = []
    nodes, initializers, infos, values, operands = [], [], [], {}, []
    for i in range(arity):
        name = f"input{i}"
        boolean = kind in {"And", "Or", "Not", "Cast"} or kind == "Where" and i == 0
        dtype = TensorProto.BOOL if boolean else TensorProto.INT8 if quantized else TensorProto.FLOAT
        data = (np.arange(np.prod(shapes[i], dtype=int)).reshape(shapes[i]) * (i + 1) + i) % 7 - 3
        values[name] = np.asarray((data > 0) if boolean else data.astype(np.int8 if quantized else np.float32))
        if variant == i + 10:
            initializers.append(numpy_helper.from_array(values.pop(name), name))
        else:
            infos.append(helper.make_tensor_value_info(name, dtype, shapes[i]))
        source = name
        if quantized and not boolean:
            initializers += _scalars(**{f"q{i}": (0.125, 0)})
            source = f"dequant{i}"
            nodes.append(helper.make_node("DequantizeLinear", [name, f"q{i}_s", f"q{i}_z"], [source]))
        operands.append(source)
    boolean_output = kind in comparisons | {"And", "Or", "Not"}
    raw = "output" if boolean_output or not quantized else "raw"
    attrs = {"to": TensorProto.FLOAT} if kind == "Cast" else {}
    nodes.append(helper.make_node(kind, operands, [raw], **attrs))
    if quantized and not boolean_output:
        initializers += _scalars(y=(1.0 if kind == "Cast" else 0.125, 0))
        nodes.append(helper.make_node("QuantizeLinear", [raw, "y_s", "y_z"], ["output"]))
    dtype = TensorProto.BOOL if boolean_output else TensorProto.INT8 if quantized else TensorProto.FLOAT
    label = f"bool_{kind.lower()}_s8{int(quantized)}_tiled{int(tiled)}_v{variant}"
    model = _model(label, nodes, infos, [helper.make_tensor_value_info("output", dtype, shape)], initializers, opset=17)
    return ContractCase(label, model, copy.deepcopy(model), values,
                        (("Transpose",) * (3 if kind == "Where" else 2) + (kind,)) if variant == 2 else (kind,),
                        mem_budget="512" if tiled else "4K", slow_budget="32K" if tiled else None,
                        expect_tiled=tiled, reference_unoptimized=True, exact=True)


def _bool_composition_case(quantized: bool, tiled: bool) -> ContractCase:
    case = _bool_case("Where", quantized, tiled)
    model = case.compile_model
    where = next(node for node in model.graph.node if node.op_type == "Where")
    operands = list(where.input[1:])
    del model.graph.input[0]
    values = {key: value for key, value in case.inputs.items() if key != "input0"}
    nodes = list(model.graph.node)
    position = nodes.index(where)
    nodes[position:position] = [helper.make_node("Equal", operands, ["equal"]),
                                helper.make_node("Not", ["equal"], ["input0"])]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return ContractCase(case.name + "_composed", model, copy.deepcopy(model), values,
                        ("Equal", "Not", "Where"), mem_budget=case.mem_budget,
                        slow_budget=case.slow_budget, expect_tiled=tiled, expect_chain=tiled, reference_unoptimized=True,
                        exact=True)


def _activation_case(kind: str, *, quantized: bool = False) -> ContractCase:
    shape = [2, 4, 16, 16]
    output_shape = shape
    data = np.random.default_rng(193).uniform(-2, 2, size=shape).astype(np.float32)
    nodes, initializers = [], []
    source = "input"
    if quantized:
        output_quant = {"LogSoftmax": (1 / 16, 127),
                        "L2Normalization": (1 / 128, 0)}.get(kind, (0.03125, -17))
        initializers += _scalars(a=(0.03125, -17), b=(0.0625, 3), y=output_quant)
        nodes += _qdq(source, "a_s", "a_z", "x")
        source = "x"
    target = "raw" if quantized else "output"
    if kind == "PRelu":
        initializers.append(numpy_helper.from_array(
            np.array([-0.5, 0.0, 0.25, 1.5], np.float32).reshape(4, 1, 1), "alpha"))
        alpha = "alpha"
        if quantized:
            nodes += _qdq(alpha, "b_s", "b_z", "slope")
            alpha = "slope"
        nodes += [helper.make_node(kind, [source, alpha], [target])]
    elif kind == "L2Pool":
        output_shape = [2, 4, 8, 8]
        nodes += [helper.make_node("Mul", [source, source], ["square"]),
                  helper.make_node("AveragePool", ["square"], ["mean"],
                                   kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 0, 0]),
                  helper.make_node("Sqrt", ["mean"], [target])]
    elif kind == "L2Normalization":
        nodes += [helper.make_node("LpNormalization", [source], [target], p=2, axis=1)]
    else:
        attrs = {"axis": 1} if kind == "LogSoftmax" else {"alpha": -0.5} if kind == "LeakyRelu" else {}
        nodes += [helper.make_node(kind, [source], [target], **attrs)]
    if quantized:
        nodes += _qdq(target, "y_s", "y_z", "output")
    label = f"{'int8' if quantized else 'float'}_activation_{kind.lower()}"
    model = _model(label, nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)], initializers)
    return ContractCase(label, model, copy.deepcopy(model), {"input": data}, (kind,),
                        mem_budget="2K" if kind == "L2Pool" else "512", expect_tiled=True,
                        reference_unoptimized=True)


def _hardswish_case(*, quantized: bool) -> ContractCase:
    """HardSwish, MobileNetV3's activation, banded behind a convolution.

    The convolution's outputs span roughly [-4, 4], so the sample covers the
    zero region below -3, the curved middle and the identity above 3.
    """
    channels, side = 8, 16
    rng = np.random.default_rng(21)
    weight = (rng.normal(size=(channels, channels, 3, 3)) * 0.25).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), conv=(0.035, 0), hs=(0.03, -40))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            *_qdq("raw", "conv_s", "conv_z", "act_in"),
            helper.make_node("HardSwish", ["act_in"], ["act"]),
            *_qdq("act", "hs_s", "hs_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("HardSwish", ["raw"], ["output"]),
        ]
    label = f"{'int8' if quantized else 'float'}_hardswish"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side, side])],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Conv", "HardSwish"), mem_budget="2K", expect_tiled=True,
                        # int8 rows are small enough that the solver streams the
                        # pair as a line-buffered chain, which bands HardSwish
                        # the other way.
                        expect_chain=quantized, expect_line_buffered=quantized)


def _constant_operand_case(*, op_type: str, quantized: bool, extent: str,
                           first: bool) -> ContractCase:
    """A binary operator with a constant operand behind a convolution: one
    value, one per channel, or one per element, on either side. The first two
    run banded; one per element carries no row offset and runs whole."""
    channels, side = 4, 12
    rng = np.random.default_rng(25)
    weight = (rng.normal(size=(channels, channels, 3, 3)) * 0.25).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    shape = {"scalar": (), "channels": (1, channels, 1, 1),
             "full": (1, channels, side, side)}[extent]
    constant = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
    if op_type == "Div":
        # A divisor near zero would magnify the convolution's rounding.
        weight = np.abs(weight) + 0.05
        data = np.abs(data) + 0.5
    initializers = [numpy_helper.from_array(weight, "weight")]
    conv = dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    operands = ["k", "c"] if first else ["c", "k"]
    if quantized:
        initializers += [numpy_helper.from_array(
            np.clip(np.round(constant / 0.01), -127, 127).astype(np.int8), "constant")]
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), conv=(0.03, 0), k=(0.01, 0),
                                 out=(0.05, -3))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "conv_s", "conv_z", "c"),
            helper.make_node("DequantizeLinear", ["constant", "k_s", "k_z"], ["k"]),
            helper.make_node(op_type, operands, ["y"]),
            *_qdq("y", "out_s", "out_z", "output"),
        ]
    else:
        initializers.append(numpy_helper.from_array(constant, "k"))
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["c"], **conv),
            helper.make_node(op_type, operands, ["output"]),
        ]
    label = (f"{'int8' if quantized else 'float'}_{op_type.lower()}_constant_{extent}"
             f"_{'first' if first else 'second'}")
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side, side])],
        initializers, opset=14,
    )
    tiled = extent != "full"
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Conv", op_type),
                        mem_budget=("1K" if quantized else "2K") if tiled else "16K",
                        expect_tiled=tiled,
                        # The rows are small enough to stream the pair as a
                        # line-buffered chain.
                        expect_chain=tiled, expect_line_buffered=tiled)


def _broadcast_case(*, op_type: str, quantized: bool, other_shape: list[int],
                    first: bool, constant: bool = False) -> ContractCase:
    """A binary operator whose operands broadcast to a larger output, behind a
    convolution. Any broadcast but one over leading stored axes, or a tensor
    operand that repeats ahead of the full one, runs untiled."""
    channels, side = 4, 12
    rng = np.random.default_rng(26)
    weight = (rng.normal(size=(channels, channels, 3, 3)) * 0.25).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    other = rng.uniform(0.5, 1.5, size=other_shape).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])]
    feeds = {"input": data}
    conv = dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    operands = ["o", "c"] if first else ["c", "o"]
    if constant and quantized:
        initializers.append(numpy_helper.from_array(
            np.clip(np.round(other / 0.01), -127, 127).astype(np.int8), "other"))
    elif constant:
        initializers.append(numpy_helper.from_array(other, "other"))
    else:
        inputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, other_shape))
        feeds["other"] = other
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), conv=(0.03, 0), o=(0.01, 0),
                                 out=(0.05, -3))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "conv_s", "conv_z", "c"),
            (helper.make_node("DequantizeLinear", ["other", "o_s", "o_z"], ["o"]) if constant
             else _qdq("other", "o_s", "o_z", "o")),
            helper.make_node(op_type, operands, ["y"]),
            *_qdq("y", "out_s", "out_z", "output"),
        ]
        nodes = [n for item in nodes for n in (item if isinstance(item, list) else [item])]
    else:
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["c"], **conv),
            helper.make_node("Identity", ["other"], ["o"]),
            helper.make_node(op_type, operands, ["output"]),
        ]
    shape_tag = "x".join(str(d) for d in other_shape)
    label = (f"{'int8' if quantized else 'float'}_{op_type.lower()}_broadcast_{shape_tag}"
             f"{'_constant' if constant else ''}_{'first' if first else 'second'}")
    model = _model(
        label, nodes, inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side, side])],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), feeds, ("Conv", op_type),
                        mem_budget="16K")


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


def _math_normalization_case() -> ContractCase:
    """Constant folding, Clip->Relu6, unary math, Mul, and Softmax."""
    nodes = [
        helper.make_node("Constant", [], ["clip_min"],
                         value=numpy_helper.from_array(np.array(0.0, dtype=np.float32))),
        helper.make_node("Constant", [], ["clip_max"],
                         value=numpy_helper.from_array(np.array(6.0, dtype=np.float32))),
        helper.make_node("Clip", ["input", "clip_min", "clip_max"], ["clipped"]),
        helper.make_node("Tanh", ["clipped"], ["tanh"]),
        helper.make_node("Sigmoid", ["clipped"], ["sigmoid"]),
        helper.make_node("Mul", ["tanh", "sigmoid"], ["product"]),
        helper.make_node("Softmax", ["product"], ["output"], axis=1),
    ]
    model = _model("math_normalization", nodes, [_vi("input", [1, 4])], [_vi("output", [1, 4])])
    return ContractCase(
        "float_math_normalization", model, model,
        {"input": np.array([[-3.0, 0.25, 2.0, 9.0]], dtype=np.float32)},
        ("Relu6", "Tanh", "Sigmoid", "Mul", "Softmax"),
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


def _legacy_softmax_case(*, shape: list[int], axis: int | None,
                         operators: tuple[str, ...]) -> ContractCase:
    """Softmax from opset 11, which normalizes everything from `axis` on.

    The older operator flattens the axes from `axis` (default 1) into one row,
    unlike opset 13's single axis, so the compiler has to keep that meaning.
    """
    attrs = {} if axis is None else {"axis": axis}
    model = _model(
        "legacy_softmax",
        [helper.make_node("Softmax", ["input"], ["output"], **attrs)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        opset=11,
    )
    data = np.linspace(-2.0, 2.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    label = f"float_opset11_softmax_rank{len(shape)}_{'default' if axis is None else axis}_axis"
    return ContractCase(label, model, model, {"input": data}, operators)


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
    """Neg preserves the sign reversal after a rectifier."""
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
        ("Relu", "Neg"),
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


def _gelu_case(approximate: str, shape: list[int], budget: str, *, two_dimensional=False) -> ContractCase:
    model = _model(
        "gelu", [helper.make_node("Gelu", ["input"], ["output"], approximate=approximate)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)], opset=20)
    values = np.linspace(-8.0, 8.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return ContractCase(
        f"float_gelu_{approximate}_{len(shape)}d_{budget}", model, model, {"input": values}, ("Gelu",),
        mem_budget=budget, slow_budget="32K", expect_tiled=True,
        expect_2d=two_dimensional, exact_untiled=True)


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


def _rank2_dense_constant_case(op_type: str) -> ContractCase:
    """A rank-2 binary operator against a constant of its own full shape, banded.

    A row band hands a constant to the kernel whole, so a constant that differs
    from row to row takes the leading band, which offsets it to each band's
    rows. Less writes bool from a float constant and Where reads a bool
    condition, so the constant's element differs from the output's. The banded
    plan has to equal the one-pass plan bit for bit.
    """
    rows, cols = 128, 16
    rng = np.random.default_rng(11)
    label = f"rank2_dense_constant_{op_type.lower()}"
    if op_type == "Where":
        constant = rng.random((rows, cols)) > 0.5
        node = helper.make_node(op_type, ["constant", "input", "other"], ["output"])
    else:
        constant = rng.uniform(-2, 2, size=(rows, cols)).astype(np.float32)
        node = helper.make_node(op_type, ["input", "constant"], ["output"])
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [rows, cols])]
    data = {"input": rng.uniform(-2, 2, size=(rows, cols)).astype(np.float32)}
    if op_type == "Where":
        inputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [rows, cols]))
        data["other"] = rng.uniform(-2, 2, size=(rows, cols)).astype(np.float32)
    output = TensorProto.BOOL if op_type == "Less" else TensorProto.FLOAT
    model = _model(
        label,
        [node],
        inputs,
        [helper.make_tensor_value_info("output", output, [rows, cols])],
        [numpy_helper.from_array(constant, "constant")],
    )
    return ContractCase(f"float_{label}", model, model, data, (op_type,),
                        mem_budget="4K", slow_budget="64K", expect_tiled=True, exact_untiled=True)


def _rank2_row_band_case(kind: str, *, constant: bool = False) -> ContractCase:
    """A rank-2 binary operator the row band cuts, against a second input or one row of constants.

    The banded plan has to equal the one-pass plan bit for bit.
    """
    rows, cols = 128, 16
    rng = np.random.default_rng(13)
    data = {"input": rng.uniform(-3, 3, size=(rows, cols)).astype(np.float32)}
    second = rng.uniform(0.5, 3, size=(cols,) if constant else (rows, cols)).astype(np.float32)
    second.reshape(-1)[::2] *= -1
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [rows, cols])]
    initializers = []
    if constant:
        initializers.append(numpy_helper.from_array(second, "second"))
    else:
        inputs.append(helper.make_tensor_value_info("second", TensorProto.FLOAT, [rows, cols]))
        data["second"] = second
    nodes, more = _elementwise_nodes(kind, "input", "second", "output")
    label = f"rank2_row_band_{kind.lower()}{'_constant' if constant else ''}"
    model = _model(label, nodes, inputs,
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, cols])],
                   initializers + more)
    return ContractCase(f"float_{label}", model, model, data, (kind,),
                        mem_budget="4K", slow_budget="64K", expect_tiled=True, exact_untiled=True)
