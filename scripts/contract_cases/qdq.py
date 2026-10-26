"""Cases for quantized (QDQ) graphs."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars


def _qdq_bilinear_upsample_case() -> ContractCase:
    """Int8 bilinear upsampling in bands, with different input and output scales.

    The four samples are mixed in the real domain and requantized, so an
    output scale that differed from the input's and went unapplied would be
    off by far more than rounding.
    """
    channels, side = 4, 8
    rng = np.random.default_rng(23)
    data = rng.uniform(-2.0, 2.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = _scalars(inp=(0.02, 3), out=(0.013, -5)) + [
        numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
    ]
    nodes = [
        *_qdq("input", "inp_s", "inp_z", "x"),
        helper.make_node("Resize", ["x", "roi", "scales"], ["raw"], mode="linear"),
        *_qdq("raw", "out_s", "out_z", "output"),
    ]
    model = _model(
        "int8_bilinear_upsample", nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 2 * side, 2 * side])],
        initializers,
    )
    return ContractCase("int8_bilinear_upsample", model, copy.deepcopy(model), {"input": data},
                        ("ResizeLinear",), mem_budget="512", expect_tiled=True)


def _quantized_batchnorm_case() -> ContractCase:
    """A quantized Conv followed by an unfolded float BatchNormalization.

    The BN sits between the Conv and the output quantization with a mixed-sign
    gamma, so the fold has to move it into the int8 weights, their per-channel
    scales and the int32 bias instead of dropping it.
    """
    rng = np.random.default_rng(31)
    channels = 4
    weight = (rng.normal(size=(channels, 3, 3, 3)) * 0.3).astype(np.float32)
    bias = np.round(rng.normal(size=channels) * 400).astype(np.int32)
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(bias, "bias_q"),
        numpy_helper.from_array(np.array(0.02 * 0.01, dtype=np.float32), "b_s"),
        numpy_helper.from_array(np.array(0, dtype=np.int32), "b_z"),
        numpy_helper.from_array(np.array([1.5, -0.75, 0.5, -2.0], dtype=np.float32), "gamma"),
        numpy_helper.from_array(np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32), "beta"),
        numpy_helper.from_array(np.array([0.05, -0.1, 0.0, 0.2], dtype=np.float32), "mean"),
        numpy_helper.from_array(np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32), "var"),
    ] + _scalars(inp=(0.02, 0), w=(0.01, 0), out=(0.04, 3))
    nodes = [
        *_qdq("input", "inp_s", "inp_z", "x"),
        *_qdq("weight", "w_s", "w_z", "wdq"),
        helper.make_node("DequantizeLinear", ["bias_q", "b_s", "b_z"], ["bdq"]),
        helper.make_node("Conv", ["x", "wdq", "bdq"], ["raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["raw", "gamma", "beta", "mean", "var"], ["normed"]),
        *_qdq("normed", "out_s", "out_z", "output"),
    ]
    model = _model(
        "int8_conv_batchnorm", nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 8, 8])],
        initializers,
    )
    data = rng.uniform(-1.0, 1.0, size=(1, 3, 8, 8)).astype(np.float32)
    return ContractCase("int8_conv_batchnorm", model, copy.deepcopy(model), {"input": data}, ("Conv",))


def _quantized_boundary_case() -> ContractCase:
    """A model that takes and returns uint8, the way a camera pipeline does.

    The input is dequantized straight off the graph input and the output is
    the QuantizeLinear itself, with no float anywhere at the boundary. Both
    fold onto int8 tensors whose declared interface stays uint8.
    """
    rng = np.random.default_rng(24)
    weight = (rng.normal(size=(4, 3, 3, 3)) * 0.3).astype(np.float32)
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "img_s"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "img_z"),
        numpy_helper.from_array(np.array(0.02, dtype=np.float32), "out_s"),
        numpy_helper.from_array(np.array(100, dtype=np.uint8), "out_z"),
    ] + _scalars(w=(0.01, 0))
    nodes = [
        helper.make_node("DequantizeLinear", ["image", "img_s", "img_z"], ["x"]),
        *_qdq("weight", "w_s", "w_z", "wdq"),
        helper.make_node("Conv", ["x", "wdq"], ["raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["raw", "out_s", "out_z"], ["mask"]),
    ]
    model = _model(
        "uint8_boundary", nodes,
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 4, 8, 8])],
        initializers,
    )
    data = rng.integers(0, 256, size=(1, 3, 8, 8)).astype(np.uint8)
    return ContractCase("uint8_boundary", model, copy.deepcopy(model), {"image": data}, ("Conv",))


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


def _qdq_rescale_passthrough_case(*, op_type: str) -> ContractCase:
    """A value-preserving int8 operator whose output rescales.

    Relu, Relu6, Reshape and nearest Resize all carry a value through
    unchanged, so each has to requantize when its output encoding differs
    from its input's.
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

    A maximum preserves the value, not its encoding. Written in the input's
    encoding, a pool whose quantizer assigned a different output scale comes
    back wrong by tens of steps with no error.
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


def _qdq_gelu_case(approximate: str, shape: list[int], *, two_dimensional=False) -> ContractCase:
    case = _qdq_erf_case()
    model = copy.deepcopy(case.compile_model)
    model.opset_import[0].version = 20
    for value in (*model.graph.input, *model.graph.output):
        value.type.tensor_type.shape.ClearField("dim")
        for extent in shape:
            value.type.tensor_type.shape.dim.add().dim_value = extent
    for node in model.graph.node:
        if node.op_type == "Erf":
            node.op_type = "Gelu"
            node.attribute.append(helper.make_attribute("approximate", approximate))
    onnx.checker.check_model(model)
    data = np.linspace(-2.0, 2.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return ContractCase(
        f"int8_gelu_{approximate}_{len(shape)}d_{int(two_dimensional)}", model, model,
        {"input": data}, ("Gelu",), mem_budget="128", slow_budget="32K", expect_tiled=True,
        expect_2d=two_dimensional, exact_untiled=True)


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


def _qdq_float_section_case(kind: str, terminal: bool, *, tiled: bool = False) -> ContractCase:
    T = TensorProto
    def q(name, s, zp=0, dt=np.int8):
        return [numpy_helper.from_array(np.array(s, np.float32), name + "_s"),
                numpy_helper.from_array(np.array(zp, dt), name + "_z")]
    def qdq(x, name):
        return [helper.make_node("QuantizeLinear", [x, name + "_s", name + "_z"], [x + "_q"]),
                helper.make_node("DequantizeLinear", [x + "_q", name + "_s", name + "_z"], [x + "_dq"])]
    def build(kind, terminal):
        rng = np.random.default_rng(0)
        inits = q("io", 0.25) + q("w", 0.05) + q("o2", 0.25)
        if kind == "gemm":
            X = helper.make_tensor_value_info("input", T.FLOAT, [1, 16])
            w = rng.integers(-60, 60, (8, 16)).astype(np.int8)
            prod = [helper.make_node("Gemm", ["input_dq", "w_dq"], ["p"], transB=1)]
            bias = rng.normal(0, 1, 8).astype(np.float32)
            oshape = [1, 8]
        else:
            X = helper.make_tensor_value_info("input", T.FLOAT, [1, 4, 8, 8])
            w = rng.integers(-60, 60, (8, 4, 3, 3)).astype(np.int8)
            prod = [helper.make_node("Conv", ["input_dq", "w_dq"], ["p"], pads=[1, 1, 1, 1])]
            bias = rng.normal(0, 1, (1, 8, 1, 1)).astype(np.float32)
            oshape = [1, 8, 8, 8]
        inits += [numpy_helper.from_array(w, "wq"), numpy_helper.from_array(bias, "b")]
        nodes = qdq("input", "io") + [helper.make_node("DequantizeLinear", ["wq", "w_s", "w_z"], ["w_dq"])] + prod + qdq("p", "io")
        nodes.append(helper.make_node("Add", ["p_dq", "b"], ["a" if not terminal else "output"]))
        if not terminal:
            nodes += [helper.make_node("QuantizeLinear", ["a", "o2_s", "o2_z"], ["a_q"]),
                      helper.make_node("DequantizeLinear", ["a_q", "o2_s", "o2_z"], ["a_dq"]),
                      helper.make_node("Relu", ["a_dq"], ["r"]),
                      helper.make_node("QuantizeLinear", ["r", "o2_s", "o2_z"], ["r_q"]),
                      helper.make_node("DequantizeLinear", ["r_q", "o2_s", "o2_z"], ["output"])]
        Y = helper.make_tensor_value_info("output", T.FLOAT, oshape)
        m = helper.make_model(helper.make_graph(nodes, "g", [X], [Y], inits), opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m

    model = build(kind, terminal)
    shape = [1, 16] if kind == "gemm" else [1, 4, 8, 8]
    values = np.random.default_rng(8041).normal(size=shape).astype(np.float32)
    operators = ("Gemm" if kind == "gemm" else "Conv", "Dequantize", "Add")
    if not terminal:
        operators += ("Quantize", "Relu")
    return ContractCase(
        f"qdq_{kind}_{'terminal' if terminal else 'inner'}_float_section{'_tiled' if tiled else ''}",
        model, model, {"input": values}, operators,
        mem_budget="512" if tiled else "128K", slow_budget="8M",
        expect_tiled=tiled, expect_chain=tiled, expect_line_buffered=tiled, reference_unoptimized=True,
    )


def _qdq_gemm_bias_add_case() -> ContractCase:
    """A float Add executes after the product rounds and clips."""
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
        ("Gemm", "Dequantize", "Add"),
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
