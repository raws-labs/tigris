"""Fail-closed validation for operators outside the binary plan schema."""

import json
import numpy as np
import onnx
import pytest
from click import ClickException
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import (
    validate_execution_dtype,
    validate_model_outputs,
    validate_operator_support,
)
from tigris.inspection import inspect_file
from tigris.capabilities import KERNEL_CAPABILITIES, effective_operators
from tigris.cli import _run_pipeline, cli
from tigris.emitters.binary.defs import OP_TYPE_MAP
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import AnalyzedGraph, OpNode, QuantParam, Stage, TensorInfo, TilePlan


@pytest.fixture
def acos_model_path(tmp_path):
    """A valid one-op ONNX model whose operator is not in OP_TYPE_MAP."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    node = helper.make_node(
        "Acos", ["input"], ["output"], name="unsupported_acos"
    )
    graph = helper.make_graph(
        [node], "unsupported_acos", [model_input], [model_output]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = tmp_path / "unsupported_acos.onnx"
    onnx.save(model, path)
    return path


@pytest.fixture
def mixed_dtype_model_path(tmp_path):
    """A float branch that no Dequantize feeds cannot run in an int8 plan."""
    quant_input = helper.make_tensor_value_info(
        "quant_input", TensorProto.FLOAT, [1, 4]
    )
    float_input = helper.make_tensor_value_info(
        "float_input", TensorProto.FLOAT, [1, 4]
    )
    quant_output = helper.make_tensor_value_info(
        "quant_output", TensorProto.FLOAT, [1, 4]
    )
    float_output = helper.make_tensor_value_info(
        "float_output", TensorProto.FLOAT, [1, 4]
    )
    scale = numpy_helper.from_array(np.array([0.1], np.float32), "scale")
    zero_point = numpy_helper.from_array(np.array([0], np.int8), "zero_point")
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["quant_input", "scale", "zero_point"],
            ["quantized_input"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["quantized_input", "scale", "zero_point"],
            ["dequantized_input"],
        ),
        helper.make_node(
            "Relu", ["dequantized_input"], ["quantized_relu"]
        ),
        helper.make_node(
            "QuantizeLinear",
            ["quantized_relu", "scale", "zero_point"],
            ["quantized_output"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["quantized_output", "scale", "zero_point"],
            ["quant_output"],
        ),
        helper.make_node("Relu", ["float_input"], ["float_output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "mixed_dtype",
        [quant_input, float_input],
        [quant_output, float_output],
        [scale, zero_point],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "mixed_dtype.onnx"
    onnx.save(model, path)
    return path


def test_normalized_unknown_operator_is_reported(acos_model_path):
    graph, _ = _run_pipeline(str(acos_model_path), ("4K",))

    validation = validate_operator_support(graph)
    findings = compute_findings(graph)

    assert not validation.supported
    assert [issue.describe() for issue in validation.issues] == [
        "unsupported_acos (Acos)"
    ]
    assert findings.verdict == "needs_work"
    assert findings.unsupported_operators == ["unsupported_acos (Acos)"]


def test_analyze_lists_unsupported_operator_as_failure(acos_model_path):
    result = CliRunner().invoke(
        cli, ["analyze", str(acos_model_path), "-m", "4K"]
    )

    assert result.exit_code == 0, result.output
    assert "cannot compile" in result.output
    assert "unsupported_acos (Acos)" in result.output


def test_compile_rejects_unsupported_operator_without_output(
    acos_model_path, tmp_path
):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(acos_model_path), "-m", "4K", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "cannot compile" in result.output
    assert "unsupported_acos (Acos)" in result.output
    assert not output.exists()


def test_writer_rejects_unsupported_operator(acos_model_path):
    graph, _ = _run_pipeline(str(acos_model_path), ("4K",))

    with pytest.raises(
        ValueError,
        match="Cannot emit a plan with unsupported operators: unsupported_acos",
    ):
        emit_binary_bytes(graph)


def test_mixed_activation_dtypes_are_never_reported_deployable(
    mixed_dtype_model_path,
):
    graph, _ = _run_pipeline(str(mixed_dtype_model_path), ("4K",))

    validation = validate_execution_dtype(graph)
    findings = compute_findings(graph)

    assert not validation.supported
    assert validation.issues == (
        "float32 activations in an int8 plan must follow a Dequantize, If or "
        "While: float_input, float_output",
    )
    assert findings.verdict == "needs_work"
    assert findings.dtype_errors
    with pytest.raises(
        ValueError, match="Cannot emit a plan with unsupported tensor dtypes"
    ):
        emit_binary_bytes(graph)


def test_compile_preserves_existing_output_on_dtype_rejection(
    mixed_dtype_model_path, tmp_path
):
    output = tmp_path / "existing.tgrs"
    output.write_bytes(b"sentinel")

    result = CliRunner().invoke(
        cli,
        ["compile", str(mixed_dtype_model_path), "-m", "4K", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "cannot compile" in result.output
    assert output.read_bytes() == b"sentinel"


def _constant_output_model(tmp_path, source):
    """A model that also returns a tensor normalization folds into a constant."""
    n = helper.make_node
    f32 = lambda name, value: numpy_helper.from_array(np.asarray(value, np.float32), name)  # noqa: E731
    x, y = (helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 8]) for name in ("x", "y"))
    if source == "constant":
        nodes = [n("Constant", [], ["k"], value=f32("v", np.full((1, 8), 0.5))), n("Add", ["x", "k"], ["y"])]
        returned, init = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 8]), []
    elif source == "shape":
        nodes = [n("Shape", ["x"], ["k"]), n("Relu", ["x"], ["r"]), n("Reshape", ["r", "k"], ["y"])]
        returned, init = helper.make_tensor_value_info("k", TensorProto.INT64, [2]), []
    else:
        int8 = lambda name, value: numpy_helper.from_array(np.asarray(value, np.int8), name)  # noqa: E731
        nodes = [n("QuantizeLinear", ["x", "s", "z"], ["xq"]), n("DequantizeLinear", ["xq", "s", "z"], ["xd"]),
                 n("DequantizeLinear", ["wq", "ws", "z"], ["k"]), n("Gemm", ["xd", "k"], ["g"], transB=1),
                 n("QuantizeLinear", ["g", "s", "z"], ["gq"]), n("DequantizeLinear", ["gq", "s", "z"], ["y"])]
        returned = helper.make_tensor_value_info("k", TensorProto.FLOAT, [8, 8])
        init = [f32("s", 0.05), int8("z", 0), int8("wq", np.arange(64).reshape(8, 8) - 32), f32("ws", 0.01)]
    model = helper.make_model(helper.make_graph(nodes, source, [x], [returned, y], init),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / f"{source}.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("source", ["constant", "shape", "dequantized_weight"])
def test_compile_refuses_an_output_no_operator_computes(tmp_path, source):
    """The plan holds no constant as an output, so it would return one output fewer."""
    path = _constant_output_model(tmp_path, source)
    graph, _ = _run_pipeline(str(path), ("4K",))
    issue = "output 'k' is not computed by any operator"
    assert validate_model_outputs(graph).issues == (issue,)
    assert validate_execution_dtype(graph).supported
    findings = compute_findings(graph)
    assert (findings.output_errors, findings.dtype_errors) == ([issue], [])
    with pytest.raises(ValueError, match=f"Cannot emit a plan that omits a declared output: {issue}"):
        emit_binary_bytes(graph)

    report = CliRunner().invoke(cli, ["analyze", str(path), "-m", "4K", "--json"])
    model = json.loads(report.output)["model"]
    assert (model["output_errors"], model["dtype_errors"]) == ([issue], [])
    output = tmp_path / "should-not-exist.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "4K", "-o", str(output)])
    assert result.exit_code != 0
    assert f"cannot compile: {issue}" in result.output
    assert not output.exists()


def test_an_output_no_operator_computes_leaves_dtype_errors_reported(mixed_dtype_model_path):
    graph, _ = _run_pipeline(str(mixed_dtype_model_path), ("4K",))
    graph.model_outputs.append("scale")
    findings = compute_findings(graph)
    assert findings.output_errors == ["output 'scale' is not computed by any operator"]
    assert findings.dtype_errors == [
        "float32 activations in an int8 plan must follow a Dequantize, If or While: "
        "float_input, float_output"]
    assert findings.verdict_text == findings.output_errors[0]


def test_ceil_mode_whose_last_window_lies_in_the_padding_is_rejected(tmp_path):
    """Kernel 1, stride 3 over five rows: rounding up adds a window that starts
    past the input, which no end padding can state."""
    node = helper.make_node("MaxPool", ["input"], ["output"], kernel_shape=[1, 1], strides=[3, 3], ceil_mode=1)
    model = helper.make_model(helper.make_graph(
        [node], "ceil_past_input", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])]),
        opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    path = tmp_path / "ceil_past_input.onnx"
    onnx.save(model, path)
    graph, _ = _run_pipeline(str(path), ("4K",))
    assert "ceil_mode=1 is not encoded" in validate_operator_support(graph).describe()


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"count_include_pad": 1}, "count_include_pad=1 with padding is not encoded"),
        ({"dilations": [2, 2]}, "pooling dilation is not implemented"),
        ({"auto_pad": "SAME_UPPER"}, "requires explicit pads"),
    ],
)
def test_unrepresentable_pool_attributes_are_rejected(
    tmp_path, attrs, message
):
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 3, 3]
    )
    node = helper.make_node(
        "AveragePool",
        ["input"],
        ["output"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        **attrs,
    )
    model = helper.make_model(
        helper.make_graph(
            [node], "unsupported_pool_attrs", [model_input], [model_output]
        ),
        opset_imports=[helper.make_opsetid("", 19)],
    )
    model.ir_version = 9
    path = tmp_path / "unsupported_pool_attrs.onnx"
    onnx.save(model, path)
    graph, _ = _run_pipeline(str(path), ("4K",))

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert message in validation.describe()
    with pytest.raises(ValueError, match=message):
        emit_binary_bytes(graph)


@pytest.mark.parametrize(
    ("op_type", "attrs", "shapes", "message"),
    [
        ("Softmax", {"axis": 0}, ([2, 4], [2, 4]), "Softmax axis must map"),
        ("Concat", {"axis": 0}, ([1, 2, 3, 4], [1, 2, 3, 4], [2, 2, 3, 4]),
         "does not concatenate on the batch axis"),
        ("Resize", {"mode": "cubic", "coordinate_transformation_mode": "asymmetric"},
         ([1, 1, 2, 2], [1, 1, 4, 4]), "Resize mode=cubic is not encoded"),
    ],
)
def test_onnx_operator_variants_outside_the_encoding_are_not_emitted(
    tmp_path, op_type, attrs, shapes, message
):
    *input_shapes, output_shape = shapes
    names = [f"input{index}" for index in range(len(input_shapes))]
    initializers = []
    if op_type == "Resize":
        initializers.append(numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"))
        names += ["", "scales"]
    path = _save(
        tmp_path, op_type.lower(),
        [helper.make_node(op_type, names, ["output"], **attrs)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
         for name, shape in zip(names, input_shapes)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )
    graph, _ = _run_pipeline(str(path), ("4K",))

    with pytest.raises(ValueError, match=message):
        emit_binary_bytes(graph)


@pytest.mark.parametrize(
    ("op", "tensors", "message"),
    [
        (
            OpNode(
                name="wrong_softmax_axis",
                op_type="Softmax",
                inputs=["input"],
                outputs=["output"],
                attrs={"axis": 0},
            ),
            {
                "input": TensorInfo("input", (2, 4), TensorProto.FLOAT),
                "output": TensorInfo("output", (2, 4), TensorProto.FLOAT),
            },
            "Softmax axis must map",
        ),
        (
            OpNode(
                name="batch_concat",
                op_type="Concat",
                inputs=["left", "right"],
                outputs=["output"],
                attrs={"kernel_shape": [0]},
            ),
            {
                "left": TensorInfo("left", (1, 2, 3, 4), TensorProto.FLOAT),
                "right": TensorInfo("right", (1, 2, 3, 4), TensorProto.FLOAT),
                "output": TensorInfo("output", (2, 2, 3, 4), TensorProto.FLOAT),
            },
            "does not concatenate on the batch axis",
        ),
        (
            OpNode(
                name="linear_resize",
                op_type="Resize",
                inputs=["input"],
                outputs=["output"],
                attrs={
                    "mode": "linear",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            ),
            {
                "input": TensorInfo("input", (1, 3, 4, 4), TensorProto.FLOAT),
                "output": TensorInfo("output", (1, 3, 8, 8), TensorProto.FLOAT),
            },
            "Resize mode=linear is not encoded",
        ),
        (
            OpNode(
                name="half_pixel_resize",
                op_type="Resize",
                inputs=["input"],
                outputs=["output"],
                attrs={"mode": "nearest"},
            ),
            {
                "input": TensorInfo("input", (1, 3, 4, 4), TensorProto.FLOAT),
                "output": TensorInfo("output", (1, 3, 8, 8), TensorProto.FLOAT),
            },
            "must state a TFLite sampling convention",
        ),
        (
            OpNode(
                name="channel_resize",
                op_type="Resize",
                inputs=["input"],
                outputs=["output"],
                attrs={
                    "mode": "nearest",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            ),
            {
                "input": TensorInfo("input", (1, 3, 4, 4), TensorProto.FLOAT),
                "output": TensorInfo("output", (1, 4, 7, 8), TensorProto.FLOAT),
            },
            "unchanged N/C",
        ),
        *[
            (
                OpNode(
                    name="conv_transpose",
                    op_type="ConvTranspose",
                    inputs=["input"],
                    outputs=["output"],
                    attrs={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [0, 0, 0, 0], **attrs},
                ),
                {
                    "input": TensorInfo("input", (1, 4, 4, 4), TensorProto.FLOAT),
                    "output": TensorInfo("output", (1, 4, 9, 9), TensorProto.FLOAT),
                },
                message,
            )
            for attrs, message in (
                ({"group": 2}, "group=2 is not implemented (group=1 only)"),
                ({"dilations": [2, 2]}, "ConvTranspose dilation is not implemented"),
            )
        ],
    ],
)
def test_unencoded_operator_variants_are_rejected(op, tensors, message):
    validation = validate_operator_support(AnalyzedGraph(ops=[op], tensors=tensors))

    assert not validation.supported
    assert message in validation.describe()


@pytest.mark.parametrize("nearest_mode", ["floor", "round_prefer_floor"])
def test_non_nearest_resize_is_refused_for_its_mode_alone(nearest_mode):
    resize = OpNode(
        name="cubic_resize",
        op_type="Resize",
        inputs=["input"],
        outputs=["output"],
        attrs={"mode": "cubic", "coordinate_transformation_mode": "asymmetric",
               "nearest_mode": nearest_mode},
    )
    tensors = {
        "input": TensorInfo("input", (1, 3, 4, 4), TensorProto.FLOAT),
        "output": TensorInfo("output", (1, 3, 8, 8), TensorProto.FLOAT),
    }

    validation = validate_operator_support(AnalyzedGraph(ops=[resize], tensors=tensors))

    assert validation.describe() == "cubic_resize (Resize: Resize mode=cubic is not encoded)"


def test_representable_operator_variants_are_supported():
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="channel_softmax",
                op_type="Softmax",
                inputs=["softmax_input"],
                outputs=["softmax_output"],
                attrs={"axis": 1},
            ),
            OpNode(
                name="channel_concat",
                op_type="Concat",
                inputs=["left", "right"],
                outputs=["concat_output"],
                attrs={"kernel_shape": [3]},
            ),
            OpNode(
                name="nearest_resize",
                op_type="Resize",
                inputs=["resize_input"],
                outputs=["resize_output"],
                attrs={
                    "mode": "nearest",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            ),
            OpNode(
                name="conv_transpose",
                op_type="ConvTranspose",
                inputs=["deconv_input"],
                outputs=["deconv_output"],
                attrs={
                    "kernel_shape": [3, 3],
                    "strides": [2, 2],
                    "pads": [0, 0, 0, 0],
                    "dilations": [1, 1],
                    "group": 1,
                },
            ),
        ],
        tensors={
            "softmax_input": TensorInfo(
                "softmax_input", (1, 4, 2, 2), TensorProto.FLOAT
            ),
            "softmax_output": TensorInfo(
                "softmax_output", (1, 4, 2, 2), TensorProto.FLOAT
            ),
            "left": TensorInfo("left", (1, 2, 3, 4), TensorProto.FLOAT),
            "right": TensorInfo("right", (1, 2, 3, 5), TensorProto.FLOAT),
            "concat_output": TensorInfo(
                "concat_output", (1, 2, 3, 9), TensorProto.FLOAT
            ),
            "resize_input": TensorInfo(
                "resize_input", (1, 3, 4, 4), TensorProto.FLOAT
            ),
            "resize_output": TensorInfo(
                "resize_output", (1, 3, 8, 8), TensorProto.FLOAT
            ),
            "deconv_input": TensorInfo(
                "deconv_input", (1, 4, 4, 4), TensorProto.FLOAT
            ),
            "deconv_output": TensorInfo(
                "deconv_output", (1, 4, 9, 9), TensorProto.FLOAT
            ),
        },
    )

    validation = validate_operator_support(graph)

    assert validation.supported, validation.describe()


def test_encodable_operator_without_a_route_is_rejected():
    routed = frozenset().union(
        *(effective_operators(backend) for backend in KERNEL_CAPABILITIES)
    )
    unrouted = sorted(set(OP_TYPE_MAP) - routed)
    assert unrouted, "no opcode lacks a route; test the gate with a synthetic op_type"
    graph = AnalyzedGraph(
        ops=[OpNode(name="lone_op", op_type=unrouted[0], inputs=["input"], outputs=["output"])],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
    )

    result = validate_operator_support(graph)

    assert not result.supported
    assert any(
        issue.reason == f"{unrouted[0]} has no kernel in the runtime"
        or "unsupported" in issue.reason.lower()
        for issue in result.issues
    )


def test_dynamic_elementwise_broadcasting_is_rejected():
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="broadcast_add",
                op_type="Add",
                inputs=["left", "right"],
                outputs=["output"],
            )
        ],
        tensors={
            "left": TensorInfo("left", (2, 3), TensorProto.FLOAT),
            "right": TensorInfo("right", (3,), TensorProto.FLOAT),
            "output": TensorInfo("output", (2, 3), TensorProto.FLOAT),
        },
    )

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert "a tensor operand at the output's rank" in validation.describe()


@pytest.mark.parametrize("height,supported", [(2049, True), (4096, False)])
def test_bilinear_reference_coordinate_range(height, supported):
    quant = QuantParam(scale=np.array([0.125], np.float32), zero_point=np.array([-17], np.int8))
    graph = AnalyzedGraph(
        ops=[OpNode("resize", "ResizeLinear", ["input"], ["output"],
                    attrs={"coordinate_transformation_mode": "align_corners"})],
        tensors={"input": TensorInfo("input", (1, 1, 3, 1), TensorProto.INT8, quant=quant),
                 "output": TensorInfo("output", (1, 1, height, 1), TensorProto.INT8, quant=quant)})
    result = validate_operator_support(graph)
    assert result.supported == supported
    if not supported:
        assert "defined TFLite reference coordinate range" in result.describe()


def test_quantized_constant_operand_without_its_quantization_is_rejected():
    graph = AnalyzedGraph(
        is_quantized=True,
        ops=[
            OpNode(
                name="constant_mul",
                op_type="Mul",
                inputs=["input", "constant"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.INT8),
            "constant": TensorInfo(
                "constant", (1, 4), TensorProto.INT8, is_constant=True
            ),
            "output": TensorInfo("output", (1, 4), TensorProto.INT8),
        },
        weight_data={"constant": np.ones((1, 4), dtype=np.int8)},
    )

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert "is not per-tensor int8" in validation.describe()


def test_exact_constant_elementwise_operand_is_rejected_when_tiled():
    op = OpNode(
        name="tiled_constant_add",
        op_type="Add",
        inputs=["input", "constant"],
        outputs=["output"],
        stage=0,
    )
    graph = AnalyzedGraph(
        ops=[op],
        stages=[
            Stage(
                stage_id=0,
                op_indices=[0],
                tile_plan=TilePlan(tileable=True, tiled_peak_bytes=16),
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "constant": TensorInfo(
                "constant", (1, 4), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
        weight_data={"constant": np.ones((1, 4), dtype=np.float32)},
    )

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert "cannot be offset for tiled execution" in validation.describe()


def test_existing_full_shape_float_constant_add_is_supported(linear_3op_path):
    graph, _ = _run_pipeline(str(linear_3op_path), ("4K",))

    validation = validate_operator_support(graph)

    assert validation.supported, validation.describe()


def test_scalar_float_constant_elementwise_operand_is_supported():
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="scalar_mul",
                op_type="Mul",
                inputs=["input", "constant"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "constant": TensorInfo(
                "constant", (), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
        weight_data={"constant": np.array(2.0, dtype=np.float32)},
    )

    validation = validate_operator_support(graph)

    assert validation.supported, validation.describe()


def test_equal_numel_different_shape_constant_is_rejected():
    """Matching element counts is not a broadcast: a (2, 2) constant against a
    (1, 4) operand does not broadcast even though the counts agree."""
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="shape_broadcast_add",
                op_type="Add",
                inputs=["input", "constant"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "constant": TensorInfo(
                "constant", (2, 2), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
        weight_data={"constant": np.ones((2, 2), dtype=np.float32)},
    )

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert "does not broadcast to the output" in validation.describe()


def test_a_per_channel_constant_is_accepted():
    """One value per channel repeats on its own: channels are stored innermost.

    This is the shape an exporter writes an input normalization in, and it is
    what a (1, C, 1, 1) mean or scale looks like after right alignment.
    """
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="normalize",
                op_type="Mul",
                inputs=["input", "scale"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 3, 8, 8), TensorProto.FLOAT),
            "scale": TensorInfo(
                "scale", (1, 3, 1, 1), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 3, 8, 8), TensorProto.FLOAT),
        },
        weight_data={"scale": np.ones((1, 3, 1, 1), dtype=np.float32)},
    )

    assert validate_operator_support(graph).supported


def test_a_per_row_constant_is_accepted():
    """A per-row constant is a broadcast inside the operand, read by output
    coordinate."""
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="per_row",
                op_type="Mul",
                inputs=["input", "scale"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 3, 8, 8), TensorProto.FLOAT),
            "scale": TensorInfo(
                "scale", (1, 1, 8, 1), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 3, 8, 8), TensorProto.FLOAT),
        },
        weight_data={"scale": np.ones((1, 1, 8, 1), dtype=np.float32)},
    )

    assert validate_operator_support(graph).supported


def test_qoperator_model_names_the_format_and_the_fix(tmp_path):
    """A QOperator graph must say what it is and how to re-export it."""
    from tigris.analysis.validation import QOPERATOR_OP_TYPES
    from tigris.loaders.onnx.loader import load_model
    from tigris.loaders.onnx.normalize import normalize

    init = [
        numpy_helper.from_array(np.float32(0.02), "x_scale"),
        numpy_helper.from_array(np.int8(0), "x_zp"),
        numpy_helper.from_array(np.zeros((8, 4, 3, 3), np.int8), "W"),
        numpy_helper.from_array(np.float32(0.01), "w_scale"),
        numpy_helper.from_array(np.int8(0), "w_zp"),
        numpy_helper.from_array(np.float32(0.03), "y_scale"),
        numpy_helper.from_array(np.int8(0), "y_zp"),
        numpy_helper.from_array(np.zeros(8, np.int32), "B"),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["x", "x_scale", "x_zp", "W", "w_scale", "w_zp", "y_scale", "y_zp", "B"],
        ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="qconv1")
    graph = helper.make_graph(
        [node], "qop",
        [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 4, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.INT8, [1, 8, 16, 16])],
        init)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "qoperator.onnx"
    onnx.save(model, str(path))

    ag = normalize(load_model(str(path)))
    assert "QLinearConv" in QOPERATOR_OP_TYPES

    support = validate_operator_support(ag)
    assert not support.supported
    described = support.describe()
    assert "QOperator" in described
    assert "QuantFormat.QDQ" in described

    # The dtype check must not also report a contradictory float32-vs-int8
    # complaint: on a QOperator graph that is a symptom of the same cause.
    assert validate_execution_dtype(ag).issues == ()


def test_non_qoperator_dtype_mismatch_is_still_reported():
    """Suppression is scoped to QOperator graphs, not to dtype mismatch."""
    ag = AnalyzedGraph(
        ops=[OpNode(name="relu1", op_type="Relu", inputs=["a"], outputs=["b"])],
        tensors={
            "a": TensorInfo(name="a", shape=(1, 4), dtype=3),
            "b": TensorInfo(name="b", shape=(1, 4), dtype=3),
        },
        model_inputs=["a"],
        model_outputs=["b"],
    )
    ag.is_quantized = False
    issues = validate_execution_dtype(ag).issues
    assert issues and "quantization metadata" in issues[0]


def _save(tmp_path, name, nodes, inputs, outputs, initializers=(), opset=13):
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, list(initializers)),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / f"{name}.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def _avg_pool_path(tmp_path, pads):
    return _save(
        tmp_path, "avg_pool",
        [helper.make_node("AveragePool", ["input"], ["output"],
                          kernel_shape=[3, 3], pads=pads, count_include_pad=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 6, 6])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT,
            [1, 2, 6, 6] if any(pads) else [1, 2, 4, 4])],
    )


def test_count_include_pad_without_padding_is_accepted(tmp_path):
    """Without padding every window lies inside the input: the flag is moot."""
    ag, _ = _run_pipeline(str(_avg_pool_path(tmp_path, [0, 0, 0, 0])), ("64K",),
                          report_bindings=False)
    assert emit_binary_bytes(ag)


def _gate_path(tmp_path, gate_shape):
    return _save(
        tmp_path, "gate",
        [helper.make_node("Mul", ["gate", "input"], ["output"])],
        [helper.make_tensor_value_info("gate", TensorProto.FLOAT, gate_shape),
         helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 5, 5])],
    )


def test_per_channel_dynamic_operand_is_accepted(tmp_path):
    """A squeeze-and-excitation gate: one value per channel, written first."""
    ag, _ = _run_pipeline(str(_gate_path(tmp_path, [1, 4, 1, 1])), ("64K",),
                          report_bindings=False)
    mul = next(op for op in ag.ops if op.op_type == "Mul")
    assert mul.inputs == ["input", "gate"], "full-shape operand goes first"
    assert emit_binary_bytes(ag)


def test_a_row_wise_dynamic_operand_is_emitted(tmp_path):
    """A row-wise operand is a broadcast inside the operand, read by output
    coordinate and never tiled."""
    ag, _ = _run_pipeline(str(_gate_path(tmp_path, [1, 1, 5, 1])), ("64K",),
                          report_bindings=False)
    assert emit_binary_bytes(ag)


@pytest.mark.parametrize("kind", ["Exp", "FloorDiv"])
def test_elementwise_without_int8_kernel_rejects_quantized_plans(kind):
    binary = kind == "FloorDiv"
    quant = QuantParam(np.array([0.125], np.float32), np.array([-17], np.int8))
    ag = AnalyzedGraph(
        ops=[OpNode(name="math", op_type=kind,
                    inputs=["a", "b"] if binary else ["a"], outputs=["y"])],
        tensors={name: TensorInfo(name, (1, 4), 3, quant=quant) for name in ("a", "b", "y")},
        is_quantized=True,
    )
    assert f"{kind} has no int8 kernel in the runtime" in validate_operator_support(ag).describe()


def test_an_operator_without_a_float32_kernel_names_the_dtype(monkeypatch):
    # Every operator has a float32 kernel today; take Relu's away.
    from tigris.analysis import validation
    from tigris.capabilities import effective_operators
    monkeypatch.setattr(validation, "effective_operators", lambda backend: (
        effective_operators(backend) - {"Relu"} if backend == "reference"
        else effective_operators(backend)))
    ag = AnalyzedGraph(
        ops=[OpNode(name="op", op_type="Relu", inputs=["a"], outputs=["y"])],
        tensors={name: TensorInfo(name, (1, 4), 1) for name in ("a", "y")},
    )
    assert "Relu has no float32 kernel in the runtime" in validate_operator_support(ag).describe()


@pytest.mark.parametrize("kind", ["Abs", "Rsqrt", "SquaredDifference", "Max", "Min", "Div"])
def test_elementwise_int8_requires_matching_shapes_and_quantization(kind):
    binary = kind in {"SquaredDifference", "Max", "Min", "Div"}
    tensors = {
        name: TensorInfo(name, (1, 4), 3, quant=QuantParam(
            np.array([0.125], np.float32), np.array([-17], np.int8)))
        for name in ("a", "b", "y")
    }
    ag = AnalyzedGraph(
        ops=[OpNode(name="math", op_type=kind,
                    inputs=["a", "b"] if binary else ["a"], outputs=["y"])],
        tensors=tensors, is_quantized=True,
    )
    assert validate_operator_support(ag).supported
    if kind == "Div":
        tensors["b"].quant.scale[0] = 0.25
        tensors["b"].quant.zero_point[0] = 31
        tensors["y"].quant.scale[0] = 0.0625
        tensors["y"].quant.zero_point[0] = -63
        assert validate_operator_support(ag).supported
    tensors["y"].shape = (2, 2)
    assert "broadcast to the output" in validate_operator_support(ag).describe()
    tensors["y"].shape = (1, 4)
    if kind in {"Max", "Min"}:
        tensors["b"].quant.scale[0] = 0.25
        assert "identical input and output quantization" in validate_operator_support(ag).describe()
    tensors["a"].quant = None
    assert "per-tensor quantization" in validate_operator_support(ag).describe()


def test_operator_from_another_domain_is_not_the_standard_one(tmp_path):
    """A model-local Relu whose body is Neg must not compile as ONNX Relu."""
    function = helper.make_function(
        "custom", "Relu", ["x"], ["y"], [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 13)])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"], name="custom_relu", domain="custom")],
        "custom_domain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])])
    model = helper.make_model(
        graph, functions=[function],
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("custom", 1)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "custom_domain.onnx"
    onnx.save(model, path)
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "4K", "-o", str(output)])

    assert result.exit_code != 0
    assert "custom_relu (custom::Relu)" in result.output
    assert not output.exists()


def test_grouped_convolution_compiles_as_one_convolution_per_group(tmp_path):
    """group=2 over four channels is neither ordinary nor depthwise: a Split,
    one convolution per group and a Concat."""
    weight = np.ones((4, 2, 1, 1), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weight"], ["output"], name="grouped", group=2)],
        "grouped_conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 2, 2])],
        [numpy_helper.from_array(weight, "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "grouped_conv.onnx"
    onnx.save(model, path)
    output = tmp_path / "grouped_conv.tgrs"

    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "4K", "-o", str(output)])

    assert result.exit_code == 0, result.output
    plan = inspect_file(output)["plan"]
    assert [op["type"] for op in plan["ops"]] == ["Split", "Conv", "Conv", "Concat"]


@pytest.mark.parametrize("pools", [["-m", "16K", "-m", "1M", "-m", "1"], ["-m", "16K+1M+1"]])
def test_more_memory_pools_than_the_planner_has_are_rejected(acos_model_path, tmp_path, pools):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(cli, ["compile", str(acos_model_path), *pools, "-o", str(output)])

    assert result.exit_code != 0
    assert "3 memory pools given" in result.output
    assert not output.exists()

@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, float("nan")])
def test_elu_rejects_alpha_other_than_one(alpha):
    graph = AnalyzedGraph(
        ops=[OpNode("elu", "Elu", ["x"], ["y"], attrs={"alpha": alpha})],
        tensors={name: TensorInfo(name, (1, 4), 1) for name in ("x", "y")},
    )
    assert "Elu requires alpha=1" in validate_operator_support(graph).describe()
    graph.ops[0].attrs["alpha"] = 1.0
    assert validate_operator_support(graph).supported


@pytest.mark.parametrize("kind,scale,zero", [
    ("LogSoftmax", 1 / 16, 127), ("L2Normalization", 1 / 128, 0),
])
@pytest.mark.parametrize("invalid", ["scale", "zero_point", "missing", "per_channel"])
def test_quantized_normalization_requires_fixed_output_encoding(kind, scale, zero, invalid):
    output_quant = QuantParam(np.array([scale], np.float32), np.array([zero], np.int8))
    graph = AnalyzedGraph(
        ops=[OpNode("normalize", kind, ["x"], ["y"])],
        tensors={
            "x": TensorInfo("x", (1, 4), 3, quant=QuantParam(
                np.array([0.125], np.float32), np.array([-17], np.int8))),
            "y": TensorInfo("y", (1, 4), 3, quant=output_quant),
        }, is_quantized=True,
    )
    assert validate_operator_support(graph).supported
    if invalid == "scale":
        output_quant.scale[0] *= 2
    elif invalid == "zero_point":
        output_quant.zero_point[0] = zero - 1
    elif invalid == "missing":
        graph.tensors["y"].quant = None
    else:
        output_quant.scale = np.array([scale, scale], np.float32)
        output_quant.zero_point = np.array([zero, zero], np.int8)
    assert f"{kind} requires output scale" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("invalid", ["dynamic", "reversed", "rank"])
def test_prelu_requires_constant_alpha_after_rank_one_to_four_input(invalid):
    shape = (1, 4) if invalid != "rank" else (1, 1, 1, 1, 4)
    graph = AnalyzedGraph(
        ops=[OpNode("prelu", "PRelu", ["x", "alpha"], ["y"])],
        tensors={
            "x": TensorInfo("x", shape, 1), "y": TensorInfo("y", shape, 1),
            "alpha": TensorInfo("alpha", (4,), 1, is_constant=True),
        }, weight_data={"alpha": np.ones(4, np.float32)},
    )
    if invalid == "dynamic":
        graph.tensors["alpha"].is_constant = False
        graph.tensors["alpha"].shape = shape
        graph.weight_data.clear()
    elif invalid == "reversed":
        graph.ops[0].inputs.reverse()
    reason = "rank 1 to 4" if invalid == "rank" else "tensor followed by a constant alpha"
    assert reason in validate_operator_support(graph).describe()


@pytest.mark.parametrize("count_include_pad,other_operand", [(0, False), (1, False), (0, True)])
def test_l2pool_fold_preserves_pool_semantics(tmp_path, count_include_pad, other_operand):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 6, 6])]
    if other_operand:
        inputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [1, 2, 6, 6]))
    attrs = {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 0, 0, 1],
             "count_include_pad": count_include_pad}
    path = _save(tmp_path, "l2_pool", [
        helper.make_node("Mul", ["x", "other" if other_operand else "x"], ["squared"]),
        helper.make_node("AveragePool", ["squared"], ["pooled"], **attrs),
        helper.make_node("Sqrt", ["pooled"], ["y"]),
    ], inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])])
    graph, _ = _run_pipeline(str(path), ("64K",), report_bindings=False)
    folded = [op for op in graph.ops if op.op_type == "L2Pool"]
    if other_operand:
        assert not folded
        assert {"Mul", "AveragePool", "Sqrt"} <= {op.op_type for op in graph.ops}
        return
    assert len(folded) == 1
    assert folded[0].inputs == ["x"]
    assert all(folded[0].attrs[key] == value for key, value in attrs.items())
    if count_include_pad:
        with pytest.raises(ValueError, match="count_include_pad=1 with padding"):
            emit_binary_bytes(graph)
    else:
        assert validate_operator_support(graph).supported
        assert emit_binary_bytes(graph)


@pytest.mark.parametrize("order", [1, 2])
def test_lpnormalization_only_lowers_l2_without_epsilon_floor(tmp_path, order):
    path = _save(tmp_path, "lp_normalization", [
        helper.make_node("LpNormalization", ["x"], ["y"], p=order, axis=1),
    ], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])])
    graph, _ = _run_pipeline(str(path), ("4K",), report_bindings=False)
    if order == 2:
        op = next(op for op in graph.ops if op.op_type == "L2Normalization")
        assert op.attrs == {"axis": 1, "epsilon": 0.0}
        assert validate_operator_support(graph).supported
        assert emit_binary_bytes(graph)
    else:
        assert graph.ops[0].op_type == "LpNormalization"
        assert not validate_operator_support(graph).supported
        with pytest.raises(ValueError, match="unsupported operators"):
            emit_binary_bytes(graph)


@pytest.mark.parametrize("axis,flat_shape", [
    (None, (2, 60)), (0, (1, 120)), (1, (2, 60)), (2, (6, 20)),
    (-2, (6, 20)), (-1, (2, 3, 4, 5)),
])
def test_legacy_logsoftmax_normalizes_the_flattened_suffix(tmp_path, axis, flat_shape):
    attrs = {} if axis is None else {"axis": axis}
    shape = [2, 3, 4, 5]
    path = _save(tmp_path, "legacy_logsoftmax", [
        helper.make_node("LogSoftmax", ["x"], ["y"], name="normalize", **attrs),
    ], [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], opset=12)
    graph, _ = _run_pipeline(str(path), ("4K",), report_bindings=False)
    op = next(op for op in graph.ops if op.op_type == "LogSoftmax")
    assert op.attrs["axis"] == -1
    assert graph.tensors[op.inputs[0]].shape == flat_shape
    assert graph.tensors[op.outputs[0]].shape == flat_shape
    assert graph.tensors[graph.model_outputs[0]].shape == tuple(shape)
    if axis != -1:
        assert [node.op_type for node in graph.ops] == [
            "Transpose", "Reshape", "LogSoftmax", "Reshape", "Transpose",
        ]
        assert graph.ops[0].attrs["perm"] == graph.ops[-1].attrs["perm"] == [0, 1, 2, 3]
    assert validate_operator_support(graph).supported
    assert emit_binary_bytes(graph)


def _reduction_graph(kind, *, axes=(1,), shape=(2, 5, 3), keep=True, quantized=False):
    result = list(shape)
    if kind != "CumSum":
        if keep:
            result[1] = 1
        else:
            result.pop(1)
    quant = QuantParam(scale=np.array([0.125], np.float32), zero_point=np.array([0], np.int8))
    graph = AnalyzedGraph(
        ops=[OpNode(name="reduce", op_type=kind, inputs=["x"], outputs=["y"],
                    attrs={"axes": list(axes), "keepdims": int(keep)})],
        tensors={"x": TensorInfo(name="x", shape=shape, dtype=3 if quantized else 1,
                                 quant=quant if quantized else None),
                 "y": TensorInfo(name="y", shape=tuple(result), dtype=3 if quantized else 1,
                                 quant=quant if quantized else None)},
        model_inputs=["x"], model_outputs=["y"])
    graph.is_quantized = quantized
    return graph


def test_int8_activations_quantized_per_axis_are_refused_for_any_operator():
    graph = _reduction_graph("ReduceMax", quantized=True)
    graph.tensors["x"].quant = QuantParam(scale=np.array([0.125, 0.25], np.float32),
                                          zero_point=np.array([0, 0], np.int8))
    assert "'x' is not quantized per tensor" in validate_operator_support(graph).describe()


def test_int8_cumsum_off_a_zero_input_zero_point_needs_tflite_semantics():
    """ONNX's CumSum sums dequantized values; only the TFLite form, which seeds
    the sum with the scaled input zero point, runs off a zero point of 0."""
    graph = _reduction_graph("CumSum", quantized=True)
    graph.tensors["x"].quant = QuantParam(scale=np.array([0.125], np.float32),
                                          zero_point=np.array([-128], np.int8))
    assert "zero point 0" in validate_operator_support(graph).describe()
    graph.ops[0].attrs["tflite_seeded"] = 1
    assert validate_operator_support(graph).supported


@pytest.mark.parametrize("kind", ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "CumSum"])
@pytest.mark.parametrize("axes,shape", [([], (2, 5, 3)), ([0, 1], (2, 5, 3)),
                                       ([3], (2, 5, 3)), ([1], (2, 5, 3, 4))])
def test_reductions_reject_axes_and_ranks_outside_native_contract(kind, axes, shape):
    graph = _reduction_graph(kind, axes=axes, shape=shape)
    assert "rank-3" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("kind", ["ReduceMax", "ReduceMin", "ReduceSum", "CumSum"])
def test_reductions_reject_wrong_shapes_and_per_channel_quantization(kind):
    graph = _reduction_graph(kind, quantized=True)
    assert validate_operator_support(graph).supported
    graph.tensors["y"].shape = (2, 2, 3)
    assert "output shape" in validate_operator_support(graph).describe()
    graph = _reduction_graph(kind, quantized=True)
    graph.tensors["y"].quant = QuantParam(
        scale=np.array([0.125, 0.125], np.float32), zero_point=np.array([0, 0], np.int8))
    assert "per-tensor" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("kind", ["ReduceMax", "ReduceMin"])
def test_extrema_require_matching_int8_quantization(kind):
    graph = _reduction_graph(kind, quantized=True)
    graph.tensors["y"].quant = QuantParam(
        scale=np.array([0.25], np.float32), zero_point=np.array([0], np.int8))
    assert "identical" in validate_operator_support(graph).describe()


def test_cumsum_refuses_invalid_options():
    graph = _reduction_graph("CumSum", quantized=True)
    graph.tensors["y"].quant = QuantParam(
        scale=np.array([2**-22], np.float32), zero_point=np.array([0], np.int8))
    assert "smaller than one" in validate_operator_support(graph).describe()
    graph.ops[0].attrs["exclusive"] = 2
    assert "exclusive and reverse" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("input_dtype", [1, 3])
@pytest.mark.parametrize("output_dtype", [6, 7])
def test_index_outputs_do_not_bypass_dispatcher_dtype_contract(input_dtype, output_dtype):
    graph = _reduction_graph("ArgMax")
    graph.tensors["x"].dtype = input_dtype
    graph.tensors["y"].dtype = output_dtype
    graph.is_quantized = input_dtype == 3
    if output_dtype == 6:
        assert validate_execution_dtype(graph).supported
        assert validate_operator_support(graph).supported
    else:
        assert not validate_execution_dtype(graph).supported
        assert not validate_operator_support(graph).supported
    graph.ops.append(OpNode(name="consumer", op_type="Relu", inputs=["y"], outputs=["z"]))
    assert not validate_execution_dtype(graph).supported


def test_reducemax_does_not_rewrite_nonspatial_axes_to_global_pool(tmp_path):
    path = _save(tmp_path, "max_nonspatial", [
        helper.make_node("ReduceMax", ["x"], ["y"], axes=[1, 2], keepdims=1),
    ], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, 5])])
    graph, _ = _run_pipeline(str(path), ("4K",), report_bindings=False)
    assert any(op.op_type == "ReduceMax" for op in graph.ops)
    assert all(op.op_type != "GlobalMaxPool" for op in graph.ops)
    assert not validate_operator_support(graph).supported


@pytest.mark.parametrize("kind", ["ArgMax", "ArgMin"])
def test_arg_normalization_preserves_indices_and_refuses_last_index(tmp_path, kind):
    from tigris.loaders.onnx.normalize import _normalize_arg_outputs

    graph = _reduction_graph(kind)
    graph.tensors["y"].dtype = 7
    graph.ops[0].attrs = {"axis": -2, "keepdims": 1, "select_last_index": 1}
    with pytest.raises(ValueError, match="select_last_index"):
        _normalize_arg_outputs(graph)
    graph.ops[0].attrs["select_last_index"] = 0
    graph.ops.append(OpNode(name="consumer", op_type="Relu", inputs=["y"], outputs=["z"]))
    _normalize_arg_outputs(graph)
    assert graph.tensors["y"].dtype == 6
    assert not validate_execution_dtype(graph).supported
    graph.ops.pop()
    graph.tensors["x"].shape = (2, 2**31, 3)
    with pytest.raises(ValueError, match="int32"):
        _normalize_arg_outputs(graph)


@pytest.mark.parametrize("kind", ["Relu", "ReduceMax"])
def test_non_arg_integer_model_output_is_not_exempt(kind):
    graph = _reduction_graph(kind)
    graph.tensors["y"].dtype = 6
    assert not validate_execution_dtype(graph).supported


def test_auxiliary_dtype_placement_and_operator_slots(monkeypatch):
    from tigris.dtypes import DATA, DTypeSignature, OP_DTYPE_SIGNATURES, check_dtype_signatures

    monkeypatch.setitem(OP_DTYPE_SIGNATURES, "predicate", DTypeSignature(outputs=(9,)))
    monkeypatch.setitem(OP_DTYPE_SIGNATURES, "choose", DTypeSignature(inputs=(9, DATA, DATA)))
    tensors = [("x", 1, False, False), ("p", 9, False, False), ("y", 1, False, False)]
    ops = [("predicate", ["x"], ["p"]), ("choose", ["p", "x", "x"], ["y"])]
    data, issues = check_dtype_signatures(tensors, ops, ["x"], ["y"])
    assert data == {1: ["x", "y"]} and not issues
    assert check_dtype_signatures(tensors, [("Relu", ["p"], ["y"])], ["p"], ["y"])[1]
    assert check_dtype_signatures(tensors, [("choose", ["x", "p", "x"], ["y"])], ["x"], ["y"])[1]
    assert check_dtype_signatures([("p", 9, False, True)], [], ["p"], ["p"])[1]


def test_a_layout_step_that_changes_the_dtype_names_both():
    from tigris.dtypes import check_dtype_signatures
    tensors = [("x", 3, False, True), ("y", 1, False, False), ("i", 6, False, False),
               ("j", 6, False, False)]
    _, issues = check_dtype_signatures(tensors, [("Reshape", ["x"], ["y"])], ["x"], ["y"])
    assert "Reshape reads int8 and writes float32; it has to keep the dtype" in issues
    _, issues = check_dtype_signatures(tensors, [("Reshape", ["i"], ["j"])], ["i"], ["j"])
    assert not issues
    _, issues = check_dtype_signatures(tensors, [("Reshape", ["i"], ["y"])], ["i"], ["y"])
    assert "Reshape reads int32 and writes float32; it has to keep the dtype" in issues


def test_add_operands_and_output_are_named_by_the_operator():
    two = AnalyzedGraph(
        ops=[OpNode(name="add", op_type="Add", inputs=["a", "b", "c"], outputs=["y"])],
        tensors={name: TensorInfo(name, (1, 4), 1) for name in ("a", "b", "c", "y")},
    )
    assert "Add takes exactly two operands" in validate_operator_support(two).describe()
    unknown = AnalyzedGraph(
        ops=[OpNode(name="add", op_type="Add", inputs=["a", "b"], outputs=["y"])],
        tensors={name: TensorInfo(name, (1, 4), 1) for name in ("a", "b")},
    )
    assert ("Add needs one output with a known shape"
            in validate_operator_support(unknown).describe())


def _refusal(ops, tensors, weights=None):
    ag = AnalyzedGraph(ops=ops, tensors=tensors, weight_data=weights or {})
    return validate_operator_support(ag).describe()


def _floats(*names, shape=(1, 4, 3)):
    return {name: TensorInfo(name, shape, 1) for name in names}


@pytest.mark.parametrize("kind, inputs, message", [
    ("Abs", ["a", "b"], "Abs takes one operand that is not a constant and gives one output"),
    ("Div", ["a"], "Div takes two operands that are not constants and gives one output"),
])
def test_elementwise_operand_count_is_named_by_the_operator(kind, inputs, message):
    op = OpNode(name="op", op_type=kind, inputs=inputs, outputs=["y"])
    assert message in _refusal([op], _floats("a", "b", "y"))


def test_axis_reductions_name_the_operator():
    arg = OpNode(name="op", op_type="ArgMax", inputs=["a"], outputs=["y"], attrs={"axes": [0, 1]})
    assert "ArgMax needs one input of rank 1 to 6 and one constant axis" in _refusal(
        [arg], _floats("a", "y"))
    reduce = OpNode(name="op", op_type="ReduceMax", inputs=["a"], outputs=["y"], attrs={"axes": [1]})
    assert "ReduceMax needs one rank-3 input and one constant axis" in _refusal(
        [reduce], _floats("a", "y", shape=(1, 4)))


def test_softmax_of_a_scalar_names_the_operator():
    op = OpNode(name="op", op_type="Softmax", inputs=["a"], outputs=["y"])
    assert "Softmax needs one input with a known shape that is not a scalar" in _refusal(
        [op], _floats("a", "y", shape=()))


def test_layer_normalization_without_a_scale_is_refused_by_name():
    op = OpNode(name="op", op_type="LayerNormalization", inputs=["a"], outputs=["y"])
    assert "LayerNormalization needs a constant scale and an optional constant bias" in _refusal(
        [op], _floats("a", "y"))


def test_a_constant_operand_without_a_tensor_operand_is_refused_by_name():
    op = OpNode(name="op", op_type="Div", inputs=["a", "w"], outputs=["y"])
    tensors = {"w": TensorInfo("w", (1,), 1, is_constant=True), **_floats("y")}
    assert ("Div takes one operand that is not a constant and at most one constant"
            in _refusal([op], tensors, {"w": np.ones(1, np.float32)}))


def test_sum_of_unequal_shapes_is_refused_by_name():
    op = OpNode(name="op", op_type="Sum", inputs=["a", "b"], outputs=["y"])
    tensors = {**_floats("a", "y"), **_floats("b", shape=(1, 4, 1))}
    assert "Sum needs inputs and an output of one shape, none of them constant" in _refusal(
        [op], tensors)


def test_int32_operands_of_an_operator_go_together():
    from tigris.dtypes import check_dtype_signatures
    tensors = [("i", 6, False, False), ("one", 6, True, False), ("j", 6, False, False),
               ("x", 1, False, False), ("y", 1, False, False)]
    data, issues = check_dtype_signatures(tensors, [("Add", ["i", "one"], ["j"])], ["i"], ["j"])
    assert not issues and data == {1: ["x", "y"]}
    _, issues = check_dtype_signatures(tensors, [("Add", ["i", "x"], ["y"])], ["i", "x"], ["y"])
    assert any("mixes int32" in issue for issue in issues)
    _, issues = check_dtype_signatures(tensors, [("Relu", ["i"], ["j"])], ["i"], ["j"])
    assert issues


def test_float32_results_stand_apart_from_int8_data():
    """Detections are float32 in an int8 plan, and no operator reads them."""
    from tigris.dtypes import check_dtype_signatures
    tensors = [("x", 3, False, True), ("s", 3, False, True), ("b", 1, False, False),
               ("c", 1, False, False), ("y", 3, False, True)]
    detect = ("DetectionPostProcess", ["x", "s"], ["b", "c"])
    data, issues = check_dtype_signatures(tensors, [detect], ["x", "s"], ["b", "c"])
    assert data == {3: ["x", "s", "y"]} and not issues
    _, issues = check_dtype_signatures(tensors, [detect, ("Relu", ["b"], ["y"])], ["x", "s"], ["y"])
    assert any("must not be read" in issue for issue in issues)


@pytest.mark.parametrize("kind", ["StridedSlice", "MirrorPad", "ReverseV2"])
def test_movement_refuses_runtime_bounds(tmp_path, kind):
    from scripts.crossrepo_contract import _movement_case
    case = _movement_case(kind, False)
    model = case.compile_model
    index_name = {"StridedSlice": "starts", "MirrorPad": "pads", "ReverseV2": "axes",
                  "DynamicUpdateSlice": "starts"}.get(kind, "indices")
    initializer = next(item for item in model.graph.initializer if item.name == index_name)
    model.graph.input.append(helper.make_tensor_value_info(index_name, initializer.data_type, initializer.dims))
    model.graph.initializer.remove(initializer)
    path = tmp_path / "dynamic_indices.onnx"
    onnx.save(model, path)
    with pytest.raises(ClickException, match="constant indices or bounds"):
        _run_pipeline(str(path), ("8K",))


@pytest.mark.parametrize("kind,changes,message", [
    ("Gather", {"indices": [3, 0, 1]}, "outside its dimension"),
    ("GatherND", {"indices": [[0, 3], [0, 0], [1, 0]]}, "outside its dimension"),
    ("StridedSlice", {"steps": [1, 0, 2]}, "step is invalid"),
    ("MirrorPad", {"pads": [2, 0, 0, 0, 0, 0]}, "reflection domain"),
    ("ReverseV2", {"axes": [0, 2]}, "contiguous"),
    ("ReverseV2", {"axes": []}, "nonempty"),
])
def test_movement_refuses_invalid_constants(tmp_path, kind, changes, message):
    from scripts.crossrepo_contract import _movement_case
    case = _movement_case(kind, False)
    for name, value in changes.items():
        initializer = next(item for item in case.compile_model.graph.initializer if item.name == name)
        initializer.CopyFrom(numpy_helper.from_array(np.array(value, np.int64), name))
    path = tmp_path / "invalid_indices.onnx"
    onnx.save(case.compile_model, path)
    with pytest.raises(ClickException, match=message):
        _run_pipeline(str(path), ("8K",))


def _bool_graph(kind, quantized=False, count=None):
    comparisons = {"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"}
    count = count or (3 if kind in {"Where", "Sum"} else 2 if kind in comparisons | {"And", "Or"} else 1)
    inputs = [f"x{i}" for i in range(count)]
    tensors = {}
    for i, name in enumerate(inputs + ["y"]):
        boolean = (kind in comparisons | {"And", "Or", "Not"} if name == "y" else
                   kind in {"And", "Or", "Not", "Cast"} or kind == "Where" and i == 0)
        dtype = 9 if boolean else 3 if quantized else 1
        quant = QuantParam(scale=np.array([1.0 if kind == "Cast" else 0.125], np.float32),
                           zero_point=np.array([0], np.int8)) if dtype == 3 else None
        tensors[name] = TensorInfo(name=name, shape=(1, 2, 3, 4), dtype=dtype, quant=quant)
    graph = AnalyzedGraph(ops=[OpNode(name="op", op_type=kind, inputs=inputs, outputs=["y"])],
                          tensors=tensors, model_inputs=inputs, model_outputs=["y"])
    graph.is_quantized = quantized
    return graph


@pytest.mark.parametrize("kind", ["Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual",
                                  "And", "Or", "Not", "Where", "Cast", "Sum"])
@pytest.mark.parametrize("quantized", [False, True])
def test_bool_signatures_and_quantization(kind, quantized):
    graph = _bool_graph(kind, quantized)
    assert validate_execution_dtype(graph).supported
    assert validate_operator_support(graph).supported
    boolean = next((t for t in graph.tensors.values() if t.dtype == 9), None)
    if boolean:
        boolean.quant = QuantParam(scale=np.array([1.0], np.float32), zero_point=np.array([0], np.int8))
        assert not validate_execution_dtype(graph).supported
        assert not validate_operator_support(graph).supported


@pytest.mark.parametrize("zero,count,valid", [(-128, 15, True), (-128, 16, False),
                                             (0, 32, True), (0, 33, False),
                                             (127, 15, True), (127, 16, False)])
def test_sum_proves_reference_accumulator_bounds(zero, count, valid):
    graph = _bool_graph("Sum", True, count)
    for name in graph.model_inputs:
        graph.tensors[name].quant.zero_point[0] = zero
    report = validate_operator_support(graph)
    assert report.supported == valid
    if not valid:
        assert "int32 reference accumulator" in report.describe()


@pytest.mark.parametrize("kind,scale,message", [
    ("Equal", 0, "valid per-tensor quantization"),
    ("Less", -0.1, "valid per-tensor quantization"),
    ("LessOrEqual", 1, "strictly between 0 and 1"),
    ("Greater", 2, "strictly between 0 and 1"),
    ("GreaterOrEqual", float("nan"), "valid per-tensor quantization"),
    ("Equal", float("inf"), "valid per-tensor quantization"),
    ("Equal", 1, "strictly between 0 and 1"),
    ("Less", 1, "strictly between 0 and 1"),
    ("GreaterOrEqual", 2, "strictly between 0 and 1"),
])
def test_comparison_rejects_scales_outside_prepare_domain(kind, scale, message):
    graph = _bool_graph(kind, True)
    graph.tensors["x1"].quant.scale[0] = scale
    report = validate_operator_support(graph)
    assert not report.supported
    assert message in report.describe()


@pytest.mark.parametrize("kind,slot", [("Where", "x1"), ("Where", "x2"), ("Where", "y"), ("Sum", "x1")])
def test_select_and_sum_require_matching_quantization(kind, slot):
    graph = _bool_graph(kind, True)
    graph.tensors[slot].quant.scale[0] = 0.25
    assert "identical quantization" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("scale,zero", [(0.5, 0), (1, -1), (2, 3), (1 / 256, -128)])
def test_cast_quantizes_bool_into_any_int8_encoding(scale, zero):
    graph = _bool_graph("Cast", True)
    graph.tensors["y"].quant.scale[0] = scale
    graph.tensors["y"].quant.zero_point[0] = zero
    assert validate_operator_support(graph).supported


@pytest.mark.parametrize("source,target", [(1, 9), (3, 9), (1, 3), (3, 1), (6, 9), (9, 6)])
def test_cast_refuses_unapproved_directions(source, target):
    graph = _bool_graph("Cast")
    graph.tensors["x0"].dtype = source
    graph.tensors["y"].dtype = target
    assert not validate_operator_support(graph).supported


def test_sum_rejects_output_multiplier_outside_reference_domain():
    graph = _bool_graph("Sum", True)
    graph.tensors["y"].quant.scale[0] = 2**-23
    assert "output multiplier" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("kind", ["Gather", "GatherND", "EmbeddingLookup", "DynamicUpdateSlice"])
@pytest.mark.parametrize("dtype", [6, 7])
def test_runtime_index_inputs_narrow_and_keep_their_slot(tmp_path, kind, dtype):
    from scripts.crossrepo_contract import _runtime_index_case
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.binary.writer import emit_binary_bytes

    case = _runtime_index_case(kind, False)
    name = "starts" if kind == "DynamicUpdateSlice" else "indices"
    info = next(i for i in case.compile_model.graph.input if i.name == name)
    info.type.tensor_type.elem_type = dtype
    path = tmp_path / "indices.onnx"
    onnx.save(case.compile_model, path)
    graph, _ = _run_pipeline(str(path), ("8K",))
    assert graph.tensors[name].dtype == 6
    assert validate_execution_dtype(graph).supported
    plan = read_binary_plan(emit_binary_bytes(graph))
    tensor = next(t for t in plan["tensors"] if t["name"] == name)
    assert tensor["dtype"] == 6 and tensor["iface_dtype"] == (7 if dtype == 7 else 0)
    graph.ops.append(OpNode(name="wrong_slot", op_type="Relu", inputs=[name], outputs=["bad"]))
    assert not validate_execution_dtype(graph).supported
    graph.ops.pop()
    info.type.tensor_type.elem_type = 1
    onnx.save(case.compile_model, path)
    with pytest.raises(ClickException, match="indices must be unquantized int32 or int64"):
        _run_pipeline(str(path), ("8K",))


def test_float_section_can_produce_a_typed_comparison():
    from tigris.dtypes import check_dtype_signatures

    tensors = [("x", 3, False, True), ("real", 1, False, False),
               ("bias", 1, True, False), ("sum", 1, False, False),
               ("condition", 9, False, False)]
    operators = [("Dequantize", ["x"], ["real"]),
                 ("Add", ["real", "bias"], ["sum"]),
                 ("Greater", ["sum", "bias"], ["condition"])]
    data, issues = check_dtype_signatures(tensors, operators, ["x"], ["condition"])
    assert not issues
    assert data == {3: ["x"]}
