"""Fail-closed validation for operators outside the binary plan schema."""

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import (
    validate_execution_dtype,
    validate_operator_support,
)
from tigris.cli import _run_pipeline, cli
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
    """One QDQ branch and one float branch cannot share one dispatcher."""
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
    assert "FAIL" in result.output
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
    assert "Cannot compile a plan with unsupported operators" in result.output
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
    assert "mixed activation dtypes" in validation.describe()
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
    assert "Cannot compile a plan with unsupported tensor dtypes" in result.output
    assert output.read_bytes() == b"sentinel"


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"count_include_pad": 1}, "count_include_pad=1 with padding is not encoded"),
        ({"ceil_mode": 1}, "ceil_mode=1 is not encoded"),
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
            "must state a TFLite sampling convention",
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
    ],
)
def test_unencoded_operator_variants_are_rejected(op, tensors, message):
    validation = validate_operator_support(AnalyzedGraph(ops=[op], tensors=tensors))

    assert not validation.supported
    assert message in validation.describe()


def test_representable_softmax_concat_and_resize_variants_are_supported():
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
        },
    )

    validation = validate_operator_support(graph)

    assert validation.supported, validation.describe()


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


def test_count_include_pad_with_padding_is_refused(tmp_path):
    ag, _ = _run_pipeline(str(_avg_pool_path(tmp_path, [1, 1, 1, 1])), ("64K",),
                          report_bindings=False)
    with pytest.raises(ValueError, match="count_include_pad=1 with padding"):
        emit_binary_bytes(ag)


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


def test_hardswish_and_int8_bilinear_have_kernels():
    from tigris.capabilities import effective_operators

    for backend in ("reference", "s8_ref", "esp-nn", "cmsis-nn"):
        assert "HardSwish" in effective_operators(backend)
        assert "ResizeLinear" in effective_operators(backend)


@pytest.mark.parametrize("kind", [
    "Neg", "Exp", "Log", "Sqrt", "Square", "Floor", "Ceil", "Round",
    "Sin", "Cos", "FloorDiv", "FloorMod",
])
def test_elementwise_without_int8_kernel_rejects_quantized_plans(kind):
    binary = kind in {"FloorDiv", "FloorMod"}
    quant = QuantParam(np.array([0.125], np.float32), np.array([-17], np.int8))
    ag = AnalyzedGraph(
        ops=[OpNode(name="math", op_type=kind,
                    inputs=["a", "b"] if binary else ["a"], outputs=["y"])],
        tensors={name: TensorInfo(name, (1, 4), 3, quant=quant) for name in ("a", "b", "y")},
        is_quantized=True,
    )
    assert "no s8_ref runtime kernel" in validate_operator_support(ag).describe()


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


def test_grouped_convolution_is_rejected_at_compile_time(tmp_path):
    """group=2 over four channels is neither ordinary nor depthwise; the runtime has no kernel."""
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
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "4K", "-o", str(output)])

    assert result.exit_code != 0
    assert "group=2 is not implemented" in result.output
    assert not output.exists()


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


@pytest.mark.parametrize("kind", ["ReduceMax", "ReduceMin", "ReduceSum", "CumSum"])
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


def test_cumsum_refuses_nonzero_input_zero_point_and_invalid_options():
    graph = _reduction_graph("CumSum", quantized=True)
    graph.tensors["x"].quant = QuantParam(
        scale=np.array([0.125], np.float32), zero_point=np.array([-17], np.int8))
    assert "input zero point 0" in validate_operator_support(graph).describe()
    graph = _reduction_graph("CumSum", quantized=True)
    graph.tensors["y"].quant = QuantParam(
        scale=np.array([2**-22], np.float32), zero_point=np.array([0], np.int8))
    assert "smaller than one" in validate_operator_support(graph).describe()
    graph.ops[0].attrs["exclusive"] = 2
    assert "exclusive and reverse" in validate_operator_support(graph).describe()


@pytest.mark.parametrize("kind", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("input_dtype", [1, 3])
@pytest.mark.parametrize("output_dtype", [6, 7])
def test_index_outputs_do_not_bypass_dispatcher_dtype_contract(kind, input_dtype, output_dtype):
    graph = _reduction_graph(kind)
    graph.tensors["x"].dtype = input_dtype
    graph.tensors["y"].dtype = output_dtype
    assert not validate_execution_dtype(graph).supported
    assert "unsupported activation tensor dtype" in validate_execution_dtype(graph).describe()
    assert not validate_operator_support(graph).supported


def test_reducemax_does_not_rewrite_nonspatial_axes_to_global_pool(tmp_path):
    path = _save(tmp_path, "max_nonspatial", [
        helper.make_node("ReduceMax", ["x"], ["y"], axes=[1, 2], keepdims=1),
    ], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, 5])])
    graph, _ = _run_pipeline(str(path), ("4K",), report_bindings=False)
    assert any(op.op_type == "ReduceMax" for op in graph.ops)
    assert all(op.op_type != "GlobalMaxPool" for op in graph.ops)
    assert not validate_operator_support(graph).supported
