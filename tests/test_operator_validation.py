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
from tigris.graph.ir import AnalyzedGraph, OpNode, Stage, TensorInfo, TilePlan


@pytest.fixture
def sin_model_path(tmp_path):
    """A valid one-op ONNX model whose operator is not in OP_TYPE_MAP."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    node = helper.make_node(
        "Sin", ["input"], ["output"], name="unsupported_sin"
    )
    graph = helper.make_graph(
        [node], "unsupported_sin", [model_input], [model_output]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = tmp_path / "unsupported_sin.onnx"
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


def test_normalized_unknown_operator_is_reported(sin_model_path):
    graph, _ = _run_pipeline(str(sin_model_path), ("4K",))

    validation = validate_operator_support(graph)
    findings = compute_findings(graph)

    assert not validation.supported
    assert [issue.describe() for issue in validation.issues] == [
        "unsupported_sin (Sin)"
    ]
    assert findings.verdict == "needs_work"
    assert findings.unsupported_operators == ["unsupported_sin (Sin)"]


def test_analyze_lists_unsupported_operator_as_failure(sin_model_path):
    result = CliRunner().invoke(
        cli, ["analyze", str(sin_model_path), "-m", "4K"]
    )

    assert result.exit_code == 0, result.output
    assert "FAIL" in result.output
    assert "unsupported_sin (Sin)" in result.output


def test_compile_rejects_unsupported_operator_without_output(
    sin_model_path, tmp_path
):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(sin_model_path), "-m", "4K", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "Cannot compile a plan with unsupported operators" in result.output
    assert "unsupported_sin (Sin)" in result.output
    assert not output.exists()


def test_writer_rejects_unsupported_operator(sin_model_path):
    graph, _ = _run_pipeline(str(sin_model_path), ("4K",))

    with pytest.raises(
        ValueError,
        match="Cannot emit a plan with unsupported operators: unsupported_sin",
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
        ({"count_include_pad": 1}, "count_include_pad=1 is not encoded"),
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
    assert "dynamic operand broadcasting is not implemented" in validation.describe()


def test_quantized_constant_elementwise_operand_is_rejected():
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
    assert "lack shape/quant metadata" in validation.describe()


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
                "constant", (4,), TensorProto.FLOAT, is_constant=True
            ),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
        weight_data={"constant": np.ones(4, dtype=np.float32)},
    )

    validation = validate_operator_support(graph)

    assert not validation.supported
    assert "requires unsupported broadcasting" in validation.describe()
