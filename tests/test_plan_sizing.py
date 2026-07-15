"""Regression tests for schema-derived plan-size accounting."""

import pytest
import onnx
from onnx import TensorProto, helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.defs import (
    HEADER_SIZE,
    HEADER_STRUCT,
    OP_ACTIVATION_STRUCT,
    OP_PREFIX_STRUCT,
    OP_SIZE,
    OP_WEIGHT_BIAS_STRUCT,
    QUANT_PARAM_SIZE,
    QUANT_PARAM_STRUCT,
    SECTION_ENTRY_SIZE,
    SECTION_ENTRY_STRUCT,
    SPATIAL_ATTRS_STRUCT,
    STAGE_SIZE,
    STAGE_STRUCT,
    TENSOR_SIZE,
    TENSOR_STRUCT,
    TILE_PLAN_SIZE,
    TILE_PLAN_STRUCT,
    WEIGHT_BLOCK_SIZE,
    WEIGHT_BLOCK_STRUCT,
    WEIGHT_ENTRY_SIZE,
    WEIGHT_ENTRY_STRUCT,
)
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.loaders import load_model


def _analyze(path, budget=65536):
    graph = load_model(path)
    graph = compute_lifetimes(graph)
    graph = compute_memory_timeline(graph)
    graph = partition_temporal(graph, budget)
    return partition_spatial(graph)


@pytest.fixture
def transpose_path(tmp_path):
    model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3])
    model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 1, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["middle"], name="relu"),
            helper.make_node(
                "Transpose", ["middle"], ["output"], name="transpose", perm=[2, 0, 1]
            ),
        ],
        "transpose_sizing",
        [model_input],
        [model_output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "transpose.onnx"
    onnx.save(model, path)
    return path


def test_exported_record_sizes_are_derived_from_canonical_layouts():
    assert HEADER_SIZE == HEADER_STRUCT.size == 48
    assert SECTION_ENTRY_SIZE == SECTION_ENTRY_STRUCT.size == 8
    assert TENSOR_SIZE == TENSOR_STRUCT.size == 16
    assert OP_SIZE == (
        OP_PREFIX_STRUCT.size
        + SPATIAL_ATTRS_STRUCT.size
        + OP_WEIGHT_BIAS_STRUCT.size
        + OP_ACTIVATION_STRUCT.size
    ) == 38
    assert STAGE_SIZE == STAGE_STRUCT.size == 28
    assert TILE_PLAN_SIZE == TILE_PLAN_STRUCT.size == 24
    assert WEIGHT_ENTRY_SIZE == WEIGHT_ENTRY_STRUCT.size == 12
    assert QUANT_PARAM_SIZE == QUANT_PARAM_STRUCT.size == 16
    assert WEIGHT_BLOCK_SIZE == WEIGHT_BLOCK_STRUCT.size == 20


@pytest.mark.parametrize(
    ("fixture_name", "budget"),
    [
        ("linear_3op_path", 65536),
        ("conv_relu_chain_path", 32768),
        ("diamond_path", 1536),
    ],
)
def test_float_plan_estimate_matches_serialized_size(request, fixture_name, budget):
    graph = _analyze(request.getfixturevalue(fixture_name), budget=budget)
    findings = compute_findings(graph)

    if fixture_name == "conv_relu_chain_path":
        assert any(stage.tile_plan is not None for stage in graph.stages)
    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))
    assert findings.plan_overhead_bytes == (
        findings.plan_size_bytes - findings.total_weight_bytes
    )


def test_plan_estimate_scales_past_the_old_fixed_string_allowance(linear_3op_path):
    graph = _analyze(linear_3op_path)
    graph.model_name = "large-model-" + "x" * 5000
    findings = compute_findings(graph)

    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))


def test_quantized_plan_estimate_includes_quant_section(qdq_conv_path):
    graph = _analyze(qdq_conv_path, budget=4096)
    findings = compute_findings(graph)

    assert graph.is_quantized
    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))


def test_plan_estimate_includes_typed_attributes(transpose_path):
    graph = _analyze(transpose_path, budget=4096)
    findings = compute_findings(graph)

    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))
