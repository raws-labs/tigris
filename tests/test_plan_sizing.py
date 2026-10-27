"""Regression tests for schema-derived plan-size accounting."""

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import _weight_sizes, compute_findings
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
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.pipeline import plan_model

from conftest import save_onnx

OPS = Path(__file__).parent / "fixtures" / "tflite" / "ops"


@pytest.fixture
def transpose_path(tmp_path):
    model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3])
    model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 1, 2])
    return save_onnx(tmp_path, "transpose_sizing",
                     [helper.make_node("Relu", ["input"], ["middle"], name="relu"),
                      helper.make_node("Transpose", ["middle"], ["output"], name="transpose",
                                       perm=[2, 0, 1])], [model_input], [model_output],
                     ir_version=8, filename="transpose.onnx")


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
    graph = plan_model(request.getfixturevalue(fixture_name), (budget,), chains=False)
    findings = compute_findings(graph)

    if fixture_name == "conv_relu_chain_path":
        assert any(stage.tile_plan is not None for stage in graph.stages)
    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))
    assert findings.plan_overhead_bytes == (
        findings.plan_size_bytes - findings.total_weight_bytes
    )


def test_plan_estimate_scales_past_the_old_fixed_string_allowance(linear_3op_path):
    graph = plan_model(linear_3op_path, (65536,))
    graph.model_name = "large-model-" + "x" * 5000
    findings = compute_findings(graph)

    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))


def test_quantized_plan_estimate_includes_quant_section(qdq_conv_path):
    graph = plan_model(qdq_conv_path, (4096,))
    findings = compute_findings(graph)

    assert graph.is_quantized
    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))


def test_plan_estimate_includes_typed_attributes(transpose_path):
    graph = plan_model(transpose_path, (4096,))
    findings = compute_findings(graph)

    assert findings.plan_size_bytes == len(emit_binary_bytes(graph))


def _assert_weights_sized_as_stored(graph, weight_bytes):
    findings = compute_findings(graph)
    plan = emit_binary_bytes(graph)

    assert sum(w["size_bytes"] for w in read_binary_plan(plan)["weights"]) == weight_bytes
    assert findings.total_weight_bytes == weight_bytes
    assert findings.plan_overhead_bytes == len(plan) - weight_bytes
    # Without a float weight the int8 profile is the plan as written.
    assert sum(_weight_sizes(graph)[1]) == weight_bytes
    assert findings.int8_plan_size_bytes == len(plan)


@pytest.mark.parametrize(("name", "weight_bytes"), [("svdf", 160), ("svdf_batch", 54)])
def test_int16_weights_are_sized_as_the_plan_stores_them(name, weight_bytes):
    _assert_weights_sized_as_stored(plan_model(OPS / f"{name}.tflite", (262144,)), weight_bytes)


def test_bool_weights_are_sized_as_the_plan_stores_them(tmp_path):
    mask = numpy_helper.from_array(np.array([[True, False, True, True, False, False, True, False]]), "mask")
    path = save_onnx(
        tmp_path, "bool_mask",
        [helper.make_node("Neg", ["x"], ["negated"]),
         helper.make_node("Where", ["mask", "x", "negated"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])], [mask])
    graph = plan_model(path, (4096,))

    assert graph.weight_data["mask"].dtype == np.bool_
    _assert_weights_sized_as_stored(graph, 8)


def test_no_lz4_estimate_when_the_writer_cannot_group_weights_by_stage(tmp_path):
    weight = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((8, 8, 3, 3)).astype(np.float32), "w")
    path = save_onnx(
        tmp_path, "shared_weight",
        [helper.make_node("Conv", ["x", "w"], ["a"], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["a"], ["b"]),
         helper.make_node("Conv", ["b", "w"], ["y"], pads=[1, 1, 1, 1])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16, 16])], [weight])
    graph = plan_model(path, (12288,), chains=False)

    assert len(graph.stages) == 2
    with pytest.raises(ValueError, match="shared across stages"):
        emit_binary_bytes(graph, compress="lz4")
    assert compute_findings(graph).lz4_plan_size_bytes == 0
