"""Tests for the binary plan emitter."""

import hashlib
import struct
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    TILE_AXIS_HEIGHT_OR_LENGTH,
)
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.defs import (
    HEADER_SIZE,
    MAGIC,
    OP_ATTR_AXES,
    OP_ATTR_BINARY_REQUANT,
    OP_ATTR_EPSILON,
    OP_TYPE_MAP,
    TENSOR_FLAG_LINEAR,
    TENSOR_FLAG_MODEL_INPUT,
    TENSOR_FLAG_MODEL_OUTPUT,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import (
    _build_quant_params,
    _build_tile_plans,
    emit_binary,
    emit_binary_bytes,
)
from tigris.graph.ir import (
    AnalyzedGraph,
    MemoryBudget,
    OpNode,
    QuantParam,
    Stage,
    TensorInfo,
    TilePlan,
)
from tigris.loaders import load_model


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag


# Header / magic tests


def test_magic_and_version(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)

    assert data[:4] == MAGIC
    version = struct.unpack_from("<I", data, 4)[0]
    assert version == SCHEMA_VERSION


def test_file_size_matches(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)

    file_size = struct.unpack_from("<I", data, 8)[0]
    assert file_size == len(data)


def test_v3_quant_data_pages_keep_offsets_in_uint16_range():
    """A large aggregate quant pool uses v3 pages instead of wrapping offsets."""
    large = QuantParam(
        scale=np.full(32768, 0.125, dtype=np.float32),
        zero_point=np.zeros(32768, dtype=np.int32),
        axis=0,
    )
    small = QuantParam(
        scale=np.array([0.25], dtype=np.float32),
        zero_point=np.array([0], dtype=np.int32),
    )
    ag = SimpleNamespace(
        is_quantized=True,
        ops=[],
        tensors=OrderedDict([
            ("large", TensorInfo("large", (32768,), 3, quant=large)),
            ("small", TensorInfo("small", (1,), 3, quant=small)),
        ]),
    )

    section, indices = _build_quant_params(ag)

    assert indices == {"large": 0, "small": 1}
    assert struct.unpack_from("<HH", section, 0) == (2, 2)
    first = struct.unpack_from("<fiHHHH", section, 4)
    second = struct.unpack_from("<fiHHHH", section, 20)
    assert first[2:] == (32768, 0, 32768, 0)
    assert second[2:] == (1, 0, 1, 1)


def test_v3_quant_param_rejects_arrays_that_cannot_fit_one_page():
    too_large = QuantParam(
        scale=np.full(32769, 0.125, dtype=np.float32),
        zero_point=np.zeros(32769, dtype=np.int32),
        axis=0,
    )
    ag = SimpleNamespace(
        is_quantized=True,
        ops=[],
        tensors=OrderedDict([
            ("too_large", TensorInfo("too_large", (32769,), 3, quant=too_large)),
        ]),
    )

    with pytest.raises(ValueError, match="v3 per-param limit"):
        _build_quant_params(ag)


# Round-trip tests


def test_roundtrip_linear(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "linear_3op"
    assert plan["num_ops"] == 3
    assert plan["version"] == SCHEMA_VERSION

    # All activation tensors (non-constant) should be present
    tensor_names = {t["name"] for t in plan["tensors"]}
    assert "input" in tensor_names
    assert "output" in tensor_names


def test_roundtrip_diamond(diamond_path):
    ag = _full_pipeline(diamond_path, budget=1536)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "diamond"
    assert plan["num_ops"] == 3
    assert plan["num_stages"] >= 1
    assert plan["budget"] == 1536


def test_roundtrip_conv_chain(conv_relu_chain_path):
    ag = _full_pipeline(conv_relu_chain_path, budget=6144)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "conv_relu_chain"
    assert plan["num_ops"] == 2  # Relu absorbed into first Conv

    # Check Conv op has spatial attrs
    conv_ops = [op for op in plan["ops"] if op["op_type"] == OP_TYPE_MAP["Conv"]]
    assert len(conv_ops) == 2
    for cop in conv_ops:
        assert cop["spatial"]["kernel_h"] == 3
        assert cop["spatial"]["kernel_w"] == 3
        assert cop["spatial"]["stride_h"] == 1
        assert cop["spatial"]["stride_w"] == 1

    # First Conv should have fused Relu activation
    from tigris.emitters.binary.defs import ACT_RELU
    assert conv_ops[0]["fused_act"] == ACT_RELU


def test_roundtrip_conv_pool(conv_pool_chain_path):
    ag = _full_pipeline(conv_pool_chain_path, budget=4096)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "conv_pool_chain"

    # Find the MaxPool op
    pool_ops = [op for op in plan["ops"] if op["op_type"] == OP_TYPE_MAP["MaxPool"]]
    assert len(pool_ops) == 1
    assert pool_ops[0]["spatial"]["kernel_h"] == 2
    assert pool_ops[0]["spatial"]["stride_h"] == 2


# Tensor validation


def test_tensor_shapes(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for t in plan["tensors"]:
        name = t["name"]
        if name in ag.tensors:
            expected = list(ag.tensors[name].shape)
            assert t["shape"] == expected, f"Shape mismatch for {name}"


def test_tensor_dtypes(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for t in plan["tensors"]:
        name = t["name"]
        if name in ag.tensors:
            assert t["dtype"] == ag.tensors[name].dtype


def test_tensor_sizes(conv_relu_chain_path):
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for t in plan["tensors"]:
        name = t["name"]
        if name in ag.tensors:
            assert t["size_bytes"] == ag.tensors[name].size_bytes


def test_tensor_flags(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for t in plan["tensors"]:
        if t["name"] in ag.model_inputs:
            assert t["flags"] & 0x02  # MODEL_INPUT
        if t["name"] in ag.model_outputs:
            assert t["flags"] & 0x04  # MODEL_OUTPUT


# Op type mapping


def test_op_types(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for i, op in enumerate(plan["ops"]):
        expected_type = OP_TYPE_MAP[ag.ops[i].op_type]
        assert op["op_type"] == expected_type


# Stage and tile plan tests


def test_stages_preserved(conv_relu_chain_path):
    ag = _full_pipeline(conv_relu_chain_path, budget=6144)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(ag.stages) >= 2
    assert plan["num_stages"] == len(ag.stages)
    for i, stage in enumerate(plan["stages"]):
        assert stage["peak_bytes"] == ag.stages[i].peak_bytes


def test_schema_v5_stage_table_supports_more_than_256_stages():
    stage_count = 300
    tensor_names = [f"value_{index}" for index in range(stage_count + 1)]
    ops = [
        OpNode(
            name=f"relu_{index}",
            op_type="Relu",
            inputs=[tensor_names[index]],
            outputs=[tensor_names[index + 1]],
            step=index,
            stage=index,
        )
        for index in range(stage_count)
    ]
    stages = [
        Stage(
            stage_id=index,
            op_indices=[index],
            input_tensors=[tensor_names[index]],
            output_tensors=[tensor_names[index + 1]],
            peak_bytes=64,
        )
        for index in range(stage_count)
    ]
    graph = AnalyzedGraph(
        model_name="many_stages",
        ops=ops,
        tensors=OrderedDict(
            (name, TensorInfo(name, (1,), 1)) for name in tensor_names
        ),
        model_inputs=[tensor_names[0]],
        model_outputs=[tensor_names[-1]],
        stages=stages,
        budget=MemoryBudget(fast=64),
        peak_memory_bytes=64,
    )

    plan = read_binary_plan(emit_binary_bytes(graph))

    assert plan["num_stages"] == stage_count
    assert [op["stage"] for op in plan["ops"]] == list(range(stage_count))


def test_tile_plans_preserved(conv_pool_chain_path):
    """If pipeline produces tile plans, verify they survive the round-trip."""
    # This fixture has a feasible 4 KiB tile plan.
    ag = _full_pipeline(conv_pool_chain_path, budget=4096)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    tiled_stages = [s for s in ag.stages if s.tile_plan is not None]
    assert tiled_stages
    assert plan["num_tile_plans"] == len(tiled_stages)

    for tp_bin, stage in zip(plan["tile_plans"], tiled_stages):
        tp = stage.tile_plan
        assert tp_bin["tileable"] == tp.tileable
        assert tp_bin["axis"] == tp.axis
        assert tp_bin["tile_height"] == tp.tile_height
        assert tp_bin["num_tiles"] == tp.num_tiles
        assert tp_bin["halo"] == tp.halo
        assert tp_bin["receptive_field"] == tp.receptive_field


def test_writer_rejects_reserved_width_axis(conv_pool_chain_path):
    ag = _full_pipeline(conv_pool_chain_path, budget=4096)
    stage = next(item for item in ag.stages if item.tile_plan is not None)
    stage.tile_plan.axis = 2

    with pytest.raises(ValueError, match="unsupported tile axis 2"):
        emit_binary_bytes(ag)


# Model I/O


def test_model_io(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(plan["model_inputs"]) == 1
    assert len(plan["model_outputs"]) == 1

    # Input tensor index should map to "input"
    inp_idx = plan["model_inputs"][0]
    assert plan["tensors"][inp_idx]["name"] == "input"

    out_idx = plan["model_outputs"][0]
    assert plan["tensors"][out_idx]["name"] == "output"


# File output


def test_file_output(linear_3op_path, tmp_path):
    ag = _full_pipeline(linear_3op_path)
    out = tmp_path / "test.tgrs"
    emit_binary(ag, out)

    assert out.exists()
    data = out.read_bytes()
    assert data[:4] == MAGIC

    plan = read_binary_plan(data)
    assert plan["model_name"] == "linear_3op"


# All fixtures produce valid binaries


def test_all_fixtures_no_budget(
    linear_3op_path, diamond_path, large_activations_path,
    conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path,
):
    for path in [
        linear_3op_path, diamond_path, large_activations_path,
        conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path,
    ]:
        ag = _full_pipeline(path)
        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)
        assert plan["magic"] == MAGIC
        assert plan["num_ops"] == len(ag.ops)


def test_all_fixtures_with_budget(
    linear_3op_path, diamond_path, large_activations_path,
    conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path,
):
    for path in [
        linear_3op_path, diamond_path, large_activations_path,
        conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path,
    ]:
        ag = _full_pipeline(path, budget=8192)
        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)
        assert plan["magic"] == MAGIC
        assert plan["num_ops"] == len(ag.ops)
        assert plan["budget"] == 8192


# Error handling


def test_immutable_supported_schema_fixtures():
    # Each artifact comes from the compiler revision that introduced the
    # corresponding schema and exercises a version-specific wire feature.
    fixtures = (
        (
            2,
            "schema-v2-linear.tgrs",
            "54597dd75d14c54fcb3c6abb3c6cb3a8c2f18a4bc2fa54ea78d9762f75cb96c5",
            0,
            0,
        ),  # b47664926e7a484d6638ecd0fd372da477236619
        (
            3,
            "schema-v3-qdq-conv.tgrs",
            "fe300785de7401dcc30f05e3c859943a74f02da83df40a0047d1fb637a46b9bb",
            3,
            0,
        ),  # 0fe37d3a53292cf532ba8628d2284c00a896fbb2
        (
            4,
            "schema-v4-transpose.tgrs",
            "c6892a319b4ca319dfca2f44cbaf6f0f36fca502cdbd493c52b67f2d2aa03e93",
            0,
            1,
        ),  # 208b322cab7f97c9960c63a8075944374fdfff2c
        (
            5,
            "schema-v5-conv1d-axis.tgrs",
            "7f0ac56ffe4100feb957917e36bace42e34c3236ebd45a7585d2d13a33b030c6",
            0,
            0,
        ),
        (
            6,
            "schema-v6-interface-dtype.tgrs",
            "1af3f363eb98db197898f89d1e1d70b4174b5cbb2356cd54778fce590a5276a9",
            3,
            0,
        ),
        (
            7,
            "schema-v7-tensor-layout.tgrs",
            "6657ccd48dafa81e1f9eeada700d705621a62f3adf61bcbf64b6af943857540f",
            0,
            2,
        ),
        (
            8,
            "schema-v8-layer-norm.tgrs",
            "cb675bf404c94af8f80e854a5d52175efd3e61ce313989b26f68c18dc5152365",
            0,
            3,
        ),
        (
            9,
            "schema-v9-binary-requant.tgrs",
            "856d250b3a6bffbdc99a752813d652d720965abf0ee4aba33e9276847baa22c0",
            3,
            1,
        ),
    )
    assert tuple(item[0] for item in fixtures) == SUPPORTED_SCHEMA_VERSIONS

    fixture_dir = Path(__file__).parent / "schema_compat"
    for version, filename, digest, quant_params, op_attributes in fixtures:
        data = (fixture_dir / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        plan = read_binary_plan(data)
        assert plan["version"] == version
        assert len(plan["quant_params"]) == quant_params
        assert len(plan["op_attributes"]) == op_attributes
        if version == 9:
            # The quantized sum states the three pairs that scale its two
            # operands and its result, which is what schema 9 added.
            attribute = plan["op_attributes"][0]
            assert attribute["type"] == OP_ATTR_BINARY_REQUANT
            assert len(attribute["data"]) == 24
        if version == 5:
            assert plan["tile_plans"][0]["axis"] == 1
            assert plan["tile_plans"][0]["num_tiles"] > 1
        if version == 7:
            # Schema 7 records storage order per tensor. The batched matrix
            # product this fixture holds converts its operand to the model's
            # own axis order and converts the result back, so the linear
            # tensors are internal and the boundaries are not.
            linear = [
                t for t in plan["tensors"] if t["flags"] & TENSOR_FLAG_LINEAR
            ]
            assert linear
            boundary = TENSOR_FLAG_MODEL_INPUT | TENSOR_FLAG_MODEL_OUTPUT
            assert not any(t["flags"] & boundary for t in linear)
        if version == 8:
            # Schema 8 opens the attribute section to kinds beyond the
            # Transpose permutation. This fixture carries a normalization
            # whose variance floor is one of them.
            epsilons = [
                a for a in plan["op_attributes"] if a["type"] == OP_ATTR_EPSILON
            ]
            assert len(epsilons) == 1
            assert struct.unpack("<f", epsilons[0]["data"])[0] > 0.0
        if version == 6:
            # A quantized boundary states the float interface the model
            # declares, which is what schema 6 added.
            for index in plan["model_inputs"] + plan["model_outputs"]:
                tensor = plan["tensors"][index]
                assert tensor["dtype"] == 3
                assert tensor["iface_dtype"] == 1


@pytest.mark.parametrize("version", [0, 1, *SUPPORTED_SCHEMA_VERSIONS, 99])
def test_schema_version_validation(linear_3op_path, version):
    ag = _full_pipeline(linear_3op_path)
    data = bytearray(emit_binary_bytes(ag))
    struct.pack_into("<I", data, 4, version)

    if version in SUPPORTED_SCHEMA_VERSIONS:
        assert read_binary_plan(bytes(data))["version"] == version
    else:
        with pytest.raises(ValueError) as exc_info:
            read_binary_plan(bytes(data))
        supported = ", ".join(str(item) for item in SUPPORTED_SCHEMA_VERSIONS)
        assert str(exc_info.value) == (
            f"Unsupported schema version: {version} (expected one of: {supported})"
        )


def test_missing_required_section_is_rejected_cleanly(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    data = bytearray(emit_binary_bytes(ag))
    struct.pack_into("<II", data, HEADER_SIZE, 0, 0)

    with pytest.raises(ValueError, match="Missing required section type"):
        read_binary_plan(bytes(data))


def test_bad_magic():
    data = b"XXXX" + b"\x00" * 100
    try:
        read_binary_plan(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Bad magic" in str(e)


def test_truncated():
    try:
        read_binary_plan(b"TGRS")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "too small" in str(e)


def test_a_tile_extent_that_does_not_fit_names_the_field_and_the_stage():
    """struct.pack alone would abort the compile with no diagnosis.

    Every solver that bands a single axis stays inside a uint16 by the nature
    of shapes that fit an embedded budget. The ones that band a product of
    axes do not, so the writer says which field overflowed on which stage
    rather than letting struct raise. tile_width is masked on the way out,
    so without this it would truncate silently instead.
    """
    ag = SimpleNamespace(stages=[
        SimpleNamespace(
            stage_id=3,
            tile_plan=TilePlan(
                tileable=True,
                axis=TILE_AXIS_HEIGHT_OR_LENGTH,
                tile_height=8192,
                num_tiles=8,
                original_height=65_536,
                tiled_peak_bytes=262_144,
            ),
        ),
    ])

    with pytest.raises(ValueError, match=r"stage 3 tile plan original_height"):
        _build_tile_plans(ag)


def _reduce_mean_model(path, shape, axes, keepdims, projected=False):
    """A ReduceMean over *axes*, optionally over a projected sequence.

    A matrix product ahead of the mean is what makes its operand state its own
    axis order, which is the difference between a mean over a sequence's
    tokens and one over a feature map's channels.
    """
    initializers = []
    nodes = []
    source = "input"
    if projected:
        initializers.append(numpy_helper.from_array(
            np.eye(shape[-1], dtype=np.float32), "w"))
        nodes.append(helper.make_node("MatMul", ["input", "w"], ["tokens"]))
        source = "tokens"
    nodes.append(helper.make_node(
        "ReduceMean", [source], ["output"], axes=list(axes),
        keepdims=1 if keepdims else 0, name="pool"))
    collapsed = {a % len(shape) for a in axes}
    if keepdims:
        out_shape = [
            1 if axis in collapsed else extent
            for axis, extent in enumerate(shape)
        ]
    else:
        out_shape = [
            extent for axis, extent in enumerate(shape)
            if axis not in collapsed
        ]
    graph = helper.make_graph(
        nodes,
        "reduce_mean",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    return path


def test_a_spatial_mean_over_both_image_axes_becomes_a_global_pool(tmp_path):
    ag = _full_pipeline(_reduce_mean_model(
        tmp_path / "gap.onnx", [1, 4, 8, 8], (2, 3), keepdims=True))
    assert [op.op_type for op in ag.ops] == ["GlobalAveragePool"]


def test_a_mean_over_one_axis_keeps_its_operator_and_states_the_axis(tmp_path):
    ag = _full_pipeline(_reduce_mean_model(
        tmp_path / "tokens.onnx", [1, 12, 8], (1,), keepdims=True,
        projected=True))
    assert "ReduceMean" in [op.op_type for op in ag.ops]
    plan = read_binary_plan(emit_binary_bytes(ag))
    axes = [a for a in plan["op_attributes"] if a["type"] == OP_ATTR_AXES]
    assert len(axes) == 1
    # The operand states its own axis order, so the emitted axis is the one
    # the model named.
    assert list(axes[0]["data"]) == [1]


def test_a_mean_over_a_stored_axis_is_emitted_where_the_runtime_holds_it(
    tmp_path,
):
    """A rank-3 feature map is stored channels-last, so its axis 1 moves.

    The compiler names ONNX axes and the runtime holds serialized ones. A mean
    over the channel axis of an NCL tensor is a mean over the last stored axis,
    and emitting the model's own number would collapse the wrong one.
    """
    ag = _full_pipeline(_reduce_mean_model(
        tmp_path / "channels.onnx", [1, 12, 8], (1,), keepdims=True))
    plan = read_binary_plan(emit_binary_bytes(ag))
    axes = [a for a in plan["op_attributes"] if a["type"] == OP_ATTR_AXES]
    assert len(axes) == 1
    assert list(axes[0]["data"]) == [2]
