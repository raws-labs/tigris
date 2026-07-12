"""Tests for the binary plan emitter."""

import struct

import pytest

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal
from tigris.analysis.partition_spatial import partition_spatial
from tigris import SCHEMA_VERSION
from tigris.emitters.binary.defs import (
    HEADER_SIZE,
    MAGIC,
    OP_TYPE_MAP,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary, emit_binary_bytes
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
        assert tp_bin["tile_height"] == tp.tile_height
        assert tp_bin["num_tiles"] == tp.num_tiles
        assert tp_bin["halo"] == tp.halo
        assert tp_bin["receptive_field"] == tp.receptive_field


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


@pytest.mark.parametrize("version", [0, 1, SCHEMA_VERSION, 99])
def test_schema_version_validation(linear_3op_path, version):
    ag = _full_pipeline(linear_3op_path)
    data = bytearray(emit_binary_bytes(ag))
    struct.pack_into("<I", data, 4, version)

    if version == SCHEMA_VERSION:
        assert read_binary_plan(bytes(data))["version"] == SCHEMA_VERSION
    else:
        with pytest.raises(ValueError) as exc_info:
            read_binary_plan(bytes(data))
        assert str(exc_info.value) == (
            f"Unsupported schema version: {version} (expected {SCHEMA_VERSION})"
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
