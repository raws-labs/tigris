"""Tests for LZ4 weight compression in the binary plan emitter."""


import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.cli import _run_pipeline, cli
from tigris.cli.compile import _run_compressed_pipeline
from tigris.emitters.binary.defs import NO_WEIGHT, OP_ATTR_CONSTANTS, OP_TYPE_MAP
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import (
    compressed_weight_reserve_bytes,
    emit_binary_bytes,
)
from tigris.frontends.tflite import STATE_KEY
from tigris.graph.ir import Stage

from conftest import full_pipeline


# Compressed weight data matches uncompressed


def test_compressed_weight_data_matches(conv_relu_chain_path):
    """Decompressed weight data from compressed plan matches uncompressed plan."""
    ag, _, _, _ = _run_compressed_pipeline(str(conv_relu_chain_path), ("32K",))

    data_plain = emit_binary_bytes(ag)
    data_lz4 = emit_binary_bytes(ag, compress="lz4")

    plan_plain = read_binary_plan(data_plain)
    plan_lz4 = read_binary_plan(data_lz4)

    # Both should have same number of weights
    assert plan_plain["num_weights"] == plan_lz4["num_weights"]

    # Gather all decompressed bytes from blocks
    all_decompressed = bytearray()
    for block in sorted(plan_lz4["weight_blocks"], key=lambda b: b["first_weight_idx"]):
        all_decompressed.extend(block["decompressed"])

    # Gather all weight bytes from plain plan
    all_plain = bytearray()
    for w in plan_plain["weights"]:
        blob_start = w["blob_base"] + w["offset"]
        all_plain.extend(data_plain[blob_start : blob_start + w["size_bytes"]])

    assert bytes(all_decompressed) == bytes(all_plain)

    # Each weight block maps to exactly one stage.
    assert len(plan_lz4["weight_blocks"]) >= 2
    for block in plan_lz4["weight_blocks"]:
        assert block["stage_idx"] < plan_lz4["num_stages"]
        assert block["num_weights"] > 0
        assert block["compressed_size"] > 0
        assert block["uncompressed_size"] > 0


def test_stored_weight_blocks_hold_the_lz4_blocks_decompressed(conv_relu_chain_path):
    ag, _, _, _ = _run_compressed_pipeline(str(conv_relu_chain_path), ("32K",))
    stored = read_binary_plan(emit_binary_bytes(ag, compress="none"))
    lz4 = read_binary_plan(emit_binary_bytes(ag, compress="lz4"))

    def layout(plan):
        return [(b["stage_idx"], b["first_weight_idx"], b["num_weights"], b["uncompressed_size"],
                 b["decompressed"]) for b in plan["weight_blocks"]]

    assert stored["weight_blocks_compression"] == 0
    assert all(b["compressed_size"] == b["uncompressed_size"] for b in stored["weight_blocks"])
    assert layout(stored) == layout(lz4)
    assert stored["weights"] == lz4["weights"]


def _compile(model: Path, compress: str, tmp_path: Path) -> tuple[bytes, dict]:
    output = tmp_path / f"{compress}.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(model), "-m", "256K", "-c", compress, "-o", str(output)])
    assert result.exit_code == 0, result.output
    data = output.read_bytes()
    return data, read_binary_plan(data)


def _assert_reads_in_stage_blocks(plain_data: bytes, plain: dict, packed: dict) -> None:
    """Each weight a stage reads is in its decoded block with the bytes the
    uncompressed plan holds, and the block covers just those."""
    constants = {a["op_index"]: struct.unpack(f"<{len(a['data']) // 2}H", a["data"])
                 for a in packed["op_attributes"] if a["type"] == OP_ATTR_CONSTANTS}
    assert any(len(listed) > 2 for listed in constants.values())
    blocks = {block["stage_idx"]: block for block in packed["weight_blocks"]}
    read: dict[int, set[int]] = {}
    for op_index, op in enumerate(packed["ops"]):
        for k in (op["weight_idx"], op["bias_idx"], *constants.get(op_index, ())):
            if k == NO_WEIGHT:
                continue
            read.setdefault(op["stage"], set()).add(k)
            entry, stored = packed["weights"][k], plain["weights"][k]
            start = stored["blob_base"] + stored["offset"]
            decoded = blocks[op["stage"]]["decompressed"]
            assert (decoded[entry["offset"]:entry["offset"] + entry["size_bytes"]]
                    == plain_data[start:start + stored["size_bytes"]]), (op_index, k)
    assert {stage: (min(ks), len(ks)) for stage, ks in read.items()} == {
        stage: (b["first_weight_idx"], b["num_weights"]) for stage, b in blocks.items()}


@pytest.mark.parametrize("name", ["float_svdf", "svdf", "float_lstm", "lstm"])
def test_every_weight_an_operator_reads_is_in_its_stage_block(name, tmp_path):
    """Svdf and Lstm name their weights in a constants list, not the weight
    and bias fields."""
    model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / f"{name}.tflite"
    plain_data, plain = _compile(model, "none", tmp_path)
    _, packed = _compile(model, "lz4", tmp_path)
    _assert_reads_in_stage_blocks(plain_data, plain, packed)


def _svdf_after(tmp_path: Path, head: str, constant: str):
    """x -> head(x, constant) -> Svdf, each operator in a stage of its own."""
    rng = np.random.default_rng(0)
    batch, features, units, rank, memory = 2, 3, 3, 2, 4
    state = [batch, units * rank * memory]
    weights = {"projection": rng.standard_normal((features, features)),
               "feature": rng.standard_normal((units * rank, features)),
               "time": rng.standard_normal((units * rank, memory)),
               "bias": rng.standard_normal(units), "state_initial": np.zeros(state)}
    nodes = [helper.make_node(head, ["x", constant], ["p"]),
             helper.make_node("Svdf", ["p", "feature", "time", "bias", "state_in"], ["y", "state_out"],
                              domain="tigris", rank=rank, activation="relu")]
    graph = helper.make_graph(
        nodes, "svdf_after",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, features]),
         helper.make_tensor_value_info("state_in", TensorProto.FLOAT, state)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, units]),
         helper.make_tensor_value_info("state_out", TensorProto.FLOAT, state)],
        [numpy_helper.from_array(w.astype(np.float32), name) for name, w in weights.items()],
        value_info=[helper.make_tensor_value_info("p", TensorProto.FLOAT, [batch, features])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("tigris", 1)])
    model.ir_version = 9
    helper.set_model_props(model, {STATE_KEY: json.dumps(
        [{"input": "state_in", "output": "state_out", "initial": "state_initial"}])})
    path = tmp_path / "svdf_after.onnx"
    onnx.save(model, str(path))
    ag = full_pipeline(path, budget=4096)
    live = [[name for name in names if name in ag.tensors and not ag.tensors[name].is_constant]
            for op in ag.ops for names in (op.inputs, op.outputs)]
    for index, op in enumerate(ag.ops):
        op.stage = index
    ag.stages = [Stage(stage_id=index, op_indices=[index], input_tensors=live[2 * index],
                       output_tensors=live[2 * index + 1], peak_bytes=ag.peak_memory_bytes)
                 for index in range(len(ag.ops))]
    return ag


def test_constants_of_a_later_stage_are_in_that_stage_block(tmp_path):
    ag = _svdf_after(tmp_path, "MatMul", "projection")
    plain_data = emit_binary_bytes(ag)
    packed = read_binary_plan(emit_binary_bytes(ag, compress="lz4"))
    assert [(op["op_type"], op["stage"]) for op in packed["ops"]] == [
        (OP_TYPE_MAP["Gemm"], 0), (OP_TYPE_MAP["Svdf"], 1)]
    assert [block["stage_idx"] for block in packed["weight_blocks"]] == [0, 1]
    _assert_reads_in_stage_blocks(plain_data, read_binary_plan(plain_data), packed)


def test_a_listed_constant_shared_across_stages_is_refused(tmp_path):
    ag = _svdf_after(tmp_path, "Add", "bias")
    with pytest.raises(ValueError, match=f"Weight {list(ag.weight_data).index('bias')} shared across stages"):
        emit_binary_bytes(ag, compress="lz4")


def test_a_weight_no_stage_reads_is_refused(conv_relu_chain_path):
    ag, _, _, _ = _run_compressed_pipeline(str(conv_relu_chain_path), ("32K",))
    ag.weight_data["unread"] = np.zeros(4, np.float32)
    with pytest.raises(ValueError, match=f"weight {len(ag.weight_data) - 1} belongs to no stage"):
        emit_binary_bytes(ag, compress="lz4")
    assert compute_findings(ag).lz4_plan_size_bytes == 0


# Uncompressed plan unchanged


def test_uncompressed_plan_unchanged(conv_relu_chain_path):
    """Without --compress, no SEC_WEIGHT_BLOCKS section should appear."""
    ag = full_pipeline(conv_relu_chain_path, budget=8192)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(plan["weight_blocks"]) == 0
    assert plan["weight_blocks_compression"] == 0


def test_compressed_plan_budget_excludes_exact_weight_reservation(
    conv_relu_chain_path,
):
    ag, total, planned_reserve, required_reserve = _run_compressed_pipeline(
        str(conv_relu_chain_path), ("32K",)
    )
    reserve = compressed_weight_reserve_bytes(ag)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)

    assert reserve > 0
    assert plan["budget"] == ag.mem_budget
    assert reserve == required_reserve
    assert ag.mem_budget + planned_reserve == total


def test_compressed_plan_fails_when_total_budget_omits_reservation(
    conv_relu_chain_path,
):
    ag = full_pipeline(conv_relu_chain_path, budget=32768)
    reserve = compressed_weight_reserve_bytes(ag)
    # The activation schedule itself fits; one byte less than activation plus
    # decompression reserve must still be rejected before emission.
    from tigris.analysis.validation import validate_memory_plan

    ag.budget = replace(ag.budget, fast=validate_memory_plan(ag).scheduled_peak_bytes + reserve - 1)

    with pytest.raises(ValueError, match="Cannot emit an infeasible memory plan"):
        emit_binary_bytes(ag, compress="lz4")


def test_decompressed_weights_that_take_the_whole_budget_leave_no_plan(tmp_path):
    """No fast memory is left for activations, so no plan is written."""
    model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / "conv_bias.tflite"
    unreserved, _ = _run_pipeline(str(model), ("256",), report_bindings=False)
    assert compressed_weight_reserve_bytes(unreserved) == 256

    output = tmp_path / "conv_bias.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(model), "-m", "256", "-c", "lz4", "-o", str(output)])
    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == (
        "Error: no plan written: decompressed weights need all 256 B of fast memory, "
        "none is left for activations")
    assert not output.exists()


def test_compile_compressed_uses_total_arena_budget(
    conv_relu_chain_path, tmp_path
):
    """The CLI partitions after reserving the runtime decompression arena."""
    from click.testing import CliRunner

    from tigris.cli import cli

    output = tmp_path / "compressed.tgrs"
    result = CliRunner().invoke(
        cli,
        [
            "compile", str(conv_relu_chain_path), "-m", "32K", "-c", "lz4",
            "-o", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    plan = read_binary_plan(output.read_bytes())
    block_bytes = {
        block["stage_idx"]: (block["uncompressed_size"] + 31) & ~31
        for block in plan["weight_blocks"]
    }
    reserve = max(block_bytes.values(), default=0)
    for stage_idx, stage in enumerate(plan["stages"]):
        if stage["chain_id"] == stage_idx and stage["chain_len"] > 1:
            reserve = max(
                reserve,
                sum(
                    block_bytes.get(member_idx, 0)
                    for member_idx in range(
                        stage_idx, stage_idx + stage["chain_len"]
                    )
                ),
            )
    # The compiler uses 32-byte tensor alignment, an intentional conservative
    # upper bound over the runtime's target-specific alignment.
    assert plan["budget"] + reserve <= 32 * 1024
    assert "holds decompressed weights" in result.output

