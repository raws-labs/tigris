"""Tests for LZ4 weight compression in the binary plan emitter."""


import json
import re
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.plan_graph import plan_decompression_reserve, slow_arena_required
from tigris.cli import cli
from tigris.emitters.binary.defs import (
    COMPRESS_LZ4,
    COMPRESS_NONE,
    HEADER_STRUCT,
    NO_WEIGHT,
    OP_ATTR_CONSTANTS,
    OP_TYPE_MAP,
    PLAN_SECTION_ALIGNMENT,
    SEC_WEIGHT_BLOCKS,
    SECTION_ENTRY_STRUCT,
    WEIGHT_BLOCK_SECTION_HEADER_STRUCT,
    WEIGHT_BLOCK_STRUCT,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import (
    compressed_weight_reserve_bytes,
    emit_binary_bytes,
)
from tigris.frontends.tflite import STATE_KEY
from tigris.graph.ir import AnalyzedGraph, Stage
from tigris.pipeline import plan_model, reserve_weights
from tigris.runtime import Session
from tigris.utils import fmt_bytes
from scripts.contract_cases.fixtures import build_conv_relu_chain


# Compressed weight data matches uncompressed


def test_compressed_weight_data_matches(conv_relu_chain_path):
    """Decompressed weight data from compressed plan matches uncompressed plan."""
    ag = plan_model(str(conv_relu_chain_path), ("32K",), compress="lz4")

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


@pytest.mark.parametrize("compress", ["none", "LZ4", ""])
def test_only_lz4_compresses(conv_relu_chain_path, compress):
    ag = plan_model(str(conv_relu_chain_path), ("32K",), compress="lz4")
    message = f"unknown weight compression {compress!r}: expected None or 'lz4'"
    with pytest.raises(ValueError, match=re.escape(message)):
        emit_binary_bytes(ag, compress=compress)
    with pytest.raises(ValueError, match=re.escape(message)):
        plan_model(str(conv_relu_chain_path), ("32K",), compress=compress)


def _seeded_conv_relu_chain(tmp_path: Path) -> Path:
    """Conv -> ReLU -> Conv with seeded random weights and biases."""
    model = build_conv_relu_chain()
    rng = np.random.default_rng(0)
    for tensor in model.graph.initializer:
        values = rng.standard_normal(tuple(tensor.dims)).astype(np.float32)
        tensor.CopyFrom(numpy_helper.from_array(values, tensor.name))
    path = tmp_path / "seeded_conv_relu_chain.onnx"
    onnx.save(model, str(path))
    return path


def _blocks_from_plain_plan(packed: dict, plain_data: bytes, plain: dict) -> list[bytes]:
    """Each weight block of an LZ4 plan, assembled from the uncompressed plan's
    weight bytes at the block offsets of the LZ4 plan's weight table."""
    blocks = []
    for block in packed["weight_blocks"]:
        data = bytearray(block["uncompressed_size"])
        for k in range(block["first_weight_idx"], block["first_weight_idx"] + block["num_weights"]):
            entry, stored = packed["weights"][k], plain["weights"][k]
            assert entry["size_bytes"] == stored["size_bytes"]
            assert entry["offset"] + entry["size_bytes"] <= len(data)
            start = stored["blob_base"] + stored["offset"]
            data[entry["offset"]:entry["offset"] + entry["size_bytes"]] = (
                plain_data[start:start + stored["size_bytes"]])
        blocks.append(bytes(data))
    return blocks


def _with_stored_weight_blocks(data: bytes, blobs: list[bytes]) -> bytes:
    """Rewrite an LZ4 plan's weight-block section to hold the given decoded
    blocks, compression type 0, which the runtime loads and copies without
    decoding."""
    blocks = read_binary_plan(data, decompress_weights=False)["weight_blocks"]
    header = list(HEADER_STRUCT.unpack_from(data, 0))
    directory = []
    position = header[3]
    while (entry := SECTION_ENTRY_STRUCT.unpack_from(data, position))[0]:
        directory.append(entry)
        position += SECTION_ENTRY_STRUCT.size
    start = dict(directory)[SEC_WEIGHT_BLOCKS]
    end = min((offset for _, offset in directory if offset > start), default=len(data))
    section = bytearray(WEIGHT_BLOCK_SECTION_HEADER_STRUCT.pack(len(blocks), COMPRESS_NONE))
    blob_offset = 0
    for block, blob in zip(blocks, blobs, strict=True):
        section += WEIGHT_BLOCK_STRUCT.pack(block["stage_idx"], block["first_weight_idx"],
                                            block["num_weights"], 0, blob_offset, len(blob), len(blob))
        blob_offset += len(blob)
    section += b"".join(blobs)
    section += bytes(-len(section) % PLAN_SECTION_ALIGNMENT)
    shift = len(section) - (end - start)
    out = bytearray(data[:start] + section + data[end:])
    header[2] = len(out)
    HEADER_STRUCT.pack_into(out, 0, *header)
    for index, (kind, offset) in enumerate(directory):
        if offset > start:
            SECTION_ENTRY_STRUCT.pack_into(out, header[3] + index * SECTION_ENTRY_STRUCT.size,
                                           kind, offset + shift)
    return bytes(out)


def _stored_lz4_and_plain_plans(tmp_path: Path) -> tuple[tuple[Path, Path, Path], list[bytes], AnalyzedGraph]:
    """The seeded chain at 32K as plans with stored, LZ4 and no weight blocks,
    the decoded blocks taken from the uncompressed plan, and the planned graph."""
    ag = plan_model(str(_seeded_conv_relu_chain(tmp_path)), ("32K",), compress="lz4")
    lz4_data, plain_data = emit_binary_bytes(ag, compress="lz4"), emit_binary_bytes(ag)
    blobs = _blocks_from_plain_plan(read_binary_plan(lz4_data, decompress_weights=False),
                                    plain_data, read_binary_plan(plain_data))
    paths = (tmp_path / "stored.tgrs", tmp_path / "lz4.tgrs", tmp_path / "plain.tgrs")
    for path, data in zip(paths, (_with_stored_weight_blocks(lz4_data, blobs), lz4_data, plain_data)):
        path.write_bytes(data)
    return paths, blobs, ag


def test_the_reader_returns_stored_weight_blocks_as_written(tmp_path):
    (stored_path, lz4_path, _), blobs, ag = _stored_lz4_and_plain_plans(tmp_path)
    stored = read_binary_plan(stored_path.read_bytes())
    lz4 = read_binary_plan(lz4_path.read_bytes())

    def layout(plan):
        return [(b["stage_idx"], b["first_weight_idx"], b["num_weights"], b["uncompressed_size"])
                for b in plan["weight_blocks"]]

    assert lz4["weight_blocks_compression"] == COMPRESS_LZ4
    assert stored["weight_blocks_compression"] == COMPRESS_NONE
    # Distinct blocks, so a read at the wrong blob offset returns other bytes.
    assert len(blobs) >= 2 and len(set(blobs)) == len(blobs)
    assert [b["blob_offset"] for b in stored["weight_blocks"]] == [
        sum(map(len, blobs[:i])) for i in range(len(blobs))]
    assert all(b["compressed_size"] == b["uncompressed_size"] for b in stored["weight_blocks"])
    assert layout(stored) == layout(lz4)
    assert stored["weights"] == lz4["weights"]
    assert [b["decompressed"] for b in stored["weight_blocks"]] == blobs
    assert [b["decompressed"] for b in lz4["weight_blocks"]] == blobs
    weights = list(ag.weight_data.values())
    for block, blob in zip(stored["weight_blocks"], blobs):
        for k in range(block["first_weight_idx"], block["first_weight_idx"] + block["num_weights"]):
            entry = stored["weights"][k]
            values = np.frombuffer(blob, np.float32, entry["size_bytes"] // 4, entry["offset"])
            np.testing.assert_array_equal(np.sort(values), np.sort(weights[k].ravel()))


@pytest.mark.usefixtures("require_host_library")
def test_the_runtime_runs_a_plan_with_stored_weight_blocks(tmp_path):
    paths, _, _ = _stored_lz4_and_plain_plans(tmp_path)
    outputs = []
    for path in paths:
        with Session(path) as session:
            (info,) = session.inputs
            x = np.random.default_rng(0).standard_normal(info["shape"]).astype(info["dtype"])
            outputs.append(session.run({info["name"]: x}))
    stored, lz4, plain = outputs
    assert stored.keys() == lz4.keys() == plain.keys()
    for key in plain:
        assert np.count_nonzero(plain[key]) > 0
        np.testing.assert_array_equal(stored[key], plain[key])
        np.testing.assert_array_equal(lz4[key], plain[key])


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


def _run(plan_path: Path, golden) -> list[dict]:
    with Session(plan_path, slow_capacity=slow_arena_required(read_binary_plan(plan_path.read_bytes()))) as session:
        return [session.run({info["name"]: golden[f"input_{position}"][sample]
                             for position, info in enumerate(session.inputs)})
                for sample in range(len(golden["input_0"]))]


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("name", ["if_conv", "float_if_feature_map", "float_while"])
def test_the_reservation_holds_the_weights_of_the_stages_control_flow_runs(name, tmp_path):
    """The stages of the subgraphs If and While run are stages of the plan, so
    the fast memory reserved for decompressed weights covers their blocks."""
    model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / f"{name}.tflite"
    _compile(model, "none", tmp_path)
    output = tmp_path / "lz4.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(model), "-m", "256K", "-c", "lz4", "-o", str(output)])
    assert result.exit_code == 0, result.output
    reserve = plan_decompression_reserve(read_binary_plan(output.read_bytes()))
    assert reserve > 0
    assert (f"LZ4, {fmt_bytes(reserve)} of the 256.00 KiB fast memory holds decompressed weights"
            in result.output)
    golden = np.load(model.with_suffix(".npz"))
    for plain, packed in zip(_run(tmp_path / "none.tgrs", golden), _run(output, golden)):
        assert plain.keys() == packed.keys()
        for key in plain:
            np.testing.assert_array_equal(plain[key], packed[key])


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
    ag = plan_model(path, (4096,))
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
    reserve_weights(ag)
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
    ag = plan_model(str(conv_relu_chain_path), ("32K",), compress="lz4")
    ag.weight_data["unread"] = np.zeros(4, np.float32)
    with pytest.raises(ValueError, match=f"weight {len(ag.weight_data) - 1} belongs to no stage"):
        emit_binary_bytes(ag, compress="lz4")
    assert compute_findings(ag).lz4_plan_size_bytes == 0


# Uncompressed plan unchanged


def test_uncompressed_plan_unchanged(conv_relu_chain_path):
    """Without --compress, no SEC_WEIGHT_BLOCKS section should appear."""
    ag = plan_model(conv_relu_chain_path, (8192,), chains=False)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(plan["weight_blocks"]) == 0
    assert plan["weight_blocks_compression"] == 0


def test_compressed_plan_budget_excludes_exact_weight_reservation(
    conv_relu_chain_path,
):
    ag = plan_model(str(conv_relu_chain_path), ("32K",), compress="lz4")
    reserve = compressed_weight_reserve_bytes(ag)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)

    assert reserve > 0
    assert plan["budget"] == ag.mem_budget
    assert reserve == ag.fast_memory_reserve_bytes
    assert ag.mem_budget + reserve == 32 * 1024


def test_compressed_plan_fails_when_the_reservation_omits_weights(conv_relu_chain_path):
    ag = plan_model(conv_relu_chain_path, ("32K",), compress="lz4")
    reserve = compressed_weight_reserve_bytes(ag)
    ag.budget = replace(ag.budget, fast_reserve=reserve - 1)

    with pytest.raises(ValueError, match=re.escape(
            f"compressed-weight reservation ({reserve:,} bytes) exceeds the declared "
            f"reservation ({reserve - 1:,} bytes)")):
        emit_binary_bytes(ag, compress="lz4")


def test_a_graph_planned_without_compression_does_not_compress(conv_relu_chain_path):
    """The fast budget of a graph is its activation arena, so a graph planned
    without a weight reservation has no room for decompressed weights."""
    ag = plan_model(conv_relu_chain_path, ("32K",))
    reserve = compressed_weight_reserve_bytes(ag)

    with pytest.raises(ValueError, match=re.escape(
            f"compressed-weight reservation ({reserve:,} bytes) exceeds the declared "
            "reservation (0 bytes)")):
        emit_binary_bytes(ag, compress="lz4")


def test_decompressed_weights_that_take_the_whole_budget_leave_no_plan(tmp_path):
    """No fast memory is left for activations, so no plan is written."""
    model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / "conv_bias.tflite"
    unreserved = plan_model(str(model), ("256",))
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

