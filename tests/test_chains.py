"""Tests for streamable chain detection and tile-through execution."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import (
    _back_propagate_tile_heights,
    _chain_fast_bytes,
    detect_and_solve_chains,
    detect_chains,
    partition_spatial,
    solve_chain_tile_height,
)
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.loaders import load_model


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
        ag = detect_and_solve_chains(ag)
    return ag


# Fixtures


@pytest.fixture
def three_conv_chain_path(tmp_path):
    """3 consecutive Conv3x3(pad=1)+Relu on [1,3,64,64].

    With a budget smaller than the peak intermediate tensor (65536 bytes)
    but larger than the chain tile buffer for tile_h=1 (~24 KB),
    the partitioner creates 5 stages that form a streamable chain.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 64, 64])

    # Conv0: 3->4 channels
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3],
                            np.random.randn(4, 3, 3, 3).astype(np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())

    # Conv1: 4->4 channels
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 4, 3, 3],
                            np.random.randn(4, 4, 3, 3).astype(np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())

    # Conv2: 4->8 channels
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [8, 4, 3, 3],
                            np.random.randn(8, 4, 3, 3).astype(np.float32).flatten().tolist())
    b2 = helper.make_tensor("b2", TensorProto.FLOAT, [8],
                            np.zeros(8, dtype=np.float32).tolist())

    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu0 = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["t2"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu1 = helper.make_node("Relu", ["t2"], ["t3"], name="relu1")
    conv2 = helper.make_node("Conv", ["t3", "w2", "b2"], ["output"], name="conv2",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])

    graph = helper.make_graph(
        [conv0, relu0, conv1, relu1, conv2], "three_conv_chain",
        [X], [Y], initializer=[w0, b0, w1, b1, w2, b2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "three_conv_chain.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def chain_with_fanout_path(tmp_path):
    """Conv -> Relu -> Conv, but the Relu output is also consumed by an Add.

    This breaks the chain because the intermediate tensor has fan-out.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 64, 64])

    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3],
                            np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 4, 3, 3],
                            np.zeros((4, 4, 3, 3), dtype=np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())

    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["t2"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    # Add uses t1 (fan-out) and t2
    add = helper.make_node("Add", ["t1", "t2"], ["output"], name="add0")

    graph = helper.make_graph(
        [conv0, relu, conv1, add], "chain_fanout",
        [X], [Y], initializer=[w0, b0, w1, b1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "chain_fanout.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def chain_with_pool_path(tmp_path):
    """Conv -> Relu -> MaxPool(stride2) -> Conv. The strided pool sits where a
    chain would otherwise form. exec_chain_tiled composes receptive fields for
    Conv/DepthwiseConv only, so a pool inside a chain would get height-preserving
    (pointwise) tile geometry -> wrong rows / OOB. The pool stage must stay out of
    any chain."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 32, 32])
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3],
                            np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32).tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [8, 4, 3, 3],
                            np.zeros((8, 4, 3, 3), dtype=np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [8], np.zeros(8, dtype=np.float32).tolist())
    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    pool = helper.make_node("MaxPool", ["t1"], ["t2"], name="pool0",
                            kernel_shape=[2, 2], strides=[2, 2])
    conv1 = helper.make_node("Conv", ["t2", "w1", "b1"], ["output"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    graph = helper.make_graph([conv0, relu, pool, conv1], "chain_pool",
                              [X], [Y], initializer=[w0, b0, w1, b1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "chain_pool.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def chain_with_model_output_mid_path(tmp_path):
    """3 convs where the middle conv's output (t2) is ALSO a model output. The
    chain intermediate is streamed tile-by-tile and never materialized to slow
    memory, so chaining across t2 would leave that model output unwritten. The
    chain must break at t2."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 64, 64])
    Y2 = helper.make_tensor_value_info("t2", TensorProto.FLOAT, [1, 4, 64, 64])
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3],
                            np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32).tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 4, 3, 3],
                            np.zeros((4, 4, 3, 3), dtype=np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32).tolist())
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [8, 4, 3, 3],
                            np.zeros((8, 4, 3, 3), dtype=np.float32).flatten().tolist())
    b2 = helper.make_tensor("b2", TensorProto.FLOAT, [8], np.zeros(8, dtype=np.float32).tolist())
    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["t2"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    conv2 = helper.make_node("Conv", ["t2", "w2", "b2"], ["output"], name="conv2",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    graph = helper.make_graph([conv0, relu, conv1, conv2], "chain_mid_output",
                              [X], [Y, Y2], initializer=[w0, b0, w1, b1, w2, b2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "chain_mid_output.onnx"
    onnx.save(model, str(path))
    return path


_POOL_OP_TYPES = {"MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"}


def _stages_then_chains(path, budget):
    """Run the pipeline up to stage creation, then return (ag, raw chain groups)."""
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    ag = partition_temporal(ag, budget)
    ag = partition_spatial(ag)
    return ag, detect_chains(ag)


def _chain_boundary_tensors(ag, chains):
    """Tensors streamed between consecutive chained stages (never materialized)."""
    boundaries: set[str] = set()
    for group in chains:
        for a, b in zip(group, group[1:]):
            boundaries |= set(ag.stages[a].output_tensors) & set(ag.stages[b].input_tensors)
    return boundaries


def test_pool_stage_not_chained(chain_with_pool_path):
    """A pool op must never appear inside a chained stage (#5: exec_chain_tiled
    handles only Conv/DepthwiseConv as spatial)."""
    ag, chains = _stages_then_chains(chain_with_pool_path, budget=24000)
    for group in chains:
        for si in group:
            optypes = {ag.ops[oi].op_type for oi in ag.stages[si].op_indices}
            assert not (optypes & _POOL_OP_TYPES), (
                f"pool op chained in stage {si}: {optypes}")


def test_chain_does_not_span_model_output(chain_with_model_output_mid_path):
    """A chain intermediate that is also a model output must not be absorbed into
    a chain (#7), else it is never written."""
    ag, chains = _stages_then_chains(chain_with_model_output_mid_path, budget=24000)
    boundaries = _chain_boundary_tensors(ag, chains)
    assert not (boundaries & set(ag.model_outputs)), (
        f"model output streamed as chain intermediate: {boundaries & set(ag.model_outputs)}")


# Chain detection tests


class TestDetectChains:
    def test_linear_chain_detected(self, three_conv_chain_path):
        """Three consecutive Conv+Relu stages should form a single chain."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        # Should have multiple stages (tight budget forces partitioning)
        assert len(ag.stages) >= 2

        # Find stages with chain_len > 0
        chained = [s for s in ag.stages if s.chain_len >= 2]
        assert len(chained) >= 2, f"Expected chain stages, got: {[(s.stage_id, s.chain_id, s.chain_len) for s in ag.stages]}"

        # All chained stages should share the same chain_id
        chain_ids = set(s.chain_id for s in chained)
        assert len(chain_ids) == 1

    def test_fanout_breaks_chain(self, chain_with_fanout_path):
        """Fan-out on intermediate tensor should prevent chaining."""
        ag = _full_pipeline(chain_with_fanout_path, budget=32000)

        # If there are chains, they should not span across the fan-out point
        # The Add stage consumes t1 (produced by Relu stage) AND t2 (produced by Conv1 stage)
        # So the Add stage has 2 inputs -> can't be part of a chain
        for s in ag.stages:
            if s.chain_len >= 2:
                # Verify the chain doesn't include stages with multi-input
                chain_stages = [
                    ag.stages[i]
                    for i in range(s.chain_id, s.chain_id + s.chain_len)
                ]
                for cs in chain_stages[1:]:  # skip first (can have any input)
                    assert len(cs.input_tensors) <= 1

    def test_no_chains_when_all_fits(self, three_conv_chain_path):
        """With a huge budget, everything fits in one stage - no chains."""
        ag = _full_pipeline(three_conv_chain_path, budget=10 * 1024 * 1024)

        # With a big budget, likely 1 stage or few stages that all fit
        chains = detect_chains(ag)
        # Chains require at least 2 stages
        if len(ag.stages) < 2:
            assert len(chains) == 0

    def test_chain_head_has_tile_h(self, three_conv_chain_path):
        """The chain head stage should have chain_tile_h > 0."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        heads = [s for s in ag.stages if s.chain_len >= 2 and s.chain_id == s.stage_id]
        for h in heads:
            assert h.chain_tile_h > 0, f"Chain head {h.stage_id} has tile_h=0"


# Tile height solver tests


class TestChainTileSolver:
    def test_back_propagate_pointwise(self):
        """Pointwise chain: tile heights are identical throughout."""
        # 3 pointwise stages: k=1, s=1, d=1
        params = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        heights = _back_propagate_tile_heights(params, 4)
        # All (in_h, out_h) should be (4, 4)
        for in_h, out_h in heights:
            assert in_h == 4
            assert out_h == 4

    def test_back_propagate_conv3x3_stride1(self):
        """Conv3x3(s=1, d=1) chain: each stage adds 2 rows of halo."""
        # 2 conv stages: k=3, s=1, d=1
        params = [(3, 1, 1), (3, 1, 1)]
        heights = _back_propagate_tile_heights(params, 4)

        # Last stage: out=4, in = 4*1 + (3-1) = 6
        assert heights[1] == (6, 4)
        # First stage: out=6, in = 6*1 + (3-1) = 8
        assert heights[0] == (8, 6)

    def test_back_propagate_stride2(self):
        """Conv3x3(s=2): out_h rows need stride*out_h + kernel-stride input rows."""
        params = [(3, 2, 1)]
        heights = _back_propagate_tile_heights(params, 3)
        # out=3, in = 3*2 + (3-2) = 7
        assert heights[0] == (7, 3)

    def test_solver_returns_positive(self, three_conv_chain_path):
        """Solver should find a valid tile height > 0."""
        ag = load_model(three_conv_chain_path)
        ag = compute_lifetimes(ag)
        ag = compute_memory_timeline(ag)
        ag = partition_temporal(ag, 32000)
        ag = partition_spatial(ag)

        chains = detect_chains(ag)
        assert len(chains) > 0, "Expected at least one chain"
        for chain in chains:
            tile_h = solve_chain_tile_height(ag, chain)
            assert tile_h > 0, f"Chain {chain} solver returned tile_h=0"


# Plan format round-trip


class TestChainPlanFormat:
    def test_chain_fields_roundtrip(self, three_conv_chain_path):
        """Chain fields should survive binary plan write -> read round-trip."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        # Must have chains for this test to be meaningful
        chained = [s for s in ag.stages if s.chain_len >= 2]
        if not chained:
            pytest.skip("No chains detected (budget may be too loose)")

        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)

        # Verify chain fields in read-back plan
        for i, stage in enumerate(plan["stages"]):
            ir_stage = ag.stages[i]
            assert stage["chain_id"] == (ir_stage.chain_id & 0xFFFF)
            assert stage["chain_len"] == (ir_stage.chain_len & 0xFFFF)
            assert stage["chain_tile_h"] == (ir_stage.chain_tile_h & 0xFFFF)

    def test_standalone_stages_have_no_chain(self, three_conv_chain_path):
        """Stages not in a chain should have chain_id=0xFFFF, chain_len=0."""
        ag = _full_pipeline(three_conv_chain_path, budget=10 * 1024 * 1024)

        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)

        for stage in plan["stages"]:
            assert stage["chain_id"] == 0xFFFF
            assert stage["chain_len"] == 0

    def test_chain_clears_individual_tile_plans(self, three_conv_chain_path):
        """Stages in a chain should not have individual tile plans."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        chained = [s for s in ag.stages if s.chain_len >= 2]
        for s in chained:
            assert s.tile_plan is None, f"Chain stage {s.stage_id} should not have individual tile_plan"


# Chain execution E2E tests


class TestChainExecution:
    def test_chain_vs_no_chain_same_graph_structure(self, three_conv_chain_path):
        """Pipeline with tight budget (chain) vs loose budget (no chain) should
        produce the same ops and tensors, differing only in stage metadata."""
        ag_chain = _full_pipeline(three_conv_chain_path, budget=32000)
        ag_loose = _full_pipeline(three_conv_chain_path, budget=10 * 1024 * 1024)

        # Same ops
        assert len(ag_chain.ops) == len(ag_loose.ops)
        for a, b in zip(ag_chain.ops, ag_loose.ops):
            assert a.op_type == b.op_type
            assert a.inputs == b.inputs
            assert a.outputs == b.outputs

        # Same tensors (keys)
        assert set(ag_chain.tensors.keys()) == set(ag_loose.tensors.keys())

    def test_chain_fields_survive_binary_roundtrip(self, three_conv_chain_path):
        """Full pipeline -> emit binary -> read back -> chain fields intact."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        chained = [s for s in ag.stages if s.chain_len >= 2]
        if not chained:
            pytest.skip("No chains detected")

        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)

        # Find chain head in the read-back plan (stage index == chain_id)
        heads = [(i, s) for i, s in enumerate(plan["stages"])
                 if s["chain_len"] >= 2 and s["chain_id"] == i]
        assert len(heads) >= 1, "Expected at least one chain head in read-back plan"

        for idx, head in heads:
            assert head["chain_tile_h"] > 0, "Chain head should have tile_h > 0"
            chain_id = head["chain_id"]
            chain_len = head["chain_len"]
            # All stages in this chain should share chain_id and chain_len
            for i in range(chain_id, chain_id + chain_len):
                s = plan["stages"][i]
                assert s["chain_id"] == chain_id
                assert s["chain_len"] == chain_len

    def test_chain_tile_h_fits_budget(self, three_conv_chain_path):
        """Solved chain_tile_h should produce tile buffers within budget."""
        ag = _full_pipeline(three_conv_chain_path, budget=32000)

        for s in ag.stages:
            if s.chain_len >= 2 and s.chain_id == s.stage_id:
                # This is a chain head - verify the tile buffer fits
                chain_indices = list(range(s.chain_id, s.chain_id + s.chain_len))
                chain_stages = [ag.stages[i] for i in chain_indices]
                chain_params = [
                    (
                        *[1, 1, 1],  # default
                    )
                    for _ in chain_stages
                ]
                # Re-derive params properly
                from tigris.analysis.partition_spatial import _get_stage_spatial_params
                chain_params = [_get_stage_spatial_params(ag, cs) for cs in chain_stages]
                heights = _back_propagate_tile_heights(chain_params, s.chain_tile_h)
                needed = _chain_fast_bytes(ag, chain_stages, heights)
                assert needed <= ag.mem_budget, (
                    f"Chain tile buffer {needed} exceeds budget {ag.mem_budget}"
                )
