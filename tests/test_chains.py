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
    _external_outputs_keep_their_rows,
    _get_stage_spatial_params,
    detect_and_solve_chains,
    detect_chains,
    partition_spatial,
    solve_chain_tile_height,
)
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import OpNode, Stage
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
    """Conv -> Relu -> MaxPool(stride2) -> Conv.

    The pool stage must participate in the streamable chain with its stride and
    receptive field represented in the chain geometry.
    """
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


_POOL_OP_TYPES = {"MaxPool", "AveragePool"}


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


def test_pool_stage_is_chained_with_spatial_geometry(chain_with_pool_path):
    ag, chains = _stages_then_chains(chain_with_pool_path, budget=24000)
    pooled_stages = {
        si
        for group in chains
        for si in group
        if {
            ag.ops[oi].op_type for oi in ag.stages[si].op_indices
        } & _POOL_OP_TYPES
    }

    assert pooled_stages, "expected the MaxPool stage in a streamable chain"
    for si in pooled_stages:
        assert _get_stage_spatial_params(ag, ag.stages[si]) == (2, 2, 1)


def test_chain_does_not_span_model_output(chain_with_model_output_mid_path):
    """A chain intermediate that is also a model output must not be absorbed into
    a chain (#7), else it is never written."""
    ag, chains = _stages_then_chains(chain_with_model_output_mid_path, budget=24000)
    boundaries = _chain_boundary_tensors(ag, chains)
    assert not (boundaries & set(ag.model_outputs)), (
        f"model output streamed as chain intermediate: {boundaries & set(ag.model_outputs)}")


class TestExternalOutputRows:
    """A stage output written before a spatial op keeps that op's input rows."""

    @staticmethod
    def _stage(stride):
        ops = [
            OpNode("point", "Conv", ["x"], ["skip"],
                   {"kernel_shape": [1, 1], "strides": [1, 1]}),
            OpNode("deep", "Conv", ["skip"], ["wide"],
                   {"kernel_shape": [3, 3], "strides": [stride, stride],
                    "pads": [1, 1, 1, 1]}),
        ]
        stage = Stage(0, [0, 1], ["x"], ["skip", "wide"])
        return stage, ops

    def test_unit_stride_after_an_escaping_output_is_admitted(self):
        stage, ops = self._stage(1)
        assert _external_outputs_keep_their_rows(stage, ops)

    def test_a_stride_after_an_escaping_output_is_refused(self):
        """At stride 2 the tiles never read the rows the strided op drops, so
        nothing would ever write them into the escaping tensor."""
        stage, ops = self._stage(2)
        assert not _external_outputs_keep_their_rows(stage, ops)

    def test_a_stride_before_the_only_escaping_output_is_admitted(self):
        ops = [
            OpNode("deep", "Conv", ["x"], ["wide"],
                   {"kernel_shape": [3, 3], "strides": [2, 2],
                    "pads": [1, 1, 1, 1]}),
            OpNode("act", "Relu", ["wide"], ["out"], {}),
        ]
        stage = Stage(0, [0, 1], ["x"], ["out"])
        assert _external_outputs_keep_their_rows(stage, ops)


def _gate_model(path, *, between_changes_rows: bool, gate_escapes: bool = False):
    """Conv -> Sigmoid -> Mul(conv, sigmoid) -> Conv, the shape of a SiLU.

    The Mul reads the tensor the Sigmoid read, so the chain has to carry it
    past the Sigmoid. `between_changes_rows` puts a strided convolution in
    that span instead, which leaves the two operands describing different
    rows. `gate_escapes` gives the Sigmoid's output a second reader after the
    chain, which nothing streams to.
    """
    c, side = 8, 24
    rng = np.random.default_rng(0)
    first = rng.normal(size=(c, c, 3, 3)).astype(np.float32) * 0.2
    second = rng.normal(size=(c, c, 3, 3)).astype(np.float32) * 0.2
    inits = [
        helper.make_tensor("first", TensorProto.FLOAT, [c, c, 3, 3],
                           first.flatten().tolist()),
        helper.make_tensor("second", TensorProto.FLOAT, [c, c, 3, 3],
                           second.flatten().tolist()),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "first"], ["gated"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    ]
    if between_changes_rows:
        nodes.append(helper.make_node("MaxPool", ["gated"], ["pooled"],
                                      kernel_shape=[2, 2], strides=[2, 2]))
        nodes.append(helper.make_node("Sigmoid", ["pooled"], ["gate"]))
        nodes.append(helper.make_node("Mul", ["gated", "gate"], ["silu"]))
    else:
        nodes.append(helper.make_node("Sigmoid", ["gated"], ["gate"]))
        nodes.append(helper.make_node("Mul", ["gated", "gate"], ["silu"]))
    nodes.append(helper.make_node("Conv", ["silu", "second"], ["convolved"],
                                 kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    if gate_escapes:
        nodes.append(helper.make_node("Add", ["convolved", "gate"], ["output"]))
    else:
        nodes.append(helper.make_node("Relu", ["convolved"], ["output"]))

    graph = helper.make_graph(
        nodes, "gate",
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, c, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, c, side, side])],
        initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


class TestCarriedOperand:
    """A chain may carry one stage's output past the next stage."""

    def test_a_gate_is_carried_through_the_chain(self, tmp_path):
        path = _gate_model(tmp_path / "gate.onnx", between_changes_rows=False)
        ag, chains = _stages_then_chains(path, budget=16 * 1024)
        carried = [
            group for group in chains
            if any(len(ag.stages[i].input_tensors) > 1 for i in group[1:])
        ]
        assert carried, "expected the Mul to join the chain it gates"
        group = carried[0]
        types = [
            [ag.ops[o].op_type for o in ag.stages[i].op_indices]
            for i in group
        ]
        assert ["Sigmoid"] in types and ["Mul"] in types

    def test_a_row_change_in_between_refuses_the_carry(self, tmp_path):
        """The carried tile holds the producer's rows, so anything that
        changes rows in between leaves the two operands disagreeing."""
        path = _gate_model(tmp_path / "pooled.onnx", between_changes_rows=True)
        ag, chains = _stages_then_chains(path, budget=16 * 1024)
        for group in chains:
            for index in group[1:]:
                assert len(ag.stages[index].input_tensors) <= 1

    def test_a_reader_after_the_chain_refuses_the_carry(self, tmp_path):
        """Nothing writes a streamed tensor to slow memory, so a reader past
        the chain would find nothing there."""
        path = _gate_model(
            tmp_path / "escapes.onnx", between_changes_rows=False,
            gate_escapes=True)
        ag, chains = _stages_then_chains(path, budget=16 * 1024)
        readers: dict[str, set[int]] = {}
        for index, stage in enumerate(ag.stages):
            for name in stage.input_tensors:
                readers.setdefault(name, set()).add(index)
        for group in chains:
            inside = set(group)
            for index in group[:-1]:
                for name in ag.stages[index].output_tensors:
                    assert readers.get(name, set()) <= inside


def _strided_stem_model(path):
    """Three stride-2 convolutions down to 16x16, then three wide layers."""
    rng = np.random.default_rng(0)
    layers = [(3, 8, 3, 2), (8, 16, 3, 2), (16, 32, 3, 2), (32, 128, 1, 1), (128, 128, 3, 1), (128, 16, 1, 1)]
    nodes, inits, previous = [], [], "input"
    for index, (cin, cout, kernel, stride) in enumerate(layers):
        inits.append(helper.make_tensor(
            f"w{index}", TensorProto.FLOAT, [cout, cin, kernel, kernel],
            rng.standard_normal((cout, cin, kernel, kernel)).astype(np.float32).ravel().tolist()))
        out = "output" if index == len(layers) - 1 else f"t{index}"
        nodes.append(helper.make_node(
            "Conv", [previous, f"w{index}"], [out], kernel_shape=[kernel, kernel],
            strides=[stride, stride], pads=[kernel // 2] * 4))
        previous = out
    graph = helper.make_graph(
        nodes, "stem",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 16])],
        inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    return path


def test_chain_geometry_takes_the_kernel_from_the_weight(tmp_path):
    """A convolution without kernel_shape keeps its 5x5 window in the chain model."""
    rng = np.random.default_rng(0)
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [16, 8, 1, 1],
                            rng.standard_normal((16, 8, 1, 1)).astype(np.float32).ravel().tolist())
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [16, 1, 5, 5],
                            rng.standard_normal((16, 1, 5, 5)).astype(np.float32).ravel().tolist())
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w1"], ["a"]),
         helper.make_node("Conv", ["a", "w2"], ["output"], group=16,
                          strides=[2, 2], pads=[1, 1, 2, 2])],
        "k5",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 16])],
        [w1, w2])
    path = str(tmp_path / "k5.onnx")
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    ag = load_model(path)
    ag.stages = [Stage(stage_id=0, op_indices=[0], input_tensors=["input"], output_tensors=["a"]),
                 Stage(stage_id=1, op_indices=[1], input_tensors=["a"], output_tensors=["output"])]

    assert _get_stage_spatial_params(ag, ag.stages[1]) == (5, 2, 1)
    heights = _back_propagate_tile_heights(
        [_get_stage_spatial_params(ag, s) for s in ag.stages], 1)
    assert heights == [(5, 5), (5, 1)]
    # Five input rows of 8 channels, five rows of the 16-channel 1x1 output,
    # one output row of the depthwise, each 32-byte aligned.
    assert _chain_fast_bytes(ag, ag.stages, heights) == (
        5 * 32 * 8 * 4 + 5 * 32 * 16 * 4 + 16 * 16 * 4)


class TestSplitChains:
    """A run too big to stream whole still streams as consecutive chains."""

    budget = 48000

    def _stages(self, tmp_path):
        ag = load_model(_strided_stem_model(str(tmp_path / "stem.onnx")))
        ag = compute_memory_timeline(compute_lifetimes(ag))
        return partition_spatial(partition_temporal(ag, self.budget))

    def test_the_whole_run_does_not_fit(self, tmp_path):
        ag = self._stages(tmp_path)
        runs = detect_chains(ag)
        assert len(runs) == 1 and len(runs[0]) == len(ag.stages)
        assert solve_chain_tile_height(ag, runs[0]) == 0

    def test_the_run_streams_as_fitting_pieces(self, tmp_path):
        ag = detect_and_solve_chains(self._stages(tmp_path))
        heads = [s for s in ag.stages if s.chain_len >= 2 and s.chain_id == s.stage_id]
        assert len(heads) >= 2
        covered = []
        for head in heads:
            piece = list(range(head.stage_id, head.stage_id + head.chain_len))
            assert all(ag.stages[i].chain_id == head.stage_id for i in piece)
            assert head.chain_tile_h > 0
            assert head.chain_tile_h == solve_chain_tile_height(ag, piece)
            covered += piece
        assert covered == sorted(set(covered))

    def test_the_stem_intermediates_stay_in_fast_memory(self, tmp_path):
        ag = detect_and_solve_chains(self._stages(tmp_path))
        first = ag.stages[0]
        assert first.chain_len >= 2
        streamed = [ag.stages[i].output_tensors[0] for i in range(first.chain_len - 1)]
        assert "t0" in streamed


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
        """A reader outside the chain keeps the chain from reaching it.

        The Add takes the Relu's output as well as the convolution's, and a
        chain only streams what it also consumes. Here the convolution between
        the two changes the rows, so the Add cannot join and read the carried
        tensor; the chain has to stop before the Relu's output leaves it.
        """
        ag = _full_pipeline(chain_with_fanout_path, budget=32000)

        readers: dict[str, set[int]] = {}
        for index, stage in enumerate(ag.stages):
            for name in stage.input_tensors:
                readers.setdefault(name, set()).add(index)

        for s in ag.stages:
            if s.chain_len < 2:
                continue
            inside = set(range(s.chain_id, s.chain_id + s.chain_len))
            for index in sorted(inside)[:-1]:
                for name in ag.stages[index].output_tensors:
                    assert readers.get(name, set()) <= inside, (
                        f"{name} is streamed but read outside the chain")

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

    def test_back_propagate_pool_stride2(self):
        """Pool2x2(s=2) uses the same spatial range algebra as convolution."""
        heights = _back_propagate_tile_heights([(2, 2, 1)], 3)
        assert heights[0] == (6, 3)

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
                chain_params = [_get_stage_spatial_params(ag, cs) for cs in chain_stages]
                heights = _back_propagate_tile_heights(chain_params, s.chain_tile_h)
                needed = _chain_fast_bytes(ag, chain_stages, heights)
                assert needed <= ag.mem_budget, (
                    f"Chain tile buffer {needed} exceeds budget {ag.mem_budget}"
                )
