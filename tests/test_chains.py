"""Tests for streamable chain detection and tile-through execution."""

from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import (
    _back_propagate_tile_heights,
    _chain_fast_bytes,
    _ag_weight_shapes,
    _external_outputs_keep_their_rows,
    _get_stage_spatial_params,
    _ops_read_the_running_band,
    _stage_2d_eligible,
    detect_and_solve_chains,
    detect_chains,
    partition_spatial,
    solve_chain_tile_height,
)
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.defs import TILE_BAND_KIND_HW
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import OpNode, Stage
from tigris.loaders import load_model
from tigris.pipeline import plan_model

from conftest import save_onnx


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

    return save_onnx(tmp_path, "three_conv_chain", [conv0, relu0, conv1, relu1, conv2], [X], [Y],
                     [w0, b0, w1, b1, w2, b2], opset=17, ir_version=8)


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

    return save_onnx(tmp_path, "chain_fanout", [conv0, relu, conv1, add], [X], [Y],
                     [w0, b0, w1, b1], opset=17, ir_version=8)


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
    return save_onnx(tmp_path, "chain_pool", [conv0, relu, pool, conv1], [X], [Y], [w0, b0, w1, b1],
                     opset=17, ir_version=8)


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
    return save_onnx(tmp_path, "chain_mid_output", [conv0, relu, conv1, conv2], [X], [Y, Y2],
                     [w0, b0, w1, b1, w2, b2], opset=17, ir_version=8)


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
        assert _get_stage_spatial_params(ag, ag.stages[si], _ag_weight_shapes(ag)) == (2, 2)


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

    return save_onnx(path.parent, "gate", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c, side, side])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c, side, side])],
                     inits, opset=17, ir_version=8, filename=path.name)


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


def _branch_model(path, *, strided):
    """A stem, then a convolution whose input a second convolution also reads.

    Strided, it is the root of a DLA: Conv(stride 2) -> MaxPool -> Conv 1x1,
    and a Conv 1x1 off the strided output that a stride-2 Conv takes to the
    Add. At unit stride both the middle op and the branch are 3x3 Convs.
    Ops: 0 and 1 the stem, 2 the shared producer, 3 and 4 the main path, 5
    the branch, 6 and 7 the join.
    """
    if strided:
        side, c0, c1, c2, stride, branch = 32, 16, 32, 64, 2, 1
    else:
        side, c0, c1, c2, stride, branch = 32, 4, 8, 8, 1, 3
    rng = np.random.default_rng(0)
    inits = []

    def weight(name, shape):
        values = rng.standard_normal(shape).astype(np.float32) * 0.3
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, shape, values.ravel().tolist()))
        return name

    def conv(source, name, cin, cout, kernel, output, step=1):
        return helper.make_node(
            "Conv", [source, weight(name, [cout, cin, kernel, kernel])], [output],
            kernel_shape=[kernel, kernel], strides=[step, step], pads=[kernel // 2] * 4)

    middle = (helper.make_node("MaxPool", ["shared"], ["pooled"], kernel_shape=[2, 2], strides=[2, 2])
              if strided else conv("shared", "wm", c1, c1, 3, "pooled"))
    nodes = [
        conv("input", "w0", 3, c0, 7, "a"),
        conv("a", "w1", c0, c0, 3, "b"),
        conv("b", "w2", c0, c1, 3, "shared", stride),
        middle,
        conv("pooled", "w3", c1, c2, 1, "main"),
        conv("shared", "w4", c1, c2, branch, "branch"),
        conv("branch", "w5", c2, c2, 3, "joined", stride),
        helper.make_node("Add", ["main", "joined"], ["output"]),
    ]
    out = side // stride // stride
    graph = helper.make_graph(
        nodes, "branch",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c2, out, out])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _fire_model(path):
    """A Conv stem, then a SqueezeNet fire module: Conv 1x1 squeeze, Conv 1x1
    and Conv 3x3 expands that both read the squeeze, Concat, then a Conv.
    Ops: 0 the stem, 1 the squeeze, 2 and 3 the expands, 4 the Concat, 5 the Conv.
    """
    rng = np.random.default_rng(0)
    inits = []

    def conv(source, name, cin, cout, kernel, output):
        values = rng.standard_normal((cout, cin, kernel, kernel)).astype(np.float32) * 0.3
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, values.shape, values.ravel().tolist()))
        return helper.make_node("Conv", [source, name], [output], kernel_shape=[kernel, kernel],
                                pads=[kernel // 2] * 4)

    nodes = [
        conv("input", "w0", 3, 8, 3, "stem"),
        conv("stem", "w1", 8, 4, 1, "squeeze"),
        conv("squeeze", "w2", 4, 8, 1, "wide"),
        conv("squeeze", "w3", 4, 8, 3, "deep"),
        helper.make_node("Concat", ["wide", "deep"], ["joined"], axis=1),
        conv("joined", "w4", 16, 8, 3, "output"),
    ]
    graph = helper.make_graph(
        nodes, "fire",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 32, 32])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


# A Conv with a 1 x k kernel: (k, width stride, width pad).
_ROW_CONVS = {"conv1x1s2": (1, 2, 0), "conv1x3": (3, 1, 0), "conv1x3p1": (3, 1, 1)}


def _stripe_model(path, middle):
    """A Conv stem to x, then `middle` to b with a Relu of x to c, then b + c.

    Ops: 0 the stem, 1 the middle op, 2 the Relu, 3 the join (a Resize of c
    first when the middle op is a Resize, and a MaxPool of c with the same
    window when it is a 1 x k Conv, so the operands agree).
    """
    rng = np.random.default_rng(0)
    inits = []

    def conv(source, name, cin, cout, kernel, output):
        values = rng.standard_normal((cout, cin, kernel, kernel)).astype(np.float32) * 0.3
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, values.shape, values.ravel().tolist()))
        return helper.make_node("Conv", [source, name], [output], kernel_shape=[kernel, kernel],
                                pads=[kernel // 2] * 4)

    def resize(source, output):
        return helper.make_node("Resize", [source, "", "scales"], [output], mode="nearest",
                                coordinate_transformation_mode="asymmetric", nearest_mode="floor")

    side = width = 32
    nodes = [conv("input", "w0", 3, 8, 3, "x")]
    if middle in _ROW_CONVS:
        kernel, stride, pad = _ROW_CONVS[middle]
        window = {"kernel_shape": [1, kernel], "strides": [1, stride], "pads": [0, pad, 0, pad]}
        width = (32 + 2 * pad - kernel) // stride + 1
        values = rng.standard_normal((8, 8, 1, kernel)).astype(np.float32) * 0.3
        inits.append(helper.make_tensor("w1", TensorProto.FLOAT, values.shape, values.ravel().tolist()))
        nodes += [helper.make_node("Conv", ["x", "w1"], ["b"], **window),
                  helper.make_node("Relu", ["x"], ["c"]),
                  helper.make_node("MaxPool", ["c"], ["narrow"], **window),
                  helper.make_node("Add", ["b", "narrow"], ["output"])]
    elif middle == "resize":
        side = width = 64
        inits.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
        nodes += [resize("x", "b"), helper.make_node("Relu", ["x"], ["c"]), resize("c", "wide"),
                  helper.make_node("Add", ["b", "wide"], ["output"])]
    else:
        kernel = {"conv3": 3, "conv1": 1}[middle]
        nodes += [conv("x", "w1", 8, 8, kernel, "b"), helper.make_node("Relu", ["x"], ["c"]),
                  helper.make_node("Add", ["b", "c"], ["output"])]
    return save_onnx(path.parent, "stripe", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, side, width])],
                     inits, ir_version=8, filename=path.name)


def _wide_model(path, kernel):
    """A Conv stem to x of 4 x 256, then a 1 x `kernel` Conv that keeps the
    width to b with a Relu of x to c, then b + c. Ops as in _stripe_model."""
    rng = np.random.default_rng(0)
    inits = []
    for name, shape in (("w0", (16, 3, 3, 3)), ("w1", (16, 16, 1, kernel))):
        values = rng.standard_normal(shape).astype(np.float32) * 0.3
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, values.shape, values.ravel().tolist()))
    nodes = [helper.make_node("Conv", ["input", "w0"], ["x"], kernel_shape=[3, 3], pads=[1] * 4),
             helper.make_node("Conv", ["x", "w1"], ["b"], kernel_shape=[1, kernel],
                              pads=[0, kernel // 2, 0, kernel // 2]),
             helper.make_node("Relu", ["x"], ["c"]),
             helper.make_node("Add", ["b", "c"], ["output"])]
    return save_onnx(path.parent, "wide", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 256])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 4, 256])],
                     inits, ir_version=8, filename=path.name)


def _stripe_stages(path, budget):
    """Stages {0}, {1, 2} and the rest, partitioned where they fit, then
    tiled at `budget`; the temporal partition never keeps more than one op in
    a stage it cannot fit, so this is the only way such a stage is tiled."""
    ag = compute_memory_timeline(compute_lifetimes(load_model(path)))
    ag = partition_temporal(ag, 1 << 22, frozenset(range(len(ag.ops))) - {2})
    ag.budget = replace(ag.budget, fast=budget)
    return partition_spatial(ag)


def _run_against_onnxruntime(path, plan):
    import onnxruntime as ort
    from tigris.runtime import Session

    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).standard_normal(reference.get_inputs()[0].shape).astype(np.float32)
    expected = reference.run(None, {"input": x})[0].transpose(0, 2, 3, 1)
    with Session(plan) as session:
        actual = session.run({"input": np.ascontiguousarray(x.transpose(0, 2, 3, 1))})["output"]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * np.abs(expected).max())


_RUNNING_BAND_MODELS = {
    "strided": lambda path: _branch_model(path, strided=True),
    "unit_stride": lambda path: _branch_model(path, strided=False),
    "fire": _fire_model,
}

# Partitions whose chains read off the band unless the compiler keeps them out.
_OFF_BAND_CHAINS = [
    ("strided", frozenset(), 120 * 1024),
    ("unit_stride", frozenset({1, 2, 6}), 100 * 1024),
    ("fire", frozenset({1, 5}), 160 * 1024),
]


class TestRunningBand:
    """Every op of a chained stage reads the band the chain executor holds."""

    @pytest.mark.parametrize("strided", [True, False])
    def test_a_branch_off_an_earlier_band_is_refused(self, tmp_path, strided):
        ag = load_model(_branch_model(tmp_path / "branch.onnx", strided=strided))
        stage = Stage(0, [2, 3, 4, 5], ["b"], ["main", "branch"])
        assert not _ops_read_the_running_band(ag, stage)
        assert _ops_read_the_running_band(ag, Stage(0, [2, 3, 4], ["b"], ["shared", "main"]))

    def test_a_one_to_one_convolution_keeps_the_band(self, tmp_path):
        """The 1x1 expand leaves the rows where they are, so the 3x3 expand
        after it still reads the band; the Concat then joins the 1x1 output,
        which holds the 3x3's input rows, with the 3x3 output."""
        ag = load_model(_fire_model(tmp_path / "fire.onnx"))
        assert _ops_read_the_running_band(ag, Stage(0, [1, 2, 3], ["stem"], ["wide", "deep"]))
        assert not _ops_read_the_running_band(ag, Stage(0, [1, 2, 3, 4], ["stem"], ["joined"]))

    def test_a_residual_operand_is_carried(self, chain_with_fanout_path):
        """The Add reads the Relu's output from before the second Conv; the
        executor offsets a binary op's operand to the band's rows."""
        ag = load_model(chain_with_fanout_path)
        stage = Stage(0, list(range(len(ag.ops))), ["input"], ["output"])
        assert ag.ops[-1].op_type == "Add"
        assert _ops_read_the_running_band(ag, stage)

    def test_the_strided_branch_stage_is_not_chained(self, tmp_path):
        ag, chains = _stages_then_chains(
            _branch_model(tmp_path / "branch.onnx", strided=True), budget=120 * 1024)
        stage = next(i for i, s in enumerate(ag.stages) if 3 in s.op_indices)
        assert {3, 5} <= set(ag.stages[stage].op_indices)
        assert all(stage not in group for group in chains)

    @pytest.mark.parametrize("model, cuts, budget", _OFF_BAND_CHAINS)
    def test_the_runtime_refuses_a_chain_off_its_band(self, tmp_path, require_host_library,
                                                       monkeypatch, model, cuts, budget):
        """A compiler without the check chains these stages. The runtime loader
        refuses that plan rather than run it with wrong output, and loads the
        plan the compiler writes."""
        from tigris.analysis import partition_spatial as spatial
        from tigris.runtime import RuntimeError as HostError, check_plan

        path = _RUNNING_BAND_MODELS[model](tmp_path / "model.onnx")

        def plan():
            ag = compute_memory_timeline(compute_lifetimes(load_model(path)))
            ag = detect_and_solve_chains(partition_spatial(partition_temporal(ag, budget, cuts)))
            return emit_binary_bytes(ag)

        assert isinstance(check_plan(plan()), dict)
        monkeypatch.setattr(spatial, "_ops_read_the_running_band", lambda *args, **kwargs: True)
        with pytest.raises(HostError) as refused:
            check_plan(plan())
        assert str(refused.value) == ("the runtime loader refuses the plan: chained stage reads a "
                                      "tensor outside its band; recompile the model with this "
                                      "TiGrIS release")

    def test_a_stripe_offsets_no_operand(self, chain_with_fanout_path):
        """The height stripe carries no operand, so the residual Add that a
        chain admits reads off the stripe's band."""
        ag = load_model(chain_with_fanout_path)
        stage = Stage(0, list(range(len(ag.ops))), ["input"], ["output"])
        assert _ops_read_the_running_band(ag, stage)
        assert not _ops_read_the_running_band(ag, stage, carries_operands=False)

    @pytest.mark.parametrize("middle", ["conv3", "resize", "conv1x1s2", "conv1x3"])
    def test_a_stripe_off_its_band_is_not_tiled(self, tmp_path, middle):
        """The Relu after the spatial op reads x, which the stripe holds at
        the spatial op's input rows and width. Tiled, the plan would run with
        wrong output (Conv 3x3, and the 1 x k Convs that narrow the width) or
        write past the Relu's output (Resize)."""
        ag = _stripe_stages(_stripe_model(tmp_path / "stripe.onnx", middle), 48 * 1024)
        stage = next(s for s in ag.stages if s.op_indices == [1, 2])
        assert not stage.tile_plan.tileable
        assert stage.tile_plan.warnings == [f"Stage {stage.stage_id} reads rows a tile does not hold"]

    def test_a_one_to_one_stripe_runs_as_the_model(self, tmp_path, require_host_library):
        """A 1x1 convolution leaves the rows where they are, so the stripe
        still tiles the stage and the Relu reads the right rows."""
        path = _stripe_model(tmp_path / "stripe.onnx", "conv1")
        ag = _stripe_stages(path, 48 * 1024)
        stage = next(s for s in ag.stages if s.op_indices == [1, 2])
        assert stage.tile_plan.tileable and stage.tile_plan.num_tiles > 1 and stage.chain_len == 0
        plan = tmp_path / "stripe.tgrs"
        plan.write_bytes(emit_binary_bytes(ag))
        _run_against_onnxruntime(path, plan)

    @pytest.mark.parametrize("middle, keeps", [
        ("conv1", True), ("conv1x3p1", True), ("conv1x1s2", False), ("conv1x3", False)])
    def test_a_conv_that_narrows_the_width_moves_the_band(self, tmp_path, middle, keeps):
        """A 1 x k Conv keeps the rows. The chain executor and the stripe
        hand the ops after it its output width, so it keeps the band only
        when its output is as wide as its input."""
        ag = load_model(_stripe_model(tmp_path / "stripe.onnx", middle))
        stage = Stage(0, [1, 2], ["x"], ["b", "c"])
        assert _ops_read_the_running_band(ag, stage) is keeps
        assert _ops_read_the_running_band(ag, stage, carries_operands=False) is keeps

    @pytest.mark.parametrize("middle", ["conv1x1s2", "conv1x3"])
    def test_the_runtime_refuses_a_chain_that_narrows_the_width(
            self, tmp_path, require_host_library, monkeypatch, middle):
        """Without the width rule the compiler chains the stage, and the
        chained plan would run with wrong output. The runtime loader refuses
        it."""
        from tigris.analysis import partition_spatial as spatial
        from tigris.runtime import RuntimeError as HostError, check_plan

        path = _stripe_model(tmp_path / "stripe.onnx", middle)
        ag = detect_and_solve_chains(_stripe_stages(path, 48 * 1024))
        stage = next(s for s in ag.stages if s.op_indices == [1, 2])
        assert stage.chain_len == 0 and not stage.tile_plan.tileable
        monkeypatch.setattr(spatial, "_ops_read_the_running_band", lambda *args, **kwargs: True)
        ag = detect_and_solve_chains(_stripe_stages(path, 48 * 1024))
        stage = next(s for s in ag.stages if s.op_indices == [1, 2])
        assert stage.chain_len >= 2
        with pytest.raises(HostError) as refused:
            check_plan(emit_binary_bytes(ag))
        assert str(refused.value) == ("the runtime loader refuses the plan: chained stage reads a "
                                      "tensor outside its band; recompile the model with this "
                                      "TiGrIS release")

    def test_a_chain_that_keeps_the_width_runs_as_the_model(self, tmp_path, require_host_library):
        """A 1x3 Conv with width pads reads a column halo but keeps the width,
        and the chain holds whole rows, so the Relu of its input stays in the
        chain and reads the right elements."""
        path = _stripe_model(tmp_path / "stripe.onnx", "conv1x3p1")
        ag = detect_and_solve_chains(_stripe_stages(path, 48 * 1024))
        assert next(s for s in ag.stages if s.op_indices == [1, 2]).chain_len >= 2
        plan = tmp_path / "stripe.tgrs"
        plan.write_bytes(emit_binary_bytes(ag))
        _run_against_onnxruntime(path, plan)

    @pytest.mark.parametrize("kernel, eligible", [(1, True), (3, False)])
    def test_an_hw_tile_reads_its_band(self, tmp_path, require_host_library, kernel, eligible):
        """An HW tile holds the spatial op's input with a column halo and
        hands the ops after it the output tile. The Relu reads x, so the stage
        is HW tiled only when the Conv maps columns one to one; tiled behind a
        1x3 Conv, the plan would run with wrong output."""
        path = _wide_model(tmp_path / "wide.onnx", kernel)
        ag = _stripe_stages(path, 40 * 1024)
        stage = next(s for s in ag.stages if s.op_indices == [1, 2])
        assert _stage_2d_eligible(ag, stage, [ag.ops[i] for i in stage.op_indices]) is eligible
        assert (stage.tile_plan.band_kind == TILE_BAND_KIND_HW) is eligible
        if eligible:
            plan = tmp_path / "wide.tgrs"
            plan.write_bytes(emit_binary_bytes(ag))
            _run_against_onnxruntime(path, plan)

    @pytest.mark.parametrize("model, cuts, budget, chained", [
        ("strided", frozenset(), 120 * 1024, [1]),
        ("unit_stride", frozenset({1, 2, 6}), 100 * 1024, [1]),
        ("fire", frozenset({1, 5}), 160 * 1024, None),
        ("fire", frozenset({1, 5}), 96 * 1024, [1, 2, 3]),
    ])
    def test_the_plan_runs_as_the_model(self, tmp_path, require_host_library,
                                        model, cuts, budget, chained):
        """Chained, the strided branch stage would stop the run with a tiling
        error, the unit-stride one would read its branch rows one row off,
        and the fire module would join two bands that start on different
        rows. The fire module without its Concat still streams. `chained`
        names the ops of a stage the plan chains."""
        import onnxruntime as ort
        from tigris.runtime import Session

        path = _RUNNING_BAND_MODELS[model](tmp_path / "model.onnx")
        ag = compute_memory_timeline(compute_lifetimes(load_model(path)))
        ag = detect_and_solve_chains(partition_spatial(partition_temporal(ag, budget, cuts)))
        if chained is not None:
            assert any(s.op_indices == chained and s.chain_len >= 2 for s in ag.stages)
        plan = tmp_path / "model.tgrs"
        plan.write_bytes(emit_binary_bytes(ag))
        x = np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)
        expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
            None, {"input": x})[0].transpose(0, 2, 3, 1)
        with Session(plan) as session:
            actual = session.run({"input": np.ascontiguousarray(x.transpose(0, 2, 3, 1))})["output"]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * np.abs(expected).max())


def _strided_stem_model(tmp_path):
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
    return save_onnx(tmp_path, "stem", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 16])],
                     inits)


def test_chain_geometry_takes_the_kernel_from_the_weight(tmp_path):
    """A convolution without kernel_shape keeps its 5x5 window in the chain model."""
    rng = np.random.default_rng(0)
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [16, 8, 1, 1],
                            rng.standard_normal((16, 8, 1, 1)).astype(np.float32).ravel().tolist())
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [16, 1, 5, 5],
                            rng.standard_normal((16, 1, 5, 5)).astype(np.float32).ravel().tolist())
    path = save_onnx(tmp_path, "k5",
                     [helper.make_node("Conv", ["input", "w1"], ["a"]),
                      helper.make_node("Conv", ["a", "w2"], ["output"], group=16, strides=[2, 2],
                                       pads=[1, 1, 2, 2])],
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 32, 32])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 16])],
                     [w1, w2])
    ag = load_model(path)
    ag.stages = [Stage(stage_id=0, op_indices=[0], input_tensors=["input"], output_tensors=["a"]),
                 Stage(stage_id=1, op_indices=[1], input_tensors=["a"], output_tensors=["output"])]

    weight_shapes = _ag_weight_shapes(ag)
    assert _get_stage_spatial_params(ag, ag.stages[1], weight_shapes) == (5, 2)
    heights = _back_propagate_tile_heights(
        [_get_stage_spatial_params(ag, s, weight_shapes) for s in ag.stages], 1)
    assert heights == [(5, 5), (5, 1)]
    # Five input rows of 8 channels, five rows of the 16-channel 1x1 output,
    # one output row of the depthwise, each 32-byte aligned.
    assert _chain_fast_bytes(ag, ag.stages, heights, weight_shapes) == (
        5 * 32 * 8 * 4 + 5 * 32 * 16 * 4 + 16 * 16 * 4)


class TestSplitChains:
    """A run too big to stream whole still streams as consecutive chains."""

    budget = 48000

    def _stages(self, tmp_path):
        ag = load_model(_strided_stem_model(tmp_path))
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


def test_padded_chain_fits_two_rows_without_unused_output_rows(tmp_path):
    shape = [1, 4, 64, 64]
    weights = np.full((4, 4, 3, 3), 0.01, dtype=np.float32)
    nodes, initializers = [], []
    source = "input"
    for index in range(4):
        name = f"w{index}"
        output = "output" if index == 3 else f"t{index}"
        initializers.append(helper.make_tensor(
            name, TensorProto.FLOAT, weights.shape, weights.ravel().tolist()))
        nodes.append(helper.make_node(
            "Conv", [source, name], [output], kernel_shape=[3, 3], pads=[1] * 4))
        source = output
    path = save_onnx(tmp_path, "padded_chain", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
                     initializers, filename="chain.onnx")
    ag = plan_model(path, (32768,))
    head = ag.stages[0]
    assert head.chain_len == 4
    assert head.chain_tile_h == 2


# Chain detection tests


class TestDetectChains:
    def test_linear_chain_detected(self, three_conv_chain_path):
        """Three consecutive Conv+Relu stages should form a single chain."""
        ag = plan_model(three_conv_chain_path, (32000,))

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
        ag = plan_model(chain_with_fanout_path, (32000,))

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

# Tile height solver tests


class TestChainTileSolver:
    @pytest.mark.parametrize("params, out_h, expected", [
        # Pointwise: tile heights are identical throughout.
        ([(1, 1), (1, 1), (1, 1)], 4, [(4, 4), (4, 4), (4, 4)]),
        # Conv3x3(s=1): each stage adds 2 rows of halo.
        ([(3, 1), (3, 1)], 4, [(8, 6), (6, 4)]),
        # Conv3x3(s=2): out_h rows need stride*out_h + kernel-stride input rows.
        ([(3, 2)], 3, [(7, 3)]),
        # Pool2x2(s=2) uses the same spatial range algebra as convolution.
        ([(2, 2)], 3, [(6, 3)]),
    ], ids=["pointwise", "conv3x3-stride1", "conv3x3-stride2", "pool2x2-stride2"])
    def test_back_propagate(self, params, out_h, expected):
        assert list(_back_propagate_tile_heights(params, out_h)) == expected


# Plan format round-trip


class TestChainPlanFormat:
    def test_chain_fields_roundtrip(self, three_conv_chain_path):
        """Chain fields survive binary write -> read, and chaining leaves the
        ops and tensors of the loose-budget pipeline unchanged."""
        ag = plan_model(three_conv_chain_path, (32000,))
        assert any(s.chain_len >= 2 for s in ag.stages)

        plan = read_binary_plan(emit_binary_bytes(ag))
        for i, stage in enumerate(plan["stages"]):
            ir_stage = ag.stages[i]
            assert stage["chain_id"] == (ir_stage.chain_id & 0xFFFF)
            assert stage["chain_len"] == (ir_stage.chain_len & 0xFFFF)
            assert stage["chain_tile_h"] == (ir_stage.chain_tile_h & 0xFFFF)

        # A chain head sits at stage index == chain_id.
        heads = [(i, s) for i, s in enumerate(plan["stages"])
                 if s["chain_len"] >= 2 and s["chain_id"] == i]
        assert len(heads) >= 1, "Expected at least one chain head in read-back plan"
        for idx, head in heads:
            assert head["chain_tile_h"] > 0, "Chain head should have tile_h > 0"
            chain_id = head["chain_id"]
            chain_len = head["chain_len"]
            for i in range(chain_id, chain_id + chain_len):
                s = plan["stages"][i]
                assert s["chain_id"] == chain_id
                assert s["chain_len"] == chain_len

        ag_loose = plan_model(three_conv_chain_path, (10 * 1024 * 1024,))
        assert len(ag.ops) == len(ag_loose.ops)
        for a, b in zip(ag.ops, ag_loose.ops):
            assert a.op_type == b.op_type
            assert a.inputs == b.inputs
            assert a.outputs == b.outputs
        assert set(ag.tensors.keys()) == set(ag_loose.tensors.keys())

    def test_standalone_stages_have_no_chain(self, three_conv_chain_path):
        """Stages not in a chain should have chain_id=0xFFFF, chain_len=0."""
        ag = plan_model(three_conv_chain_path, (10 * 1024 * 1024,))

        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)

        assert not any(s.line_buffered for s in ag.stages)
        for stage in plan["stages"]:
            assert stage["chain_id"] == 0xFFFF
            assert stage["chain_len"] == 0
            assert stage["flags"] == 0

    def test_chain_clears_individual_tile_plans(self, three_conv_chain_path):
        """Stages in a chain should not have individual tile plans."""
        ag = plan_model(three_conv_chain_path, (32000,))

        chained = [s for s in ag.stages if s.chain_len >= 2]
        for s in chained:
            assert s.tile_plan is None, f"Chain stage {s.stage_id} should not have individual tile_plan"


# Chain execution E2E tests


class TestChainExecution:
    def test_chain_tile_h_fits_budget(self, three_conv_chain_path):
        """Solved chain_tile_h should produce tile buffers within budget."""
        ag = plan_model(three_conv_chain_path, (32000,))

        for s in ag.stages:
            if s.chain_len >= 2 and s.chain_id == s.stage_id:
                # This is a chain head - verify the tile buffer fits
                chain_indices = list(range(s.chain_id, s.chain_id + s.chain_len))
                chain_stages = [ag.stages[i] for i in chain_indices]
                assert s.chain_tile_h > 0
                weight_shapes = _ag_weight_shapes(ag)
                chain_params = [_get_stage_spatial_params(ag, cs, weight_shapes)
                                for cs in chain_stages]
                heights = _back_propagate_tile_heights(chain_params, s.chain_tile_h)
                needed = _chain_fast_bytes(ag, chain_stages, heights, weight_shapes)
                assert needed <= ag.mem_budget, (
                    f"Chain tile buffer {needed} exceeds budget {ag.mem_budget}"
                )
