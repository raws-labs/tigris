"""Shared fixtures: small hand-crafted ONNX graphs for testing."""

import os
from pathlib import Path

import pytest
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import detect_and_solve_chains, partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary import defs
from tigris.loaders import load_model
from scripts.contract_cases.fixtures import build_conv_relu_chain, build_linear_3op


def vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def save_onnx(tmp_path, name, nodes, inputs, outputs, init=(), opset=13,
              ir_version=None, filename=None, check=False):
    """Save a one-graph model as tmp_path/<filename or name.onnx>; return the path."""
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, list(init)),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    if ir_version is not None:
        model.ir_version = ir_version
    if check:
        onnx.checker.check_model(model)
    path = tmp_path / (filename or f"{name}.onnx")
    onnx.save(model, str(path))
    return path


def section_entries(data):
    """Map each section kind of a plan to (directory entry offset, payload offset)."""
    entries = {}
    offset = defs.HEADER_STRUCT.unpack_from(data)[3]
    while True:
        kind, start = defs.SECTION_ENTRY_STRUCT.unpack_from(data, offset)
        if not kind:
            return entries
        entries[kind] = (offset, start)
        offset += defs.SECTION_ENTRY_SIZE


@pytest.fixture
def require_host_library(monkeypatch):
    """Skip without the host library staged in the tigris package; fail under TIGRIS_REQUIRE_HOST."""
    monkeypatch.delenv("TIGRIS_HOST_LIBRARY", raising=False)
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Host library is not staged in the tigris package")
        pytest.skip("Host library has not been staged in this source checkout")


def normalized(tmp_path, name, nodes, inputs, outputs, init=(), **kwargs):
    """Save a one-graph model and load it through the normalizer."""
    return load_model(save_onnx(tmp_path, name, nodes, inputs, outputs, init, **kwargs))


def full_pipeline(path, budget=0, spatial=True, chains=False):
    """Load, compute lifetimes and memory; with a budget, partition the graph."""
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        if spatial:
            ag = partition_spatial(ag)
            if chains:
                ag = detect_and_solve_chains(ag)
    return ag


@pytest.fixture
def linear_3op_path(tmp_path):
    """A -> B -> C -> D  (3 ops, linear chain, float32, batch=1).

    Op0: Relu(input) -> t0           shape [1, 64]
    Op1: Add(t0, w0) -> t1           shape [1, 64]
    Op2: Add(t1, w1) -> output       shape [1, 64]

    Activation sizes: input=256B, t0=256B, t1=256B, output=256B

    The Relu leads so the chain keeps three operators: an activation that
    follows a fusable producer is absorbed into it.
    """
    path = tmp_path / "linear_3op.onnx"
    onnx.save(build_linear_3op(), str(path))
    return path


@pytest.fixture
def diamond_path(tmp_path):
    """Diamond graph: input branches into two paths that merge.

    Op0: Relu(input) -> left         shape [1, 128]
    Op1: Sigmoid(input) -> right     shape [1, 128]
    Op2: Add(left, right) -> output  shape [1, 128]

    Peak memory: at Op2, left + right + output are all live.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])

    relu = helper.make_node("Relu", ["input"], ["left"], name="relu")
    sigmoid = helper.make_node("Sigmoid", ["input"], ["right"], name="sigmoid")
    add = helper.make_node("Add", ["left", "right"], ["output"], name="add")

    return save_onnx(
        tmp_path, "diamond", [relu, sigmoid, add], [X], [Y], opset=17, ir_version=8
    )


@pytest.fixture
def large_activations_path(tmp_path):
    """5-op chain with large activations to test partitioning.

    Each op produces a [1, 1024] float32 tensor = 4096 bytes.
    Budget of ~8K should force at least 2 stages.

    Op0: Relu(input) -> t0
    Op1: Relu(t0) -> t1
    Op2: Relu(t1) -> t2
    Op3: Relu(t2) -> t3
    Op4: Relu(t3) -> output
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1024])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1024])

    nodes = []
    for i in range(5):
        inp = "input" if i == 0 else f"t{i-1}"
        out = "output" if i == 4 else f"t{i}"
        nodes.append(helper.make_node("Relu", [inp], [out], name=f"relu{i}"))

    return save_onnx(tmp_path, "large_chain", nodes, [X], [Y], opset=17, ir_version=8)


@pytest.fixture
def loop_past_512_tensors_path(tmp_path):
    """float_while_if_through with a tree of 519 Adds over its loop output,
    which takes the main graph past 512 tensors; the plan also holds the loop
    and branch bodies. Returns the path and the model's main-graph tensors."""
    from tigris.frontends import tflite_writer as tw
    from tigris.frontends.tflite_schema import BUILTIN_OPERATORS

    fixture = Path(__file__).parent / "fixtures" / "tflite" / "ops" / "float_while_if_through.tflite"
    model = tw.read(fixture.read_bytes())
    graph = model.subgraphs[0]
    add = BUILTIN_OPERATORS.index("ADD")
    model.operator_codes.append(tw.OperatorCode(add, 1, None, add))
    level = [graph.outputs[0]] * 520
    while len(level) > 1:
        sums = []
        for a, b in zip(level[::2], level[1::2]):
            graph.tensors.append(tw.Tensor([1, 4], 0, 0, b"sum%d" % len(graph.tensors)))
            graph.operators.append(tw.Operator(len(model.operator_codes) - 1, [a, b],
                                               [len(graph.tensors) - 1]))
            sums.append(len(graph.tensors) - 1)
        level = sums + level[len(sums) * 2:]
    graph.outputs = level
    source = tmp_path / "loop.tflite"
    source.write_bytes(tw.write(model))
    return source, len(graph.tensors)


@pytest.fixture
def conv_relu_chain_path(tmp_path):
    """Conv3x3 -> Relu -> Conv3x3 on [1,3,32,32] - tileable, RF=5.

    Two 3x3 convolutions with a relu in between, NCHW layout.
    """
    path = tmp_path / "conv_relu_chain.onnx"
    onnx.save(build_conv_relu_chain(), str(path))
    return path


@pytest.fixture
def conv_with_flatten_path(tmp_path):
    """Conv -> Relu -> Flatten - untileable due to Flatten.

    NCHW input, Flatten collapses spatial dims.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 588])

    w_data = np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist()
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3], w_data)
    b_data = np.zeros(4, dtype=np.float32).tolist()
    b = helper.make_tensor("b", TensorProto.FLOAT, [4], b_data)

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="conv0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    flatten = helper.make_node("Flatten", ["t1"], ["output"], name="flatten0", axis=1)

    return save_onnx(
        tmp_path, "conv_flatten", [conv, relu, flatten], [X], [Y], [w, b],
        opset=17, ir_version=8,
    )


@pytest.fixture
def conv_pool_chain_path(tmp_path):
    """Conv3x3 -> Relu -> MaxPool2x2(stride=2) -> Conv3x3 - tests stride effect on RF.

    RF=8, halo=7
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 13, 13])

    # Conv0: 3->8, 3x3
    w0_data = np.zeros((8, 3, 3, 3), dtype=np.float32).flatten().tolist()
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [8, 3, 3, 3], w0_data)
    b0_data = np.zeros(8, dtype=np.float32).tolist()
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [8], b0_data)

    # Conv1: 8->8, 3x3
    w1_data = np.zeros((8, 8, 3, 3), dtype=np.float32).flatten().tolist()
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [8, 8, 3, 3], w1_data)
    b1_data = np.zeros(8, dtype=np.float32).tolist()
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [8], b1_data)

    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    pool = helper.make_node("MaxPool", ["t1"], ["t2"], name="pool0",
                            kernel_shape=[2, 2], strides=[2, 2])
    conv1 = helper.make_node("Conv", ["t2", "w1", "b1"], ["output"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])

    return save_onnx(
        tmp_path, "conv_pool_chain", [conv0, relu, pool, conv1], [X], [Y], [w0, b0, w1, b1],
        opset=17, ir_version=8,
    )


@pytest.fixture
def strided_conv_path(tmp_path):
    """Conv3x3 with row stride 2, pad 1, [1,4,32,8] -> [1,8,16,8]."""
    weights = numpy_helper.from_array(
        np.linspace(-1.0, 1.0, 8 * 4 * 9, dtype=np.float32).reshape(8, 4, 3, 3), "weights")
    conv = helper.make_node("Conv", ["input", "weights"], ["output"], name="conv0",
                            kernel_shape=[3, 3], strides=[2, 1], pads=[1, 1, 1, 1])
    return save_onnx(tmp_path, "strided_conv", [conv], [vi("input", [1, 4, 32, 8])],
                     [vi("output", [1, 8, 16, 8])], [weights], opset=17, ir_version=8)


@pytest.fixture
def edge_conv_path(tmp_path):
    """Conv5x5 with pad 2, [1,4,14,8] -> [1,8,14,8]."""
    weights = numpy_helper.from_array(
        np.linspace(-1.0, 1.0, 8 * 4 * 25, dtype=np.float32).reshape(8, 4, 5, 5), "weights")
    conv = helper.make_node("Conv", ["input", "weights"], ["output"], name="conv0",
                            kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    return save_onnx(tmp_path, "edge_conv", [conv], [vi("input", [1, 4, 14, 8])],
                     [vi("output", [1, 8, 14, 8])], [weights], opset=17, ir_version=8)


@pytest.fixture
def qdq_conv_path(tmp_path):
    """Quantized Conv->Relu with QDQ operators (int8 weights + activation Q/DQ).

    Pattern:
      input   -> QuantizeLinear -> DequantizeLinear -\
      float_W -> QuantizeLinear -> DequantizeLinear -> Conv -> Relu -> Q -> DQ -> output

    Single Conv with 1 input channel, 2 output channels, 3x3 kernel.
    Input shape: [1, 1, 4, 4] (NCHW).
    """
    np.random.seed(42)

    # Input
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])

    # Weight (float) - will be quantized by QL/DQL pair
    W_float = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.1
    W_init = numpy_helper.from_array(W_float, name="W_float")

    # Bias
    B_float = np.zeros(2, dtype=np.float32)
    B_init = numpy_helper.from_array(B_float, name="B_float")

    # Per-tensor scale/zp for input activation
    inp_scale = numpy_helper.from_array(np.array([0.05], dtype=np.float32), "inp_scale")
    inp_zp = numpy_helper.from_array(np.array([0], dtype=np.int8), "inp_zp")

    # Per-channel scale/zp for weight
    w_scale = numpy_helper.from_array(np.array([0.01, 0.01], dtype=np.float32), "w_scale")
    w_zp = numpy_helper.from_array(np.array([0, 0], dtype=np.int8), "w_zp")

    # Per-tensor scale/zp for output activation
    out_scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), "out_scale")
    out_zp = numpy_helper.from_array(np.array([0], dtype=np.int8), "out_zp")

    # Nodes: input Q/DQ
    n_inp_q = helper.make_node("QuantizeLinear", ["input", "inp_scale", "inp_zp"],
                               ["input_q"], name="inp_ql")
    n_inp_dq = helper.make_node("DequantizeLinear", ["input_q", "inp_scale", "inp_zp"],
                                ["input_dq"], name="inp_dql")

    # Nodes: weight Q/DQ
    n_w_q = helper.make_node("QuantizeLinear", ["W_float", "w_scale", "w_zp"],
                             ["W_q"], name="w_ql", axis=0)
    n_w_dq = helper.make_node("DequantizeLinear", ["W_q", "w_scale", "w_zp"],
                              ["W_dq"], name="w_dql", axis=0)

    # Conv + Relu
    n_conv = helper.make_node("Conv", ["input_dq", "W_dq", "B_float"], ["conv_out"],
                              name="conv0", kernel_shape=[3, 3], strides=[1, 1],
                              pads=[0, 0, 0, 0])
    n_relu = helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu0")

    # Output Q/DQ
    n_out_q = helper.make_node("QuantizeLinear", ["relu_out", "out_scale", "out_zp"],
                               ["out_q"], name="out_ql")
    n_out_dq = helper.make_node("DequantizeLinear", ["out_q", "out_scale", "out_zp"],
                                ["output"], name="out_dql")

    return save_onnx(
        tmp_path, "qdq_conv",
        [n_inp_q, n_inp_dq, n_w_q, n_w_dq, n_conv, n_relu, n_out_q, n_out_dq], [X], [Y],
        [W_init, B_init, inp_scale, inp_zp, w_scale, w_zp, out_scale, out_zp], ir_version=8,
    )
