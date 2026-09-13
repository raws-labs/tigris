"""Shared fixtures: small hand-crafted ONNX graphs for testing."""

import pytest
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


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
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [1, 64], np.zeros((1, 64), dtype=np.float32).flatten().tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [1, 64], np.zeros((1, 64), dtype=np.float32).flatten().tolist())

    relu = helper.make_node("Relu", ["input"], ["t0"], name="relu0")
    add0 = helper.make_node("Add", ["t0", "w0"], ["t1"], name="add0")
    add1 = helper.make_node("Add", ["t1", "w1"], ["output"], name="add1")

    graph = helper.make_graph([relu, add0, add1], "linear_3op", [X], [Y], initializer=[w0, w1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "linear_3op.onnx"
    onnx.save(model, str(path))
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

    graph = helper.make_graph([relu, sigmoid, add], "diamond", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "diamond.onnx"
    onnx.save(model, str(path))
    return path


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

    graph = helper.make_graph(nodes, "large_chain", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "large_chain.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def conv_relu_chain_path(tmp_path):
    """Conv3x3 -> Relu -> Conv3x3 on [1,3,32,32] - tileable, RF=5.

    Two 3x3 convolutions with a relu in between, NCHW layout.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 28, 28])

    # Conv1: 3->8 channels, 3x3 kernel, no padding
    w0_data = np.zeros((8, 3, 3, 3), dtype=np.float32).flatten().tolist()
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [8, 3, 3, 3], w0_data)
    b0_data = np.zeros(8, dtype=np.float32).tolist()
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [8], b0_data)

    # Conv2: 8->8 channels, 3x3 kernel, no padding
    w1_data = np.zeros((8, 8, 3, 3), dtype=np.float32).flatten().tolist()
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [8, 8, 3, 3], w1_data)
    b1_data = np.zeros(8, dtype=np.float32).tolist()
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [8], b1_data)

    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["output"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])

    graph = helper.make_graph([conv0, relu, conv1], "conv_relu_chain",
                              [X], [Y], initializer=[w0, b0, w1, b1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "conv_relu_chain.onnx"
    onnx.save(model, str(path))
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

    graph = helper.make_graph([conv, relu, flatten], "conv_flatten",
                              [X], [Y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "conv_flatten.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def conv_pool_chain_path(tmp_path):
    """Conv3x3 -> Relu -> MaxPool2x2(stride=2) -> Conv3x3 - tests stride effect on RF.

    RF calculation: conv1(k=3,s=1) -> relu(passthrough) -> pool(k=2,s=2) -> conv0(k=3,s=1)
    Walking in reverse: rf=1, then conv1: rf=3, jump=1; pool: rf=4, jump=2; conv0: rf=8, jump=2
    Actually: rf=1 -> conv1(k=3,s=1): rf = 1+(3-1)*1 = 3, j=1 -> pool(k=2,s=2): rf = 3+(2-1)*1 = 4, j=2 -> relu: pass -> conv0(k=3,s=1): rf = 4+(3-1)*2 = 8, j=2
    Wait, we reverse: [conv0, relu, pool, conv1]. reversed = [conv1, pool, relu, conv0]
    conv1(k=3,s=1): rf=1+(3-1)*1=3, j=1*1=1
    pool(k=2,s=2): rf=3+(2-1)*1=4, j=1*2=2
    relu: pass
    conv0(k=3,s=1): rf=4+(3-1)*2=8, j=2*1=2
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

    graph = helper.make_graph([conv0, relu, pool, conv1], "conv_pool_chain",
                              [X], [Y], initializer=[w0, b0, w1, b1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = tmp_path / "conv_pool_chain.onnx"
    onnx.save(model, str(path))
    return path


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
                             ["W_q"], name="w_ql")
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

    graph = helper.make_graph(
        [n_inp_q, n_inp_dq, n_w_q, n_w_dq, n_conv, n_relu, n_out_q, n_out_dq],
        "qdq_conv",
        [X], [Y],
        initializer=[W_init, B_init, inp_scale, inp_zp, w_scale, w_zp, out_scale, out_zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    path = tmp_path / "qdq_conv.onnx"
    onnx.save(model, str(path))
    return path
