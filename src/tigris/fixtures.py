"""Inline ONNX model builders for test fixtures.

Not part of the CLI. Used by:
- Python tests (test_plan_contract.py)
- scripts/gen_fixtures.py (fixture generation for tigris-runtime C tests)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_linear_3op() -> onnx.ModelProto:
    """ReLU -> Add -> Add.  Input/output: [1, 64].

    The activation leads so the chain keeps three operators: a Relu that
    follows a fusable producer is absorbed into it.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [1, 64],
                            np.zeros((1, 64), dtype=np.float32).flatten().tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [1, 64],
                            np.zeros((1, 64), dtype=np.float32).flatten().tolist())
    relu = helper.make_node("Relu", ["input"], ["t0"], name="relu0")
    add0 = helper.make_node("Add", ["t0", "w0"], ["t1"], name="add0")
    add1 = helper.make_node("Add", ["t1", "w1"], ["output"], name="add1")
    graph = helper.make_graph([relu, add0, add1], "linear_3op", [X], [Y], initializer=[w0, w1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def build_conv_relu_chain() -> onnx.ModelProto:
    """Conv -> ReLU -> Conv.  Input [1,3,32,32], output [1,8,28,28]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 28, 28])
    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [8, 3, 3, 3],
                            np.zeros((8, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [8], np.zeros(8, dtype=np.float32).tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [8, 8, 3, 3],
                            np.zeros((8, 8, 3, 3), dtype=np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [8], np.zeros(8, dtype=np.float32).tolist())
    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["output"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    graph = helper.make_graph([conv0, relu, conv1], "conv_relu_chain",
                              [X], [Y], initializer=[w0, b0, w1, b1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def build_ds_cnn() -> onnx.ModelProto:
    """MLPerf Tiny DS-CNN for keyword spotting."""
    np.random.seed(42)

    def _conv_bn_relu(prefix, x_name, out_name, c_in, c_out, kh, kw, sh, sw, group=1):
        nodes, inits = [], []
        w = np.random.randn(c_out, c_in // group, kh, kw).astype(np.float32) * 0.1
        b = np.zeros(c_out, dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"{prefix}_w"))
        inits.append(numpy_helper.from_array(b, f"{prefix}_b"))
        pad_h, pad_w = max(0, kh - 1), max(0, kw - 1)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        nodes.append(helper.make_node("Conv", [x_name, f"{prefix}_w", f"{prefix}_b"],
                                      [f"{prefix}_conv_out"], name=f"{prefix}_conv",
                                      kernel_shape=[kh, kw], strides=[sh, sw],
                                      pads=[pt, pl, pb, pr], group=group))
        for suffix, data in [("scale", np.ones(c_out)), ("bias", np.zeros(c_out)),
                             ("mean", np.zeros(c_out)), ("var", np.ones(c_out))]:
            inits.append(numpy_helper.from_array(data.astype(np.float32), f"{prefix}_{suffix}"))
        nodes.append(helper.make_node("BatchNormalization",
                                      [f"{prefix}_conv_out", f"{prefix}_scale", f"{prefix}_bias",
                                       f"{prefix}_mean", f"{prefix}_var"],
                                      [f"{prefix}_bn_out"], name=f"{prefix}_bn"))
        nodes.append(helper.make_node("Relu", [f"{prefix}_bn_out"], [out_name], name=f"{prefix}_relu"))
        return nodes, inits

    all_nodes, all_inits = [], []
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 49, 10])
    n, i = _conv_bn_relu("block0", "input", "b0_out", 1, 64, 10, 4, 2, 2)
    all_nodes.extend(n)
    all_inits.extend(i)
    prev = "b0_out"
    for idx in range(1, 5):
        n, i = _conv_bn_relu(f"dw{idx}", prev, f"dw{idx}_out", 64, 64, 3, 3, 1, 1, group=64)
        all_nodes.extend(n)
        all_inits.extend(i)
        n, i = _conv_bn_relu(f"pw{idx}", f"dw{idx}_out", f"pw{idx}_out", 64, 64, 1, 1, 1, 1)
        all_nodes.extend(n)
        all_inits.extend(i)
        prev = f"pw{idx}_out"
    all_nodes.append(helper.make_node("GlobalAveragePool", [prev], ["pool_out"], name="gap"))
    all_nodes.append(helper.make_node("Flatten", ["pool_out"], ["flat_out"], name="flatten", axis=1))
    fc_w = np.random.randn(12, 64).astype(np.float32) * 0.1
    fc_b = np.zeros(12, dtype=np.float32)
    all_inits.append(numpy_helper.from_array(fc_w, "fc_w"))
    all_inits.append(numpy_helper.from_array(fc_b, "fc_b"))
    all_nodes.append(helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["output"], name="fc", transB=1))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 12])
    graph = helper.make_graph(all_nodes, "ds_cnn", [X], [Y], initializer=all_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_fc_autoencoder() -> onnx.ModelProto:
    """MLPerf Tiny FC autoencoder for anomaly detection."""
    np.random.seed(43)
    all_nodes, all_inits = [], []
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128])
    layer_dims = [128, 128, 128, 128, 128, 8, 128, 128, 128, 128]
    prev, prev_dim = "input", 128
    for idx, dim_out in enumerate(layer_dims):
        prefix = f"fc{idx}"
        w = np.random.randn(dim_out, prev_dim).astype(np.float32) * 0.1
        b = np.zeros(dim_out, dtype=np.float32)
        all_inits.append(numpy_helper.from_array(w, f"{prefix}_w"))
        all_inits.append(numpy_helper.from_array(b, f"{prefix}_b"))
        all_nodes.append(helper.make_node("Gemm", [prev, f"{prefix}_w", f"{prefix}_b"],
                                          [f"{prefix}_out"], name=prefix, transB=1))
        if idx < len(layer_dims) - 1:
            all_nodes.append(helper.make_node("Relu", [f"{prefix}_out"], [f"{prefix}_relu_out"],
                                              name=f"{prefix}_relu"))
            prev = f"{prefix}_relu_out"
        else:
            all_nodes.append(helper.make_node("Sigmoid", [f"{prefix}_out"], ["output"],
                                              name="output_sigmoid"))
        prev_dim = dim_out
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])
    graph = helper.make_graph(all_nodes, "fc_autoencoder", [X], [Y], initializer=all_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_tcn() -> onnx.ModelProto:
    """Gated TCN for activity recognition."""
    np.random.seed(44)

    def _conv1d_bn(prefix, x_name, out_name, c_in, c_out, k, dilation, act):
        nodes, inits = [], []
        w = np.random.randn(c_out, c_in, k).astype(np.float32) * 0.1
        b = np.zeros(c_out, dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"{prefix}_w"))
        inits.append(numpy_helper.from_array(b, f"{prefix}_b"))
        nodes.append(helper.make_node("Conv", [x_name, f"{prefix}_w", f"{prefix}_b"],
                                      [f"{prefix}_conv_out"], name=f"{prefix}_conv",
                                      kernel_shape=[k], strides=[1],
                                      pads=[dilation * (k - 1), 0], dilations=[dilation]))
        for suffix, data in [("scale", np.ones(c_out)), ("bias", np.zeros(c_out)),
                             ("mean", np.zeros(c_out)), ("var", np.ones(c_out))]:
            inits.append(numpy_helper.from_array(data.astype(np.float32), f"{prefix}_{suffix}"))
        nodes.append(helper.make_node("BatchNormalization",
                                      [f"{prefix}_conv_out", f"{prefix}_scale", f"{prefix}_bias",
                                       f"{prefix}_mean", f"{prefix}_var"],
                                      [f"{prefix}_bn_out"], name=f"{prefix}_bn"))
        nodes.append(helper.make_node(act, [f"{prefix}_bn_out"], [out_name],
                                      name=f"{prefix}_{act.lower()}"))
        return nodes, inits

    all_nodes, all_inits = [], []
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128])
    n, i = _conv1d_bn("block1", "input", "b1_out", 3, 16, 3, 1, "Relu")
    all_nodes.extend(n)
    all_inits.extend(i)
    n, i = _conv1d_bn("block2", "b1_out", "b2_out", 16, 16, 3, 2, "Relu")
    all_nodes.extend(n)
    all_inits.extend(i)
    n, i = _conv1d_bn("gate_t", "b2_out", "gate_tanh_out", 16, 16, 3, 4, "Tanh")
    all_nodes.extend(n)
    all_inits.extend(i)
    n, i = _conv1d_bn("gate_s", "b2_out", "gate_sig_out", 16, 16, 3, 4, "Sigmoid")
    all_nodes.extend(n)
    all_inits.extend(i)
    all_nodes.append(helper.make_node("Mul", ["gate_tanh_out", "gate_sig_out"], ["gate_out"],
                                      name="gate_mul"))
    n, i = _conv1d_bn("block4", "gate_out", "b4_out", 16, 16, 3, 8, "Relu")
    all_nodes.extend(n)
    all_inits.extend(i)
    all_nodes.append(helper.make_node("Flatten", ["b4_out"], ["flat_out"], name="flatten", axis=1))
    fc_w = np.random.randn(6, 2048).astype(np.float32) * 0.1
    fc_b = np.zeros(6, dtype=np.float32)
    all_inits.append(numpy_helper.from_array(fc_w, "fc_w"))
    all_inits.append(numpy_helper.from_array(fc_b, "fc_b"))
    all_nodes.append(helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["output"],
                                      name="fc", transB=1))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6])
    graph = helper.make_graph(all_nodes, "tcn", [X], [Y], initializer=all_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model
