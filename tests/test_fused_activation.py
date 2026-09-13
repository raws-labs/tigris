"""Tests for fused activation absorption pass."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.emitters.binary.defs import ACT_RELU, OP_TYPE_MAP
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders import load_model
from tigris.loaders.onnx.normalize import _decompose_silu


# Helpers


def _make_model(nodes, name, X, Y, initializers=None):
    graph = helper.make_graph(nodes, name, [X], [Y], initializer=initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _save_and_load(model, tmp_path, filename="model.onnx"):
    path = tmp_path / filename
    onnx.save(model, str(path))
    return load_model(path)


# SiLU decomposition


def test_silu_decomposes_to_supported_sigmoid_mul():
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="silu",
                op_type="Silu",
                inputs=["input"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", (1, 4), TensorProto.FLOAT),
            "output": TensorInfo("output", (1, 4), TensorProto.FLOAT),
        },
        model_inputs=["input"],
        model_outputs=["output"],
    )

    normalized = _decompose_silu(graph)

    assert [op.op_type for op in normalized.ops] == ["Sigmoid", "Mul"]
    assert normalized.ops[0].inputs == ["input"]
    assert normalized.ops[1].inputs == ["input", "input_sigmoid"]
    assert normalized.ops[1].outputs == ["output"]
    assert normalized.tensors["input_sigmoid"].shape == (1, 4)
    assert [op.step for op in normalized.ops] == [0, 1]


# Conv -> Relu fusion


def test_conv_relu_fused(tmp_path):
    """Conv followed by Relu should be fused into one op."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3],
                           np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4],
                           np.zeros(4, dtype=np.float32).tolist())

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="conv0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["output"], name="relu0")

    model = _make_model([conv, relu], "conv_relu", X, Y, [w, b])
    ag = _save_and_load(model, tmp_path)

    assert len(ag.ops) == 1
    assert ag.ops[0].op_type == "Conv"
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"
    assert ag.ops[0].outputs == ["output"]
    assert "t0" not in ag.tensors


def test_conv_relu6_fused(tmp_path):
    """Conv followed by Relu6 (from Clip(0,6)) should be fused."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3],
                           np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4],
                           np.zeros(4, dtype=np.float32).tolist())

    min_val = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "clip_min")
    max_val = numpy_helper.from_array(np.array([6.0], dtype=np.float32), "clip_max")

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="conv0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    clip = helper.make_node("Clip", ["t0", "clip_min", "clip_max"], ["output"], name="clip0")

    model = _make_model([conv, clip], "conv_relu6", X, Y, [w, b, min_val, max_val])
    ag = _save_and_load(model, tmp_path)

    assert len(ag.ops) == 1
    assert ag.ops[0].op_type == "Conv"
    assert ag.ops[0].attrs.get("fused_activation") == "Relu6"


def test_depthwise_relu_fused(tmp_path):
    """DepthwiseConv followed by Relu should be fused."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    # Depthwise: group=4, weight shape [4, 1, 3, 3]
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 1, 3, 3],
                           np.zeros((4, 1, 3, 3), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4],
                           np.zeros(4, dtype=np.float32).tolist())

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="dw0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0], group=4)
    relu = helper.make_node("Relu", ["t0"], ["output"], name="relu0")

    model = _make_model([conv, relu], "dw_relu", X, Y, [w, b])
    ag = _save_and_load(model, tmp_path)

    assert len(ag.ops) == 1
    assert ag.ops[0].op_type == "DepthwiseConv"
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"


def test_gemm_relu_fused(tmp_path):
    """Gemm followed by Relu should be fused."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])

    w = helper.make_tensor("w", TensorProto.FLOAT, [8, 16],
                           np.zeros((8, 16), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [8],
                           np.zeros(8, dtype=np.float32).tolist())

    gemm = helper.make_node("Gemm", ["input", "w", "b"], ["t0"], name="gemm0",
                            transB=1)
    relu = helper.make_node("Relu", ["t0"], ["output"], name="relu0")

    model = _make_model([gemm, relu], "gemm_relu", X, Y, [w, b])
    ag = _save_and_load(model, tmp_path)

    assert len(ag.ops) == 1
    assert ag.ops[0].op_type == "Gemm"
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"


# Cases that should NOT fuse


def test_add_relu_is_fused(tmp_path):
    """Relu following Add is absorbed into the Add.

    A quantizer writes a residual block's activation after the sum, on an edge
    the sum's own output requantization already covers, so the Add has to carry
    it.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

    w = helper.make_tensor("w", TensorProto.FLOAT, [1, 64],
                           np.zeros((1, 64), dtype=np.float32).flatten().tolist())

    add = helper.make_node("Add", ["input", "w"], ["t0"], name="add0")
    relu = helper.make_node("Relu", ["t0"], ["output"], name="relu0")

    model = _make_model([add, relu], "add_relu", X, Y, [w])
    ag = _save_and_load(model, tmp_path)

    assert len(ag.ops) == 1
    assert ag.ops[0].op_type == "Add"
    assert ag.ops[0].attrs["fused_activation"] == "Relu"
    assert ag.ops[0].outputs == ["output"]


def test_conv_two_consumers_not_fused(tmp_path):
    """Conv whose output feeds both Relu and another op should NOT be fused."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3],
                           np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4],
                           np.zeros(4, dtype=np.float32).tolist())

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="conv0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    # Second consumer of t0
    add = helper.make_node("Add", ["t1", "t0"], ["output"], name="add0")

    model = _make_model([conv, relu, add], "conv_branch", X, Y, [w, b])
    ag = _save_and_load(model, tmp_path)

    # Conv should not be fused because t0 has two consumers
    conv_op = [op for op in ag.ops if op.op_type == "Conv"][0]
    assert conv_op.attrs.get("fused_activation") is None


# Step indices contiguous


def test_step_indices_contiguous(tmp_path):
    """After fusion, step indices should be 0, 1, 2, ... without gaps."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    w0 = helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3],
                            np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 4, 3, 3],
                            np.zeros((4, 4, 3, 3), dtype=np.float32).flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [4],
                            np.zeros(4, dtype=np.float32).tolist())

    conv0 = helper.make_node("Conv", ["input", "w0", "b0"], ["t0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["t1"], name="relu0")
    conv1 = helper.make_node("Conv", ["t1", "w1", "b1"], ["output"], name="conv1",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])

    model = _make_model([conv0, relu, conv1], "conv_relu_conv", X, Y, [w0, b0, w1, b1])
    ag = _save_and_load(model, tmp_path)

    # Relu absorbed -> 2 ops
    assert len(ag.ops) == 2
    for i, op in enumerate(ag.ops):
        assert op.step == i


# Binary round-trip


def test_binary_roundtrip_preserves_fused_act(tmp_path):
    """fused_act field survives binary serialization round-trip."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])

    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3],
                           np.zeros((4, 3, 3, 3), dtype=np.float32).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4],
                           np.zeros(4, dtype=np.float32).tolist())

    conv = helper.make_node("Conv", ["input", "w", "b"], ["t0"], name="conv0",
                            kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["t0"], ["output"], name="relu0")

    model = _make_model([conv, relu], "conv_relu_rt", X, Y, [w, b])
    ag = _save_and_load(model, tmp_path)

    # Pipeline for binary
    from tigris.analysis.lifetime import compute_lifetimes
    from tigris.analysis.memory import compute_memory_timeline
    from tigris.analysis.partition_temporal import partition_temporal
    from tigris.analysis.partition_spatial import partition_spatial

    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    ag = partition_temporal(ag, 4096)
    ag = partition_spatial(ag)

    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["num_ops"] == 1
    conv_op = plan["ops"][0]
    assert conv_op["op_type"] == OP_TYPE_MAP["Conv"]
    assert conv_op["fused_act"] == ACT_RELU
    assert conv_op["act_min"] == -128
    assert conv_op["act_max"] == 127


def test_quantized_intermediate_blocks_fusion(tmp_path):
    """An intermediate carrying its own scale is a quantization step of its own.

    Folding the activation past it would drop that rounding, so the pass leaves
    both operators in place.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "zero_point"),
        numpy_helper.from_array(
            np.zeros((1, 64), dtype=np.float32), "bias"
        ),
    ]
    nodes = [
        helper.make_node("Add", ["input", "bias"], ["summed"]),
        # The sum is quantized before the activation reads it.
        helper.make_node(
            "QuantizeLinear", ["summed", "scale", "zero_point"], ["summed_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["summed_q", "scale", "zero_point"], ["summed_dq"]
        ),
        helper.make_node("Relu", ["summed_dq"], ["activated"]),
        helper.make_node(
            "QuantizeLinear", ["activated", "scale", "zero_point"], ["out_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["out_q", "scale", "zero_point"], ["output"]
        ),
    ]
    ag = _save_and_load(_make_model(nodes, "quantized_edge", X, Y, initializers), tmp_path)

    assert [op.op_type for op in ag.ops] == ["Add", "Relu"]
    assert "fused_activation" not in ag.ops[0].attrs


def test_quantized_add_relu_carries_the_output_scale(tmp_path):
    """The fused Add takes the activation's output quantization.

    quantize(Relu(x)) == max(quantize(x), zero_point), so the plan encodes the
    clamp as a lower bound at the output zero point.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "in_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "in_zp"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "out_scale"),
        numpy_helper.from_array(np.array([-8], dtype=np.int8), "out_zp"),
        numpy_helper.from_array(np.zeros((1, 64), dtype=np.float32), "bias"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_scale", "in_zp"], ["in_q"]),
        helper.make_node(
            "DequantizeLinear", ["in_q", "in_scale", "in_zp"], ["in_dq"]
        ),
        helper.make_node("Add", ["in_dq", "bias"], ["summed"]),
        helper.make_node("Relu", ["summed"], ["activated"]),
        helper.make_node(
            "QuantizeLinear", ["activated", "out_scale", "out_zp"], ["out_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["out_q", "out_scale", "out_zp"], ["output"]
        ),
    ]
    ag = _save_and_load(_make_model(nodes, "qdq_add_relu", X, Y, initializers), tmp_path)

    add = next(op for op in ag.ops if op.op_type == "Add")
    assert add.attrs["fused_activation"] == "Relu"
    quant = ag.tensors[add.outputs[0]].quant
    assert quant is not None
    assert float(quant.scale[0]) == 0.5
    assert int(quant.zero_point[0]) == -8
