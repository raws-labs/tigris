"""Tests for tigris.loaders."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model

from conftest import save_onnx


def test_linear_load(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.model_name == "linear_3op"
    # relu0 must come before add0, add0 before add1
    assert [op.op_type for op in ag.ops] == ["Relu", "Add", "Add"]
    assert ag.model_inputs == ["input"]
    assert ag.model_outputs == ["output"]

    info = ag.tensors["input"]
    assert info.shape == (1, 64)
    assert info.size_bytes == 1 * 64 * 4  # float32

    assert ag.tensors["w0"].is_constant is True
    assert ag.tensors["w1"].is_constant is True
    assert ag.tensors["input"].is_constant is False


def test_loader_extracts_spatial_attrs(conv_relu_chain_path, conv_pool_chain_path):
    conv_ops = [op for op in load_model(conv_relu_chain_path).ops if op.op_type == "Conv"]
    assert len(conv_ops) == 2
    for op in conv_ops:
        assert op.attrs["kernel_shape"] == [3, 3]
        assert op.attrs["strides"] == [1, 1]

    pool_ops = [op for op in load_model(conv_pool_chain_path).ops if op.op_type == "MaxPool"]
    assert len(pool_ops) == 1
    assert pool_ops[0].attrs["kernel_shape"] == [2, 2]
    assert pool_ops[0].attrs["strides"] == [2, 2]


def _save_model(tmp_path, name, nodes, inputs, outputs, initializers):
    return save_onnx(
        tmp_path, name, nodes, inputs, outputs, initializers, opset=17, ir_version=8
    )


def test_reshape_target_shape_is_not_a_weight(tmp_path):
    """A shape vector is metadata; binding it as a weight makes a plan the
    runtime refuses to load."""
    path = _save_model(
        tmp_path,
        "reshape_init",
        [
            helper.make_node("Relu", ["input"], ["act"]),
            helper.make_node("Reshape", ["act", "target_shape"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "target_shape")],
    )

    ag = load_model(path)

    reshape = next(op for op in ag.ops if op.op_type == "Reshape")
    # The operand is gone, whichever tensor the layout pass leaves feeding it:
    # regrouping a feature map needs it held in the model's own order first.
    assert reshape.inputs == [reshape.inputs[0]]
    assert "target_shape" not in reshape.inputs
    assert "target_shape" not in ag.weight_data


def test_clip_bounds_are_not_weights(tmp_path):
    path = _save_model(
        tmp_path,
        "clip_init",
        [helper.make_node("Clip", ["input", "lo", "hi"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "lo"),
            numpy_helper.from_array(np.array(6.0, dtype=np.float32), "hi"),
        ],
    )

    ag = load_model(path)

    clip = ag.ops[0]
    assert clip.op_type == "Relu6"
    assert clip.inputs == ["input"]
    assert ag.weight_data == {}


def test_runtime_computed_metadata_operand_is_kept(tmp_path):
    """Only constants are metadata the compiler can drop."""
    path = _save_model(
        tmp_path,
        "reshape_dynamic",
        [
            helper.make_node("Relu", ["input"], ["act"]),
            helper.make_node("Reshape", ["act", "target_shape"], ["output"]),
        ],
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("target_shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [],
    )

    ag = load_model(path)

    reshape = next(op for op in ag.ops if op.op_type == "Reshape")
    # A shape the graph computes at runtime stays an operand, whichever
    # tensor the layout pass leaves as the data one.
    assert len(reshape.inputs) == 2
    assert reshape.inputs[1] == "target_shape"


def test_orphan_value_info_is_not_a_tensor(tmp_path):
    """A graph optimizer leaves value_info behind for tensors its fusion removed.

    Those entries describe nothing the graph computes. Carrying them into the
    IR makes a fully quantized model look mixed-dtype and be rejected.
    """
    path = _save_model(
        tmp_path,
        "orphan_value_info",
        [helper.make_node("Relu", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [],
    )
    model = onnx.load(str(path))
    model.graph.value_info.append(
        helper.make_tensor_value_info("fused_away", TensorProto.FLOAT, [1, 4])
    )
    onnx.save(model, path)

    ag = load_model(path)

    assert "fused_away" not in ag.tensors
    assert set(ag.tensors) == {"input", "output"}
