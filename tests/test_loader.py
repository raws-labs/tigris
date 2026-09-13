"""Tests for tigris.loaders."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model


def test_linear_load(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.model_name == "linear_3op"
    assert len(ag.ops) == 3
    assert ag.model_inputs == ["input"]
    assert ag.model_outputs == ["output"]


def test_linear_topo_order(linear_3op_path):
    ag = load_model(linear_3op_path)

    # Steps should be 0, 1, 2 in a valid topological order
    op_types = [op.op_type for op in ag.ops]
    # relu0 must come before add0, add0 before add1
    assert op_types == ["Relu", "Add", "Add"]


def test_diamond_load(diamond_path):
    ag = load_model(diamond_path)

    assert len(ag.ops) == 3
    assert "input" in ag.model_inputs
    assert "output" in ag.model_outputs


def test_tensors_have_shapes(linear_3op_path):
    ag = load_model(linear_3op_path)

    info = ag.tensors["input"]
    assert info.shape == (1, 64)
    assert info.size_bytes == 1 * 64 * 4  # float32


def test_constants_marked(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.tensors["w0"].is_constant is True
    assert ag.tensors["w1"].is_constant is True
    assert ag.tensors["input"].is_constant is False


def _save_model(tmp_path, name, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return path


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
    assert reshape.inputs == ["act"]
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
    assert reshape.inputs == ["act", "target_shape"]
