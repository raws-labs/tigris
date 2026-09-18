"""Layout as a tensor property rather than a rule inferred from rank."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import Layout, TensorInfo, serialized_axis_map
from tigris.cli import _run_pipeline


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _plan(tmp_path, name, nodes, inputs, outputs, init=(), budget="64K"):
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, list(init)),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    ag, _ = _run_pipeline(str(path), (budget,))
    return ag, read_binary_plan(emit_binary_bytes(ag))


def test_tensors_are_spatial_by_default():
    assert TensorInfo("a", (1, 3, 4, 4), 1).layout is Layout.SPATIAL


def test_serialized_axis_map_follows_the_layout():
    # Spatial tensors are held channels-last; linear ones are already in order.
    assert serialized_axis_map(4, Layout.SPATIAL) == [0, 3, 1, 2]
    assert serialized_axis_map(3, Layout.SPATIAL) == [0, 2, 1]
    assert serialized_axis_map(4, Layout.LINEAR) == [0, 1, 2, 3]
    assert serialized_axis_map(3, Layout.LINEAR) == [0, 1, 2]
    # Rank 2 and below cannot disagree.
    assert serialized_axis_map(2, Layout.SPATIAL) == [0, 1]


def test_spatial_activation_is_stored_channels_last(tmp_path):
    ag, plan = _plan(
        tmp_path, "spatial",
        [helper.make_node("Conv", ["x", "w"], ["y"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv1")],
        [_vi("x", [1, 2, 5, 7])], [_vi("y", [1, 4, 5, 7])],
        [numpy_helper.from_array(np.zeros((4, 2, 3, 3), np.float32), "w")])

    assert ag.tensors["y"].layout is Layout.SPATIAL
    stored = next(t for t in plan["tensors"] if t["name"] == "y")
    assert tuple(stored["shape"]) == (1, 5, 7, 4)   # NCHW [1,4,5,7] -> NHWC


def test_terminal_transpose_output_is_linear_and_keeps_onnx_shape(tmp_path):
    """The boundary a model observes keeps ONNX order, so it is a linear tensor."""
    ag, plan = _plan(
        tmp_path, "terminal_transpose",
        [
            helper.make_node("Conv", ["x", "w"], ["c"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv1"),
            helper.make_node("Transpose", ["c"], ["y"], perm=[0, 2, 3, 1],
                             name="t1"),
        ],
        [_vi("x", [1, 2, 5, 7])], [_vi("y", [1, 5, 7, 4])],
        [numpy_helper.from_array(np.zeros((4, 2, 3, 3), np.float32), "w")])

    assert ag.tensors["c"].layout is Layout.SPATIAL
    assert ag.tensors["y"].layout is Layout.LINEAR
    stored = next(t for t in plan["tensors"] if t["name"] == "y")
    assert tuple(stored["shape"]) == (1, 5, 7, 4)   # unpermuted, as declared


def _operand_layouts(ag, op):
    return [
        ag.tensors[name].layout
        for name in op.inputs
        if name and name in ag.tensors
        and not ag.tensors[name].is_constant
        and len(ag.tensors[name].shape) >= 3
    ]


def test_a_residual_over_a_matrix_product_unifies_its_operands(tmp_path):
    """The skip arrives in storage order and the product's output does not.

    An elementwise Add reads both operands at the same offset, so two operands
    held in different axis orders add unrelated elements. Where the two shapes
    differ the loader catches it; where they match, as they do whenever the
    sequence length equals the width, nothing does and the answer is wrong.
    """
    tokens, width = 8, 8
    ag, _ = _plan(
        tmp_path, "residual_square",
        [
            helper.make_node("MatMul", ["x", "w"], ["p"], name="mm"),
            helper.make_node("Add", ["x", "p"], ["y"], name="add"),
        ],
        [_vi("x", [1, tokens, width])], [_vi("y", [1, tokens, width])],
        [numpy_helper.from_array(
            np.zeros((width, width), np.float32), "w")],
        budget="4M")

    add = next(op for op in ag.ops if op.op_type == "Add")
    layouts = _operand_layouts(ag, add)
    assert len(layouts) == 2
    assert len(set(layouts)) == 1, (
        f"the Add's operands disagree about layout: {layouts}")


def test_every_layout_agnostic_operator_has_operands_that_agree(tmp_path):
    """The invariant, over the shapes that put a graph in both layouts."""
    tokens, width = 8, 8
    ag, _ = _plan(
        tmp_path, "mixed",
        [
            helper.make_node("MatMul", ["x", "w"], ["p"], name="mm"),
            helper.make_node("Add", ["x", "p"], ["s"], name="add"),
            helper.make_node("Mul", ["s", "x"], ["m"], name="mul"),
            helper.make_node("Softmax", ["m"], ["sm"], axis=-1, name="sm"),
            helper.make_node("Sub", ["sm", "x"], ["y"], name="sub"),
        ],
        [_vi("x", [1, tokens, width])], [_vi("y", [1, tokens, width])],
        [numpy_helper.from_array(
            np.zeros((width, width), np.float32), "w")],
        budget="4M")

    for op in ag.ops:
        if op.op_type not in ("Add", "Sub", "Mul", "Concat"):
            continue
        layouts = _operand_layouts(ag, op)
        assert len(set(layouts)) <= 1, (
            f"{op.name} ({op.op_type}) operands disagree: {layouts}")
