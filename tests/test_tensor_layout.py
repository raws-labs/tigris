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
