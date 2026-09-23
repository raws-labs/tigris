"""Operators an exporter leaves behind that compute nothing at inference."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize, _stored_extent


def _compile_graph(tmp_path, name, nodes, inputs, outputs, init=(), opset=13):
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, list(init)),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return normalize(load_model(str(path)))


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_dropout_and_identity_are_removed(tmp_path):
    ag = _compile_graph(
        tmp_path, "identities",
        [
            helper.make_node("Identity", ["x"], ["a"], name="id1"),
            helper.make_node("Dropout", ["a"], ["b"], ratio=0.5, name="drop1"),
            helper.make_node("Relu", ["b"], ["y"], name="relu1"),
        ],
        [_vi("x", [1, 8])], [_vi("y", [1, 8])], opset=11)

    types = [op.op_type for op in ag.ops]
    assert types == ["Relu"], types
    # The survivor must read the original graph input, not a dropped tensor.
    assert ag.ops[0].inputs[0] == "x"


def test_dropout_keeping_its_mask_is_left_alone(tmp_path):
    """The mask output only exists for training; an op whose mask is read stays."""
    ag = _compile_graph(
        tmp_path, "dropout_mask",
        [
            helper.make_node("Dropout", ["x"], ["b", "mask"], name="drop1"),
            helper.make_node("Relu", ["b"], ["y"], name="relu1"),
        ],
        [_vi("x", [1, 8])],
        [_vi("y", [1, 8]), helper.make_tensor_value_info(
            "mask", TensorProto.BOOL, [1, 8])],
        opset=12)

    assert "Dropout" in [op.op_type for op in ag.ops]


def test_squeeze_becomes_reshape_when_element_order_survives(tmp_path):
    """Dropping the trailing unit axes of a pooled tensor is a pure reshape."""
    ag = _compile_graph(
        tmp_path, "squeeze_safe",
        [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="sq1")],
        [_vi("x", [1, 6, 1, 1])], [_vi("y", [1, 6])],
        [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])

    assert [op.op_type for op in ag.ops] == ["Reshape"]
    # The axes operand is gone: the plan takes the shape from the tensor table.
    assert len(ag.ops[0].inputs) == 1


def test_unsqueeze_becomes_reshape_when_element_order_survives(tmp_path):
    ag = _compile_graph(
        tmp_path, "unsqueeze_safe",
        [helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="un1")],
        [_vi("x", [1, 6])], [_vi("y", [1, 6, 1, 1])],
        [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])

    assert [op.op_type for op in ag.ops] == ["Reshape"]


def test_a_squeeze_that_would_transpose_is_converted_first(tmp_path):
    """Rank 4 to rank 3 dropping the channel axis reorders under NHWC/NLC.

    NHWC [1, H, W, 1] walks (h, w); the resulting NLC [1, W, H] walks (w, h).
    Copying the bytes through would transpose the picture, so the operand is
    converted to the model's own order first and the regrouping then moves
    nothing. The conversions are explicit Transposes around the reshape, and
    the last one puts the model output back in the layout callers expect.
    """
    ag = _compile_graph(
        tmp_path, "squeeze_unsafe",
        [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="sq1")],
        [_vi("x", [1, 1, 8, 4])], [_vi("y", [1, 8, 4])],
        [numpy_helper.from_array(np.array([1], np.int64), "axes")])

    types = [op.op_type for op in ag.ops]
    assert types.count("Reshape") == 1
    assert types[0] == "Transpose" and types[-1] == "Transpose"


def test_a_single_channel_squeeze_needs_no_conversion(tmp_path):
    """The permutation that formally moves a unit channel axis moves nothing."""
    ag = _compile_graph(
        tmp_path, "squeeze_unit_channel",
        [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="sq1")],
        [_vi("x", [1, 6, 1, 1])], [_vi("y", [1, 6])],
        [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])

    assert [op.op_type for op in ag.ops] == ["Reshape"]


def test_stored_extent_orders_axes_the_way_the_runtime_serializes_them():
    # rank 4 is NCHW -> NHWC, rank 3 is NCL -> NLC, everything else is as-is.
    assert _stored_extent((1, 1, 8, 4)) == (8, 4)     # N,H,W,C over [1,1,8,4]
    assert _stored_extent((1, 8, 4)) == (4, 8)        # N,L,C over [1,8,4]
    assert _stored_extent((1, 6, 1, 1)) == (6,)
    assert _stored_extent((1, 6)) == (6,)


def test_a_gather_of_one_index_becomes_a_cut(tmp_path):
    """Reading one index out of an axis nothing precedes is a run of bytes.

    The runtime carries no Gather, and it does not need one: the wanted index
    starts a contiguous run, which is what a Split cuts.
    """
    proj = np.eye(8, dtype=np.float32)
    ag = _compile_graph(
        tmp_path, "gather_one",
        [
            helper.make_node("MatMul", ["x", "proj"], ["mixed"], name="mix"),
            helper.make_node("Gather", ["mixed", "first"], ["y"], axis=1,
                             name="pick"),
        ],
        [_vi("x", [1, 4, 8])], [_vi("y", [1, 8])],
        init=[numpy_helper.from_array(proj, "proj"),
              numpy_helper.from_array(np.array(0, dtype=np.int64), "first")],
    )
    types = [op.op_type for op in ag.ops]
    assert "Gather" not in types
    assert "Split" in types
    cut = ag.ops[types.index("Split")]
    parts = [ag.tensors[name].shape for name in cut.outputs]
    assert parts == [(1, 8), (3, 8)]


def test_a_gather_of_a_middle_index_keeps_both_sides(tmp_path):
    proj = np.eye(8, dtype=np.float32)
    ag = _compile_graph(
        tmp_path, "gather_middle",
        [
            helper.make_node("MatMul", ["x", "proj"], ["mixed"], name="mix"),
            helper.make_node("Gather", ["mixed", "second"], ["y"], axis=1,
                             name="pick"),
        ],
        [_vi("x", [1, 4, 8])], [_vi("y", [1, 8])],
        init=[numpy_helper.from_array(proj, "proj"),
              numpy_helper.from_array(np.array(2, dtype=np.int64), "second")],
    )
    types = [op.op_type for op in ag.ops]
    assert "Gather" not in types
    cut = ag.ops[types.index("Split")]
    assert [ag.tensors[name].shape for name in cut.outputs] == [
        (2, 8), (1, 8), (1, 8)]


def test_a_gather_on_an_axis_something_precedes_is_left_alone(tmp_path):
    """The wanted index is then scattered through the stored bytes."""
    proj = np.eye(8, dtype=np.float32)
    ag = _compile_graph(
        tmp_path, "gather_inner",
        [
            helper.make_node("MatMul", ["x", "proj"], ["mixed"], name="mix"),
            helper.make_node("Gather", ["mixed", "first"], ["y"], axis=2,
                             name="pick"),
        ],
        [_vi("x", [1, 4, 8])], [_vi("y", [1, 4])],
        init=[numpy_helper.from_array(proj, "proj"),
              numpy_helper.from_array(np.array(0, dtype=np.int64), "first")],
    )
    assert "Gather" in [op.op_type for op in ag.ops]
