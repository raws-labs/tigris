"""A float per-channel constant Add folded into its producer's bias."""

import numpy as np
from onnx import helper, numpy_helper

from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize
from tigris.loaders.onnx.normalize.fusion import _axis_broadcast_size, _bias_broadcast_size

from conftest import save_onnx, vi


def _normalized(tmp_path, name, nodes, inputs, outputs, init=()):
    path = save_onnx(tmp_path, name, nodes, inputs, outputs, init)
    return normalize(load_model(str(path)))


def _weight(name="weight"):
    return numpy_helper.from_array(
        np.full((3, 2, 3, 3), 0.25, np.float32), name)


def _channel(values, shape, name="channel"):
    return numpy_helper.from_array(
        np.array(values, np.float32).reshape(shape), name)


def test_channel_add_becomes_the_producers_bias(tmp_path):
    ag = _normalized(
        tmp_path, "as_bias",
        [
            helper.make_node("Conv", ["x", "weight"], ["p"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv1"),
            helper.make_node("Add", ["p", "channel"], ["y"], name="add1"),
        ],
        [vi("x", [1, 2, 3, 3])], [vi("y", [1, 3, 3, 3])],
        [_weight(), _channel([0.5, -1.25, 2.0], (1, 3, 1, 1))])

    assert [op.op_type for op in ag.ops] == ["Conv"]
    conv = ag.ops[0]
    assert len(conv.inputs) == 3
    assert np.allclose(ag.weight_data[conv.inputs[2]], [0.5, -1.25, 2.0])
    # The plan must still name what the model named.
    assert conv.outputs == ["y"]


def test_channel_add_sums_into_an_existing_bias(tmp_path):
    ag = _normalized(
        tmp_path, "onto_bias",
        [
            helper.make_node("Conv", ["x", "weight", "bias"], ["p"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv1"),
            helper.make_node("Add", ["p", "channel"], ["y"], name="add1"),
        ],
        [vi("x", [1, 2, 3, 3])], [vi("y", [1, 3, 3, 3])],
        [_weight(),
         numpy_helper.from_array(np.array([1.0, 2.0, 3.0], np.float32), "bias"),
         _channel([0.5, -1.25, 2.0], (1, 3, 1, 1))])

    assert [op.op_type for op in ag.ops] == ["Conv"]
    assert np.allclose(
        ag.weight_data[ag.ops[0].inputs[2]], [1.5, 0.75, 5.0])


def test_add_without_a_bias_producer_is_left_alone(tmp_path):
    """Nothing to fold into: the Add must survive and be reported unsupported."""
    ag = _normalized(
        tmp_path, "no_producer",
        [helper.make_node("Add", ["x", "channel"], ["y"], name="add1")],
        [vi("x", [1, 3, 4, 4])], [vi("y", [1, 3, 4, 4])],
        [_channel([0.5, -1.25, 2.0], (1, 3, 1, 1))])

    assert [op.op_type for op in ag.ops] == ["Add"]


def test_add_feeding_a_second_consumer_is_left_alone(tmp_path):
    """Folding past another reader of the product would change what it sees."""
    ag = _normalized(
        tmp_path, "shared_product",
        [
            helper.make_node("Conv", ["x", "weight"], ["p"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv1"),
            helper.make_node("Add", ["p", "channel"], ["a"], name="add1"),
            helper.make_node("Relu", ["p"], ["b"], name="relu1"),
            helper.make_node("Add", ["a", "b"], ["y"], name="add2"),
        ],
        [vi("x", [1, 2, 3, 3])], [vi("y", [1, 3, 3, 3])],
        [_weight(), _channel([0.5, -1.25, 2.0], (1, 3, 1, 1))])

    assert "Add" in [op.op_type for op in ag.ops]


def test_axis_broadcast_size_reads_the_aligned_axis():
    ref = (1, 8, 4, 8)
    # Right-aligned forms that address the channel axis.
    assert _axis_broadcast_size((1, 8, 1, 1), ref, 1) == 8
    assert _axis_broadcast_size((8, 1, 1), ref, 1) == 8
    # Width has the same extent as C here: size alone would confuse the two.
    assert _axis_broadcast_size((1, 1, 1, 8), ref, 1) is None
    assert _axis_broadcast_size((8,), ref, 1) is None
    assert _axis_broadcast_size((4, 1), ref, 1) is None


def test_which_axis_a_bias_addresses_follows_its_producer():
    """A convolution biases the channel; a matrix product the last axis.

    The two disagree on a token sequence, where the feature axis is last and
    the axis the model states second is the sequence length.
    """
    image = (1, 8, 16, 16)
    tokens = (4, 64, 192)

    assert _bias_broadcast_size("Conv", (8, 1, 1), image) == 8
    # Right-alignment puts a bare rank-1 constant on the width axis of an
    # image, which is why a convolution needs the trailing ones spelled out.
    assert _bias_broadcast_size("Conv", (8,), image) is None
    assert _bias_broadcast_size("Conv", (192,), tokens) is None

    for producer in ("Gemm", "MatMul"):
        assert _bias_broadcast_size(producer, (192,), tokens) == 192
        assert _bias_broadcast_size(producer, (1, 1, 192), tokens) == 192
        # The sequence length is not a feature the product biases.
        assert _bias_broadcast_size(producer, (64,), tokens) is None
        assert _bias_broadcast_size(producer, (1, 64, 1), tokens) is None
