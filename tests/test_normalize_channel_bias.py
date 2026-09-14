"""A float per-channel constant Add folded into its producer's bias."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize, _channel_broadcast_size


def _normalized(tmp_path, name, nodes, inputs, outputs, init=()):
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, list(init)),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return normalize(load_model(str(path)))


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


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
        [_vi("x", [1, 2, 3, 3])], [_vi("y", [1, 3, 3, 3])],
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
        [_vi("x", [1, 2, 3, 3])], [_vi("y", [1, 3, 3, 3])],
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
        [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 4, 4])],
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
        [_vi("x", [1, 2, 3, 3])], [_vi("y", [1, 3, 3, 3])],
        [_weight(), _channel([0.5, -1.25, 2.0], (1, 3, 1, 1))])

    assert "Add" in [op.op_type for op in ag.ops]


def test_channel_broadcast_size_reads_the_aligned_axis():
    ref = (1, 8, 4, 8)
    # Right-aligned forms that address the channel axis.
    assert _channel_broadcast_size((1, 8, 1, 1), ref) == 8
    assert _channel_broadcast_size((8, 1, 1), ref) == 8
    # Width has the same extent as C here: size alone would confuse the two.
    assert _channel_broadcast_size((1, 1, 1, 8), ref) is None
    assert _channel_broadcast_size((8,), ref) is None
    assert _channel_broadcast_size((4, 1), ref) is None
