"""Tests for tigris.analysis.lifetime."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes, pure_reinterpretations


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _analyzed(tmp_path, name, nodes, inputs, outputs, init=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(init))
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return compute_lifetimes(load_model(path))


def test_model_input_birth(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert ag.lifetimes["input"].birth_step == -1


def test_model_output_death(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert ag.lifetimes["output"].death_step == len(ag.ops)


def test_constants_excluded(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    assert "w0" not in ag.lifetimes
    assert "w1" not in ag.lifetimes


def test_intermediate_lifetime(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)

    # t0 is produced by add0 and consumed by relu0
    lt = ag.lifetimes["t0"]
    assert lt.birth_step >= 0
    assert lt.death_step > lt.birth_step


def test_diamond_lifetimes(diamond_path):
    ag = load_model(diamond_path)
    ag = compute_lifetimes(ag)

    # Both left and right should be alive until the Add consumes them
    assert "left" in ag.lifetimes
    assert "right" in ag.lifetimes

    # input is consumed by both relu and sigmoid - death = max of those steps
    assert ag.lifetimes["input"].death_step >= 0


def test_a_reshape_that_moves_no_byte_shares_its_input(tmp_path):
    """The output is the same bytes read through a different shape.

    Both tensors state their own axis order, so regrouping the shape moves
    nothing and the executor gives the two the same buffer. The model counts
    one allocation, living until the last of the pair is read.
    """
    tokens, width = 8, 8
    ag = _analyzed(
        tmp_path, "reshape_alias",
        [
            helper.make_node("MatMul", ["x", "w"], ["p"], name="mm"),
            helper.make_node("Reshape", ["p", "shp"], ["r"], name="rs"),
            helper.make_node("Erf", ["r"], ["y"], name="erf"),
        ],
        [_vi("x", [1, tokens, width])], [_vi("y", [1, tokens, 2, width // 2])],
        [
            numpy_helper.from_array(
                np.zeros((width, width), np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, tokens, 2, width // 2], np.int64), "shp"),
        ])

    alias = pure_reinterpretations(ag)
    assert "r" in alias, alias
    assert ag.lifetimes["r"].size_bytes == 0
    # The root's storage has to survive the alias's last read.
    root = alias["r"]
    assert ag.lifetimes[root].death_step >= ag.lifetimes["r"].death_step


def test_an_unfold_over_the_spatial_axes_shares_its_input(tmp_path):
    """A feature map regrouped into patches is the same bytes channels-last."""
    channels, side = 8, 4
    ag = _analyzed(
        tmp_path, "unfold_alias",
        [
            helper.make_node(
                "Conv", ["x", "w"], ["c"], kernel_shape=[1, 1], name="cv"),
            helper.make_node("Reshape", ["c", "shp"], ["u"], name="rs"),
        ],
        [_vi("x", [1, channels, side, side])],
        [_vi("u", [1, channels, side * side, 1])],
        [
            numpy_helper.from_array(
                np.zeros((channels, channels, 1, 1), np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, channels, side * side, 1], np.int64), "shp"),
        ])

    alias = pure_reinterpretations(ag)
    assert "u" in alias, alias
    assert ag.lifetimes["u"].size_bytes == 0


def test_a_reshape_that_moves_channels_is_not_an_alias(tmp_path):
    """Changing the channel extent regroups the stored order, not the shape."""
    channels, side = 8, 4
    ag = _analyzed(
        tmp_path, "regroup_channels",
        [
            helper.make_node(
                "Conv", ["x", "w"], ["c"], kernel_shape=[1, 1], name="cv"),
            helper.make_node("Reshape", ["c", "shp"], ["u"], name="rs"),
        ],
        [_vi("x", [1, channels, side, side])],
        [_vi("u", [1, side * side, channels, 1])],
        [
            numpy_helper.from_array(
                np.zeros((channels, channels, 1, 1), np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, side * side, channels, 1], np.int64), "shp"),
        ])

    assert "u" not in pure_reinterpretations(ag)
    assert ag.lifetimes["u"].size_bytes > 0
