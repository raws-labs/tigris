"""Tests for tigris.analysis.lifetime."""

import numpy as np
from onnx import helper, numpy_helper

from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes, pure_reinterpretations

from conftest import save_onnx, vi


def _analyzed(tmp_path, name, nodes, inputs, outputs, init=()):
    path = save_onnx(tmp_path, name, nodes, inputs, outputs, init, opset=17, ir_version=8)
    return compute_lifetimes(load_model(path))


def _lifetimes(ag):
    return {name: (lt.birth_step, lt.death_step, lt.size_bytes) for name, lt in ag.lifetimes.items()}


def test_linear_lifetimes(linear_3op_path):
    """Model inputs are born at -1, outputs die after the last op, constants are absent."""
    ag = compute_lifetimes(load_model(linear_3op_path))

    assert _lifetimes(ag) == {
        "input": (-1, 0, 256), "t0": (0, 1, 256), "t1": (1, 2, 256), "output": (2, 3, 256),
    }


def test_diamond_lifetimes(diamond_path):
    """The input lives until its second reader; both branches live until the Add."""
    ag = compute_lifetimes(load_model(diamond_path))

    assert _lifetimes(ag) == {
        "input": (-1, 1, 512), "right": (0, 2, 512), "left": (1, 2, 512), "output": (2, 3, 512),
    }


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
        [vi("x", [1, tokens, width])], [vi("y", [1, tokens, 2, width // 2])],
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
        [vi("x", [1, channels, side, side])],
        [vi("u", [1, channels, side * side, 1])],
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
        [vi("x", [1, channels, side, side])],
        [vi("u", [1, side * side, channels, 1])],
        [
            numpy_helper.from_array(
                np.zeros((channels, channels, 1, 1), np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, side * side, channels, 1], np.int64), "shp"),
        ])

    assert "u" not in pure_reinterpretations(ag)
    assert ag.lifetimes["u"].size_bytes > 0
