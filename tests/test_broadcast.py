"""Which binary broadcasts a tile can carry."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.broadcast import DENSE, GENERAL, PERIODIC, access
from tigris.analysis.partition_spatial import TileCategory, op_category
from tigris.loaders.onnx.loader import load_model
from tigris.loaders.onnx.normalize import normalize


@pytest.mark.parametrize("stored, how", [
    ((1, 6, 6, 4), DENSE),
    ((1, 1, 1, 4), PERIODIC),
    ((1, 1, 6, 4), PERIODIC),
    ((1, 1, 1, 1), PERIODIC),
    ((1, 6, 1, 4), GENERAL),
    ((1, 6, 6, 1), GENERAL),
    ((1, 3, 6, 4), None),
])
def test_access_follows_the_stored_shape(stored, how):
    assert access(stored, (1, 6, 6, 4)) == how


def _binary(tmp_path, first_shape, second_shape, constant=False):
    inputs = [helper.make_tensor_value_info("a", TensorProto.FLOAT, first_shape)]
    init = []
    if constant:
        init.append(numpy_helper.from_array(np.ones(second_shape, np.float32), "b"))
    else:
        inputs.append(helper.make_tensor_value_info("b", TensorProto.FLOAT, second_shape))
    output = [max(x, y) for x, y in zip(first_shape, second_shape)]
    model = helper.make_model(
        helper.make_graph([helper.make_node("Add", ["a", "b"], ["y"])], "g", inputs,
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, output)], init),
        opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    ag = normalize(load_model(str(path)))
    return next(op for op in ag.ops if op.op_type == "Add")


@pytest.mark.parametrize("second, constant, tileable", [
    ([1, 4, 1, 1], False, True),    # one value per channel, second
    ([1, 4, 1, 1], True, True),     # a constant that repeats
    ([1, 4, 6, 1], False, False),   # one value per row: read by coordinate
    ([1, 1, 6, 6], False, False),   # a tensor that repeats but is not per channel
    ([1, 4, 6, 1], True, False),    # a constant read by coordinate
])
def test_only_carried_broadcasts_stay_tileable(tmp_path, second, constant, tileable):
    op = _binary(tmp_path, [1, 4, 6, 6], second, constant)
    assert (op_category(op) is not TileCategory.UNTILEABLE) == tileable
