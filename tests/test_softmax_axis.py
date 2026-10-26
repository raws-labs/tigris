"""Which Softmax axis is reducible is a property of the tensor's layout."""

import pytest
from onnx import TensorProto, helper

from tigris.analysis.validation import validate_operator_support
from tigris.graph.ir import Layout

from conftest import normalized


def _normalized(tmp_path, shape, axis):
    node = helper.make_node("Softmax", ["x"], ["y"], axis=axis, name="sm1")
    return normalized(tmp_path, "softmax_axis", [node],
                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
                      filename="softmax.onnx")


@pytest.mark.parametrize("shape", [[1, 8, 16], [1, 4, 3, 5]])
def test_last_axis_softmax_reads_the_models_own_order(tmp_path, shape):
    ag = _normalized(tmp_path, shape, -1)
    softmax = next(op for op in ag.ops if op.op_type == "Softmax")
    assert ag.tensors[softmax.inputs[0]].layout is Layout.LINEAR
    assert validate_operator_support(ag).supported


@pytest.mark.parametrize("shape", [[1, 8, 16], [1, 4, 3, 5]])
def test_channel_axis_softmax_stays_spatial(tmp_path, shape):
    ag = _normalized(tmp_path, shape, 1)
    softmax = next(op for op in ag.ops if op.op_type == "Softmax")
    assert ag.tensors[softmax.inputs[0]].layout is Layout.SPATIAL
    assert validate_operator_support(ag).supported


def test_middle_axis_softmax_has_no_layout_that_helps(tmp_path):
    """Rank-4 axis 2 is neither the channel axis nor the last one."""
    ag = _normalized(tmp_path, [1, 4, 3, 5], 2)
    support = validate_operator_support(ag)
    assert not support.supported
    assert "final dimension" in support.describe()


def test_rank2_softmax_needs_no_conversion(tmp_path):
    """Below rank 3 the two orders coincide, so nothing is inserted."""
    ag = _normalized(tmp_path, [4, 8], -1)
    assert [op.op_type for op in ag.ops] == ["Softmax"]
    assert validate_operator_support(ag).supported
