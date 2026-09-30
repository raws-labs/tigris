"""A height stripe is sized by its band buffers as the runtime aligns them."""

import onnx
from onnx import TensorProto, helper

from tigris.cli import _run_pipeline


def _mul(tmp_path):
    shape = [1, 4, 5, 5]
    model = helper.make_model(
        helper.make_graph([helper.make_node("Mul", ["x", "y"], ["z"])], "mul",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
                           helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)]),
        opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "mul.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_a_one_row_band_counts_each_buffer_aligned(tmp_path):
    # Three 80-byte rows, each aligned to 96 bytes: 288, not 240.
    ag, _ = _run_pipeline(_mul(tmp_path), ("288",), report_bindings=False)
    plan = ag.stages[0].tile_plan
    assert plan.tileable and plan.tile_height == 1
    assert plan.tiled_peak_bytes == 288
    assert not plan.warnings


def test_a_band_that_does_not_fit_aligned_narrows_to_a_2d_tile(tmp_path):
    # One aligned row needs 288 bytes, so a 256-byte budget takes a 2D tile.
    ag, _ = _run_pipeline(_mul(tmp_path), ("256",), report_bindings=False)
    plan = ag.stages[0].tile_plan
    assert plan.tileable and plan.tile_width > 0
    assert plan.tiled_peak_bytes <= 256
