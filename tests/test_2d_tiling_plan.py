"""Compiler serializes the HW tile axis and tile_width, and refuses to emit
a plan when even a 1x1 2D core tile does not fit the fast-memory budget.

Byte-level, mirrors test_linebuffer_plan.py: builds a real ONNX model, runs
the shared CLI pipeline, and inspects the emitted binary tile-plan record.
"""


import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris import TILE_AXIS_HW
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import TILE_BAND_KIND_HW
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes


def _build_high_res_conv(h: int, w: int, c: int) -> onnx.ModelProto:
    """A single 3x3 stride-1 pad-1 Conv on a [1,c,h,w] float32 activation.

    Same channel count in and out, so the stage's live-tensor peak is exactly
    two copies of the [1,c,h,w] tensor.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c, h, w])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c, h, w])
    w0 = numpy_helper.from_array(np.zeros((c, c, 3, 3), dtype=np.float32), "w0")
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [c], np.zeros(c, dtype=np.float32).tolist())
    conv0 = helper.make_node(
        "Conv", ["input", "w0", "b0"], ["output"], name="conv0",
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([conv0], "high_res_conv", [X], [Y], initializer=[w0, b0])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def compile_high_res_conv_to_bytes(h: int, w: int, c: int, budget: str, tmp_path) -> bytes:
    """Build the graph, run the shared pipeline, and emit the binary plan bytes.

    Follows the crossrepo _compile_plan pattern: validate feasibility before
    emitting so an infeasible plan raises with a descriptive reason instead
    of being silently written out.
    """
    model_path = tmp_path / "high_res_conv.onnx"
    onnx.save(_build_high_res_conv(h, w, c), str(model_path))
    ag, _ = _run_pipeline(str(model_path), (budget,))
    validation = validate_memory_plan(ag)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise AssertionError(f"compiler produced an infeasible graph: {details}")
    return emit_binary_bytes(ag)


def test_2d_plan_encodes_hw_axis_and_tile_width(tmp_path):
    plan_bytes = compile_high_res_conv_to_bytes(h=256, w=256, c=256, budget="24K", tmp_path=tmp_path)
    tp = read_binary_plan(plan_bytes)["tile_plans"][0]
    assert tp["tileable"]
    assert tp["axis"] == TILE_AXIS_HW
    assert tp["tile_extent"] >= 1 and tp["output_width_extent"] >= 1
    assert (tp["band_kind"], tp["band_mode"]) == (TILE_BAND_KIND_HW, 0)


def test_even_2d_min_tile_infeasible_fails_closed(tmp_path):
    # Budget so small even a 1x1 2D core will not fit -> compile refuses.
    with pytest.raises(Exception) as exc:
        compile_high_res_conv_to_bytes(h=64, w=64, c=64, budget="1K", tmp_path=tmp_path)
    assert "2D tile" in str(exc.value)
