"""The reported minimum fast arena is what compile would require, not a stage's untiled peak."""

import onnx
from onnx import TensorProto, helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline


def _global_pool(path, channels=64, side=8):
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["output"], name="gap")],
        "gap",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 1, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def test_a_pool_groups_below_its_full_row_floor(tmp_path):
    path = tmp_path / "gap.onnx"
    _global_pool(path)
    graph, _ = _run_pipeline(str(path), ("2K",))
    assert validate_memory_plan(graph).feasible
    tile = graph.stages[0].tile_plan
    assert tile.input_channel_group > 0
    assert tile.input_spatial_interval > 0
    assert tile.tiled_peak_bytes <= 2048
    assert compute_findings(graph).blocking_stages == []


def test_grouped_pool_has_three_aligned_buffers_at_its_floor(tmp_path):
    path = tmp_path / "gap.onnx"
    _global_pool(path)
    for budget, feasible in ((95, False), (96, True)):
        graph, _ = _run_pipeline(str(path), (str(budget),))
        assert validate_memory_plan(graph).feasible is feasible
        if not feasible:
            assert [(issue.required_bytes, issue.reason) for issue in validate_memory_plan(graph).issues] == [
                (96, "smallest tile")]
            assert graph.stages[0].tile_plan.warnings


def test_the_smallest_band_fits_one_budget_up(tmp_path):
    """The reported requirement is sufficient: that budget compiles."""
    path = tmp_path / "gap.onnx"
    _global_pool(path)
    ag, _ = _run_pipeline(str(path), (str(64 * 4 + 8 * 64 * 4 + 64 * 4),))
    assert validate_memory_plan(ag).feasible
    assert compute_findings(ag).blocking_stages == []
