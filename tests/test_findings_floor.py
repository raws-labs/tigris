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


def test_a_pool_whose_smallest_band_is_too_large_reports_that_band(tmp_path):
    """64 channels over 8x8: the smallest band is one row (8 x 64 floats) plus the
    per-channel sums and the output, 2560 bytes, not the 16 KiB untiled input."""
    path = tmp_path / "gap.onnx"
    _global_pool(path)
    ag, _ = _run_pipeline(str(path), ("2K",))

    band = 64 * 4 + 8 * 64 * 4 + 64 * 4
    issues = validate_memory_plan(ag).issues
    assert [(issue.required_bytes, issue.reason) for issue in issues] == [(band, "smallest tile")]

    findings = compute_findings(ag)
    assert findings.stages_untileable == 0
    assert findings.min_fast_for_partition == band
    assert [(stage.required_bytes, stage.reason) for stage in findings.blocking_stages] == [
        (band, "smallest tile")]


def test_the_smallest_band_fits_one_budget_up(tmp_path):
    """The reported requirement is sufficient: that budget compiles."""
    path = tmp_path / "gap.onnx"
    _global_pool(path)
    ag, _ = _run_pipeline(str(path), (str(64 * 4 + 8 * 64 * 4 + 64 * 4),))
    assert validate_memory_plan(ag).feasible
    assert compute_findings(ag).blocking_stages == []
