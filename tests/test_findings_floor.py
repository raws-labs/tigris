"""The reported minimum fast arena is what compile would require, not a stage's untiled peak."""

import json

import numpy as np
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline, cli

from conftest import save_onnx


def _global_pool(path, channels=64, side=8):
    save_onnx(path.parent, "gap",
              [helper.make_node("GlobalAveragePool", ["input"], ["output"], name="gap")],
              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 1, 1])],
              ir_version=8, filename=path.name)


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


def _conv(source, weight, result):
    return helper.make_node("Conv", [source, weight], [result], kernel_shape=[3, 3], pads=[1, 1, 1, 1])


def _conv_model(path, branched=False):
    """A 3x3 Conv on 32x32x32; branched, an If whose branches run a 256-channel Conv on its result."""
    def act(name, channels=32):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, 32, 32])

    def branch(name):
        return helper.make_graph([_conv("c", "w2", f"{name}_y")], name, [], [act(f"{name}_y", 256)])

    weights = [numpy_helper.from_array(np.ones((32, 32, 3, 3), np.float32), "w")]
    nodes, inputs, output = [_conv("x", "w", "c")], [act("x")], act("c")
    if branched:
        weights.append(numpy_helper.from_array(np.ones((256, 32, 3, 3), np.float32), "w2"))
        nodes.append(helper.make_node("If", ["cond"], ["y"], then_branch=branch("then"),
                                      else_branch=branch("else")))
        inputs.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]))
        output = act("y", 256)
    save_onnx(path.parent, "conv", nodes, inputs, [output], weights, ir_version=8,
              filename=path.name)


def _tiling_counts(path, budget):
    report = json.loads(CliRunner().invoke(cli, ["analyze", str(path), "-m", budget, "--json"]).output)
    return tuple(report["fast"][key] for key in
                 ("stages_needing_tiling", "stages_tileable", "stages_untileable"))


@pytest.mark.parametrize("build, budget, solver_tileable, counted", [
    (_global_pool, "95", False, 0), (_global_pool, "96", True, 1),
    (_conv_model, "600", True, 0), (_conv_model, "20K", True, 1),
], ids=["pool_refused", "pool_fits", "conv_over_budget", "conv_fits"])
def test_only_a_tile_plan_the_validator_accepts_counts_as_tileable(
        tmp_path, build, budget, solver_tileable, counted):
    """Over budget, the pool solver refuses its smallest tile while the row and 2D solvers
    keep theirs as tileable with a warning; the stage is in neither count either way."""
    path = tmp_path / "model.onnx"
    build(path)
    graph, _ = _run_pipeline(str(path), (budget,))
    tile = graph.stages[0].tile_plan
    assert tile.tileable is solver_tileable and not tile.untileable_ops
    assert validate_memory_plan(graph).feasible is bool(counted)
    assert _tiling_counts(path, budget) == (1, counted, 0)


def test_a_blocked_subgraph_stage_does_not_hide_a_main_graph_stage_with_its_id(tmp_path):
    path = tmp_path / "branched.onnx"
    _conv_model(path, branched=True)
    graph, _ = _run_pipeline(str(path), ("2K",))
    issues = validate_memory_plan(graph).issues
    assert [(issue.stage_id, issue.subgraph) for issue in issues] == [(0, 1), (0, 2)]
    assert graph.stages[0].tile_plan.tileable
    assert _tiling_counts(path, "2K") == (2, 1, 0)
