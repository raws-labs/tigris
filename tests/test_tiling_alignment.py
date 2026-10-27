"""A height stripe is sized by its band buffers as the runtime aligns them."""

import pytest
from onnx import TensorProto, helper

from tigris.analysis.validation import validate_memory_plan
from tigris.pipeline import plan_model

from conftest import save_onnx


def _mul(tmp_path):
    shape = [1, 4, 5, 5]
    path = save_onnx(tmp_path, "mul", [helper.make_node("Mul", ["x", "y"], ["z"])],
                     [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
                      helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
                     [helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)])
    return str(path)


def test_a_one_row_band_counts_each_buffer_aligned(tmp_path):
    # Three 80-byte rows, each aligned to 96 bytes: 288, not 240.
    ag = plan_model(_mul(tmp_path), ("288",))
    plan = ag.stages[0].tile_plan
    assert plan.tileable and plan.tile_height == 1
    assert plan.tiled_peak_bytes == 288
    assert not plan.warnings


def test_a_band_that_does_not_fit_aligned_narrows_to_a_2d_tile(tmp_path):
    # One aligned row needs 288 bytes, so a 256-byte budget takes a 2D tile.
    ag = plan_model(_mul(tmp_path), ("256",))
    plan = ag.stages[0].tile_plan
    assert plan.tileable and plan.tile_width > 0
    assert plan.tiled_peak_bytes <= 256


@pytest.mark.parametrize("budget,feasible", [(191, False), (192, True), (256, True)])
def test_transpose_band_counts_unchanged_trailing_block(tmp_path, budget, feasible):
    shape = [1, 2, 2, 3, 2, 3]
    perm = [0, 1, 3, 2, 4, 5]
    path = save_onnx(tmp_path, "transpose",
                     [helper.make_node("Transpose", ["x"], ["y"], perm=perm)],
                     [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [shape[i] for i in perm])])
    graph = plan_model(str(path), (str(budget),))
    assert validate_memory_plan(graph).feasible == feasible
    plan = graph.stages[0].tile_plan
    if feasible:
        # Two buffers of batch(2) x rows(2) x band(1) x block(6) x float(4).
        assert plan.tile_height == 1
        assert plan.tiled_peak_bytes == 192


@pytest.mark.parametrize("reserve", [0, 64])
def test_memory_contract_rejects_peak_above_schedule_in_roomy_arena(reserve):
    from scripts.crossrepo_contract import _assert_memory_contract

    plan = {"budget": 256, "scheduled_peak": 192, "_compiler_scheduled_peak": 192,
            "_compiler_slow_peak": 512, "_compiler_slow_budget": 0}
    report = (f"TIGRIS_CONTRACT_MEMORY budget=256 activation_limit=256 "
              f"reserve={reserve} required={192 + reserve} allocated={256 + reserve} "
              f"peak={240 + reserve} slow_peak=128\n")
    with pytest.raises(AssertionError, match="runtime peak .* exceeds compiler"):
        _assert_memory_contract("roomy", plan, report, full_budget=True)


def test_memory_contract_holds_the_runtime_to_the_stated_peak():
    from scripts.crossrepo_contract import _assert_memory_contract

    report = ("TIGRIS_CONTRACT_MEMORY budget=256 activation_limit=192 reserve=0 "
              "required={required} allocated=192 peak=192 slow_peak=0\n")
    plan = {"budget": 256, "scheduled_peak": 160, "_compiler_scheduled_peak": 192,
            "_compiler_slow_peak": 0, "_compiler_slow_budget": 0}
    with pytest.raises(AssertionError, match="plan states scheduled peak 160"):
        _assert_memory_contract("stated", plan, report.format(required=192))
    plan["scheduled_peak"] = 192
    with pytest.raises(AssertionError, match=r"required arena 256 != activations \+ reserve"):
        _assert_memory_contract("stated", plan, report.format(required=256))
    _assert_memory_contract("stated", plan, report.format(required=192))
    plan["scheduled_peak"] = None
    _assert_memory_contract("older", plan, report.format(required=256))
