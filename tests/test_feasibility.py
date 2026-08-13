"""Fail-closed validation for deployment memory budgets."""

from dataclasses import replace

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline, cli
from tigris.emitters.binary.writer import emit_binary_bytes


def _write_oversized_height_op_model(tmp_path, op_type):
    if op_type == "GlobalAveragePool":
        input_shape = [1, 8, 64, 64]
        output_shape = [1, 8, 1, 1]
        node = helper.make_node(
            op_type, ["input"], ["output"], name="global_average_pool"
        )
        initializers = []
    else:
        input_shape = [1, 4, 8, 8]
        output_shape = [1, 4, 16, 16]
        scales = numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"
        )
        node = helper.make_node(
            "Resize",
            ["input", "", "scales"],
            ["output"],
            name="resize",
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        initializers = [scales]

    graph = helper.make_graph(
        [node],
        f"oversized_{op_type}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / f"{op_type}.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("op_type", ["GlobalAveragePool", "Resize"])
def test_height_changing_or_reducing_stage_cannot_claim_height_tiling(
    tmp_path, op_type
):
    model_path = _write_oversized_height_op_model(tmp_path, op_type)

    graph, _ = _run_pipeline(str(model_path), ("4K",))
    tile_plans = [
        stage.tile_plan for stage in graph.stages if stage.tile_plan is not None
    ]

    assert tile_plans
    assert all(not tile_plan.tileable for tile_plan in tile_plans)
    assert any(
        op_type in untileable
        for tile_plan in tile_plans
        for untileable in tile_plan.untileable_ops
    )
    assert not validate_memory_plan(graph).feasible


@pytest.mark.parametrize("op_type", ["GlobalAveragePool", "Resize"])
def test_compile_refuses_unsafe_height_tiling_but_keeps_untiled_support(
    tmp_path, op_type
):
    model_path = _write_oversized_height_op_model(tmp_path, op_type)
    rejected = tmp_path / f"{op_type}-tight.tgrs"
    accepted = tmp_path / f"{op_type}-roomy.tgrs"

    tight = CliRunner().invoke(
        cli,
        ["compile", str(model_path), "-m", "4K", "-o", str(rejected)],
    )
    roomy = CliRunner().invoke(
        cli,
        ["compile", str(model_path), "-m", "256K", "-o", str(accepted)],
    )

    assert tight.exit_code != 0
    assert "Cannot compile an infeasible memory plan" in tight.output
    assert not rejected.exists()
    assert roomy.exit_code == 0, roomy.output
    assert accepted.exists()


def test_minimum_tile_that_exceeds_budget_is_infeasible(conv_relu_chain_path):
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("1K",))

    validation = validate_memory_plan(graph)

    assert not validation.feasible
    assert validation.scheduled_peak_bytes > graph.mem_budget
    assert any(issue.reason == "minimum spatial tile" for issue in validation.issues)


def test_findings_never_pass_when_scheduled_peak_exceeds_budget(conv_relu_chain_path):
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("1K",))

    findings = compute_findings(graph)

    assert findings.scheduled_peak_bytes > findings.budget
    assert findings.verdict == "needs_work"
    assert findings.feasibility_errors


def test_compile_refuses_infeasible_plan_without_creating_output(
    conv_relu_chain_path, tmp_path
):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(conv_relu_chain_path), "-m", "1K", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "Cannot compile an infeasible memory plan" in result.output
    assert "requires" in result.output
    assert not output.exists()


@pytest.mark.parametrize("budget", ["0", "-1K"])
def test_compile_rejects_nonpositive_budget(linear_3op_path, tmp_path, budget):
    output = tmp_path / "should-not-exist.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(linear_3op_path), "-m", budget, "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "Fast-memory budget must be greater than zero" in result.output
    assert not output.exists()


def test_compile_rejects_budget_above_plan_format_limit(
    linear_3op_path, tmp_path
):
    output = tmp_path / "existing.tgrs"
    output.write_bytes(b"sentinel")

    result = CliRunner().invoke(
        cli,
        [
            "compile",
            str(linear_3op_path),
            "-m",
            "4096M",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "uint32 plan-format limit" in result.output
    assert output.read_bytes() == b"sentinel"


def test_writer_defensively_rejects_infeasible_graph(conv_relu_chain_path):
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("1K",))

    with pytest.raises(ValueError, match="Cannot emit an infeasible memory plan"):
        emit_binary_bytes(graph)


def test_writer_defensively_rejects_budget_above_plan_format_limit(
    linear_3op_path,
):
    graph, _ = _run_pipeline(str(linear_3op_path), ("4K",))
    graph.budget = replace(graph.budget, fast=0x1_0000_0000)

    with pytest.raises(ValueError, match="uint32 plan-format limit"):
        emit_binary_bytes(graph)


def test_feasible_plan_still_compiles(conv_relu_chain_path, tmp_path):
    output = tmp_path / "feasible.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(conv_relu_chain_path), "-m", "64K", "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_analyze_displays_failing_verdict(conv_relu_chain_path):
    result = CliRunner().invoke(
        cli,
        ["analyze", str(conv_relu_chain_path), "-m", "1K"],
    )

    assert result.exit_code == 0
    assert "FAIL" in result.output
    assert "Infeasible" in result.output


def test_run_pipeline_records_slow_tier(conv_relu_chain_path):
    ag, total = _run_pipeline(str(conv_relu_chain_path), ("64K", "8M"))
    assert ag.budget.slow == 8 * 1024 * 1024
    assert ag.budget.fast + ag.budget.fast_reserve == total


def test_run_pipeline_no_slow_tier_is_zero(conv_relu_chain_path):
    ag, _ = _run_pipeline(str(conv_relu_chain_path), ("64K",))
    assert ag.budget.slow == 0


def test_compile_rejects_nonpositive_slow_tier(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "out.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K", "-m", "0",
               "-o", str(output)])
    assert result.exit_code != 0
    assert "Slow-memory budget must be greater than zero" in result.output
    assert not output.exists()


def test_compile_refuses_slow_overflow_without_output(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "should-not-exist.tgrs"
    # 16K is below the naive peak (52.6K) so stages need tiling and are
    # still fast-feasible; 1K is a slow tier the stage in+out cannot fit.
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "16K", "-m", "1K",
               "-o", str(output)])
    assert result.exit_code != 0
    assert "overflows slow memory" in result.output
    assert not output.exists()


def test_slow_within_budget_compiles(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "out.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "16K", "-m", "64M",
               "-o", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_compile_refuses_flash_overflow_without_output(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "should-not-exist.tgrs"
    # -f far below any real plan size.
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K",
               "-f", "1", "-o", str(output)])
    assert result.exit_code != 0
    assert "exceeds the flash budget" in result.output
    assert not output.exists()


def test_compile_refuses_flash_overflow_compressed(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "should-not-exist.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K",
               "--compress", "lz4", "-f", "1", "-o", str(output)])
    assert result.exit_code != 0
    assert "exceeds the flash budget" in result.output
    assert not output.exists()


def test_analyze_allows_nonpositive_slow_tier(conv_relu_chain_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    result = CliRunner().invoke(
        cli, ["analyze", str(conv_relu_chain_path), "-m", "256K", "-m", "0"])
    assert result.exit_code == 0
    assert "Slow-memory budget must be greater than zero" not in result.output


def test_flash_within_budget_compiles(conv_relu_chain_path, tmp_path):
    from click.testing import CliRunner
    from tigris.cli import cli
    output = tmp_path / "out.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K",
               "-f", "16M", "-o", str(output)])
    assert result.exit_code == 0
    assert output.exists()
