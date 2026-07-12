"""Fail-closed validation for deployment memory budgets."""

import pytest
from click.testing import CliRunner

from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline, cli
from tigris.emitters.binary.writer import emit_binary_bytes


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
    graph.mem_budget = 0x1_0000_0000

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
