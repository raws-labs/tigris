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


def test_a_global_reduction_tiles_along_its_input(tmp_path):
    """It has no output axis to tile, so the band count comes from the input."""
    model_path = _write_oversized_height_op_model(
        tmp_path, "GlobalAveragePool")

    graph, _ = _run_pipeline(str(model_path), ("4K",))
    tile_plans = [
        stage.tile_plan for stage in graph.stages if stage.tile_plan is not None
    ]

    assert tile_plans
    assert all(tile_plan.tileable for tile_plan in tile_plans)
    assert all(tile_plan.num_tiles > 1 for tile_plan in tile_plans)
    assert all(tile_plan.original_height == 64 for tile_plan in tile_plans)
    assert validate_memory_plan(graph).feasible


def test_a_global_reduction_cuts_a_row_that_does_not_fit(tmp_path):
    """A spatial interval can end inside a row without changing the sum."""
    model_path = _write_oversized_height_op_model(
        tmp_path, "GlobalAveragePool")

    graph, _ = _run_pipeline(str(model_path), ("1K",))
    tile_plans = [
        stage.tile_plan for stage in graph.stages if stage.tile_plan is not None
    ]

    assert tile_plans
    assert all(tile_plan.tileable for tile_plan in tile_plans)
    assert all(0 < tile_plan.input_spatial_interval < 64 for tile_plan in tile_plans)
    assert validate_memory_plan(graph).feasible


def test_a_resample_compiles_at_a_budget_below_its_output(tmp_path):
    """An upsample has more output rows than input rows.

    The tile loop cuts its output into bands and divides to find the source
    band each one reads. The model upsamples 8x8 to 16x16 over 4 channels, so
    its output alone is 4 KiB and the two tensors together are 5 KiB.
    Compiling it to a 4 KiB arena only works if the stage is cut.
    """
    model_path = _write_oversized_height_op_model(tmp_path, "Resize")
    plan = tmp_path / "resize-tight.tgrs"

    result = CliRunner().invoke(
        cli,
        ["compile", str(model_path), "-m", "4K", "-o", str(plan)],
    )

    assert result.exit_code == 0, result.output
    assert plan.exists()


def test_minimum_tile_that_exceeds_budget_is_infeasible(conv_relu_chain_path):
    # 300 bytes is below the 1x1 2D core for either stage (halo 2x2), so a
    # 2D tile solve is attempted once the 1D single-row tile also overflows
    # and reports its own distinct "minimum 2D tile" reason, not the generic
    # 1D "minimum spatial tile" fallback.
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("300",))

    validation = validate_memory_plan(graph)

    assert not validation.feasible
    assert validation.scheduled_peak_bytes > graph.mem_budget
    assert any(issue.reason == "smallest 2D tile" for issue in validation.issues)


def test_findings_never_pass_when_scheduled_peak_exceeds_budget(conv_relu_chain_path):
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("300",))

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
        # 300 bytes stays below the 1x1 2D core too; see
        # test_minimum_tile_that_exceeds_budget_is_infeasible.
        ["compile", str(conv_relu_chain_path), "-m", "300", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "does not fit" in result.output and "no plan written" in result.output
    assert "smallest 2D tile" in result.output
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
    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("300",))

    with pytest.raises(ValueError, match="Cannot emit an infeasible memory plan"):
        emit_binary_bytes(graph)


def test_writer_defensively_rejects_budget_above_plan_format_limit(
    linear_3op_path,
):
    graph, _ = _run_pipeline(str(linear_3op_path), ("4K",))
    graph.budget = replace(graph.budget, fast=0x1_0000_0000)

    with pytest.raises(ValueError, match="uint32 plan-format limit"):
        emit_binary_bytes(graph)


@pytest.mark.parametrize("budget", [
    ["-m", "64K"],
    ["-m", "16K", "-m", "64M"],
    ["-m", "64K", "-f", "16M"],
], ids=["fast", "slow", "flash"])
def test_plan_within_budget_compiles(conv_relu_chain_path, tmp_path, budget):
    output = tmp_path / "out.tgrs"

    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), *budget, "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_analyze_displays_failing_verdict(conv_relu_chain_path):
    result = CliRunner().invoke(
        cli,
        ["analyze", str(conv_relu_chain_path), "-m", "300"],
    )

    assert result.exit_code == 0
    assert "does not fit 300 B fast memory" in result.output
    # The blocking stage is named once, with what it needs.
    assert result.output.count("smallest 2D tile") == 1
    assert "none at this budget" in result.output


def test_analyze_prints_the_trace(conv_relu_chain_path):
    trace = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "64K", "--trace"])
    assert trace.exit_code == 0, trace.output
    assert "traced on runtime" in trace.output and "their bytes equal the runtime's own counters" in trace.output
    assert CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "--json",
                                    "--trace"]).exit_code != 0


@pytest.mark.parametrize("command", ["plan", "simulate"])
def test_plan_and_simulate_are_removed(conv_relu_chain_path, command):
    result = CliRunner().invoke(cli, [command, str(conv_relu_chain_path), "-m", "64K"])
    assert result.exit_code == 2
    assert f"No such command '{command}'" in result.output


def test_run_pipeline_records_slow_tier(conv_relu_chain_path):
    ag, total = _run_pipeline(str(conv_relu_chain_path), ("64K", "8M"))
    assert ag.budget.slow == 8 * 1024 * 1024
    assert ag.budget.fast + ag.budget.fast_reserve == total


def test_run_pipeline_no_slow_tier_is_zero(conv_relu_chain_path):
    ag, _ = _run_pipeline(str(conv_relu_chain_path), ("64K",))
    assert ag.budget.slow == 0


def test_compile_refuses_slow_overflow_without_output(conv_relu_chain_path, tmp_path):
    output = tmp_path / "should-not-exist.tgrs"
    # 16K is below the naive peak (52.6K) so stages need tiling and are
    # still fast-feasible; 1K is a slow tier the stage in+out cannot fit.
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "16K", "-m", "1K",
               "-o", str(output)])
    assert result.exit_code != 0
    assert "slow memory" in result.output and "needed" in result.output and "given" in result.output
    assert not output.exists()


@pytest.mark.parametrize("extra", [[], ["--compress", "lz4"]], ids=["raw", "lz4"])
def test_compile_refuses_flash_overflow_without_output(conv_relu_chain_path, tmp_path, extra):
    output = tmp_path / "should-not-exist.tgrs"
    # -f far below any real plan size.
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K", *extra,
               "-f", "1", "-o", str(output)])
    assert result.exit_code != 0
    assert "exceeds it by" in result.output and "no plan written" in result.output
    assert not output.exists()


def test_a_plan_beyond_the_default_build_is_written_and_names_the_build(tmp_path):
    """Nineteen stage outputs exceed the runtime's default of 16, a build
    option: the plan is written and every report names the -D it needs."""
    import json

    outputs = [f"part{k}" for k in range(19)]
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Split", ["input"], outputs, axis=1, num_outputs=19)], "parts",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 19])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]) for name in outputs]),
        opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    source = tmp_path / "parts.onnx"
    onnx.save(model, source)
    plan = tmp_path / "parts.tgrs"
    expected = "runtime build   -DTIGRIS_MAX_STAGE_OUTPUTS=19"
    for args in (["compile", str(source), "-m", "16K", "-o", str(plan)],
                 ["analyze", str(source), "-m", "16K"], ["inspect", str(plan)]):
        result = CliRunner().invoke(cli, args, env={"NO_COLOR": "1"})
        assert result.exit_code == 0, result.output
        assert expected in result.output
    report = json.loads(CliRunner().invoke(cli, ["analyze", str(source), "-m", "16K", "--json"]).output)
    assert report["runtime_build"] == [{"name": "TIGRIS_MAX_STAGE_OUTPUTS", "needed": 19, "default": 16}]


def test_a_control_flow_plan_names_the_tensor_count_of_the_whole_plan(tmp_path, loop_past_512_tensors_path):
    """The plan's tensor table also holds the loop and branch bodies, so
    compile names the tensor build limit with that count and no other."""
    import re

    from tigris.inspection import inspect_file

    source, model_tensors = loop_past_512_tensors_path
    plan = tmp_path / "loop.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "256K", "-o", str(plan)],
                                env={"NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    tensors = len(inspect_file(plan)["plan"]["tensors"])
    assert 512 < model_tensors < tensors
    assert re.findall(r"TIGRIS_MAX_TENSORS=(\d+)", result.output) == [str(tensors)]
    assert f"runtime build   -DTIGRIS_MAX_TENSORS={tensors}" in result.output


@pytest.mark.parametrize("budget", ["128", "32M"])
def test_int8_global_pool_refuses_an_overflowing_sum(tmp_path, budget):
    from tigris.analysis.validation import validate_operator_support
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.runtime import Session, RuntimeError as HostError
    from scripts.crossrepo_contract import _model, _qdq, _scalars

    count = ((1 << 31) - 1) // 255 + 1
    model = _model("large_mean", [*_qdq("input", "x_s", "x_z", "x"),
                  helper.make_node("GlobalAveragePool", ["x"], ["raw"]),
                  *_qdq("raw", "y_s", "y_z", "output")],
                  [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, count])],
                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])],
                  _scalars(x=(0.125, -128), y=(0.25, 9)))
    source = tmp_path / "large.onnx"
    onnx.save(model, source)
    graph, _ = _run_pipeline(str(source), (budget, "32M"))
    assert not validate_operator_support(graph).supported
    path = tmp_path / "large.tgrs"
    with pytest.raises(ValueError):
        emit_binary_bytes(graph)
    from tigris.emitters.binary.writer import _emit
    path.write_bytes(_emit(graph, compress=None, validate=False))
    assert read_binary_plan(path.read_bytes())["version"] == 11
    with pytest.raises(HostError):
        Session(path)
