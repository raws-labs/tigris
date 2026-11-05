"""Fail-closed validation for deployment memory budgets."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris import SCHEMA_VERSION
from tigris.analysis.findings import compute_findings
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import cli
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.pipeline import plan_model

from conftest import save_onnx


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

    return save_onnx(tmp_path, f"oversized_{op_type}", [node],
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
                     initializers, ir_version=8, check=True, filename=f'{op_type}.onnx')


def test_a_global_reduction_tiles_along_its_input(tmp_path):
    """It has no output axis to tile, so the band count comes from the input."""
    model_path = _write_oversized_height_op_model(
        tmp_path, "GlobalAveragePool")

    graph = plan_model(str(model_path), ("4K",))
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

    graph = plan_model(str(model_path), ("1K",))
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
    graph = plan_model(str(conv_relu_chain_path), ("300",))

    validation = validate_memory_plan(graph)

    assert not validation.feasible
    assert validation.scheduled_peak_bytes > graph.mem_budget
    assert any(issue.reason == "smallest 2D tile" for issue in validation.issues)


def test_findings_never_pass_when_scheduled_peak_exceeds_budget(conv_relu_chain_path):
    graph = plan_model(str(conv_relu_chain_path), ("300",))

    findings = compute_findings(graph)

    assert findings.scheduled_peak_bytes > graph.mem_budget
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
    graph = plan_model(str(conv_relu_chain_path), ("300",))

    with pytest.raises(ValueError, match="Cannot emit an infeasible memory plan"):
        emit_binary_bytes(graph)


def test_writer_defensively_rejects_budget_above_plan_format_limit(
    linear_3op_path,
):
    graph = plan_model(str(linear_3op_path), ("4K",))
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


@pytest.mark.parametrize("flags, xip, names", [
    ([], False, False),
    (["--keep-names"], False, True),
    (["--xip"], True, False),
], ids=["plain", "names", "xip"])
def test_compile_writes_the_writer_bytes_for_its_flags(conv_relu_chain_path, tmp_path, flags, xip, names):
    output = tmp_path / "out.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "16K", *flags, "-o", str(output)])
    assert result.exit_code == 0, result.output
    graph = plan_model(str(conv_relu_chain_path), ("16K",))
    assert output.read_bytes() == emit_binary_bytes(graph, xip=xip, names=names)


def test_a_graph_without_activations_is_gated_like_any_other(tmp_path):
    """A constant output leaves no tensor with a lifetime; findings still carry the gate."""
    value = numpy_helper.from_array(np.ones((1, 4), np.float32), "v")
    model = helper.make_model(
        helper.make_graph([helper.make_node("Constant", [], ["output"], value=value)], "const_only", [],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])]),
        opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "const_only.onnx"
    onnx.save(model, path)
    graph = plan_model(str(path), ("16K",))
    assert not graph.lifetimes
    findings = compute_findings(graph)
    assert findings.verdict == "needs_work"
    assert not findings.gate.usable and not findings.gate.passes
    output = tmp_path / "should-not-exist.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "16K", "-o", str(output)])
    assert result.exit_code == 1
    assert "output 'output' is not computed by any operator" in result.output
    assert "Error: no plan written" in result.output
    assert not output.exists()


def test_compile_reports_the_writer_refusal_of_a_plan_the_gate_passes(
    conv_relu_chain_path, tmp_path, monkeypatch
):
    import tigris.emitters.binary.writer

    def refuse(*args, **kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(tigris.emitters.binary.writer, "emit_binary_bytes", refuse)
    output = tmp_path / "should-not-exist.tgrs"
    result = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "16K", "-o", str(output)])
    assert result.exit_code == 1
    assert "Error: no plan written: refused" in result.output
    assert not output.exists()


def test_analyze_displays_failing_verdict(conv_relu_chain_path):
    result = CliRunner().invoke(
        cli,
        ["analyze", str(conv_relu_chain_path), "-m", "300"],
    )

    assert result.exit_code == 1
    assert "does not fit 300 B fast memory" in result.output
    # The blocking stage is named once, with what it needs.
    assert result.output.count("smallest 2D tile") == 1
    assert "none at this budget" in result.output


@pytest.mark.usefixtures("require_host_library")
def test_analyze_prints_the_trace(conv_relu_chain_path):
    trace = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "64K", "--trace"])
    assert trace.exit_code == 0, trace.output
    assert "traced on runtime" in trace.output and "their bytes equal the runtime's own counters" in trace.output
    assert CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "--json",
                                    "--trace"]).exit_code != 0


@pytest.mark.parametrize("budget, rows", [
    ("64K", ["0 0-1 52.62 KiB 1 1 untiled"]),
    ("16K", ["0 0 40.12 KiB 1 1 chain of 2, tile height 5", "1 1 52.62 KiB 1 1 in chain 0"]),
    ("4K", ["0 0 40.12 KiB 1 1 16 tiles of 9 x 9, halo 2", "1 1 52.62 KiB 1 1 20 tiles of 6 x 8, halo 2"]),
])
def test_analyze_verbose_lists_the_stages(conv_relu_chain_path, budget, rows):
    plain = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", budget])
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", budget, "-v"])
    assert plain.exit_code == 0 and result.exit_code == 0, result.output
    assert result.output.startswith(plain.output)
    table = [" ".join(line.split()) for line in result.output[len(plain.output):].splitlines() if line.strip()]
    assert table == ["stages", "stage ops untiled peak in out tiling", *rows]


@pytest.mark.parametrize("command", ["plan", "simulate"])
def test_plan_and_simulate_are_removed(conv_relu_chain_path, command):
    result = CliRunner().invoke(cli, [command, str(conv_relu_chain_path), "-m", "64K"])
    assert result.exit_code == 2
    assert f"No such command '{command}'" in result.output


def test_plan_model_records_slow_tier(conv_relu_chain_path):
    ag = plan_model(str(conv_relu_chain_path), ("64K", "8M"))
    assert ag.budget.slow == 8 * 1024 * 1024
    assert ag.budget.fast + ag.budget.fast_reserve == 64 * 1024


def test_plan_model_no_slow_tier_is_zero(conv_relu_chain_path):
    ag = plan_model(str(conv_relu_chain_path), ("64K",))
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
    source = save_onnx(tmp_path, "parts",
                       [helper.make_node("Split", ["input"], outputs, axis=1, num_outputs=19)],
                       [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 19])],
                       [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]) for name in outputs],
                       opset=18, ir_version=8)
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
@pytest.mark.usefixtures("require_host_library")
def test_int8_global_pool_refuses_an_overflowing_sum(tmp_path, budget):
    from tigris.analysis.validation import validate_operator_support
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.runtime import Session, RuntimeError as HostError
    from scripts.contract_cases.common import _model, _qdq, _scalars

    count = ((1 << 31) - 1) // 255 + 1
    model = _model("large_mean", [*_qdq("input", "x_s", "x_z", "x"),
                  helper.make_node("GlobalAveragePool", ["x"], ["raw"]),
                  *_qdq("raw", "y_s", "y_z", "output")],
                  [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, count])],
                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])],
                  _scalars(x=(0.125, -128), y=(0.25, 9)))
    source = tmp_path / "large.onnx"
    onnx.save(model, source)
    graph = plan_model(str(source), (budget, "32M"))
    assert not validate_operator_support(graph).supported
    path = tmp_path / "large.tgrs"
    with pytest.raises(ValueError):
        emit_binary_bytes(graph)
    from tigris.emitters.binary.writer import _emit
    path.write_bytes(_emit(graph, compress=None, validate=False))
    assert read_binary_plan(path.read_bytes())["version"] == SCHEMA_VERSION
    with pytest.raises(HostError):
        Session(path)


def test_analyze_lists_five_blocking_stages_then_counts_the_rest():
    model = Path(__file__).parent / "fixtures" / "tflite" / "vww_96_int8.tflite"
    result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "1K"])
    assert result.exit_code == 1
    assert "  and 2 more; --json lists all" in result.output


def test_a_stage_that_runs_untiled_names_its_operator_in_full(tmp_path):
    w = numpy_helper.from_array(np.ones((10, 256), np.float32), "w")
    nodes = [helper.make_node("Gemm", ["input", "w"], ["output"], name="fc (head)", transB=1)]
    path = save_onnx(tmp_path, "fc_head", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
                     [w], ir_version=8)
    graph = plan_model(str(path), ("256",))

    assert [issue.reason for issue in validate_memory_plan(graph).issues] == ["runs untiled: fc (head)"]
    assert compute_findings(graph).untileable_op_types == ["Gemm"]


_TFLITE = Path(__file__).parent / "fixtures" / "tflite"


def _wide_if(tmp_path):
    """An If whose then branch widens x [1, 4] to [256, 4] and sums it back,
    so the branch needs more memory than the main graph."""
    def branch(name, wide):
        x, y = f"{name}_x", f"{name}_y"
        nodes = [helper.make_node("Relu", [x], [y])]
        constants = []
        if wide:
            nodes = [helper.make_node("Mul", [x, f"{name}_c"], [f"{name}_wide"]),
                     helper.make_node("Reshape", [f"{name}_wide", f"{name}_s"], [f"{name}_w3"]),
                     helper.make_node("ReduceSum", [f"{name}_w3", f"{name}_ax"], [y], keepdims=0)]
            constants = [numpy_helper.from_array(np.ones((256, 4), np.float32), f"{name}_c"),
                         numpy_helper.from_array(np.asarray([1, 256, 4], np.int64), f"{name}_s"),
                         numpy_helper.from_array(np.asarray([1], np.int64), f"{name}_ax")]
        return helper.make_graph(nodes, name, [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 4])],
                                 [helper.make_tensor_value_info(y, TensorProto.FLOAT, [1, 4])], constants)

    weights = [numpy_helper.from_array(np.asarray(0.0, np.float32), "zero"),
               numpy_helper.from_array(np.asarray([2], np.int64), "axes"),
               numpy_helper.from_array(np.asarray([1, 1, 4], np.int64), "rows")]
    nodes = [helper.make_node("Reshape", ["x", "rows"], ["row"]),
             helper.make_node("ReduceSum", ["row", "axes"], ["sum"], keepdims=0),
             helper.make_node("Greater", ["sum", "zero"], ["positive"]),
             helper.make_node("If", ["positive", "x"], ["y"], domain="tigris",
                              then_branch=branch("then", True), else_branch=branch("else", False))]
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(nodes, "wide_if", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [y], weights, value_info=[y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("tigris", 1)])
    model.ir_version = 9
    path = tmp_path / "wide_if.onnx"
    onnx.save(model, str(path))
    return path


def _cli_model(name, tmp_path, conv_relu_chain_path, qdq_conv_path=None):
    """A model for the analyze and compile parity cases, by short name."""
    if name == "chain":
        return conv_relu_chain_path
    if name == "qdq":
        return qdq_conv_path
    if name == "wide_if":
        return _wide_if(tmp_path)
    if name == "relu":
        return save_onnx(tmp_path, "relu", [helper.make_node("Relu", ["input"], ["output"])],
                         [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 64])],
                         [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 64])],
                         opset=17, ir_version=8)
    if name == "acos":
        return save_onnx(tmp_path, "acos", [helper.make_node("Acos", ["input"], ["output"])],
                         [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                         [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
                         ir_version=8)
    return {"vww": _TFLITE / "vww_96_int8.tflite", "while": _TFLITE / "ops" / "float_while.tflite"}[name]


@pytest.mark.parametrize("model, options, writes", [
    ("chain", ["-m", "64K"], True),
    ("chain", ["-m", "64K", "-f", "1M"], True),
    ("chain", ["-m", "300"], False),
    ("chain", ["-m", "16K", "-m", "64K"], True),
    ("chain", ["-m", "16K", "-m", "8K"], False),
    ("chain", ["-m", "64K", "-f", "1K"], False),
    ("chain", [], True),
    ("acos", ["-m", "4K"], False),
    ("acos", [], False),
    ("vww", ["-m", "32K", "-m", "1M"], True),
    ("vww", ["-m", "1K"], False),
    ("while", ["-m", "1K", "-m", "64"], False),
    ("while", [], True),
    ("wide_if", [], True),
    ("wide_if", ["-m", "64"], False),
    ("wide_if", ["-m", "64K"], True),
    ("chain", ["-m", "65535"], True),
    ("chain", ["-m", "319"], False),
    ("qdq", ["--int8-io"], True),
    ("qdq", ["-m", "100", "--int8-io"], True),
    ("qdq", ["-m", "32", "--int8-io"], False),
    ("relu", ["--input-shape", "input:16x64"], True),
    ("relu", ["-m", "4K", "-m", "2K", "--input-shape", "input:16x64"], False),
    ("relu", ["-m", "8K", "-m", "4K", "--input-shape", "input:16x64"], True),
    ("chain", ["-f", "1M"], True),
    ("chain", ["-f", "1K"], False),
], ids=["fits", "flash_fits", "fast_too_small", "slow_fits", "slow_too_small", "flash_too_small",
        "no_budget", "unsupported", "unsupported_no_budget", "tflite_fits", "tflite_fast_too_small",
        "slow_too_small_at_every_budget", "control_flow_no_budget", "wide_branch_no_budget",
        "wide_branch_fast_too_small", "wide_branch_fits", "unaligned_fits", "unaligned_too_small",
        "int8_io_no_budget", "int8_io_unaligned_fits", "int8_io_fast_too_small",
        "input_shape_no_budget", "input_shape_slow_too_small", "input_shape_slow_fits",
        "flash_fits_no_budget", "flash_too_small_no_budget"])
def test_analyze_exits_1_exactly_when_compile_writes_no_plan(
        tmp_path, conv_relu_chain_path, qdq_conv_path, model, options, writes):
    """Without -m, compile runs at the smallest budget analyze reports, or at
    64M when it reports none. Compile with the other options writes a plan at
    the smallest budget and none at the budget below it."""
    import json

    path = str(_cli_model(model, tmp_path, conv_relu_chain_path, qdq_conv_path))
    text_run = CliRunner().invoke(cli, ["analyze", path, *options])
    json_run = CliRunner().invoke(cli, ["analyze", path, *options, "--json"])
    report = json.loads(json_run.output)
    fit = report["smallest_fit"]
    if fit and fit["budget_bytes"]:
        for fast, fast_writes in ((fit["budget_bytes"], True), (fit["below_bytes"], False)):
            if fast is not None:
                assert _compile_writes(tmp_path, path, _at_fast(options, fast)) == fast_writes, fast
    if "-m" not in options:
        fit = report["smallest_fit"]
        options = ["-m", str(fit["budget_bytes"] if fit and fit["budget_bytes"] else "64M"), *options]
    output = tmp_path / "m.tgrs"
    compiled = CliRunner().invoke(cli, ["compile", path, *options, "-o", str(output)])
    assert output.exists() == writes, compiled.output
    assert [text_run.exit_code, json_run.exit_code, compiled.exit_code] == [0 if writes else 1] * 3, (
        text_run.output)


def _at_fast(options, fast):
    """The options with the fast budget `fast`, the other options kept."""
    if "-m" not in options:
        return ["-m", str(fast), *options]
    at = options.index("-m")
    return [*options[:at + 1], str(fast), *options[at + 2:]]


def _compile_writes(tmp_path, path, options):
    output = tmp_path / "at_budget.tgrs"
    output.unlink(missing_ok=True)
    CliRunner().invoke(cli, ["compile", path, *options, "-o", str(output)])
    return output.exists()


def test_the_smallest_budget_is_where_the_plan_fits_the_flash_budget(tmp_path, conv_relu_chain_path):
    """The plan is larger at the smallest budget that fits memory than at
    64K; with a flash budget between the two, the smallest budget is where
    compile writes a plan, above the one that only fits memory."""
    import json

    path = str(conv_relu_chain_path)

    def smallest(options):
        report = json.loads(CliRunner().invoke(cli, ["analyze", path, *options, "--json"]).output)
        return report["smallest_fit"]

    def plan_size(fast):
        output = tmp_path / f"{fast}.tgrs"
        CliRunner().invoke(cli, ["compile", path, "-m", str(fast), "-o", str(output)])
        return output.stat().st_size

    memory_fit = smallest([])["budget_bytes"]
    large, small = plan_size(memory_fit), plan_size(64 * 1024)
    assert large > small
    flash = ["-f", str(large - 1)]
    fit = smallest(["-m", "64K", *flash])
    assert fit["budget_bytes"] > memory_fit and fit == smallest(flash)
    assert _compile_writes(tmp_path, path, ["-m", str(fit["budget_bytes"]), *flash])
    assert not _compile_writes(tmp_path, path, ["-m", str(fit["below_bytes"]), *flash])
    # The budget below fits memory: only the flash budget refuses it.
    assert _compile_writes(tmp_path, path, ["-m", str(fit["below_bytes"])])
    result = CliRunner().invoke(cli, ["analyze", path, "-m", str(memory_fit), *flash])
    assert result.exit_code == 1


def _row(output, label):
    return " ".join(next(line for line in output.splitlines() if line.strip().startswith(label)).split())


@pytest.mark.parametrize("model, pools, rest", [
    ("chain", ["64K"], []),
    ("chain", ["300"], []),
    ("chain", ["16K", "8K"], []),
    ("chain", [], []),
    ("chain", ["1000"], []),
    ("wide_if", ["64"], []),
    ("qdq", ["100"], ["--int8-io"]),
    ("relu", ["8K", "4K"], ["--input-shape", "input:16x64"]),
], ids=["fits", "fast_too_small", "slow_too_small", "no_budget", "unaligned", "wide_branch",
        "int8_io", "input_shape"])
def test_analyze_reports_the_smallest_budget_and_compiles_both_ends(
        tmp_path, conv_relu_chain_path, qdq_conv_path, model, pools, rest):
    """The smallest budget fits and the one 32 bytes below does not, with the
    slow pool and the other options given; both are compiled to check it."""
    import json

    from tigris.analysis.memory import TENSOR_ALIGN
    from tigris.pipeline import parse_size
    from tigris.utils import fmt_bytes

    path = str(_cli_model(model, tmp_path, conv_relu_chain_path, qdq_conv_path))
    options = [*(word for pool in pools for word in ("-m", pool)), *rest]
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, *options, "--json"]).output)
    assert report["version"] == 2 and "budgets_tried" not in report
    fit = report["smallest_fit"]
    smallest, below, slow = fit["budget_bytes"], fit["below_bytes"], fit["slow_budget_bytes"]
    assert slow == (parse_size(pools[1]) if len(pools) > 1 else None)
    assert smallest % TENSOR_ALIGN == 0 and below == smallest - TENSOR_ALIGN
    pool = ["-m", str(slow)] if slow else []
    for fast, writes in ((smallest, True), (below, False)):
        output = tmp_path / f"{fast}.tgrs"
        CliRunner().invoke(cli, ["compile", path, "-m", str(fast), *pool, *rest, "-o", str(output)])
        assert output.exists() == writes
    result = CliRunner().invoke(cli, ["analyze", path, *options], env={"NO_COLOR": "1"})
    clause = f"with {fmt_bytes(slow)} slow memory; " if slow else ""
    assert _row(result.output, "smallest budget") == (
        f"smallest budget {fmt_bytes(smallest)} {clause}{fmt_bytes(below)} does not fit")
    assert "fits at" not in result.output


@pytest.mark.parametrize("model, budgets", [
    ("chain", [[], ["-m", "64K"], ["-m", "65535"], ["-m", "1000"], ["-m", "320"]]),
    ("wide_if", [[], ["-m", "64"], ["-m", "64K"], ["-m", "1056"]]),
], ids=["unaligned_budgets", "wide_branch"])
def test_the_smallest_budget_does_not_depend_on_the_budget_given(
        tmp_path, conv_relu_chain_path, model, budgets):
    """Candidates are multiples of TENSOR_ALIGN whatever -m is, and the search
    reaches what a subgraph needs above the main graph's peak."""
    import json

    from tigris.analysis.memory import TENSOR_ALIGN

    path = str(_cli_model(model, tmp_path, conv_relu_chain_path))
    reports = [json.loads(CliRunner().invoke(cli, ["analyze", path, *budget, "--json"]).output)
               for budget in budgets]
    fits = [report["smallest_fit"] for report in reports]
    assert all(fit == fits[0] for fit in fits), fits
    assert fits[0]["budget_bytes"] % TENSOR_ALIGN == 0
    if model == "wide_if":
        assert fits[0]["budget_bytes"] > reports[0]["model"]["unscheduled_peak_bytes"]


def test_analyze_reports_no_smallest_budget_when_the_slow_pool_never_suffices():
    """The loop's result always lives in slow memory, 128 bytes of it."""
    import json

    path = str(_TFLITE / "ops" / "float_while.tflite")
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, "-m", "1K", "-m", "64", "--json"]).output)
    assert report["smallest_fit"] == {"budget_bytes": None, "below_bytes": None, "slow_budget_bytes": 64,
                                      "reason": "memory"}
    result = CliRunner().invoke(cli, ["analyze", path, "-m", "1K", "-m", "64"], env={"NO_COLOR": "1"})
    assert _row(result.output, "smallest budget") == (
        "smallest budget none no fast budget fits with 64 B slow memory")


@pytest.mark.parametrize("pools", [[], ["-m", "16K", "-m", "64K"]], ids=["no_budget", "slow"])
def test_analyze_reports_no_smallest_budget_when_no_plan_fits_the_flash_budget(
        conv_relu_chain_path, pools):
    import json

    path = str(conv_relu_chain_path)
    options = [*pools, "-f", "1K"]
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, *options, "--json"]).output)
    assert report["smallest_fit"] == {"budget_bytes": None, "below_bytes": None,
                                      "slow_budget_bytes": 65536 if pools else None,
                                      "reason": "flash"}
    result = CliRunner().invoke(cli, ["analyze", path, *options], env={"NO_COLOR": "1"})
    assert result.exit_code == 1
    assert _row(result.output, "smallest budget") == (
        "smallest budget none no plan fits the 1.00 KiB flash budget")


@pytest.mark.parametrize("pools", [[], ["-m", "16K", "-m", "64K"]], ids=["no_budget", "slow"])
def test_analyze_reports_no_smallest_budget_when_the_loader_refuses_every_plan(
        conv_relu_chain_path, monkeypatch, pools):
    import json

    import tigris.runtime

    def refuse(data):
        raise tigris.runtime.RuntimeError("refused")

    monkeypatch.setattr(tigris.runtime, "check_plan", refuse)
    path = str(conv_relu_chain_path)
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, *pools, "--json"]).output)
    assert report["smallest_fit"]["budget_bytes"] is None
    assert report["smallest_fit"]["reason"] == "loader"
    result = CliRunner().invoke(cli, ["analyze", path, *pools], env={"NO_COLOR": "1"})
    assert result.exit_code == 1
    assert _row(result.output, "smallest budget") == (
        "smallest budget none the runtime loader refuses the plan at every fast budget")


def test_analyze_reports_no_smallest_budget_when_the_writer_refuses_every_plan(
        conv_relu_chain_path, monkeypatch):
    """A plan the writer refuses at the largest budget is a model that does
    not compile: no smallest_fit, no row."""
    import json

    import tigris.emitters.binary.writer

    def refuse(*args, **kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(tigris.emitters.binary.writer, "emit_binary_bytes", refuse)
    path = str(conv_relu_chain_path)
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, "--json"]).output)
    assert report["smallest_fit"] is None
    result = CliRunner().invoke(cli, ["analyze", path])
    assert result.exit_code == 1 and "smallest budget" not in result.output


def test_a_found_smallest_budget_has_no_reason(conv_relu_chain_path):
    import json

    report = json.loads(CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "--json"]).output)
    assert report["smallest_fit"]["budget_bytes"] and report["smallest_fit"]["reason"] is None


def test_analyze_reports_no_smallest_budget_for_a_model_that_cannot_compile(tmp_path, conv_relu_chain_path):
    import json

    path = str(_cli_model("acos", tmp_path, conv_relu_chain_path))
    report = json.loads(CliRunner().invoke(cli, ["analyze", path, "--json"]).output)
    assert report["smallest_fit"] is None
    assert "smallest budget" not in CliRunner().invoke(cli, ["analyze", path]).output


@pytest.mark.parametrize("slow, detail", [(None, ""), (4 << 20, "with 4.00 MiB slow memory")])
def test_a_smallest_budget_with_nothing_below_names_no_budget_below(slow, detail):
    """Only reached when the lowest candidate fits, so the row is built directly."""
    import click

    from tigris.cli.analyze import smallest_row

    row = smallest_row({"budget_bytes": 32, "below_bytes": None, "slow_budget_bytes": slow,
                        "reason": None}, 0)
    assert [click.unstyle(cell) for cell in row] == ["smallest budget", "32 B", detail]


def test_no_smallest_budget_without_a_slow_budget_leaves_the_row_out():
    """Not expected with an unbounded slow pool; the report still says null."""
    from tigris.cli.analyze import smallest_row

    assert smallest_row({"budget_bytes": None, "below_bytes": None, "slow_budget_bytes": None,
                         "reason": "memory"}, 0) is None
