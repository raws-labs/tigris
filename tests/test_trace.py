"""The execution trace groups and checks runtime events without a host library."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tigris.cli import _run_pipeline, cli
from tigris.cli.trace import check, render, units
from tigris.graph.ir import AnalyzedGraph, OpNode
from tigris.graph.subgraphs import input_dependent_loop

OPS = Path(__file__).parent / "fixtures" / "tflite" / "ops"
# Loops whose condition reads a value that starts from, or is updated from, a model input.
INPUT_DEPENDENT = {"float_while", "float_while_feature_map", "float_while_if_through",
                   "float_while_through", "float_while_two_variables"}


def _event(kind, **fields):
    event = {"kind": kind, "stage": 0, "tensor": None, "op": None, "pool": None, "offset": 0,
             "src_offset": 0, "bytes": 0, "row0": None, "row1": None, "col0": None, "col1": None,
             "fast_used": 0, "slow_used": 0}
    event.update(fields)
    return event


PLAN = {
    "stages": [{"chain_len": 2}, {"chain_len": 2}, {"chain_len": 1}],
    "tensors": [{"name": "input"}, {"name": "middle"}, {"name": "output"}],
    "ops": [{"name": "Conv_0"}, {"name": "Conv_1"}, {"name": "Relu_2"}],
}

EVENTS = [
    _event("stage_begin", path="chain", fast_used=0, slow_used=300),
    _event("tile_begin", op=0, row0=0, row1=4),
    _event("load", tensor=0, pool="fast", offset=0, src_offset=0, bytes=64, row0=0, row1=4, fast_used=64,
           slow_used=300),
    _event("alloc", tensor=1, pool="fast", offset=64, bytes=48, fast_used=112, slow_used=300),
    _event("op", op=0, fast_used=112, slow_used=300),
    _event("spill", tensor=1, pool="slow", offset=300, src_offset=64, bytes=32, row0=0, row1=2,
           fast_used=112, slow_used=332),
    _event("reset", pool="fast", fast_used=0, slow_used=332),
    _event("tile_begin", op=1, row0=2, row1=6),
    _event("load", tensor=0, pool="fast", offset=0, src_offset=32, bytes=64, row0=2, row1=6, fast_used=64,
           slow_used=332),
    _event("spill", tensor=1, pool="slow", offset=332, src_offset=64, bytes=32, row0=2, row1=4,
           fast_used=112, slow_used=364),
    _event("stage_end", fast_used=0, slow_used=364),
    _event("stage_begin", stage=2, path="normal", fast_used=0, slow_used=364),
    _event("load", stage=2, tensor=1, pool="fast", offset=0, src_offset=300, bytes=64, fast_used=64,
           slow_used=364),
    _event("op", stage=2, op=2, fast_used=64, slow_used=364),
    _event("spill", stage=2, tensor=2, pool="slow", offset=364, src_offset=0, bytes=64, fast_used=64,
           slow_used=428),
    _event("stage_end", stage=2, fast_used=0, slow_used=428),
]
COUNTERS = {"load_bytes": 192, "spill_bytes": 128, "weight_bytes": 0, "copy_bytes": 0, "compactions": 0,
            "tiles": 2}


def test_units_group_events_per_stage_with_their_traffic():
    chain, untiled = units(PLAN, EVENTS)
    assert (chain["stage"], chain["kind"], chain["tiles"], chain["read"], chain["written"]) == \
        (0, "chain 2", 2, 128, 64)
    assert (chain["fast_peak"], chain["slow_used"]) == (112, 364)
    assert (untiled["stage"], untiled["kind"], untiled["tiles"], untiled["read"], untiled["written"]) == \
        (2, "untiled", 0, 64, 64)


def test_check_compares_event_sums_with_the_runtime_counters():
    assert check(EVENTS, COUNTERS) == []
    problems = check(EVENTS, {**COUNTERS, "spill_bytes": 100, "tiles": 3})
    assert problems == ["spill events add up to 128 bytes, the runtime counted 100",
                        "2 tile events, the runtime counted 3 tiles"]


def test_render_prints_totals_units_and_with_verbose_every_event(capsys):
    memory = {"fast_capacity_bytes": 128}
    info = {"version": "9.9.9", "input": "zero input", "align": 32}
    render("model.onnx", PLAN, EVENTS, COUNTERS, memory, info, verbose=True)
    out = capsys.readouterr().out
    assert out.startswith("model.onnx   traced on runtime 9.9.9, host reference backend, zero input\n")
    assert "moved       128 B written, 192 B read" in out
    assert "fast peak   112 B of 128 B" in out
    assert "stage 0   chain 2" in out
    assert "    load    input      slow+32   fast+0     64 B   rows 2-5" in out
    assert "    0   chain 2       2   128 B      64 B       112 B       364 B" in out
    assert "16 events; their bytes equal the runtime's own counters." in out


def test_trace_needs_a_budget(conv_relu_chain_path):
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "--trace"])
    assert result.exit_code != 0
    assert "needs a fast-memory budget" in result.output


def test_input_is_only_for_trace(conv_relu_chain_path, tmp_path):
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "64K",
                                      "--input", str(tmp_path / "x.bin")])
    assert result.exit_code != 0
    assert "--input is for --trace" in result.output


def test_analyze_says_when_traffic_is_not_measured(conv_relu_chain_path, monkeypatch):
    import tigris.runtime

    def unavailable():
        raise tigris.runtime.RuntimeError("Bundled host runtime is unavailable")

    monkeypatch.setattr(tigris.runtime, "runtime_info", unavailable)
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "8K"])
    assert result.exit_code == 0, result.output
    assert "not measured: no host runtime in this install" in result.output
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "8K", "--json"])
    assert json.loads(result.output)["slow"]["traffic"] is None


@pytest.mark.parametrize("name", sorted(path.stem for path in OPS.glob("*.tflite")
                                        if {"if", "while"} & set(path.stem.split("_"))))
def test_input_dependent_loops_are_told_from_fixed_ones(name):
    ag, _ = _run_pipeline(str(OPS / f"{name}.tflite"), ("64K",), report_bindings=False)
    assert input_dependent_loop(ag) is (name in INPUT_DEPENDENT)


def _graph(inputs, ops, outputs, subgraphs=()):
    return AnalyzedGraph(ops=ops, model_inputs=inputs, model_outputs=outputs, subgraphs=list(subgraphs))


def _counting(body, ops=(), subgraphs=()):
    """x carried as `v` through a While whose condition runs `ops` and then
    compares the counter `i`, which starts from a constant, with a limit."""
    names = body.model_inputs
    condition = _graph(names, [*ops, OpNode("less", "Less", ["i", "limit"], ["going"])], ["going"], subgraphs)
    loop = OpNode("loop", "While", ["x" if name == "v" else "zero" for name in names],
                  [f"out_{name}" for name in names], {"cond_branch": 0, "body_branch": 1})
    return _graph(["x"], [loop], ["out_v"], [condition, body])


def _up_to(bound):
    """A While that counts from zero to `bound`, and its two subgraphs."""
    condition = _graph(["a", "b"], [OpNode("less", "Less", ["a", "b"], ["going"])], ["going"])
    body = _graph(["a", "b"], [OpNode("step", "Add", ["a", "one"], ["a2"])], ["a2", "b"])
    loop = OpNode("inner", "While", ["zero", bound], ["steps", "bound"], {"cond_branch": 0, "body_branch": 1})
    return loop, [condition, body]


def _counter_steps_by(source):
    return _counting(_graph(["i", "v"], [OpNode("add", "Add", ["i", source], ["i2"])], ["i2", "v"]))


def _counter_steps_by_a_variable_set_from(source):
    # The counter reads `source` only through j, one body pass later.
    return _counting(_graph(["i", "j", "v"], [OpNode("add", "Add", ["i", "j"], ["i2"]),
                                              OpNode("copy", "Identity", [source], ["j2"])], ["i2", "j2", "v"]))


def _if_on(source):
    step = _graph(["i"], [OpNode("add", "Add", ["i", "one"], ["r"])], ["r"])
    stay = _graph(["i"], [OpNode("copy", "Identity", ["i"], ["r"])], ["r"])
    return _counting(_graph(["i", "v"], [OpNode("test", "Greater", [source, "half"], ["p"]),
                                         OpNode("if", "If", ["p", "i"], ["i2"],
                                                {"then_branch": 0, "else_branch": 1})],
                            ["i2", "v"], [step, stay]))


def _while_in_body_up_to(source):
    inner, subgraphs = _up_to(source)
    return _counting(_graph(["i", "v"], [OpNode("add", "Add", ["i", "one"], ["i2"]), inner],
                            ["i2", "v"], subgraphs))


def _while_in_condition_up_to(source):
    inner, subgraphs = _up_to(source)
    return _counting(_graph(["i", "v"], [OpNode("add", "Add", ["i", "one"], ["i2"])], ["i2", "v"]),
                     [inner], subgraphs)


def _while_in_if_branch_up_to(source):
    inner, subgraphs = _up_to(source)
    loops = _graph(["v"], [inner], ["bound"], subgraphs)
    plain = _graph(["v"], [OpNode("copy", "Identity", ["v"], ["bound"])], ["bound"])
    choice = OpNode("if", "If", ["flag", "x"], ["y"], {"then_branch": 0, "else_branch": 1})
    return _graph(["x"], [choice], ["y"], [loops, plain])


@pytest.mark.parametrize("model", [_counter_steps_by, _counter_steps_by_a_variable_set_from, _if_on,
                                   _while_in_body_up_to, _while_in_condition_up_to,
                                   _while_in_if_branch_up_to])
@pytest.mark.parametrize("source, dependent", [("v", True), ("one", False)])
def test_a_loop_count_that_reads_an_input_makes_the_model_input_dependent(model, source, dependent):
    assert input_dependent_loop(model(source)) is dependent


def test_analyze_does_not_run_an_input_dependent_loop_on_zeros(monkeypatch):
    import tigris.cli.trace

    def never(*args, **kwargs):
        raise AssertionError("the plan ran")

    monkeypatch.setattr(tigris.cli.trace, "_run", never)
    model = str(OPS / "float_while_two_variables.tflite")
    reason = "the number of While iterations depends on the input values"
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "64K"])
    assert result.exit_code == 0, result.output
    assert f"not measured: {reason}" in result.output
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "64K", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["slow"]["traffic"] is None
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "64K", "--trace"])
    assert result.exit_code == 2, result.output
    assert f"--trace needs --input here: {reason}" in result.output
