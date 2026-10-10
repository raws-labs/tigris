"""The execution trace groups and checks runtime events without a host library."""

import json

from click.testing import CliRunner

from tigris.cli import cli
from tigris.cli.trace import check, render, units


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
