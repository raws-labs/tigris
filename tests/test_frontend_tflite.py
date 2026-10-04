"""TFLite models reach the compiler through their QDQ ONNX expression."""

import os
import struct
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner

from tigris.cli import cli
from tigris.emitters.binary.reader import read_binary_plan
from tigris.frontends import tflite
from tigris.frontends.flatbuffer import Table
from tigris.inspection import inspect_file

FIXTURES = Path(__file__).parent / "fixtures" / "tflite"
KWS = FIXTURES / "kws_ref_model.tflite"


def test_description_reads_the_models_own_facts():
    report = tflite.describe(KWS.read_bytes())
    graph = report["graph"]
    assert report["subgraphs"] == 1
    assert graph["inputs"] == [{"name": "input_1", "kind": "tensor", "dtype": "int8",
                                "shape": [1, 49, 10, 1]}]
    assert graph["outputs"][0]["name"] == "Identity"
    kinds = [op["type"] for op in graph["operators"]]
    assert kinds.count("CONV_2D") == 5 and kinds.count("DEPTHWISE_CONV_2D") == 4
    assert report["unsupported"] == []


def test_conversion_keeps_the_interface_names():
    model = tflite.to_onnx(KWS.read_bytes(), "kws")
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["input_1"]
    assert [o.name for o in model.graph.output] == ["Identity"]


def test_compile_and_analyze_accept_a_tflite_file(tmp_path):
    plan_path = tmp_path / "kws.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(KWS), "-m", "16K", "-o", str(plan_path)])
    assert result.exit_code == 0, result.output
    plan = read_binary_plan(plan_path.read_bytes())
    names = [[plan["tensors"][i]["name"] for i in plan[key]] for key in ("model_inputs", "model_outputs")]
    assert names == [["input_1"], ["Identity"]]
    result = CliRunner().invoke(cli, ["analyze", str(KWS), "-m", "16K"])
    assert result.exit_code == 0, result.output


def _session():
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Installed wheel has no host library")
        pytest.skip("Host library has not been staged in this source checkout")
    from tigris.runtime import Session
    return Session


def _assert_matches_tflite_micro(model: Path, golden: dict, budget: str, tmp_path: Path,
                                 tight: bool = False):
    """Runs the compiled plan on TFLite Micro's recorded inputs, in the
    model's own dtypes, and compares the outputs bit for bit."""
    Session = _session()
    plan_path = tmp_path / "model.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", budget, "-o", str(plan_path)])
    if tight and result.exit_code != 0 and "fast-memory budget" in result.output:
        pytest.skip("does not fit the tight budget: " + result.output.strip().splitlines()[-1])
    assert result.exit_code == 0, result.output
    _, graphs = tflite._read(model.read_bytes())
    _, tensors, inputs, outputs, _ = graphs[0]
    with Session(plan_path) as session:
        assert [tuple(i["shape"]) for i in session.inputs] == [tuple(tensors[i].shape) for i in inputs]
        for sample in range(len(golden["input_0"])):
            feed = {}
            for position, index in enumerate(inputs):
                feed[tensors[index].name] = golden[f"input_{position}"][sample]
            produced = session.run(feed)
            for position, index in enumerate(outputs):
                tensor = tensors[index]
                got = produced[session.outputs[position]["name"]]
                assert got.shape == tuple(tensor.shape)
                assert got.dtype == golden[f"output_{position}"].dtype
                np.testing.assert_array_equal(got, golden[f"output_{position}"][sample],
                                              err_msg=f"sample {sample}, output {position}")


def test_plan_reproduces_tflite_micro_outputs(tmp_path):
    """Recorded TFLite Micro reference outputs, compared bit for bit."""
    recorded = np.load(FIXTURES / "kws_ref_model_tflm.npz")
    golden = {"input_0": recorded["inputs"], "output_0": recorded["outputs"].reshape(-1, 1, 12)}
    _assert_matches_tflite_micro(KWS, golden, "16K", tmp_path)


@pytest.mark.parametrize("name", ["vww_96_int8", "pretrainedResnet_quant"])
@pytest.mark.parametrize("budget,tiled", [("256K", False), ("16K", True)])
def test_reference_model_matches_tflite_micro(name, budget, tiled, tmp_path):
    """Whole MLPerf Tiny models, in one stage and at a budget that tiles
    most of them."""
    model = FIXTURES / f"{name}.tflite"
    _assert_matches_tflite_micro(model, np.load(FIXTURES / f"{name}_tflm.npz"), budget, tmp_path)
    stages = read_binary_plan((tmp_path / "model.tgrs").read_bytes())["stages"]
    tiled_stages = [s for s in stages if s["tile_plan_idx"] != 0xFFFF or s["chain_id"] != 0xFFFF]
    assert bool(tiled_stages) == tiled


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("*.tflite")), ids=lambda p: p.stem)
def test_single_operator_matches_tflite_micro(model, tmp_path):
    assert tflite.unsupported(model.read_bytes()) == []
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256K", tmp_path)


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("*.tflite")), ids=lambda p: p.stem)
def test_single_operator_matches_tflite_micro_when_tiled(model, tmp_path):
    """The same comparison at a budget that forces tiling wherever the
    operator tiles; one that cannot fit it untiled is reported as a skip."""
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path,
                                 tight=True)


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("resize_*_down.tflite")), ids=lambda p: p.stem)
def test_resize_downscales_in_height_bands(model, tmp_path):
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)
    plan = read_binary_plan((tmp_path / "model.tgrs").read_bytes())
    assert any(tile["num_tiles"] > 1 for tile in plan["tile_plans"])



@pytest.mark.parametrize("name,kind", [
    ("cumsum", "CumSum"), ("cumsum_offset", "CumSum"),
    ("cumsum_offset_exclusive_reverse", "CumSum"),
    ("dynamic_update_slice", "DynamicUpdateSlice"), ("float_gather_indices", "Gather"),
    ("float_reverse_channels", "ReverseV2"),
    ("reverse_channels", "ReverseV2"), ("mirror_pad_reflect", "MirrorPad"),
    ("mirror_pad_symmetric", "MirrorPad"),
])
def test_independent_axis_uses_tight_budget(name, kind, tmp_path):
    from tigris.emitters.binary.defs import OP_TYPE_MAP

    model = FIXTURES / "ops" / (name + ".tflite")
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)
    plan = read_binary_plan((tmp_path / "model.tgrs").read_bytes())
    stage = next(stage for stage in plan["stages"]
                 if any(plan["ops"][i]["op_type"] == OP_TYPE_MAP[kind] for i in stage["ops"]))
    assert stage["tile_plan_idx"] != 65535
    tile = plan["tile_plans"][stage["tile_plan_idx"]]
    assert tile["tileable"] and tile["num_tiles"] > 1 and tile["tiled_peak_bytes"] <= 256


@pytest.mark.parametrize("name,kind", [
    ("float_cumsum_offset", "CumSum"), ("float_reduce_max_spatial", "ReduceMax"),
    ("float_reduce_min_rows", "ReduceMin"), ("float_sum_spatial", "ReduceSum"),
])
def test_independent_axis_does_not_hide_an_infeasible_reshape(name, kind, tmp_path):
    from tigris.cli import _run_pipeline

    model = FIXTURES / "ops" / (name + ".tflite")
    graph, _ = _run_pipeline(str(model), ("256",), report_bindings=False)
    stage = next(stage for stage in graph.stages
                 if any(graph.ops[i].op_type == kind for i in stage.op_indices))
    assert stage.tile_plan.tileable and stage.tile_plan.num_tiles > 1
    assert stage.tile_plan.tiled_peak_bytes <= 256
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256",
                                     "-o", str(tmp_path / "model.tgrs")])
    assert result.exit_code != 0
    assert "untileable operators: Reshape" in result.output


def _with_operator_replaced(data: bytes, kind: str, code: int) -> bytes:
    """A copy whose operator code for `kind` names builtin `code` instead."""
    buffer = bytearray(data)
    for entry in Table.root(data).tables(1):
        builtin = max(entry.scalar(3, "i"), entry.scalar(0, "b"))
        if tflite._BUILTIN_OPERATORS[builtin] == kind:
            for slot, fmt in ((3, "<i"), (0, "<b")):
                offset = entry._field(slot)
                if offset:
                    struct.pack_into(fmt, buffer, offset, code)
    return bytes(buffer)


def test_an_unsupported_operator_is_refused_by_name(tmp_path):
    svdf = tflite._BUILTIN_OPERATORS.index("SVDF")
    path = tmp_path / "kws_svdf.tflite"
    path.write_bytes(_with_operator_replaced(KWS.read_bytes(), "FULLY_CONNECTED", svdf))

    report = inspect_file(path)
    assert report["format"] == "tflite"
    assert any("SVDF: not supported" in reason for reason in report["unsupported"])
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "16K", "-o", str(tmp_path / "x.tgrs")])
    assert result.exit_code != 0
    assert "SVDF: not supported" in result.output
    assert not (tmp_path / "x.tgrs").exists()


def test_a_truncated_file_is_an_error_not_a_misread():
    data = KWS.read_bytes()[:4096]
    with pytest.raises(ValueError):
        tflite.describe(data)


def test_an_int8_operator_tflite_micro_runs_in_float_only_is_refused():
    negate = tflite._BUILTIN_OPERATORS.index("NEG")
    data = _with_operator_replaced((FIXTURES / "ops" / "abs.tflite").read_bytes(), "ABS", negate)
    assert any("NEG: TFLite Micro runs it in float32 only" in reason
               for reason in tflite.unsupported(data))


def test_analyze_reports_shapes_in_the_files_axis_order():
    result = CliRunner().invoke(cli, ["analyze", str(KWS), "-m", "16K"])
    assert result.exit_code == 0, result.output
    assert "input_1 1x49x10x1" in result.output
    assert "1x25x5x64" in result.output


def test_an_input_shape_the_file_does_not_state_is_refused():
    same = CliRunner().invoke(cli, ["analyze", str(KWS), "-m", "16K", "--input-shape", "input_1:1x49x10x1"])
    assert same.exit_code == 0, same.output
    other = CliRunner().invoke(cli, ["analyze", str(KWS), "-m", "16K", "--input-shape", "input_1:1x1x49x10"])
    assert other.exit_code != 0
    assert "fixes every tensor shape" in other.output


def test_a_strided_slice_with_a_zero_stride_is_refused():
    model = (FIXTURES / "ops" / "strided_slice.tflite").read_bytes()
    _, graphs = tflite._read(model)
    _, tensors, _, _, operators = graphs[0]
    strides = tensors[operators[0].inputs[3]]
    data = bytearray(model)
    offset = model.index(strides.data)
    data[offset + 4:offset + 8] = np.array([0], np.int32).tobytes()
    assert any("a zero stride" in reason for reason in tflite.unsupported(bytes(data)))


def test_an_int8_l2_pool_is_refused():
    l2_pool = tflite._BUILTIN_OPERATORS.index("L2_POOL_2D")
    data = _with_operator_replaced((FIXTURES / "ops" / "avg_pool_same.tflite").read_bytes(),
                                   "AVERAGE_POOL_2D", l2_pool)
    assert any("L2_POOL_2D: TFLite Micro runs it in float32 only" in reason
               for reason in tflite.unsupported(data))


def test_a_reduction_over_axes_that_are_not_adjacent_is_refused():
    model = (FIXTURES / "ops" / "sum_spatial.tflite").read_bytes()
    _, graphs = tflite._read(model)
    _, tensors, _, _, operators = graphs[0]
    axes = tensors[operators[0].inputs[1]]
    data = bytearray(model)
    offset = model.index(axes.data)
    data[offset:offset + 8] = np.array([1, 3], np.int32).tobytes()
    assert any("axes that are not adjacent" in reason for reason in tflite.unsupported(bytes(data)))


def _read_edited(monkeypatch, name, edit):
    """The fixture `name` as the frontend reads it, after `edit(tensors, operators)`."""
    data = (FIXTURES / "ops" / f"{name}.tflite").read_bytes()
    model, graphs = tflite._read(data)
    edit(graphs[0][1], graphs[0][4])
    monkeypatch.setattr(tflite, "_read", lambda _: (model, graphs))
    return tflite.unsupported(data)


def test_run_time_indices_come_from_a_model_input_or_an_arg_max(monkeypatch):
    """Indices stay positions: an index tensor feeds only index operands, and
    indices computed at run time come from ARG_MAX or ARG_MIN."""
    def swap(tensors, operators):
        gather = next(op for op in operators if op.kind == "GATHER")
        gather.inputs = [gather.inputs[1], gather.inputs[0]]
    reasons = _read_edited(monkeypatch, "float_arg_max_gather", swap)
    assert any("feeds an operand other than indices" in r for r in reasons)
    assert any("other than ARG_MAX or ARG_MIN" in r for r in reasons)


def test_int64_run_time_indices_are_refused(monkeypatch):
    def widen(tensors, operators):
        tensors[operators[0].inputs[1]].type = "INT64"
    reasons = _read_edited(monkeypatch, "float_gather_runtime", widen)
    assert any("INT64 run-time indices; the runtime takes int32" in r for r in reasons)


def test_an_int8_cumsum_keeps_tflite_semantics_in_the_compilers_own_form():
    model = tflite.to_onnx((FIXTURES / "ops" / "cumsum_offset.tflite").read_bytes(), "cumsum")
    scans = [node for node in model.graph.node if node.op_type == "CumSum"]
    assert [node.domain for node in scans] == ["tigris"]


def test_a_variable_is_kept_across_runs_and_reset(tmp_path):
    """The window variable carries from run to run, as TFLite Micro keeps it,
    and a reset starts it over."""
    Session = _session()
    model = FIXTURES / "ops" / "float_variable_window.tflite"
    golden = np.load(model.with_suffix(".npz"))
    plan = tmp_path / "window.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    assert read_binary_plan(plan.read_bytes())["version"] == 10
    with Session(plan) as session:
        assert [i["name"] for i in session.inputs] == ["serving_default_x:0"]
        assert len(session.outputs) == 1
        name = session.outputs[0]["name"]
        first = session.run({"serving_default_x:0": golden["input_0"][0]})[name]
        session.run({"serving_default_x:0": golden["input_0"][1]})
        session.reset_state()
        again = session.run({"serving_default_x:0": golden["input_0"][0]})[name]
    np.testing.assert_array_equal(first, golden["output_0"][0])
    np.testing.assert_array_equal(again, first)
