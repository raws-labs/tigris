"""TFLite models reach the compiler through their QDQ ONNX expression."""

import json
import re
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
    if tight and result.exit_code != 0 and "does not fit" in result.output:
        lines = result.output.splitlines()
        verdict = next(k for k, line in enumerate(lines) if line.startswith("does not fit"))
        pytest.skip(" ".join(line.strip() for line in lines[verdict:verdict + 2]))
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
    ("dynamic_update_slice_runtime", "DynamicUpdateSlice"),
    ("gather_runtime", "Gather"), ("float_gather_runtime", "Gather"),
    ("float_gather_nd_runtime", "GatherND"), ("float_gather_nd", "GatherND"),
    ("float_logistic_tails", "Sigmoid"), ("float_lstm_time_major_clip", "Lstm"),
    ("float_svdf_rank2_relu", "Svdf"), ("float_mean_rows", "ReduceMean"),
    ("float_strided_slice_steps", "StridedSlice"), ("reverse_spatial", "ReverseV2"),
    ("float_arg_max_channels", "ArgMax"), ("arg_max_channels", "ArgMax"),
    ("float_split", "Split"), ("float_slice", "Split"), ("float_unpack", "Split"),
    ("float_concat_rows", "Concat"), ("float_broadcast_to", "Concat"),
    ("float_concat_outer", "Concat"),
    ("pad", "Pad"), ("padv2", "Pad"),
    ("add_broadcast_rows", "Add"), ("add_constant_rows", "Add"),
    ("add_constant_full", "Add"), ("float_add_rank", "Add"),
    ("sub_broadcast_first", "Sub"), ("float_div_broadcast", "Div"),
    ("float_maximum_broadcast", "Max"), ("float_maximum_rank", "Max"),
    ("float_less_rank", "Less"), ("float_select_v2_broadcast", "Where"),
    ("float_select_v2_rank", "Where"), ("float_gather_matrix", "Transpose"),
    ("float_gather_batch_matrix", "Transpose"),
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
def test_independent_axis_and_reshape_fit_tight_budget(name, kind, tmp_path):
    from tigris.cli import _run_pipeline

    model = FIXTURES / "ops" / (name + ".tflite")
    graph, _ = _run_pipeline(str(model), ("256",), report_bindings=False)
    stage = next(stage for stage in graph.stages
                 if any(graph.ops[i].op_type == kind for i in stage.op_indices))
    assert stage.tile_plan.tileable and stage.tile_plan.num_tiles > 1
    assert stage.tile_plan.tiled_peak_bytes <= 256
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)
    reshapes = [stage for stage in graph.stages if stage.peak_bytes > 256
                and any(graph.ops[i].op_type == "Reshape" for i in stage.op_indices)]
    assert reshapes and all(s.tile_plan.tileable and s.tile_plan.num_tiles > 1 for s in reshapes)


@pytest.mark.parametrize("name", ["float_depth_to_space", "float_space_to_depth", "float_reshape"])
def test_reshape_rank_boundary_fits_tight_budget(name, tmp_path):
    model = FIXTURES / "ops" / (name + ".tflite")
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)


@pytest.mark.parametrize("name,kind,count", [
    ("float_reduce_max_channels", "ReduceMax", 2),
    ("float_sum_channels", "ReduceSum", 2),
    ("float_cumsum_exclusive_reverse", "CumSum", 2),
])
def test_contiguous_reshape_and_leading_operator_fit_tight_budget(name, kind, count, tmp_path):
    from tigris.cli import _run_pipeline
    from tigris.analysis.validation import validate_memory_plan

    model = FIXTURES / "ops" / (name + ".tflite")
    graph, _ = _run_pipeline(str(model), ("256",), report_bindings=False)
    reshapes = [s for s in graph.stages if any(graph.ops[i].op_type == "Reshape" for i in s.op_indices)]
    assert len(reshapes) == count
    oversized = [s for s in reshapes if s.peak_bytes > 256]
    assert oversized and all(s.tile_plan.tileable and s.tile_plan.num_tiles > 1 for s in oversized)
    assert validate_memory_plan(graph).feasible
    stage = next(s for s in graph.stages if any(graph.ops[i].op_type == kind for i in s.op_indices))
    assert len(stage.op_indices) == 1 and stage.chain_len == 0
    assert stage.tile_plan.original_height == 36 and stage.tile_plan.num_tiles > 1
    assert stage.tile_plan.tiled_peak_bytes <= 256
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)
    assert read_binary_plan((tmp_path / "model.tgrs").read_bytes())["version"] == 9



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
    sparse = tflite._BUILTIN_OPERATORS.index("EMBEDDING_LOOKUP_SPARSE")
    path = tmp_path / "kws_sparse.tflite"
    path.write_bytes(_with_operator_replaced(KWS.read_bytes(), "FULLY_CONNECTED", sparse))

    report = inspect_file(path)
    assert report["format"] == "tflite"
    assert any("EMBEDDING_LOOKUP_SPARSE: not supported" in reason for reason in report["unsupported"])
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "16K", "-o", str(tmp_path / "x.tgrs")])
    assert result.exit_code != 0
    assert "EMBEDDING_LOOKUP_SPARSE: not supported" in result.output
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
    assert re.search(r"input_1 +1x49x10x1", result.output)
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
    assert any("other than ARG_MAX, ARG_MIN or int32 arithmetic" in r for r in reasons)


def test_int64_run_time_indices_are_refused(monkeypatch):
    def widen(tensors, operators):
        tensors[operators[0].inputs[1]].type = "INT64"
    reasons = _read_edited(monkeypatch, "float_gather_runtime", widen)
    assert any("INT64 run-time indices; the runtime takes int32" in r for r in reasons)


@pytest.mark.parametrize(("name", "arena"), [
    # TFLite Micro's own "Arena allocation head" (tflite-micro 0.dev20260925222358) for
    # models whose kernels request no scratch buffer.
    ("vww_96_int8.tflite", 73728), ("pretrainedResnet_quant.tflite", 49152),
    ("ops/conv_bias.tflite", 512),
])
def test_tflm_tensor_arena_reproduces_its_planner(name, arena):
    assert tflite.tflm_tensor_arena((FIXTURES / name).read_bytes()) == arena


def test_tflm_tensor_arena_declines_several_subgraphs():
    assert tflite.tflm_tensor_arena((FIXTURES / "ops" / "while_conv.tflite").read_bytes()) is None


def test_analyze_reports_json_with_tflite_micro_arena_and_exact_plan_size(tmp_path):
    model = FIXTURES / "ops" / "while_conv.tflite"
    result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "256K", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert (report["report"], report["version"]) == ("tigris-analysis", 1)
    assert report["model"]["format"] == "tflite" and report["model"]["dtype"] == "int8"
    assert report["model"]["inputs"] == [{"name": "x", "shape": [1, 6, 6, 4]}]
    # Several subgraphs: TFLite Micro's figure is not modelled.
    assert report["tflite_micro"] is None
    plan = tmp_path / "m.tgrs"
    assert CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(plan)]).exit_code == 0
    # Subgraph weights and sections count: the size is the serialized plan's.
    assert report["flash"]["plan_bytes"] == plan.stat().st_size
    single = json.loads(CliRunner().invoke(
        cli, ["analyze", str(FIXTURES / "ops" / "conv_bias.tflite"), "-m", "256K", "--json"]).output)
    assert single["tflite_micro"]["tensor_arena_bytes"] == 512


def test_flexbuffer_map_reads_scalar_options():
    from tigris.frontends.flatbuffer import flexbuffer_map
    data = bytes.fromhex(
        "6d61785f646574656374696f6e73006e6d735f696f755f7468726573686f6c64007573655f726567756c61"
        "725f6e6d730003322413000000060000000100000003000000030000000000003f01000000060e6a0f2601")
    assert flexbuffer_map(data) == {"max_detections": 3, "nms_iou_threshold": 0.5,
                                    "use_regular_nms": True}
    with pytest.raises(ValueError):
        flexbuffer_map(data[40:])


@pytest.mark.parametrize(("option", "value", "reason"), [
    ("max_classes_per_detection", 2, "more than one class per detection in the fast form"),
    ("nms_iou_threshold", 1.5, "an IoU threshold outside (0, 1]"),
    ("num_classes", 1, "scores for other than the classes and at most one background column"),
    ("max_detections", None, "option max_detections is missing"),
])
def test_a_detection_tflite_micro_cannot_run_is_refused(monkeypatch, option, value, reason):
    """TFLite Micro's fast form writes past its outputs for more than one
    class per detection; the other options it checks or requires."""
    read = tflite.flexbuffer_map
    monkeypatch.setattr(tflite, "flexbuffer_map", lambda data: {**read(data), option: value})
    data = (FIXTURES / "ops" / "float_detection_fast.tflite").read_bytes()
    assert any(reason in r for r in tflite.unsupported(data))


def test_detection_anchors_are_a_constant(monkeypatch):
    def run_time(tensors, operators):
        tensors[operators[0].inputs[2]].buffer = 0
    reasons = _read_edited(monkeypatch, "float_detection_regular", run_time)
    assert any("anchors that are not a constant" in r for r in reasons)


def test_an_svdf_without_bias_is_refused(monkeypatch):
    """TFLite Micro's SVDF Prepare reads the bias whether or not it is there."""
    def drop(tensors, operators):
        operators[0].inputs = [*operators[0].inputs[:3], -1, operators[0].inputs[4]]
    reasons = _read_edited(monkeypatch, "float_svdf", drop)
    assert any("no bias, which TFLite Micro requires" in r for r in reasons)


def test_an_int8_svdf_keeps_int16_state(monkeypatch):
    def narrow(tensors, operators):
        for position in (2, 4):
            tensors[operators[0].inputs[position]].type = "INT8"
    reasons = _read_edited(monkeypatch, "svdf", narrow)
    assert any("int8 state; the converter writes int16" in r for r in reasons)


def test_a_variable_tensor_is_only_svdf_state(monkeypatch):
    def mark(tensors, operators):
        tensors[operators[0].inputs[0]].variable = True
    reasons = _read_edited(monkeypatch, "float_svdf", mark)
    assert any("is not the state of one SVDF" in r for r in reasons)


@pytest.mark.parametrize(("edit", "reason"), [
    (lambda ops: ops[0].inputs.__setitem__(1, -1), "a missing gate, which TFLite Micro requires"),
    (lambda ops: ops[0].inputs.__setitem__(9, 5), "peepholes, projection or layer normalization"),
])
def test_an_lstm_tflite_micro_cannot_run_is_refused(monkeypatch, edit, reason):
    def change(tensors, operators):
        operators[0].inputs = list(operators[0].inputs)
        edit(operators)
    reasons = _read_edited(monkeypatch, "float_lstm", change)
    assert any(reason in r for r in reasons)


def test_an_lstm_cell_activation_other_than_tanh_is_refused(monkeypatch):
    def relu(tensors, operators):
        operators[0].option = lambda slot, fmt, default=0: 1 if slot == 0 else default
    reasons = _read_edited(monkeypatch, "float_lstm", relu)
    assert any("cell activation relu" in r for r in reasons)


def test_an_int8_lstm_cell_is_symmetric_int16(monkeypatch):
    def offset(tensors, operators):
        tensors[operators[0].inputs[19]].zero_point = np.asarray([5], np.int64)
    reasons = _read_edited(monkeypatch, "lstm", offset)
    assert any("cell state other than symmetric int16" in r for r in reasons)


def test_an_int8_lstm_with_per_channel_weights_is_refused(monkeypatch):
    """TFLite Micro's LSTM reads one multiplier per gate projection."""
    def per_channel(tensors, operators):
        weight = tensors[operators[0].inputs[1]]
        weight.scale = np.repeat(weight.scale, weight.shape[0])
    reasons = _read_edited(monkeypatch, "lstm", per_channel)
    assert any("quantized per tensor" in r for r in reasons)


def test_an_if_on_operands_other_than_float32_is_refused(monkeypatch):
    def narrow(tensors, operators):
        branching = next(op for op in operators if op.kind == "IF")
        tensors[branching.inputs[1]].type = "INT8"
    reasons = _read_edited(monkeypatch, "float_if", narrow)
    assert any("operands other than float32" in r for r in reasons)


def test_state_inside_an_if_branch_is_refused(monkeypatch):
    data = (FIXTURES / "ops" / "float_if.tflite").read_bytes()
    model, graphs = tflite._read(data)
    graphs[1][4][0].kind = "READ_VARIABLE"
    monkeypatch.setattr(tflite, "_read", lambda _: (model, graphs))
    assert any("state inside a subgraph" in r for r in tflite.unsupported(data))


def _counter_reasons(monkeypatch, edit):
    """float_while_counter's reasons after `edit(graphs)`; graph 2 is its body."""
    data = (FIXTURES / "ops" / "float_while_counter.tflite").read_bytes()
    model, graphs = tflite._read(data)
    edit(graphs)
    monkeypatch.setattr(tflite, "_read", lambda _: (model, graphs))
    return tflite.unsupported(data)


def _body_op(graphs, kind):
    return next(op for op in graphs[2][4] if op.kind == kind and graphs[2][1][op.inputs[0]].type == "INT32")


@pytest.mark.parametrize(("edit", "reason"), [
    (lambda g: setattr(g[2][1][_body_op(g, "ADD").inputs[1]], "type", "FLOAT32"),
     "int32 mixed with other operands"),
    (lambda g: setattr(g[2][1][_body_op(g, "CAST").outputs[0]], "type", "INT8"),
     "int32 cast to other than float32"),
    (lambda g: setattr(g[0][1][g[0][4][0].inputs[0]], "type", "INT16"),
     "loop variables other than float32 or int32"),
])
def test_int32_computation_tflite_micro_runs_differently_is_refused(monkeypatch, edit, reason):
    assert any(reason in r for r in _counter_reasons(monkeypatch, edit))


def test_a_loop_starting_from_a_constant_does_not_compress(tmp_path):
    result = CliRunner().invoke(cli, ["compile", str(FIXTURES / "ops" / "float_while_counter.tflite"),
                                      "-m", "256K", "-c", "lz4", "-o", str(tmp_path / "m.tgrs")])
    assert result.exit_code != 0
    assert "a loop starting from a constant does not compress" in result.output


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
