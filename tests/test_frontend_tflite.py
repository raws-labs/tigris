"""TFLite models reach the compiler through their QDQ ONNX expression."""

import itertools
import json
import re
import os
import struct
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner

from tigris.analysis.plan_graph import slow_arena_required
from tigris.cli import cli
from tigris.emitters.binary.reader import read_binary_plan
from tigris.frontends import tflite
from tigris.frontends.flatbuffer import Table
from tigris.frontends.tflite_export import tflm_tensor_arena
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


def _session():
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Installed wheel has no host library")
        pytest.skip("Host library has not been staged in this source checkout")
    from tigris.runtime import Session
    return Session


def _assert_matches_tflite_micro(model: Path, golden: dict, budget: str, tmp_path: Path):
    """Runs the compiled plan on TFLite Micro's recorded inputs, in the
    model's own dtypes, and compares the outputs bit for bit. The slow arena
    is exactly what generated applications reserve for the plan."""
    Session = _session()
    plan_path = tmp_path / "model.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", budget, "-o", str(plan_path)])
    assert result.exit_code == 0, result.output
    _, graphs = tflite._read(model.read_bytes())
    _, tensors, inputs, outputs, _ = graphs[0]
    slow = slow_arena_required(read_binary_plan(plan_path.read_bytes()))
    with Session(plan_path, slow_capacity=slow) as session:
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
                expected = golden[f"output_{position}"][sample]
                if ("reference" in golden and golden["reference"].item() == "TFLite BUILTIN_REF"
                        and expected.dtype == np.float32):
                    tolerance = 4 * np.finfo(np.float32).eps * float(np.max(np.abs(expected)))
                    np.testing.assert_allclose(got, expected, rtol=0, atol=tolerance,
                                               err_msg=f"sample {sample}, output {position}")
                else:
                    np.testing.assert_array_equal(got, expected,
                                                  err_msg=f"sample {sample}, output {position}")


def test_plan_reproduces_tflite_micro_outputs(tmp_path):
    """Recorded TFLite Micro reference outputs, compared bit for bit."""
    recorded = np.load(FIXTURES / "kws_ref_model_tflm.npz")
    golden = {"input_0": recorded["inputs"], "output_0": recorded["outputs"].reshape(-1, 1, 12)}
    _assert_matches_tflite_micro(KWS, golden, "16K", tmp_path)
    plan = read_binary_plan((tmp_path / "model.tgrs").read_bytes())
    names = [[plan["tensors"][i]["name"] for i in plan[key]] for key in ("model_inputs", "model_outputs")]
    assert names == [["input_1"], ["Identity"]]


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


# The fixtures that compile untiled at the tight budget.
FITS_UNTILED = {
    "arg_max", "arg_max_gather", "arg_min_rows", "batch_matmul", "batch_to_space", "broadcast_to",
    "concat_outer", "concat_rank2", "depth_to_space", "embedding_lookup",
    "embedding_lookup_runtime", "expand_dims", "expand_dims_op", "float_arg_max",
    "float_arg_max_gather", "float_arg_min_rows", "float_broadcast_args", "float_concat_rank2",
    "float_embedding_lookup", "float_embedding_lookup_runtime", "float_fill",
    "float_fully_connected_bias", "float_if", "float_if_if", "float_if_int32", "float_if_through",
    "float_if_two_operands", "float_log_softmax", "float_lstm", "float_lstm_unrolled",
    "float_pack_outer", "float_shape_arithmetic", "float_shape_pack_reshape", "float_shape_reshape",
    "float_shape_reverse_reshape", "float_softmax", "float_svdf", "float_unpack_one",
    "float_variable_window", "float_while", "float_while_countdown", "float_while_counter",
    "float_while_feature_map", "float_while_if_through", "float_while_through",
    "float_while_two_variables", "float_zeros_like",
    "fully_connected_bias", "fully_connected_rank3", "fully_connected_rank3_relu",
    "fully_connected_row_relu", "gather_batch", "gather_batch_matrix", "gather_index_arithmetic",
    "log_softmax", "lstm", "lstm_time_major_clip", "lstm_unrolled", "mean_spatial",
    "mean_spatial_flat", "mul_broadcast_both", "pack", "pack_outer", "requantize_shared", "reshape",
    "resize_bilinear", "resize_bilinear_asymmetric", "resize_nearest", "shape_pack_reshape",
    "softmax", "space_to_batch", "space_to_depth", "split", "split_v", "squeeze", "squeeze_op",
    "svdf", "svdf_batch", "transpose", "unpack", "unpack_one",
}

# The fixtures refused at 256 bytes, by what keeps a stage over it: the type of
# an operator that runs untiled, or the refusal's reason.
REFUSED = {
    "DetectionPostProcess": {"detection_fast", "detection_regular", "float_detection_fast",
                             "float_detection_regular"},
    "ReduceMean": {"float_mean_all"},
    "ReduceSum": {"float_if_feature_map"},
    "smallest 2D tile": {"float_conv_dilated"},
    "smallest tile": {"float_batch_to_space", "float_conv_split", "float_cumsum",
                      "float_cumsum_offset_exclusive_reverse", "float_dynamic_update_slice",
                      "float_dynamic_update_slice_runtime", "float_gelu_approximate_wide",
                      "float_gelu_wide", "float_if_feature_map", "float_mean_all",
                      "float_mean_nonadjacent_image", "float_mean_nonadjacent_keep",
                      "float_mean_nonadjacent_negative", "float_mirror_pad_reflect",
                      "float_mirror_pad_symmetric", "float_pad", "float_padv2",
                      "float_reverse_spatial", "float_space_to_batch", "float_transpose_conv",
                      "gelu_approximate_offset", "gelu_wide", "hard_swish_offset",
                      "hard_swish_wide", "mean_nonadjacent_image", "mean_nonadjacent_keep",
                      "mean_nonadjacent_negative"},
}

# Refused fixtures that tile at 1K.
TILES_AT_1K = {
    "float_conv_dilated", "float_conv_split", "float_cumsum",
    "float_cumsum_offset_exclusive_reverse", "float_dynamic_update_slice",
    "float_dynamic_update_slice_runtime", "float_mean_all", "float_mirror_pad_reflect",
    "float_mirror_pad_symmetric", "float_pad", "float_padv2", "float_reverse_spatial",
    "float_transpose_conv", "gelu_approximate_offset", "gelu_wide", "hard_swish_offset",
    "hard_swish_wide",
}

# The fixtures whose tight plan runs a chain of stages.
CHAINED = {
    "cast_bool", "float_logical_not", "float_not_equal", "if_conv", "logical_not", "not_equal",
    "while_conv",
}

# The operator whose own stage tiles, not only a stage next to it.
INDEPENDENT_AXIS = {
    "cumsum": "CumSum", "cumsum_offset": "CumSum", "cumsum_offset_exclusive_reverse": "CumSum",
    "dynamic_update_slice": "DynamicUpdateSlice", "float_gather_indices": "Gather",
    "float_reverse_channels": "ReverseV2", "reverse_channels": "ReverseV2",
    "mirror_pad_reflect": "MirrorPad", "mirror_pad_symmetric": "MirrorPad",
    "dynamic_update_slice_runtime": "DynamicUpdateSlice", "gather_runtime": "Gather",
    "float_gather_runtime": "Gather", "float_gather_nd_runtime": "GatherND",
    "float_gather_nd": "GatherND", "float_logistic_tails": "Sigmoid",
    "float_lstm_time_major_clip": "Lstm", "float_svdf_rank2_relu": "Svdf",
    "float_mean_rows": "ReduceMean", "float_strided_slice_steps": "StridedSlice",
    "reverse_spatial": "ReverseV2", "float_arg_max_channels": "ArgMax",
    "arg_max_channels": "ArgMax", "float_split": "Split", "float_slice": "Split",
    "float_unpack": "Split", "float_concat_rows": "Concat", "float_broadcast_to": "Concat",
    "float_concat_outer": "Concat", "pad": "Pad", "padv2": "Pad", "add_broadcast_rows": "Add",
    "add_constant_rows": "Add", "add_constant_full": "Add", "float_add_rank": "Add",
    "sub_broadcast_first": "Sub", "float_div_broadcast": "Div", "float_maximum_broadcast": "Max",
    "float_maximum_rank": "Max", "float_less_rank": "Less", "float_select_v2_broadcast": "Where",
    "float_select_v2_rank": "Where", "float_gather_matrix": "Transpose",
    "float_gather_batch_matrix": "Transpose",
}


def _blockers(output: str) -> set[str]:
    """The operator types that run untiled and the other reasons in a refusal."""
    found = set()
    for reason in re.findall(r"^\s+stage \d+\s+[\d.]+ \w+\s+(.+)$", output, re.M):
        untiled = re.match(r"runs untiled: (\w+?)(?:_\d+)?$", reason)
        found.add(untiled.group(1) if untiled else reason.strip())
    return found


def test_the_tight_budget_tables_name_fixtures():
    stems = {model.stem for model in (FIXTURES / "ops").glob("*.tflite")}
    refused = set().union(*REFUSED.values())
    assert FITS_UNTILED | refused | CHAINED | set(INDEPENDENT_AXIS) <= stems
    assert TILES_AT_1K <= refused and not FITS_UNTILED & refused
    assert not (CHAINED | set(INDEPENDENT_AXIS)) & (FITS_UNTILED | refused)


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("*.tflite")), ids=lambda p: p.stem)
def test_single_operator_at_a_tight_budget(model, tmp_path):
    """The same comparison at 256 bytes, or at 1K for the fixtures in
    TILES_AT_1K, which forces tiling wherever the operator tiles. Whether
    each fixture tiles, fits untiled or is refused is recorded above, so a
    change in either direction fails."""
    name = model.stem
    budget, limit = "256", 256
    expected = {blocker for blocker, names in REFUSED.items() if name in names}
    if expected:
        result = CliRunner().invoke(cli, ["compile", str(model), "-m", budget,
                                         "-o", str(tmp_path / "model.tgrs")])
        assert result.exit_code != 0
        assert "does not fit 256 B fast memory" in result.output, result.output
        assert _blockers(result.output) == expected, result.output
        if name not in TILES_AT_1K:
            return
        budget, limit = "1K", 1024
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), budget, tmp_path)
    plan = read_binary_plan((tmp_path / "model.tgrs").read_bytes())
    for tile in plan["tile_plans"]:
        assert tile["tileable"] and tile["tiled_peak_bytes"] <= limit
    multi_tile = any(tile["num_tiles"] > 1 for tile in plan["tile_plans"])
    chained = any(stage["chain_id"] != 0xFFFF for stage in plan["stages"])
    assert chained == (name in CHAINED)
    assert (multi_tile or chained) == (name not in FITS_UNTILED)
    if name in INDEPENDENT_AXIS:
        from tigris.emitters.binary.defs import OP_TYPE_MAP

        op_type = OP_TYPE_MAP[INDEPENDENT_AXIS[name]]
        stage = next(stage for stage in plan["stages"]
                     if any(plan["ops"][i]["op_type"] == op_type for i in stage["ops"]))
        assert stage["tile_plan_idx"] != 0xFFFF
        assert plan["tile_plans"][stage["tile_plan_idx"]]["num_tiles"] > 1


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
    assert read_binary_plan((tmp_path / "model.tgrs").read_bytes())["version"] == 11



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


class _StridedSlice:
    """A STRIDED_SLICE operator that sets only its begin, end and shrink masks
    and its offset flag."""
    kind = "STRIDED_SLICE"

    def __init__(self, begin_mask: int, end_mask: int, shrink: int, offset: bool = False):
        self.masks = {0: begin_mask, 1: end_mask, 4: shrink, 5: offset}

    def option(self, slot, fmt, default=0):
        return self.masks.get(slot, default)


class _Constant:
    type, scale, zero_point, buffer = "INT32", None, None, 1

    def __init__(self, values):
        self.values = np.asarray(values, np.int32)
        self.shape, self.data = list(self.values.shape), self.values.tobytes()

    def array(self):
        return self.values


def test_every_strided_slice_path_takes_one_rule():
    """One axis of extent 0-5, begin and end -7..7, strides +-1..3 and every
    begin, end and shrink mask: the folded constant, the stride-1 bounds and
    the shape check follow _strided_slice_axes, which is Python's slice of the
    axis, or the element TFLite Micro takes from a dropped axis; it writes none
    for a dropped axis with a negative stride."""
    for extent, begin, end, stride, begin_mask, end_mask, shrink in itertools.product(
            range(6), range(-7, 8), range(-7, 8), (-3, -2, -1, 1, 2, 3), (0, 1), (0, 1), (0, 1)):
        op, x = _StridedSlice(begin_mask, end_mask, shrink), np.arange(extent, dtype=np.int32)
        ins = [_Constant(x), _Constant([begin]), _Constant([end]), _Constant([stride])]
        case = (extent, begin, end, stride, begin_mask, end_mask, shrink)
        if shrink:
            index = ((0 if stride > 0 else extent - 1) if begin_mask
                     else begin + extent if begin < 0 else begin)
            expected = [index] if 0 <= index < extent and stride > 0 else None
        else:
            expected = list(range(*slice(None if begin_mask else begin, None if end_mask else end,
                                         stride).indices(extent)))
        axes = tflite._strided_slice_axes(tflite._StridedSliceOptions.of(op), [extent],
                                          [begin], [end], [stride])
        assert (None if axes is None else list(range(*axes[0][:3]))) == expected, case
        folded = tflite._constant_move(op, ins)
        assert (None if folded is None else folded.reshape(-1).tolist()) == expected, case
        if stride == 1:
            bounds = tflite._strided_slice_bounds(op, ins)
            assert bounds == (None if not expected else
                              [(expected[0], len(expected), bool(shrink))]), case
        if not (shrink and stride < 0):
            out = _Constant(np.zeros(() if shrink else len(expected or ()), np.int32))
            reason = tflite._option_shape_reason(op, ins, [out])
            assert reason == ("" if expected is not None
                              else "a dropped axis at an index outside the axis"), case


def test_a_constant_strided_slice_reads_one_entry_per_axis_and_adds_the_offset():
    """TFLite Micro reads begin, end and strides at each input axis: with fewer
    entries the slice does not fold, entries past the last axis are not read,
    and with offset set the end of an axis is begin + end."""
    x = np.arange(6, dtype=np.int32).reshape(2, 3)
    ins = [_Constant(x), _Constant([1]), _Constant([2]), _Constant([1])]
    assert tflite._constant_move(_StridedSlice(0, 0, 0), ins) is None
    ins = [_Constant(x), _Constant([1, 0, 9]), _Constant([2, 3, 9]), _Constant([1, 1, 0])]
    assert tflite._constant_move(_StridedSlice(0, 0, 0), ins).tolist() == [[3, 4, 5]]
    ins = [_Constant(np.arange(6, dtype=np.int32)), _Constant([1]), _Constant([2]), _Constant([1])]
    assert tflite._constant_move(_StridedSlice(0, 0, 0), ins).tolist() == [1]
    assert tflite._constant_move(_StridedSlice(0, 0, 0, offset=True), ins).tolist() == [1, 2]


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


def _with_option(data: bytes, kind: str, slot: int, fmt: str, value) -> bytes:
    """A copy whose first `kind` operator stores `value` in options field `slot`,
    which the file already holds."""
    model = tflite._Model(data)
    operators = [tflite._Operator(model, t, i) for i, t in enumerate(model.subgraphs[0].tables(3))]
    position = next(op for op in operators if op.kind == kind).options.present().get(slot)
    assert position, f"{kind} stores no field {slot}"
    buffer = bytearray(data)
    struct.pack_into(fmt, buffer, position, value)
    return bytes(buffer)


def _fc_quant_spec(payload: bytes) -> Table:
    """FullyConnectedOptions that set only quant_spec, to `payload`."""
    vtable = struct.pack("<8H", 16, 8, 0, 0, 0, 0, 0, 4)
    return Table.root(struct.pack("<I", 4 + len(vtable)) + vtable
                      + struct.pack("<iII", len(vtable), 4, len(payload)) + payload)


def test_a_fully_connected_quantization_contract_is_refused(monkeypatch):
    """A runtime that does not know the contract quant_spec names must refuse
    the operator."""
    def contract(tensors, operators):
        operators[0].options = _fc_quant_spec(b"\x00\x01\x02")
    assert _read_edited(monkeypatch, "fully_connected_bias", contract) == [
        "operator 0 FULLY_CONNECTED: a non-standard quantization contract in quant_spec"]


def test_an_empty_fully_connected_quant_spec_converts(monkeypatch):
    """An empty quant_spec names no contract, and TFLite Micro runs the operator
    as standard FULLY_CONNECTED."""
    options = _fc_quant_spec(b"")
    assert 5 in options.present()

    def empty(tensors, operators):
        operators[0].options = options
    assert _read_edited(monkeypatch, "fully_connected_bias", empty) == []


def test_a_depth_multiplier_that_disagrees_with_the_channels_is_refused():
    """TFLite Micro maps the channels by depth_multiplier, not by the shapes."""
    model = (FIXTURES / "ops" / "depthwise_multiplier.tflite").read_bytes()
    data = _with_option(model, "DEPTHWISE_CONV_2D", 3, "<i", 1)
    assert tflite.unsupported(data) == [
        "operator 0 DEPTHWISE_CONV_2D: depth_multiplier 1 for 4 input and 8 output channels"]


def test_a_window_whose_padding_does_not_give_its_shapes_is_refused(tmp_path):
    """TFLite Micro pads by the padding option and runs over the stated output."""
    model = (FIXTURES / "ops" / "pad_conv.tflite").read_bytes()
    path = tmp_path / "same.tflite"
    path.write_bytes(_with_option(model, "CONV_2D", 0, "<b", 0))
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "256K", "-o", str(tmp_path / "x.tgrs")])
    assert result.exit_code != 0
    assert "CONV_2D: a 6x8 output where SAME padding gives 8x10" in result.output


@pytest.mark.parametrize(("name", "padding", "reason"), [
    ("transpose_conv", 1,
     "operator 0 TRANSPOSE_CONV: a 4x4 input where VALID padding of the output gives 3x3"),
    ("max_pool_same", 2, "operator 0 MAX_POOL_2D: padding 2, neither SAME nor VALID"),
])
def test_a_padding_option_the_shapes_do_not_follow_is_refused(monkeypatch, name, padding, reason):
    def pad(tensors, operators):
        stated = operators[0].option
        operators[0].option = lambda slot, fmt, default=0: padding if slot == 0 else stated(slot, fmt, default)
    assert _read_edited(monkeypatch, name, pad) == [reason]


@pytest.mark.parametrize(("name", "kind", "count", "reason"), [
    ("pack", "PACK", 1, "operator 0 PACK: values_count 1 for 2 inputs"),
    ("shape_pack_reshape", "PACK", 1, "operator 2 PACK: values_count 1 for 2 inputs"),
    ("unpack", "UNPACK", 1, "operator 0 UNPACK: num 1 for 3 outputs"),
    ("unpack_one", "UNPACK", 2, "operator 0 UNPACK: num 2 for 1 output"),
])
def test_a_pack_count_that_disagrees_with_the_operands_is_refused(name, kind, count, reason):
    """TFLite Micro packs and unpacks as many tensors as the count says; a
    constant PACK is kept for the same check."""
    model = (FIXTURES / "ops" / f"{name}.tflite").read_bytes()
    assert reason in tflite.unsupported(_with_option(model, kind, 0, "<i", count))


@pytest.mark.parametrize(("name", "kind", "block", "reason"), [
    ("depth_to_space", "DEPTH_TO_SPACE", 1, "a 1x6x6x2 output where block_size 1 gives 1x3x3x8"),
    ("depth_to_space", "DEPTH_TO_SPACE", 3, "block_size 3 for a 1x3x3x8 input"),
    ("space_to_depth", "SPACE_TO_DEPTH", 1, "a 1x2x3x12 output where block_size 1 gives 1x4x6x3"),
    ("space_to_depth", "SPACE_TO_DEPTH", 0, "block_size 0 for a 1x4x6x3 input"),
])
def test_a_block_size_the_shapes_do_not_follow_is_refused(name, kind, block, reason):
    """TFLite Micro shapes the output by block_size, not by the file."""
    model = (FIXTURES / "ops" / f"{name}.tflite").read_bytes()
    assert tflite.unsupported(_with_option(model, kind, 0, "<i", block)) == [f"operator 0 {kind}: {reason}"]


def test_an_lstm_with_diagonal_recurrent_weights_is_refused(monkeypatch):
    """TFLite Micro fails to allocate an LSTM whose recurrent weights are vectors."""
    def diagonal(tensors, operators):
        for position in range(5, 9):
            weight = tensors[operators[0].inputs[position]]
            weight.shape = weight.shape[:1]
    assert _read_edited(monkeypatch, "float_lstm", diagonal) == [
        "operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: diagonal recurrent weights, which TFLite Micro does not run"]


def _read_edited(monkeypatch, name, edit):
    """The fixture `name` as the frontend reads it, after `edit(tensors, operators)`."""
    data = (FIXTURES / "ops" / f"{name}.tflite").read_bytes()
    model, graphs = tflite._read(data)
    edit(graphs[0][1], graphs[0][4])
    monkeypatch.setattr(tflite, "_read", lambda _: (model, graphs))
    return tflite.unsupported(data)


@pytest.mark.parametrize("weights, advice", [("INT8", True), ("INT16", False)])
def test_dynamic_range_weights_name_the_conversion(monkeypatch, weights, advice):
    def quantize_weights(tensors, operators):
        tensors[operators[0].inputs[1]].type = weights
    reasons = _read_edited(monkeypatch, "float_conv_bias", quantize_weights)
    refusal = next(r for r in reasons if "must be constant float32" in r)
    hint = "; convert to full int8 or to float32 without dynamic-range quantization"
    assert refusal.endswith(hint) is advice


def test_run_time_indices_come_from_a_model_input_or_an_arg_max(monkeypatch):
    """Indices stay positions: an index tensor feeds only index operands, and
    indices computed at run time come from ARG_MAX or ARG_MIN."""
    def swap(tensors, operators):
        gather = next(op for op in operators if op.kind == "GATHER")
        gather.inputs = [gather.inputs[1], gather.inputs[0]]
    reasons = _read_edited(monkeypatch, "float_arg_max_gather", swap)
    assert any("feeds an operand other than indices" in r for r in reasons)
    assert any("other than ARG_MAX, ARG_MIN or int32 arithmetic" in r for r in reasons)


def test_int64_run_time_indices_with_an_excessive_range_are_refused(monkeypatch):
    def widen(tensors, operators):
        index = tensors[operators[0].inputs[1]]
        index.type = "INT64"
        index.maximum = [float(1 << 40)]
    reasons = _read_edited(monkeypatch, "float_gather_runtime", widen)
    assert any("INT64 run-time indices; the runtime takes int32" in r for r in reasons)


@pytest.mark.parametrize(("name", "arena"), [
    # TFLite Micro's own "Arena allocation head" (the dev-extra tflite-micro) for
    # models whose kernels request no scratch buffer or a reduction's.
    ("vww_96_int8.tflite", 73728), ("pretrainedResnet_quant.tflite", 49152),
    ("ops/conv_bias.tflite", 512), ("ops/mean_channels.tflite", 368),
    ("ops/sum_spatial.tflite", 208), ("ops/float_reduce_max_channels.tflite", 752),
    ("ops/reduce_min_rows.tflite", 208),
])
def test_tflm_tensor_arena_reproduces_its_planner(name, arena):
    assert tflm_tensor_arena((FIXTURES / name).read_bytes()) == arena


def test_tflm_tensor_arena_declines_several_subgraphs():
    assert tflm_tensor_arena((FIXTURES / "ops" / "while_conv.tflite").read_bytes()) is None


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
    assert single["tflite_micro"]["excludes"] == [
        "kernel scratch buffers other than those of MEAN, SUM, REDUCE_MAX, REDUCE_MIN and "
        "REDUCE_ALL", "persistent allocations"]
    assert set(single["tflite_micro"]["export"]) == {
        "arena_head_before_bytes", "arena_head_after_bytes", "persistent_growth_bytes",
        "change_bytes", "region", "bands", "pads_folded", "macs", "extra_macs", "target",
        "reason"}


def test_analyze_shows_the_lz4_estimate_only_where_lz4_compiles():
    # VWW's weights decompress to about 211 KiB for one stage, more than 32K holds.
    model = FIXTURES / "vww_96_int8.tflite"
    for budget, compiles in (("32K", False), ("256K", True)):
        written = CliRunner().invoke(cli, ["compile", str(model), "-m", budget, "-m", "1M", "-c", "lz4",
                                           "-o", os.devnull])
        assert (written.exit_code == 0) == compiles, written.output
        result = CliRunner().invoke(cli, ["analyze", str(model), "-m", budget, "-m", "1M"])
        assert ("with -c lz4" in result.output) == compiles, result.output
        report = json.loads(CliRunner().invoke(
            cli, ["analyze", str(model), "-m", budget, "-m", "1M", "--json"]).output)
        assert (report["flash"]["estimates"]["lz4_plan_bytes"] is not None) == compiles


def test_compile_keeps_only_interface_names_unless_asked(tmp_path):
    model = FIXTURES / "vww_96_int8.tflite"
    plans = {}
    for name, extra in (("stripped", []), ("named", ["--keep-names"])):
        path = tmp_path / f"{name}.tgrs"
        result = CliRunner().invoke(cli, ["compile", str(model), "-m", "32K", "-m", "1M", "-o", str(path),
                                          *extra])
        assert result.exit_code == 0, result.output
        plans[name] = read_binary_plan(path.read_bytes())
        plans[name]["size"] = path.stat().st_size
    stripped, named = plans["stripped"], plans["named"]
    interface = {named["tensors"][i]["name"] for i in (*named["model_inputs"], *named["model_outputs"])}
    assert interface == {"input_1_int8", "Identity_int8"}
    assert stripped["model_name"] == named["model_name"] == "vww_96_int8"
    assert {t["name"] for t in stripped["tensors"]} == interface | {""}
    assert {op["name"] for op in stripped["ops"]} == {""}
    assert {w["name"] for w in stripped["weights"]} == {""}
    assert all(op["name"] for op in named["ops"])
    assert stripped["size"] < named["size"]
    report = json.loads(CliRunner().invoke(cli, ["analyze", str(model), "-m", "32K", "-m", "1M", "--json"]).output)
    assert report["flash"]["plan_bytes"] == stripped["size"]


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
    assert read_binary_plan(plan.read_bytes())["version"] == 11
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


@pytest.mark.parametrize("name", ["mean_spatial", "mean_spatial_flat", "float_mean_spatial", "float_mean_spatial_flat"])
def test_grouped_pool_preserves_recorded_outputs(name, tmp_path):
    from dataclasses import replace
    from tigris.cli import _run_pipeline
    from tigris.emitters.binary.writer import emit_binary_bytes

    source = FIXTURES / "ops" / (name + ".tflite")
    graph, _ = _run_pipeline(str(source), ("96", "16K"))
    stage = next(stage for stage in graph.stages if
                 len(stage.op_indices) == 1 and graph.ops[stage.op_indices[0]].op_type == "GlobalAveragePool")
    batch, channels, height, width = graph.tensors[stage.input_tensors[0]].shape
    stage.tile_plan = replace(stage.tile_plan, tile_height=0, input_channel_group=3,
                              input_spatial_interval=1, tiled_peak_bytes=96,
                              num_tiles=batch * -(-channels // 3) * height * width,
                              original_height=height * width, band_axis=1)
    path = tmp_path / "grouped.tgrs"
    path.write_bytes(emit_binary_bytes(graph))
    plan = read_binary_plan(path.read_bytes())
    assert plan["version"] == 11
    Session = _session()
    golden = np.load(source.with_suffix(".npz"))
    with Session(path, slow_capacity=slow_arena_required(plan)) as session:
        for sample in range(len(golden["input_0"])):
            actual = session.run({session.inputs[0]["name"]: golden["input_0"][sample]})[session.outputs[0]["name"]]
            np.testing.assert_array_equal(actual, golden["output_0"][sample])
