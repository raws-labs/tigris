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


def _assert_matches_tflite_micro(model: Path, golden: dict, budget: str, tmp_path: Path):
    """Runs the compiled plan on TFLite Micro's recorded inputs, in the
    model's own dtypes, and compares the outputs bit for bit."""
    Session = _session()
    plan_path = tmp_path / "model.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", budget, "-o", str(plan_path)])
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


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("*.tflite")), ids=lambda p: p.stem)
def test_single_operator_matches_tflite_micro(model, tmp_path):
    assert tflite.unsupported(model.read_bytes()) == []
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256K", tmp_path)


@pytest.mark.parametrize("model", sorted((FIXTURES / "ops").glob("resize_*_down.tflite")), ids=lambda p: p.stem)
def test_resize_downscales_in_height_bands(model, tmp_path):
    _assert_matches_tflite_micro(model, np.load(model.with_suffix(".npz")), "256", tmp_path)
    plan = read_binary_plan((tmp_path / "model.tgrs").read_bytes())
    assert any(tile["num_tiles"] > 1 for tile in plan["tile_plans"])


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


def test_a_strided_slice_with_a_stride_is_refused():
    model = (FIXTURES / "ops" / "strided_slice.tflite").read_bytes()
    _, graphs = tflite._read(model)
    _, tensors, _, _, operators = graphs[0]
    strides = tensors[operators[0].inputs[3]]
    data = bytearray(model)
    offset = model.index(strides.data)
    data[offset + 4:offset + 8] = np.array([2], np.int32).tobytes()
    assert any("only stride-1 slices" in reason for reason in tflite.unsupported(bytes(data)))
