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


def test_plan_reproduces_tflite_micro_outputs(tmp_path):
    """Recorded TFLite Micro reference outputs, compared bit for bit."""
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Installed wheel has no host library")
        pytest.skip("Host library has not been staged in this source checkout")
    from tigris.runtime import Session

    plan_path = tmp_path / "kws.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(KWS), "-m", "16K", "-o", str(plan_path)])
    assert result.exit_code == 0, result.output
    golden = np.load(FIXTURES / "kws_ref_model_tflm.npz")
    _, graphs = tflite._read(KWS.read_bytes())
    _, tensors, inputs, outputs, _ = graphs[0]
    source, result_tensor = tensors[inputs[0]], tensors[outputs[0]]
    with Session(plan_path) as session:
        for quantized, expected in zip(golden["inputs"], golden["outputs"]):
            value = ((quantized.astype(np.float32) - np.float32(source.zero_point[0]))
                     * source.scale[0]).astype(np.float32)
            produced = session.run({"input_1": value})["Identity"].reshape(-1)
            got = np.round(produced / result_tensor.scale[0]).astype(np.int32) + int(result_tensor.zero_point[0])
            np.testing.assert_array_equal(got, expected.astype(np.int32))


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
    logistic = tflite._BUILTIN_OPERATORS.index("LOGISTIC")
    path = tmp_path / "kws_logistic.tflite"
    path.write_bytes(_with_operator_replaced(KWS.read_bytes(), "FULLY_CONNECTED", logistic))

    report = inspect_file(path)
    assert report["format"] == "tflite"
    assert any("LOGISTIC: not supported" in reason for reason in report["unsupported"])
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "16K", "-o", str(tmp_path / "x.tgrs")])
    assert result.exit_code != 0
    assert "LOGISTIC: not supported" in result.output
    assert not (tmp_path / "x.tgrs").exists()


def test_a_truncated_file_is_an_error_not_a_misread():
    data = KWS.read_bytes()[:4096]
    with pytest.raises(ValueError):
        tflite.describe(data)
