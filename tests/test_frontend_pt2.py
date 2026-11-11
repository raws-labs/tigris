"""Exported PyTorch programs reach the compiler through torch.onnx."""

import importlib
import json
import os
import sys
import zipfile

import pytest
from click.testing import CliRunner

from tigris.cli import cli
from tigris.frontends import pt2
from tigris.inspection import inspect_file


def _require(module: str):
    """Skip without `module`; fail under TIGRIS_REQUIRE_PYTORCH."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if os.environ.get("TIGRIS_REQUIRE_PYTORCH"):
            pytest.fail(f"{module} is not installed")
        pytest.skip(f"{module} is not installed")


def _archive(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def _model(torch):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4)).eval()


@pytest.fixture(scope="module")
def float_program(tmp_path_factory):
    torch = _require("torch")
    _require("onnxscript")
    path = tmp_path_factory.mktemp("float") / "net.pt2"
    torch.export.save(torch.export.export(_model(torch), (torch.randn(1, 3, 16, 16),)), path)
    return path


@pytest.fixture(scope="module")
def quantized_program(tmp_path_factory):
    """The guide's recipe: PT2E with ExecuTorch's Cortex-M quantizer on
    channels-last inputs, saved as a PT2 archive and as its ONNX export."""
    torch = _require("torch")
    _require("onnxscript")
    _require("executorch")
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    torch.manual_seed(1)
    samples = [torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last)
               for _ in range(8)]
    example = (samples[0],)
    graph = torch.export.export(_model(torch), example).module(check_guards=False)
    graph = prepare_pt2e(graph, CortexMQuantizer())
    for x in samples:
        graph(x)
    program = torch.export.export(convert_pt2e(graph), example)
    root = tmp_path_factory.mktemp("quantized")
    (root / "pt2").mkdir()
    (root / "onnx").mkdir()
    torch.export.save(program, root / "pt2" / "net.pt2")
    torch.onnx.export(program, example, dynamo=True, verbose=False).save(str(root / "onnx" / "net.onnx"))
    return root / "pt2" / "net.pt2", root / "onnx" / "net.onnx"


def test_without_torch_the_error_names_the_packages(tmp_path, monkeypatch):
    path = _archive(tmp_path / "net.pt2", {"net/archive_format": b"pt2"})
    monkeypatch.setitem(sys.modules, "torch", None)
    result = CliRunner().invoke(cli, ["analyze", str(path)])
    assert result.exit_code == 1
    assert ("Error: reading net.pt2 needs torch and onnxscript: pip install torch onnxscript"
            in result.output)


def test_a_zip_that_is_no_program_is_not_a_model(tmp_path):
    path = _archive(tmp_path / "net.pt2", {"net/archive_format": b"other"})
    assert not pt2.is_pt2(path)
    result = CliRunner().invoke(cli, ["analyze", str(path)])
    assert result.exit_code == 1
    assert ("net.pt2 is not an ONNX model, a TFLite model, an exported PyTorch program "
            "or a plan") in result.output


def test_an_archive_torch_cannot_load_is_refused(tmp_path):
    _require("torch")
    _require("onnxscript")
    path = _archive(tmp_path / "net.pt2", {"net/archive_format": b"pt2"})
    result = CliRunner().invoke(cli, ["analyze", str(path)])
    assert result.exit_code == 1
    assert "Error: cannot read net.pt2: " in result.output


def test_a_quantized_program_compiles_as_its_onnx_export(quantized_program, tmp_path):
    program, exported = quantized_program
    plans = []
    for model in (program, exported):
        out = tmp_path / f"{model.suffix[1:]}.tgrs"
        result = CliRunner().invoke(cli, ["compile", str(model), "-m", "64K", "--int8-io",
                                          "-o", str(out)])
        assert result.exit_code == 0, result.output
        plans.append(out.read_bytes())
    assert plans[0] == plans[1]


def test_analyze_reports_executorchs_arena(quantized_program):
    program, _ = quantized_program
    result = CliRunner().invoke(cli, ["analyze", str(program), "-m", "64K", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["model"]["format"] == "pt2"
    assert report["model"]["dtype"] == "int8"
    arena = report["executorch"]
    assert arena["reason"] is None and arena["cpu"] == "cortex-m7"
    # It holds at least the largest activation, the first convolution's int8 output.
    assert arena["arena_bytes"] >= 8 * 16 * 16
    assert arena["arena_bytes"] == pt2.executorch_arena(program)[0]
    text = CliRunner().invoke(cli, ["analyze", str(program), "-m", "64K"]).output
    assert "its Cortex-M7 lowering, CMSIS-NN scratch included" in text


def test_without_executorch_the_arena_is_not_measured(float_program, monkeypatch):
    for name in ["executorch", *(name for name in sys.modules if name.startswith("executorch."))]:
        monkeypatch.setitem(sys.modules, name, None)
    assert pt2.executorch_arena(float_program) == (None, "executorch is not installed")
    result = CliRunner().invoke(cli, ["analyze", str(float_program), "-m", "64K"])
    assert result.exit_code == 0, result.output
    assert "ExecuTorch arena" in result.output
    assert "not measured: executorch is not installed" in result.output


def test_inspect_and_run_know_a_program(float_program, tmp_path):
    report = inspect_file(float_program)
    assert report["format"] == "pt2"
    assert report["operator_counts"]["Conv"] == 2
    out = CliRunner().invoke(cli, ["run", str(float_program), "--input", str(tmp_path / "x.npy"),
                                   "--output", str(tmp_path / "y.npy")])
    assert out.exit_code != 0
    assert "not a plan: compile net.pt2 with tigris compile first" in out.output
