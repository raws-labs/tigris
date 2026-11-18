"""tigris export and tigris analyze on ExecuTorch programs: the report, the
files written, the JSON and header, and the exit codes."""

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tigris import __version__
from tigris.cli import cli
from tigris.cli.export import pte_title
from tigris.frontends import pte
from tigris.frontends import pte_export as pe
from tigris.frontends.pte_cortex_m import DSP, SCALAR
from tigris.frontends.pte_schema import HEADER, HEADER_OFFSET

FIXTURES = Path(__file__).parent / "fixtures" / "pte"
_RAM = "ExecuTorch RAM on Cortex-M, from the program"
_RUNNER = "  plus the runner's own input and output buffers, stack and statics"


def _run(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def _export(name: str, tmp_path: Path, *options):
    out = tmp_path / name.replace(".pte", "_export.pte")
    return _run("export", FIXTURES / name, "-o", out, *options), out


def test_written_program_report(tmp_path):
    result, out = _export("bands_m7.pte", tmp_path)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "bands_m7.pte   ExecuTorch program, 10 instructions, CMSIS-NN DSP kernels (Cortex-M4, "
        "M7, M33, M35P)",
        "",
        "rewrite",
        "  instructions 2-7     6 row bands, output in place",
        "  extra MACs           +42.87%",
        "  instructions after   40",
        "",
        _RAM,
        "  model memory       13.31 KiB   27.56 KiB before   2.07x less; activations and "
        "CMSIS-NN scratch",
        "  method allocator    7.60 KiB    4.11 KiB before   ExecuTorch's records for the method",
        "  temp allocator         448 B       448 B before   needed to load the method",
        "  total              21.35 KiB   32.11 KiB before   1.50x less",
        _RUNNER,
        "",
        "model input memory reused once read, as in the original",
        "",
        f"wrote {out}   23.16 KiB   +4.25 KiB",
    ]
    assert out.read_bytes() == pe.export((FIXTURES / "bands_m7.pte").read_bytes()).data


def test_default_output_name(tmp_path):
    source = tmp_path / "bands.pte"
    source.write_bytes((FIXTURES / "bands_m7.pte").read_bytes())
    result = _run("export", source)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "bands_export.pte").exists()


def test_input_lines(tmp_path):
    from scripts.pte_host_check import input_past_plan

    moved = tmp_path / "moved.pte"
    moved.write_bytes(input_past_plan((FIXTURES / "bands_m7.pte").read_bytes()))
    kept = _run("export", moved, "-o", tmp_path / "kept.pte")
    assert kept.exit_code == 0, kept.output
    assert "model input untouched" in kept.output.splitlines()
    allowed = _run("export", moved, "-o", tmp_path / "allowed.pte", "--allow-input-overwrite")
    assert allowed.exit_code == 0, allowed.output
    assert ("model input memory reused once read (--allow-input-overwrite)"
            in allowed.output.splitlines())


def test_no_gain_writes_an_unchanged_copy(tmp_path):
    result, out = _export("resblock_m7.pte", tmp_path)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "resblock_m7.pte   ExecuTorch program, 6 instructions, CMSIS-NN DSP kernels (Cortex-M4, "
        "M7, M33, M35P)",
        f"no smaller total found; {out} is an unchanged copy",
    ]
    assert out.read_bytes() == (FIXTURES / "resblock_m7.pte").read_bytes()
    gain, _ = _export("resblock_m7.pte", tmp_path, "--require-gain")
    assert gain.exit_code == 3
    assert f"no smaller total found; {out} is an unchanged copy" in gain.output


def test_no_gain_within_the_mac_limit(tmp_path):
    result, out = _export("resblock_m7.pte", tmp_path, "--max-extra-macs", "5%")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1] == (
        f"no smaller total found within --max-extra-macs 5%; {out} is an unchanged copy")
    report_path = tmp_path / "r.json"
    _export("resblock_m7.pte", tmp_path, "--max-extra-macs", "5%", "--json", report_path)
    assert json.loads(report_path.read_text())["reason"] == (
        "no smaller total found within --max-extra-macs 5%")
    dry = _run("analyze", FIXTURES / "resblock_m7.pte", "--max-extra-macs", "0.5")
    assert dry.exit_code == 0, dry.output
    assert dry.output.splitlines()[4] == (
        "  after tigris export   6.42 KiB   no smaller total found within --max-extra-macs 0.5%")


@pytest.mark.parametrize("options, within", [
    (["--method", "4212"], "--method 4212"),
    (["--arena", "64K", "--method", "4212"], "--arena 64K and --method 4212"),
    (["--method", "4212", "--max-extra-macs", "5%"], "--method 4212 and --max-extra-macs 5%"),
])
def test_no_gain_within_size_limits(options, within, tmp_path):
    """Smaller programs exist, but none within the limits, which the original
    meets: the line names the limits, and the program is kept."""
    report_path = tmp_path / "r.json"
    result, out = _export("bands_m7.pte", tmp_path, *options, "--json", report_path)
    assert result.exit_code == 0, result.output
    assert pe.export((FIXTURES / "bands_m7.pte").read_bytes()).written
    assert result.output.splitlines()[1:] == [
        f"no smaller total found within {within}; {out} is an unchanged copy"]
    assert out.read_bytes() == (FIXTURES / "bands_m7.pte").read_bytes()
    report = json.loads(report_path.read_text())
    assert report["written"] is False and report["reason"] == f"no smaller total found within {within}"
    gain, _ = _export("bands_m7.pte", tmp_path, *options, "--require-gain")
    assert gain.exit_code == 3
    assert gain.output.splitlines()[1:] == [
        f"no smaller total found within {within}; {out} is an unchanged copy"]


def test_a_written_program_meets_require_gain(tmp_path):
    result, _ = _export("bands_m7.pte", tmp_path, "--require-gain")
    assert result.exit_code == 0, result.output


def test_limit_not_met(tmp_path):
    result, out = _export("bands_m7.pte", tmp_path, "--arena", "1000", "--method", "4000")
    assert result.exit_code == 3
    assert result.output.splitlines()[1:] == [
        "smallest model memory found is 11.06 KiB, above --arena 1000 (total 22.89 KiB)",
        "smallest method allocator found is 4.11 KiB, above --method 4000 (total 32.11 KiB)",
        f"{out} is an unchanged copy",
    ]
    assert out.read_bytes() == (FIXTURES / "bands_m7.pte").read_bytes()
    report_path = tmp_path / "r.json"
    _export("bands_m7.pte", tmp_path, "--arena", "1000", "--method", "4000", "--json", report_path)
    assert json.loads(report_path.read_text())["reason"] == (
        "smallest model memory found is 11.06 KiB, above --arena 1000 (total 22.89 KiB); "
        "smallest method allocator found is 4.11 KiB, above --method 4000 (total 32.11 KiB)")
    both, _ = _export("bands_m7.pte", tmp_path, "--arena", "20000", "--method", "4212",
                      "--json", report_path)
    assert both.exit_code == 3
    assert both.output.splitlines()[1:] == [
        "no program found meets --arena 20000 and --method 4212 together",
        f"{out} is an unchanged copy"]
    assert json.loads(report_path.read_text())["reason"] == (
        "no program found meets --arena 20000 and --method 4212 together")
    # The smallest model memory found is written when it is the limit.
    smallest, _ = _export("bands_m7.pte", tmp_path, "--arena", "11328")
    assert smallest.exit_code == 0, smallest.output
    assert "  model memory       11.06 KiB" in smallest.output


def test_limits_that_are_met(tmp_path):
    result, _ = _export("bands_m7.pte", tmp_path, "--arena", "20000", "--method", "10K",
                        "--max-extra-macs", "50%")
    assert result.exit_code == 0, result.output
    capped, _ = _export("bands_m7.pte", tmp_path, "--max-extra-macs", "0")
    assert capped.exit_code == 0, capped.output
    assert "  extra MACs           +0.00%" in capped.output.splitlines()


@pytest.mark.parametrize("name, line", [
    ("bands_m55.pte", "warning: lowered for MVE kernels (Cortex-M55, M85); firmware without MVE "
                      "runs its linear layers without their bias"),
    ("chain_m55.pte", "warning: lowered for MVE kernels (Cortex-M55, M85); it runs on firmware "
                      "built with MVE"),
])
def test_mve_warning(name, line, tmp_path):
    result, _ = _export(name, tmp_path)
    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [line]
    assert "CMSIS-NN MVE kernels (Cortex-M55, M85)" in result.stdout.splitlines()[0]


def test_cpu_is_checked(tmp_path):
    result, _ = _export("bands_m7.pte", tmp_path, "--cpu", "M7")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith(
        "CMSIS-NN DSP kernels (Cortex-M4, M7, M33, M35P), checked against --cpu M7")
    refused, out = _export("chain_m55.pte", tmp_path, "--cpu", "M7")
    assert refused.exit_code == 1
    assert refused.stderr.splitlines()[-1] == (
        "Error: chain_m55.pte was lowered for MVE kernels (Cortex-M55, M85); --cpu M7 runs DSP "
        "kernels")
    assert not out.exists()
    unknown, _ = _export("bands_m7.pte", tmp_path, "--cpu", "M5")
    assert unknown.exit_code == 2
    assert "unknown Cortex-M CPU 'M5'; expected one of M0, M0+, M3" in unknown.output


def test_several_classes_are_named_together(monkeypatch):
    found = pe.PteExport(b"", classes=(SCALAR, DSP), instructions_before=5)
    monkeypatch.setenv("NO_COLOR", "1")
    assert pte_title("m.pte", found, None) == (
        "m.pte   ExecuTorch program, 5 instructions, CMSIS-NN scalar or DSP kernels")
    found.classes = (SCALAR,)
    assert pte_title("m.pte", found, "M0+").endswith(
        "CMSIS-NN scalar kernels (Cortex-M0, M0+, M3, M23), checked against --cpu M0+")


def test_refusal_writes_nothing(tmp_path):
    program = pte.read((FIXTURES / "bands_m7.pte").read_bytes())
    program.execution_plan[0].delegates = [pte.BackendDelegate(id="EthosUBackend")]
    source = tmp_path / "m.pte"
    source.write_bytes(pte.write(program))
    out = tmp_path / "out.pte"
    result = _run("export", source, "-o", out, "--header", tmp_path / "m.h",
                  "--json", tmp_path / "m.json")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: m.pte delegates to EthosUBackend; tigris export rewrites programs that run on "
        "cortex_m operators"]
    assert not out.exists() and not (tmp_path / "m.h").exists()
    assert not (tmp_path / "m.json").exists()


def test_layout_refusal(tmp_path):
    # A non-zero byte in the padding after the extended header.
    data = bytearray((FIXTURES / "resblock_align64_m7.pte").read_bytes())
    data[HEADER_OFFSET + HEADER.size + 5] = 1
    source = tmp_path / "m.pte"
    source.write_bytes(bytes(data))
    out = tmp_path / "out.pte"
    result = _run("export", source, "-o", out)
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: m.pte is laid out differently from the programs ExecuTorch 1.6 writes; tigris "
        "export rewrites programs it writes back byte for byte"]
    assert not out.exists()


def test_nhwc_refusal(tmp_path):
    program = pte.read((FIXTURES / "bands_m7.pte").read_bytes())
    plan = program.execution_plan[0]
    plan.operators[plan.chains[0].instructions[0].op_index].name = (
        "cortex_m::quantized_conv2d_nhwc")
    source = tmp_path / "m.pte"
    source.write_bytes(pte.write(program))
    result = _run("export", source, "-o", tmp_path / "out.pte")
    assert result.exit_code == 1
    assert ("Error: m.pte uses the explicit NHWC layout (cortex_m::quantized_conv2d_nhwc), which "
            "tigris export does not rewrite") in result.output.splitlines()


_KEYS = {"runtime", "source", "tigris_version", "input_sha256", "output_sha256", "written",
         "reason", "warnings", "cmsis_class", "cpu", "model_memory_bytes",
         "method_allocator_bytes", "temp_allocator_bytes", "total_bytes", "macs", "extra_macs",
         "pte_bytes", "regions", "input_overwritten", "instruction_map"}


def test_json_report(tmp_path):
    report_path = tmp_path / "r.json"
    result, out = _export("bands_m7.pte", tmp_path, "--json", report_path, "--cpu", "M7")
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    source = (FIXTURES / "bands_m7.pte").read_bytes()
    assert set(report) == _KEYS
    assert report["runtime"] == "executorch" and report["source"] == "program"
    assert report["tigris_version"] == __version__
    assert report["input_sha256"] == hashlib.sha256(source).hexdigest()
    assert report["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert report["written"] is True and report["reason"] is None and report["warnings"] == []
    assert report["cmsis_class"] == ["DSP"] and report["cpu"] == "M7"
    assert report["model_memory_bytes"] == {"before": 28224, "after": 13632}
    assert report["method_allocator_bytes"] == {"before": 4212, "after": 7780}
    assert report["temp_allocator_bytes"] == {"before": 448, "after": 448}
    assert report["total_bytes"] == {"before": 32884, "after": 21860}
    assert report["pte_bytes"] == {"before": len(source), "after": out.stat().st_size}
    assert report["regions"] == [{"first": 2, "last": 7, "bands": 6, "in_place": True}]
    assert report["input_overwritten"] == {"before": True, "after": True}
    written = pte.read(out.read_bytes()).execution_plan[0]
    assert report["macs"] + report["extra_macs"] == pe.program_macs(written)
    assert f"{100 * report['extra_macs'] / report['macs']:+.2f}%" == "+42.87%"
    mapping = report["instruction_map"]
    assert len(mapping) == 40
    assert [m["original"] for m in mapping if m["band"] is None] == [0, 1, 8, 9]
    banded = [m for m in mapping if m["band"] is not None]
    assert {m["band"] for m in banded} == set(range(6))
    assert {m["original"] for m in banded} == set(range(2, 8))
    assert [m["original"] for m in mapping[:2] + mapping[-2:]] == [0, 1, 8, 9]
    # Each instruction runs the operator of the original instruction it maps to.
    original = pte.read(source).execution_plan[0]
    names = [[plan.operators[i.op_index].name for i in plan.chains[0].instructions]
             for plan in (written, original)]
    assert names[0] == [names[1][m["original"]] for m in mapping]


def test_a_written_program_reads_back(tmp_path):
    """analyze prints the figures of a program export wrote, and export
    keeps it as it is."""
    first, written = _export("bands_m7.pte", tmp_path)
    assert first.exit_code == 0, first.output
    dry = _run("analyze", written)
    assert dry.exit_code == 0, dry.output
    assert dry.output.splitlines() == [
        "bands_m7_export.pte   ExecuTorch program, 40 instructions, CMSIS-NN DSP kernels "
        "(Cortex-M4, M7, M33, M35P)",
        "",
        _RAM,
        "  as lowered            21.35 KiB   model memory 13.31 KiB, method 7.60 KiB, temp 448 B",
        "  after tigris export   21.35 KiB   no smaller total found",
        _RUNNER,
    ]
    out = tmp_path / "again.pte"
    report_path = tmp_path / "again.json"
    again = _run("export", written, "-o", out, "--json", report_path)
    assert again.exit_code == 0, again.output
    assert again.output.splitlines() == [
        "bands_m7_export.pte   ExecuTorch program, 40 instructions, CMSIS-NN DSP kernels "
        "(Cortex-M4, M7, M33, M35P)",
        f"no smaller total found; {out} is an unchanged copy",
    ]
    assert out.read_bytes() == written.read_bytes()
    report = json.loads(report_path.read_text())
    assert report["written"] is False and report["reason"] == "no smaller total found"
    assert report["total_bytes"] == {"before": 21860, "after": 21860}


def test_json_report_of_a_copy(tmp_path):
    report_path = tmp_path / "r.json"
    result, out = _export("resblock_m7.pte", tmp_path, "--json", report_path)
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert set(report) == _KEYS
    assert report["written"] is False and report["reason"] == "no smaller total found"
    assert report["output_sha256"] == report["input_sha256"]
    assert report["total_bytes"]["before"] == report["total_bytes"]["after"] == 3648 + 2504 + 420
    assert report["instruction_map"] == [{"original": j, "band": None} for j in range(6)]
    assert report["regions"] == [] and report["extra_macs"] == 0


def test_header(tmp_path):
    header = tmp_path / "pools.h"
    result, out = _export("bands_m7.pte", tmp_path, "--header", header)
    assert result.exit_code == 0, result.output
    plan = pte.read(out.read_bytes()).execution_plan[0]
    offset = plan.values[plan.inputs[0]].allocation_info.offset
    out_offset = plan.values[plan.outputs[0]].allocation_info.offset
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    assert header.read_text().splitlines() == [
        f"/* Pool sizes for {out.name}, from tigris export {__version__} (sha256 {digest}). */",
        "#define TIGRIS_ET_PLANNED_BUFFER_1_BYTES 13632u",
        "#define TIGRIS_ET_METHOD_ALLOCATOR_MIN_BYTES 7780u",
        "#define TIGRIS_ET_TEMP_ALLOCATOR_MIN_BYTES 448u",
        f"#define TIGRIS_ET_INPUT_0_OFFSET {offset}u",
        f"#define TIGRIS_ET_OUTPUT_0_OFFSET {out_offset}u",
    ]


def test_header_of_a_copy(tmp_path):
    header = tmp_path / "pools.h"
    result, out = _export("resblock_m7.pte", tmp_path, "--header", header)
    assert result.exit_code == 0, result.output
    lines = header.read_text().splitlines()
    assert lines[1:4] == ["#define TIGRIS_ET_PLANNED_BUFFER_1_BYTES 3648u",
                          "#define TIGRIS_ET_METHOD_ALLOCATOR_MIN_BYTES 2504u",
                          "#define TIGRIS_ET_TEMP_ALLOCATOR_MIN_BYTES 420u"]
    assert hashlib.sha256(out.read_bytes()).hexdigest()[:16] in lines[0]


@pytest.mark.parametrize("option", [["--target", "lp64"]])
def test_tflite_options_are_a_usage_error(option, tmp_path):
    result, out = _export("bands_m7.pte", tmp_path, *option)
    assert result.exit_code == 2
    assert "Error: --target applies to TFLite models" in result.output
    assert not out.exists()


def test_analyze_dry_run(tmp_path):
    before = {p.name for p in FIXTURES.iterdir()}
    result = _run("analyze", FIXTURES / "bands_m7.pte")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "bands_m7.pte   ExecuTorch program, 10 instructions, CMSIS-NN DSP kernels (Cortex-M4, "
        "M7, M33, M35P)",
        "",
        _RAM,
        "  as lowered            32.11 KiB   model memory 27.56 KiB, method 4.11 KiB, temp 448 B",
        "  after tigris export   21.35 KiB   1.50x less; model memory 13.31 KiB, method 7.60 KiB, "
        "temp 448 B; +42.87% MACs",
        _RUNNER,
    ]
    assert {p.name for p in FIXTURES.iterdir()} == before
    copy = _run("analyze", FIXTURES / "resblock_m7.pte")
    assert copy.exit_code == 0, copy.output
    assert copy.output.splitlines()[3:5] == [
        "  as lowered            6.42 KiB   model memory 3.56 KiB, method 2.45 KiB, temp 420 B",
        "  after tigris export   6.42 KiB   no smaller total found"]


def test_analyze_json_carries_the_export_figures(tmp_path):
    result = _run("analyze", FIXTURES / "bands_m7.pte", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["model"] == {"name": "bands_m7.pte", "format": "pte"}
    assert set(report) == {"report", "version", "model", "executorch_export"}
    figures = report["executorch_export"]
    assert set(figures) == _KEYS
    written = tmp_path / "r.json"
    _export("bands_m7.pte", tmp_path, "--json", written)
    assert figures == json.loads(written.read_text())


@pytest.mark.parametrize("option", [["-m", "64K"], ["-f", "4M"], ["--trace"], ["--verbose"],
                                    ["--input-shape", "x:1x2"], ["--int8-io"]])
def test_analyze_options_of_compiled_models_are_a_usage_error(option):
    result = _run("analyze", FIXTURES / "bands_m7.pte", *option)
    assert result.exit_code == 2
    assert f"Error: {option[0]} applies to ONNX, TFLite and .pt2 models" in result.output


def test_analyze_executorch_options_are_a_usage_error_for_other_models():
    tflite = Path(__file__).parent / "fixtures" / "tflite" / "vww_96_int8.tflite"
    result = _run("analyze", tflite, "-m", "64K", "--cpu", "M7")
    assert result.exit_code == 2
    assert "Error: --cpu applies to ExecuTorch programs" in result.output


def test_analyze_takes_the_export_options():
    result = _run("analyze", FIXTURES / "bands_m7.pte", "--cpu", "M7", "--max-extra-macs", "0")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith("checked against --cpu M7")
    assert result.output.splitlines()[4].endswith("; +0.00% MACs")
    refused = _run("analyze", FIXTURES / "bands_m7.pte", "--cpu", "M55")
    assert refused.exit_code == 1
    assert ("Error: bands_m7.pte was lowered for DSP kernels (Cortex-M4, M7, M33, M35P); --cpu "
            "M55 runs MVE kernels") in refused.output


def test_inspect_program():
    result = _run("inspect", FIXTURES / "bands_m7.pte")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "bands_m7.pte   ExecuTorch program, 10 instructions, CMSIS-NN DSP kernels (Cortex-M4, "
        "M7, M33, M35P)",
        "  input    0   1x48x48x3 int8, dim order 0,2,3,1",
        "  output   0   1x10 int8",
        "",
        _RAM,
        "  model memory       27.56 KiB   activations and CMSIS-NN scratch",
        "  method allocator    4.11 KiB   ExecuTorch's records for the method",
    ]
    report = json.loads(_run("inspect", FIXTURES / "bands_m7.pte", "--json").output)
    assert report["format"] == "pte"
    program = report["program"]
    assert program["cmsis_nn_classes"] == ["DSP"] and program["instructions"] == 10
    assert program["inputs"] == [{"kind": "tensor", "dtype": "int8", "shape": [1, 3, 48, 48],
                                  "dim_order": [0, 2, 3, 1]}]
    assert program["memory"] == {"planned_buffer_bytes": 28224, "method_allocator_bytes": 4212}
    assert report["operator_counts"]["cortex_m::quantized_conv2d"] >= 1


def test_inspect_reads_a_program_export_does_not_write_back(tmp_path):
    # A non-zero byte in the padding after the extended header.
    data = bytearray((FIXTURES / "resblock_align64_m7.pte").read_bytes())
    data[HEADER_OFFSET + HEADER.size + 5] = 1
    source = tmp_path / "m.pte"
    source.write_bytes(bytes(data))
    result = _run("inspect", source)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].startswith("m.pte   ExecuTorch program, ")


@pytest.mark.parametrize("edit, line", [
    (lambda plan: setattr(plan.chains[0].instructions[0], "op_index", 99),
     "Error: Invalid ExecuTorch program: a kernel call names operator 99, which does not exist"),
    (lambda plan: setattr(plan, "inputs", [999]),
     "Error: Invalid ExecuTorch program: the method input is value 999, which does not exist"),
    (lambda plan: setattr(plan, "outputs", [-1]),
     "Error: Invalid ExecuTorch program: the method output is value -1, which does not exist"),
], ids=["operator", "input", "output"])
def test_inspect_names_an_index_outside_the_program(edit, line, tmp_path):
    program = pte.read((FIXTURES / "chain_m7.pte").read_bytes())
    edit(program.execution_plan[0])
    source = tmp_path / "m.pte"
    source.write_bytes(pte.write(program))
    result = _run("inspect", source)
    assert result.exit_code == 1
    assert result.output.splitlines() == [line]


def test_analyze_refuses_a_plan_that_runs_past_its_buffer(tmp_path):
    program = pte.read((FIXTURES / "chain_m7.pte").read_bytes())
    program.execution_plan[0].non_const_buffer_sizes[1] = 100
    source = tmp_path / "m.pte"
    source.write_bytes(pte.write(program))
    result = _run("analyze", source)
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: m.pte has a memory plan that fails the check: value 16 (768 bytes at 2048) runs "
        "past buffer 1 of 100 bytes"]


@pytest.mark.parametrize("command", ["analyze", "export"])
@pytest.mark.parametrize("share", ["nan", "-1", "inf", "1e400"])
def test_a_share_that_is_no_finite_number_is_a_usage_error(command, share, tmp_path):
    out = tmp_path / "o.pte"
    output = ["-o", out] if command == "export" else []
    result = _run(command, FIXTURES / "bands_m7.pte", *output, "--max-extra-macs", share)
    assert result.exit_code == 2
    assert (f"invalid percentage {share!r}: expected a number of 0 or more with an optional %, "
            f"for example 5%") in " ".join(result.output.split())
    assert not out.exists()
