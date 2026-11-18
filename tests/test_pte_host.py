"""ExecuTorch programs on the host: input layouts, the checkout and cache
checks, `tigris run` on a .pte with a stand-in runner, and, with an
ExecuTorch checkout, the host runner it builds."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from tigris.cli import cli
from tigris.cli.run import read_inputs
from tigris.frontends import pte
from tigris.frontends import pte_cortex_m as cm
from tigris.frontends import pte_export as pe
from tigris.frontends import pte_host as ph

FIXTURES = Path(__file__).parent / "fixtures" / "pte"
# Method input of the bands fixtures: int8 1x3x48x48 in NHWC order.
SPEC = ph.MethodTensor(1, (1, 3, 48, 48), (0, 2, 3, 1))


def _run(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def _linear_offset_changed(data: bytes) -> bytes:
    """The program with the output offset of its last quantized_linear moved
    by one, so its outputs differ from the original's."""
    program = pte.read(data)
    plan = program.execution_plan[0]
    call = cm.kernel_calls(plan)[-1]
    assert call.signature.name == "cortex_m::quantized_linear"
    index = call.args[call.signature.index("output_offset")]
    plan.values[index] = pte.Int(plan.values[index].int_val + 1)
    return pte.write(replace(program, execution_plan=[plan]))


# Input layouts

def test_flat_npy_input_is_taken_as_it_is():
    row = np.arange(SPEC.nbytes, dtype=np.int64).astype(np.int8)
    assert ph.npy_bytes("flat.npy", row, SPEC) == row.tobytes()


def test_npy_input_in_dim_order_is_taken_as_it_is():
    nhwc = np.arange(SPEC.nbytes, dtype=np.int64).astype(np.int8).reshape(1, 48, 48, 3)
    assert ph.npy_bytes("nhwc.npy", nhwc, SPEC) == nhwc.tobytes()
    assert ph.npy_bytes("hwc.npy", nhwc[0], SPEC) == nhwc.tobytes()


def test_npy_input_in_the_method_sizes_is_laid_out_in_dim_order():
    nhwc = np.arange(SPEC.nbytes, dtype=np.int64).astype(np.int8).reshape(1, 48, 48, 3)
    assert ph.npy_bytes("nchw.npy", nhwc.transpose(0, 3, 1, 2), SPEC) == nhwc.tobytes()
    assert ph.npy_bytes("chw.npy", nhwc.transpose(0, 3, 1, 2)[0], SPEC) == nhwc.tobytes()


def test_array_in_the_method_sizes_reads_the_bytes_in_dim_order():
    nhwc = np.arange(SPEC.nbytes, dtype=np.int64).astype(np.int8).reshape(1, 48, 48, 3)
    array = SPEC.array(nhwc.tobytes())
    assert array.shape == (1, 3, 48, 48)
    np.testing.assert_array_equal(array, nhwc.transpose(0, 3, 1, 2))
    assert ph.npy_bytes("x.npy", array, SPEC) == nhwc.tobytes()


def test_npy_input_of_other_shape_is_refused():
    with pytest.raises(ValueError) as error:
        ph.npy_bytes("rows.npy", np.zeros((48, 144), np.int8), SPEC)
    assert str(error.value) == ("rows.npy has shape 48x144; the method input is 1x48x48x3 int8, "
                                "dim order 0,2,3,1; a .npy input has shape 1x48x48x3, shape "
                                "1x3x48x48 as PyTorch shows the tensor, or one dimension")
    spec = ph.MethodTensor(1, (1, 10), (0, 1))
    with pytest.raises(ValueError) as error:
        ph.npy_bytes("x.npy", np.zeros((2, 5), np.int8), spec)
    assert str(error.value) == ("x.npy has shape 2x5; the method input is 1x10 int8; a .npy "
                                "input has shape 1x10 or one dimension")


def test_npy_input_that_reads_both_ways_is_refused():
    spec = ph.MethodTensor(1, (1, 3, 3, 3), (0, 2, 3, 1))
    cube = np.arange(27, dtype=np.int8).reshape(1, 3, 3, 3)
    with pytest.raises(ValueError) as error:
        ph.npy_bytes("cube.npy", cube, spec)
    assert str(error.value) == ("cube.npy has shape 1x3x3x3, which matches the method input "
                                "(1x3x3x3 int8, dim order 0,2,3,1) both in memory order and as "
                                "PyTorch shows the tensor; a .npy input with one dimension or a "
                                ".bin file gives the bytes in memory order")
    # One channel: both readings hold the same bytes.
    spec = ph.MethodTensor(1, (1, 1, 4, 4), (0, 2, 3, 1))
    plane = np.arange(16, dtype=np.int8).reshape(1, 1, 4, 4)
    assert ph.npy_bytes("plane.npy", plane, spec) == plane.tobytes()


def test_npy_input_of_other_dtype_is_refused():
    with pytest.raises(ValueError) as error:
        ph.npy_bytes("float.npy", np.zeros(SPEC.nbytes // 4, np.float32), SPEC)
    assert str(error.value) == ("float.npy holds float32 values; the method input is "
                                "1x48x48x3 int8, dim order 0,2,3,1")


def test_method_input_and_outputs_of_a_program():
    plan = ph.forward_plan(pte.read((FIXTURES / "bands_m7.pte").read_bytes()))
    assert ph.method_input(plan) == SPEC
    assert ph.method_outputs(plan) == [ph.MethodTensor(1, (1, 10), (0, 1))]
    assert ph.MethodTensor(1, (1, 10), (0, 1)).describe() == "1x10 int8"
    assert SPEC.describe() == "1x48x48x3 int8, dim order 0,2,3,1"


def test_a_dim_order_that_is_no_permutation_is_refused():
    plan = ph.forward_plan(pte.read((FIXTURES / "bands_m7.pte").read_bytes()))
    plan.values[plan.inputs[0]].dim_order = [0, 1]
    with pytest.raises(ValueError) as error:
        ph.method_input(plan)
    assert str(error.value) == "method forward has an input with dim order 0,1 for 4 dimensions"


def test_a_tensor_kernels_take_channels_last_is_named_in_memory_order():
    tensor = pte.Tensor(scalar_type=1, sizes=[1, 3, 8, 8], dim_order=[0, 1, 2, 3])
    with pytest.raises(ValueError) as error:
        cm._nhwc(tensor, "cortex_m::quantized_conv2d input")
    assert str(error.value) == ("the cortex_m::quantized_conv2d input is 1x3x8x8 int8, expected "
                                "a channels-last 4-D tensor (dim order 0,2,3,1)")


def test_wrote_line_shows_a_program_output_as_inspect_does():
    from tigris.cli.run import _shapes

    assert _shapes([{"dtype": "int8", "shape": [1, 10], "dim_order": [0, 1]}]) == "1x10 int8"
    assert _shapes([{"dtype": "int8", "shape": [1, 3, 8, 8], "dim_order": [0, 2, 3, 1]},
                    {"dtype": "int8", "shape": [1, 10], "dim_order": [0, 1]}]) == (
        "1x8x8x3 int8, dim order 0,2,3,1; 1x10 int8")
    assert _shapes([{"dtype": "float32", "shape": [1, 4]}, {"dtype": "int8", "shape": []}]) == (
        "1x4 float32, scalar int8")


# Checkout, cache and build output

def _tree(root: Path, version: str = "1.6.0", submodules: bool = True) -> Path:
    (root / "backends" / "cortex_m").mkdir(parents=True)
    (root / "version.txt").write_text(version + "\n")
    if submodules:
        for module in ph.SUBMODULES:
            (root / module).mkdir(parents=True)
            (root / module / "CMakeLists.txt").write_text("")
    return root


def test_checkout_reads_release_commit_and_changes(tmp_path):
    root = _tree(tmp_path / "executorch")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "version.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "t"], check=True)
    tree = ph.checkout(root)
    assert tree.version == "1.6.0" and len(tree.commit) == 40 and tree.changes == ""
    (root / "untracked.txt").write_text("left out")
    assert ph.checkout(root).changes == ""
    (root / "version.txt").write_text("1.6.0\n# a\n")
    changed = ph.checkout(root)
    assert changed.commit == tree.commit and len(changed.changes) == 64
    (root / "version.txt").write_text("1.6.0\n# b\n")
    assert ph.checkout(root).changes not in ("", changed.changes)


@pytest.mark.parametrize("name, version, submodules, message", [
    ("et", "1.6.0", True, "is not a directory named executorch, the name ExecuTorch builds from"),
    ("executorch", "1.5.1", True, "is not an ExecuTorch 1.6 checkout; it holds ExecuTorch 1.5.1"),
    ("executorch", "1.6.0", False, "is not a checkout with its submodules"),
    ("executorch", "1.6.0", True, "is not a git checkout"),
])
def test_checkout_refusals(tmp_path, name, version, submodules, message):
    root = _tree(tmp_path / name, version, submodules)
    with pytest.raises(ValueError, match=message):
        ph.checkout(root)


def test_checkout_whose_git_status_fails_is_refused(tmp_path):
    root = _tree(tmp_path / "executorch")
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q"], check=True)
    module = ph.SUBMODULES[1]
    subprocess.run([*git, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},{module}"],
                   check=True)
    subprocess.run([*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m",
                    "t"], check=True)
    (root / module / ".git").write_text(f"gitdir: {tmp_path / 'gone'}\n")
    with pytest.raises(ValueError) as error:
        ph.checkout(root)
    assert str(error.value).startswith(f"{root}: git status fails: fatal: ")


def test_a_directory_without_a_source_tree_is_refused(tmp_path):
    (tmp_path / "executorch").mkdir()
    with pytest.raises(ValueError, match="is not an ExecuTorch source tree with backends/cortex_m"):
        ph.checkout(tmp_path / "executorch")


def test_runner_key_follows_commit_changes_and_source():
    key = ph.runner_key("a" * 40, b"source")
    assert key == ph.runner_key("a" * 40, b"source") == ph.runner_key("a" * 40, b"source", "")
    assert key != ph.runner_key("b" * 40, b"source")
    assert key != ph.runner_key("a" * 40, b"source changed")
    assert key != ph.runner_key("a" * 40, b"source", "c" * 64)
    assert ph.runner_key("a" * 40, b"source", "c" * 64) != ph.runner_key("a" * 40, b"source",
                                                                         "d" * 64)
    assert key.startswith("a" * 12 + "-")


def _fake_step(name, command, log, cwd=None):
    """A build step that leaves what cached_runner reads."""
    if name == "configure":
        build = Path(command[command.index("-B") + 1])
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CMSIS_NN_VERSION:STRING=v8.0.0\n")
    elif name == "link":
        Path(command[-1]).write_text("runner")
    return ""


@pytest.mark.parametrize("runner, record", [(False, {"executorch": "1.6.0", "commit": "a" * 40,
                                                     "cmsis_nn": "v8.0.0"}),
                                            (True, {}), (True, [])],
                         ids=["no-runner", "empty-record", "list-record"])
def test_a_stale_cache_entry_is_rebuilt(runner, record, tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "missing_modules", lambda python: [])
    monkeypatch.setattr(ph, "_step", _fake_step)
    tree = ph.Checkout(tmp_path / "executorch", "1.6.0", "a" * 40)
    cache = tmp_path / "cache"
    entry = cache / ph.runner_key(tree.commit, ph.RUNNER_SOURCE.read_bytes())
    entry.mkdir(parents=True)
    (entry / "runner.json").write_text(json.dumps(record))
    if runner:
        (entry / ph.RUNNER_NAME).write_text("runner")
    assert ph.cached_runner(tree, cache) is None
    built = ph.build_runner(tree, cache, python="/usr/bin/python3")
    assert built == ph.cached_runner(tree, cache)
    assert (built.executorch, built.cmsis_nn) == ("1.6.0", "v8.0.0")


# The Python that runs ExecuTorch's code generator

def test_missing_modules_are_named():
    assert ph.missing_modules(sys.executable, ("json", "tigris_no_such_module")) == [
        "tigris_no_such_module"]


def test_python_without_codegen_modules_is_refused(monkeypatch):
    monkeypatch.setattr(ph, "missing_modules", lambda python: ["torchgen"])
    with pytest.raises(ValueError) as error:
        ph.check_python("/usr/bin/python3")
    assert str(error.value) == ("/usr/bin/python3 cannot import torchgen; the runner build runs "
                                "ExecuTorch's code generator, which needs a Python with torch and "
                                "PyYAML")


def test_python_that_does_not_start_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"python does not start: "):
        ph.check_python(str(tmp_path / "python"))


def test_build_does_not_start_without_codegen_modules(tmp_path, monkeypatch):
    steps = []
    monkeypatch.setattr(ph, "missing_modules", lambda python: ["torchgen", "yaml"])
    monkeypatch.setattr(ph, "_step", lambda *args, **kwargs: steps.append(args))
    tree = ph.Checkout(tmp_path / "executorch", "1.6.0", "a" * 40)
    with pytest.raises(ValueError, match=r"cannot import torchgen or yaml"):
        ph.build_runner(tree, tmp_path / "cache", python="/usr/bin/python3")
    assert steps == [] and not (tmp_path / "cache").exists()


def test_cache_follows_xdg_cache_home(tmp_path, monkeypatch):
    if sys.platform in ("win32", "darwin"):
        pytest.skip("XDG_CACHE_HOME applies on Linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ph.cache_root() == tmp_path / "tigris" / "pte-runner"


def test_error_lines_keep_each_message_with_its_details():
    output = "\n".join([
        "-- Configuring done",
        "CMake Error at CMakeLists.txt:358 (add_subdirectory):",
        "  The source directory",
        "",
        "    /src/executorch/third-party/x",
        "",
        "  does not contain a CMakeLists.txt file.",
        "-- Generating",
        "ninja: build stopped: subcommand failed.",
    ])
    assert ph.error_lines(output) == [
        "CMake Error at CMakeLists.txt:358 (add_subdirectory):",
        "  The source directory",
        "    /src/executorch/third-party/x",
        "  does not contain a CMakeLists.txt file.",
    ]
    assert ph.error_lines("a\nb\nc", count=2) == ["b", "c"]


# tigris run on a program, with a stand-in runner

_FAKE_RUNNER = f"""#!{sys.executable}
import sys
program, feed, out = sys.argv[1:4]
data = open(feed, "rb").read()
outputs = bytearray()
for k in range(0, len(data), {SPEC.nbytes}):
    outputs += data[k:k + 10]
if "changed" in program:
    outputs[3] ^= 1
open(out, "wb").write(bytes(outputs))
"""
_TITLE = "ExecuTorch 1.6.0 at 4776f71b6f0d, CMSIS-NN v8.0.0 portable kernels"


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    """A runner that copies the first 10 bytes of each input to its output,
    with one byte flipped for a program named *changed*, behind an ExecuTorch
    checkout the checks accept."""
    path = tmp_path / "runner"
    path.write_text(_FAKE_RUNNER)
    path.chmod(0o755)
    runner = ph.Runner(path, "1.6.0", "4776f71b6f0d" + "0" * 28, "v8.0.0")
    tree = ph.Checkout(tmp_path / "executorch", "1.6.0", runner.commit)
    (tmp_path / "executorch").mkdir()
    monkeypatch.setattr(ph, "checkout", lambda root: tree)
    monkeypatch.setattr(ph, "cached_runner", lambda tree, cache=None: runner)
    return tmp_path


def _input(directory: Path, suffix: str = ".bin") -> tuple[Path, bytes]:
    """The fixture's first input, as raw bytes in memory order or as a .npy
    array in the method input's sizes."""
    row = np.load(FIXTURES / "bands_m7.npz")["inputs"][0]
    path = directory / f"x{suffix}"
    if suffix == ".npy":
        np.save(path, row[None].transpose(0, 3, 1, 2))
    else:
        row.tofile(path)
    return path, row.tobytes()


def test_run_writes_the_output_bytes(fake_runner):
    given, data = _input(fake_runner, ".npy")
    out = fake_runner / "y.bin"
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output", out,
                  "--executorch", fake_runner / "executorch")
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == data[:10]
    assert result.output.splitlines() == [
        f"bands_m7.pte   {_TITLE}",
        "",
        f"wrote {out}   1x10 int8",
    ]


def test_a_program_and_its_export_write_the_same_bytes(fake_runner):
    given, _ = _input(fake_runner)
    written = fake_runner / "bands_m7_export.pte"
    written.write_bytes(pe.export((FIXTURES / "bands_m7.pte").read_bytes()).data)
    outputs = []
    for k, program in enumerate((FIXTURES / "bands_m7.pte", written)):
        outputs.append(fake_runner / f"{k}.bin")
        result = _run("run", program, "--input", given, "--output", outputs[-1],
                      "--executorch", fake_runner / "executorch")
        assert result.exit_code == 0, result.output
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_run_json_and_npy_output(fake_runner):
    given, data = _input(fake_runner)
    out = fake_runner / "y.npy"
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", f"0={given}", "--output", out,
                  "--executorch", fake_runner / "executorch", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report == {
        "runtime": "executorch", "runtime_version": "1.6.0",
        "runtime_commit": "4776f71b6f0d" + "0" * 28, "runtime_changes": None,
        "runtime_source": str(fake_runner / "runner"), "backend": "cmsis-nn",
        "cmsis_nn_version": "v8.0.0", "output": str(out),
        "tensors": [{"name": "0", "dtype": "int8", "shape": [1, 10], "dim_order": [0, 1]}],
    }
    np.testing.assert_array_equal(np.load(out), np.frombuffer(data[:10], np.int8)[None])


def test_input_of_other_size_is_refused(fake_runner):
    (fake_runner / "short.bin").write_bytes(bytes(1000))
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", fake_runner / "short.bin",
                  "--output", fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: short.bin holds 1000 bytes; the method input takes 6912 (1x48x48x3 int8, dim "
        "order 0,2,3,1)"]


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04garbage", "npz"],
                         ids=["empty", "zip-magic", "npz"])
def test_input_that_is_not_a_npy_file_is_refused(content, fake_runner):
    path = fake_runner / "x.npy"
    if content == "npz":
        with path.open("wb") as stream:
            np.savez(stream, x=np.zeros((1, 3, 48, 48), np.int8))
    else:
        path.write_bytes(content)
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", path, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    assert result.output.splitlines() == ["Error: x.npy is not a .npy file"]
    interface = [{"name": "x", "dtype": np.dtype(np.int8), "shape": (1, 3, 48, 48),
                  "size_bytes": 6912}]
    with pytest.raises(ValueError, match=r"^x\.npy is not a \.npy file$"):
        read_inputs(interface, [str(path)])


def test_build_message_then_build_failure_with_error_lines(fake_runner, monkeypatch):
    def fail(tree, cache=None, python=None):
        raise ph.BuildError("build", ["x.cpp:3:1: error: expected ';'", "    3 | int x"],
                            Path("/cache/build-k.log"))

    monkeypatch.setattr(ph, "cached_runner", lambda tree, cache=None: None)
    monkeypatch.setattr(ph, "missing_modules", lambda python: [])
    monkeypatch.setattr(ph, "build_runner", fail)
    monkeypatch.setattr(ph, "cache_root", lambda: Path("/cache"))
    given, _ = _input(fake_runner)
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "building the ExecuTorch host runner at 4776f71b6f0d, once; it is kept in /cache",
        "Error: the host runner build stopped at its build step:",
        "  x.cpp:3:1: error: expected ';'",
        "      3 | int x",
        "full log: /cache/build-k.log",
    ]


def test_build_message_writes_the_home_directory_as_tilde(fake_runner, monkeypatch):
    def fail(tree, cache=None, python=None):
        raise ph.BuildError("build", ["x.cpp:3:1: error: expected ';'"], Path("build-k.log"))

    monkeypatch.setattr(ph, "cached_runner", lambda tree, cache=None: None)
    monkeypatch.setattr(ph, "missing_modules", lambda python: [])
    monkeypatch.setattr(ph, "build_runner", fail)
    monkeypatch.setattr(ph, "cache_root", lambda: Path.home() / ".cache" / "tigris" / "pte-runner")
    given, _ = _input(fake_runner)
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.output.splitlines()[0] == (
        "building the ExecuTorch host runner at 4776f71b6f0d, once; it is kept in "
        + str(Path("~/.cache/tigris/pte-runner")))


def test_python_without_codegen_modules_stops_before_the_build(fake_runner, monkeypatch):
    built = []
    monkeypatch.setattr(ph, "cached_runner", lambda tree, cache=None: None)
    monkeypatch.setattr(ph, "missing_modules", lambda python: ["torchgen"])
    monkeypatch.setattr(ph, "build_runner", lambda *args, **kwargs: built.append(args))
    given, _ = _input(fake_runner)
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1, result.output
    assert built == [] and "building" not in result.output
    assert result.output.splitlines() == [
        f"Error: {sys.executable} cannot import torchgen; the runner build runs ExecuTorch's "
        f"code generator, which needs a Python with torch and PyYAML"]


def test_runner_failure_exit_1(fake_runner, tmp_path):
    given, _ = _input(fake_runner)
    (tmp_path / "runner").write_text(f"#!{sys.executable}\nimport sys\nprint('execute failed: "
                                     f"0x12', file=sys.stderr)\nsys.exit(1)\n")
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: the host runner stopped on bands_m7.pte: execute failed: 0x12"]
    assert not (fake_runner / "y.bin").exists()


def test_mve_program_is_refused(fake_runner):
    given, _ = _input(fake_runner)
    result = _run("run", FIXTURES / "bands_m55.pte", "--input", given, "--output",
                  fake_runner / "y.bin", "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: bands_m55.pte was lowered for MVE kernels (Cortex-M55, M85); the host runs "
        "programs lowered for scalar or DSP kernels"]


@pytest.mark.parametrize("name, overload", [("aten::relu", "out"),
                                            ("cortex_m::quantized_conv2d", "default"),
                                            ("cortex_m::quantized_conv2d_nhwc", "out"),
                                            ("cortex_m::quantize_per_tensor", "out")])
def test_operators_export_does_not_rewrite_reach_the_runner(fake_runner, name, overload):
    """run leaves operators outside export's table to the runtime that runs
    the program; only the calls that tell the kernel classes apart decide
    a refusal."""
    program = pte.read((FIXTURES / "bands_m7.pte").read_bytes())
    plan = program.execution_plan[0]
    first = plan.chains[0].instructions[0]
    plan.operators[first.op_index] = replace(plan.operators[first.op_index], name=name,
                                             overload=overload)
    model = fake_runner / "m.pte"
    model.write_bytes(pte.write(replace(program, execution_plan=[plan])))
    given, data = _input(fake_runner)
    out = fake_runner / "y.bin"
    result = _run("run", model, "--input", given, "--output", out,
                  "--executorch", fake_runner / "executorch")
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == data[:10]


def test_truncated_program_fails_to_parse(fake_runner):
    model = fake_runner / "trunc.pte"
    model.write_bytes((FIXTURES / "bands_m7.pte").read_bytes()[:3000])
    given, _ = _input(fake_runner)
    result = _run("run", model, "--input", given, "--output", fake_runner / "y.bin",
                  "--executorch", fake_runner / "executorch")
    assert result.exit_code == 1
    (line,) = result.output.splitlines()
    assert line.startswith("Error: trunc.pte fails to parse: the extended header (program ")
    assert line.endswith("does not match the file of 3000 bytes")


def test_program_without_executorch_is_refused(tmp_path):
    given, _ = _input(tmp_path)
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  tmp_path / "y.bin")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: running an ExecuTorch program needs --executorch, an ExecuTorch 1.6 checkout"]


def test_checkout_error_exit_2(tmp_path):
    given, _ = _input(tmp_path)
    _tree(tmp_path / "et")
    result = _run("run", FIXTURES / "bands_m7.pte", "--input", given, "--output",
                  tmp_path / "y.bin", "--executorch", tmp_path / "et")
    assert result.exit_code == 2
    assert "Invalid value for '--executorch':" in result.output
    assert "is not a directory named executorch" in " ".join(result.output.split())


def test_run_imports_no_torch():
    code = ("import sys; from tigris.cli import cli; import tigris.cli.run, "
            "tigris.frontends.pte_host; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# The host runner, built against an ExecuTorch checkout

@pytest.fixture(scope="module")
def executorch_tree():
    """Needs TIGRIS_EXECUTORCH_SOURCE, an ExecuTorch 1.6 checkout with its
    submodules in a directory named executorch; the first run builds the
    runner into the user's cache, which takes minutes, and needs CMake, a C++
    compiler, network access for CMSIS-NN and a Python with torch."""
    root = os.environ.get("TIGRIS_EXECUTORCH_SOURCE")
    if not root:
        pytest.skip("TIGRIS_EXECUTORCH_SOURCE is not set")
    return Path(root)


def test_host_runner_runs_a_program_and_its_export_to_the_recorded_bytes(executorch_tree,
                                                                         tmp_path):
    """Every recorded input through `tigris run`: the program and its export
    give the recorded output bytes, a changed program others; the runner
    gives the same bytes under two fills of the planned memory."""
    record = np.load(FIXTURES / "bands_m7.npz")
    original = FIXTURES / "bands_m7.pte"
    written = tmp_path / "bands_m7_export.pte"
    written.write_bytes(pe.export(original.read_bytes()).data)
    changed = tmp_path / "bands_m7_changed.pte"
    changed.write_bytes(_linear_offset_changed(original.read_bytes()))
    differ = []
    for k, (row, expected) in enumerate(zip(record["inputs"], record["output_bytes"])):
        row.tofile(tmp_path / f"x{k}.bin")
        for program in (original, written, changed):
            out = tmp_path / f"{program.stem}_{k}.bin"
            result = _run("run", program, "--input", tmp_path / f"x{k}.bin", "--output", out,
                          "--executorch", executorch_tree)
            assert result.exit_code == 0, result.output
            if program is changed:
                differ.append(out.read_bytes() != expected.tobytes())
            else:
                assert out.read_bytes() == expected.tobytes()
    assert any(differ)
    runner = ph.cached_runner(ph.checkout(executorch_tree))
    inputs = [row.tobytes() for row in record["inputs"]]
    for program in (original, written):
        fills = []
        for poison in (0xA5, 0x5A):
            work = tmp_path / f"{program.stem}_{poison:02x}"
            work.mkdir()
            fills.append(ph.run([str(runner.path)], program, inputs, work, poison))
        assert fills[0] == fills[1] == [row.tobytes() for row in record["output_bytes"]]
