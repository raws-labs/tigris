"""Numerical and interface checks against an installed host library."""

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from tigris.cli import cli, _run_pipeline
from tigris.cli.compile import _run_compressed_pipeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris import SCHEMA_VERSION, runtime
from tigris.runtime import Session, runtime_info, runtime_version

from conftest import save_onnx


pytestmark = pytest.mark.usefixtures("require_host_library")


@pytest.fixture
def linear_plan(linear_3op_path, tmp_path):
    graph, _ = _run_pipeline(str(linear_3op_path), ("64K",))
    path = tmp_path / "linear.tgrs"
    path.write_bytes(emit_binary_bytes(graph))
    return path


@pytest.mark.parametrize("path", sorted((Path(__file__).parent / "schema_compat").glob("*.tgrs")),
                         ids=lambda path: path.stem)
def test_runtime_places_inputs_where_the_compiler_says(path):
    """The runtime reads model inputs in the fast pool exactly for the plans
    the compiler sizes without slow memory for them, older schemas included."""
    from tigris.analysis.plan_graph import graph_from_plan
    from tigris.analysis.validation import io_in_fast
    from tigris.emitters.binary.reader import read_binary_plan

    in_fast = io_in_fast(graph_from_plan(read_binary_plan(path.read_bytes())))
    with Session(path) as session:
        inputs = {item["name"]: np.zeros(item["shape"], item["dtype"]) for item in session.inputs}
        _, events, counters = session.trace(inputs)
    allocations = [event["pool"] for event in events if event["kind"] == "alloc"]
    assert allocations[0] == ("fast" if in_fast else "slow")
    assert (counters["load_bytes"] == 0) == in_fast


@pytest.mark.parametrize("path", sorted((Path(__file__).parent / "schema_compat").glob("*.tgrs")),
                         ids=lambda path: path.stem)
def test_a_slow_arena_that_cannot_hold_an_input_is_reported(path):
    """With one byte of slow memory, a plan that reads its inputs in place
    runs and any other plan reports that the arena cannot hold the input."""
    from tigris.analysis.plan_graph import graph_from_plan
    from tigris.analysis.validation import io_in_fast
    from tigris.emitters.binary.reader import read_binary_plan

    in_fast = io_in_fast(graph_from_plan(read_binary_plan(path.read_bytes())))
    with Session(path, slow_capacity=1) as session:
        inputs = {item["name"]: np.zeros(item["shape"], item["dtype"]) for item in session.inputs}
        if in_fast:
            session.run(inputs)
        else:
            with pytest.raises(runtime.RuntimeError, match="^arena cannot hold the tensor$"):
                session.run(inputs)


@pytest.mark.parametrize("budget,rows,bands", [("640", 1, 16), ("1280", 2, 8), ("2688", 4, 4)])
def test_a_strided_stripe_runs_the_bands_its_plan_states(strided_conv_path, tmp_path, budget, rows, bands):
    """The runtime runs the output bands the plan states within the working set it states,
    and they reproduce the unbanded result exactly. At 640 bytes the stage stripes one output
    row at a time; at 2688 bytes it runs 4 bands of 4 rows."""
    from tigris.emitters.binary.reader import read_binary_plan

    results = []
    for size in (budget, "64K"):
        graph, _ = _run_pipeline(str(strided_conv_path), (size,), report_bindings=False)
        path = tmp_path / f"strided-{size}.tgrs"
        path.write_bytes(emit_binary_bytes(graph))
        with Session(path) as session:
            (item,) = session.inputs
            image = np.random.default_rng(0).standard_normal(item["shape"]).astype(np.float32)
            outputs, _, counters = session.trace({item["name"]: image})
        results.append((read_binary_plan(path.read_bytes())["tile_plans"], outputs, counters))
    (banded, outputs, counters), (unbanded, reference, _) = results
    assert [(record["axis"], record["tile_extent"], record["num_tiles"]) for record in banded] == [
        (1, rows, bands)] and not unbanded
    assert counters["tiles"] == bands
    assert outputs.keys() == reference.keys()
    for key in outputs:
        np.testing.assert_array_equal(outputs[key], reference[key])


@pytest.fixture
def override_library(tmp_path):
    directory = Path(runtime.__file__).with_name("native")
    manifest = json.loads((directory / "manifest.json").read_text())
    return Path(shutil.copy2(directory / manifest["library"], tmp_path / manifest["library"]))


def test_override_execution_and_origin(linear_plan, override_library, monkeypatch, tmp_path):
    expected_version = runtime_version()
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", str(override_library))
    source = f"TIGRIS_HOST_LIBRARY={override_library.resolve()}"
    assert runtime_info() == {"version": expected_version, "source": source}
    x = np.arange(-32, 32, dtype=np.float32).reshape(1, 64)
    with Session(linear_plan) as session:
        monkeypatch.delenv("TIGRIS_HOST_LIBRARY")
        assert session.runtime_source == source
        np.testing.assert_array_equal(session.run({"input": x})["output"], np.maximum(x, 0))
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", str(override_library))
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert source in result.output
    assert expected_version in result.output
    np.save(tmp_path / "input.npy", x)
    result = CliRunner().invoke(cli, ["run", str(linear_plan), "--input", str(tmp_path / "input.npy"),
                                     "--output", str(tmp_path / "result.npy"), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["runtime_source"] == source
    assert report["runtime_version"] == expected_version
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), np.maximum(x, 0))


@pytest.mark.parametrize("kind", ["missing", "unloadable", "empty"])
def test_invalid_override_never_falls_back(kind, linear_plan, monkeypatch, tmp_path):
    path = tmp_path / "library"
    if kind == "unloadable":
        path.write_text("Not a shared library")
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", "" if kind == "empty" else str(path))
    with pytest.raises(ValueError, match="Cannot load TIGRIS_HOST_LIBRARY"):
        Session(linear_plan)
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 1
    assert "Cannot load TIGRIS_HOST_LIBRARY" in result.output


def test_override_rejects_wrong_abi(override_library, monkeypatch):
    lib = runtime.ct.CDLL(str(override_library))
    lib.tigris_host_abi = lambda: 999
    monkeypatch.setattr(runtime.ct, "CDLL", lambda path: lib)
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", str(override_library))
    with pytest.raises(ValueError, match="Unsupported host library ABI.*TIGRIS_HOST_LIBRARY"):
        runtime_info()


def test_unresolvable_override_names_variable(monkeypatch):
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", "library")

    def unresolved(path, **kwargs):
        raise RuntimeError("Cannot resolve library path")

    monkeypatch.setattr(Path, "resolve", unresolved)
    with pytest.raises(ValueError, match="Cannot load TIGRIS_HOST_LIBRARY"):
        runtime_info()


def test_override_reports_library_version(linear_plan, override_library, monkeypatch):
    lib = runtime.ct.CDLL(str(override_library))
    lib.tigris_host_version = lambda: b"9.8.7"
    monkeypatch.setattr(runtime.ct, "CDLL", lambda path: lib)
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", str(override_library))
    assert runtime_version() == "9.8.7"
    with Session(linear_plan) as session:
        assert session.runtime_version == "9.8.7"


def test_float_inference_and_repeated_calls(linear_plan):
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert "; bundled" in result.output
    x = np.arange(-32, 32, dtype=np.float32).reshape(1, 64)
    with Session(linear_plan) as session:
        assert session.runtime_version == runtime_version()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: session.run({"input": x}), range(6)))
        for result in results:
            np.testing.assert_array_equal(result["output"], np.maximum(x, 0))
        assert session.memory["fast_peak_bytes"] <= session.memory["fast_capacity_bytes"]
    with pytest.raises(ValueError, match="closed"):
        session.run({"input": x})


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_conv_matches_onnx_runtime(conv_relu_chain_path, qdq_conv_path, tmp_path, quantized, compressed):
    import onnxruntime as ort

    model = qdq_conv_path if quantized else conv_relu_chain_path
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.5, 0.5, reference.get_inputs()[0].shape).astype(np.float32)
    expected = reference.run(None, {reference.get_inputs()[0].name: x})[0]
    if compressed:
        graph, _, _, _ = _run_compressed_pipeline(str(model), ("64K",))
    else:
        graph, _ = _run_pipeline(str(model), ("16K",))
    path = tmp_path / "conv.tgrs"
    path.write_bytes(emit_binary_bytes(graph, compress="lz4" if compressed else None))
    with Session(path) as session:
        inputs = {session.inputs[0]["name"]: x.transpose(0, 2, 3, 1)}
        actual = next(iter(session.run(inputs).values())).transpose(0, 3, 1, 2)
    # Quantized kernels may differ by one output quantization step at rounding boundaries.
    np.testing.assert_allclose(actual, expected, atol=0.100001 if quantized else 1e-5, rtol=0 if quantized else 1e-5)


def test_fixed_int8_reference():
    # A 1x1 int8 Conv of weight 2 at scale 0.25 between float boundaries at scale 0.25.
    path = Path(__file__).parent / "schema_compat" / "schema-v6-interface-dtype.tgrs"
    x = (np.arange(16, dtype=np.float32) * 2 - 16).reshape(1, 4, 4, 1) * 0.25
    with Session(path) as session:
        actual = session.run({"input": x})
    np.testing.assert_array_equal(actual["conv"], x * 0.5)


def test_input_validation_and_recovery(linear_plan):
    with Session(linear_plan) as session:
        for inputs in ({}, {"wrong": np.zeros((1, 64), np.float32)},
                       {"input": np.zeros((64,), np.float32)}, {"input": np.zeros((1, 64), np.float64)}):
            with pytest.raises(ValueError):
                session.run(inputs)
        assert session.run({"input": np.ones((1, 64), np.float32)})["output"].sum() == 64


def test_bad_plan(tmp_path):
    path = tmp_path / "bad.tgrs"
    path.write_bytes(b"TGRS")
    with pytest.raises(ValueError):
        Session(path)


def test_incompatible_schema(linear_plan):
    data = bytearray(linear_plan.read_bytes())
    data[4:8] = (9999).to_bytes(4, "little")
    linear_plan.write_bytes(data)
    with pytest.raises(ValueError, match="version"):
        Session(linear_plan)


def test_compile_refuses_a_plan_the_runtime_loader_refuses(tmp_path, monkeypatch):
    """With the writer's contract check turned off, a Softmax that reads only
    a constant reaches the plan; compile refuses it with the loader's reason
    and writes nothing."""
    import importlib

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from tigris.emitters.binary import writer

    compile_command = importlib.import_module("tigris.cli.compile")

    model = helper.make_model(helper.make_graph(
        [helper.make_node("MatMul", ["input", "w"], ["h"], name="product"),
         helper.make_node("Softmax", ["h"], ["output"], axis=-1, name="scores")],
        "contract", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 8])],
        [numpy_helper.from_array(np.eye(8, dtype=np.float32), "w")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    source = tmp_path / "contract.onnx"
    onnx.save(model, source)

    def pipeline(*args, **kwargs):
        graph, budget = _run_pipeline(*args, **kwargs)
        next(op for op in graph.ops if op.op_type == "Softmax").inputs = ["w"]
        return graph, budget

    monkeypatch.setattr(compile_command, "_run_pipeline", pipeline)
    monkeypatch.setattr(writer, "_contract_violations", lambda *args: [])
    plan = tmp_path / "contract.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 1, result.output
    assert result.output.endswith("Error: no plan written: the runtime loader refuses the plan: "
                                  "operator contract is not executable\n")
    assert not plan.exists()


def test_a_plan_beyond_the_default_build_limits_loads_and_states_its_needs(tmp_path, loop_past_512_tensors_path):
    """The host runtime allows every build limit at its maximum, so a plan
    with more than 512 tensors loads, and the loader states the same needs as
    the compiler's build-limit report."""
    from tigris.inspection import _plan, _plan_requirements

    source, _ = loop_past_512_tensors_path
    plan = tmp_path / "loop.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "256K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    data = plan.read_bytes()
    needs = runtime.check_plan(data)
    assert needs["TIGRIS_MAX_TENSORS"] > 512
    assert needs == {limit["name"]: limit["needed"] for limit in _plan_requirements(_plan(data))["build_limits"]}
    with Session(plan) as session:
        assert session.inputs


def test_the_loader_states_a_chain_s_needs_as_the_compiler_does(tmp_path):
    from tigris.inspection import _plan, _plan_requirements

    source = Path(__file__).parent / "fixtures" / "tflite" / "vww_96_int8.tflite"
    plan = tmp_path / "vww.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "16K", "-m", "8M", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    data = plan.read_bytes()
    needs = runtime.check_plan(data)
    assert needs == {limit["name"]: limit["needed"] for limit in _plan_requirements(_plan(data))["build_limits"]}
    assert needs["TIGRIS_MAX_CHAIN_STAGES"] == 5 and needs["TIGRIS_MAX_SPATIAL_OPS_PER_STAGE"] == 1


def test_compile_without_a_host_runtime_writes_the_plan_unchecked(linear_3op_path, tmp_path, monkeypatch):
    """A pure wheel has no host runtime to load the plan with; the writer's
    contract check is all compile has."""
    def unavailable():
        raise AssertionError("no host runtime is loaded without one installed")

    monkeypatch.setattr(runtime, "_NATIVE", tmp_path / "native")
    monkeypatch.setattr(runtime, "_library", unavailable)
    assert runtime.check_plan(b"TGRS") is None
    plan = tmp_path / "linear.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(linear_3op_path), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    assert plan.exists()


def test_check_plan_names_the_loader_s_refusal(linear_plan):
    data = bytearray(linear_plan.read_bytes())
    data[0] ^= 0xFF
    with pytest.raises(ValueError, match="^the runtime loader refuses the plan: bad magic"):
        runtime.check_plan(bytes(data))


@pytest.fixture
def zoo_plan(linear_plan):
    """The linear plan as a fetched zoo artifact would hold it, with its manifest and receipt."""
    plan = linear_plan.rename(linear_plan.with_name("model.tgrs"))
    data = plan.read_bytes()
    manifest = {
        "format_version": 1, "id": "example", "model": "example", "category": "classification",
        "quantization": "float32", "published_at": "2026-01-01T00:00:00Z", "schema": SCHEMA_VERSION,
        "runtime": {"min": "0.0.1", "max": None}, "backends": ["reference"],
        "memory": {"fast_bytes": 65536, "slow_bytes": 0, "flash_bytes": len(data)},
        "inputs": [{"name": "input", "shape": [1, 64], "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, 64], "dtype": "float32"}],
        "license": "apache-2.0", "compiler": {"version": "0.0.1"}, "source": {"model": "example"},
        "evaluation": {"test_cases": 1},
        "files": [{"path": "model.tgrs", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    receipt = {"artifact_id": "example", "runtime": {"min": "0.0.1", "max": None}}
    return plan, manifest, receipt


def _write(plan, name, value):
    plan.with_name(name).write_text(json.dumps(value))


def _cli_run(plan):
    np.save(plan.with_name("input.npy"), np.zeros((1, 64), np.float32))
    output = plan.with_name("output.npy")
    result = CliRunner().invoke(cli, ["run", str(plan), "--input", str(plan.with_name("input.npy")),
                                      "--output", str(output)])
    return result, output.exists()


def test_zoo_dependency_record(zoo_plan):
    plan, manifest, receipt = zoo_plan
    _write(plan, "manifest.json", manifest)
    with Session(plan):
        pass
    _write(plan, "download.json", receipt)
    with Session(plan):
        pass
    _write(plan, "download.json", dict(receipt, runtime={"min": "999.0.0", "max": None}))
    with pytest.raises(ValueError, match="requires runtime"):
        Session(plan)
    _write(plan, "download.json", dict(receipt, artifact_id="other"))
    with pytest.raises(ValueError, match="belongs to another artifact"):
        Session(plan)
    plan.with_name("download.json").unlink()
    _write(plan, "manifest.json", dict(manifest, files=[dict(manifest["files"][0], sha256="0" * 64)]))
    with pytest.raises(ValueError, match="checksum differs"):
        Session(plan)


@pytest.mark.parametrize("manifest", [
    {"id": "other", "files": [{"path": "other.tgrs", "sha256": "0" * 64}]},
    {}, [], "model.tgrs", {"files": "model.tgrs"}, {"files": ["model.tgrs"]}, {"files": None},
], ids=["other-plan", "empty", "list", "string", "files-string", "files-names", "files-null"])
def test_manifest_not_listing_the_plan_is_ignored(zoo_plan, manifest):
    plan, _, _ = zoo_plan
    _write(plan, "manifest.json", manifest)
    _write(plan, "download.json", {})
    with Session(plan):
        pass
    result, written = _cli_run(plan)
    assert result.exit_code == 0, result.output
    assert written


@pytest.mark.parametrize("field, value, message", [
    ("runtime", None, "Invalid zoo manifest: malformed artifact metadata: 'runtime'"),
    ("runtime", {"min": "0.0.1"}, "Invalid zoo manifest: runtime requires min and max"),
    ("format_version", None, "Invalid zoo manifest: malformed artifact metadata: 'format_version'"),
])
def test_malformed_zoo_manifest_is_named(zoo_plan, field, value, message):
    plan, manifest, receipt = zoo_plan
    manifest = dict(manifest, **{field: value}) if value is not None else \
        {key: item for key, item in manifest.items() if key != field}
    _write(plan, "manifest.json", manifest)
    _write(plan, "download.json", receipt)
    with pytest.raises(ValueError, match=re.escape(message)):
        Session(plan)
    result, written = _cli_run(plan)
    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert isinstance(result.exception, SystemExit) and not written


@pytest.mark.parametrize("receipt, message", [
    ({"artifact_id": "example"}, "runtime requires min and max (null when no upper bound is known)"),
    ([], "runtime requires min and max (null when no upper bound is known)"),
    ({"artifact_id": "example", "runtime": {"min": "latest", "max": None}},
     "runtime versions must be release versions such as 0.9.1"),
])
def test_malformed_download_record_is_named(zoo_plan, receipt, message):
    plan, manifest, _ = zoo_plan
    _write(plan, "manifest.json", manifest)
    _write(plan, "download.json", receipt)
    message = f"Invalid zoo download record: {message}"
    with pytest.raises(ValueError, match=re.escape(message)):
        Session(plan)
    result, written = _cli_run(plan)
    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert isinstance(result.exception, SystemExit) and not written


@pytest.mark.parametrize("suffix", [".bin", ".npy", ".npz"])
def test_cli_execution(linear_plan, tmp_path, suffix):
    x = np.arange(-32, 32, dtype=np.float32).reshape(1, 64)
    input_path = tmp_path / "input.npy"
    np.save(input_path, x)
    output = tmp_path / f"result{suffix}"
    args = ["run", str(linear_plan), "--input", str(input_path), "--output", str(output), "--json"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["runtime_version"] == runtime_version()
    assert json.loads(result.output)["runtime_source"] == "bundled"
    if suffix == ".bin":
        actual = np.fromfile(output, dtype="<f4").reshape(x.shape)
    elif suffix == ".npy":
        actual = np.load(output, allow_pickle=False)
    else:
        with np.load(output, allow_pickle=False) as values:
            actual = values["output"]
    np.testing.assert_array_equal(actual, np.maximum(x, 0))
    assert CliRunner().invoke(cli, args).exit_code == 1


def test_multiple_inputs_and_named_outputs(tmp_path):
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in ("a", "b")]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in ("file", "allow_pickle")]
    nodes = [helper.make_node("Add", ["a", "b"], ["file"]),
             helper.make_node("Sub", ["a", "b"], ["allow_pickle"])]
    source = save_onnx(tmp_path, "two_inputs", nodes, inputs, outputs, opset=17, ir_version=8,
                       filename="model.onnx")
    graph, _ = _run_pipeline(str(source), ("4K",))
    path = tmp_path / "model.tgrs"
    path.write_bytes(emit_binary_bytes(graph))
    a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    b = np.array([[4, 3, 2, 1]], dtype=np.float32)
    a.tofile(tmp_path / "a.bin")
    np.save(tmp_path / "b.npy", b)
    destination = tmp_path / "outputs.npz"
    result = CliRunner().invoke(cli, ["run", str(path), "--input", f"a={tmp_path / 'a.bin'}",
                                     "--input", f"b={tmp_path / 'b.npy'}", "--output", str(destination)])
    assert result.exit_code == 0, result.output
    with np.load(destination, allow_pickle=False) as values:
        np.testing.assert_array_equal(values["file"], a + b)
        np.testing.assert_array_equal(values["allow_pickle"], a - b)


def test_bool_passthrough_boundary_selects_data_dispatcher(tmp_path):
    from onnx import TensorProto, helper
    from tigris.emitters.codegen import generate_c

    flag = helper.make_tensor_value_info("flag", TensorProto.BOOL, [4])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    model_path = save_onnx(tmp_path, "boundary", [helper.make_node("Relu", ["data"], ["output"])],
                           [flag, data], [flag, output], ir_version=8)
    graph, _ = _run_pipeline(str(model_path), ("4K",))
    plan = emit_binary_bytes(graph)
    assert "t->dtype == 9u" in generate_c(plan, "reference")
    path = tmp_path / "boundary.tgrs"
    path.write_bytes(plan)
    flags = np.array([True, False, True, False], np.bool_)
    with Session(path) as session:
        result = session.run({"flag": flags, "data": np.array([-1, 2, -3, 4], np.float32)})
        assert result["flag"].dtype == np.dtype("bool")
        np.testing.assert_array_equal(result["flag"], flags)
        np.testing.assert_array_equal(result["output"], [0, 2, 0, 4])


@pytest.mark.parametrize("compressed", [False, True])
def test_inspect_fast_arena_matches_the_runtime(conv_relu_chain_path, tmp_path, compressed):
    import platform

    from tigris.inspection import inspect_file

    if compressed:
        graph, _, _, _ = _run_compressed_pipeline(str(conv_relu_chain_path), ("64K",))
    else:
        graph, _ = _run_pipeline(str(conv_relu_chain_path), ("16K",))
    path = tmp_path / "conv.tgrs"
    path.write_bytes(emit_binary_bytes(graph, compress="lz4" if compressed else None))
    needs = inspect_file(path)["requirements"]
    assert (needs["decompression_bytes"] > 0) == compressed
    with Session(path) as session:
        required = session.memory["fast_capacity_bytes"]
    # inspect sizes the decompression buffer at the largest tensor alignment,
    # which is the host's own on x86_64.
    assert needs["fast_arena_bytes"] >= required
    if platform.machine() == "x86_64":
        assert needs["fast_arena_bytes"] == required


def test_analyze_trace_runs_the_plan_and_matches_the_counters(conv_relu_chain_path):
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "8K", "--trace", "-v"])
    assert result.exit_code == 0, result.output
    assert "traced on runtime" in result.output
    assert "their bytes equal the runtime's own counters" in result.output
    assert "mismatch" not in result.output
    assert "tile 0" in result.output


def test_analyze_reports_measured_slow_traffic(conv_relu_chain_path):
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "8K", "--trace"])
    moved = next(line for line in result.output.splitlines() if line.strip().startswith("moved"))
    result = CliRunner().invoke(cli, ["analyze", str(conv_relu_chain_path), "-m", "8K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "slow traffic" in line)
    written, read = moved.split(None, 1)[1].replace(" written", "").replace(" read", "").split(", ")
    assert f"per inference: {written} written, {read} read" in row
    report = json.loads(CliRunner().invoke(
        cli, ["analyze", str(conv_relu_chain_path), "-m", "8K", "--json"]).output)
    assert report["slow"]["traffic"]["written"] > 0 and report["slow"]["traffic"]["read"] > 0


def test_analyze_traces_and_measures_a_plan_whose_slow_pool_fits(conv_relu_chain_path):
    model = str(conv_relu_chain_path)
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "16K", "-m", "64K", "--trace"])
    assert result.exit_code == 0, result.output
    assert "their bytes equal the runtime's own counters" in result.output
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "16K", "-m", "64K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "slow traffic" in line)
    assert "per inference:" in row


def test_analyze_measures_a_loop_with_a_fixed_iteration_count():
    model = str(Path(__file__).parent / "fixtures" / "tflite" / "ops" / "float_while_counter.tflite")
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "64K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "slow traffic" in line)
    assert "per inference:" in row
    report = json.loads(CliRunner().invoke(cli, ["analyze", model, "-m", "64K", "--json"]).output)
    assert report["slow"]["traffic"]["written"] > 0 and report["slow"]["traffic"]["read"] > 0


def test_analyze_names_why_the_host_runtime_cannot_run_the_plan(tmp_path):
    """The host runtime sizes its slow arena at twice the bytes of every
    tensor; three 1 GiB tensors exceed its 32-bit capacity, so the session
    is refused and analyze reports the refusal instead of the traffic."""
    from onnx import TensorProto, helper

    shape = [1, 4, 8192, 8192]
    model = str(save_onnx(tmp_path, "large", [helper.make_node("Relu", ["x"], ["h"]),
                                              helper.make_node("Relu", ["h"], ["y"])],
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]))
    result = CliRunner().invoke(cli, ["analyze", model, "-m", "1M"], env={"NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "slow traffic" in line)
    assert row.split() == ["slow", "traffic", "not", "measured:", "Host", "slow", "capacity", "exceeds",
                           "the", "runtime", "limit"]
    report = json.loads(CliRunner().invoke(cli, ["analyze", model, "-m", "1M", "--json"]).output)
    assert report["slow"]["traffic"] is None


def test_trace_runs_an_input_dependent_loop_on_the_given_input(tmp_path):
    ops = Path(__file__).parent / "fixtures" / "tflite" / "ops"
    recorded = np.load(ops / "float_while_two_variables.npz")
    for name, key in (("x", "input_0"), ("y", "input_1")):
        np.save(tmp_path / f"{name}.npy", recorded[key][0])
    result = CliRunner().invoke(cli, ["analyze", str(ops / "float_while_two_variables.tflite"), "-m", "64K",
                                      "--trace", "--input", f"x={tmp_path / 'x.npy'}",
                                      "--input", f"y={tmp_path / 'y.npy'}"])
    assert result.exit_code == 0, result.output
    assert "traced on runtime" in result.output and "given input" in result.output
    assert "mismatch" not in result.output


def test_session_trace_reports_events_in_order(linear_plan):
    with Session(linear_plan) as session:
        outputs, events, counters = session.trace({"input": np.ones((1, 64), np.float32)})
    assert outputs["output"].sum() == 64
    # One stage: the input is allocated in fast first and read there.
    kinds = [event["kind"] for event in events]
    assert (kinds[0], events[0]["pool"], events[0]["stage"]) == ("alloc", "fast", None)
    assert counters["load_bytes"] == 0
    assert kinds.count("stage_begin") == kinds.count("stage_end") == 1
    assert kinds.index("stage_begin") < kinds.index("stage_end") == len(kinds) - 2
    assert [event["op"] for event in events if event["kind"] == "op"] == [0, 1, 2]
    assert sum(event["bytes"] for event in events if event["kind"] == "load") == counters["load_bytes"]


def test_a_trace_that_outgrows_its_buffer_reruns_from_the_session_state(tmp_path, monkeypatch):
    """A trace whose first buffer holds too few events runs again from the
    variables the session held before the call, so its outputs, events and the
    state it leaves are those of a trace whose buffer fits."""
    model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / "float_variable_window.tflite"
    x = np.load(model.with_suffix(".npz"))["input_0"]
    plan = tmp_path / "window.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    with Session(plan) as fits, Session(plan) as retried:
        name, output = fits.inputs[0]["name"], fits.outputs[0]["name"]
        for session in (fits, retried):
            session.run({name: x[0]})
        expected = fits.trace({name: x[1]})
        monkeypatch.setattr(runtime, "_TRACE_EVENTS", 1)
        traced = retried.trace({name: x[1]})
        assert len(traced[1]) > 1
        np.testing.assert_array_equal(traced[0][output], expected[0][output])
        assert traced[1:] == expected[1:]
        np.testing.assert_array_equal(retried.run({name: x[2]})[output], fits.run({name: x[2]})[output])


@pytest.mark.parametrize("axes_input", [False, True])
def test_spatial_mean_without_kept_axes_runs_as_a_global_pool(tmp_path, axes_input):
    """ReduceMean over H and W with keepdims=0, as ShuffleNet and MNASNet end."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.random.default_rng(3).normal(size=(6, 3, 3, 3)).astype(np.float32), "w")
    mean_inputs, opset, attrs = ["conv"], 13, {"axes": [2, 3]}
    initializers = [weight]
    if axes_input:
        initializers.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"))
        mean_inputs, opset, attrs = ["conv", "axes"], 18, {}
    nodes = [helper.make_node("Conv", ["input", "w"], ["conv"], pads=[1, 1, 1, 1]),
             helper.make_node("ReduceMean", mean_inputs, ["output"], keepdims=0, **attrs)]
    source = save_onnx(tmp_path, "mean", nodes,
                       [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
                       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6])],
                       initializers, opset=opset, ir_version=8)
    plan = tmp_path / "mean.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "4K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.random.default_rng(4).uniform(-1, 1, (1, 3, 8, 8)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with Session(plan) as session:
        actual = session.run({"input": x.transpose(0, 2, 3, 1)})["output"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", ["depthwise", "dilated_depthwise", "multiplier", "no_kernel_shape",
                                  "no_kernel_shape_stride_1"])
def test_one_dimensional_convolutions_match_onnx_runtime(tmp_path, form):
    """A grouped 1D convolution runs as a 2D depthwise over a height of one; a
    1D convolution that leaves kernel_shape to its weight is still 1D."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(5)
    channels, length = 6, 20
    out_channels = 12 if form == "multiplier" else channels
    group = 1 if form.startswith("no_kernel_shape") else channels
    in_per_group = channels if form.startswith("no_kernel_shape") else 1
    weight = rng.normal(size=(out_channels, in_per_group, 3)).astype(np.float32)
    attrs = {"group": group, "pads": [2, 2] if form == "dilated_depthwise" else [1, 1]}
    if form == "dilated_depthwise":
        attrs["dilations"] = [2]
    if not form.startswith("no_kernel_shape"):
        attrs["kernel_shape"] = [3]
    elif form == "no_kernel_shape":
        attrs["strides"] = [2]
    nodes = [helper.make_node("Conv", ["input", "w", "b"], ["conv"], **attrs),
             helper.make_node("Relu", ["conv"], ["output"])]
    initializers = [numpy_helper.from_array(weight, "w"),
                    numpy_helper.from_array(rng.normal(size=out_channels).astype(np.float32), "b")]
    model = helper.make_model(helper.make_graph(
        nodes, "conv1d", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, length])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)], initializers),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    source = tmp_path / "conv1d.onnx"
    onnx.save(model, source)
    plan = tmp_path / "conv1d.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "8K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = rng.uniform(-1, 1, (1, channels, length)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with Session(plan) as session:
        actual = session.run({"input": x.transpose(0, 2, 1)})["output"]
    np.testing.assert_allclose(actual.transpose(0, 2, 1), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("also_output", [False, True])
def test_a_flatten_into_a_product_keeps_the_model_order(tmp_path, also_output):
    """The flatten a Gemm reads may stay in stored order, its weights permuted,
    but not when the model also returns it."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(6)
    nodes = [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
             helper.make_node("Flatten", ["c"], ["flat"]),
             helper.make_node("Gemm", ["flat", "fw"], ["y"], transB=1)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])]
    if also_output:
        outputs.insert(0, helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 120]))
    source = save_onnx(tmp_path, "flatten", nodes,
                       [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 6, 4])],
                       outputs,
                       [numpy_helper.from_array(rng.normal(size=(5, 2, 3, 3)).astype(np.float32), "w"),
                        numpy_helper.from_array(rng.normal(size=(3, 120)).astype(np.float32), "fw")],
                       opset=17, ir_version=8)
    plan = tmp_path / "flatten.tgrs"
    assert CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)]).exit_code == 0
    x = rng.uniform(-1, 1, (1, 2, 6, 4)).astype(np.float32)
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    expected = dict(zip([o.name for o in outputs], session.run(None, {"x": x})))
    with Session(plan) as runtime_session:
        actual = runtime_session.run({"x": x.transpose(0, 2, 3, 1)})
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name].reshape(value.shape), value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op_type, auto_pad, stride, size", [
    ("Conv", "SAME_UPPER", 2, 9), ("Conv", "SAME_UPPER", 2, 8), ("Conv", "SAME_LOWER", 2, 9),
    ("Conv", "VALID", 1, 8), ("MaxPool", "SAME_UPPER", 2, 9), ("AveragePool", "SAME_UPPER", 2, 7),
    ("AveragePool", "SAME_LOWER", 1, 6)])
def test_auto_pad_matches_onnx_runtime(tmp_path, op_type, auto_pad, stride, size):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(8)
    inputs, initializers = ["x"], []
    attrs = {"auto_pad": auto_pad, "strides": [stride, stride], "kernel_shape": [3, 3]}
    if op_type == "Conv":
        inputs.append("w")
        initializers.append(numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "w"))
    model = helper.make_model(helper.make_graph(
        [helper.make_node(op_type, inputs, ["y"], **attrs)], "auto_pad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], initializers),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    source = tmp_path / "auto_pad.onnx"
    onnx.save(model, source)
    plan = tmp_path / "auto_pad.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = rng.uniform(-1, 1, (1, 3, size, size)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    with Session(plan) as session:
        actual = session.run({"x": x.transpose(0, 2, 3, 1)})["y"].transpose(0, 3, 1, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("group, out_channels, rank", [(2, 4, 4), (4, 8, 4), (2, 6, 3)])
def test_grouped_convolutions_match_onnx_runtime(tmp_path, group, out_channels, rank):
    """A grouped convolution that is not depthwise runs as one per group."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(9)
    channels = 8
    spatial = [7, 6] if rank == 4 else [11]
    kernel = [3] * (rank - 2)
    weight = rng.normal(size=(out_channels, channels // group, *kernel)).astype(np.float32)
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], group=group, pads=[1] * (2 * (rank - 2)),
                          strides=[2] * (rank - 2))], "grouped",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, *spatial])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [numpy_helper.from_array(weight, "w"),
         numpy_helper.from_array(rng.normal(size=out_channels).astype(np.float32), "b")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    source = tmp_path / "grouped.onnx"
    onnx.save(model, source)
    plan = tmp_path / "grouped.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = rng.uniform(-1, 1, (1, channels, *spatial)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    order = (0, *range(2, rank), 1)
    with Session(plan) as session:
        actual = session.run({"x": x.transpose(order)})["y"]
    np.testing.assert_allclose(actual.transpose(np.argsort(order)), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op_type, size, kernel, stride, pads", [
    ("MaxPool", 7, 3, 2, 0), ("MaxPool", 8, 2, 3, 0), ("AveragePool", 7, 3, 2, 0),
    ("AveragePool", 10, 3, 2, 1), ("MaxPool", 6, 3, 2, 1)])
def test_ceil_mode_pools_match_onnx_runtime(tmp_path, op_type, size, kernel, stride, pads):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    model = helper.make_model(helper.make_graph(
        [helper.make_node(op_type, ["x"], ["y"], kernel_shape=[kernel, kernel], strides=[stride, stride],
                          pads=[pads] * 4, ceil_mode=1)], "ceil",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    source = tmp_path / "ceil.onnx"
    onnx.save(model, source)
    plan = tmp_path / "ceil.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.random.default_rng(10).uniform(-1, 1, (1, 3, size, size)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    with Session(plan) as session:
        actual = session.run({"x": x.transpose(0, 2, 3, 1)})["y"].transpose(0, 3, 1, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_a_requantization_tiles_and_matches_its_untiled_plan(tmp_path):
    """An activation in two encodings: the int8 Quantize between them tiles as a
    pointwise operator, and the tiled plan's outputs equal the untiled plan's."""
    from onnx import TensorProto, helper, numpy_helper

    from tigris.inspection import inspect_file

    s = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.float32), name)  # noqa: E731
    z = lambda name: numpy_helper.from_array(np.array(0, dtype=np.int8), name)  # noqa: E731
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
        helper.make_node("Relu", ["xd"], ["r"]),
        helper.make_node("QuantizeLinear", ["r", "as", "az"], ["rq"]),
        helper.make_node("DequantizeLinear", ["rq", "as", "az"], ["rd"]),
        helper.make_node("QuantizeLinear", ["rd", "as", "az"], ["first_q"]),
        helper.make_node("DequantizeLinear", ["first_q", "as", "az"], ["first"]),
        helper.make_node("QuantizeLinear", ["rd", "bs", "bz"], ["second_q"]),
        helper.make_node("DequantizeLinear", ["second_q", "bs", "bz"], ["second"]),
    ]
    shape = [1, 8, 32, 32]
    source = save_onnx(tmp_path, "two_encodings", nodes,
                       [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                       [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("first", "second")],
                       [s("xs", 0.05), z("xz"), s("as", 0.02), z("az"), s("bs", 0.07), z("bz")],
                       ir_version=8, filename="two.onnx")
    outputs = {}
    for budget in ("4K", "1M"):
        plan = tmp_path / f"two_{budget}.tgrs"
        result = CliRunner().invoke(cli, ["compile", str(source), "-m", budget, "-m", "1M", "-o", str(plan)])
        assert result.exit_code == 0, result.output
        report = inspect_file(plan)["plan"]
        quantize = next(k for k, op in enumerate(report["ops"]) if op["type"] == "Quantize")
        stage = next(st for st in report["stages"] if quantize in st["ops"])
        tiled = stage["tile_plan_idx"] is not None or (stage["chain_len"] or 0) >= 2
        assert tiled == (budget == "4K")
        x = np.random.default_rng(11).uniform(-1, 6, shape).astype(np.float32)
        with Session(plan) as session:
            outputs[budget] = session.run({"input": x.transpose(0, 2, 3, 1)})
    for name in ("first", "second"):
        np.testing.assert_array_equal(outputs["4K"][name], outputs["1M"][name])


@pytest.mark.parametrize("op_type, constant", [("Div", 6.0), ("Mul", 0.5), ("Add", 3.0), ("Sub", 1.25),
                                              ("Mul", [[[[2.0]], [[2.0]], [[2.0]]]])])
def test_a_uniform_float_constant_of_an_int8_operator_matches_onnx_runtime(tmp_path, op_type, constant):
    """A QDQ export can leave a scalar operand in float, as in x * relu6(x + 3) / 6."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    s = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.float32), name)  # noqa: E731
    z = lambda name: numpy_helper.from_array(np.array(0, dtype=np.int8), name)  # noqa: E731
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
        helper.make_node(op_type, ["xd", "c"], ["y"]),
        helper.make_node("QuantizeLinear", ["y", "ys", "yz"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["output"]),
    ]
    source = save_onnx(tmp_path, "constant", nodes,
                       [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
                       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
                       [s("xs", 0.04), z("xz"), s("ys", 0.05), z("yz"), s("c", constant)],
                       ir_version=8)
    plan = tmp_path / "constant.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.random.default_rng(12).uniform(-3, 3, (1, 3, 4, 4)).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with Session(plan) as session:
        actual = session.run({"input": x.transpose(0, 2, 3, 1)})["output"].transpose(0, 3, 1, 2)
    # One output step: the int8 kernels round as TFLite does, ONNX Runtime in float.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.05 + 1e-6)


def test_a_batch_normalization_no_convolution_absorbs_matches_onnx_runtime(tmp_path):
    """Pre-activation networks normalize the sum of a residual Add."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(13)
    channels = 6
    params = [numpy_helper.from_array(v.astype(np.float32), n) for n, v in (
        ("gamma", rng.uniform(0.5, 2, channels)), ("beta", rng.normal(size=channels)),
        ("mean", rng.normal(size=channels)), ("var", rng.uniform(0.2, 3, channels)))]
    source = save_onnx(tmp_path, "preact",
                       [helper.make_node("Add", ["a", "b"], ["sum"]),
                        helper.make_node("BatchNormalization", ["sum", "gamma", "beta", "mean", "var"], ["bn"], epsilon=1e-5),
                        helper.make_node("Relu", ["bn"], ["output"])],
                       [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, channels, 5, 7]) for n in ("a", "b")],
                       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 5, 7])],
                       params, opset=15, ir_version=8)
    plan = tmp_path / "preact.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    a, b = (rng.uniform(-2, 2, (1, channels, 5, 7)).astype(np.float32) for _ in range(2))
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"a": a, "b": b})[0]
    with Session(plan) as session:
        actual = session.run({"a": a.transpose(0, 2, 3, 1), "b": b.transpose(0, 2, 3, 1)})["output"]
    np.testing.assert_allclose(actual.transpose(0, 3, 1, 2), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("steps", [["Reshape"], ["Transpose"], ["Reshape", "Transpose"]])
def test_a_layout_step_before_the_first_quantize_matches_onnx_runtime(tmp_path, steps):
    """Exporters move a float input into NCHW and quantize the result."""
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(14)
    shape = [1, 60] if steps[0] == "Reshape" else [1, 4, 5, 3]
    target = [1, 4, 5, 3] if len(steps) > 1 else [1, 3, 4, 5]
    nodes, x_name = [], "input"
    for index, step in enumerate(steps):
        out = f"moved{index}"
        if step == "Reshape":
            nodes.append(helper.make_node("Reshape", [x_name, "target"], [out]))
        else:
            nodes.append(helper.make_node("Transpose", [x_name], [out], perm=[0, 3, 1, 2]))
        x_name = out
    nodes += [
        helper.make_node("QuantizeLinear", [x_name, "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wd"]),
        helper.make_node("Conv", ["xd", "wd"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["y", "ys", "yz"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["output"]),
    ]
    f = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.float32), name)  # noqa: E731
    z = lambda name: numpy_helper.from_array(np.array(0, dtype=np.int8), name)  # noqa: E731
    params = [f("xs", 0.03), z("xz"), f("ws", 0.02), z("wz"), f("ys", 0.1), z("yz"),
              # 7-bit weights: ONNX Runtime's u8s8 kernels on x86 without VNNI saturate
              # int16 pair sums of full-range weights, so the reference would depend on the CPU.
              numpy_helper.from_array(rng.integers(-63, 64, (4, 3, 3, 3)).astype(np.int8), "w"),
              numpy_helper.from_array(np.array(target, np.int64), "target")]
    source = save_onnx(tmp_path, "layout", nodes,
                       [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 4, 5])],
                       params, ir_version=8)
    plan = tmp_path / "layout.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = rng.uniform(-3, 3, shape).astype(np.float32)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with Session(plan) as session:
        feed = x if len(shape) == 2 else x.transpose(0, 2, 3, 1)
        actual = session.run({"input": feed})["output"].transpose(0, 3, 1, 2)
    # One output step: the int8 kernels round as TFLite does, ONNX Runtime in float.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.1 + 1e-6)


def _fold_case(kind, rng):
    from onnx import TensorProto, helper, numpy_helper

    f = lambda name, v: numpy_helper.from_array(np.asarray(v, dtype=np.float32), name)  # noqa: E731
    node = helper.make_node
    conv = [f("w", rng.normal(size=(4, 3, 3, 3)))]
    if kind == "add_relu":
        return ([node("Add", ["x", "s"], ["y"]), node("Relu", ["y"], ["s_next"])],
                {"x": [1, 8], "s": [1, 8]}, {"y": [1, 8], "s_next": [1, 8]}, [])
    if kind == "conv_relu":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), node("Relu", ["c"], ["y"])],
                {"x": [1, 3, 6, 5]}, {"c": [1, 4, 6, 5], "y": [1, 4, 6, 5]}, conv)
    if kind == "conv_bn":
        params = [f(n, v) for n, v in (("g", rng.uniform(0.5, 2, 4)), ("b", rng.normal(size=4)),
                                       ("m", rng.normal(size=4)), ("v", rng.uniform(0.5, 2, 4)))]
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
                 node("BatchNormalization", ["c", "g", "b", "m", "v"], ["y"])],
                {"x": [1, 3, 6, 5]}, {"c": [1, 4, 6, 5], "y": [1, 4, 6, 5]}, conv + params)
    if kind == "conv_channel_add":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), node("Add", ["c", "bias"], ["y"])],
                {"x": [1, 3, 6, 5]}, {"c": [1, 4, 6, 5], "y": [1, 4, 6, 5]},
                conv + [f("bias", rng.normal(size=(1, 4, 1, 1)))])
    if kind == "pad_conv":
        return ([node("Pad", ["x", "pads"], ["p"]), node("Conv", ["p", "w"], ["y"])],
                {"x": [1, 3, 6, 5]}, {"p": [1, 3, 8, 7], "y": [1, 4, 6, 5]},
                conv + [numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")])
    if kind == "matmul_split":
        return ([node("MatMul", ["x", "w"], ["qkv"]), node("Split", ["qkv", "parts"], ["q", "k", "v"], axis=2)],
                {"x": [1, 5, 8]}, {"qkv": [1, 5, 12], "q": [1, 5, 4], "k": [1, 5, 4], "v": [1, 5, 4]},
                [f("w", rng.normal(size=(8, 12))), numpy_helper.from_array(np.array([4, 4, 4], np.int64), "parts")])
    if kind == "split_parts_out":
        return ([node("MatMul", ["x", "w"], ["qkv"]), node("Split", ["qkv", "parts"], ["q", "k", "v"], axis=2)],
                {"x": [1, 5, 8]}, {"q": [1, 5, 4], "k": [1, 5, 4], "v": [1, 5, 4]},
                [f("w", rng.normal(size=(8, 12))), numpy_helper.from_array(np.array([4, 4, 4], np.int64), "parts")])
    if kind == "split_parts_read":
        return ([node("MatMul", ["x", "w"], ["product"]), node("Add", ["product", "b"], ["qkv"]),
                 node("Split", ["qkv", "parts"], ["q", "k", "v"], axis=2),
                 *(node("Relu", [n], [f"r{n}"]) for n in "qkv")],
                {"x": [1, 5, 8]}, {"rq": [1, 5, 4], "rk": [1, 5, 4], "rv": [1, 5, 4]},
                [f("w", rng.normal(size=(8, 12))), f("b", rng.normal(size=12)),
                 numpy_helper.from_array(np.array([4, 4, 4], np.int64), "parts")])
    if kind == "identity_of_output":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), node("Identity", ["c"], ["y"])],
                {"x": [1, 3, 6, 5]}, {"c": [1, 4, 6, 5], "y": [1, 4, 6, 5]}, conv)
    if kind == "identity_of_tiled_output":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), node("Identity", ["c"], ["y"])],
                {"x": [1, 3, 64, 64]}, {"c": [1, 8, 64, 64], "y": [1, 8, 64, 64]},
                [f("w", rng.normal(size=(8, 3, 3, 3)))])
    head = [node("Relu", ["c"], ["r"]), node("GlobalAveragePool", ["r"], ["gp"]), node("Flatten", ["gp"], ["fl"]),
            node("MatMul", ["fl", "fw"], ["g"])]
    if kind == "identity_of_last_stage_output":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), *head, node("Identity", ["g"], ["y"])],
                {"x": [1, 3, 64, 64]}, {"g": [1, 10], "y": [1, 10]},
                [f("w", rng.normal(size=(8, 3, 3, 3))), f("fw", rng.normal(size=(8, 10)))])
    if kind == "identity_of_earlier_stage_output":
        return ([node("Relu", ["x"], ["s"]), node("Conv", ["s", "w"], ["c"], pads=[1, 1, 1, 1]), *head,
                 node("Identity", ["s"], ["y"])],
                {"x": [1, 3, 32, 32]}, {"s": [1, 3, 32, 32], "g": [1, 10], "y": [1, 3, 32, 32]},
                [f("w", rng.normal(size=(32, 3, 3, 3))), f("fw", rng.normal(size=(32, 10)))])
    if kind == "identity_of_input_the_last_stage_reads":
        return ([node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), *head, node("Add", ["g", "v"], ["s"]),
                 node("Identity", ["v"], ["y"])],
                {"x": [1, 3, 64, 64], "v": [1, 10]}, {"s": [1, 10], "y": [1, 10]},
                [f("w", rng.normal(size=(8, 3, 3, 3))), f("fw", rng.normal(size=(8, 10)))])
    if kind == "identity_of_input":
        return ([node("Identity", ["x"], ["y"]), node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1])],
                {"x": [1, 3, 6, 5]}, {"y": [1, 3, 6, 5], "c": [1, 4, 6, 5]}, conv)
    if kind == "two_identities":
        return ([node("MatMul", ["x", "w"], ["p"]), node("Identity", ["p"], ["y1"]), node("Dropout", ["p"], ["y2"])],
                {"x": [1, 5, 8]}, {"y1": [1, 5, 8], "y2": [1, 5, 8]}, [f("w", rng.normal(size=(8, 8)))])
    z = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.int8), name)  # noqa: E731
    nodes = [node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]), node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
             node("Add", ["xd", "bias"], ["summed"]), node("Relu", ["summed"], ["relu"]),
             node("QuantizeLinear", ["relu", "ys", "yz"], ["yq"]), node("DequantizeLinear", ["yq", "ys", "yz"], ["y"])]
    params = [f("xs", 0.05), z("xz", 0), f("ys", 0.1), z("yz", -8), f("bias", rng.normal(size=(1, 16)))]
    if kind == "quantized_add_relu":
        return nodes, {"x": [1, 16]}, {"summed": [1, 16], "y": [1, 16]}, params
    # Both ends of the output's QuantizeLinear and DequantizeLinear pair, in either order.
    outputs = {"yq": (TensorProto.INT8, [1, 16]), "y": [1, 16]}
    return nodes, {"x": [1, 16]}, dict(reversed(outputs.items())) if kind.endswith("dq_q") else outputs, params


def _check_fold_case(tmp_path, kind, without_copy=False):
    import onnxruntime as ort
    from onnx import TensorProto, helper

    rng = np.random.default_rng(15)
    nodes, inputs, outputs, params = _fold_case(kind, rng)
    if without_copy:
        # The same model without its last node, the Identity that writes y.
        nodes, outputs, kind = nodes[:-1], {n: s for n, s in outputs.items() if n != "y"}, f"{kind}_without_copy"
    source = save_onnx(tmp_path, kind, nodes,
                       [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs.items()],
                       [helper.make_tensor_value_info(n, *(s if isinstance(s, tuple) else (TensorProto.FLOAT, s)))
                        for n, s in outputs.items()], params, opset=17, ir_version=8)
    plan = tmp_path / f"{kind}.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    feed = {n: rng.uniform(-1, 1, s).astype(np.float32) for n, s in inputs.items()}
    expected = dict(zip(outputs, ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
                        .run(None, feed)))
    with Session(plan) as session:
        assert [item["name"] for item in session.outputs] == list(outputs)
        actual = session.run({n: v if v.ndim < 3 else np.moveaxis(v, 1, -1) for n, v in feed.items()})
    # One output step for the int8 case: its kernels round as TFLite does.
    atol = 0.1 + 1e-6 if kind.startswith("quantized") else 1e-5
    for name, value in expected.items():
        got = actual[name] if value.ndim < 3 else np.moveaxis(actual[name], -1, 1)
        assert got.dtype == value.dtype, name
        np.testing.assert_allclose(got, value, rtol=1e-5, atol=1 if value.dtype == np.int8 else atol)
    return plan


@pytest.mark.parametrize("kind", ["add_relu", "conv_relu", "conv_bn", "conv_channel_add", "pad_conv",
                                  "matmul_split", "quantized_add_relu"])
def test_a_fused_intermediate_the_model_returns_matches_onnx_runtime(tmp_path, kind):
    """A fold that would absorb the operator writing a model output leaves it in place."""
    _check_fold_case(tmp_path, kind)


@pytest.mark.parametrize("kind", ["split_parts_out", "split_parts_read"])
def test_a_split_folded_into_its_weight_matches_onnx_runtime(tmp_path, kind):
    """Each part of a split read straight off a product is written by its own product."""
    _check_fold_case(tmp_path, kind)


@pytest.mark.parametrize("kind", ["identity_of_output", "identity_of_input", "two_identities",
                                  "quantized_q_dq", "quantized_dq_q"])
def test_every_declared_output_gets_its_own_tensor(tmp_path, kind):
    """An output the model also returns under another name, or passes an input through to, is lowered to a copy."""
    _check_fold_case(tmp_path, kind)


@pytest.mark.parametrize("kind, own_stage, reload", [
    ("identity_of_tiled_output", True, True),
    ("identity_of_last_stage_output", False, False),
    ("identity_of_earlier_stage_output", False, True),
    ("identity_of_input_the_last_stage_reads", False, False),
])
def test_an_output_copy_shares_the_last_stage_when_it_has_room(tmp_path, kind, own_stage, reload):
    """In a plan of several stages the copy has a slow buffer of its own, gets a
    stage of its own only when the last stage cannot take it, and loads its
    source again unless its stage computes it or already reads it."""
    runs = []
    for without_copy in (True, False):
        plan = _check_fold_case(tmp_path, kind, without_copy=without_copy)
        with Session(plan) as session:
            sizes = {item["name"]: int(np.prod(item["shape"])) * 4 for item in session.outputs}
            _, events, counters = session.trace(
                {item["name"]: np.zeros(item["shape"], np.float32) for item in session.inputs})
        runs.append(([event["path"] for event in events if event["kind"] == "stage_begin"], counters))
    (paths, before), (paths_with_copy, after) = runs
    assert paths_with_copy[:len(paths)] == paths
    assert len(paths_with_copy) - len(paths) == own_stage
    assert after["spill_bytes"] - before["spill_bytes"] == sizes["y"]
    assert after["load_bytes"] - before["load_bytes"] == (sizes["y"] if reload else 0)


# Each case: nodes, outputs, the variable's next value, one step (x, s) -> (outputs, next s).
_STATE_CASES = {
    "next_returned": (
        [("Add", ["x", "s"], ["y"]), ("Identity", ["y"], ["s_out"])], ["y", "s_out"], "s_out",
        lambda x, s: ({"y": x + s}, x + s)),
    "next_declared_first": (
        [("Add", ["x", "s"], ["y"]), ("Identity", ["y"], ["s_out"])], ["s_out", "y"], "s_out",
        lambda x, s: ({"y": x + s}, x + s)),
    "old_and_next_returned": (
        [("Identity", ["s"], ["old"]), ("Add", ["x", "s"], ["y"]), ("Identity", ["y"], ["s_out"])],
        ["old", "y", "s_out"], "s_out", lambda x, s: ({"old": s, "y": x + s}, x + s)),
    "next_is_an_input": (
        [("Add", ["x", "s"], ["y"]), ("Identity", ["x"], ["s_out"])], ["y", "s_out"], "s_out",
        lambda x, s: ({"y": x + s}, x)),
    "unchanged": (
        [("Add", ["x", "s"], ["y"]), ("Identity", ["s"], ["s_out"])], ["y", "s_out"], "s_out",
        lambda x, s: ({"y": x + s}, s)),
    "unchanged_by_name": (
        [("Add", ["x", "s"], ["y"])], ["y", "s"], "s", lambda x, s: ({"y": x + s}, s)),
}


@pytest.mark.parametrize("kind", list(_STATE_CASES))
def test_a_variable_the_model_also_returns_keeps_every_output(tmp_path, kind):
    """A variable's next value can be a tensor the model also returns, an
    input, or the variable itself; the variable and every declared output
    still get their values."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, outputs, following, step = _STATE_CASES[kind]
    info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 8])  # noqa: E731
    initial = np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 8)
    model = helper.make_model(helper.make_graph(
        [helper.make_node(*node) for node in nodes], kind, [info("x"), info("s")],
        [info(name) for name in outputs], [numpy_helper.from_array(initial, "s_initial")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    helper.set_model_props(model, {"tigris.state": json.dumps(
        [{"input": "s", "output": following, "initial": "s_initial"}])})
    source, plan = tmp_path / f"{kind}.onnx", tmp_path / f"{kind}.tgrs"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    rng = np.random.default_rng(16)
    state = initial
    with Session(plan) as session:
        assert [item["name"] for item in session.inputs] == ["x"]
        assert [item["name"] for item in session.outputs] == [n for n in outputs if n != following]
        for _ in range(3):
            x = rng.uniform(-1, 1, (1, 8)).astype(np.float32)
            expected, state = step(x, state)
            actual = session.run({"x": x})
            for name, value in expected.items():
                np.testing.assert_array_equal(actual[name], value, err_msg=name)


def _if_reading_enclosing_values(form):
    """A model whose If branches read values and constants of enclosing graphs,
    and the standard ONNX model ONNX Runtime runs for it."""
    from onnx import TensorProto, helper, numpy_helper

    def info(name, kind=TensorProto.FLOAT, dims=(1, 4)):
        return helper.make_tensor_value_info(name, kind, list(dims))

    def const(name, value, dtype=np.float32):
        return numpy_helper.from_array(np.asarray(value, dtype), name)

    def branch(name, nodes, output, constants=(), inputs=()):
        return helper.make_graph(nodes, name, list(inputs), [output], list(constants))

    def qdq(source, scale, zero, result):
        return [helper.make_node("QuantizeLinear", [source, scale, zero], [f"{result}_q"]),
                helper.make_node("DequantizeLinear", [f"{result}_q", scale, zero], [result])]

    k = np.asarray([[0.5, -1.0, 2.0, 0.25]], np.float32)
    inputs, outputs, constants = [info("x"), info("cond", TensorProto.BOOL, (1,))], [info("y")], []
    if form == "values":
        nodes = [helper.make_node("Relu", ["x"], ["a"]), helper.make_node("Mul", ["x", "k"], ["b"]),
                 helper.make_node("If", ["cond"], ["y"], then_branch=branch(
                     "then", [helper.make_node("Add", ["a", "t_one"], ["t_y"])], info("t_y"),
                     [const("t_one", 1.0)]),
                     else_branch=branch("else", [helper.make_node("Sub", ["b", "e_three"], ["e_y"])],
                                        info("e_y"), [const("e_three", 3.0)]))]
        constants = [const("k", 2.5)]
    elif form == "weight":
        nodes = [helper.make_node("Sigmoid", ["x"], ["act"]),
                 helper.make_node("If", ["cond"], ["y"], then_branch=branch(
                     "then", [helper.make_node("MatMul", ["act", "w"], ["t_y"])], info("t_y")),
                     else_branch=branch("else", [helper.make_node("Add", ["act", "k"], ["e_y"])], info("e_y")))]
        constants = [const("w", np.linspace(-1, 1, 16).reshape(4, 4)), const("k", k)]
    elif form == "constant_node":
        nodes = [helper.make_node("Constant", [], ["k"], value=const("k", k)),
                 helper.make_node("Relu", ["x"], ["act"]),
                 helper.make_node("If", ["cond"], ["y"], then_branch=branch(
                     "then", [helper.make_node("Add", ["act", "k"], ["t_y"])], info("t_y")),
                     else_branch=branch("else", [helper.make_node("Tanh", ["act"], ["e_y"])], info("e_y")))]
    elif form == "nested":
        # The inner If reads act and k from the main graph and t_r from the then branch.
        inner = helper.make_node("If", ["t_large"], ["t_y"], then_branch=branch(
            "inner_then", [helper.make_node("Mul", ["act", "k"], ["i_y"])], info("i_y")),
            else_branch=branch("inner_else", [helper.make_node("Sigmoid", ["t_r"], ["j_y"])], info("j_y")))
        nodes = [helper.make_node("Sigmoid", ["x"], ["act"]),
                 helper.make_node("If", ["cond"], ["y"], then_branch=branch("then", [
                     helper.make_node("Relu", ["act"], ["t_r"]),
                     helper.make_node("Reshape", ["t_r", "rows"], ["t_row"]),
                     helper.make_node("ReduceMax", ["t_row"], ["t_max"], axes=[2], keepdims=0),
                     helper.make_node("Greater", ["t_max", "half"], ["t_large"]), inner], info("t_y")),
                     else_branch=branch("else", [helper.make_node("Tanh", ["act"], ["e_y"])], info("e_y")))]
        constants = [const("k", k), const("rows", [1, 1, 4], np.int64), const("half", 0.5)]
    elif form == "beside_operands":
        def model(tigris):
            x_then, x_else = ("t_x", "e_x") if tigris else ("x", "x")
            return [helper.make_node("Sigmoid", ["x"], ["b"]), helper.make_node(
                "If", ["cond", "x"] if tigris else ["cond"], ["y"], **({"domain": "tigris"} if tigris else {}),
                then_branch=branch("then", [helper.make_node("Mul", [x_then, "t_two"], ["t_y"])], info("t_y"),
                                   [const("t_two", 2.0)], [info("t_x")] if tigris else []),
                else_branch=branch("else", [helper.make_node("Add", [x_else, "b"], ["e_y"])], info("e_y"),
                                   (), [info("e_x")] if tigris else []))]
        nodes, reference = model(True), model(False)
    else:
        # int8: the branches requantize a, which a DequantizeLinear in the main graph gives.
        shape = (1, 8)
        nodes = [helper.make_node("DequantizeLinear", ["image", "in_s", "in_z"], ["xd"]),
                 helper.make_node("Sigmoid", ["xd"], ["sg"]), *qdq("sg", "a_s", "a_z", "a"),
                 helper.make_node("If", ["cond"], ["yf"], then_branch=branch(
                     "then", [*qdq("a", "t_s", "t_z", "t_v"), helper.make_node("Relu", ["t_v"], ["t_r"]),
                              *qdq("t_r", "t_s", "t_z", "t_y")], info("t_y", dims=shape)),
                     else_branch=branch("else", qdq("a", "e_s", "e_z", "e_y"), info("e_y", dims=shape))),
                 helper.make_node("QuantizeLinear", ["yf", "out_s", "out_z"], ["y"])]
        inputs = [info("image", TensorProto.INT8, shape), inputs[1]]
        outputs = [info("y", TensorProto.INT8, shape)]
        constants = [const(name, value, np.int8 if name.endswith("_z") else np.float32) for name, value in (
            ("in_s", 0.04), ("in_z", -2), ("out_s", 0.035), ("out_z", -10), ("a_s", 0.03), ("a_z", -5),
            ("t_s", 0.02), ("t_z", 3), ("e_s", 0.05), ("e_z", -7))]

    def built(graph_nodes):
        opsets = [helper.make_opsetid("", 17)] + (
            [helper.make_opsetid("tigris", 1)] if any(n.domain for n in graph_nodes) else [])
        result = helper.make_model(helper.make_graph(graph_nodes, form, inputs, outputs, constants),
                                   opset_imports=opsets)
        result.ir_version = 9
        return result

    compiled = built(nodes)
    return compiled, (built(reference) if form == "beside_operands" else compiled)


@pytest.mark.parametrize("form", ["values", "weight", "constant_node", "nested", "beside_operands", "int8"])
def test_an_if_whose_branches_read_enclosing_values_matches_onnx_runtime(tmp_path, form):
    """ONNX's If reads its inputs from the enclosing graphs; the compiler passes
    them as operands."""
    import onnx
    import onnxruntime as ort

    compiled, reference = _if_reading_enclosing_values(form)
    source = tmp_path / f"{form}.onnx"
    onnx.save(compiled, source)
    plan = tmp_path / f"{form}.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    rng = np.random.default_rng(15)
    if form == "int8":
        feeds = [{"image": rng.integers(-128, 128, (1, 8)).astype(np.int8)} for _ in range(2)]
    else:
        x = rng.uniform(0.1, 2.0, (1, 4)).astype(np.float32)
        # Both signs, so the nested inner condition takes both branches.
        feeds = [{"x": x}, {"x": -x}]
    expected_session = ort.InferenceSession(reference.SerializeToString(), providers=["CPUExecutionProvider"])
    with Session(plan) as session:
        for feed in feeds:
            for take in (True, False):
                inputs = {**feed, "cond": np.asarray([take])}
                expected = expected_session.run(None, inputs)[0]
                actual = session.run(inputs)["y"]
                if form == "int8":
                    # One output step: the int8 kernels round as TFLite does, ONNX Runtime in float.
                    np.testing.assert_allclose(actual.astype(np.int32), expected.astype(np.int32),
                                               rtol=0, atol=1)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
