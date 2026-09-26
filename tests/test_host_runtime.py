"""Numerical and interface checks against an installed host library."""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from tigris.cli import cli, _run_pipeline
from tigris.cli.compile import _run_compressed_pipeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris import runtime
from tigris.runtime import Session, runtime_info, runtime_version


@pytest.fixture(autouse=True)
def require_library(monkeypatch):
    monkeypatch.delenv("TIGRIS_HOST_LIBRARY", raising=False)
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Installed wheel has no host library")
        pytest.skip("Host library has not been staged in this source checkout")


@pytest.fixture
def linear_plan(linear_3op_path, tmp_path):
    graph, _ = _run_pipeline(str(linear_3op_path), ("64K",))
    path = tmp_path / "linear.tgrs"
    path.write_bytes(emit_binary_bytes(graph))
    return path


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
    path = Path(__file__).parent / "schema_compat" / "schema-v3-qdq-conv.tgrs"
    with Session(path) as session:
        actual = session.run({"input": np.arange(16, dtype=np.int8).reshape(1, 4, 4, 1)})
    expected = np.array([1, 0, 1, 0, 2, 0, 2, 0], dtype=np.int8).reshape(1, 2, 2, 2)
    np.testing.assert_array_equal(actual["relu_out"], expected)


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


def test_zoo_dependency_record(linear_plan):
    manifest = {"id": "example", "runtime": {"min": "0.0.1", "max": None},
                "files": [{"path": linear_plan.name, "sha256": hashlib.sha256(linear_plan.read_bytes()).hexdigest()}]}
    linear_plan.with_name("manifest.json").write_text(json.dumps(manifest))
    with Session(linear_plan):
        pass
    receipt = {"artifact_id": "example", "runtime": {"min": "999.0.0", "max": None}}
    linear_plan.with_name("download.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="requires runtime"):
        Session(linear_plan)


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
    import onnx
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in ("a", "b")]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in ("file", "allow_pickle")]
    nodes = [helper.make_node("Add", ["a", "b"], ["file"]),
             helper.make_node("Sub", ["a", "b"], ["allow_pickle"])]
    model = helper.make_model(helper.make_graph(nodes, "two_inputs", inputs, outputs),
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    source = tmp_path / "model.onnx"
    onnx.save(model, source)
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
