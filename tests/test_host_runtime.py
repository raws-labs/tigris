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


def test_bool_passthrough_boundary_selects_data_dispatcher(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    from tigris.emitters.codegen import generate_c

    flag = helper.make_tensor_value_info("flag", TensorProto.BOOL, [4])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    model = helper.make_model(helper.make_graph([helper.make_node("Relu", ["data"], ["output"])],
                                                "boundary", [flag, data], [flag, output]),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model_path = tmp_path / "boundary.onnx"
    onnx.save(model, model_path)
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


def test_session_trace_reports_events_in_order(linear_plan):
    with Session(linear_plan) as session:
        outputs, events, counters = session.trace({"input": np.ones((1, 64), np.float32)})
    assert outputs["output"].sum() == 64
    # The input is placed in slow memory before the stage and the fast pool is reset after it.
    kinds = [event["kind"] for event in events]
    assert (kinds[0], events[0]["pool"], events[0]["stage"]) == ("alloc", "slow", None)
    assert kinds.count("stage_begin") == kinds.count("stage_end") == 1
    assert kinds.index("stage_begin") < kinds.index("stage_end") == len(kinds) - 2
    assert [event["op"] for event in events if event["kind"] == "op"] == [0, 1, 2]
    assert sum(event["bytes"] for event in events if event["kind"] == "load") == counters["load_bytes"]


@pytest.mark.parametrize("axes_input", [False, True])
def test_spatial_mean_without_kept_axes_runs_as_a_global_pool(tmp_path, axes_input):
    """ReduceMean over H and W with keepdims=0, as ShuffleNet and MNASNet end."""
    import onnx
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
    model = helper.make_model(helper.make_graph(
        nodes, "mean", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6])], initializers),
        opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    source = tmp_path / "mean.onnx"
    onnx.save(model, source)
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
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(6)
    nodes = [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
             helper.make_node("Flatten", ["c"], ["flat"]),
             helper.make_node("Gemm", ["flat", "fw"], ["y"], transB=1)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])]
    if also_output:
        outputs.insert(0, helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 120]))
    model = helper.make_model(helper.make_graph(
        nodes, "flatten", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 6, 4])], outputs,
        [numpy_helper.from_array(rng.normal(size=(5, 2, 3, 3)).astype(np.float32), "w"),
         numpy_helper.from_array(rng.normal(size=(3, 120)).astype(np.float32), "fw")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    source = tmp_path / "flatten.onnx"
    onnx.save(model, source)
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
    import onnx
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
    model = helper.make_model(helper.make_graph(
        nodes, "two_encodings", [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("first", "second")],
        [s("xs", 0.05), z("xz"), s("as", 0.02), z("az"), s("bs", 0.07), z("bz")]),
        opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    source = tmp_path / "two.onnx"
    onnx.save(model, source)
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
    import onnx
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
    model = helper.make_model(helper.make_graph(
        nodes, "constant", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
        [s("xs", 0.04), z("xz"), s("ys", 0.05), z("yz"), s("c", constant)]),
        opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    source = tmp_path / "constant.onnx"
    onnx.save(model, source)
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
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(13)
    channels = 6
    params = [numpy_helper.from_array(v.astype(np.float32), n) for n, v in (
        ("gamma", rng.uniform(0.5, 2, channels)), ("beta", rng.normal(size=channels)),
        ("mean", rng.normal(size=channels)), ("var", rng.uniform(0.2, 3, channels)))]
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["sum"]),
         helper.make_node("BatchNormalization", ["sum", "gamma", "beta", "mean", "var"], ["bn"], epsilon=1e-5),
         helper.make_node("Relu", ["bn"], ["output"])], "preact",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, channels, 5, 7]) for n in ("a", "b")],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 5, 7])], params),
        opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    source = tmp_path / "preact.onnx"
    onnx.save(model, source)
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
    import onnx
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
              numpy_helper.from_array(rng.integers(-127, 128, (4, 3, 3, 3)).astype(np.int8), "w"),
              numpy_helper.from_array(np.array(target, np.int64), "target")]
    model = helper.make_model(helper.make_graph(
        nodes, "layout", [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 4, 5])], params),
        opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    source = tmp_path / "layout.onnx"
    onnx.save(model, source)
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
