"""Contract cases run through the host library; needs the source checkout."""

import os
from pathlib import Path

import numpy as np
import pytest

from tigris.cli import _run_pipeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.runtime import Session


@pytest.fixture(autouse=True)
def require_library(monkeypatch):
    monkeypatch.delenv("TIGRIS_HOST_LIBRARY", raising=False)
    import tigris
    if not (Path(tigris.__file__).parent / "native" / "manifest.json").exists():
        if os.environ.get("TIGRIS_REQUIRE_HOST"):
            pytest.fail("Host library has not been staged")
        pytest.skip("Host library has not been staged in this source checkout")


@pytest.mark.parametrize("kind", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_arg_output_interface(tmp_path, kind, axis, quantized, keep):
    import onnx
    from scripts.crossrepo_contract import _arg_case
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.codegen import generate_c

    case = _arg_case(kind, axis, quantized, keep)
    model_path = tmp_path / "arg.onnx"
    onnx.save(case.compile_model, model_path)
    graph, _ = _run_pipeline(str(model_path), ("4K",))
    data = emit_binary_bytes(graph)
    plan = read_binary_plan(data)
    output = plan["tensors"][plan["model_outputs"][0]]
    assert output["dtype"] == 6 and output["iface_dtype"] == 7
    for backend in (("reference", "cmsis-nn", "esp-nn") if quantized else ("reference",)):
        for fmt in ("app", "core"):
            assert generate_c(data, backend, fmt)
    path = tmp_path / "arg.tgrs"
    path.write_bytes(data)
    source = case.inputs["input"]
    expected = (np.argmax if kind == "ArgMax" else np.argmin)(source, axis=axis, keepdims=keep)
    with Session(path) as session:
        actual = session.run({"input": np.ascontiguousarray(source.transpose(0, 2, 1))})["output"]
        assert actual.dtype == np.dtype("int64")
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind,variant", [("Gather", 0), ("Gather", 1), ("Gather", 2),
                                         ("GatherND", 0), ("StridedSlice", 0), ("StridedSlice", 1), ("StridedSlice", 2),
                                         ("MirrorPad", 0), ("MirrorPad", 1), ("ReverseV2", 0),
                                         ("EmbeddingLookup", 0), ("DynamicUpdateSlice", 0), ("DynamicUpdateSlice", 1)])
@pytest.mark.parametrize("quantized", [False, True])
def test_movement_output_is_exact(tmp_path, kind, variant, quantized):
    import onnx
    import onnxruntime as ort
    from scripts.crossrepo_contract import _movement_case, _to_runtime_layout

    case = _movement_case(kind, quantized, variant)
    model_path = tmp_path / "movement.onnx"
    onnx.save(case.compile_model, model_path)
    graph, _ = _run_pipeline(str(model_path), ("8K",))
    path = tmp_path / "movement.tgrs"
    path.write_bytes(emit_binary_bytes(graph))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(case.reference_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    expected = _to_runtime_layout(reference.run(None, case.inputs)[0])
    with Session(path) as session:
        actual = session.run({name: _to_runtime_layout(value) for name, value in case.inputs.items()})["output"]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
