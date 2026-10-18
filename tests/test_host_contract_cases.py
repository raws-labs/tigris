"""Contract cases run through the host library; needs the source checkout."""

import numpy as np
import pytest

from tigris.cli import _run_pipeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.runtime import RuntimeError, Session


pytestmark = pytest.mark.usefixtures("require_host_library")


def _compile(tmp_path, case, budgets):
    import onnx

    model_path = tmp_path / "case.onnx"
    onnx.save(case.compile_model, model_path)
    graph, _ = _run_pipeline(str(model_path), budgets)
    return emit_binary_bytes(graph)


def _reference(case, inputs=None):
    import onnxruntime as ort
    from scripts.crossrepo_contract import _to_runtime_layout

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(case.reference_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return _to_runtime_layout(reference.run(None, inputs or case.inputs)[0])


@pytest.mark.parametrize("kind,axis,quantized,keep", [("ArgMax", 1, False, False), ("ArgMin", 2, True, True)])
def test_arg_output_interface(tmp_path, kind, axis, quantized, keep):
    from scripts.crossrepo_contract import _arg_case
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.codegen import generate_c

    case = _arg_case(kind, axis, quantized, keep)
    data = _compile(tmp_path, case, ("4K",))
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


@pytest.mark.parametrize("kind,variant,quantized", [("StridedSlice", 2, False), ("StridedSlice", 2, True),
                                                   ("Gather", 0, True), ("MirrorPad", 0, False),
                                                   ("ReverseV2", 0, True), ("DynamicUpdateSlice", 0, False)])
def test_movement_output_is_exact(tmp_path, kind, variant, quantized):
    from scripts.crossrepo_contract import _movement_case, _to_runtime_layout

    case = _movement_case(kind, quantized, variant)
    path = tmp_path / "movement.tgrs"
    path.write_bytes(_compile(tmp_path, case, ("8K",)))
    expected = _reference(case)
    with Session(path) as session:
        actual = session.run({name: _to_runtime_layout(value) for name, value in case.inputs.items()})["output"]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind,variant", [("Gather", 1), ("GatherND", 0), ("DynamicUpdateSlice", 1)])
@pytest.mark.parametrize("quantized", [False, True])
def test_runtime_indices(tmp_path, kind, variant, quantized):
    from scripts.crossrepo_contract import _runtime_index_case, _to_runtime_layout
    from tigris.emitters.codegen import generate_c

    case = _runtime_index_case(kind, quantized, variant)
    binary = _compile(tmp_path, case, ("8K",))
    assert generate_c(binary, "reference", "core")
    plan = tmp_path / "indices.tgrs"
    plan.write_bytes(binary)
    inputs = {name: _to_runtime_layout(value) if value.dtype.kind == "f" else value
              for name, value in case.inputs.items()}
    with Session(plan) as session:
        assert session.run(inputs)["output"].tobytes() == _reference(case).tobytes()
        name = "starts" if kind == "DynamicUpdateSlice" else "indices"
        changed = {key: value.copy() for key, value in case.inputs.items()}
        changed[name].fill(0)
        inputs[name] = changed[name]
        assert session.run(inputs)["output"].tobytes() == _reference(case, changed).tobytes()
        if kind != "DynamicUpdateSlice":
            for invalid in (-1, 2**31 - 1):
                inputs[name].fill(invalid)
                with pytest.raises(RuntimeError):
                    session.run(inputs)


@pytest.mark.parametrize("kind,quantized,tiled", [("Where", True, True), ("Less", True, True),
                                                   ("And", False, False)])
def test_bool_model_boundaries_and_codegen(tmp_path, kind, quantized, tiled):
    from scripts.crossrepo_contract import _bool_case, _to_runtime_layout
    from tigris.emitters.codegen import generate_c

    case = _bool_case(kind, quantized, tiled)
    budgets = (case.mem_budget, case.slow_budget) if tiled else (case.mem_budget,)
    data = _compile(tmp_path, case, budgets)
    for backend in ("reference", "cmsis-nn", "esp-nn"):
        for fmt in ("app", "core"):
            assert generate_c(data, backend, fmt)
    path = tmp_path / "bool.tgrs"
    path.write_bytes(data)
    expected = _reference(case)
    with Session(path) as session:
        actual = session.run({name: _to_runtime_layout(value) for name, value in case.inputs.items()})["output"]
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
