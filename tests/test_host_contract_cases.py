"""Contract cases run through the host library; needs the source checkout."""

import numpy as np
import pytest

from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.pipeline import plan_model
from tigris.runtime import RuntimeError, Session


pytestmark = pytest.mark.usefixtures("require_host_library")


def _compile(tmp_path, case, budgets):
    import onnx

    model_path = tmp_path / "case.onnx"
    onnx.save(case.compile_model, model_path)
    graph = plan_model(str(model_path), budgets)
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
    from scripts.contract_cases.reduction import _arg_case
    from tigris.emitters.codegen import generate_c

    case = _arg_case(kind, axis, quantized, keep)
    data = _compile(tmp_path, case, ("4K",))
    plan = read_binary_plan(data)
    output = plan["tensors"][plan["model_outputs"][0]]
    assert output["dtype"] == 6 and output["iface_dtype"] == 7
    for backend in (("reference", "cmsis-nn", "esp-nn") if quantized else ("reference",)):
        for fmt in ("app", "core"):
            assert generate_c(read_binary_plan(data), backend, fmt)
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
    from scripts.contract_cases.movement import _movement_case
    from scripts.crossrepo_contract import _to_runtime_layout

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
    from scripts.contract_cases.movement import _runtime_index_case
    from scripts.crossrepo_contract import _to_runtime_layout
    from tigris.emitters.codegen import generate_c

    case = _runtime_index_case(kind, quantized, variant)
    binary = _compile(tmp_path, case, ("8K",))
    assert generate_c(read_binary_plan(binary), "reference", "core")
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
    from scripts.contract_cases.elementwise import _bool_case
    from scripts.crossrepo_contract import _to_runtime_layout
    from tigris.emitters.codegen import generate_c

    case = _bool_case(kind, quantized, tiled)
    budgets = (case.mem_budget, case.slow_budget) if tiled else (case.mem_budget,)
    data = _compile(tmp_path, case, budgets)
    for backend in ("reference", "cmsis-nn", "esp-nn"):
        for fmt in ("app", "core"):
            assert generate_c(read_binary_plan(data), backend, fmt)
    path = tmp_path / "bool.tgrs"
    path.write_bytes(data)
    expected = _reference(case)
    with Session(path) as session:
        actual = session.run({name: _to_runtime_layout(value) for name, value in case.inputs.items()})["output"]
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind,constant", [
    *[(kind, False) for kind in ("Div", "Max", "Min", "SquaredDifference", "FloorDiv", "FloorMod")],
    *[(kind, True) for kind in ("Max", "Min", "SquaredDifference")]])
def test_rank2_row_band_equals_one_pass(tmp_path, kind, constant):
    import onnx
    from scripts.contract_cases.elementwise import _rank2_row_band_case
    from tigris.analysis.partition_spatial import _stage_is_row_tiled

    case = _rank2_row_band_case(kind, constant=constant)
    model_path = tmp_path / "case.onnx"
    onnx.save(case.compile_model, model_path)
    outputs = []
    for budgets in ((case.mem_budget, case.slow_budget), ("64K",)):
        graph = plan_model(str(model_path), budgets)
        [stage] = graph.stages
        ops = [graph.ops[i] for i in stage.op_indices]
        assert [op.op_type for op in ops] == [kind]
        assert _stage_is_row_tiled(graph, stage, ops)
        tile = stage.tile_plan
        banded = tile is not None and tile.tileable and tile.num_tiles > 1
        assert banded is (budgets[0] == case.mem_budget)
        path = tmp_path / f"{budgets[0]}.tgrs"
        path.write_bytes(emit_binary_bytes(graph))
        with Session(path) as session:
            outputs.append(session.run(case.inputs)["output"])
    assert outputs[0].tobytes() == outputs[1].tobytes()
    np.testing.assert_allclose(outputs[0], _reference(case), rtol=1e-5, atol=1e-5)


def test_int8_interface_is_bit_exact_with_onnx_runtime(tmp_path):
    """With the int8 interface the plan takes the int8 tensor the model's input
    QuantizeLinear makes and returns the one its output DequantizeLinear reads,
    byte for byte as ONNX Runtime computes both from the same float input."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    from scripts.contract_cases.qdq import _int8_interface_case
    from scripts.crossrepo_contract import _to_runtime_layout

    case = _int8_interface_case()
    model_path = tmp_path / "case.onnx"
    onnx.save(case.compile_model, model_path)
    path = tmp_path / "int8.tgrs"
    path.write_bytes(emit_binary_bytes(plan_model(str(model_path), ("4K",), int8_io=True)))

    oracle = onnx.ModelProto()
    oracle.CopyFrom(case.compile_model)
    del oracle.graph.output[:]
    oracle.graph.output.extend([helper.make_tensor_value_info("input_q", TensorProto.INT8, [1, 3, 8, 8]),
                                helper.make_tensor_value_info("output_q", TensorProto.INT8, [1, 4, 8, 8])])
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(oracle.SerializeToString(), options, providers=["CPUExecutionProvider"])
    data = np.random.default_rng(5).normal(scale=8.0, size=(1, 3, 8, 8)).astype(np.float32)
    # Past the input's int8 range at scale 0.25, so the input quantization saturates.
    data[0, 0, 0, :2] = (-100.0, 100.0)
    quantized, expected = reference.run(None, {"input": data})
    with Session(path) as session:
        assert [item["dtype"] for item in session.inputs + session.outputs] == [np.dtype("int8")] * 2
        actual = session.run({"input": _to_runtime_layout(quantized)})["output"]
    assert actual.dtype == np.int8
    assert np.count_nonzero(expected) > 0
    assert actual.tobytes() == _to_runtime_layout(expected).tobytes()
