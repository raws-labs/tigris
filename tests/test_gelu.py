"""GELU exporter forms and bounded pointwise execution."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.validation import validate_budget
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import Layout
from tigris.pipeline import plan_model
from tigris.runtime import Session

pytestmark = pytest.mark.usefixtures("require_host_library")

FIXTURES = Path(__file__).parent / "fixtures/tflite/ops"


def gelu_model(shape, approximate, form):
    constants = {"half": 0.5, "one": 1.0, "root_two": np.sqrt(2.0),
                 "root_two_pi": np.sqrt(2.0 / np.pi), "cubic_scale": 0.044715, "three": 3.0}
    if form == "native":
        nodes = [helper.make_node("Gelu", ["input"], ["output"], approximate=approximate)]
    else:
        if approximate == "none":
            nodes = [helper.make_node("Div", ["input", "root_two"], ["scaled"]),
                     helper.make_node("Erf", ["scaled"], ["activation"])]
        else:
            nodes = ([helper.make_node("Pow", ["input", "three"], ["cubed"])]
                     if form == "pow" else
                     [helper.make_node("Mul", ["input", "input"], ["squared"]),
                      helper.make_node("Mul", ["squared", "input"], ["cubed"])])
            nodes += [helper.make_node("Mul", ["cubic_scale", "cubed"], ["term"]),
                      helper.make_node("Add", ["input", "term"], ["polynomial"]),
                      helper.make_node("Mul", ["polynomial", "root_two_pi"], ["scaled"]),
                      helper.make_node("Tanh", ["scaled"], ["activation"])]
        nodes.append(helper.make_node("Add", ["activation", "one"], ["gate"]))
        if form == "erf_last_half":
            nodes += [helper.make_node("Mul", ["input", "gate"], ["gated"]),
                      helper.make_node("Mul", ["gated", "half"], ["output"])]
        else:
            nodes += [helper.make_node("Mul", ["input", "half"], ["halved"]),
                      helper.make_node("Mul", ["gate", "halved"], ["output"])]
    graph = helper.make_graph(nodes, "gelu", [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
                              [numpy_helper.from_array(np.array(value, np.float32), name)
                               for name, value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10)
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("shape", [(7, 31), (1, 4, 33), (2, 3, 5, 7)])
@pytest.mark.parametrize("approximate,form", [("none", "native"), ("tanh", "native"),
                                             ("none", "erf"), ("none", "erf_last_half"),
                                             ("tanh", "pow"), ("tanh", "mul")])
def test_onnx_gelu_forms_match_tiled_and_untiled(shape, approximate, form, tmp_path):
    model = gelu_model(shape, approximate, form)
    source = tmp_path / "model.onnx"
    onnx.save(model, source)
    values = np.linspace(-8.0, 8.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, {"input": values})[0]
    produced = []
    for budget in ("256", "1M"):
        ag = plan_model(str(source), (budget, "8M"))
        assert [op.op_type for op in ag.ops] == ["Gelu"]
        assert validate_budget(ag).feasible
        plan = tmp_path / f"{budget}.tgrs"
        binary = emit_binary_bytes(ag)
        plan.write_bytes(binary)
        if budget == "256":
            assert any(t['tileable'] and t['num_tiles'] > 1 for t in read_binary_plan(binary)['tile_plans'])
        order = (0, *range(2, len(shape)), 1) if len(shape) in (3, 4) and ag.tensors['input'].layout is Layout.SPATIAL else tuple(range(len(shape)))
        with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
            actual = session.run({'input': values.transpose(order)})['output'].transpose(np.argsort(order))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        produced.append(actual)
    np.testing.assert_array_equal(*produced)


@pytest.mark.parametrize("name", ["gelu", "gelu_approximate", "gelu_wide", "gelu_approximate_offset",
                                   "float_gelu", "float_gelu_approximate", "float_gelu_wide", "float_gelu_approximate_wide"])
def test_tflite_gelu_matches_builtin_reference(name, tmp_path):
    model = FIXTURES / f"{name}.tflite"
    golden = np.load(model.with_suffix('.npz'))
    assert golden['reference'].item() == 'TFLite BUILTIN_REF'
    if golden['input_0'].dtype == np.int8:
        assert np.unique(golden['input_0']).size == 256
    tight = "2K" if "wide" in name or "offset" in name else "256"
    for budget in (tight, "1M"):
        ag = plan_model(str(model), (budget, '8M'))
        assert any(op.op_type == 'Gelu' for op in ag.ops)
        plan = tmp_path / f'{budget}.tgrs'
        binary = emit_binary_bytes(ag)
        plan.write_bytes(binary)
        if budget == tight:
            assert any(t['tileable'] and t['num_tiles'] > 1 for t in read_binary_plan(binary)['tile_plans'])
        with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
            for x, expected in zip(golden['input_0'], golden['output_0']):
                actual = session.run({session.inputs[0]['name']: x})[session.outputs[0]['name']]
                if expected.dtype == np.float32:
                    tolerance = 4 * np.finfo(np.float32).eps * float(np.max(np.abs(expected)))
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
                else:
                    np.testing.assert_array_equal(actual, expected)
