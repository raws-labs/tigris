"""Values the graph settles before it runs are computed as ONNX defines them,
in the main graph and in an If branch alike."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.runtime import Session

from conftest import save_onnx

pytestmark = pytest.mark.usefixtures("require_host_library")


def _const(name, value):
    return numpy_helper.from_array(np.asarray(value, np.int64), name)


def _scaled(p, nodes, factor):
    """``x`` times ``factor`` cast to float."""
    return nodes + [helper.make_node("Cast", [factor], [f"{p}f"], to=TensorProto.FLOAT),
                    helper.make_node("Mul", ["x", f"{p}f"], [f"{p}y"])]


def _integer_div(p):
    nodes = [helper.make_node("Shape", ["x"], [f"{p}s"]),
             helper.make_node("Gather", [f"{p}s", f"{p}one"], [f"{p}n"], axis=0),
             helper.make_node("Max", [f"{p}n", f"{p}one"], [f"{p}m"]),
             helper.make_node("Div", [f"{p}m", f"{p}three"], [f"{p}d"])]
    return _scaled(p, nodes, f"{p}d"), [_const(f"{p}one", 1), _const(f"{p}three", 3)], (1, 8)


def _negative_div(p):
    nodes = [helper.make_node("Div", [f"{p}a", f"{p}b"], [f"{p}d"])]
    return _scaled(p, nodes, f"{p}d"), [_const(f"{p}a", -7), _const(f"{p}b", 2)], (1, 4)


def _shape_start(p):
    nodes = [helper.make_node("Shape", ["x"], [f"{p}s"], start=-1)]
    return _scaled(p, nodes, f"{p}s"), [], (1, 4)


def _unsqueeze_negative_axes(p):
    nodes = [helper.make_node("Unsqueeze", [f"{p}v", f"{p}axes"], [f"{p}u"])]
    return (_scaled(p, nodes, f"{p}u"), [_const(f"{p}v", [1, 2, 3]), _const(f"{p}axes", [-1, -2])],
            (1, 3, 1, 1))


def _scalar_shape(p):
    nodes = [helper.make_node("Shape", [f"{p}k"], [f"{p}s"]),
             helper.make_node("Concat", [f"{p}s", f"{p}four"], [f"{p}c"], axis=0)]
    return (_scaled(p, nodes, f"{p}c"), [_const(f"{p}k", 5), _const(f"{p}four", [4])], (1, 4))


CASES = {"integer_div": _integer_div, "negative_div": _negative_div, "shape_start": _shape_start,
         "unsqueeze_negative_axes": _unsqueeze_negative_axes, "scalar_shape": _scalar_shape}


def _model(case, where):
    build = CASES[case]
    shape = build("")[2]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    inputs = [x]
    if where == "graph":
        nodes, constants, _ = build("")
    else:
        def branch(name):
            body, constants, _ = build(f"{name}_")
            out = helper.make_tensor_value_info(f"{name}_y", TensorProto.FLOAT, list(shape))
            return helper.make_graph(body, name, [], [out], constants)

        inputs.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]))
        nodes = [helper.make_node("If", ["cond"], ["y"], then_branch=branch("then"),
                                  else_branch=branch("else"))]
        constants = []
    graph = helper.make_graph(nodes, case, inputs, [y], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# In a branch, Shape of a rank-0 value is typed as a zero-extent vector, which
# the loader refuses before any fold runs.
@pytest.mark.parametrize("case,where", [(case, where) for case in sorted(CASES)
                                        for where in ("graph", "branch")
                                        if (case, where) != ("scalar_shape", "branch")])
def test_static_values_match_onnx_runtime(tmp_path, case, where):
    model = _model(case, where)
    source, plan = tmp_path / f"{case}.onnx", tmp_path / f"{case}.tgrs"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.random.default_rng(3).uniform(0.5, 2.0, CASES[case]("")[2]).astype(np.float32)
    feed = {"x": x} if where == "graph" else {"x": x, "cond": np.asarray([True])}
    expected = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"]).run(None, feed)[0]
    channels_last = {**feed, "x": np.moveaxis(x, 1, -1)} if x.ndim == 4 else feed
    with Session(plan) as session:
        actual = session.run(channels_last)["y"]
    if x.ndim == 4:
        actual = np.moveaxis(actual, -1, 1)
    np.testing.assert_array_equal(actual, expected)


def test_branch_constants_become_constants_and_keep_the_operator_order(tmp_path):
    """The sort places a Constant first, which decides where the Relu goes."""
    from tigris.loaders.onnx.loader import load_model as load_raw

    def branch(name):
        k = numpy_helper.from_array(np.full((1, 4), 2.0, np.float32), f"{name}_kvalue")
        out = helper.make_tensor_value_info(f"{name}_y", TensorProto.FLOAT, [1, 4])
        return helper.make_graph([
            helper.make_node("Constant", [], [f"{name}_k"], value=k),
            helper.make_node("Relu", ["x"], [f"{name}_r"]),
            helper.make_node("Add", ["x", f"{name}_k"], [f"{name}_a"]),
            helper.make_node("Mul", [f"{name}_r", f"{name}_a"], [f"{name}_y"])], name, [], [out])

    save_onnx(tmp_path, "branch_constants",
              [helper.make_node("If", ["cond"], ["y"], then_branch=branch("then"), else_branch=branch("else"))],
              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
               helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])],
              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], opset=17,
              ir_version=8)

    ag = load_raw(tmp_path / "branch_constants.onnx")

    assert len(ag.subgraphs) == 2
    for sub in ag.subgraphs:
        name = sub.ops[0].outputs[0].removesuffix("_r")
        assert [op.op_type for op in sub.ops] == ["Relu", "Add", "Mul"]
        assert [op.step for op in sub.ops] == [0, 1, 2]
        np.testing.assert_array_equal(sub.weight_data[f"{name}_k"], np.full((1, 4), 2.0, np.float32))
        assert sub.tensors[f"{name}_k"].is_constant


def test_an_untyped_constant_takes_the_dtype_of_its_value():
    from tigris.loaders.onnx.loader import _graph_to_ag

    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["k"], value=_const("kvalue", [1, 4])),
         helper.make_node("Reshape", ["x", "k"], ["y"])],
        "untyped", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])

    ag = _graph_to_ag(graph, [helper.make_opsetid("", 17)])

    assert [op.op_type for op in ag.ops] == ["Reshape"]
    assert ag.tensors["k"].dtype == TensorProto.INT64
    assert ag.tensors["k"].shape == (2,)


@pytest.mark.parametrize("kind", ["Mul", "Reshape"])
def test_a_value_folded_at_load_in_an_ir_version_3_model_matches_onnx_runtime(tmp_path, kind):
    """Before IR version 4 an initializer is also a graph input; one the load
    fold adds is declared the same way, or shape inference cannot type the
    MatMul that reads it."""
    weight = np.arange(32, dtype=np.float32).reshape(2, 16) / 32
    if kind == "Mul":
        weight = weight.reshape(16, 2)
        operand = numpy_helper.from_array(np.asarray(2.0, np.float32), "operand")
    else:
        operand = _const("operand", [16, 2])
    constants = [numpy_helper.from_array(weight, "w"), operand]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])] + [
        helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)) for t in constants]
    nodes = [helper.make_node(kind, ["w", "operand"], ["wt"]),
             helper.make_node("MatMul", ["x", "wt"], ["product"]),
             helper.make_node("Relu", ["product"], ["y"])]
    graph = helper.make_graph(nodes, "ir3", inputs,
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 7)])
    model.ir_version = 3
    source, plan = tmp_path / "ir3.onnx", tmp_path / "ir3.tgrs"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.random.default_rng(4).uniform(-1, 1, (1, 16)).astype(np.float32)
    expected = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    with Session(plan) as session:
        np.testing.assert_allclose(session.run({"x": x})["y"], expected, rtol=1e-6, atol=1e-6)


def _i64(value):
    return np.asarray(value, np.int64)


def _f32(value):
    return np.asarray(value, np.float32)


REFERENCE_CASES = [
    ("Div", {}, [_i64(-7), _i64(2)]), ("Div", {}, [_i64([7, -7, 6, -6]), _i64([-2, 2, -4, 4])]),
    ("Div", {}, [_f32([1, -3]), _f32([2, 2])]),
    ("Unsqueeze", {}, [_i64([1, 2, 3]), _i64([-1, -2])]), ("Unsqueeze", {}, [_i64(5), _i64([0])]),
    ("Squeeze", {}, [_i64([[[1, 2, 3]]]), _i64([0, -2])]), ("Squeeze", {}, [_i64([[[1, 2, 3]]])]),
    ("Gather", {"axis": 0}, [_i64([1, 8, 4]), _i64(-1)]), ("Gather", {"axis": 0}, [_i64([1, 8, 4]), _i64([2, 0])]),
    ("Concat", {"axis": 0}, [_i64([]), _i64([4])]),
    ("Slice", {}, [_i64([1, 2, 3, 4]), _i64([-1]), _i64([-5]), _i64([0]), _i64([-1])]),
    ("Slice", {}, [_i64([1, 2, 3, 4]), _i64([1]), _i64([2**62])]),
    ("Slice", {}, [_i64([[1, 2], [3, 4]]), _i64([1]), _i64([2]), _i64([-1])]),
    ("Reshape", {}, [_i64([[1, 2], [3, 4]]), _i64([0, -1])]),
    ("Cast", {"to": TensorProto.FLOAT}, [_i64([2, 3])]), ("Cast", {"to": TensorProto.INT32}, [_f32([2.7, -2.7])]),
    ("Max", {}, [_i64(1), _i64(8), _i64(3)]), ("Min", {}, [_f32([1, 5]), _f32([3, 2])]),
    ("Pow", {}, [_f32([2, 3]), _i64([2, 3])]), ("Sqrt", {}, [_f32([4, 2])]),
    ("Reciprocal", {}, [_f32([4, 3])]), ("Neg", {}, [_i64([1, -2])]),
    ("Add", {}, [_i64([1]), _i64(2)]), ("Sub", {}, [_i64(1), _i64(2)]), ("Mul", {}, [_f32(1.5), _f32([2, 3])]),
]


@pytest.mark.parametrize("kind,attrs,values", REFERENCE_CASES,
                         ids=[f"{case[0]}{i}" for i, case in enumerate(REFERENCE_CASES)])
def test_static_value_matches_the_onnx_reference(kind, attrs, values):
    from onnx.reference import ReferenceEvaluator

    from tigris.loaders.onnx.shapes import static_value

    names = [f"in{i}" for i in range(len(values))]
    graph = helper.make_graph([helper.make_node(kind, names, ["out"], **attrs)], kind, [],
                              [helper.make_tensor_value_info("out", TensorProto.UNDEFINED, None)],
                              [numpy_helper.from_array(v, n) for v, n in zip(values, names)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    expected = ReferenceEvaluator(model).run(None, {})[0]

    actual = static_value(kind, attrs, names, dict(zip(names, values)).get, lambda name: None)

    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("attrs,expected", [({}, [1, 3, 8]), ({"start": -1}, [8]),
                                            ({"start": 1, "end": -1}, [3]), ({"end": 10}, [1, 3, 8])])
def test_shape_takes_start_and_end_as_onnx_does(attrs, expected):
    from tigris.loaders.onnx.shapes import static_value

    value = static_value("Shape", attrs, ["x"], lambda name: None, {"x": (1, 3, 8)}.get)
    assert value.dtype == np.int64 and value.tolist() == expected


@pytest.mark.parametrize("attrs,expected", [({"value_int": 3}, np.int64(3)),
                                            ({"value_ints": [1, 2]}, _i64([1, 2])),
                                            ({"value_float": 0.5}, np.float32(0.5)),
                                            ({"value_floats": [0.5, 2.0]}, _f32([0.5, 2.0]))])
def test_constant_attribute_forms_take_their_onnx_dtype(attrs, expected):
    from tigris.loaders.onnx.shapes import constant_value

    value = constant_value(attrs)
    assert value.dtype == np.asarray(expected).dtype and value.shape == np.shape(expected)
    np.testing.assert_array_equal(value, expected)
