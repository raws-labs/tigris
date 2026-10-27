"""A fold that rewrites a constant leaves every other reader of it as it was,
and a name a pass generates never replaces a tensor the graph already has."""

from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders import load_model
from tigris.loaders.onnx.normalize.adoption import _adopt_compiler_ops
from tigris.loaders.onnx.normalize.relabels import _transpose_product_weights


def _f32(name, value):
    return numpy_helper.from_array(np.asarray(value, np.float32), name)


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _model(nodes, inputs, outputs, constants, opset=17):
    graph = helper.make_graph(nodes, "case", [_vi(n, s) for n, s in inputs],
                              [_vi(n, s) for n, s in outputs], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def _conv(x, w, out, *bias, name=None):
    return helper.make_node("Conv", [x, w, *bias], [out], kernel_shape=[1, 1], name=name)


def _bn(x, out, gamma, beta, mean, var):
    return helper.make_node("BatchNormalization", [x, gamma, beta, mean, var], [out], epsilon=0.0)


IMAGE = [1, 2, 4, 4]
ONES = np.ones((2, 2, 1, 1))


def _bias_name_taken():
    """The folded BN bias of a.weight would be named a.bias, which conv_b reads."""
    return _model(
        [_conv("x", "a.weight", "c1"), _bn("c1", "y1", "g", "beta", "mean", "var"),
         _conv("x", "b.weight", "y2", "a.bias")],
        [("x", IMAGE)], [("y1", IMAGE), ("y2", IMAGE)],
        [_f32("a.weight", ONES), _f32("b.weight", 2 * ONES), _f32("a.bias", [10, 20]),
         _f32("g", [1, 1]), _f32("beta", [5, 5]), _f32("mean", [0, 0]), _f32("var", [1, 1])])


def _conv_weight_shared():
    return _model(
        [_conv("x", "w", "c1"), _bn("c1", "y1", "g", "beta", "mean", "var"), _conv("x", "w", "y2")],
        [("x", IMAGE)], [("y1", IMAGE), ("y2", IMAGE)],
        [_f32("w", ONES), _f32("g", [3, 3]), _f32("beta", [0, 0]), _f32("mean", [0, 0]),
         _f32("var", [1, 1])])


def _conv_bias_shared():
    return _model(
        [_conv("x", "wa", "c1", "b"), _bn("c1", "y1", "g", "beta", "mean", "var"),
         _conv("x", "wb", "y2", "b")],
        [("x", IMAGE)], [("y1", IMAGE), ("y2", IMAGE)],
        [_f32("wa", ONES), _f32("wb", 2 * ONES), _f32("b", [1, 2]), _f32("g", [3, 3]),
         _f32("beta", [1, 1]), _f32("mean", [0, 0]), _f32("var", [1, 1])])


def _bn_parameters_shared():
    """Two BNs read one zeros initializer as beta and as mean."""
    return _model(
        [_conv("x", "wa", "c1"), _bn("c1", "y1", "g", "zeros", "zeros", "var"),
         _conv("x", "wb", "c2"), _bn("c2", "y2", "g", "zeros", "zeros", "var")],
        [("x", IMAGE)], [("y1", IMAGE), ("y2", IMAGE)],
        [_f32("wa", ONES), _f32("wb", 2 * ONES), _f32("g", [3, 3]), _f32("zeros", [0, 0]),
         _f32("var", [1, 1])])


def _gemm(*inputs, out, **attrs):
    return helper.make_node("Gemm", list(inputs), [out], transB=1, **attrs)


def _gemm_weight_shared():
    return _model([_gemm("x", "w", out="y1", alpha=2.0), _gemm("x", "w", out="y2")],
                  [("x", [1, 3])], [("y1", [1, 2]), ("y2", [1, 2])],
                  [_f32("w", np.arange(6).reshape(2, 3) / 4)])


def _gemm_bias_shared():
    return _model([_gemm("x", "w", "c", out="y1", beta=3.0), _gemm("x", "w2", "c", out="y2")],
                  [("x", [1, 3])], [("y1", [1, 2]), ("y2", [1, 2])],
                  [_f32("w", np.ones((2, 3))), _f32("w2", np.full((2, 3), 0.5)), _f32("c", [1, -2])])


def _reduce_axes_shared():
    """Opset 18 states ReduceMean axes as an input, here one initializer for both."""
    nodes = [helper.make_node("Relu", ["x"], ["r"]),
             helper.make_node("ReduceMean", ["x", "axes"], ["y1"], keepdims=1),
             helper.make_node("ReduceMean", ["r", "axes"], ["y2"], keepdims=1)]
    return _model(nodes, [("x", IMAGE)], [("y1", [1, 2, 1, 1]), ("y2", [1, 2, 1, 1])],
                  [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")], opset=18)


def _constant_value_ints():
    nodes = [helper.make_node("Constant", [], ["shape"], value_ints=[1, -1]),
             helper.make_node("Reshape", ["x", "shape"], ["r"]),
             helper.make_node("Relu", ["r"], ["y"])]
    return _model(nodes, [("x", [1, 2, 4])], [("y", [1, 8])], [])


def _softmax_rows():
    """Softmax before opset 13 over a leading axis becomes rows named after it."""
    nodes = [helper.make_node("Softmax", ["x"], ["y"], axis=1, name="sm"),
             helper.make_node("Neg", ["x"], ["sm_rows_in"])]
    return _model(nodes, [("x", [2, 3, 4])], [("y", [2, 3, 4]), ("sm_rows_in", [2, 3, 4])], [],
                  opset=11)


def _pad_fill():
    pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
    nodes = [helper.make_node("Pad", ["x", "pads", "value"], ["y1"], name="pad"),
             helper.make_node("Add", ["x", "pad_fill"], ["y2"])]
    return _model(nodes, [("x", [1, 1, 4, 4])], [("y1", [1, 1, 6, 6]), ("y2", [1, 1, 4, 4])],
                  [pads, _f32("value", 1.5), _f32("pad_fill", [2.0])])


def _product(*extra, squatter=None):
    """A product over a rank-3 activation, read in model order, converts it."""
    nodes = [helper.make_node("MatMul", ["x", "w"], ["p"]), helper.make_node("Add", ["x", "p"], ["y"]),
             *extra]
    outputs = [("y", [1, 8, 8])] + ([(squatter, [1, 8, 8])] if squatter else [])
    return _model(nodes, [("x", [1, 8, 8])], outputs,
                  [_f32("w", np.arange(64).reshape(8, 8) / 64)])


def _layout_conversion():
    return _product(helper.make_node("Neg", ["x"], ["x_to_linear_1"]), squatter="x_to_linear_1")


def _layout_output():
    return _product(helper.make_node("Neg", ["x"], ["y_linear_2"]), squatter="y_linear_2")


def _product_rows():
    return _product(helper.make_node("Neg", ["x"], ["p_rows_1"]), squatter="p_rows_1")


def _split_part():
    """A split of a product's columns becomes one product per part of the weight."""
    nodes = [helper.make_node("MatMul", ["x", "w"], ["p"]),
             helper.make_node("Split", ["p"], ["a", "b"], axis=2),
             helper.make_node("Add", ["v", "w_part0"], ["c"])]
    return _model(nodes, [("x", [1, 8, 8]), ("v", [1, 4])],
                  [("a", [1, 8, 4]), ("b", [1, 8, 4]), ("c", [1, 4])],
                  [_f32("w", np.arange(64).reshape(8, 8) / 64), _f32("w_part0", [[1, 2, 3, 4]])])


def _matmul_weight_mixed():
    """One weight read as stored by a Gemm and transposed by a MatMul."""
    nodes = [_gemm("x", "w", out="y1"), helper.make_node("MatMul", ["x", "w"], ["y2"])]
    return _model(nodes, [("x", [3, 4])], [("y1", [3, 4]), ("y2", [3, 4])],
                  [_f32("w", np.arange(16).reshape(4, 4) / 16)])


def _matmul_weight_shared_rows():
    """Two batched products read one weight."""
    nodes = [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Relu", ["h"], ["r"]),
             helper.make_node("MatMul", ["r", "w"], ["y"])]
    return _model(nodes, [("x", [1, 8, 8])], [("y", [1, 8, 8])],
                  [_f32("w", np.arange(64).reshape(8, 8) / 64 - 0.5)])


def _matmul_weight_across_passes(other_reader=False):
    """A rank-2 and a batched product read one weight, optionally next to an Add."""
    nodes = [helper.make_node("MatMul", ["a", "w"], ["ya"]),
             helper.make_node("MatMul", ["b", "w"], ["yb"])]
    inputs, outputs = [("a", [4, 8]), ("b", [1, 4, 8])], [("ya", [4, 8]), ("yb", [1, 4, 8])]
    if other_reader:
        nodes.append(helper.make_node("Add", ["d", "w"], ["yd"]))
        inputs.append(("d", [8, 8]))
        outputs.append(("yd", [8, 8]))
    return _model(nodes, inputs, outputs, [_f32("w", np.arange(64).reshape(8, 8) / 64 - 0.5)])


CASES = {
    "bias_name_taken": _bias_name_taken,
    "conv_weight_shared": _conv_weight_shared,
    "conv_bias_shared": _conv_bias_shared,
    "bn_parameters_shared": _bn_parameters_shared,
    "gemm_weight_shared": _gemm_weight_shared,
    "gemm_bias_shared": _gemm_bias_shared,
    "reduce_axes_shared": _reduce_axes_shared,
    "constant_value_ints": _constant_value_ints,
    "softmax_rows": _softmax_rows,
    "pad_fill": _pad_fill,
    "layout_conversion": _layout_conversion,
    "layout_output": _layout_output,
    "product_rows": _product_rows,
    "split_part": _split_part,
    "matmul_weight_mixed": _matmul_weight_mixed,
    "matmul_weight_shared_rows": _matmul_weight_shared_rows,
    "matmul_weight_across_passes": _matmul_weight_across_passes,
    "matmul_weight_across_passes_and_add": lambda: _matmul_weight_across_passes(other_reader=True),
}


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("case", sorted(CASES))
def test_normalized_model_matches_onnx_runtime(tmp_path, case):
    _assert_matches_onnx_runtime(tmp_path, CASES[case](), "256K")


def _assert_matches_onnx_runtime(tmp_path, model, budget):
    from tigris.runtime import Session

    source, plan = tmp_path / "model.onnx", tmp_path / "model.tgrs"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", budget, "-o", str(plan)])
    assert result.exit_code == 0, result.output
    rng = np.random.default_rng(5)
    feed = {i.name: rng.uniform(-2, 2, [d.dim_value for d in i.type.tensor_type.shape.dim]).astype(np.float32)
            for i in model.graph.input}
    expected = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"]).run(None, feed)
    with Session(plan) as session:
        actual = session.run({n: v if v.ndim < 3 else np.moveaxis(v, 1, -1) for n, v in feed.items()})
    for output, value in zip(model.graph.output, expected):
        got = actual[output.name]
        np.testing.assert_allclose(got if value.ndim < 3 else np.moveaxis(got, -1, 1), value,
                                   rtol=1e-5, atol=1e-5, err_msg=output.name)


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("budget", ["64K", "1M"])
@pytest.mark.parametrize("softmax", [False, True])
def test_products_sharing_a_weight_transpose_it_once(tmp_path, softmax, budget):
    weight = np.random.default_rng(1).standard_normal((128, 128)) * 0.05
    nodes = [helper.make_node("MatMul", ["x", "w"], ["h"])]
    if softmax:
        nodes.append(helper.make_node("Softmax", ["h"], ["p"], axis=-1))
    nodes.append(helper.make_node("MatMul", ["p" if softmax else "h", "w"], ["y"]))
    model = _model(nodes, [("x", [512, 128])], [("y", [512, 128])], [_f32("w", weight)])

    ag = load_model(_save(tmp_path, model))

    assert [op.inputs[1] for op in ag.ops if op.op_type == "Gemm"] == ["w", "w"]
    np.testing.assert_array_equal(ag.weight_data["w"], weight.astype(np.float32).T)
    _assert_matches_onnx_runtime(tmp_path, model, budget)


@pytest.mark.parametrize("other_reader", [False, True])
def test_products_before_and_after_layouts_read_one_transposed_copy(tmp_path, other_reader):
    """The rank-2 product is relabeled before layouts and the batched one is
    lowered after them; both read the copy the first pass made."""
    ag = load_model(_save(tmp_path, _matmul_weight_across_passes(other_reader)))

    weight = np.arange(64, dtype=np.float32).reshape(8, 8) / 64 - 0.5
    copies = {op.inputs[1] for op in ag.ops if op.op_type == "Gemm"}
    assert len(copies) == 1
    np.testing.assert_array_equal(ag.weight_data[copies.pop()], weight.T)
    assert len(ag.weight_data) == (2 if other_reader else 1)
    if other_reader:
        add = next(op for op in ag.ops if op.op_type == "Add")
        np.testing.assert_array_equal(ag.weight_data[add.inputs[1]], weight)


@pytest.mark.parametrize("stale", [False, True])
def test_a_recorded_copy_is_read_only_while_it_holds_the_transpose(stale):
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    graph = AnalyzedGraph(
        ops=[OpNode(name="product", op_type="MatMul", inputs=["x", "w"], outputs=["y"])],
        tensors={"w": TensorInfo("w", (2, 3), 1, is_constant=True),
                 "w_1": TensorInfo("w_1", (3, 2), 1, is_constant=True)},
        weight_data={"w": weight, "w_1": weight.T + (1 if stale else 0)})

    _transpose_product_weights(graph, {"w": graph.ops}, {"w": "w_1"})

    read = graph.ops[0].inputs[1]
    assert read == ("w" if stale else "w_1")
    np.testing.assert_array_equal(graph.weight_data[read], weight.T)
    assert graph.tensors[read].shape == (3, 2)


def _producers(ag):
    return Counter(name for op in ag.ops for name in op.outputs)


def _save(tmp_path, model):
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def test_a_dequantized_weight_the_model_returns_stays_its_output(tmp_path):
    """The Transpose of a returned weight is not folded into it, so the output
    keeps its tensor and is refused as every constant output is."""
    constants = [numpy_helper.from_array(np.arange(6, dtype=np.int8).reshape(2, 3), "wq"),
                 _f32("ws", 0.1), numpy_helper.from_array(np.array(0, np.int8), "wz"),
                 _f32("xs", 0.05), numpy_helper.from_array(np.array(0, np.int8), "xz")]
    nodes = [helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
             helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
             helper.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["wf"]),
             helper.make_node("Transpose", ["wf"], ["wt"], perm=[1, 0]),
             helper.make_node("MatMul", ["xd", "wt"], ["y"])]
    path = _save(tmp_path, _model(nodes, [("x", [1, 3])], [("y", [1, 2]), ("wf", [2, 3])], constants))

    ag = load_model(path)

    assert "wf" in ag.model_outputs and "wf" in ag.tensors
    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "64K", "-o", str(tmp_path / "p.tgrs")])
    assert result.exit_code == 1
    assert "output   wf   2x3 float32" in result.output
    assert "cannot compile: output 'wf' is not computed by any operator" in result.output


def test_an_operator_without_operands_is_refused_by_name(tmp_path):
    nodes = [helper.make_node("RandomUniform", [], ["r"], shape=[1, 3], name="random"),
             helper.make_node("Add", ["x", "r"], ["y"])]
    path = _save(tmp_path, _model(nodes, [("x", [1, 3])], [("y", [1, 3])], []))

    result = CliRunner().invoke(cli, ["compile", str(path), "-m", "64K", "-o", str(tmp_path / "p.tgrs")])

    assert isinstance(result.exception, SystemExit), result.exception
    assert "random (RandomUniform)" in result.output


def _qdq(tensor, scale, out, zero="zero"):
    return [helper.make_node("QuantizeLinear", [tensor, scale, zero], [f"{out}_q"]),
            helper.make_node("DequantizeLinear", [f"{out}_q", scale, zero], [out])]


def _int8_constants(**values):
    return ([_f32(name, value) for name, value in values.items()]
            + [numpy_helper.from_array(np.array(0, np.int8), "zero")])


def test_an_input_quantized_first_keeps_a_tensor_of_the_same_name(tmp_path):
    """The input is quantized ahead of its Transpose as x_input_q, which the Relu writes."""
    nodes = [helper.make_node("Transpose", ["x"], ["t"], perm=[0, 2, 1]), *_qdq("t", "s", "td"),
             helper.make_node("Relu", ["td"], ["x_input_q"]), *_qdq("x_input_q", "s", "rd"),
             helper.make_node("Transpose", ["rd"], ["o"], perm=[0, 2, 1]), *_qdq("o", "s", "y")]
    path = _save(tmp_path, _model(nodes, [("x", [1, 3, 4])], [("y", [1, 3, 4])],
                                  _int8_constants(s=0.05), opset=13))

    ag = load_model(path)

    assert max(_producers(ag).values()) == 1
    assert [op.outputs for op in ag.ops if op.op_type == "Relu"] == [["x_input_q"]]
    assert ag.tensors["x_input_q"].shape == (1, 4, 3)


def test_an_int32_bias_copy_keeps_a_tensor_of_the_same_name(tmp_path):
    """Two int8 Convs share a float bias, so each gets an int32 copy named
    after it; conv_a's output already has that name."""
    w = np.full((2, 2, 1, 1), 0.5, np.float32)
    constants = _int8_constants(s=0.05, ws=0.01, b=[0.1, -0.2], wa=w, wb=-w)
    nodes = [*_qdq("x", "s", "xd"), *_qdq("wa", "ws", "wad"), *_qdq("wb", "ws", "wbd"),
             _conv("xd", "wad", "b__int32_conv_a", "b", name="conv_a"),
             *_qdq("b__int32_conv_a", "s", "hd"), _conv("hd", "wbd", "c2", "b", name="conv_b"),
             *_qdq("c2", "s", "y")]
    path = _save(tmp_path, _model(nodes, [("x", IMAGE)], [("y", IMAGE)], constants, opset=13))

    ag = load_model(path)

    conv_a = next(op for op in ag.ops if op.name == "conv_a")
    assert conv_a.outputs == ["b__int32_conv_a"]
    assert not ag.tensors["b__int32_conv_a"].is_constant
    assert conv_a.inputs[2] != "b__int32_conv_a"
    assert ag.weight_data[conv_a.inputs[2]].dtype == np.int32


def _quantized_conv_bn(output, second):
    """An int8 Conv without bias, w quantized at 0.01, then a BN with gamma
    [2, -0.5], then conv_b over the BN's output with weight ``second``."""
    weight = np.array([[[[0.5]], [[-0.25]]], [[[0.3]], [[0.1]]]], np.float32)
    constants = _int8_constants(s=0.1, ws=0.01, w=weight, w2=-weight, g=[2.0, -0.5],
                                beta=[0.5, 0.5], mean=[0, 0], var=[1, 1])
    nodes = [*_qdq("x", "s", "xd"), *_qdq("w", "ws", "wd"), *_qdq("w2", "ws", "w2d"),
             _conv("xd", "wd", "raw", name="conv_a"), _bn("raw", output, "g", "beta", "mean", "var"),
             *_qdq(output, "s", "hd"), _conv("hd", second, "c2", name="conv_b"), *_qdq("c2", "s", "y")]
    return _model(nodes, [("x", [1, 2, 2, 2])], [("y", [1, 2, 2, 2])], constants, opset=13)


def test_a_quantized_bn_bias_keeps_a_tensor_of_the_same_name(tmp_path):
    """The bias the fold adds would be named w_bn_bias, the BN's own output."""
    ag = load_model(_save(tmp_path, _quantized_conv_bn("w_bn_bias", "w2d")))

    conv_a = next(op for op in ag.ops if op.name == "conv_a")
    assert conv_a.outputs == ["w_bn_bias"] and not ag.tensors["w_bn_bias"].is_constant
    assert conv_a.inputs[2] != "w_bn_bias" and ag.weight_data[conv_a.inputs[2]].dtype == np.int32


def test_a_quantized_bn_leaves_a_shared_weight_to_its_other_reader(tmp_path):
    """The fold negates and rescales its Conv's weight; conv_b reads the original."""
    ag = load_model(_save(tmp_path, _quantized_conv_bn("normed", "wd")))

    conv_b = next(op for op in ag.ops if op.name == "conv_b")
    folded = next(op for op in ag.ops if op.name == "conv_a")
    assert ag.weight_data[conv_b.inputs[1]][:, :, 0, 0].tolist() == [[50, -25], [30, 10]]
    np.testing.assert_allclose(ag.tensors[conv_b.inputs[1]].quant.scale, [0.01])
    assert ag.weight_data[folded.inputs[1]][:, :, 0, 0].tolist() == [[50, -25], [-30, -10]]


def test_a_detection_scratch_output_keeps_a_tensor_of_the_same_name():
    shapes = {"boxes": (1, 4, 6), "scores": (1, 3, 6), "d_boxes": (1, 4, 2), "d_classes": (1, 2),
              "d_scores": (1, 2), "d_count": (1,), "d_boxes_scratch": (5,)}
    graph = AnalyzedGraph(
        ops=[OpNode(name="detect", op_type="tigris::DetectionPostProcess",
                    inputs=["boxes", "scores", "anchors"],
                    outputs=["d_boxes", "d_classes", "d_scores", "d_count"],
                    attrs={"num_classes": 2, "max_detections": 2, "scales": [10, 10, 5, 5]})],
        tensors={name: TensorInfo(name, shape, TensorProto.FLOAT) for name, shape in shapes.items()},
        weight_data={"anchors": np.zeros((6, 4), np.float32)})

    ag = _adopt_compiler_ops(graph)

    scratch = ag.ops[0].outputs[4]
    assert scratch != "d_boxes_scratch" and ag.tensors["d_boxes_scratch"].shape == (5,)
