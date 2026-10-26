"""A split of a fused projection, pushed back through shape-only operators into
the weight of the product that computed it."""

import importlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.graph.ir import QuantParam
from tigris.loaders.onnx.loader import load_model

# The package exports a function under the module's name, so the module is looked up directly.
normalize_module = importlib.import_module("tigris.loaders.onnx.normalize")


def _head_split(seq, widths, product="MatMul"):
    """Product -> Reshape [.., seq, parts, 8] -> Transpose parts first -> Split.

    The stacked axis has extent sum(widths) after the Reshape and the Transpose;
    seq equal to that extent puts the same extent on the sequence axis too.
    ``product`` is MatMul against a constant, Gemm against a transposed
    constant, or MatMul against a second model input of the same rank, since
    the runtime's MatMul of two activations does not broadcast.
    """
    total = sum(widths)
    weight = np.random.default_rng(7).standard_normal((8, 8 * total)).astype(np.float32)
    constants = [numpy_helper.from_array(np.array(widths, np.int64), "parts")]
    inputs = []
    if product == "Gemm":
        lead = [seq]
        nodes = [helper.make_node("Gemm", ["x", "w"], ["p"], name="mm", transB=1)]
        constants.append(numpy_helper.from_array(np.ascontiguousarray(weight.T), "w"))
    else:
        lead = [1, seq]
        nodes = [helper.make_node("MatMul", ["x", "w"], ["p"], name="mm")]
        if product == "MatMul":
            constants.append(numpy_helper.from_array(weight, "w"))
        else:
            inputs.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, *weight.shape]))
    perm = [len(lead), *range(len(lead)), len(lead) + 1]
    constants.append(numpy_helper.from_array(np.array([*lead, total, 8], np.int64), "shape"))
    names = [f"y{i}" for i in range(len(widths))]
    nodes += [helper.make_node("Reshape", ["p", "shape"], ["r"], name="reshape"),
              helper.make_node("Transpose", ["r"], ["t"], name="transpose", perm=perm),
              helper.make_node("Split", ["t", "parts"], names, name="split", axis=0)]
    graph = helper.make_graph(
        nodes, "head_split", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [*lead, 8]), *inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [w, *lead, 8]) for n, w in zip(names, widths)],
        constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _quantized_head_split(seq, widths, per_channel):
    """``_head_split`` in int8 QDQ form: every activation passes a QuantizeLinear and
    DequantizeLinear pair, and the weight is int8 with one scale or one per column."""
    total = sum(widths)
    rng = np.random.default_rng(11)
    weight = rng.integers(-127, 128, (8, 8 * total)).astype(np.int8)
    weight_scale = (rng.uniform(0.005, 0.02, 8 * total) if per_channel else np.array(0.01)).astype(np.float32)
    names = [f"y{i}" for i in range(len(widths))]
    constants = [
        numpy_helper.from_array(weight, "w"),
        numpy_helper.from_array(weight_scale, "ws"),
        numpy_helper.from_array(np.zeros(weight_scale.shape, np.int8), "wz"),
        numpy_helper.from_array(np.array(1 / 127, np.float32), "xs"),
        numpy_helper.from_array(np.array(0.05, np.float32), "ps"),
        numpy_helper.from_array(np.array(0, np.int8), "z"),
        numpy_helper.from_array(np.array(widths, np.int64), "parts"),
        numpy_helper.from_array(np.array([1, seq, total, 8], np.int64), "shape"),
    ]

    def qdq(source, scale, result):
        return [helper.make_node("QuantizeLinear", [source, scale, "z"], [f"{source}_q"]),
                helper.make_node("DequantizeLinear", [f"{source}_q", scale, "z"], [result])]

    nodes = [*qdq("x", "xs", "xd"),
             helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wd"], axis=1),
             helper.make_node("MatMul", ["xd", "wd"], ["p"], name="mm"), *qdq("p", "ps", "pd"),
             helper.make_node("Reshape", ["pd", "shape"], ["r"], name="reshape"), *qdq("r", "ps", "rd"),
             helper.make_node("Transpose", ["rd"], ["t"], name="transpose", perm=[2, 0, 1, 3]),
             *qdq("t", "ps", "td"),
             helper.make_node("Split", ["td", "parts"], [f"{n}_float" for n in names], name="split", axis=0)]
    for name in names:
        nodes += qdq(f"{name}_float", "ps", name)
    graph = helper.make_graph(
        nodes, "quantized_head_split", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, seq, 8])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [w, 1, seq, 8]) for n, w in zip(names, widths)],
        constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, weight_scale


def _compile(tmp_path, model, run):
    # The plan records the model's file name, so every run uses the same one.
    (tmp_path / run).mkdir()
    source, plan = tmp_path / run / "model.onnx", tmp_path / run / "model.tgrs"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "256K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    return source, plan


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("seq, widths", [(4, [1, 1, 1]), (3, [1, 1, 1]), (4, [2, 1]), (3, [2, 1]), (5, [2, 1])])
def test_each_part_is_computed_by_its_own_product(tmp_path, seq, widths):
    """Every part's chain cuts the stacked axis, also where another axis has the same extent."""
    from tigris.runtime import Session

    model = _head_split(seq, widths)
    source, plan = _compile(tmp_path, model, "compiled")
    ag = normalize_module.normalize(load_model(str(source)))
    assert "Split" not in [op.op_type for op in ag.ops]
    products = [op for op in ag.ops if op.op_type == "Gemm"]
    assert [ag.weight_data[op.inputs[1]].size for op in products] == [64 * w for w in widths]

    x = np.random.default_rng(3).uniform(-1, 1, (1, seq, 8)).astype(np.float32)
    expected = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"]).run(None, {"x": x})
    with Session(plan) as session:
        actual = session.run({"x": np.moveaxis(x, 1, -1)})
    # A Transpose writes each part, so the outputs keep the model's own axis order.
    for output, value in zip(model.graph.output, expected):
        np.testing.assert_allclose(actual[output.name], value, rtol=1e-5, atol=1e-5,
                                   err_msg=output.name)


@pytest.mark.parametrize("product", ["Gemm", "MatMul of two inputs"])
def test_a_refused_fold_leaves_the_plan_as_without_the_pass(tmp_path, monkeypatch, product):
    """A product the fold cannot cut stays whole, and no part it began is left in the plan."""
    model = _head_split(3, [2, 1], product)
    fold, attempts = normalize_module._fold_split_onto, []
    monkeypatch.setattr(normalize_module, "_fold_split_onto",
                        lambda *args: attempts.append(fold(*args)) or attempts[-1])
    _, folded = _compile(tmp_path, model, "attempted")
    assert attempts == [False]
    monkeypatch.setattr(normalize_module, "_fold_split_into_its_weight", lambda ag: ag)
    _, untouched = _compile(tmp_path, model, "skipped")
    assert folded.read_bytes() == untouched.read_bytes()


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("per_channel", [False, True], ids=["per_tensor", "per_channel"])
@pytest.mark.parametrize("seq, widths", [(3, [2, 1]), (4, [1, 1, 1])])
def test_an_int8_part_keeps_its_share_of_the_weight_quantization(tmp_path, seq, widths, per_channel):
    """Every part weight carries the whole weight's scale, or its own columns' scales."""
    from tigris.runtime import Session

    model, weight_scale = _quantized_head_split(seq, widths, per_channel)
    source, plan = _compile(tmp_path, model, "compiled")
    ag = normalize_module.normalize(load_model(str(source)))
    assert "Split" not in [op.op_type for op in ag.ops]
    columns = np.cumsum([0, *widths]) * 8
    for position, op in enumerate(op for op in ag.ops if op.op_type == "Gemm"):
        quant = ag.tensors[op.inputs[1]].quant
        expected_scale = weight_scale[columns[position]:columns[position + 1]] if per_channel else weight_scale
        np.testing.assert_array_equal(np.asarray(quant.scale).reshape(-1), expected_scale.reshape(-1))

    x = np.random.default_rng(3).uniform(-1, 1, (1, seq, 8)).astype(np.float32)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    expected = ort.InferenceSession(model.SerializeToString(), options,
                                    providers=["CPUExecutionProvider"]).run(None, {"x": x})
    with Session(plan) as session:
        actual = session.run({"x": np.moveaxis(x, 1, -1)})
    for output, value in zip(model.graph.output, expected):
        np.testing.assert_allclose(actual[output.name], value, rtol=0, atol=0.05 + 1e-6, err_msg=output.name)


def test_a_per_channel_stacked_activation_keeps_its_split(tmp_path, monkeypatch):
    """The parts of a product quantized per channel have no quantization of their own to carry."""
    model, _ = _quantized_head_split(3, [2, 1], per_channel=False)
    fold = normalize_module._fold_split_into_its_weight

    def per_channel_product(ag):
        product = next(op for op in ag.ops if op.op_type == "MatMul")
        info = ag.tensors[product.outputs[0]]
        features = info.shape[-1]
        info.quant = QuantParam(scale=np.full(features, 0.05, np.float32),
                                zero_point=np.zeros(features, np.int8), axis=len(info.shape) - 1)
        return fold(ag)

    monkeypatch.setattr(normalize_module, "_fold_split_into_its_weight", per_channel_product)
    onnx.save(model, tmp_path / "model.onnx")
    ag = normalize_module.normalize(load_model(str(tmp_path / "model.onnx")))
    assert "Split" in [op.op_type for op in ag.ops]
    assert not [name for name in ag.tensors if "_part" in name]


def test_a_block_quantization_is_cut_only_along_its_own_axis():
    """Per-tensor holds for every block; per-channel is sliced along its axis or refused."""
    blocks = [(0, 4), (4, 2)]
    per_tensor = QuantParam(scale=np.array([0.1], np.float32), zero_point=np.array([0], np.int8))
    assert [q is per_tensor for q in normalize_module._quant_of_blocks(per_tensor, 2, 1, 6, blocks)] == [True, True]
    assert normalize_module._quant_of_blocks(None, 2, 1, 6, blocks) == [None, None]

    scale = np.arange(1, 7, dtype=np.float32)
    columns = QuantParam(scale=scale, zero_point=np.arange(6, dtype=np.int8), axis=-1)
    first, second = normalize_module._quant_of_blocks(columns, 2, 1, 6, blocks)
    np.testing.assert_array_equal(first.scale, scale[:4])
    np.testing.assert_array_equal(second.zero_point, np.arange(4, 6))
    assert (first.axis, second.axis) == (1, 1)

    rows = QuantParam(scale=scale, zero_point=np.zeros(6, np.int8), axis=0)
    assert normalize_module._quant_of_blocks(rows, 2, 1, 6, blocks) is None
