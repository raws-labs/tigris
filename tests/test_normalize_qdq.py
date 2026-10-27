"""Tests for QDQ folding in the normalize pass."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model
from tigris.loaders.onnx.loader import load_model as load_raw
from tigris.loaders.onnx.normalize import normalize
from tigris.loaders.onnx.normalize.quantization import _fold_qdq
from tigris.graph.ir import QuantParam
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.defs import NO_QUANT_PARAM
from tigris.pipeline import plan_model

from conftest import save_onnx


def test_qdq_fold_removes_ql_dql_ops(qdq_conv_path):
    """All QuantizeLinear and DequantizeLinear ops should be removed."""
    ag = load_raw(qdq_conv_path)
    qdq_types = {"QuantizeLinear", "DequantizeLinear"}
    assert any(op.op_type in qdq_types for op in ag.ops), "fixture should have Q/DQ ops"

    ag = _fold_qdq(ag)
    remaining = [op.op_type for op in ag.ops]
    assert not any(t in qdq_types for t in remaining), f"Q/DQ ops remain: {remaining}"


def test_qdq_fold_quantizes_the_conv(qdq_conv_path):
    """Folding leaves one int8 Conv+Relu with weight, input and output quant params."""
    ag = load_model(qdq_conv_path)

    # Relu is absorbed as fused activation
    op_types = [op.op_type for op in ag.ops]
    assert "Conv" in op_types
    assert len(ag.ops) == 1
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"
    assert ag.is_quantized is True

    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    w_name = conv_op.inputs[1]
    assert w_name in ag.weight_data
    assert ag.weight_data[w_name].dtype == np.int8

    w_info = ag.tensors[w_name]
    assert w_info.quant is not None
    assert isinstance(w_info.quant, QuantParam)
    assert w_info.quant.scale.shape[0] == 2  # per-channel, 2 output channels

    input_info = ag.tensors[ag.model_inputs[0]]
    assert input_info.quant is not None
    assert input_info.quant.scale.size == 1  # per-tensor
    np.testing.assert_allclose(input_info.quant.scale, [0.05])

    # The Conv output inherits the fused Relu output's quant param
    conv_info = ag.tensors[conv_op.outputs[0]]
    assert conv_info.quant is not None
    np.testing.assert_allclose(conv_info.quant.scale, [0.1])

    for name in ["inp_scale", "inp_zp", "w_scale", "w_zp", "out_scale", "out_zp"]:
        assert name not in ag.weight_data, f"{name} still in weight_data"


# Binary round-trip tests


def test_qdq_binary_roundtrip(qdq_conv_path):
    """The quantized plan carries valid quant params and keeps the ONNX output name."""
    ag = plan_model(qdq_conv_path)
    assert ag.model_outputs == ["output"]
    assert ag.tensors["output"].quant is not None

    plan = read_binary_plan(emit_binary_bytes(ag))

    assert plan["model_name"] == "qdq_conv"
    assert plan["num_ops"] == 1  # Conv with fused Relu
    assert len(plan["quant_params"]) > 0

    tensors_with_qp = [t for t in plan["tensors"] if t["quant_param_idx"] != NO_QUANT_PARAM]
    assert len(tensors_with_qp) > 0
    for t in tensors_with_qp:
        assert t["quant_param_idx"] < len(plan["quant_params"])

    # The weight has per-channel quant (2 output channels)
    per_channel = [qp for qp in plan["quant_params"] if qp["num_channels"] > 1]
    assert len(per_channel) > 0
    qp = per_channel[0]
    assert qp["num_channels"] == 2
    assert len(qp["multipliers"]) == 2
    assert len(qp["shifts"]) == 2
    assert all(m > 0 for m in qp["multipliers"])

    names = [plan["tensors"][index]["name"] for index in plan["model_outputs"]]
    assert names == ["output"]


def test_float_model_no_quant_params(linear_3op_path):
    """A float model is left unquantized and emits no quant params."""
    ag = load_raw(linear_3op_path)
    orig_ops = len(ag.ops)
    ag = normalize(ag)
    assert ag.is_quantized is False
    assert len(ag.ops) == orig_ops

    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    plan = read_binary_plan(emit_binary_bytes(ag))

    assert len(plan["quant_params"]) == 0
    for t in plan["tensors"]:
        assert t["quant_param_idx"] == NO_QUANT_PARAM


def _qdq_gemm_bias_model(tmp_path, shared_scale: bool):
    """A quantized Gemm whose bias arrives as an unfused float Add."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    scale_names = ("io_scale", "io_zp") if shared_scale else ("w_scale", "w_zp")
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zp"),
        numpy_helper.from_array(
            np.array([[0.5, -0.25, 0.75, 0.25], [-0.5, 0.25, 0.5, -0.75]],
                     dtype=np.float32),
            "weight",
        ),
        numpy_helper.from_array(np.array([0.5, -0.25], dtype=np.float32), "bias"),
    ]
    if not shared_scale:
        initializers.extend([
            numpy_helper.from_array(np.array([0.25], dtype=np.float32), "w_scale"),
            numpy_helper.from_array(np.array([0], dtype=np.int8), "w_zp"),
        ])
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "io_scale", "io_zp"], ["in_q"]),
        helper.make_node("DequantizeLinear", ["in_q", "io_scale", "io_zp"], ["in_dq"]),
        helper.make_node("QuantizeLinear", ["weight", *scale_names], ["w_q"]),
        helper.make_node("DequantizeLinear", ["w_q", *scale_names], ["w_dq"]),
        helper.make_node("Gemm", ["in_dq", "w_dq"], ["product"], transB=1),
        helper.make_node(
            "QuantizeLinear", ["product", "io_scale", "io_zp"], ["product_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["product_q", "io_scale", "io_zp"], ["product_dq"]
        ),
        helper.make_node("Add", ["product_dq", "bias"], ["output"]),
    ]
    return save_onnx(
        tmp_path, "qdq_gemm_bias", nodes, [X], [Y], initializers,
        opset=17, ir_version=8, filename=f"qdq_gemm_bias_{int(shared_scale)}.onnx",
    )


def test_constant_add_after_dequantize_stays_float(tmp_path):
    ag = load_model(_qdq_gemm_bias_model(tmp_path, shared_scale=False))
    assert [op.op_type for op in ag.ops] == ["Gemm", "Dequantize", "Add"]
    assert len(ag.ops[0].inputs) == 2
    assert ag.tensors[ag.ops[0].outputs[0]].dtype == 3
    assert ag.weight_data["bias"].dtype == np.float32
    assert ag.tensors["output"].dtype == 1
    assert ag.tensors["output"].quant is None
    assert ag.model_output_dtypes == [1]


@pytest.mark.parametrize("kind", ["Add", "Mul", "Sub", "Div"])
@pytest.mark.parametrize("ending", ["quantized", "terminal", "float_only", "nonuniform"])
def test_constant_section_boundary_respects_integer_lowering(tmp_path, kind, ending):
    path = _qdq_gemm_bias_model(tmp_path, shared_scale=False)
    model = onnx.load(path)
    constant = np.array([0.5, 0.75] if ending == "nonuniform" else [0.5, 0.5], np.float32)
    next(t for t in model.graph.initializer if t.name == "bias").CopyFrom(
        numpy_helper.from_array(constant, "bias"))
    model.graph.node[-1].op_type = kind
    if ending != "terminal":
        model.graph.node[-1].output[0] = "adjusted"
        source = "adjusted"
        if ending == "float_only":
            model.graph.node.extend([helper.make_node("Exp", [source], ["expanded"])])
            source = "expanded"
        model.graph.node.extend([
            helper.make_node("QuantizeLinear", [source, "io_scale", "io_zp"], ["encoded"]),
            helper.make_node("DequantizeLinear", ["encoded", "io_scale", "io_zp"], ["output"]),
        ])
    onnx.save(model, path)
    ag = load_model(path)
    binary = next(op for op in ag.ops if op.op_type == ("Add" if kind in {"Add", "Sub"} else "Mul"))
    expected_dtype = 3 if ending == "quantized" else 1
    assert ag.tensors[binary.outputs[0]].dtype == expected_dtype
    assert ag.weight_data[binary.inputs[1]].dtype == (np.int8 if ending == "quantized" else np.float32)
    assert any(op.op_type == "Dequantize" for op in ag.ops) == (ending != "quantized")


def test_uniform_constant_shared_by_float_and_int8_adds(tmp_path):
    from tigris.runtime import Session

    path = _qdq_gemm_bias_model(tmp_path, shared_scale=False)
    model = onnx.load(path)
    next(t for t in model.graph.initializer if t.name == "bias").CopyFrom(
        numpy_helper.from_array(np.full(2, 0.5, np.float32), "bias"))
    model.graph.node.extend([
        helper.make_node("Add", ["product_dq", "bias"], ["adjusted"]),
        helper.make_node("QuantizeLinear", ["adjusted", "io_scale", "io_zp"], ["encoded"]),
        helper.make_node("DequantizeLinear", ["encoded", "io_scale", "io_zp"], ["quantized"]),
    ])
    model.graph.output.extend([helper.make_tensor_value_info("quantized", TensorProto.FLOAT, [1, 2])])
    onnx.save(model, path)
    ag = load_model(path)
    adds = {ag.tensors[op.outputs[0]].dtype: op for op in ag.ops if op.op_type == "Add"}
    assert set(adds) == {1, 3}
    assert ag.weight_data[adds[1].inputs[1]].dtype == np.float32
    assert ag.weight_data[adds[3].inputs[1]].dtype == np.int8
    ag = plan_model(str(path), ("1K",))
    binary = tmp_path / "shared.tgrs"
    binary.write_bytes(emit_binary_bytes(ag))
    with Session(binary) as session:
        actual = session.run({"input": np.full((1, 4), 31.75, np.float32)})
    assert actual["output"][0, 0] == np.float32(32.25)
    assert actual["quantized"][0, 0] == np.float32(31.75)


def test_bias_after_quantization_preserves_product_clipping(tmp_path):
    import onnxruntime as ort

    from tigris.runtime import Session

    path = _qdq_gemm_bias_model(tmp_path, shared_scale=False)
    binary = tmp_path / "bias.tgrs"
    values = np.full((1, 4), 31.75, dtype=np.float32)
    expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"input": values}
    )[0]
    for compress, xip in ((None, False), ("lz4", False), (None, True)):
        ag = plan_model(str(path), ("16K",), compress=compress)
        binary.write_bytes(emit_binary_bytes(ag, compress=compress, xip=xip))
        with Session(binary) as session:
            actual = session.run({"input": values})["output"]
        np.testing.assert_array_equal(actual, expected)


def test_shared_scale_initializer_still_folds_activations(tmp_path):
    ag = load_model(_qdq_gemm_bias_model(tmp_path, shared_scale=True))
    assert [op.op_type for op in ag.ops] == ["Gemm", "Dequantize", "Add"]
    assert ag.tensors["input"].dtype == 3
    assert ag.tensors["output"].dtype == 1


def test_float_constant_add_becomes_the_gemm_bias(tmp_path):
    """A float bias Add folds into the float Gemm bias slot."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    initializers = [
        numpy_helper.from_array(
            np.array([[0.5, -0.25, 0.75, 0.25], [-0.5, 0.25, 0.5, -0.75]],
                     dtype=np.float32),
            "weight",
        ),
        numpy_helper.from_array(np.array([0.5, -0.25], dtype=np.float32), "bias"),
    ]
    nodes = [
        helper.make_node("Gemm", ["input", "weight"], ["product"], transB=1),
        helper.make_node("Add", ["product", "bias"], ["output"]),
    ]
    path = save_onnx(
        tmp_path, "float_gemm_bias", nodes, [X], [Y], initializers, opset=17, ir_version=8
    )

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    gemm = ag.ops[0]
    assert len(gemm.inputs) == 3
    assert np.allclose(ag.weight_data[gemm.inputs[2]], [0.5, -0.25])
    assert gemm.outputs == ["output"]
    assert ag.tensors["output"].dtype == 1


# MatMul relabeling


def _matmul_model(tmp_path, op_type, weight_constant=True, **attrs):
    """A rank-2 product against a [4, 3] weight."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    inputs = [X]
    initializers = []
    if weight_constant:
        initializers.append(numpy_helper.from_array(values, "weight"))
    else:
        inputs.append(
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, [4, 3])
        )
    node = helper.make_node(op_type, ["input", "weight"], ["output"], **attrs)
    path = save_onnx(
        tmp_path, "product", [node], inputs, [Y], initializers,
        opset=17, ir_version=8, filename=f"{op_type}_{int(weight_constant)}.onnx",
    )
    return path, values


def test_matmul_becomes_a_transposed_gemm(tmp_path):
    """The kernels index the weight as [OC, IC]; ONNX MatMul states [IC, OC]."""
    path, values = _matmul_model(tmp_path, "MatMul")

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    gemm = ag.ops[0]
    assert gemm.attrs["transB"] == 1
    assert np.array_equal(ag.weight_data["weight"], values.T)
    assert ag.tensors["weight"].shape == (3, 4)


def test_gemm_without_transb_is_transposed(tmp_path):
    """A Gemm left at the default transB states the weight the same way."""
    path, values = _matmul_model(tmp_path, "Gemm")

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    assert ag.ops[0].attrs["transB"] == 1
    assert np.array_equal(ag.weight_data["weight"], values.T)


def test_gemm_with_transb_is_left_alone(tmp_path):
    """A Gemm that already states [OC, IC] must not be transposed again."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    node = helper.make_node("Gemm", ["input", "weight"], ["output"], transB=1)
    path = save_onnx(
        tmp_path, "gemm_t", [node], [X], [Y], [numpy_helper.from_array(values, "weight")],
        opset=17, ir_version=8,
    )

    ag = load_model(path)

    assert np.array_equal(ag.weight_data["weight"], values)


def test_matmul_with_a_dynamic_operand_is_left_alone(tmp_path):
    """Without a constant matrix there is nothing to transpose ahead of time."""
    path, _ = _matmul_model(tmp_path, "MatMul", weight_constant=False)

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["MatMul"]


def test_matmul_weight_shared_with_another_op_gets_a_transposed_copy(tmp_path):
    """Transposing in place would misstate the weight for the other reader."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3])
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["product"]),
        helper.make_node("Add", ["product", "weight"], ["output"]),
    ]
    path = save_onnx(
        tmp_path, "shared", nodes, [X], [Y], [numpy_helper.from_array(values, "weight")],
        opset=17, ir_version=8,
    )

    ag = load_model(path)

    product = next(op for op in ag.ops if op.outputs == ["product"])
    add = next(op for op in ag.ops if op.op_type == "Add")
    assert product.op_type == "Gemm" and product.inputs[1] != "weight"
    assert np.array_equal(ag.weight_data[product.inputs[1]], values.T)
    assert "weight" in add.inputs and np.array_equal(ag.weight_data["weight"], values)


def test_quantized_matmul_weight_keeps_its_channel_axis(tmp_path):
    """A per-channel weight's channel axis moves with the transpose."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    initializers = [
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "io_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int8), "io_zp"),
        numpy_helper.from_array(
            np.array([0.5, 0.25, 0.125], dtype=np.float32), "w_scale"
        ),
        numpy_helper.from_array(np.zeros(3, dtype=np.int8), "w_zp"),
        numpy_helper.from_array(
            np.arange(12, dtype=np.float32).reshape(4, 3), "weight"
        ),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "io_scale", "io_zp"], ["in_q"]),
        helper.make_node(
            "DequantizeLinear", ["in_q", "io_scale", "io_zp"], ["in_dq"]
        ),
        helper.make_node(
            "QuantizeLinear", ["weight", "w_scale", "w_zp"], ["w_q"], axis=1
        ),
        helper.make_node(
            "DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], axis=1
        ),
        helper.make_node("MatMul", ["in_dq", "w_dq"], ["product"]),
        helper.make_node(
            "QuantizeLinear", ["product", "io_scale", "io_zp"], ["out_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["out_q", "io_scale", "io_zp"], ["output"]
        ),
    ]
    path = save_onnx(
        tmp_path, "qdq_matmul", nodes, [X], [Y], initializers, opset=17, ir_version=8
    )

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    quant = ag.tensors["weight"].quant
    assert quant.axis == 0
    assert quant.scale.tolist() == [0.5, 0.25, 0.125]


# uint8 activations


def _quint8_model(tmp_path, weight_unsigned=False, act_zp=np.uint8(114)):
    """A 1x1 Conv whose activations take the dtype of act_zp, uint8 by default as ORT emits."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    weight_values = (
        np.array([[[[200]]]], dtype=np.uint8)
        if weight_unsigned
        else np.array([[[[72]]]], dtype=np.int8)
    )
    initializers = [
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "act_scale"),
        numpy_helper.from_array(np.array(act_zp), "act_zp"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(weight_values, "weight_q"),
        numpy_helper.from_array(
            np.array(128 if weight_unsigned else 0,
                     dtype=np.uint8 if weight_unsigned else np.int8),
            "w_zp",
        ),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "act_scale", "act_zp"], ["in_q"]),
        helper.make_node(
            "DequantizeLinear", ["in_q", "act_scale", "act_zp"], ["in_dq"]
        ),
        helper.make_node(
            "DequantizeLinear", ["weight_q", "w_scale", "w_zp"], ["w_dq"]
        ),
        helper.make_node("Conv", ["in_dq", "w_dq"], ["conv"]),
        helper.make_node(
            "QuantizeLinear", ["conv", "act_scale", "act_zp"], ["out_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["out_q", "act_scale", "act_zp"], ["output"]
        ),
    ]
    return save_onnx(
        tmp_path, "quint8", nodes, [X], [Y], initializers,
        opset=17, ir_version=8, filename=f"quint8_{int(weight_unsigned)}.onnx",
    )


def test_unsigned_activation_moves_to_the_signed_domain(tmp_path):
    """uint8 v and int8 v - 128 denote the same real value under shifted zero points."""
    ag = load_model(_quint8_model(tmp_path))

    activation = ag.tensors["input"]
    assert activation.dtype == 3  # INT8, as the plan executes
    assert int(activation.quant.zero_point[0]) == 114 - 128
    assert float(activation.quant.scale[0]) == 0.25
    # The caller hands over what the model declares, whatever the plan stores.
    assert ag.model_input_dtypes == [1]


def test_unsigned_weight_data_moves_with_its_zero_point(tmp_path):
    """A stored uint8 weight is restated, not reinterpreted."""
    ag = load_model(_quint8_model(tmp_path, weight_unsigned=True))

    weight = ag.weight_data["weight_q"]
    assert weight.dtype == np.int8
    assert weight.ravel().tolist() == [200 - 128]
    info = ag.tensors["weight_q"]
    assert info.dtype == 3
    assert int(info.quant.zero_point[0]) == 128 - 128


def test_signed_activations_are_untouched(tmp_path):
    """An int8 graph keeps the zero points the model states."""
    ag = load_model(_quint8_model(tmp_path, act_zp=np.int8(-14)))

    assert int(ag.tensors["input"].quant.zero_point[0]) == -14
    assert ag.weight_data["weight_q"].ravel().tolist() == [72]


def _int_weight_conv(path, weight_type):
    """Q/DQ input, a constant integer weight dequantized per channel, Conv, Q/DQ output."""
    weights = np.array([-8, -3, 0, 2, 7, 1, -1, 5, -6] * 2, dtype=np.int8).reshape(2, 1, 3, 3)
    initializers = [
        helper.make_tensor("w", weight_type, weights.shape, weights.flatten().tolist()),
        numpy_helper.from_array(np.array([0.02, 0.03], dtype=np.float32), "w_scale"),
        helper.make_tensor("w_zp", weight_type, [2], [0, 0]),
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "x_zp"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "y_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]),
        helper.make_node("DequantizeLinear", ["w", "w_scale", "w_zp"], ["w_dq"], axis=0),
        helper.make_node("Conv", ["x_dq", "w_dq"], ["conv"], name="conv"),
        helper.make_node("QuantizeLinear", ["conv", "y_scale", "y_zp"], ["y_q"]),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["output"]),
    ]
    return save_onnx(path.parent, "int_weight_conv", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
                     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
                     initializers, opset=21, ir_version=10, filename=path.name)


def test_int4_weights_compile_as_the_same_int8_values(tmp_path):
    """4-bit weights hold int8 values exactly; the plan is the int8 model's."""
    plans = []
    for weight_type in (TensorProto.INT4, TensorProto.INT8):

        (tmp_path / str(weight_type)).mkdir()
        graph = plan_model(str(_int_weight_conv(tmp_path / str(weight_type) / "conv.onnx", weight_type)), ("64K",))
        assert graph.weight_data["w"].dtype == np.int8
        plans.append(emit_binary_bytes(graph))
    assert plans[0] == plans[1]


def test_a_second_encoding_of_an_activation_is_a_requantization(tmp_path):
    """Relu's output is quantized for its own readers and again, at another
    scale, for a second one: it keeps its encoding and a Quantize makes the other."""
    s = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.float32), name)  # noqa: E731
    z = lambda name: numpy_helper.from_array(np.array(0, dtype=np.int8), name)  # noqa: E731
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"]),
        helper.make_node("Relu", ["xd"], ["r"]),
        helper.make_node("QuantizeLinear", ["r", "as", "az"], ["rq"]),
        helper.make_node("DequantizeLinear", ["rq", "as", "az"], ["rd"]),
        helper.make_node("Neg", ["rd"], ["n"]),
        helper.make_node("QuantizeLinear", ["n", "as", "az"], ["nq"]),
        helper.make_node("DequantizeLinear", ["nq", "as", "az"], ["first"]),
        helper.make_node("QuantizeLinear", ["rd", "bs", "bz"], ["second_q"]),
        helper.make_node("DequantizeLinear", ["second_q", "bs", "bz"], ["second"]),
    ]
    path = save_onnx(tmp_path, "two_encodings", nodes,
                     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
                     [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 8]) for n in ("first", "second")],
                     [s("xs", 0.05), z("xz"), s("as", 0.02), z("az"), s("bs", 0.07), z("bz")],
                     ir_version=8, filename="two.onnx")
    ag = _fold_qdq(load_raw(path))
    relu = next(op for op in ag.ops if op.op_type == "Relu")
    assert float(ag.tensors[relu.outputs[0]].quant.scale[0]) == np.float32(0.02)
    requantize = [op for op in ag.ops if op.op_type == "Quantize"
                  and ag.tensors[op.inputs[0]].dtype == 3]
    assert len(requantize) == 1 and requantize[0].inputs == relu.outputs
    assert float(ag.tensors[requantize[0].outputs[0]].quant.scale[0]) == np.float32(0.07)


def test_dequantize_float_section_keeps_int8_model_input(tmp_path):
    from tigris.runtime import Session

    source = _qdq_gemm_bias_model(tmp_path, shared_scale=False)
    model = onnx.load(source)
    kept = [node for node in model.graph.node if node.op_type == "Add"]
    kept.insert(0, helper.make_node("DequantizeLinear", ["input", "io_scale", "io_zp"], ["product_dq"]))
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT8
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        dim.dim_value = 1
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 2
    onnx.save(model, source)
    ag = plan_model(str(source), ("1K",))
    assert ag.is_quantized
    assert [op.op_type for op in ag.ops] == ["Dequantize", "Add"]
    binary = tmp_path / "section.tgrs"
    binary.write_bytes(emit_binary_bytes(ag))
    values = np.array([[127, -128]], dtype=np.int8)
    with Session(binary) as session:
        actual = session.run({"input": values})["output"]
    np.testing.assert_array_equal(actual, [[32.25, -32.25]])


def test_float_only_kernel_between_quantized_tensors(tmp_path):
    from tigris.runtime import Session

    path = _qdq_gemm_bias_model(tmp_path, shared_scale=False)
    model = onnx.load(path)
    model.graph.node[-1].op_type = "Exp"
    del model.graph.node[-1].input[1:]
    model.graph.node[-1].output[0] = "expanded"
    model.graph.node.extend([
        helper.make_node("QuantizeLinear", ["expanded", "io_scale", "io_zp"], ["encoded"]),
        helper.make_node("DequantizeLinear", ["encoded", "io_scale", "io_zp"], ["output"]),
    ])
    onnx.save(model, path)
    ag = plan_model(str(path), ("1K",))
    assert [op.op_type for op in ag.ops] == ["Gemm", "Dequantize", "Exp", "Quantize"]
    binary = tmp_path / "exp.tgrs"
    binary.write_bytes(emit_binary_bytes(ag))
    with Session(binary) as session:
        actual = session.run({"input": np.zeros((1, 4), dtype=np.float32)})["output"]
    np.testing.assert_array_equal(actual, [[1.0, 1.0]])


def _qdq_float_bias_model(tmp_path, op_type):
    """An int8 QDQ Conv or Gemm whose bias is a plain float32 initializer, as
    PyTorch's PT2E quantizers export it."""
    rng = np.random.default_rng(7)
    if op_type == "Conv":
        x_shape, out_shape, w = [1, 3, 6, 6], [1, 4, 6, 6], rng.normal(0, 0.3, (4, 3, 3, 3))
        attrs = {"pads": [1, 1, 1, 1]}
    else:
        x_shape, out_shape, w = [1, 6], [1, 4], rng.normal(0, 0.3, (4, 6))
        attrs = {"transB": 1}
    w = w.astype(np.float32)
    w_scale = (np.abs(w.reshape(4, -1)).max(axis=1) / 127).astype(np.float32)
    w_q = np.clip(np.round(w / w_scale.reshape([4] + [1] * (w.ndim - 1))), -127, 127).astype(np.int8)
    initializers = [
        numpy_helper.from_array(np.array(0.02, np.float32), "in_scale"),
        numpy_helper.from_array(np.array(0, np.int8), "zp"),
        numpy_helper.from_array(w_q, "w_q"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(np.zeros(4, np.int8), "w_zp"),
        numpy_helper.from_array(rng.normal(0, 0.5, 4).astype(np.float32), "bias"),
        numpy_helper.from_array(np.array(0.05, np.float32), "out_scale"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_scale", "zp"], ["in_q"]),
        helper.make_node("DequantizeLinear", ["in_q", "in_scale", "zp"], ["in_dq"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], axis=0),
        helper.make_node(op_type, ["in_dq", "w_dq", "bias"], ["product"], **attrs),
        helper.make_node("QuantizeLinear", ["product", "out_scale", "zp"], ["out_q"]),
        helper.make_node("DequantizeLinear", ["out_q", "out_scale", "zp"], ["output"]),
    ]
    return save_onnx(
        tmp_path, f"qdq_{op_type.lower()}_float_bias", nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, x_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        initializers, opset=13, ir_version=8,
    )


@pytest.mark.parametrize("op_type", ["Conv", "Gemm"])
def test_float_bias_of_an_int8_product_becomes_int32(tmp_path, op_type):
    """The int8 kernels add an int32 bias at the accumulator scale, so a float
    bias is quantized to it, rounding as the TFLite converter does."""
    path = _qdq_float_bias_model(tmp_path, op_type)
    raw = onnx.load(path)
    bias = numpy_helper.to_array(next(t for t in raw.graph.initializer if t.name == "bias"))
    w_scale = numpy_helper.to_array(next(t for t in raw.graph.initializer if t.name == "w_scale"))
    ag = load_model(path)
    product = next(op for op in ag.ops if op.op_type == op_type)
    stored = ag.weight_data[product.inputs[2]]
    assert stored.dtype == np.int32
    scale = np.float64(0.02) * w_scale.astype(np.float64)
    np.testing.assert_array_equal(stored.reshape(-1), np.round(bias / scale).astype(np.int32))
    quant = ag.tensors[product.inputs[2]].quant
    np.testing.assert_allclose(quant.scale, scale.astype(np.float32))
    assert ag.tensors[product.inputs[2]].dtype == 6


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("op_type", ["Conv", "Gemm"])
def test_float_bias_of_an_int8_product_matches_onnx_runtime(tmp_path, op_type):
    import onnxruntime as ort

    from tigris.runtime import Session

    path = _qdq_float_bias_model(tmp_path, op_type)
    rng = np.random.default_rng(3)
    shape = [1, 3, 6, 6] if op_type == "Conv" else [1, 6]
    values = rng.uniform(-1, 1, shape).astype(np.float32)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    expected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(
        None, {"input": values})[0]
    ag = plan_model(str(path), ("16K",))
    binary = tmp_path / "float_bias.tgrs"
    binary.write_bytes(emit_binary_bytes(ag))
    stored = np.transpose(values, (0, 2, 3, 1)) if op_type == "Conv" else values
    with Session(binary) as session:
        actual = session.run({"input": np.ascontiguousarray(stored)})["output"]
    if op_type == "Conv":
        actual = np.transpose(actual, (0, 3, 1, 2))
    assert np.abs(np.round((actual - expected) / 0.05)).max() <= 1


def test_int8_product_with_a_non_int32_bias_is_refused(tmp_path):
    from tigris.analysis.validation import validate_operator_support

    ag = load_model(_qdq_float_bias_model(tmp_path, "Conv"))
    conv = next(op for op in ag.ops if op.op_type == "Conv")
    ag.weight_data[conv.inputs[2]] = ag.weight_data[conv.inputs[2]].astype(np.float32)
    ag.tensors[conv.inputs[2]].dtype = 1
    result = validate_operator_support(ag)
    assert not result.supported
    assert "bias" in result.describe()


def _qdq_linear_transposed_weight(tmp_path):
    """PyTorch's nn.Linear in QDQ form: the weight is dequantized as [out, in]
    and transposed before the Gemm."""
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.3, (4, 6)).astype(np.float32)
    w_scale = (np.abs(w).max(axis=1) / 127).astype(np.float32)
    w_q = np.clip(np.round(w / w_scale[:, None]), -127, 127).astype(np.int8)
    b_scale = np.float32(0.02) * w_scale
    initializers = [
        numpy_helper.from_array(np.array(0.02, np.float32), "in_scale"),
        numpy_helper.from_array(np.array(0, np.int8), "zp"),
        numpy_helper.from_array(w_q, "w_q"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(np.zeros(4, np.int8), "w_zp"),
        numpy_helper.from_array(np.round(rng.normal(0, 0.5, 4) / b_scale).astype(np.int32), "b_q"),
        numpy_helper.from_array(b_scale, "b_scale"),
        numpy_helper.from_array(np.zeros(4, np.int32), "b_zp"),
        numpy_helper.from_array(np.array(0.05, np.float32), "out_scale"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_scale", "zp"], ["in_q"]),
        helper.make_node("DequantizeLinear", ["in_q", "in_scale", "zp"], ["in_dq"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], axis=0),
        helper.make_node("Transpose", ["w_dq"], ["w_t"], perm=[1, 0]),
        helper.make_node("DequantizeLinear", ["b_q", "b_scale", "b_zp"], ["b_dq"], axis=0),
        helper.make_node("Gemm", ["in_dq", "w_t", "b_dq"], ["product"]),
        helper.make_node("QuantizeLinear", ["product", "out_scale", "zp"], ["out_q"]),
        helper.make_node("DequantizeLinear", ["out_q", "out_scale", "zp"], ["output"]),
    ]
    return save_onnx(
        tmp_path, "qdq_linear", nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers, opset=13, ir_version=8,
    )


def test_transposed_quantized_weight_folds_into_the_weight(tmp_path):
    ag = load_model(_qdq_linear_transposed_weight(tmp_path))
    assert [op.op_type for op in ag.ops] == ["Gemm"]
    gemm = ag.ops[0]
    weights = ag.weight_data[gemm.inputs[1]]
    # Gemm weights are canonical [out, in], quantized per output channel.
    assert weights.dtype == np.int8 and weights.shape == (4, 6)
    assert ag.tensors[gemm.inputs[1]].quant.scale.size == 4


@pytest.mark.usefixtures("require_host_library")
def test_transposed_quantized_weight_matches_onnx_runtime(tmp_path):
    import onnxruntime as ort

    from tigris.runtime import Session

    path = _qdq_linear_transposed_weight(tmp_path)
    values = np.random.default_rng(5).uniform(-1, 1, (1, 6)).astype(np.float32)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    expected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(
        None, {"input": values})[0]
    ag = plan_model(str(path), ("16K",))
    binary = tmp_path / "linear.tgrs"
    binary.write_bytes(emit_binary_bytes(ag))
    with Session(binary) as session:
        actual = session.run({"input": values})["output"]
    assert np.abs(np.round((actual - expected) / 0.05)).max() <= 1
