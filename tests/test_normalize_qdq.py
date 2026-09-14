"""Tests for QDQ folding in the normalize pass."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model
from tigris.loaders.onnx.loader import load_model as load_raw
from tigris.loaders.onnx.normalize import normalize, _fold_qdq
from tigris.graph.ir import QuantParam
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.defs import NO_QUANT_PARAM


def _load_and_fold(path):
    """Load raw ONNX (no normalize) then apply full normalize."""
    ag = load_raw(path)
    return normalize(ag)


def test_qdq_fold_removes_ql_dql_ops(qdq_conv_path):
    """All QuantizeLinear and DequantizeLinear ops should be removed."""
    ag = load_raw(qdq_conv_path)
    qdq_types = {"QuantizeLinear", "DequantizeLinear"}
    assert any(op.op_type in qdq_types for op in ag.ops), "fixture should have Q/DQ ops"

    ag = _fold_qdq(ag)
    remaining = [op.op_type for op in ag.ops]
    assert not any(t in qdq_types for t in remaining), f"Q/DQ ops remain: {remaining}"


def test_qdq_fold_preserves_compute_ops(qdq_conv_path):
    """Conv should survive folding; Relu is absorbed as fused activation."""
    ag = _load_and_fold(qdq_conv_path)
    op_types = [op.op_type for op in ag.ops]
    assert "Conv" in op_types
    assert len(ag.ops) == 1  # Conv with fused Relu
    assert ag.ops[0].attrs.get("fused_activation") == "Relu"


def test_qdq_fold_sets_is_quantized(qdq_conv_path):
    """ag.is_quantized should be True after folding."""
    ag = _load_and_fold(qdq_conv_path)
    assert ag.is_quantized is True


def test_qdq_fold_weight_int8(qdq_conv_path):
    """After folding, weight data should be int8."""
    ag = _load_and_fold(qdq_conv_path)

    # Find the conv op's weight input
    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    w_name = conv_op.inputs[1]

    assert w_name in ag.weight_data
    assert ag.weight_data[w_name].dtype == np.int8


def test_qdq_fold_weight_quant_param(qdq_conv_path):
    """Weight tensor should have a QuantParam attached."""
    ag = _load_and_fold(qdq_conv_path)

    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    w_name = conv_op.inputs[1]
    w_info = ag.tensors[w_name]

    assert w_info.quant is not None
    assert isinstance(w_info.quant, QuantParam)
    assert w_info.quant.scale.shape[0] == 2  # per-channel, 2 output channels


def test_qdq_fold_activation_quant_param(qdq_conv_path):
    """Input activation should have a QuantParam from the input Q/DQ pair."""
    ag = _load_and_fold(qdq_conv_path)

    # The model input tensor should have quant param
    input_name = ag.model_inputs[0]
    input_info = ag.tensors[input_name]
    assert input_info.quant is not None
    assert input_info.quant.scale.size == 1  # per-tensor
    np.testing.assert_allclose(input_info.quant.scale, [0.05])


def test_qdq_fold_output_quant_param(qdq_conv_path):
    """Output of fused Conv+Relu should have quant param from the output Q/DQ."""
    ag = _load_and_fold(qdq_conv_path)

    # Relu is fused into Conv; the Conv output inherits the Relu output's quant param
    conv_op = next(op for op in ag.ops if op.op_type == "Conv")
    conv_out = conv_op.outputs[0]
    conv_info = ag.tensors[conv_out]
    assert conv_info.quant is not None
    np.testing.assert_allclose(conv_info.quant.scale, [0.1])


def test_qdq_fold_no_effect_on_float_model(linear_3op_path):
    """Float model should be unchanged by QDQ folding."""
    ag = load_raw(linear_3op_path)
    orig_ops = len(ag.ops)
    ag = normalize(ag)
    assert ag.is_quantized is False
    assert len(ag.ops) == orig_ops


def test_qdq_fold_scale_zp_cleaned(qdq_conv_path):
    """Scale/zero_point initializers should be removed from weight_data."""
    ag = _load_and_fold(qdq_conv_path)

    # Scale/zp tensor names should not remain in weight_data
    for name in ["inp_scale", "inp_zp", "w_scale", "w_zp", "out_scale", "out_zp"]:
        assert name not in ag.weight_data, f"{name} still in weight_data"


# Binary round-trip tests


def _full_pipeline_qdq(path):
    """Load QDQ model through full pipeline (normalize, lifetimes, memory)."""
    ag = load_raw(path)
    ag = normalize(ag)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    return ag


def test_qdq_binary_roundtrip(qdq_conv_path):
    """Quantized model emits valid binary with quant params section."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["model_name"] == "qdq_conv"
    assert plan["num_ops"] == 1  # Conv with fused Relu
    assert len(plan["quant_params"]) > 0


def test_qdq_binary_quant_param_content(qdq_conv_path):
    """Quant params have correct scale and zero_point values."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    # Find tensor with quant param
    tensors_with_qp = [t for t in plan["tensors"] if t["quant_param_idx"] != NO_QUANT_PARAM]
    assert len(tensors_with_qp) > 0

    # Check that quant param indices reference valid entries
    for t in tensors_with_qp:
        qp_idx = t["quant_param_idx"]
        assert qp_idx < len(plan["quant_params"])


def test_qdq_binary_weight_per_channel(qdq_conv_path):
    """Per-channel weight quant params have multipliers and shifts."""
    ag = _full_pipeline_qdq(qdq_conv_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    # The weight has per-channel quant (2 output channels)
    per_channel = [qp for qp in plan["quant_params"] if qp["num_channels"] > 1]
    assert len(per_channel) > 0

    qp = per_channel[0]
    assert qp["num_channels"] == 2
    assert "multipliers" in qp
    assert "shifts" in qp
    assert len(qp["multipliers"]) == 2
    assert len(qp["shifts"]) == 2
    # Multipliers should be positive non-zero
    assert all(m > 0 for m in qp["multipliers"])


def test_float_model_no_quant_params(linear_3op_path):
    """Float model should have no quant params in binary."""
    ag = load_raw(linear_3op_path)
    ag = normalize(ag)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

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
    graph = helper.make_graph(nodes, "qdq_gemm_bias", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"qdq_gemm_bias_{int(shared_scale)}.onnx"
    onnx.save(model, str(path))
    return path


def test_constant_add_becomes_the_operator_bias(tmp_path):
    """The float bias add is requantized into the product's int32 domain."""
    ag = load_model(_qdq_gemm_bias_model(tmp_path, shared_scale=False))

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    gemm = ag.ops[0]
    assert len(gemm.inputs) == 3
    assert gemm.outputs == ["output"]
    bias = ag.weight_data[gemm.inputs[2]]
    assert bias.dtype == np.int32
    # bias scale is input_scale * weight_scale = 0.25 * 0.25
    assert bias.tolist() == [8, -4]
    assert ag.tensors["output"].dtype == 3
    assert ag.tensors["output"].quant is not None
    # The model declares a float output, which the plan still presents.
    assert ag.model_output_dtypes == [1]


def test_shared_scale_initializer_still_folds_activations(tmp_path):
    """A scale shared by a weight pair and the activation pairs survives pass 1."""
    ag = load_model(_qdq_gemm_bias_model(tmp_path, shared_scale=True))

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    assert ag.tensors["input"].dtype == 3
    assert ag.tensors["output"].dtype == 3


def test_float_constant_add_becomes_the_gemm_bias(tmp_path):
    """A float bias Add folds into Gemm's bias slot, as the quantized one does.

    The two paths reach it differently: the quantized fold has to requantize
    into the accumulator domain, while this one just moves float data. Leaving
    the Add standing made the graph uncompilable, because its operand is
    per-channel and the Add kernel takes two operands of one shape.
    """
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
    graph = helper.make_graph(nodes, "float_gemm_bias", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "float_gemm_bias.onnx"
    onnx.save(model, str(path))

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
    graph = helper.make_graph([node], "product", inputs, [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"{op_type}_{int(weight_constant)}.onnx"
    onnx.save(model, str(path))
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
    graph = helper.make_graph(
        [node], "gemm_t", [X], [Y], [numpy_helper.from_array(values, "weight")]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "gemm_t.onnx"
    onnx.save(model, str(path))

    ag = load_model(path)

    assert np.array_equal(ag.weight_data["weight"], values)


def test_matmul_with_a_dynamic_operand_is_left_alone(tmp_path):
    """Without a constant matrix there is nothing to transpose ahead of time."""
    path, _ = _matmul_model(tmp_path, "MatMul", weight_constant=False)

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["MatMul"]


def test_matmul_weight_shared_with_another_op_is_left_alone(tmp_path):
    """Transposing in place would misstate the weight for the other reader."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 3])
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["product"]),
        helper.make_node("Add", ["product", "weight"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "shared", [X], [Y], [numpy_helper.from_array(values, "weight")]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "shared.onnx"
    onnx.save(model, str(path))

    ag = load_model(path)

    assert [op.op_type for op in ag.ops][0] == "MatMul"
    assert np.array_equal(ag.weight_data["weight"], values)


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
    graph = helper.make_graph(nodes, "qdq_matmul", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "qdq_matmul.onnx"
    onnx.save(model, str(path))

    ag = load_model(path)

    assert [op.op_type for op in ag.ops] == ["Gemm"]
    quant = ag.tensors["weight"].quant
    assert quant.axis == 0
    assert quant.scale.tolist() == [0.5, 0.25, 0.125]


# uint8 activations


def _quint8_model(tmp_path, weight_unsigned=False):
    """A Conv whose activations are stated as uint8, the ORT quantizer default."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    weight_values = (
        np.array([[[[200]]]], dtype=np.uint8)
        if weight_unsigned
        else np.array([[[[72]]]], dtype=np.int8)
    )
    initializers = [
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "act_scale"),
        numpy_helper.from_array(np.array(114, dtype=np.uint8), "act_zp"),
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
    graph = helper.make_graph(nodes, "quint8", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"quint8_{int(weight_unsigned)}.onnx"
    onnx.save(model, str(path))
    return path


def test_unsigned_activation_moves_to_the_signed_domain(tmp_path):
    """uint8 v and int8 v - 128 denote the same real value under shifted zero points."""
    ag = load_model(_quint8_model(tmp_path))

    activation = ag.tensors["input"]
    assert activation.dtype == 3  # INT8
    assert int(activation.quant.zero_point[0]) == 114 - 128
    assert float(activation.quant.scale[0]) == 0.25


def test_unsigned_weight_data_moves_with_its_zero_point(tmp_path):
    """A stored uint8 weight is restated, not reinterpreted."""
    ag = load_model(_quint8_model(tmp_path, weight_unsigned=True))

    weight = ag.weight_data["weight_q"]
    assert weight.dtype == np.int8
    assert weight.ravel().tolist() == [200 - 128]
    info = ag.tensors["weight_q"]
    assert info.dtype == 3
    assert int(info.quant.zero_point[0]) == 128 - 128


def test_unsigned_model_input_keeps_its_declared_interface(tmp_path):
    """The caller hands over what the model declares, whatever the plan stores."""
    ag = load_model(_quint8_model(tmp_path))

    assert ag.model_input_dtypes == [1]  # float32, as the model states
    assert ag.tensors["input"].dtype == 3  # int8, as the plan executes


def test_signed_activations_are_untouched(tmp_path):
    """An int8 graph keeps the zero points the model states."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    initializers = [
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "act_scale"),
        numpy_helper.from_array(np.array(-14, dtype=np.int8), "act_zp"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array([[[[72]]]], dtype=np.int8), "weight_q"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "w_zp"),
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
    graph = helper.make_graph(nodes, "signed", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "signed.onnx"
    onnx.save(model, str(path))

    ag = load_model(path)

    assert int(ag.tensors["input"].quant.zero_point[0]) == -14
    assert ag.weight_data["weight_q"].ravel().tolist() == [72]
