"""MatMul and Gemm cases."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _vi


def _float_gemm_bias_add_case() -> ContractCase:
    """A float Gemm whose bias arrives as a separate Add, the rank-2 form.

    The operand is per-channel against a [1, C] product, so the Add kernel
    cannot take it as written; the bias slot can. ONNX Runtime evaluates the
    two-op graph and TiGrIS must match it with one op.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([[0.5, -0.25, 0.75, 0.25], [-0.5, 0.25, 0.5, -0.75]],
                     dtype=np.float32),
            "weight",
        ),
        numpy_helper.from_array(
            np.array([0.5, -0.25], dtype=np.float32), "bias"),
    ]
    model = _model(
        "float_gemm_bias_add",
        [
            helper.make_node("Gemm", ["input", "weight"], ["product"], transB=1),
            helper.make_node("Add", ["product", "bias"], ["output"]),
        ],
        [model_input],
        [model_output],
        initializers,
    )
    return ContractCase(
        "float_gemm_bias_add",
        model,
        model,
        {"input": np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)},
        ("Gemm",),
    )


def _batched_matmul_case() -> ContractCase:
    """A rank-3 constant-weight MatMul, the per-position linear layer.

    ONNX Runtime evaluates the batched product directly. TiGrIS makes the
    operand linear, collapses its leading axes and runs the fully-connected
    kernel, so this checks the layout conversion and the lowering together.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 5, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 5, 3]
    )
    weight = numpy_helper.from_array(
        np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(4, 3), "weight"
    )
    model = _model(
        "batched_matmul",
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        [model_input],
        [model_output],
        [weight],
    )
    return ContractCase(
        "float_batched_matmul",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 20, dtype=np.float32
            ).reshape(1, 5, 4)
        },
        # Model boundaries keep the channels-last convention callers rely on,
        # so reaching a linear operand costs a conversion at each end. Those
        # Transposes are the layout change made explicit.
        ("Transpose", "Reshape", "Gemm", "Reshape", "Transpose"),
    )


def _dynamic_matmul_case(*, batched: bool) -> ContractCase:
    """A matrix product of two activations, which has no constant to fold.

    Neither operand is a weight, so this is the form the fully-connected kernel
    cannot express and the MatMul kernel exists for. The batched variant also
    exercises the layout conversion: rank-3 model boundaries keep the
    channels-last convention, so reaching the model's own axis order costs a
    Transpose on each operand and one on the result.
    """
    lhs_shape = [2, 3, 4] if batched else [3, 4]
    rhs_shape = [2, 4, 2] if batched else [4, 2]
    out_shape = [2, 3, 2] if batched else [3, 2]
    model = _model(
        "dynamic_matmul",
        [helper.make_node("MatMul", ["lhs", "rhs"], ["output"])],
        [
            helper.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs_shape),
            helper.make_tensor_value_info("rhs", TensorProto.FLOAT, rhs_shape),
        ],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
    )
    lhs = np.linspace(
        -1.0, 1.0, int(np.prod(lhs_shape)), dtype=np.float32
    ).reshape(lhs_shape)
    rhs = np.linspace(
        0.5, -0.5, int(np.prod(rhs_shape)), dtype=np.float32
    ).reshape(rhs_shape)
    operators = (
        ("Transpose", "Transpose", "MatMul", "Transpose")
        if batched else ("MatMul",)
    )
    return ContractCase(
        f"float_dynamic_matmul{'_batched' if batched else ''}",
        model,
        model,
        {"lhs": lhs, "rhs": rhs},
        operators,
    )


def _gemm_scaled_case() -> ContractCase:
    """A Gemm carrying alpha and beta, folded into the constants they scale.

    The plan has no field for either and the kernel computes Y = X * W^T + B,
    so without the fold this compiles and returns a silently wrong answer.
    ONNX Runtime applies both, which is what makes this case decisive.
    """
    weight = numpy_helper.from_array(
        np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(4, 3), "weight")
    bias = numpy_helper.from_array(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "bias")
    model = _model(
        "gemm_scaled",
        [
            helper.make_node(
                "Gemm", ["input", "weight", "bias"], ["output"],
                alpha=2.0, beta=3.0, transB=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [weight, bias],
    )
    return ContractCase(
        "float_gemm_scaled",
        model,
        model,
        {"input": np.array([[1.0, -2.0, 0.5]], dtype=np.float32)},
        ("Gemm",),
    )


def _matmul_case() -> ContractCase:
    """A rank-2 MatMul against a constant weight.

    ONNX states the product with the weight as [K, N] while the kernels index
    it as [OC, IC], so the compiler transposes the constant and relabels the
    operator. A deliberately asymmetric weight makes a missed transpose change
    the result rather than hide in a symmetric matrix.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    weight = numpy_helper.from_array(
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ),
        "weight",
    )
    model = _model(
        "matmul",
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        [model_input],
        [model_output],
        [weight],
    )
    return ContractCase(
        "float_matmul",
        model,
        model,
        {"input": np.array([[0.5, -1.0, 2.0, -0.25]], dtype=np.float32)},
        ("Gemm",),
    )


def _gemm_no_transpose_case() -> ContractCase:
    """A Gemm that leaves transB at its default.

    The weight is then [K, N] like MatMul's, so it needs the same transpose
    before it means what the kernels compute.
    """
    initializers = [
        numpy_helper.from_array(np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32), "weight"),
        numpy_helper.from_array(np.array([0.5, -1.5, 2.0], dtype=np.float32), "bias"),
    ]
    model = _model(
        "gemm_no_transpose", [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
        [_vi("input", [1, 4])], [_vi("output", [1, 3])], initializers,
    )
    return ContractCase(
        "float_gemm_no_transpose", model, model,
        {"input": np.array([[0.5, -1.0, 2.0, -0.25]], dtype=np.float32)},
        ("Gemm",),
    )
