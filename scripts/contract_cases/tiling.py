"""Cases the planner bands along one axis."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model


def _banded_attention_case() -> ContractCase:
    """An attention region banded over its query axis.

    The band cuts the second to last axis of a rank-4 tensor, so the head axis
    is batch that the band spans rather than cuts, and the matrix product's
    second operand is read whole the way a weight is. Sized so the budget
    forces the band: the same graph at a roomy budget runs whole and exercises
    none of it.
    """
    heads, tokens, width = 4, 64, 16
    shape = [1, heads, tokens, width]
    model = _model(
        "banded_attention",
        [
            helper.make_node(
                "Transpose", ["input"], ["keys"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["input", "keys"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["output"], axis=-1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, heads, tokens, tokens])],
    )
    data = np.linspace(
        -1.0, 1.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_banded_attention",
        model,
        model,
        {"input": data},
        ("Transpose", "Transpose", "Transpose", "MatMul", "Softmax",
         "Transpose"),
        mem_budget="64K",
        expect_tiled=True,
    )


def _row_band_alias_case() -> ContractCase:
    """A banded stage that writes its output over its own input.

    The scores a Softmax normalizes are dead the moment it has read them, and
    a row band is gathered out of the input before any op runs and scattered
    back to the same rows, so one slow buffer can carry both tensors. The
    compiler sizes the slow pool on that rule and the runtime fills the pool
    on its own copy of it, so the slow budget here is the tight one: 96 KiB is
    what the pair costs shared, and 144 KiB is what it costs apart. A runtime
    that stops aliasing overruns the budget the compiler accepted.
    """
    heads, tokens, width = 4, 64, 16
    shape = [1, heads, tokens, width]
    model = _model(
        "row_band_alias",
        [
            helper.make_node(
                "Transpose", ["input"], ["keys"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["input", "keys"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
            helper.make_node("MatMul", ["probs", "input"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -1.0, 1.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_row_band_alias",
        model,
        model,
        {"input": data},
        ("Transpose", "Transpose", "Transpose", "MatMul", "Softmax", "MatMul",
         "Transpose"),
        mem_budget="32K",
        slow_budget="96K",
        expect_tiled=True,
    )


def _row_band_onto_output_case() -> ContractCase:
    """A banded stage whose input is also a model output.

    The Softmax is the last stage to read the scores, but the runtime keeps a
    model output until the run ends, so it never writes the Softmax result
    over them. The compiler has to count both buffers: 768 KiB, where sharing
    one would be 640 KiB. The plan runs in exactly the figure the compiler
    states.
    """
    rows, depth, width, out = 512, 16, 128, 64
    rng = np.random.default_rng(7)
    first = (rng.standard_normal((depth, width)) * 0.1).astype(np.float32)
    second = (rng.standard_normal((width, out)) * 0.1).astype(np.float32)
    model = _model(
        "row_band_onto_output",
        [
            helper.make_node("MatMul", ["input", "first"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
            helper.make_node("MatMul", ["probs", "second"], ["mixed"]),
            helper.make_node("Add", ["mixed", "skip"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [rows, depth]),
         helper.make_tensor_value_info("skip", TensorProto.FLOAT, [rows, out])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, out]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [rows, width])],
        [numpy_helper.from_array(first, "first"), numpy_helper.from_array(second, "second")],
    )
    inputs = {"input": rng.standard_normal((rows, depth)).astype(np.float32),
              "skip": rng.standard_normal((rows, out)).astype(np.float32)}
    return ContractCase(
        "float_row_band_onto_output",
        model,
        model,
        inputs,
        ("Gemm", "Softmax", "Gemm", "Add"),
        mem_budget="32K",
        expect_tiled=True,
        exact_slow=True,
    )


def _tiled_softmax_case() -> ContractCase:
    """Softmax on a stage the solver has to tile.

    Normalization runs along the final stored dimension and the tile cuts
    stored axis 1, so each tile holds whole rows. The budget is set well below
    the tensor so the plan cannot be a single tile: if the kernel ignored its
    tile geometry it would normalize the wrong span and the values would not
    match ONNX Runtime.
    """
    shape = [1, 6, 512]
    model = _model(
        "tiled_softmax",
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_softmax",
        model,
        model,
        {"input": data},
        ("Softmax",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _reshape_alias_release_case() -> ContractCase:
    """A stage whose reshapes hand their buffers on and then let go.

    Lowering a matrix product leaves a reshape on either side of it, and each
    one shares its input's buffer rather than copying. An executor that pins
    that buffer for the rest of the stage fills the fast pool with buffers
    nothing reads again and pushes every later output into slow memory. The
    slow budget here is what the stage costs when a shared buffer is released
    at the last read of either tensor holding it.
    """
    tokens, width, depth = 64, 64, 6
    rng = np.random.default_rng(0)
    nodes = []
    weights = []
    current = "input"
    for index in range(depth):
        weight = (rng.normal(size=(width, width)) * 0.2).astype(np.float32)
        weights.append(numpy_helper.from_array(weight, f"w{index}"))
        nodes.append(helper.make_node("MatMul", [current, f"w{index}"],
                                      [f"product{index}"]))
        nodes.append(helper.make_node("Sigmoid", [f"product{index}"],
                                      [f"gated{index}"]))
        current = f"gated{index}"
    nodes.append(helper.make_node("Add", [current, "input"], ["output"]))
    model = _model(
        "reshape_alias_release", nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        initializers=weights,
    )
    data = (rng.normal(size=(1, tokens, width)) * 0.5).astype(np.float32)
    return ContractCase(
        "float_reshape_alias_release",
        model,
        model,
        {"input": data},
        ("Transpose",)
        + ("Reshape", "Gemm", "Reshape", "Sigmoid") * depth
        + ("Add", "Transpose"),
        mem_budget="64K",
        slow_budget="32K",
    )


def _tiled_rank4_binary_case(*, op_type: str) -> ContractCase:
    """A rank-4 binary pointwise stage the solver has to tile.

    The compiler and the runtime each read the height-stripe contract, so a
    binary operator the compiler tiles has to load and run tiled. One case
    per binary operator checks that.
    """
    shape = [1, 4, 32, 32]
    model = _model(
        f"tiled_rank4_{op_type.lower()}",
        [helper.make_node(op_type, ["left", "right"], ["output"])],
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    count = int(np.prod(shape))
    return ContractCase(
        f"float_tiled_rank4_{op_type.lower()}",
        model,
        model,
        {
            "left": np.linspace(-1.0, 1.0, count, dtype=np.float32).reshape(shape),
            "right": np.linspace(0.5, -0.5, count, dtype=np.float32).reshape(shape),
        },
        (op_type,),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_softmax_rank4_case() -> ContractCase:
    """Softmax on a rank-4 stage the solver has to tile.

    The rank-3 twin above covers the NLC contract. This covers the NHWC one,
    which the compiler, the loader and the executor each state separately.
    """
    shape = [1, 4, 32, 32]
    model = _model(
        "tiled_softmax_rank4",
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_softmax_rank4",
        model,
        model,
        {"input": data},
        ("Softmax",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_last_axis_softmax_case(*, rank: int) -> ContractCase:
    """Softmax over the last axis on a stage the solver has to tile.

    Normalization runs along the model's own final axis, which is not the axis
    the runtime stores last, so the graph compiles to a conversion, the
    Softmax, and a conversion back. At this budget all three have to tile, and
    the two conversions tile along opposite transfers.

    The rank-4 shape is the one an attention block produces: the conversion
    there moves the channel axis past two spatial axes rather than one, which
    collapses to the same transpose only because those two keep their order.
    """
    shape = [1, 512, 6] if rank == 3 else [1, 6, 16, 16]
    name = f"tiled_last_axis_softmax_rank{rank}"
    model = _model(
        name,
        [helper.make_node("Softmax", ["input"], ["output"], axis=-1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        f"float_{name}",
        model,
        model,
        {"input": data},
        ("Transpose", "Softmax", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_layer_norm_case() -> ContractCase:
    """LayerNormalization on a stage the solver has to tile.

    A tile cuts an axis ahead of the one the kernel normalizes, so each tile
    holds whole normalization rows. If the kernel took its statistics over the
    wrong span the values would not match ONNX Runtime.
    """
    shape = [1, 256, 16]
    initializers = [
        numpy_helper.from_array(
            np.linspace(0.75, 1.25, shape[-1], dtype=np.float32), "gamma"),
        numpy_helper.from_array(
            np.linspace(-0.2, 0.2, shape[-1], dtype=np.float32), "beta"),
    ]
    model = _model(
        "tiled_layer_norm",
        [helper.make_node(
            "LayerNormalization", ["input", "gamma", "beta"], ["output"],
            axis=-1, epsilon=1e-5)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
        opset=17,
    )
    row = np.linspace(-3.0, 3.0, shape[-1], dtype=np.float32)
    gains = (1.0 + 0.01 * np.arange(shape[1], dtype=np.float32))[:, None]
    data = (row[None, None, :] * gains[None, :, :]).astype(np.float32)
    return ContractCase(
        "float_tiled_layer_norm",
        model,
        model,
        {"input": data},
        ("Transpose", "LayerNormalization", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_erf_case() -> ContractCase:
    """Erf on a stage the solver has to tile."""
    shape = [1, 8, 512]
    model = _model(
        "tiled_erf",
        [helper.make_node("Erf", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = np.linspace(
        -3.0, 3.0, int(np.prod(shape)), dtype=np.float32
    ).reshape(shape)
    return ContractCase(
        "float_tiled_erf",
        model,
        model,
        {"input": data},
        ("Erf",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_matrix_rows_case() -> ContractCase:
    """A projection pipeline the solver has to band along its rows.

    A matrix product against a constant reads one row to write one row, so a
    band of rows computes exactly the rows it holds. Rank 2 has no axis the
    height contract names, which is why this needs its own path: the
    projections of a transformer lower to exactly this shape.
    """
    rows, width, hidden = 256, 32, 64
    initializers = [
        numpy_helper.from_array(
            (np.arange(width * hidden, dtype=np.float32).reshape(width, hidden)
             / (width * hidden) - 0.5), "w1"),
        numpy_helper.from_array(
            (np.arange(hidden * width, dtype=np.float32).reshape(hidden, width)
             / (hidden * width) - 0.5), "w2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["input", "w1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["act"]),
        helper.make_node("MatMul", ["act", "w2"], ["output"]),
    ]
    model = _model(
        "tiled_matrix_rows",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, rows, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, rows, width])],
        initializers,
    )
    data = np.linspace(
        -1.0, 1.0, rows * width, dtype=np.float32
    ).reshape(1, rows, width)
    return ContractCase(
        "float_tiled_matrix_rows",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Relu",
         "Reshape", "Gemm", "Reshape", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )


def _tiled_attention_transpose_case() -> ContractCase:
    """A transpose the model itself asks for, on a stage that has to tile.

    An attention block transposes its keys between two operands that are both
    already in the model's own order, so no layout conversion is involved.
    The permutation of stored axes is the same one a conversion emits, which
    is what lets the same band serve it.
    """
    tokens, width = 256, 16
    weight = numpy_helper.from_array(
        (np.arange(width * width, dtype=np.float32).reshape(width, width)
         / (width * width) - 0.5), "wk")
    nodes = [
        helper.make_node("MatMul", ["input", "wk"], ["keys"]),
        helper.make_node("Transpose", ["keys"], ["output"], perm=[0, 2, 1]),
    ]
    model = _model(
        "tiled_attention_transpose",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, width, tokens])],
        [weight],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        "float_tiled_attention_transpose",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Transpose"),
        mem_budget="8K",
        expect_tiled=True,
    )
