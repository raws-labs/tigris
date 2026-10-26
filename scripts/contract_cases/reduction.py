"""Reduction, arg, cumulative sum and pooling cases."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _qdq_operator_model, _scalars


def _linear_spatial_max_case(*, quantized: bool) -> ContractCase:
    shape = [1, 3, 2, 4]
    source, target = "input", "output"
    nodes, initializers = [], []
    if quantized:
        initializers = _scalars(x=(0.03125, -17), y=(1 / 16, 127))
        nodes += _qdq(source, "x_s", "x_z", "x")
        source, target = "x", "raw"
    nodes += [helper.make_node("LogSoftmax", [source], ["normalized"], axis=-1)]
    normalized = "normalized"
    if quantized:
        nodes += _qdq(normalized, "y_s", "y_z", "normalized_qdq")
        normalized = "normalized_qdq"
    nodes += [helper.make_node("ReduceMax", [normalized], [target], axes=[2, 3], keepdims=1)]
    if quantized:
        nodes += _qdq(target, "y_s", "y_z", "output")
    label = f"{'int8' if quantized else 'float'}_linear_spatial_max"
    model = _model(label, nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])], initializers)
    data = np.random.default_rng(416).uniform(-2, 2, size=shape).astype(np.float32)
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Transpose", "LogSoftmax", "Transpose", "GlobalMaxPool"),
                        reference_unoptimized=True)


def _reduction_case(kind: str, axis: int, *, quantized: bool, keep: bool = True,
                    exclusive: bool = False, reverse: bool = False, tiled: bool = False,
                    leading: bool = False) -> ContractCase:
    shape = [2, 137 if tiled else 7, 4]
    if kind == "ReduceMean" and tiled:
        shape = [2, 4, 137]
        axis = 1
    if leading:
        shape = [37, 4, 1] if kind == "ReduceMean" else [37, 1, 4]
    output_shape = shape.copy()
    if kind != "CumSum":
        if keep:
            output_shape[axis] = 1
        else:
            output_shape.pop(axis)
    data = np.random.default_rng(411).uniform(-1, 1, size=shape).astype(np.float32)
    nodes, initializers = [], []
    source = "input"
    if quantized:
        iq = (0.03125, 0 if kind == "CumSum" else -17)
        oq = iq if kind in {"ReduceMax", "ReduceMin"} else (0.0625, 31)
        initializers += _scalars(x=iq, y=oq)
        nodes += _qdq(source, "x_s", "x_z", "x")
        source = "x"
    target = "raw" if quantized else "output"
    if kind in {"CumSum", "ReduceSum"}:
        initializers.append(numpy_helper.from_array(
            np.array(axis - 3 if kind == "CumSum" else [axis - 3], np.int64), "axis"))
        attrs = {"exclusive": int(exclusive), "reverse": int(reverse)} if kind == "CumSum" else {"keepdims": int(keep)}
        nodes += [helper.make_node(kind, [source, "axis"], [target], **attrs)]
    else:
        nodes += [helper.make_node(kind, [source], [target], axes=[axis - 3], keepdims=int(keep))]
    if quantized:
        nodes += _qdq(target, "y_s", "y_z", "output")
    label = (f"{'int8' if quantized else 'float'}_{kind.lower()}_axis{axis}"
             f"_keep{int(keep)}_exclusive{int(exclusive)}_reverse{int(reverse)}")
    label += "_leading" if leading else "_tiled" if tiled else ""
    model = _model(label, nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)], initializers)
    operators = ("Transpose", kind, "Transpose") if keep or kind == "CumSum" else ("Transpose", kind)
    if kind == "ReduceMean":
        operators = (kind,)
    return ContractCase(label, model, copy.deepcopy(model), {"input": data}, operators,
                        mem_budget=("128" if quantized else "384") if leading else "256" if tiled else "4K",
                        slow_budget="16K" if tiled else None,
                        expect_tiled=tiled, reference_unoptimized=True, exact_untiled=leading)


def _reduce_all_case(axis: int, keep: bool, tiled: bool = False,
                   leading: bool = False) -> ContractCase:
    """All of one bool axis, stated as the uint8 minimum ONNX can express."""
    shape = [37, 1, 4] if leading else [2, 137 if tiled else 7, 4]
    output_shape = shape.copy()
    if keep:
        output_shape[axis] = 1
    else:
        output_shape.pop(axis)
    data = np.random.default_rng(417 + axis).uniform(size=shape) > 0.15
    nodes = [helper.make_node("Cast", ["input"], ["widened"], to=TensorProto.UINT8),
             helper.make_node("ReduceMin", ["widened"], ["lowest"], axes=[axis - 3], keepdims=int(keep)),
             helper.make_node("Cast", ["lowest"], ["output"], to=TensorProto.BOOL)]
    label = f"bool_reduceall_axis{axis}_keep{int(keep)}"
    label += "_leading" if leading else "_tiled" if tiled else ""
    model = _model(label, nodes,
                   [helper.make_tensor_value_info("input", TensorProto.BOOL, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.BOOL, output_shape)], [])
    operators = ("Transpose", "ReduceAll", "Transpose") if keep else ("Transpose", "ReduceAll")
    return ContractCase(label, model, copy.deepcopy(model), {"input": data}, operators,
                        mem_budget="128" if leading else "256" if tiled else "4K",
                        slow_budget="16K" if tiled else None,
                        expect_tiled=tiled, reference_unoptimized=True, exact_untiled=leading)


def _arg_case(kind: str, axis: int, quantized: bool, keep: bool, tiled: bool = False,
                   leading: bool = False) -> ContractCase:
    shape = [37, 1, 4] if leading else [2, 137 if tiled else 7, 4]
    target = shape.copy()
    if keep:
        target[axis] = 1
    else:
        target.pop(axis)
    data = (np.arange(np.prod(shape)).reshape(shape) % 5).astype(np.float32)
    nodes, initializers = [], []
    source = "input"
    if quantized:
        initializers = _scalars(x=(0.125, -17))
        nodes = _qdq(source, "x_s", "x_z", "x")
        source = "x"
    nodes.append(helper.make_node(kind, [source], ["output"], axis=axis - 3, keepdims=int(keep)))
    label = f"{kind.lower()}_axis{axis}_int8{int(quantized)}_keep{int(keep)}"
    label += "_leading" if leading else "_tiled" if tiled else ""
    model = _model(label, nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.INT64, target)], initializers)
    return ContractCase(label, model, copy.deepcopy(model), {"input": data}, ("Transpose", kind),
                        mem_budget=("128" if quantized else "384") if leading else "256" if tiled else "4K",
                        slow_budget="16K" if tiled else None,
                        expect_tiled=tiled, reference_unoptimized=True, exact_untiled=leading)


def _whole_map_average_pool_case() -> ContractCase:
    """An AveragePool whose window is the whole map, as F.avg_pool2d writes it.

    count_include_pad=1 is PyTorch's default and changes nothing without
    padding. The op becomes the global pool, which streams in bands instead of
    needing the whole input at once.
    """
    channels, side = 8, 12
    rng = np.random.default_rng(25)
    data = rng.normal(size=(1, channels, side, side)).astype(np.float32)
    model = _model(
        "whole_map_average_pool",
        [helper.make_node("AveragePool", ["input"], ["output"], kernel_shape=[side, side],
                          strides=[side, side], count_include_pad=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 1, 1])],
    )
    return ContractCase("float_whole_map_average_pool", model, copy.deepcopy(model), {"input": data},
                        ("GlobalAveragePool",), mem_budget="2K", expect_tiled=True)


def _reduce_mean_case(*, quantized: bool, keepdims: bool) -> ContractCase:
    """A mean over the token axis, which is what a pooled sequence head is.

    The encoder leaves one vector per token and the head wants one vector for
    the sequence, so the mean collapses the rows of a rank-3 tensor. Both the
    kept and the dropped row axis are here because the two write the same
    bytes and only the declared rank differs, which is exactly the kind of
    difference the loader has to police rather than guess at.
    """
    tokens, width = 12, 8
    in_shape = [1, tokens, width]
    out_shape = [1, 1, width] if keepdims else [1, width]
    kd = 1 if keepdims else 0
    name = f"reduce_mean_{'keepdims' if keepdims else 'flat'}"
    if quantized:
        in_scale, out_scale = 0.05, 0.03
        initializers = [
            numpy_helper.from_array(
                np.array(in_scale, dtype=np.float32), "in_s"),
            numpy_helper.from_array(np.array(2, dtype=np.int8), "in_z"),
            numpy_helper.from_array(
                np.array(out_scale, dtype=np.float32), "out_s"),
            numpy_helper.from_array(np.array(-5, dtype=np.int8), "out_z"),
        ]
        nodes = [
            helper.make_node(
                "QuantizeLinear", ["input", "in_s", "in_z"], ["iq"]),
            helper.make_node(
                "DequantizeLinear", ["iq", "in_s", "in_z"], ["idq"]),
            helper.make_node(
                "ReduceMean", ["idq"], ["raw"], axes=[1], keepdims=kd),
            helper.make_node(
                "QuantizeLinear", ["raw", "out_s", "out_z"], ["oq"]),
            helper.make_node(
                "DequantizeLinear", ["oq", "out_s", "out_z"], ["output"]),
        ]
        rng = np.random.default_rng(11)
        data = (rng.integers(-100, 100, size=in_shape).astype(np.float32)
                * in_scale)
        label = f"int8_{name}"
    else:
        initializers = []
        nodes = [
            helper.make_node(
                "ReduceMean", ["input"], ["output"], axes=[1], keepdims=kd),
        ]
        data = np.linspace(
            -1.5, 1.5, int(np.prod(in_shape)), dtype=np.float32
        ).reshape(in_shape)
        label = f"float_{name}"
    model = _model(
        label,
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    return ContractCase(
        label,
        model,
        model,
        {"input": data},
        ("ReduceMean",),
    )


def _nonadjacent_mean_case(suffix: str) -> ContractCase:
    from tigris.frontends import tflite

    path = Path(__file__).resolve().parents[2] / "tests/fixtures/tflite/ops" / f"float_mean_nonadjacent_{suffix}.tflite"
    model = tflite.to_onnx(path.read_bytes(), path.stem)
    samples = np.load(path.with_suffix(".npz"))
    x = samples["input_0"][0]
    native = x.transpose(0, x.ndim - 1, *range(1, x.ndim - 1))
    axes = [0, 1] if x.ndim == 3 else [1, 2]
    keep = suffix in {"keep", "negative"}
    reference = _model(path.stem + "_reference", [helper.make_node(
        "ReduceMean", [model.graph.input[0].name], [model.graph.output[0].name],
        axes=axes, keepdims=int(keep))], list(model.graph.input), list(model.graph.output))
    budget = "2K" if suffix == "keep" else "512" if suffix == "tokens" else "1536"
    # The batch of one makes the token mean adjacent: only the layout move remains.
    transposes = 1 if suffix == "tokens" else 4 + int(keep)
    expected = ("Transpose",) * transposes + ("Reshape", "ReduceMean", "Reshape")
    return ContractCase(path.stem, model, reference, {model.graph.input[0].name: native},
                        expected, mem_budget=budget, expect_tiled=True)


def _reduce_case(*, maximum: bool) -> ContractCase:
    """A spatial ReduceMean or ReduceMax must execute as its global pool."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 1, 1]
    )
    operator = "ReduceMax" if maximum else "ReduceMean"
    pool = "GlobalMaxPool" if maximum else "GlobalAveragePool"
    model = _model(
        f"{operator.lower()}_to_pool",
        [
            helper.make_node(
                operator,
                ["input"],
                ["output"],
                axes=[2, 3],
                keepdims=1,
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        f"float_{operator.lower()}_to_pool",
        model,
        model,
        {
            "input": np.arange(
                18, dtype=np.float32
            ).reshape(1, 3, 2, 3)
        },
        (pool,),
    )


def _global_max_pool_case() -> ContractCase:
    """GlobalMaxPool, the counterpart of GlobalAveragePool."""
    model = _model(
        "global_max_pool",
        [helper.make_node("GlobalMaxPool", ["input"], ["output"])],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 3, 2, 4])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    data = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(1, 3, 2, 4)
    return ContractCase(
        "float_global_max_pool",
        model,
        model,
        {"input": data},
        ("GlobalMaxPool",),
    )


def _tiled_pool_case() -> ContractCase:
    """A pool case large enough to require standalone tiled execution."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 32, 32]
    )
    model = _model(
        "tiled_pool",
        [
            helper.make_node(
                "AveragePool",
                ["input"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_tiled_averagepool",
        model,
        model,
        {"input": np.arange(4096, dtype=np.float32).reshape(1, 1, 64, 64) / 64.0},
        ("AveragePool",),
        mem_budget="4K",
        expect_tiled=True,
    )


def _grouped_global_reduction_case(quantized: bool, width_only: bool) -> ContractCase:
    shape = [2, 13, 1, 41] if width_only else [2, 31, 3, 19]
    result_shape = [shape[0], shape[1], 1, 1]
    values = ((np.arange(np.prod(shape)).reshape(shape) % 31) - 15).astype(np.float32) * 0.125
    nodes, constants = [], []
    source, target = "input", "output"
    if quantized:
        constants = _scalars(x=(0.125, -17), y=(0.25, 9))
        nodes += _qdq("input", "x_s", "x_z", "x")
        source, target = "x", "raw"
    nodes.append(helper.make_node("GlobalAveragePool", [source], [target]))
    if quantized:
        nodes += _qdq("raw", "y_s", "y_z", "output")
    model = _model("grouped_global_reduction", nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, result_shape)], constants)
    return ContractCase(f"{'int8' if quantized else 'float'}_grouped_global_reduction_{int(width_only)}",
                        model, model, {"input": values}, ("GlobalAveragePool",),
                        mem_budget="128" if quantized or width_only else "256", slow_budget="16K",
                        expect_tiled=True, expect_grouped=True, exact_untiled=True,
                        tiled_operators=("GlobalAveragePool",), reference_unoptimized=True)


def _tiled_global_reduction_case() -> ContractCase:
    """A GlobalAveragePool too large for the budget, so it reduces in bands."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 8, 64, 64]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 1, 1]
    )
    model = _model(
        "tiled_global_reduction",
        [
            helper.make_node(
                "GlobalAveragePool", ["input"], ["output"], name="gap"
            )
        ],
        [model_input],
        [model_output],
    )
    # 4096 samples per channel, kept inside [-1, 1] so the running sum stays
    # small: a mean over values of magnitude 30 accumulates past 1e5, where
    # float32 summation order alone moves the result by more than the 1e-5
    # bound and the case would measure numpy against ONNX Runtime rather than
    # tiled against untiled.
    values = np.arange(1 * 8 * 64 * 64, dtype=np.float32).reshape(1, 8, 64, 64)
    return ContractCase(
        "float_tiled_global_reduction",
        model,
        model,
        {"input": (values % 17.0) * 0.125 - 1.0},
        ("GlobalAveragePool",),
        mem_budget="16K",
        expect_tiled=True,
    )


def _qdq_tiled_global_reduction_case() -> ContractCase:
    """The int8 sibling: the banded sum must requantize exactly once."""
    model = _qdq_operator_model("qdq_tiled_global_reduction", "GlobalAveragePool",
                                [1, 8, 64, 64], [1, 8, 1, 1], (0.25, -3), (0.125, 7), name="gap")
    values = np.arange(1 * 8 * 64 * 64, dtype=np.float32).reshape(1, 8, 64, 64)
    return ContractCase(
        "int8_tiled_global_reduction", model, model, {"input": (values % 61) * 0.25 - 7.5},
        ("GlobalAveragePool",), mem_budget="8K", expect_tiled=True,
    )


def _tiled_ceil_pool_case() -> ContractCase:
    """MaxPool -> AveragePool with ceil_mode, each rounding up to one more
    window, tiled so the end padding that states it reaches the last band."""
    model = _model(
        "tiled_ceil_pool",
        [
            helper.make_node("MaxPool", ["input"], ["max"], kernel_shape=[3, 3],
                             strides=[2, 2], ceil_mode=1),
            helper.make_node("AveragePool", ["max"], ["output"], kernel_shape=[3, 3],
                             strides=[2, 2], ceil_mode=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 16, 16])],
    )
    return ContractCase(
        "float_tiled_ceil_pool",
        model,
        model,
        {"input": np.linspace(-2.0, 3.0, 8192, dtype=np.float32).reshape(1, 2, 64, 64)},
        ("MaxPool", "AveragePool"),
        mem_budget="4K",
        expect_tiled=True,
    )
