"""Transpose, reshape, gather, concat, split, pad and resize cases."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars


def _fixed_axis_transpose_case() -> ContractCase:
    shape, perm = [1, 2, 5, 17, 3], [0, 1, 4, 3, 2]
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.125
    model = _model("transpose_fixed_axis", [helper.make_node("Transpose", ["x"], ["y"], perm=perm)],
                   [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, [shape[i] for i in perm])])
    return ContractCase("transpose_fixed_axis", model, model, {"x": values}, ("Transpose",),
                        mem_budget="512", slow_budget="16K", expect_tiled=True)


def _reshape_alias_case(*, spatial: bool) -> ContractCase:
    """A reshape that moves no byte, so the two tensors share a buffer.

    The linear case regroups a matrix's shape; the spatial one regroups a
    feature map's spatial axes into patches, which is what a vision
    transformer's unfold is. Both leave the stored bytes untouched, so the
    executor gives the output the input's buffer and the compiler counts one
    allocation. A wrong answer here means the two do not agree on when that is
    safe.
    """
    if spatial:
        channels, side = 8, 4
        nodes = [
            helper.make_node(
                "Conv", ["input", "w"], ["features"], kernel_shape=[1, 1]),
            helper.make_node("Reshape", ["features", "shape"], ["patches"]),
            helper.make_node("Relu", ["patches"], ["output"]),
        ]
        initializers = [
            numpy_helper.from_array(
                (np.random.default_rng(0).normal(
                    size=(channels, channels, 1, 1)) * 0.3
                 ).astype(np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, channels, side * side, 1], np.int64), "shape"),
        ]
        in_shape = [1, channels, side, side]
        out_shape = [1, channels, side * side, 1]
    else:
        tokens, width = 8, 8
        nodes = [
            helper.make_node("MatMul", ["input", "w"], ["projected"]),
            helper.make_node("Reshape", ["projected", "shape"], ["heads"]),
            helper.make_node("Erf", ["heads"], ["output"]),
        ]
        initializers = [
            numpy_helper.from_array(
                (np.random.default_rng(0).normal(size=(width, width)) * 0.3
                 ).astype(np.float32), "w"),
            numpy_helper.from_array(
                np.array([1, tokens, 2, width // 2], np.int64), "shape"),
        ]
        in_shape = [1, tokens, width]
        out_shape = [1, tokens, 2, width // 2]
    kind = "spatial" if spatial else "linear"
    model = _model(
        f"reshape_alias_{kind}",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    data = np.linspace(
        -1.5, 1.5, int(np.prod(in_shape)), dtype=np.float32
    ).reshape(in_shape)
    ops = (("Conv", "Reshape", "Relu") if spatial
           else ("Transpose", "Reshape", "Gemm", "Reshape", "Reshape", "Erf",
                 "Transpose"))
    return ContractCase(
        f"float_reshape_alias_{kind}",
        model,
        model,
        {"input": data},
        ops,
    )


def _reshape_group_case() -> ContractCase:
    """Three reshapes of one matrix product, each read at a different step.

    All three share the product's buffer, and releasing one of them releases
    it for all. The buffer has to stay until the last read of any of them,
    including a member the stage walks before the others.
    """
    tokens, width = 16, 32
    rng = np.random.default_rng(0)
    nodes = [
        helper.make_node("MatMul", ["x", "wm"], ["p"]),
        helper.make_node("Reshape", ["p", "h1"], ["q1"]),
        helper.make_node("Reshape", ["p", "h2"], ["q2"]),
        helper.make_node("Reshape", ["p", "h3"], ["q3"]),
        helper.make_node("Erf", ["q3"], ["e3"]),
        helper.make_node("Reshape", ["e3", "back"], ["b3"]),
        helper.make_node("Relu", ["q1"], ["e1"]),
        helper.make_node("Reshape", ["e1", "back"], ["b1"]),
        helper.make_node("Reshape", ["q2", "back"], ["b2"]),
        helper.make_node("Add", ["b1", "b3"], ["s"]),
        helper.make_node("Mul", ["s", "b2"], ["s2"]),
        helper.make_node("MatMul", ["s2", "wo"], ["y"]),
    ]
    shapes = {"h1": [1, tokens, 4, width // 4], "h2": [1, tokens * width],
              "h3": [1, tokens // 2, 2 * width], "back": [1, tokens, width]}
    model = _model(
        "reshape_group",
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array((rng.normal(size=(width, width)) * 0.3).astype(np.float32), name)
         for name in ("wm", "wo")]
        + [numpy_helper.from_array(np.array(shape, np.int64), name) for name, shape in shapes.items()],
    )
    data = np.linspace(-1.5, 1.5, tokens * width, dtype=np.float32).reshape(1, tokens, width)
    return ContractCase(
        "float_reshape_group", model, model, {"x": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Reshape", "Relu", "Reshape", "Reshape", "Reshape",
         "Reshape", "Erf", "Reshape", "Add", "Mul", "Reshape", "Gemm", "Reshape", "Transpose"),
        mem_budget="8K", slow_budget="1M",
    )


def _flattened_token_head_case() -> ContractCase:
    """A sequence flattened whole and fed to a matrix product.

    A classifier or forecast head flattens every token into one vector. When
    the tensor is one the runtime stores channels-last, the flat order differs
    from the model's and the weight's columns are permuted to compensate. A
    tensor that states its own axis order is already stored the way the model
    states it, so permuting it there is a silent wrong answer: the rank is the
    same, only the layout differs.
    """
    tokens, width, outputs = 9, 16, 5
    rng = np.random.default_rng(13)
    project = (rng.normal(size=(width, width)) * 0.3).astype(np.float32)
    head = (rng.normal(size=(outputs, tokens * width)) * 0.05).astype(np.float32)
    bias = (rng.normal(size=(outputs,)) * 0.1).astype(np.float32)
    nodes = [
        # The product is what makes the sequence state its own axis order.
        helper.make_node("MatMul", ["input", "w"], ["tokens"]),
        helper.make_node("Reshape", ["tokens", "flat"], ["vector"]),
        helper.make_node(
            "Gemm", ["vector", "head", "bias"], ["output"],
            alpha=1.0, beta=1.0, transB=1),
    ]
    model = _model(
        "flattened_token_head",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, outputs])],
        [numpy_helper.from_array(project, "w"),
         numpy_helper.from_array(
             np.array([1, tokens * width], np.int64), "flat"),
         numpy_helper.from_array(head, "head"),
         numpy_helper.from_array(bias, "bias")],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        "float_flattened_token_head",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Reshape", "Gemm"),
    )


def _bilinear_upsample_case(*, tiled: bool = False) -> ContractCase:
    """Bilinear upsampling, which is how a dense prediction returns to size.

    It is its own operator rather than a mode on the nearest one, so a runtime
    that does not carry it refuses the plan instead of quietly resampling a
    different way. The coordinates are half-pixel: an output sample sits at
    (o + 0.5) / scale - 0.5 and the border clamps.
    """
    channels, side = 3, 6
    rng = np.random.default_rng(1)
    model = _model(
        "bilinear_upsample",
        [
            helper.make_node(
                "Resize", ["input", "roi", "scales"], ["output"],
                mode="linear"),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, channels, 2 * side, 2 * side])],
        initializers=[
            numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
            numpy_helper.from_array(
                np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
        ],
    )
    data = (rng.normal(size=(1, channels, side, side))).astype(np.float32)
    if tiled:
        # In bands, every band after the first starts a source row ahead of
        # out / scale, which the band has to reach.
        return ContractCase(
            "float_bilinear_upsample_tiled", model, model, {"input": data},
            ("ResizeLinear",), mem_budget="2K", expect_tiled=True,
        )
    return ContractCase(
        "float_bilinear_upsample",
        model,
        model,
        {"input": data},
        ("ResizeLinear",),
    )


def _resize_case(mode: str, coordinate: str, geometry: str, quantized: bool) -> ContractCase:
    nearest = "round_prefer_ceil" if coordinate in {"align_corners", "half_pixel"} else "floor"
    ih, iw = (17, 9) if geometry == "down" else (7, 9)
    oh, ow = (7, 5) if geometry == "down" else (11, 4)
    scales = np.array([1, 1, 2.35, 0.7], np.float32) if geometry == "scales" else None
    if scales is not None:
        oh, ow = (int(np.floor(np.float32(n) * s)) for n, s in zip((ih, iw), scales[2:]))
    shape, target = [1, 4, ih, iw], [1, 4, oh, ow]
    name = f"{'int8' if quantized else 'float'}_resize_{mode}_{coordinate}_{geometry}"
    initializers = [numpy_helper.from_array(scales, "scales") if scales is not None else
                    numpy_helper.from_array(np.array(target, np.int64), "sizes")]
    nodes = _qdq("input", "q_s", "q_z", "x") if quantized else []
    nodes.append(helper.make_node("Resize", ["x" if quantized else "input", "", "scales"]
                                  if scales is not None else ["x" if quantized else "input", "", "", "sizes"],
                                  ["raw" if quantized else "output"], mode=mode,
                                  coordinate_transformation_mode=coordinate, nearest_mode=nearest))
    if quantized:
        initializers += _scalars(q=(0.125, -17))
        nodes += _qdq("raw", "q_s", "q_z", "output")
    model = _model(name, nodes, [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, target)], initializers)
    data = np.random.default_rng(1709).uniform(-5, 5, shape).astype(np.float32)
    return ContractCase(name, model, copy.deepcopy(model), {"input": data},
                        ("ResizeLinear" if mode == "linear" else "Resize",),
                        mem_budget="256" if quantized else "1K", expect_tiled=True)


def _reshape_band_case(quantized: bool, merge: bool, contiguous: bool = False) -> ContractCase:
    source, target = ([2, 33, 4], [2, 11, 12]) if merge else ([2, 11, 12], [2, 33, 4])
    if contiguous:
        source, target = ([36, 1, 4], [1, 6, 6, 4]) if merge else ([1, 6, 6, 4], [36, 1, 4])
    data = ((np.arange(np.prod(source)).reshape(source) % 101) - 50).astype(np.float32) * 0.125
    initializers = [numpy_helper.from_array(np.array(target, np.int64), "shape")]
    nodes = []
    if quantized:
        initializers += _scalars(x=(0.125, -17), y=(0.125, -17))
        nodes += _qdq("input", "x_s", "x_z", "x")
    nodes.append(helper.make_node("Reshape", ["x" if quantized else "input", "shape"],
                                  ["raw" if quantized else "output"]))
    if quantized:
        nodes += _qdq("raw", "y_s", "y_z", "output")
    model = _model("reshape_band", nodes,
                   [helper.make_tensor_value_info("input", TensorProto.FLOAT, source)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, target)], initializers)
    label = f"{'int8' if quantized else 'float'}_reshape_{'merge' if merge else 'split'}"
    if contiguous:
        label += "_contiguous"
    # Budgets also fit interface transposes; the 576/144-byte reshape still tiles.
    return ContractCase(label,
                        model, model, {"input": data}, ("Transpose", "Reshape", "Transpose"),
                        mem_budget=("128" if quantized else "384") if contiguous else "256",
                        slow_budget="16K", expect_tiled=True, exact_untiled=True)


def _movement_case(kind: str, quantized: bool, variant: int = 0, tiled: bool = False,
                   leading: bool = False) -> ContractCase:
    shape = [2, 37 if tiled else 3, 4]
    if leading:
        shape = [37, 1, 4]
    data = ((np.arange(np.prod(shape)).reshape(shape) * 7) % 31 - 15).astype(np.float32) * 0.125
    nodes, initializers, reference_nodes = [], [], []
    inputs = {"input": data}
    model_inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)]
    source = "input"
    if quantized:
        initializers += _scalars(x=(0.125, -17), y=(0.125, -17))
        nodes += _qdq(source, "x_s", "x_z", "x")
        source = "x"
    reference_nodes.extend(nodes)
    target = "raw" if quantized else "output"

    def integer(name, value):
        initializers.append(numpy_helper.from_array(np.array(value, np.int64), name))
        return name

    if kind == "Gather":
        axis = variant % 3
        indices = [shape[axis] - 1, 0] if tiled and axis == 0 else [shape[axis] - 1, 0, shape[axis] - 1]
        node = helper.make_node(kind, [source, integer("indices", indices)], [target], axis=axis)
        nodes.append(node)
        reference_nodes.append(copy.deepcopy(node))
        expected = np.take(data, indices, axis=axis)
    elif kind == "GatherND":
        indices = [[1], [0]] if tiled else [[1, 2], [0, 0], [1, 0]]
        node = helper.make_node(kind, [source, integer("indices", indices)], [target])
        nodes.append(node)
        reference_nodes.append(copy.deepcopy(node))
        expected = data[tuple(np.array(indices).T)]
    elif kind == "StridedSlice":
        if variant == 2:
            args = [source, integer("starts", [0, -1, 3]), integer("ends", [2, 0, 0]), integer("steps", [1, 1, -2])]
            nodes.append(helper.make_node(kind, args, [target], domain="tigris", end_mask=4, shrink_axis_mask=2))
            reference_nodes += [helper.make_node("Slice", [source, integer("ref_starts", [0, 2, 3]),
                                integer("ref_ends", [2, 3, -2**63]), integer("axes", [0, 1, 2]), "steps"], ["sliced"]),
                                helper.make_node("Squeeze", ["sliced", integer("squeeze_axis", [1])], [target])]
            expected = data[:, -1, ::-2]
        else:
            starts, ends, steps = ([1, 2, 3], [-2**63] * 3, [-1, -2, -2]) if variant else ([0, 0, 0], shape, [1, 2, 2])
            if tiled:
                steps = [1, 1, 2]
            args = [source, integer("starts", starts), integer("ends", ends), integer("axes", [0, 1, 2]), integer("steps", steps)]
            node = helper.make_node("Slice", args, [target])
            nodes.append(node)
            reference_nodes.append(copy.deepcopy(node))
            expected = data[tuple(slice(s, e, t) for s, e, t in zip(starts, ends, steps))]
    elif kind == "MirrorPad":
        mode = "symmetric" if variant else "reflect"
        pads = [[0, 0], [0, 0], [1, 2]] if tiled else [[1, 1], [2, 1], [0, 2]]
        integer("pads", np.array(pads).T.reshape(-1))
        if variant:
            nodes.append(helper.make_node(kind, [source, "pads"], [target], mode=mode, domain="tigris"))
        else:
            nodes.append(helper.make_node("Pad", [source, "pads"], [target], mode=mode))
        previous = source
        for axis, (dim, pad) in enumerate(zip(shape, pads)):
            indices = np.pad(np.arange(dim), pad, mode=mode)
            name = target if axis == 2 else f"pad_axis{axis}"
            reference_nodes.append(helper.make_node("Gather", [previous, integer(f"mirror{axis}", indices)], [name], axis=axis))
            previous = name
        expected = np.pad(data, pads, mode=mode)
    elif kind == "ReverseV2":
        if tiled:
            nodes.append(helper.make_node(kind, [source, integer("axes", [2])], [target], domain="tigris"))
            reference_nodes.append(helper.make_node("Gather", [source, integer("last", [3, 2, 1, 0])], [target], axis=2))
        else:
            nodes.append(helper.make_node(kind, [source, integer("axes", [0, 1])], [target], domain="tigris"))
            reference_nodes += [helper.make_node("Gather", [source, integer("first", [1, 0])], ["rev"], axis=0),
                                helper.make_node("Gather", ["rev", integer("last", [2, 1, 0])], [target], axis=1)]
        expected = data[:, :, ::-1] if tiled else data[::-1, ::-1, :]
    elif kind == "EmbeddingLookup":
        indices = [1, 0] if tiled else [1, 0, 1]
        name = integer("indices", indices)
        nodes.append(helper.make_node(kind, [source, name], [target], domain="tigris"))
        reference_nodes.append(helper.make_node("Gather", [source, name], [target], axis=0))
        expected = data[indices]
    else:
        update = np.full((shape[0], shape[1], 2) if tiled else (1, 2, 2), 3.5, np.float32)
        starts = [99, -4, 1]
        if variant:
            initializers.append(numpy_helper.from_array(update, "update"))
        else:
            inputs["update"] = update
            model_inputs.append(helper.make_tensor_value_info("update", TensorProto.FLOAT, update.shape))
        update_name = "update"
        if quantized:
            prefix = _qdq("update", "x_s", "x_z", "u")
            nodes.extend(prefix)
            reference_nodes.extend(copy.deepcopy(prefix))
            update_name = "u"
        nodes.append(helper.make_node(kind, [source, update_name, integer("starts", starts)], [target], domain="tigris"))
        clamped = np.clip(starts, 0, np.array(shape) - update.shape)
        coordinates = np.array(list(np.ndindex(update.shape))) + clamped
        reference_nodes += [helper.make_node("Reshape", [update_name, integer("flat", [-1])], ["updates_flat"]),
                            helper.make_node("ScatterND", [source, integer("positions", coordinates), "updates_flat"], [target])]
        expected = data.copy()
        expected[tuple(slice(int(s), int(s) + d) for s, d in zip(clamped, update.shape))] = update
    if quantized:
        nodes.extend(_qdq(target, "y_s", "y_z", "output"))
        reference_nodes.extend(_qdq(target, "y_s", "y_z", "output"))
    label = f"{'int8' if quantized else 'float'}_{kind.lower()}_{variant}"
    label += "_leading" if leading else "_tiled" if tiled else ""
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, expected.shape)

    def build(body):
        value_info = ([helper.make_tensor_value_info(target, TensorProto.FLOAT, expected.shape),
                       helper.make_tensor_value_info("output_q", TensorProto.INT8, expected.shape)]
                      if quantized else [])
        model = helper.make_model(helper.make_graph(body, label, model_inputs, [output], initializers,
                                                    value_info=value_info),
                                  opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("tigris", 1)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model

    operators = ["Transpose"]
    if kind == "DynamicUpdateSlice" and not variant:
        operators.append("Transpose")
    operators.append(kind)
    if expected.ndim >= 3:
        operators.append("Transpose")
    return ContractCase(label, build(nodes), build(reference_nodes), inputs, tuple(operators),
                        mem_budget=("128" if quantized else "384") if leading else "256" if tiled else "8K",
                        slow_budget="16K" if tiled else None,
                        expect_tiled=tiled, reference_unoptimized=True, exact_untiled=leading, exact=True,
                        compression="lz4" if kind == "Gather" and variant == 0 and not tiled else None)


def _leading_index_case(kind: str, quantized: bool) -> ContractCase:
    shape = [1, 1, 37, 1, 4]
    data = ((np.arange(148).reshape(37, 1, 4) % 23) - 11).astype(np.float32) * 0.125
    indices = np.array([[[0, 0]]] if kind == "GatherND" else [0], np.int64)
    initializers = [numpy_helper.from_array(indices, "indices"),
                    numpy_helper.from_array(np.array([3, 2, 1, 0], np.int64), "order"),
                    numpy_helper.from_array(np.array(shape, np.int64), "expanded_shape"),
                    numpy_helper.from_array(np.array([37, 1, 4], np.int64), "interface_shape")]
    nodes = []
    source, target = "input", "output"
    if quantized:
        initializers += _scalars(x=(0.125, -17), y=(0.125, -17))
        nodes += _qdq("input", "x_s", "x_z", "x")
        source, target = "x", "raw"
    nodes.append(helper.make_node("Gather", [source, "order"], ["ordered"], axis=2))
    if quantized:
        nodes += _qdq("ordered", "x_s", "x_z", "ordered_encoded")
    nodes.append(helper.make_node("Reshape", ["ordered_encoded" if quantized else "ordered", "expanded_shape"], ["expanded"]))
    if quantized:
        nodes += _qdq("expanded", "x_s", "x_z", "expanded_encoded")
    reference = copy.deepcopy(nodes)
    nodes.append(helper.make_node(kind, ["expanded_encoded" if quantized else "expanded", "indices"], ["indexed"],
                                  domain="tigris" if kind == "EmbeddingLookup" else ""))
    reference.append(helper.make_node("Gather" if kind == "EmbeddingLookup" else kind,
                                     ["expanded_encoded" if quantized else "expanded", "indices"], ["indexed"]))
    if quantized:
        nodes += _qdq("indexed", "x_s", "x_z", "indexed_encoded")
        reference += _qdq("indexed", "x_s", "x_z", "indexed_encoded")
    tail = helper.make_node("Reshape", ["indexed_encoded" if quantized else "indexed", "interface_shape"], [target])
    nodes.append(tail)
    reference.append(copy.deepcopy(tail))
    if quantized:
        nodes += _qdq(target, "y_s", "y_z", "output")
        reference += _qdq(target, "y_s", "y_z", "output")
    def model(body):
        m = helper.make_model(helper.make_graph(body, "leading_index",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [37, 1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [37, 1, 4])], initializers,
            value_info=[helper.make_tensor_value_info("indexed", TensorProto.FLOAT, shape)]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("tigris", 1)])
        m.ir_version = 8
        return m
    return ContractCase(f"{kind.lower()}_leading_int8{int(quantized)}", model(nodes), model(reference),
                        {"input": data}, ("Transpose", "Gather", "Reshape", kind, "Reshape", "Transpose"),
                        mem_budget="128" if quantized else "384",
                        slow_budget="16K", expect_tiled=True, exact_untiled=True, reference_unoptimized=True)


def _runtime_index_case(kind: str, quantized: bool, variant: int = 0, arg: str | None = None,
                        cast: bool = False, tiled: bool = False) -> ContractCase:
    case = _movement_case(kind, quantized, variant, tiled=tiled)
    name = "starts" if kind == "DynamicUpdateSlice" else "indices"
    original = next(t for t in case.compile_model.graph.initializer if t.name == name)
    values = numpy_helper.to_array(original).copy()
    inputs = dict(case.inputs)
    for model in (case.compile_model, case.reference_model):
        graph = model.graph
        initializer = next(t for t in graph.initializer if t.name == name)
        graph.initializer.remove(initializer)
        if arg:
            # The source has unique extrema at exactly the requested indices.
            shape = [*values.shape, 4]
            scores = np.zeros(shape, np.float32)
            np.put_along_axis(scores, values[..., None], -1.0 if arg == "ArgMin" else 1.0, axis=-1)
            inputs["scores"] = scores
            graph.input.append(helper.make_tensor_value_info("scores", TensorProto.FLOAT, shape))
            prefix = []
            source = "scores"
            if quantized:
                prefix = _qdq(source, "x_s", "x_z", "scores_dq")
                source = "scores_dq"
            prefix.append(helper.make_node(arg, [source], [name + "_wide" if cast else name], axis=-1, keepdims=0))
            if cast:
                prefix.append(helper.make_node("Cast", [name + "_wide"], [name], to=TensorProto.INT32))
            for node in reversed(prefix):
                graph.node.insert(0, node)
        else:
            # Keep the reference's int64 interface to exercise narrowing as well.
            graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT64, values.shape))
            inputs[name] = values
        if kind == "DynamicUpdateSlice" and model is case.reference_model:
            # ScatterND positions are computed from the same run-time starts.
            positions = next(t for t in graph.initializer if t.name == "positions")
            graph.initializer.remove(positions)
            update = case.inputs.get("update")
            if update is None:
                update = numpy_helper.to_array(next(t for t in graph.initializer if t.name == "update"))
            update_shape = update.shape
            limits = np.array(case.inputs["input"].shape) - update_shape
            for key, value in (("start_min", [0, 0, 0]), ("start_max", limits),
                               ("update_coords", list(np.ndindex(update_shape)))):
                graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), key))
            prefix = [helper.make_node("Max", [name, "start_min"], ["positive_starts"]),
                      helper.make_node("Min", ["positive_starts", "start_max"], ["clamped_starts"]),
                      helper.make_node("Add", ["update_coords", "clamped_starts"], ["positions"])]
            for node in reversed(prefix):
                graph.node.insert(0, node)
    operators = ((arg,) if arg else ()) + case.expected_operators
    return replace(case, name=case.name + "_runtime_" + (arg or "input") + ("_cast" if cast else ""), inputs=inputs,
                   expected_operators=operators, compression=None)


def _inner_axis_concat_case(*, axis: int, quantized: bool) -> ContractCase:
    """Two feature maps joined along height or width behind a convolution.
    No band holds the joined axis whole, so the stage runs untiled."""
    channels, height, width = 4, 12, 12
    rng = np.random.default_rng(23 + axis)
    weight = (rng.normal(size=(channels, channels, 3, 3)) * 0.25).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, height, width)).astype(np.float32)
    other = rng.uniform(-1.0, 1.0, size=(1, channels, height, width)).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    conv = dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), act=(0.03, 0))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("other", "act_s", "act_z", "y"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "act_s", "act_z", "c"),
            helper.make_node("Concat", ["c", "y"], ["joined"], axis=axis),
            *_qdq("joined", "act_s", "act_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["c"], **conv),
            helper.make_node("Concat", ["c", "other"], ["output"], axis=axis),
        ]
    shape = [1, channels, height, width]
    shape[axis] *= 2
    label = f"{'int8' if quantized else 'float'}_concat_axis{axis}"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, height, width]),
         helper.make_tensor_value_info("other", TensorProto.FLOAT, [1, channels, height, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data, "other": other},
                        ("Conv", "Concat"), mem_budget="16K")


def _channel_split_case(*, quantized: bool) -> ContractCase:
    """A feature map cut into unequal channel groups, each an interleaved run
    per pixel, recombined by a product."""
    rng = np.random.default_rng(24)
    data = rng.uniform(-1.0, 1.0, size=(1, 6, 5, 5)).astype(np.float32)
    initializers = [numpy_helper.from_array(np.array([3, 3], np.int64), "parts")]
    if quantized:
        initializers += _scalars(inp=(0.008, 0), prod=(0.0001, -5))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            helper.make_node("Split", ["x", "parts"], ["a", "b"], axis=1),
            *_qdq("a", "inp_s", "inp_z", "aq"),
            *_qdq("b", "inp_s", "inp_z", "bq"),
            helper.make_node("Mul", ["aq", "bq"], ["prod"]),
            *_qdq("prod", "prod_s", "prod_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("Split", ["input", "parts"], ["a", "b"], axis=1),
            helper.make_node("Mul", ["a", "b"], ["output"]),
        ]
    label = f"{'int8' if quantized else 'float'}_channel_split"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 6, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 5, 5])],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("Split", "Mul"))


def _standalone_pad_case(*, quantized: bool) -> ContractCase:
    """A Pad that no convolution absorbs, with a nonzero fill, padding both
    spatial axes unevenly."""
    channels, side = 3, 6
    rng = np.random.default_rng(27)
    weight = (rng.normal(size=(channels, channels, 3, 3)) * 0.25).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 2, 1], np.int64), "pads"),
        numpy_helper.from_array(np.array(0.75, np.float32), "fill"),
    ]
    conv = dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    if quantized:
        initializers += _scalars(inp=(0.008, 0), w=(0.01, 0), act=(0.03, -5))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("Conv", ["x", "wdq"], ["raw"], **conv),
            *_qdq("raw", "act_s", "act_z", "c"),
            helper.make_node("Pad", ["c", "pads", "fill"], ["padded"], mode="constant"),
            *_qdq("padded", "act_s", "act_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["c"], **conv),
            helper.make_node("Pad", ["c", "pads", "fill"], ["output"], mode="constant"),
        ]
    label = f"{'int8' if quantized else 'float'}_standalone_pad"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, channels, side + 3, side + 1])],
        initializers, opset=14,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data}, ("Conv", "Pad"),
                        mem_budget="16K")


def _unfolded_feature_map_case() -> ContractCase:
    """A feature map regrouped into patches, which is an unfold.

    A reshape renames axes over one sequence and the kernel copies straight
    through, which states the model's regrouping only when the runtime holds
    the elements in that order. This one folds the channel axis into the
    leading one, which a tensor held channels-last does not, so a straight
    copy of it comes back wrong. Holding the operand in the model's own
    order first is what makes it expressible.
    """
    channels, side, patch = 6, 8, 2
    cells = side // patch
    nodes = [
        helper.make_node(
            "Conv", ["input", "w"], ["features"], kernel_shape=[1, 1]),
        # [1, C, 8, 8] -> [C * cells, patch, cells, patch]: the channel axis
        # joins the leading one, which the stored order does not hold next to
        # it.
        helper.make_node("Reshape", ["features", "patches"], ["grouped"]),
        helper.make_node("Erf", ["grouped"], ["output"]),
    ]
    model = _model(
        "unfolded_feature_map",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT,
            [channels * cells, patch, cells, patch])],
        [numpy_helper.from_array(
            (np.random.default_rng(3).normal(size=(channels, channels, 1, 1))
             * 0.4).astype(np.float32), "w"),
         numpy_helper.from_array(
             np.array([channels * cells, patch, cells, patch], np.int64),
             "patches")],
    )
    data = np.linspace(
        -1.0, 1.0, channels * side * side, dtype=np.float32
    ).reshape(1, channels, side, side)
    return ContractCase(
        "float_unfolded_feature_map",
        model,
        model,
        {"input": data},
        ("Conv", "Transpose", "Reshape", "Erf", "Transpose"),
    )


def _prepended_token_case() -> ContractCase:
    """A learned token concatenated onto a sequence, then read back out.

    This is how a vision transformer carries a class token: one constant row
    is prepended to the patch sequence, travels through the blocks, and the
    head reads index 0 back out. The plan names the constant as the
    operator's weight, since only activations get tensor-table entries, and
    the index read is a cut of the stored bytes rather than an operator of
    its own.
    """
    tokens, width = 4, 8
    rng = np.random.default_rng(0)
    cls = (rng.normal(size=(1, 1, width)) * 0.3).astype(np.float32)
    proj = (rng.normal(size=(width, width)) * 0.3).astype(np.float32)
    head = (rng.normal(size=(width, 3)) * 0.3).astype(np.float32)
    model = _model(
        "prepended_token",
        [
            helper.make_node("Concat", ["cls", "input"], ["sequence"], axis=1),
            helper.make_node("MatMul", ["sequence", "proj"], ["mixed"]),
            helper.make_node("Gather", ["mixed", "first"], ["token"], axis=1),
            helper.make_node("MatMul", ["token", "head"], ["output"]),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        initializers=[
            numpy_helper.from_array(cls, "cls"),
            numpy_helper.from_array(proj, "proj"),
            numpy_helper.from_array(head, "head"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), "first"),
        ],
    )
    data = (rng.normal(size=(1, tokens, width)) * 0.7).astype(np.float32)
    return ContractCase(
        "float_prepended_token",
        model,
        model,
        {"input": data},
        ("Concat", "Transpose", "Reshape", "Gemm", "Reshape", "Reshape",
         "Split", "Reshape", "Gemm"),
    )


def _split_case() -> ContractCase:
    """A tensor cut into contiguous parts, which is how a fused qkv unbinds.

    One projection produces query, key and value together and the model cuts
    them apart along the axis they were stacked on. The parts are runs of the
    input, so each is that many bytes taken in order; a cut anywhere else
    interleaves and is refused rather than copied wrongly.
    """
    parts, tokens, width = 3, 5, 8
    nodes = [
        helper.make_node(
            "Split", ["input", "parts"], ["first", "second", "third"], axis=0),
        helper.make_node("Add", ["first", "second"], ["pair"]),
        helper.make_node("Mul", ["pair", "third"], ["output"]),
    ]
    model = _model(
        "split_parts",
        nodes,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [parts, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array(np.array([1, 1, 1], np.int64), "parts")],
    )
    data = np.linspace(
        -2.0, 2.0, parts * tokens * width, dtype=np.float32
    ).reshape(parts, tokens, width)
    return ContractCase(
        "float_split_parts",
        model,
        model,
        {"input": data},
        ("Split", "Add", "Mul"),
    )


def _constant_table_case(kind: str, quantized: bool, vector: bool) -> ContractCase:
    shape = [32, 16] if kind == "Gather" else [6, 6, 16]
    index_shape = ([31] if vector else [1, 31]) + ([2] if kind == "GatherND" else [])
    indices = np.arange(math.prod(index_shape), dtype=np.int64).reshape(index_shape) % (32 if kind == "Gather" else 6)
    indices.flat[0] = -1
    data = ((np.arange(math.prod(shape)).reshape(shape) * 7) % 127 - 63).astype(np.float32) * 0.125
    constants = [numpy_helper.from_array(data, "table")]
    nodes = []
    operand = "table"
    target = "output"
    if quantized:
        constants += _scalars(x=(0.125, 0), y=(0.125, 0))
        nodes += _qdq("table", "x_s", "x_z", "data")
        operand, target = "data", "raw"
    nodes.append(helper.make_node(kind, [operand, "indices"], [target]))
    if quantized:
        nodes += _qdq("raw", "y_s", "y_z", "output")
    output_shape = index_shape[:-1] + [16] if kind == "GatherND" else index_shape + [16]
    name = f"{'int8' if quantized else 'float'}_constant_{kind.lower()}_{'vector' if vector else 'matrix'}"
    model = _model(name, nodes, [helper.make_tensor_value_info("indices", TensorProto.INT64, index_shape)],
                   [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)], constants)
    expected = (kind,) if vector else (kind, "Transpose")
    return ContractCase(name, model, copy.deepcopy(model), {"indices": indices}, expected,
                        mem_budget="256", expect_tiled=True)


def _head_permutation_case() -> ContractCase:
    """The permutation that moves an attention block into its head layout.

    Stored (0, 2, 1, 3): a token by head transpose carrying the head width at
    each position.
    """
    tokens, heads, width = 64, 4, 16
    model = _model(
        "head_permutation",
        [helper.make_node(
            "Transpose", ["input"], ["output"], perm=[0, 2, 1, 3])],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, heads, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, heads, tokens, width])],
    )
    data = np.linspace(
        -1.0, 1.0, tokens * heads * width, dtype=np.float32
    ).reshape(1, tokens, heads, width)
    return ContractCase(
        "float_head_permutation",
        model,
        model,
        {"input": data},
        ("Transpose",),
        mem_budget="8K",
        expect_tiled=True,
    )


def _layout_mixing_residual_case(*, square: bool) -> ContractCase:
    """A residual connection around a matrix product.

    The matrix product needs the model's own axis order and the skip arrives
    in storage order, so the Add's two operands disagree about layout and the
    normalizer has to unify them. Two shapes: without the unification an
    oblong block fails the Add shape check at load, and a square one passes
    it and adds transposed data. The square case is the one that matters,
    and it is the ordinary shape for a small transformer whose sequence
    length equals its width.
    """
    tokens, width = (8, 8) if square else (8, 4)
    rng = np.random.default_rng(0)
    weight = (rng.normal(size=(width, width)) * 0.3).astype(np.float32)
    model = _model(
        f"layout_mixing_residual_{'square' if square else 'oblong'}",
        [
            helper.make_node("MatMul", ["input", "w"], ["projected"]),
            helper.make_node("Add", ["input", "projected"], ["output"]),
        ],
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, tokens, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, tokens, width])],
        [numpy_helper.from_array(weight, "w")],
    )
    data = np.linspace(
        -1.5, 1.5, tokens * width, dtype=np.float32
    ).reshape(1, tokens, width)
    return ContractCase(
        f"float_residual_over_matmul_{'square' if square else 'oblong'}",
        model,
        model,
        {"input": data},
        ("Transpose", "Reshape", "Gemm", "Reshape", "Add", "Transpose"),
    )


def _output_transpose_case() -> ContractCase:
    """A public Transpose must retain its ONNX shape and element ordering."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 3]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [3, 1, 2]
    )
    model = _model(
        "output_transpose",
        [
            helper.make_node(
                "Transpose", ["input"], ["output"], perm=[2, 0, 1]
            )
        ],
        [model_input],
        [model_output],
    )
    return ContractCase(
        "float_output_transpose",
        model,
        model,
        {"input": np.array([[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]], dtype=np.float32)},
        ("Transpose",),
    )


def _inference_identity_case() -> ContractCase:
    """Dropout and Identity must vanish, and Squeeze must execute as Reshape.

    The reference model keeps all three so ONNX Runtime evaluates the graph an
    exporter actually emits; TiGrIS must reach the same values with them gone.
    The Squeeze here drops the two trailing unit axes of a pooled [1, C, 1, 1]
    tensor, which leaves the runtime's element order untouched.
    """
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes")
    model = _model(
        "inference_identity",
        [
            helper.make_node("Identity", ["input"], ["same"]),
            helper.make_node("Dropout", ["same"], ["kept"]),
            helper.make_node("GlobalAveragePool", ["kept"], ["pooled"]),
            helper.make_node("Squeeze", ["pooled", "axes"], ["output"]),
        ],
        [model_input],
        [model_output],
        [axes],
    )
    return ContractCase(
        "float_inference_identity",
        model,
        model,
        {
            "input": np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
        },
        ("GlobalAveragePool", "Reshape"),
    )


def _resize_concat_case() -> ContractCase:
    """Resize scale extraction and NCHW-to-NHWC Concat-axis mapping."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            "left_scales",
        ),
        numpy_helper.from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            "right_scales",
        ),
        numpy_helper.from_array(
            np.array(1.5, dtype=np.float32), "multiplier"
        ),
    ]
    resize_attrs = {
        "coordinate_transformation_mode": "asymmetric",
        "mode": "nearest",
        "nearest_mode": "floor",
    }
    model = _model(
        "resize_concat",
        [
            helper.make_node(
                "Resize",
                ["input", "", "left_scales"],
                ["left"],
                **resize_attrs,
            ),
            helper.make_node(
                "Mul", ["input", "multiplier"], ["scaled"]
            ),
            helper.make_node(
                "Resize",
                ["scaled", "", "right_scales"],
                ["right"],
                **resize_attrs,
            ),
            helper.make_node(
                "Concat", ["left", "right"], ["output"], axis=1
            ),
        ],
        [model_input],
        [model_output],
        initializers,
    )
    return ContractCase(
        "float_resize_concat",
        model,
        model,
        {
            "input": np.array(
                [[[[1.0, -2.0], [3.0, 0.5]]]], dtype=np.float32
            )
        },
        ("Resize", "Mul", "Resize", "Concat"),
    )
