"""Whole-network, recurrent and detection cases."""

from __future__ import annotations

import copy
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.frontends.tflite import STATE_KEY

from .common import ContractCase, _model, _qdq, _scalars
from .fixtures import build_ds_cnn, build_fc_autoencoder, build_tcn


def _svdf_case(projected: bool = False) -> ContractCase:
    """A float SVDF with a fused Relu, run once from its zero initial state, so
    each filter's time projection sees only the newest slot of its memory.
    With ``projected``, a MatMul by a constant feeds the SVDF."""
    rng = np.random.default_rng(85)
    batch, features, units, rank, memory = 2, 6, 3, 2, 4
    filters = units * rank
    feature = rng.normal(0.0, 0.5, (filters, features)).astype(np.float32)
    time = rng.normal(0.0, 0.5, (filters, memory)).astype(np.float32)
    bias = rng.normal(0.0, 0.3, (units,)).astype(np.float32)
    x = rng.uniform(-2.0, 2.0, (batch, features)).astype(np.float32)
    shape_x, shape_y, shape_s = [batch, features], [batch, units], [batch, filters * memory]
    head, source, head_weights = [], "x", []
    if projected:
        head = [helper.make_node("MatMul", ["x", "w_in"], ["x_in"])]
        source = "x_in"
        head_weights = [numpy_helper.from_array(
            rng.normal(0.0, 0.5, (features, features)).astype(np.float32), "w_in")]
    weights = [numpy_helper.from_array(feature, "feature"), numpy_helper.from_array(time, "time"),
               numpy_helper.from_array(bias, "bias"),
               numpy_helper.from_array(np.zeros(shape_s, np.float32), "state_initial"), *head_weights]
    compile_graph = helper.make_graph(
        [*head, helper.make_node("Svdf", [source, "feature", "time", "bias", "state_in"], ["y", "state_out"],
                                 domain="tigris", rank=rank, activation="relu")],
        "svdf",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x),
         helper.make_tensor_value_info("state_in", TensorProto.FLOAT, shape_s)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
         helper.make_tensor_value_info("state_out", TensorProto.FLOAT, shape_s)],
        weights,
        value_info=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y),
                    helper.make_tensor_value_info("state_out", TensorProto.FLOAT, shape_s),
                    *([helper.make_tensor_value_info(source, TensorProto.FLOAT, shape_x)] if projected else [])])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    helper.set_model_props(compile_model, {STATE_KEY: json.dumps(
        [{"input": "state_in", "output": "state_out", "initial": "state_initial"}])})
    reference = _model(
        "svdf_reference",
        [*head, helper.make_node("MatMul", [source, "feature_t"], ["projection"]),
         helper.make_node("Mul", ["projection", "newest"], ["weighted"]),
         helper.make_node("Reshape", ["weighted", "grouped_shape"], ["grouped"]),
         helper.make_node("ReduceSum", ["grouped", "rank_axis"], ["summed"], keepdims=0),
         helper.make_node("Add", ["summed", "bias"], ["biased"]),
         helper.make_node("Relu", ["biased"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)],
        [numpy_helper.from_array(np.ascontiguousarray(feature.T), "feature_t"),
         numpy_helper.from_array(np.ascontiguousarray(time[:, -1]), "newest"),
         numpy_helper.from_array(np.asarray([batch, units, rank], np.int64), "grouped_shape"),
         numpy_helper.from_array(np.asarray([2], np.int64), "rank_axis"),
         numpy_helper.from_array(bias, "bias"), *head_weights])
    return ContractCase(name="projected_svdf_relu" if projected else "svdf_relu", compile_model=compile_model,
                        reference_model=reference, inputs={"x": x},
                        expected_operators=("Gemm", "Svdf") if projected else ("Svdf",))


def _detection_case() -> ContractCase:
    """Fast-form DetectionPostProcess over boxes that do not overlap, so the
    detections are the best-scoring decoded boxes, against a reference that
    decodes them and takes the top scores."""
    rng = np.random.default_rng(89)
    boxes, detections = 6, 3
    scales = [10.0, 10.0, 5.0, 5.0]
    # Anchors a unit apart and a third of a unit wide, as [y, x, h, w].
    anchors = np.stack([np.arange(boxes) * 1.0, np.full(boxes, 0.5),
                        np.full(boxes, 0.3), np.full(boxes, 0.3)], 1).astype(np.float32)
    encodings = rng.uniform(-0.5, 0.5, (1, 4, boxes)).astype(np.float32)
    scores = np.stack([rng.uniform(0.0, 0.2, boxes), rng.permutation(np.linspace(0.3, 0.9, boxes))])
    scores = scores[None].astype(np.float32)
    shapes = {"encodings": [1, 4, boxes], "scores": [1, 2, boxes], "boxes": [1, 4, detections],
              "classes": [1, detections], "best": [1, detections], "count": [1]}
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shapes[n]) for n in ("encodings", "scores")]
    outputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shapes[n])
               for n in ("boxes", "classes", "best", "count")]
    compile_model = helper.make_model(helper.make_graph(
        [helper.make_node("DetectionPostProcess", ["encodings", "scores", "anchors"],
                          ["boxes", "classes", "best", "count"], domain="tigris",
                          max_detections=detections, detections_per_class=100, use_regular_nms=0,
                          score_threshold=0.1, iou_threshold=0.5, num_classes=1, scales=scales)],
        "detection", inputs, outputs, [numpy_helper.from_array(anchors, "anchors")],
        value_info=outputs), opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9

    def row(values, name):
        return numpy_helper.from_array(np.asarray(values, np.float32).reshape(1, 1, boxes), name)

    reference = _model(
        "detection_reference",
        [helper.make_node("Split", ["encodings", "four"], ["ey", "ex", "eh", "ew"], axis=1),
         helper.make_node("Div", ["ey", "sy"], ["dy"]), helper.make_node("Mul", ["dy", "ah"], ["my"]),
         helper.make_node("Add", ["my", "ay"], ["cy"]),
         helper.make_node("Div", ["ex", "sx"], ["dx"]), helper.make_node("Mul", ["dx", "aw"], ["mx"]),
         helper.make_node("Add", ["mx", "ax"], ["cx"]),
         helper.make_node("Div", ["eh", "sh"], ["dh"]), helper.make_node("Exp", ["dh"], ["gh"]),
         helper.make_node("Mul", ["gh", "half_ah"], ["hh"]),
         helper.make_node("Div", ["ew", "sw"], ["dw"]), helper.make_node("Exp", ["dw"], ["gw"]),
         helper.make_node("Mul", ["gw", "half_aw"], ["hw"]),
         helper.make_node("Sub", ["cy", "hh"], ["ymin"]), helper.make_node("Sub", ["cx", "hw"], ["xmin"]),
         helper.make_node("Add", ["cy", "hh"], ["ymax"]), helper.make_node("Add", ["cx", "hw"], ["xmax"]),
         helper.make_node("Concat", ["ymin", "xmin", "ymax", "xmax"], ["corners"], axis=1),
         helper.make_node("Slice", ["scores", "one", "two", "one"], ["class_scores"]),
         helper.make_node("Reshape", ["class_scores", "flat"], ["flat_scores"]),
         helper.make_node("TopK", ["flat_scores", "k"], ["top", "order"]),
         helper.make_node("Gather", ["corners", "order"], ["boxes"], axis=2),
         helper.make_node("Reshape", ["top", "row_shape"], ["best"]),
         helper.make_node("Mul", ["best", "zero"], ["classes"]),
         helper.make_node("Shape", ["order"], ["found"]),
         helper.make_node("Cast", ["found"], ["count"], to=TensorProto.FLOAT)],
        inputs, outputs,
        [numpy_helper.from_array(np.asarray([1, 1, 1, 1], np.int64), "four"),
         *(numpy_helper.from_array(np.float32(v), n) for n, v in zip(("sy", "sx", "sh", "sw"), scales)),
         row(anchors[:, 0], "ay"), row(anchors[:, 1], "ax"), row(anchors[:, 2], "ah"),
         row(anchors[:, 3], "aw"), row(0.5 * anchors[:, 2], "half_ah"), row(0.5 * anchors[:, 3], "half_aw"),
         numpy_helper.from_array(np.asarray([1], np.int64), "one"),
         numpy_helper.from_array(np.asarray([2], np.int64), "two"),
         numpy_helper.from_array(np.asarray([boxes], np.int64), "flat"),
         numpy_helper.from_array(np.asarray([detections], np.int64), "k"),
         numpy_helper.from_array(np.asarray([1, detections], np.int64), "row_shape"),
         numpy_helper.from_array(np.float32(0.0), "zero")])
    return ContractCase(name="detection_fast", compile_model=compile_model, reference_model=reference,
                        inputs={"encodings": encodings, "scores": scores},
                        expected_operators=("DetectionPostProcess",))


def _lstm_case() -> ContractCase:
    """A float LSTM over three steps from its zero initial state, against ONNX's
    own LSTM operator, which orders the gates i, o, f, c."""
    rng = np.random.default_rng(86)
    batch, steps, features, units = 2, 3, 3, 4
    w_in = rng.normal(0.0, 0.5, (4, units, features)).astype(np.float32)
    w_rec = rng.normal(0.0, 0.5, (4, units, units)).astype(np.float32)
    bias = rng.normal(0.0, 0.3, (4, units)).astype(np.float32)
    x = rng.uniform(-2.0, 2.0, (batch, steps, features)).astype(np.float32)
    shape_s, shape_y = [batch, units], [batch, steps, units]
    weights = [numpy_helper.from_array(w, f"w{k}")
               for k, w in enumerate([*w_in, *w_rec, *bias])]
    weights += [numpy_helper.from_array(np.zeros(shape_s, np.float32), name)
                for name in ("hidden_initial", "cell_initial")]
    states = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape_s)
              for name in ("hidden_out", "cell_out")]
    compile_graph = helper.make_graph(
        [helper.make_node("Lstm", ["x", *(f"w{k}" for k in range(12)), "hidden_in", "cell_in"],
                          ["y", "hidden_out", "cell_out"], domain="tigris", time_major=0,
                          cell_clip=0.0)],
        "lstm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
         *(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape_s)
           for name in ("hidden_in", "cell_in"))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y), *states],
        weights,
        value_info=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y), *states])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    helper.set_model_props(compile_model, {STATE_KEY: json.dumps([
        {"input": "hidden_in", "output": "hidden_out", "initial": "hidden_initial"},
        {"input": "cell_in", "output": "cell_out", "initial": "cell_initial"}])})
    onnx_order = [0, 3, 1, 2]
    reference = _model(
        "lstm_reference",
        # ONNX Runtime's LSTM runs time-major only.
        [helper.make_node("Transpose", ["x"], ["steps_first"], perm=[1, 0, 2]),
         helper.make_node("LSTM", ["steps_first", "W", "R", "B"], ["sequence"], hidden_size=units),
         helper.make_node("Squeeze", ["sequence", "direction_axis"], ["squeezed"]),
         helper.make_node("Transpose", ["squeezed"], ["y"], perm=[1, 0, 2])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)],
        [numpy_helper.from_array(w_in[onnx_order].reshape(1, 4 * units, features), "W"),
         numpy_helper.from_array(w_rec[onnx_order].reshape(1, 4 * units, units), "R"),
         numpy_helper.from_array(np.concatenate(
             [bias[onnx_order].reshape(-1), np.zeros(4 * units, np.float32)])[None], "B"),
         numpy_helper.from_array(np.asarray([1], np.int64), "direction_axis")],
        opset=14)
    return ContractCase(name="lstm", compile_model=compile_model, reference_model=reference,
                        inputs={"x": x},
                        # Rank-3 boundaries are stored channels-last; Lstm reads model order.
                        expected_operators=("Transpose", "Lstm", "Transpose"))


def _squeeze_excitation_case(*, quantized: bool) -> ContractCase:
    """A squeeze-and-excitation gate: one value per channel scales the map.

    The gate is the Mul's first operand, as exporters write it, so the case
    also covers putting the full-shape operand first. The map is larger than
    the budget, so the Mul runs in bands with the gate loaded whole each time.
    """
    channels, side = 16, 12
    rng = np.random.default_rng(22)
    weight = (rng.normal(size=(channels, channels, 1, 1)) * 0.5).astype(np.float32)
    bias = (rng.normal(size=(channels,)) * 0.2).astype(np.float32)
    data = rng.uniform(-1.0, 1.0, size=(1, channels, side, side)).astype(np.float32)
    initializers = [numpy_helper.from_array(weight, "weight")]
    if not quantized:
        initializers.append(numpy_helper.from_array(bias, "bias"))
    if quantized:
        initializers += _scalars(inp=(0.008, 3), pool=(0.004, 0), w=(0.008, 0),
                                 fc=(0.02, 0), gate=(1.0 / 256.0, -128), out=(0.007, 1))
        # An int32 bias at the accumulator scale, dequantized only, which is
        # how a QDQ export states it.
        bias_scale = np.float32(0.004 * 0.008)
        initializers.append(numpy_helper.from_array(
            np.round(bias / bias_scale).astype(np.int32), "bias_q"))
        initializers.append(numpy_helper.from_array(np.array(bias_scale, dtype=np.float32), "b_s"))
        initializers.append(numpy_helper.from_array(np.array(0, dtype=np.int32), "b_z"))
        nodes = [
            *_qdq("input", "inp_s", "inp_z", "x"),
            helper.make_node("GlobalAveragePool", ["x"], ["pooled_raw"]),
            *_qdq("pooled_raw", "pool_s", "pool_z", "pooled"),
            *_qdq("weight", "w_s", "w_z", "wdq"),
            helper.make_node("DequantizeLinear", ["bias_q", "b_s", "b_z"], ["bdq"]),
            helper.make_node("Conv", ["pooled", "wdq", "bdq"], ["fc_raw"], kernel_shape=[1, 1]),
            *_qdq("fc_raw", "fc_s", "fc_z", "fc"),
            helper.make_node("Sigmoid", ["fc"], ["gate_raw"]),
            *_qdq("gate_raw", "gate_s", "gate_z", "gate"),
            helper.make_node("Mul", ["gate", "x"], ["scaled"]),
            *_qdq("scaled", "out_s", "out_z", "output"),
        ]
    else:
        nodes = [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Conv", ["pooled", "weight", "bias"], ["fc"], kernel_shape=[1, 1]),
            helper.make_node("Sigmoid", ["fc"], ["gate"]),
            helper.make_node("Mul", ["gate", "input"], ["output"]),
        ]
    label = f"{'int8' if quantized else 'float'}_squeeze_excitation"
    model = _model(
        label, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side, side])],
        initializers,
    )
    return ContractCase(label, model, copy.deepcopy(model), {"input": data},
                        ("GlobalAveragePool", "Conv", "Sigmoid", "Mul"),
                        mem_budget="2K", expect_tiled=True)


def _many_stage_case() -> ContractCase:
    """Schema v5 derives full stage IDs from the uint16 stage table."""
    stage_count = 300
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    nodes = []
    previous = "input"
    for index in range(stage_count):
        output = "output" if index == stage_count - 1 else f"value_{index}"
        nodes.append(helper.make_node("Relu", [previous], [output]))
        previous = output
    model = _model(
        "many_stage_relu", nodes, [model_input], [model_output]
    )
    return ContractCase(
        "float_schema_v5_many_stage",
        model,
        model,
        {"input": np.array([[-2.0, -0.5, 0.25, 3.0]], dtype=np.float32)},
        ("Relu",) * stage_count,
        mem_budget="64",
        force_one_op_stages=True,
    )


def _tcn_16k_case() -> ContractCase:
    """The TCN at a 16K budget, which tiles its gated pointwise path."""
    model = build_tcn()
    return ContractCase(
        "float_tcn_16k",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 3 * 128, dtype=np.float32
            ).reshape(1, 3, 128)
        },
        (
            "Conv1D",
            "Conv1D",
            "Conv1D",
            "Tanh",
            "Conv1D",
            "Sigmoid",
            "Mul",
            "Conv1D",
            "Flatten",
            "Gemm",
        ),
        mem_budget="16K",
        expect_tiled=True,
    )


def _ds_cnn_256k_case() -> ContractCase:
    """The project DS-CNN keyword spotter, whole model."""
    model = build_ds_cnn()
    return ContractCase(
        "float_ds_cnn_256k",
        model,
        model,
        {
            "input": np.linspace(
                -1.0, 1.0, 49 * 10, dtype=np.float32
            ).reshape(1, 1, 49, 10)
        },
        ("Conv", "DepthwiseConv") * 4
        + ("Conv", "GlobalAveragePool", "Flatten", "Gemm"),
        mem_budget="256K",
    )


def _fc_autoencoder_256k_case() -> ContractCase:
    """The project fully connected autoencoder, whole model."""
    model = build_fc_autoencoder()
    return ContractCase(
        "float_fc_autoencoder_256k",
        model,
        model,
        {"input": np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(1, 128)},
        ("Gemm",) * 10 + ("Sigmoid",),
        mem_budget="256K",
    )


def _normalized_classifier_case() -> ContractCase:
    """BN, Relu6 fusion, shape folding, pooling, reshape, FC, flatten."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    initializers = [
        numpy_helper.from_array(
            np.array([[[[0.75]]], [[[-0.5]]]], dtype=np.float32),
            "conv_weights",
        ),
        numpy_helper.from_array(
            np.array([0.1, -0.2], dtype=np.float32), "conv_bias"
        ),
        numpy_helper.from_array(
            np.array([1.2, 0.8], dtype=np.float32), "bn_scale"
        ),
        numpy_helper.from_array(
            np.array([0.1, -0.1], dtype=np.float32), "bn_bias"
        ),
        numpy_helper.from_array(
            np.array([0.2, -0.3], dtype=np.float32), "bn_mean"
        ),
        numpy_helper.from_array(
            np.array([0.5, 0.25], dtype=np.float32), "bn_var"
        ),
        numpy_helper.from_array(
            np.linspace(-0.4, 0.5, 32, dtype=np.float32).reshape(4, 8),
            "fc_weights",
        ),
        numpy_helper.from_array(
            np.array([0.05, -0.1, 0.15, 0.2], dtype=np.float32),
            "fc_bias",
        ),
    ]
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["clip_min"],
            value=numpy_helper.from_array(np.array(0.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["clip_max"],
            value=numpy_helper.from_array(np.array(6.0, dtype=np.float32)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["shape_index"],
            value=numpy_helper.from_array(np.array(0, dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["unsqueeze_axes"],
            value=numpy_helper.from_array(np.array([0], dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["flat_tail"],
            value=numpy_helper.from_array(np.array([-1], dtype=np.int64)),
        ),
        helper.make_node(
            "Conv",
            ["input", "conv_weights", "conv_bias"],
            ["conv"],
            kernel_shape=[1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["normalized"],
        ),
        helper.make_node(
            "Clip",
            ["normalized", "clip_min", "clip_max"],
            ["activated"],
        ),
        helper.make_node(
            "MaxPool",
            ["activated"],
            ["pooled"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node("Shape", ["pooled"], ["pool_shape"]),
        helper.make_node(
            "Gather", ["pool_shape", "shape_index"], ["batch"], axis=0
        ),
        helper.make_node(
            "Unsqueeze", ["batch", "unsqueeze_axes"], ["batch_vector"]
        ),
        helper.make_node(
            "Concat",
            ["batch_vector", "flat_tail"],
            ["target_shape"],
            axis=0,
        ),
        helper.make_node(
            "Reshape", ["pooled", "target_shape"], ["reshaped"]
        ),
        helper.make_node(
            "Gemm",
            ["reshaped", "fc_weights", "fc_bias"],
            ["classified"],
            transB=1,
        ),
        helper.make_node("Flatten", ["classified"], ["output"], axis=1),
    ]
    model = _model(
        "normalized_classifier",
        nodes,
        [model_input],
        [model_output],
        initializers,
    )
    # The deployment shape is static even though the ONNX graph computes its
    # Reshape input through Shape/Gather/Unsqueeze/Concat. Preserve the
    # compiler's fail-closed dynamic-shape rule by declaring that known value.
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "reshaped", TensorProto.FLOAT, [1, 8]
            )
        ]
    )
    onnx.checker.check_model(model)
    return ContractCase(
        "float_normalized_classifier",
        model,
        model,
        {
            "input": np.array(
                [
                    [
                        [
                            [-1.0, 0.0, 1.0, 2.0],
                            [3.0, 4.0, 5.0, 6.0],
                            [2.5, 1.5, 0.5, -0.5],
                            [4.5, 3.5, 2.5, 1.5],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        },
        ("Conv", "MaxPool", "Reshape", "Gemm", "Flatten"),
        compression="lz4",
    )


def _two_input_compressed_case() -> ContractCase:
    """Two model inputs read in place by one stage whose weights are compressed.

    Only the first input placed leaves room for the decompressed weights below
    it, so the inputs sit together above one room and the run needs no slow
    arena at the stated fast figure.
    """
    shape = [1, 2, 4, 4]
    weight = (np.arange(54, dtype=np.float32).reshape(3, 2, 3, 3) % 7 - 3) * 0.125
    bias = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    model = _model(
        "two_input_compressed",
        [helper.make_node("Add", ["left", "right"], ["sum"]),
         helper.make_node("Conv", ["sum", "weight", "bias"], ["output"], pads=[1, 1, 1, 1])],
        [helper.make_tensor_value_info("left", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("right", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")],
    )
    values = np.arange(32, dtype=np.float32).reshape(shape) * 0.0625
    return ContractCase("two_input_compressed", model, model,
                        {"left": values, "right": values[:, ::-1].copy() - 1.0}, ("Add", "Conv"),
                        compression="lz4", expect_inputs_in_fast=True)
