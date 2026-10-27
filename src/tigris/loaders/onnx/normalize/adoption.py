"""Operators the frontends write in the compiler's own domain, checked and given their plan names."""

import numpy as np

from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders.onnx.normalize.common import _fresh_name


def _adopt_compiler_ops(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Give each operator of the compiler's own domain its plan name and check
    it, in graph order."""
    for op in ag.ops:
        adopt = _ADOPT.get(op.op_type)
        if adopt is not None:
            op.op_type = op.op_type.removeprefix("tigris::")
            adopt(ag, op)
    return ag


def _adopt_tflite_gelu(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own Gelu carries TFLite GELU below ONNX opset 20."""


def _adopt_tflite_cumsum(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own CumSum states TFLite's int8 semantics, which seed the
    sum with the scaled input zero point; ONNX's CumSum sums dequantized
    values, which differs whenever that zero point is not 0."""
    op.attrs["tflite_seeded"] = 1


def _adopt_svdf(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own SVDF: input [batch, features], constant feature
    [filters, features], time [filters, memory] and bias [units] weights, and a
    state [batch, filters * memory] passed in and written back."""
    if len(op.inputs) != 5 or len(op.outputs) != 2:
        raise ValueError("Svdf requires five inputs and two outputs")
    x, y = ag.tensors[op.inputs[0]], ag.tensors[op.outputs[0]]
    feature, time, bias = (ag.weight_data.get(name) for name in op.inputs[1:4])
    state, kept = ag.tensors[op.inputs[4]], ag.tensors[op.outputs[1]]
    rank = int(op.attrs.get("rank", 0))
    if feature is None or time is None or bias is None:
        raise ValueError("Svdf requires constant feature, time and bias weights")
    if len(x.shape) != 2 or feature.ndim != 2 or time.ndim != 2 or bias.ndim != 1:
        raise ValueError("Svdf requires a rank-2 input and rank-2 weights")
    batch, features = x.shape
    filters, memory = time.shape
    if (rank <= 0 or filters % rank or feature.shape != (filters, features)
            or bias.shape != (filters // rank,) or tuple(y.shape) != (batch, filters // rank)
            or tuple(state.shape) != (batch, filters * memory) or kept.shape != state.shape):
        raise ValueError("Svdf shapes do not agree with its rank")
    activation = op.attrs.pop("activation", "none")
    if isinstance(activation, bytes):
        activation = activation.decode()
    if activation not in ("none", "relu", "relu6"):
        raise ValueError(f"Svdf activation {activation!r} is not supported")
    if activation != "none":
        op.attrs["fused_activation"] = {"relu": "Relu", "relu6": "Relu6"}[activation]


def _adopt_quantize(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own Quantize and Dequantize: float32 to the int8 encoding
    of the output, and int8 to float32, where an int8 model keeps a
    control-flow boundary in float32."""
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(f"{op.op_type} requires one input and one output")


def detection_scratch_bytes(boxes: int, detections: int, candidates: int) -> int:
    """The runtime's TIGRIS_DETECTION_SCRATCH_BYTES."""
    return 4 * (9 * boxes + max(boxes, candidates) + 2 * candidates + detections) + boxes


def _adopt_detection(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own DetectionPostProcess: box encodings [1, 4, boxes] and
    scores [1, columns, boxes] in NCL, constant anchors [boxes, 4]; float32
    boxes [1, 4, detections], classes and scores [1, detections] and the count
    [1]. A float32 output holding the runtime's working memory is added."""
    if len(op.inputs) != 3 or len(op.outputs) != 4:
        raise ValueError("DetectionPostProcess requires three inputs and four outputs")
    anchors = ag.weight_data.get(op.inputs[2])
    boxes, scores = (ag.tensors[name] for name in op.inputs[:2])
    a = op.attrs
    count, classes, detections = (anchors.shape[0] if anchors is not None else 0,
                                  int(a.get("num_classes", 0)), int(a.get("max_detections", 0)))
    regular = int(a.get("use_regular_nms", 0))
    if (anchors is None or anchors.shape != (count, 4) or tuple(boxes.shape) != (1, 4, count)
            or len(scores.shape) != 3 or scores.shape[0] != 1 or scores.shape[2] != count
            or not 1 <= classes <= scores.shape[1] <= classes + 1 or not 0 < count <= 65535):
        raise ValueError("DetectionPostProcess shapes do not agree")
    expected = [(1, 4, detections), (1, detections), (1, detections), (1,)]
    if detections <= 0 or [tuple(ag.tensors[n].shape) for n in op.outputs] != expected:
        raise ValueError("DetectionPostProcess outputs do not agree with its detections")
    per_class = int(a.get("detections_per_class", 100))
    if regular not in (0, 1) or (regular and per_class <= 0) or len(a.get("scales", ())) != 4:
        raise ValueError("DetectionPostProcess options are not supported")
    a.update(max_detections=detections, detections_per_class=per_class, num_classes=classes,
             use_regular_nms=regular, scales=[float(v) for v in a["scales"]])
    size = detection_scratch_bytes(count, detections, detections + (per_class if regular else 0))
    scratch = _fresh_name(ag, f"{op.outputs[0]}_scratch")
    ag.tensors[scratch] = TensorInfo(name=scratch, shape=(-(-size // 4),), dtype=1)
    op.outputs.append(scratch)


def _adopt_lstm(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own LSTM: a sequence [batch, steps, features], or
    [steps, batch, features] when time_major, then the input and recurrent
    weights and the biases of gates i, f, c, o, then hidden and cell states
    [batch, units]; it writes the hidden sequence and both states back."""
    if len(op.inputs) != 15 or len(op.outputs) != 3:
        raise ValueError("Lstm requires fifteen inputs and three outputs")
    x, y = ag.tensors[op.inputs[0]], ag.tensors[op.outputs[0]]
    weights = [ag.weight_data.get(name) for name in op.inputs[1:13]]
    if any(w is None for w in weights):
        raise ValueError("Lstm requires constant weights and biases")
    if len(x.shape) != 3:
        raise ValueError("Lstm requires a rank-3 input")
    time_major = int(op.attrs.get("time_major", 0))
    steps, batch = (x.shape[0], x.shape[1]) if time_major else (x.shape[1], x.shape[0])
    features = x.shape[2]
    units = weights[0].shape[0]
    expected = [(units, features)] * 4 + [(units, units)] * 4 + [(units,)] * 4
    sequence = (steps, batch, units) if time_major else (batch, steps, units)
    states = [ag.tensors[name] for name in (*op.inputs[13:], *op.outputs[1:])]
    if ([w.shape for w in weights] != expected or tuple(y.shape) != sequence
            or any(tuple(s.shape) != (batch, units) for s in states)):
        raise ValueError("Lstm shapes do not agree")
    if float(op.attrs.get("cell_clip", 0.0)) < 0.0:
        raise ValueError("Lstm cell clip must not be negative")
    if "weight_scales" in op.attrs and (len(op.attrs["weight_scales"]) != 8
                                        or "cell_scale" not in op.attrs):
        raise ValueError("an integer Lstm states its eight weight scales and its cell scale")


def _adopt_if(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own If: a one-element bool condition and operands in;
    each branch a subgraph taking the operands and giving the results, in the
    same shapes and dtypes."""
    branches = [ag.subgraphs[op.attrs[key]] for key in ("then_branch", "else_branch")]
    condition = ag.tensors[op.inputs[0]]
    if condition.dtype != 9 or int(np.prod(condition.shape)) != 1:
        raise ValueError("If requires a one-element bool condition")
    operands = _forms(ag, op.inputs[1:])
    results = _forms(ag, op.outputs)
    for branch in branches:
        if (_forms(branch, branch.model_inputs) != operands
                or _forms(branch, branch.model_outputs) != results):
            raise ValueError("If branches must take its operands and give its results")


def _forms(graph: AnalyzedGraph, names) -> list:
    return [(graph.tensors[n].shape, graph.tensors[n].dtype) for n in names]


def _adopt_while(ag: AnalyzedGraph, op: OpNode) -> None:
    """The compiler's own While: loop variables in and out; the condition
    subgraph takes them and gives one bool, the body takes and gives them."""
    condition = ag.subgraphs[op.attrs["cond_branch"]]
    body = ag.subgraphs[op.attrs["body_branch"]]
    variables = _forms(ag, op.inputs)
    verdict = [condition.tensors[n] for n in condition.model_outputs]
    if (len(op.outputs) != len(op.inputs)
            or _forms(ag, op.outputs) != variables
            or _forms(condition, condition.model_inputs) != variables
            or _forms(body, body.model_inputs) != variables
            or _forms(body, body.model_outputs) != variables):
        raise ValueError("While condition and body must take and give its loop variables")
    if len(verdict) != 1 or verdict[0].dtype != 9 or int(np.prod(verdict[0].shape)) != 1:
        raise ValueError("While condition must give one bool")


_ADOPT = {
    "tigris::Gelu": _adopt_tflite_gelu,
    "tigris::CumSum": _adopt_tflite_cumsum,
    "tigris::Svdf": _adopt_svdf,
    "tigris::Lstm": _adopt_lstm,
    "tigris::If": _adopt_if,
    "tigris::While": _adopt_while,
    "tigris::DetectionPostProcess": _adopt_detection,
    "tigris::Quantize": _adopt_quantize,
    "tigris::Dequantize": _adopt_quantize,
}
