"""Operators restated as operators the runtime has kernels for."""

import math
from dataclasses import replace

import numpy as np

from tigris.graph.ir import AnalyzedGraph, Layout, OpNode, QuantParam, TensorInfo
from tigris.loaders.onnx.normalize.attributes import _reduction_axes
from tigris.loaders.onnx.normalize.common import _forget_unread, _fresh_name, _own_constant, _Uses


def _lower_legacy_softmax(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Restate a Softmax from before opset 13 in the later operator's terms.

    Up to opset 12, Softmax flattens every axis from ``axis`` onward into one
    row and normalizes that row, with ``axis`` defaulting to 1. From opset 13
    it normalizes the single axis ``axis``, defaulting to -1. The two agree
    when ``axis`` is the last one; otherwise the old operator becomes a Reshape
    to [outer, inner], a Softmax over the inner axis and a Reshape back.
    """
    if ag.opset == 0 or ag.opset >= 13:
        return ag
    rewritten: list[OpNode] = []
    for op in ag.ops:
        if op.op_type not in {"Softmax", "LogSoftmax"}:
            rewritten.append(op)
            continue
        info = ag.tensors.get(op.inputs[0])
        out_info = ag.tensors.get(op.outputs[0])
        if info is None or out_info is None or not info.shape:
            raise ValueError(
                f"Softmax '{op.name}' before opset 13 needs a known input shape")
        rank = len(info.shape)
        axis = int(op.attrs.get("axis", 1))
        axis = axis + rank if axis < 0 else axis
        if not 0 <= axis < rank:
            raise ValueError(f"Softmax '{op.name}' has axis {axis} outside rank {rank}")
        if axis == rank - 1:
            op.attrs = {**op.attrs, "axis": -1}
            rewritten.append(op)
            continue
        outer = int(np.prod(info.shape[:axis], dtype=np.int64))
        inner = int(np.prod(info.shape[axis:], dtype=np.int64))
        flat_in = _fresh_name(ag, f"{op.name}_rows_in")
        ag.tensors[flat_in] = TensorInfo(
            name=flat_in, shape=(outer, inner), dtype=info.dtype,
            quant=info.quant, layout=Layout.LINEAR)
        flat_out = _fresh_name(ag, f"{op.name}_rows_out")
        ag.tensors[flat_out] = TensorInfo(
            name=flat_out, shape=(outer, inner), dtype=out_info.dtype,
            quant=out_info.quant, layout=Layout.LINEAR)
        rewritten.append(OpNode(
            name=f"{op.name}_rows", op_type="Reshape",
            inputs=[op.inputs[0]], outputs=[flat_in]))
        rewritten.append(OpNode(
            name=op.name, op_type=op.op_type,
            inputs=[flat_in], outputs=[flat_out], attrs={"axis": -1}))
        rewritten.append(OpNode(
            name=f"{op.name}_shape", op_type="Reshape",
            inputs=[flat_out], outputs=[op.outputs[0]]))
    ag.ops = rewritten
    return ag


def _fold_float_square(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Keep quantized multiplication distinct from the float-only square kernel."""
    for op in ag.ops:
        if op.op_type == "Mul" and len(op.inputs) == 2 and op.inputs[0] == op.inputs[1]:
            tensor = ag.tensors.get(op.inputs[0])
            if tensor is not None and tensor.dtype == 1:
                op.op_type, op.inputs = "Square", op.inputs[:1]
    return ag


def _standalone_bn_as_affine(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Run a float BatchNormalization no convolution absorbed as Mul then Add.

    Inference-time batch normalization is y = x * a + c per channel, with
    a = gamma / sqrt(var + eps) and c = beta - mean * a; pre-activation networks
    place it after an Add, where no weight can take it. An int8 one would need
    per-channel int8 constants and stays refused.
    """
    if ag.is_quantized:
        return ag
    rewritten: list[OpNode] = []
    for op in ag.ops:
        source = ag.tensors.get(op.inputs[0]) if op.inputs else None
        params = [ag.weight_data.get(n) for n in op.inputs[1:5]]
        if (op.op_type != "BatchNormalization" or len(op.inputs) < 5 or source is None
                or source.quant is not None or any(v is None for v in params) or len(source.shape) < 2):
            rewritten.append(op)
            continue
        gamma, beta, mean, var = (np.asarray(v, dtype=np.float32).reshape(-1) for v in params)
        eps = np.float32(op.attrs.get("epsilon", 1e-5))
        scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
        shift = (beta - mean * scale).astype(np.float32)
        broadcast = (scale.size, *[1] * (len(source.shape) - 2))
        names = []
        for label, value in (("scale", scale), ("shift", shift)):
            name = _fresh_name(ag, f"{op.name or op.outputs[0]}_{label}")
            ag.weight_data[name] = value.reshape(broadcast)
            ag.tensors[name] = TensorInfo(name=name, shape=broadcast, dtype=1, is_constant=True)
            names.append(name)
        scaled = _fresh_name(ag, f"{op.outputs[0]}_scaled")
        ag.tensors[scaled] = replace(ag.tensors[op.outputs[0]], name=scaled)
        rewritten.append(OpNode(name=f"{op.name}_scale", op_type="Mul", inputs=[op.inputs[0], names[0]],
                                outputs=[scaled]))
        rewritten.append(OpNode(name=op.name, op_type="Add", inputs=[scaled, names[1]],
                                outputs=list(op.outputs[:1])))
    ag.ops = rewritten
    return ag


def _fold_gemm_scalars(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold Gemm's alpha and beta into the constants they scale.

    The plan has no field for either, and the fully-connected kernel computes
    Y = X * W^T + B, so a Gemm carrying them would return a silently wrong
    answer if they were left in place. Scaling a constant weight by alpha and a
    constant bias by beta is exact and leaves nothing for the plan to express.
    transA transposes an activation, which no constant can absorb; validation
    refuses it.
    """
    reads = _Uses(ag).reads
    for op in ag.ops:
        if op.op_type != "Gemm" or len(op.inputs) < 2:
            continue

        alpha = float(op.attrs.get("alpha", 1.0))
        if alpha != 1.0:
            weight = ag.weight_data.get(op.inputs[1])
            if weight is not None and weight.dtype == np.float32:
                scaled = _own_constant(ag, op, 1, reads)
                ag.weight_data[scaled] = np.ascontiguousarray(weight * np.float32(alpha))
                op.attrs["alpha"] = 1.0

        beta = float(op.attrs.get("beta", 1.0))
        if beta != 1.0:
            if len(op.inputs) < 3:
                # beta scales C; with no C there is no term for it to scale.
                op.attrs["beta"] = 1.0
            else:
                bias = ag.weight_data.get(op.inputs[2])
                if bias is not None and bias.dtype == np.float32:
                    scaled = _own_constant(ag, op, 2, reads)
                    ag.weight_data[scaled] = np.ascontiguousarray(bias * np.float32(beta))
                    op.attrs["beta"] = 1.0
    return ag


def _relabel_matmul_to_gemm(ag: AnalyzedGraph, transposed: dict[str, str]) -> AnalyzedGraph:
    """Put a constant-weight matrix product in the layout the kernels read.

    The fully-connected kernels index the weight as ``W[oc * IC + ic]``, which
    is ONNX ``Gemm`` with ``transB=1``. An ONNX ``MatMul`` states the same
    product with the weight the other way round, and so does a ``Gemm`` that
    leaves ``transB`` at its default, so both need the constant transposed
    before they mean what the kernels compute.

    A product whose second operand is not a constant matrix is left alone: the
    runtime has no kernel for it and rejects the plan rather than guessing.
    ``transposed`` is the record ``_transpose_product_weights`` keeps.
    """
    products: dict[str, list[OpNode]] = {}
    for op in ag.ops:
        if op.op_type == "MatMul":
            pass
        elif op.op_type == "Gemm" and int(op.attrs.get("transB", 0)) != 1:
            pass
        else:
            continue
        if op.attrs.get("transA"):
            continue
        if float(op.attrs.get("alpha", 1.0)) != 1.0:
            continue
        if float(op.attrs.get("beta", 1.0)) != 1.0:
            continue
        if len(op.inputs) < 2:
            continue

        weight = ag.weight_data.get(op.inputs[1])
        if weight is None or weight.ndim != 2:
            continue
        data_info = ag.tensors.get(op.inputs[0])
        if data_info is None or len(data_info.shape) != 2:
            continue
        products.setdefault(op.inputs[1], []).append(op)

    _transpose_product_weights(ag, products, transposed)
    for op in (op for readers in products.values() for op in readers):
        op.op_type = "Gemm"
        op.attrs["transB"] = 1
    return ag


def _transpose_product_weights(
    ag: AnalyzedGraph, products: dict[str, list[OpNode]], transposed: dict[str, str],
) -> None:
    """Transpose each constant matrix for the products that read it as their weight.

    ``products`` maps a constant to the products that need it transposed. When
    they are all its readers it is transposed in place, once; otherwise they
    read one transposed copy and every other reader keeps the original.
    ``transposed`` maps a constant to the copy an earlier call made of it, so
    products relabeled before layouts and products lowered after them read
    one copy, as long as it still holds the transpose.
    """
    reads = _Uses(ag).reads
    for name, readers in products.items():
        copy = transposed.get(name)
        if copy is not None and _holds_transpose(ag, copy, name):
            for op in readers:
                op.inputs[1] = copy
            continue
        target = name
        if reads[name] != len(readers):
            target = _fresh_name(ag, name)
            transposed[name] = target
            for op in readers:
                op.inputs[1] = target
        ag.weight_data[target] = np.ascontiguousarray(ag.weight_data[name].T)
        info = ag.tensors.get(name)
        if info is not None:
            ag.tensors[target] = _transposed_info(info, target)


def _transposed_info(info: TensorInfo, name: str) -> TensorInfo:
    """``info`` for the transpose of its matrix, under ``name``."""
    quant = info.quant
    if quant is not None and quant.axis in (0, 1):
        quant = QuantParam(scale=quant.scale, zero_point=quant.zero_point, axis=1 - quant.axis)
    return replace(info, name=name, shape=tuple(reversed(info.shape)), quant=quant)


def _holds_transpose(ag: AnalyzedGraph, copy: str, name: str) -> bool:
    """Whether constant ``copy`` reads as the transpose of constant ``name``."""
    if copy not in ag.weight_data or name not in ag.weight_data:
        return False
    value, held = ag.weight_data[name].T, ag.weight_data[copy]
    if (held.dtype, held.shape) != (value.dtype, value.shape) or (
            held.tobytes() != value.tobytes()):
        return False
    info, copy_info = ag.tensors.get(name), ag.tensors.get(copy)
    if info is None or copy_info is None:
        return info is copy_info
    expected = _transposed_info(info, copy).quant
    if copy_info.dtype != info.dtype or (expected is None) != (copy_info.quant is None):
        return False
    return expected is None or (expected.axis == copy_info.quant.axis
                                and expected.same_scale_and_zero_point(copy_info.quant))


def _fold_sub_constant_to_add(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Rewrite a constant subtrahend as an added negation.

    Sub does not commute and the plan records only that an operand is constant,
    not which side it was on, so the runtime takes two tensor operands. A
    constant on the right has an exact commutative equivalent, x + (-c), which
    the Add path already carries, including folding it into a producer's bias.
    A constant on the left has no such equivalent and is left for validation to
    reject.
    """
    for op in ag.ops:
        if op.op_type != "Sub" or len(op.inputs) != 2:
            continue
        subtrahend = op.inputs[1]
        if subtrahend not in ag.weight_data or op.inputs[0] in ag.weight_data:
            continue
        constant = ag.weight_data[subtrahend]
        if constant.dtype != np.float32:
            continue
        ag.weight_data[subtrahend] = np.ascontiguousarray(-constant)
        op.op_type = "Add"
    return ag


def _fold_div_constant_to_mul(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Rewrite a constant divisor as a multiplication by its reciprocal.

    Div requires two dynamic operands; a constant divisor uses the Mul
    constant path: a GELU exports as ``Erf(x / sqrt(2))`` and an attention
    scale as a division by the head width. A divisor on the left has no such
    equivalent, and a zero in one would turn an infinity the model states into
    a finite number, so both are left for validation to reject.

    The reciprocal is taken in double precision and stored back as float32,
    which is the nearest representable inverse rather than the one a float32
    division would produce. The two differ by at most a half ulp of the
    reciprocal, well inside the tolerance the contract gate holds float to.

    It goes into a tensor of its own rather than over the divisor. One
    constant commonly feeds several divisions, since an exporter writes the
    same sqrt(2) into every GELU and a deduplicating pass then hands them all
    one initializer; inverting that in place would invert it once per reader.
    The same constant may also be read as something other than a divisor,
    which a rewrite in place would corrupt outright.
    """
    reciprocals: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type != "Div" or len(op.inputs) != 2:
            continue
        divisor = op.inputs[1]
        if divisor not in ag.weight_data or op.inputs[0] in ag.weight_data:
            continue
        name = reciprocals.get(divisor)
        if name is None:
            constant = ag.weight_data[divisor]
            if constant.dtype != np.float32 or constant.size == 0:
                continue
            if not np.all(np.isfinite(constant)) or np.any(constant == 0.0):
                continue
            reciprocal = (1.0 / constant.astype(np.float64)).astype(np.float32)
            if not np.all(np.isfinite(reciprocal)):
                continue
            name = _fresh_name(ag, f"{divisor}_reciprocal")
            ag.weight_data[name] = np.ascontiguousarray(reciprocal)
            ag.tensors[name] = replace(ag.tensors[divisor], name=name)
            reciprocals[divisor] = name
        op.inputs[1] = name
        op.op_type = "Mul"
    return ag


def _decompose_silu(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace Silu ops with Sigmoid + Mul sequence.

    SiLU(x) = x * sigmoid(x). Some exporters (e.g. Ultralytics opset-11)
    already emit Sigmoid + Mul - this pass handles the case where a single
    ``Silu`` op is present.
    """
    new_ops: list[OpNode] = []

    for op in ag.ops:
        if op.op_type != "Silu":
            new_ops.append(op)
            continue

        x_name = op.inputs[0]
        y_name = op.outputs[0]

        # Create intermediate tensor for sigmoid output
        sig_name = _fresh_name(ag, f"{x_name}_sigmoid")
        x_info = ag.tensors.get(x_name)
        if x_info:
            ag.tensors[sig_name] = TensorInfo(
                name=sig_name,
                shape=x_info.shape,
                dtype=x_info.dtype,
                is_constant=False,
                quant=x_info.quant,
            )

        # Sigmoid op
        sig_op = OpNode(
            name=f"{op.name}/Sigmoid",
            op_type="Sigmoid",
            inputs=[x_name],
            outputs=[sig_name],
            attrs={},
        )
        # Mul op: x * sigmoid(x)
        mul_op = OpNode(
            name=f"{op.name}/Mul",
            op_type="Mul",
            inputs=[x_name, sig_name],
            outputs=[y_name],
            attrs={},
        )
        new_ops.append(sig_op)
        new_ops.append(mul_op)

    ag.ops = new_ops
    return ag


def _relabel_depthwise(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel Conv ops with group == C_in as DepthwiseConv.

    Each group reads one input channel and may write several output channels,
    the channel multiplier, so the output channel count is a multiple of group.
    """
    for op in ag.ops:
        if op.op_type != "Conv":
            continue
        group = op.attrs.get("group", 1)
        if group <= 1:
            continue
        # For depthwise: group == C_in. Check via weight shape.
        w_name = op.inputs[1] if len(op.inputs) >= 2 else None
        if w_name and w_name in ag.weight_data:
            W = ag.weight_data[w_name]
            # Standard Conv weight: [OC, IC/G, KH, KW]
            # Depthwise: IC/G == 1 and OC a multiple of group
            if W.ndim >= 2 and W.shape[1] == 1 and W.shape[0] % group == 0:
                op.op_type = "DepthwiseConv"
        else:
            # No weight data: check tensor info
            if w_name and w_name in ag.tensors:
                w_info = ag.tensors[w_name]
                if (
                    len(w_info.shape) >= 2
                    and w_info.shape[1] == 1
                    and w_info.shape[0] % group == 0
                ):
                    op.op_type = "DepthwiseConv"

    return ag


def _split_grouped_convs(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Run a grouped convolution as one convolution per group.

    A Conv with 1 < group that is not depthwise reads, per group, its own run of
    input channels and writes its own run of output channels: a Split of the
    input along the channel axis, a Conv of group 1 on each part with that
    part's filters, bias and per-channel quantization, and a Concat of the
    results. Each output channel sums the same products as before.
    """
    rewritten: list[OpNode] = []
    for op in ag.ops:
        if op.op_type != "Conv" or len(op.inputs) < 2 or not op.outputs:
            rewritten.append(op)
            continue
        group = int(op.attrs.get("group", 1))
        weight = ag.weight_data.get(op.inputs[1])
        source = ag.tensors.get(op.inputs[0])
        result = ag.tensors.get(op.outputs[0])
        if (group <= 1 or weight is None or source is None or result is None
                or len(source.shape) < 3 or source.shape[1] % group or weight.shape[0] % group):
            rewritten.append(op)
            continue
        inputs_per, outputs_per = source.shape[1] // group, weight.shape[0] // group
        w_info = ag.tensors[op.inputs[1]]
        bias = ag.weight_data.get(op.inputs[2]) if len(op.inputs) > 2 and op.inputs[2] else None
        parts_in, parts_out = [], []
        for g in range(group):
            part = _fresh_name(ag, f"{op.inputs[0]}_group{g}")
            ag.tensors[part] = replace(source, name=part, shape=(source.shape[0], inputs_per, *source.shape[2:]))
            parts_in.append(part)
            out = _fresh_name(ag, f"{op.outputs[0]}_group{g}")
            ag.tensors[out] = replace(result, name=out, shape=(result.shape[0], outputs_per, *result.shape[2:]))
            parts_out.append(out)
        rewritten.append(OpNode(name=f"{op.name}_groups", op_type="Split", inputs=[op.inputs[0]],
                                outputs=parts_in, attrs={"axis": 1}))
        attrs = {**op.attrs, "group": 1}
        for g in range(group):
            rows = slice(g * outputs_per, (g + 1) * outputs_per)
            w_name = _fresh_name(ag, f"{op.inputs[1]}_group{g}")
            ag.weight_data[w_name] = np.ascontiguousarray(weight[rows])
            quant = w_info.quant
            if quant is not None and np.asarray(quant.scale).size > 1:
                quant = QuantParam(scale=np.asarray(quant.scale)[rows], zero_point=np.asarray(quant.zero_point)[rows],
                                   axis=quant.axis)
            ag.tensors[w_name] = replace(w_info, name=w_name, shape=ag.weight_data[w_name].shape, quant=quant)
            inputs = [parts_in[g], w_name]
            if bias is not None:
                b_info = ag.tensors[op.inputs[2]]
                b_name = _fresh_name(ag, f"{op.inputs[2]}_group{g}")
                ag.weight_data[b_name] = np.ascontiguousarray(bias[rows])
                b_quant = b_info.quant
                if b_quant is not None and np.asarray(b_quant.scale).size > 1:
                    b_quant = QuantParam(scale=np.asarray(b_quant.scale)[rows],
                                         zero_point=np.asarray(b_quant.zero_point)[rows], axis=b_quant.axis)
                ag.tensors[b_name] = replace(b_info, name=b_name, shape=ag.weight_data[b_name].shape, quant=b_quant)
                inputs.append(b_name)
            rewritten.append(OpNode(name=f"{op.name}_group{g}", op_type="Conv", inputs=inputs,
                                    outputs=[parts_out[g]], attrs=dict(attrs)))
        rewritten.append(OpNode(name=op.name, op_type="Concat", inputs=parts_out,
                                outputs=[op.outputs[0]], attrs={"axis": 1}))
    ag.ops = rewritten
    return ag


def _relabel_conv1d(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel 1D Conv ops as Conv1D, and run a 1D depthwise one as 2D.

    A convolution is 1D by its kernel_shape or, when the export leaves that
    out, by its weight's rank. Conv1D has no groups, so a 1D DepthwiseConv runs
    as a 2D one over a height of one: [N, C, L] and [N, C, 1, L] are the same
    bytes in channels-last storage, and two Reshapes carry the tensor across.

    Must run after BN fold (which only matches Conv->BN, not Conv1D->BN)
    and after depthwise relabeling (which already changed group==C_in convs).
    """
    rewritten: list[OpNode] = []
    for op in ag.ops:
        rewritten.append(op)
        if op.op_type not in ("Conv", "DepthwiseConv"):
            continue
        weight = ag.weight_data.get(op.inputs[1]) if len(op.inputs) > 1 else None
        kernel = op.attrs.get("kernel_shape")
        one_dimensional = len(kernel) == 1 if kernel is not None else weight is not None and weight.ndim == 3
        if not one_dimensional:
            continue
        if op.op_type == "Conv":
            op.op_type = "Conv1D"
            continue
        source, result = ag.tensors[op.inputs[0]], ag.tensors[op.outputs[0]]
        if len(source.shape) != 3 or len(result.shape) != 3 or weight is None:
            continue
        flat = _fresh_name(ag, f"{op.inputs[1]}_2d")
        ag.weight_data[flat] = weight.reshape(weight.shape[0], weight.shape[1], 1, weight.shape[2])
        ag.tensors[flat] = replace(ag.tensors[op.inputs[1]], name=flat, shape=ag.weight_data[flat].shape)
        wide_in = _fresh_name(ag, f"{op.inputs[0]}_2d")
        wide_out = _fresh_name(ag, f"{op.outputs[0]}_2d")
        ag.tensors[wide_in] = replace(source, name=wide_in, shape=(*source.shape[:2], 1, source.shape[2]))
        ag.tensors[wide_out] = replace(result, name=wide_out, shape=(*result.shape[:2], 1, result.shape[2]))
        pads = [int(p) for p in op.attrs.get("pads", [0, 0])]
        op.attrs = {**op.attrs,
                    "kernel_shape": [1, int(weight.shape[2])],
                    "strides": [1, *[int(v) for v in op.attrs.get("strides", [1])]],
                    "dilations": [1, *[int(v) for v in op.attrs.get("dilations", [1])]],
                    "pads": [0, pads[0], 0, pads[1]]}
        rewritten[-1:] = [
            OpNode(name=f"{op.name}_to_2d", op_type="Reshape", inputs=[op.inputs[0]], outputs=[wide_in]),
            op,
            OpNode(name=f"{op.name}_to_1d", op_type="Reshape", inputs=[wide_out], outputs=[op.outputs[0]])]
        op.inputs = [wide_in, flat, *op.inputs[2:]]
        op.outputs = [wide_out]
    ag.ops = rewritten
    return ag


def _clip_to_relu6(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace a Clip that states an activation the runtime already has.

    Clip(0, 6) is Relu6. Clip(0, unbounded) is Relu, which exporters emit in
    place of a Relu often enough to be worth recognizing: an absent upper bound,
    an infinite one, or one at the float maximum all mean the same thing.
    Anything else keeps its bounds and has no kernel to run on.
    """
    for op in ag.ops:
        if op.op_type != "Clip":
            continue

        min_val = None
        max_val = None

        # Clip can have min/max as attributes (opset < 11) or inputs (opset >= 11)
        if "min" in op.attrs:
            min_val = op.attrs["min"]
        if "max" in op.attrs:
            max_val = op.attrs["max"]

        # Opset 11 and later pass min and max as inputs.
        if min_val is None and len(op.inputs) >= 2 and op.inputs[1] in ag.weight_data:
            arr = ag.weight_data[op.inputs[1]]
            if arr.size == 1:
                min_val = float(arr.flat[0])
        if max_val is None and len(op.inputs) >= 3 and op.inputs[2] in ag.weight_data:
            arr = ag.weight_data[op.inputs[2]]
            if arr.size == 1:
                max_val = float(arr.flat[0])

        if min_val is None or abs(min_val) >= 1e-6:
            continue

        unbounded_above = (
            max_val is None
            or math.isinf(max_val)
            or max_val >= np.finfo(np.float32).max
        )
        if max_val is not None and abs(max_val - 6.0) < 1e-6:
            op.op_type = "Relu6"
        elif unbounded_above:
            op.op_type = "Relu"
        else:
            continue

        # Keep only the data input, drop min/max constant inputs
        op.inputs = [op.inputs[0]]
        op.attrs = {}

    return ag


def _reduce_mean_to_gap(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace a spatial ReduceMean or ReduceMax with its global pool.

    Handles both attribute-based axes (opset < 18) and input-based axes
    (opset >= 18).
    """
    reductions = {"ReduceMean": "GlobalAveragePool", "ReduceMax": "GlobalMaxPool"}

    ops: list[OpNode] = []
    dropped_axes: list[str] = []
    for op in ag.ops:
        ops.append(op)
        if op.op_type not in reductions:
            continue

        axes = _reduction_axes(ag, op, prefer="attribute")
        if axes is None:
            continue

        source = ag.tensors.get(op.inputs[0])
        rank = len(source.shape) if source is not None else 0
        if rank == 0:
            continue
        axes_norm = set(int(a) % rank for a in axes)

        # An operator that keeps its own type still has to carry its axes in
        # one place, because the emitter reads them from the attribute and an
        # opset-18 model states them as an input.
        op.attrs["axes"] = sorted(axes_norm)

        # Only a rank-4 mean over both spatial axes is the global pool. The
        # axes read the same in either order the graph may state them.
        if rank != 4 or axes_norm not in ({2, 3}, {1, 2}):
            continue

        kept = int(op.attrs.get("keepdims", 1)) == 1
        if op.op_type == "ReduceMax" and (axes_norm != {2, 3} or not kept):
            continue
        op.op_type = reductions[op.op_type]
        op.attrs = {}
        if not kept:
            # The pool keeps rank 4; with both reduced axes at 1, a Flatten
            # from axis 1 gives the [N, C] the reduction produced.
            result = ag.tensors[op.outputs[0]]
            pooled = _fresh_name(ag, f"{op.outputs[0]}_pooled")
            shape = tuple(1 if axis in axes_norm else dim for axis, dim in enumerate(source.shape))
            ag.tensors[pooled] = TensorInfo(name=pooled, shape=shape, dtype=result.dtype,
                                            quant=result.quant)
            ops.append(OpNode(name=f"{op.name}_flatten", op_type="Flatten", inputs=[pooled],
                              outputs=[op.outputs[0]], attrs={"axis": 1}))
            op.outputs = [pooled]

        # Remove axes input and clean up axes tensor
        if len(op.inputs) >= 2:
            dropped_axes.append(op.inputs[1])
            op.inputs = [op.inputs[0]]

    ag.ops = ops
    _forget_unread(ag, dropped_axes)
    return ag


def _whole_map_average_pool_to_gap(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace an AveragePool whose one window is the whole map with its global pool.

    ``F.avg_pool2d(x, x.shape[2:])`` exports as AveragePool with the kernel set
    to the input's height and width. With no padding it averages the same
    samples by the same count as GlobalAveragePool, which the compiler streams
    through fast memory a band at a time; as a pooling window it spans the
    whole height, so no band could be smaller than the input. An int8 pool is
    marked so it keeps AveragePool rounding, which differs from a mean's.
    """
    for op in ag.ops:
        if op.op_type != "AveragePool" or len(op.inputs) != 1:
            continue
        source = ag.tensors.get(op.inputs[0])
        result = ag.tensors.get(op.outputs[0]) if op.outputs else None
        if source is None or result is None:
            continue
        if len(source.shape) != 4 or len(result.shape) != 4:
            continue
        kernel = [int(k) for k in op.attrs.get("kernel_shape", [])]
        pads = [int(p) for p in op.attrs.get("pads", [0, 0, 0, 0])]
        if kernel != [int(source.shape[2]), int(source.shape[3])]:
            continue
        if any(pads) or int(op.attrs.get("ceil_mode", 0)) != 0:
            continue
        if tuple(result.shape[2:]) != (1, 1):
            continue
        op.op_type = "GlobalAveragePool"
        # An int8 AveragePool rounds its integer mean like TFLite
        # AVERAGE_POOL_2D, a global pool like TFLite MEAN; the attribute keeps
        # the pool's own rounding after the rewrite.
        op.attrs = {"pool_rounding": "average"} if source.quant is not None else {}
    return ag
