"""QDQ folding and the int8 encodings of inputs, weights, biases and constants."""

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from tigris.capabilities import effective_operators
from tigris.graph.ir import AnalyzedGraph, OpNode, QuantParam, TensorInfo
from tigris.loaders.onnx.normalize.common import _fresh_name, _own_constant, _Uses


def _quantize_uniform_constants(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Encode a float constant operand of an int8 Add or Mul whose elements are
    all one value.

    QDQ exporters leave such a constant in float, as in a hard-swish written out
    as x * relu6(x + 3) / 6. One value v is exactly 127 at scale |v| / 127, so the
    int8 operator reads the same number; a constant of differing values would
    lose precision and stays refused.
    """
    if not ag.is_quantized:
        return ag
    float_constants = {name for op in ag.ops
                       if any(n in ag.tensors and not ag.tensors[n].is_constant
                              and ag.tensors[n].dtype == 1 for n in op.inputs)
                       for name in op.inputs if name in ag.weight_data}
    encoded_constants: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type not in ("Add", "Mul") or len(op.inputs) != 2:
            continue
        activations = [ag.tensors.get(n) for n in op.inputs if n not in ag.weight_data]
        if len(activations) != 1 or activations[0] is None or activations[0].quant is None:
            continue
        name = next(n for n in op.inputs if n in ag.weight_data)
        info, value = ag.tensors.get(name), np.asarray(ag.weight_data[name])
        if info is None or info.quant is not None or value.dtype.kind != "f" or value.size == 0:
            continue
        first = float(value.reshape(-1)[0])
        if not np.all(value == value.reshape(-1)[0]) or not np.isfinite(first):
            continue
        if name in float_constants:
            encoded = encoded_constants.get(name)
            if encoded is not None:
                op.inputs = [encoded if n == name else n for n in op.inputs]
                continue
            encoded = _fresh_name(ag, f"{name}_int8")
            encoded_constants[name] = encoded
            op.inputs = [encoded if n == name else n for n in op.inputs]
            name, info = encoded, replace(info, name=encoded)
            ag.tensors[name] = info
        scale = np.float32(abs(first) / 127.0) if first != 0.0 else np.float32(1.0)
        level = 0 if first == 0.0 else (127 if first > 0 else -127)
        ag.weight_data[name] = np.full(value.shape, level, dtype=np.int8)
        info.dtype = 3
        info.quant = QuantParam(scale=np.array([scale], dtype=np.float32),
                                zero_point=np.array([0], dtype=np.int32))
    return ag


def _transpose_quantized_weights(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold a Transpose of a dequantized constant weight into the weight.

    PyTorch exports nn.Linear as DequantizeLinear on the [out, in] weight, then
    Transpose, then Gemm. The int8 data is transposed in place of the float
    result and the quantization axis moves with it, so the Gemm reads an
    int8 weight like any other.
    """
    uses = _Uses(ag)
    removed: set[int] = set()
    for index, op in enumerate(ag.ops):
        if op.op_type != "Transpose" or len(op.inputs) != 1:
            continue
        dql = uses.producer(op.inputs[0])
        if (dql is None or dql.op_type != "DequantizeLinear" or uses.reads[op.inputs[0]] != 1
                or dql.inputs[0] not in ag.weight_data or uses.reads[dql.inputs[0]] != 1
                or any(n and n not in ag.weight_data for n in dql.inputs[1:])):
            continue
        data = ag.weight_data[dql.inputs[0]]
        perm = list(op.attrs.get("perm") or reversed(range(data.ndim)))
        if sorted(perm) != list(range(data.ndim)):
            continue
        scale = ag.weight_data[dql.inputs[1]]
        if scale.size > 1:
            axis = dql.attrs.get("axis", 1) % data.ndim
            dql.attrs["axis"] = perm.index(axis)
        ag.weight_data[dql.inputs[0]] = np.ascontiguousarray(np.transpose(data, perm))
        source = ag.tensors.get(dql.inputs[0])
        if source is not None:
            source.shape = tuple(source.shape[p] for p in perm)
        old, new = dql.outputs[0], op.outputs[0]
        dql.outputs[0] = new
        ag.tensors.pop(old, None)
        removed.add(index)
    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
    return ag


def _quantize_float_biases(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Restate a float32 bias of an int8 Conv, ConvTranspose or Gemm as int32.

    The int8 kernels add an int32 bias to the accumulator, whose scale is the
    input scale times the weight scale of each output channel. ONNX QDQ models
    from PyTorch's quantizers leave the bias in float32 with no
    DequantizeLinear, so it is rounded to that scale here, as the TFLite
    converter does. A bias this cannot express is left as it is and refused by
    validation.
    """
    uses = _Uses(ag)
    for op in ag.ops:
        if op.op_type not in {"Conv", "ConvTranspose", "Gemm"} or len(op.inputs) < 3:
            continue
        bias_name, weight_name = op.inputs[2], op.inputs[1]
        bias = ag.weight_data.get(bias_name)
        weights = ag.weight_data.get(weight_name)
        x_info = ag.tensors.get(op.inputs[0])
        w_info = ag.tensors.get(weight_name)
        if (bias is None or bias.dtype != np.float32 or weights is None
                or weights.dtype != np.int8 or x_info is None or x_info.dtype != 3
                or x_info.quant is None or x_info.quant.scale.size != 1
                or w_info is None or w_info.quant is None):
            continue
        channels = bias.size
        w_scale = np.asarray(w_info.quant.scale, np.float64).reshape(-1)
        if w_scale.size not in (1, channels):
            continue
        if w_scale.size > 1:
            # The per-channel scale has to run along the output channels.
            out_axis = {"Conv": 0, "ConvTranspose": 1}.get(
                op.op_type, 0 if op.attrs.get("transB", 0) else weights.ndim - 1)
            if w_info.quant.axis % weights.ndim != out_axis:
                continue
        scale = float(x_info.quant.scale.reshape(-1)[0]) * np.broadcast_to(w_scale, (channels,))
        if np.any(scale <= 0) or not np.all(np.isfinite(bias)):
            continue
        quantized = np.round(bias.astype(np.float64).reshape(-1) / scale)
        if np.any(np.abs(quantized) > np.iinfo(np.int32).max):
            continue
        target = bias_name
        if uses.reader_count(bias_name) > 1:
            target = _fresh_name(ag, f"{bias_name}__int32_{op.name or len(ag.weight_data)}")
            op.inputs[2] = target
        ag.weight_data[target] = quantized.astype(np.int32).reshape(bias.shape)
        ag.tensors[target] = TensorInfo(
            name=target, shape=tuple(bias.shape), dtype=6, is_constant=True,
            quant=QuantParam(scale=scale.astype(np.float32),
                             zero_point=np.zeros(channels, dtype=np.int32), axis=0))
    return ag


_LAYOUT_OPS = frozenset({"Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze"})


def _quantize_inputs_first(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Quantize a float model input before the layout steps that lead to its
    first QuantizeLinear.

    Exporters put a Transpose or Reshape on the float input and quantize its
    result, so the layout step reads float32 and writes int8. Per-tensor
    quantization commutes with moving elements, so a QuantizeLinear and
    DequantizeLinear with the same parameters on the input and after each
    intermediate step give the same values, and the fold turns the input into
    an int8 boundary and every step into an int8 one.
    """
    uses = _Uses(ag)
    inserted: dict[int, list[OpNode]] = {}
    for name in ag.model_inputs:
        info = ag.tensors.get(name)
        if info is None or info.dtype != 1 or name in ag.model_outputs:
            continue
        held = [name]
        op = None
        while uses.reader_count(held[-1]) == 1:
            op = ag.ops[uses.readers(held[-1])[0][0]]
            if op.op_type not in _LAYOUT_OPS or op.inputs[0] != held[-1]:
                break
            if op.outputs[0] in ag.model_outputs:
                op = None
                break
            held.append(op.outputs[0])
        if (op is None or op.op_type != "QuantizeLinear" or len(held) < 2
                or op.inputs[1] not in ag.weight_data
                or ag.weight_data[op.inputs[1]].size != 1):
            continue
        params = op.inputs[1:]
        quantized = ag.tensors.get(op.outputs[0])
        dtype = quantized.dtype if quantized is not None else 3
        for tensor in held[:-1]:
            first = ag.ops[uses.readers(tensor)[0][0]]
            shape = ag.tensors[tensor].shape
            q = _fresh_name(ag, f"{tensor}_input_q")
            ag.tensors[q] = TensorInfo(name=q, shape=shape, dtype=dtype)
            dq = _fresh_name(ag, f"{tensor}_input_dq")
            ag.tensors[dq] = TensorInfo(name=dq, shape=shape, dtype=1)
            inserted.setdefault(id(first), []).extend([
                OpNode(name=q, op_type="QuantizeLinear", inputs=[tensor, *params],
                       outputs=[q]),
                OpNode(name=dq, op_type="DequantizeLinear", inputs=[q, *params],
                       outputs=[dq]),
            ])
            first.inputs[0] = dq

    if not inserted:
        return ag
    ops: list[OpNode] = []
    for op in ag.ops:
        ops.extend(inserted.get(id(op), []))
        ops.append(op)
    ag.ops = ops
    return ag


_QDQ_TYPES = frozenset({"QuantizeLinear", "DequantizeLinear"})
_BINARY_OPS = frozenset({"Add", "Sub", "Mul", "Div", "PRelu", "Max", "Min",
                         "SquaredDifference", "FloorDiv", "FloorMod"})
_TYPED_OPS = frozenset({"ArgMax", "ArgMin", "Cast", "DetectionPostProcess", "If", "While",
                        "Quantize", "Dequantize", "Svdf", "Lstm"})


@dataclass
class _QdqFold:
    """What the two passes of one QDQ fold share."""

    uses: _Uses
    float_ops: set[int]
    crossing: set[int]
    float_results: set[str]
    subgraph: bool
    removed: set[int] = field(default_factory=set)
    # A scale or zero point may be shared by a weight pair and an activation
    # pair. Deleting it as soon as the weight pair is folded leaves the
    # activation pairs with no parameters to read, so every removal waits until
    # both passes have run.
    deferred_cleanup: set[str] = field(default_factory=set)


def _fold_qdq(ag: AnalyzedGraph, subgraph: bool = False) -> AnalyzedGraph:
    """Fold QuantizeLinear / DequantizeLinear ops.

    ONNX QDQ format places Q/DQ pairs around weights and activations:
    - Weight: float_W -> QuantizeLinear -> int8_W -> DequantizeLinear -> fake_float_W
    - Activation: tensor -> QuantizeLinear -> int8 -> DequantizeLinear -> fake_float

    This pass:
    1. Detects whether the model has any Q/DQ ops (early exit if not).
    2. For DequantizeLinear on weight inputs: extracts scale/zp, stores
       QuantParam on the weight TensorInfo, keeps raw int8 data.
    3. For QuantizeLinear on activations: extracts scale/zp, stores
       QuantParam on the quantized tensor, sets dtype to INT8.
    4. Retains conversions at float sections and folds encoded operator edges.
    5. Sets ag.is_quantized = True.

    Values an If or While passes in or out stay float32, so a pair at that
    boundary becomes a Dequantize or Quantize: around the operator, and in a
    ``subgraph`` at its inputs and outputs.
    """
    has_qdq = any(op.op_type in _QDQ_TYPES for op in ag.ops)
    if not has_qdq:
        return ag

    uses = _Uses(ag)
    fold = _QdqFold(uses, *_float_sections(ag, uses, subgraph), subgraph)
    _fold_weight_dequantizers(ag, fold)
    requantized = _fold_activation_pairs(ag, fold)

    # Now clean up all deferred scale/zp constants
    for name in fold.deferred_cleanup:
        if name in ag.weight_data:
            del ag.weight_data[name]
        if name in ag.tensors:
            del ag.tensors[name]

    if fold.removed:
        replacements = dict(requantized)
        ag.ops = [replacements.get(idx, op) for idx, op in enumerate(ag.ops)
                  if idx not in fold.removed or idx in replacements]
        ag.is_quantized = True

    return ag


def _float_sections(
    ag: AnalyzedGraph, uses: _Uses, subgraph: bool,
) -> tuple[set[int], set[int], set[str]]:
    """The operators that retain float arithmetic, the If and While operators,
    and the values that stay float32: the results of both and, in a
    ``subgraph``, its inputs."""
    # Operators with unencoded operands or results retain their float arithmetic.
    encoded = {op.inputs[0] for op in ag.ops if op.op_type == "QuantizeLinear"}
    float_only = effective_operators("reference") - effective_operators("s8_ref")
    float_required = {i for i, op in enumerate(ag.ops) if op.op_type in float_only}
    float_required.update(uses.producers[n] for n in ag.model_outputs
                          if n in uses.producers and ag.tensors[n].dtype == 1)
    pending = list(float_required)
    while pending:
        index = pending.pop()
        if ag.ops[index].op_type in _QDQ_TYPES:
            continue
        for name in ag.ops[index].inputs:
            parent = uses.producers.get(name)
            if parent is not None and parent not in float_required:
                float_required.add(parent)
                pending.append(parent)

    integer_ops = {i for i, op in enumerate(ag.ops)
                   if op.op_type not in _QDQ_TYPES | float_only and any(n in encoded for n in op.outputs)
                   and not _float_constant_boundary(ag, uses, op, float_required)}
    pending = list(integer_ops)
    while pending:
        for name in ag.ops[pending.pop()].inputs:
            parent = uses.producers.get(name)
            if (parent is not None and parent not in integer_ops
                    and ag.ops[parent].op_type not in _QDQ_TYPES | float_only
                    and not _float_constant_boundary(ag, uses, ag.ops[parent], float_required)):
                integer_ops.add(parent)
                pending.append(parent)
    float_ops: set[int] = set()
    for i, op in enumerate(ag.ops):
        if op.op_type in _QDQ_TYPES:
            continue
        unencoded_constant = _float_constant_boundary(ag, uses, op, float_required)
        from_dequantize = _from_activation_dequantize(ag, uses, op)
        from_section = any(uses.producers.get(n) in float_ops and n in ag.tensors
                           and ag.tensors[n].dtype == 1 for n in op.inputs)
        float_output = any(n in ag.tensors and ag.tensors[n].dtype == 1 for n in op.outputs)
        if (op.op_type not in _TYPED_OPS and float_output
                and (unencoded_constant or from_section or (from_dequantize and i not in integer_ops))):
            float_ops.add(i)
    float_results = {n for i in float_ops for n in ag.ops[i].outputs}
    crossing = {i for i, op in enumerate(ag.ops) if op.op_type in {"If", "While"}}
    float_results.update(n for i in crossing for n in ag.ops[i].outputs)
    if subgraph:
        float_results.update(ag.model_inputs)
    return float_ops, crossing, float_results


def _from_activation_dequantize(ag: AnalyzedGraph, uses: _Uses, op: OpNode) -> bool:
    """Whether ``op`` reads the DequantizeLinear of an activation."""
    for name in op.inputs:
        producer = uses.producer(name)
        if (producer is not None and producer.op_type == "DequantizeLinear"
                and producer.inputs[0] not in ag.weight_data):
            return True
    return False


def _float_constant_boundary(
    ag: AnalyzedGraph, uses: _Uses, op: OpNode, float_required: set[int],
) -> bool:
    """Whether ``op``, a binary operator over a dequantized activation and a
    float32 constant, retains its float arithmetic.

    It does when its result is required in float, or when it is not an Add or
    Mul, or a Sub or Div by a constant on the right, over one constant holding
    a single finite value (for Div, a finite reciprocal).
    """
    constants = [n for n in op.inputs if n in ag.weight_data
                 and ag.weight_data[n].dtype == np.float32]
    if op.op_type not in _BINARY_OPS or not constants or not _from_activation_dequantize(ag, uses, op):
        return False
    if any(uses.producers.get(n) in float_required for n in op.outputs):
        return True
    # Sub and Div with a right-hand constant lower to Add and Mul.
    lowerable = op.op_type in {"Add", "Mul"} or (
        op.op_type in {"Sub", "Div"} and len(op.inputs) == 2 and op.inputs[1] in constants)
    if not lowerable or len(constants) != 1 or len(op.inputs) != 2:
        return True
    value = ag.weight_data[constants[0]]
    if value.size == 0 or not np.all(np.isfinite(value)):
        return True
    if op.op_type == "Div":
        if np.any(value == 0.0):
            return True
        value = (1.0 / value.astype(np.float64)).astype(np.float32)
    return not (np.all(np.isfinite(value)) and np.all(value == value.reshape(-1)[0]))


def _bypass_dql(ag: AnalyzedGraph, fold: _QdqFold, index: int, source: str, *,
                constant: bool = False) -> None:
    """Point the readers of DequantizeLinear ``index`` at ``source``.

    A float operator, an If or While, or a subgraph output that reads it keeps
    a float32 value: a dequantized copy of a ``constant``, or a Dequantize.
    """
    dql = ag.ops[index]
    target = dql.outputs[0]
    fold.deferred_cleanup.update(n for n in dql.inputs[1:] if n)
    keep = any(c in fold.float_ops | fold.crossing for c, _ in fold.uses.readers(target)) or (
        fold.subgraph and target in ag.model_outputs)
    for consumer, position in fold.uses.readers(target):
        if consumer not in fold.float_ops | fold.crossing:
            ag.ops[consumer].inputs[position] = source
    if keep and constant:
        quant = ag.tensors[source].quant
        raw = ag.weight_data[source].astype(np.float64)
        shape = [1] * raw.ndim
        if quant.scale.size > 1:
            shape[quant.axis] = quant.scale.size
        scale = quant.scale.reshape(shape) if quant.scale.size > 1 else quant.scale[0]
        zero = quant.zero_point.reshape(shape) if quant.zero_point.size > 1 else quant.zero_point[0]
        ag.weight_data[target] = ((raw - zero) * scale).astype(np.float32)
        ag.tensors[target] = TensorInfo(name=target, shape=tuple(raw.shape), dtype=1, is_constant=True)
    elif keep:
        dql.op_type = "Dequantize"
        dql.inputs = [source]
        ag.tensors[target].dtype = 1
        ag.tensors[target].quant = None
    if not keep or constant:
        fold.removed.add(index)
    if not keep:
        if target in ag.model_outputs:
            ag.model_outputs = [source if n == target else n for n in ag.model_outputs]
        if target != source:
            ag.tensors.pop(target, None)


def _get_quant_param(ag: AnalyzedGraph, op: OpNode) -> tuple[QuantParam | None, bool]:
    """Extract QuantParam from a QuantizeLinear or DequantizeLinear op.

    Inputs: x, y_scale, y_zero_point (optional).
    For DQL: x, x_scale, x_zero_point (optional).

    Returns the parameters in the signed domain the kernels work in, and
    whether the model stated them as unsigned. uint8 value ``v`` and int8
    value ``v - 128`` denote the same real number under zero points that
    differ by the same 128, so shifting both is exact rather than a
    reinterpretation. The caller shifts stored data to match.
    """
    if len(op.inputs) < 2:
        return None, False
    scale_name = op.inputs[1]
    if scale_name not in ag.weight_data:
        return None, False
    scale = ag.weight_data[scale_name].astype(np.float32).flatten()

    unsigned = False
    if len(op.inputs) >= 3 and op.inputs[2] and op.inputs[2] in ag.weight_data:
        zp = ag.weight_data[op.inputs[2]].flatten()
        unsigned = zp.dtype == np.uint8
    else:
        # An omitted zero point means zero in the operator's own output
        # type, which ONNX defaults to uint8.
        produced = ag.tensors.get(op.outputs[0])
        unsigned = produced is not None and produced.dtype == 2
        zp = np.zeros_like(scale, dtype=np.int8)
    if unsigned:
        zp = (zp.astype(np.int32) - 128).astype(np.int32)

    axis = op.attrs.get("axis", 1)
    if scale.size == 1:
        axis = -1  # per-tensor

    return QuantParam(scale=scale, zero_point=zp, axis=axis), unsigned


def _to_signed(arr: np.ndarray) -> np.ndarray:
    """Restate uint8 storage in the signed domain its zero point moved to."""
    return (arr.astype(np.int16) - 128).astype(np.int8)


def _fold_weight_dequantizers(ag: AnalyzedGraph, fold: _QdqFold) -> None:
    """Fold each DequantizeLinear of a constant, of a QuantizeLinear of a float
    constant, and of a quantized model input."""
    uses = fold.uses
    # Pattern: weight_init -> QuantizeLinear -> int8 -> DequantizeLinear -> fake_float
    # Or simply: int8_init -> DequantizeLinear -> fake_float
    for i, op in enumerate(ag.ops):
        if op.op_type != "DequantizeLinear":
            continue

        dql_input = op.inputs[0]
        # Check if the input is a weight (constant/initializer)
        is_weight_dql = False

        # Case 1: int8 initializer -> DequantizeLinear
        if dql_input in ag.weight_data:
            is_weight_dql = True
            qp, unsigned = _get_quant_param(ag, op)
            if qp is None:
                continue

            if unsigned:
                ag.weight_data[dql_input] = _to_signed(ag.weight_data[dql_input])

            # Store QuantParam on the weight tensor
            if dql_input in ag.tensors:
                ag.tensors[dql_input].quant = qp
                ag.tensors[dql_input].dtype = 3  # INT8

            _bypass_dql(ag, fold, i, dql_input, constant=True)

        # Case 2: QuantizeLinear -> DequantizeLinear on a weight
        elif dql_input in uses.producers:
            ql_idx = uses.producers[dql_input]
            ql_op = ag.ops[ql_idx]
            if ql_op.op_type == "QuantizeLinear" and ql_op.inputs[0] in ag.weight_data:
                is_weight_dql = True
                original_weight = ql_op.inputs[0]

                # Get quant params from the DQL op
                qp, unsigned = _get_quant_param(ag, op)
                if qp is None:
                    continue

                # Quantize the float weight to int8
                scale = qp.scale
                zp = qp.zero_point.astype(np.int32)
                float_w = ag.weight_data[original_weight]

                if scale.size == 1:
                    int8_w = np.clip(
                        np.round(float_w / scale[0]) + zp[0], -128, 127
                    ).astype(np.int8)
                else:
                    # Per-channel: reshape scale for broadcasting
                    axis = qp.axis if qp.axis >= 0 else 1
                    bc_shape = [1] * float_w.ndim
                    bc_shape[axis] = scale.size
                    s = scale.reshape(bc_shape)
                    z = zp.reshape(bc_shape)
                    int8_w = np.clip(np.round(float_w / s) + z, -128, 127).astype(
                        np.int8
                    )

                # Replace float weight with int8
                ag.weight_data[original_weight] = int8_w
                if original_weight in ag.tensors:
                    ag.tensors[original_weight].dtype = 3  # INT8
                    ag.tensors[original_weight].quant = qp

                _bypass_dql(ag, fold, i, original_weight, constant=True)
                ag.tensors.pop(dql_input, None)
                fold.removed.add(ql_idx)

        # Case 3: a quantized model input -> DequantizeLinear. The input keeps
        # its place and becomes the int8 tensor the kernels read; the dtype
        # the model declares stays recorded, so the plan converts at the
        # boundary instead of carrying a float copy of the input.
        if (
            not is_weight_dql
            and dql_input in ag.model_inputs
            and dql_input not in uses.producers
            and dql_input in ag.tensors
            and ag.tensors[dql_input].dtype in (2, 3)  # uint8, int8
            and all(
                ag.ops[cons_idx].op_type == "DequantizeLinear"
                for cons_idx, _ in uses.readers(dql_input)
            )
        ):
            qp, _ = _get_quant_param(ag, op)
            if qp is not None:
                ag.is_quantized = True
                ag.tensors[dql_input].quant = qp
                ag.tensors[dql_input].dtype = 3  # INT8
                _bypass_dql(ag, fold, i, dql_input)


def _fold_activation_pairs(ag: AnalyzedGraph, fold: _QdqFold) -> list[tuple[int, OpNode]]:
    """Fold each QuantizeLinear of an activation into the tensor it encodes and
    bypass its DequantizeLinear readers. Returns, by index, the QuantizeLinear
    operators that become a Quantize."""
    uses = fold.uses
    # Pattern: activation -> QuantizeLinear -> int8 -> DequantizeLinear -> consumer
    requantized: list[tuple[int, OpNode]] = []
    for i, op in enumerate(ag.ops):
        if i in fold.removed:
            continue
        if op.op_type != "QuantizeLinear":
            continue

        ql_output = op.outputs[0]
        ql_input = op.inputs[0]

        # Skip if input is a weight (already handled above)
        if ql_input in ag.weight_data:
            continue

        # Get quant params from this QL op
        qp, unsigned = _get_quant_param(ag, op)
        if qp is None:
            continue

        # An activation another QuantizeLinear already encoded keeps that
        # encoding; a different one is a requantization of its own, which
        # writes the QuantizeLinear's output as an int8 tensor.
        source = ag.tensors.get(ql_input)
        if (source is not None and (ql_input in fold.float_results or (
                source.quant is not None and source.dtype == 3
                and not source.quant.same_scale_and_zero_point(qp)))):
            ag.tensors[ql_output] = TensorInfo(name=ql_output, shape=source.shape, dtype=3, quant=qp)
            requantize = OpNode(name=op.name, op_type="Quantize", inputs=[ql_input],
                                outputs=[ql_output], step=op.step)
            requantized.append((i, requantize))
            for dql_idx in [c for c, _ in uses.readers(ql_output)
                            if ag.ops[c].op_type == "DequantizeLinear"]:
                _bypass_dql(ag, fold, dql_idx, ql_output)
            fold.removed.add(i)
            fold.deferred_cleanup.update(name for name in op.inputs[1:] if name)
            continue

        # Store quant param on the activation tensor and set dtype to INT8
        if ql_input in ag.tensors:
            ag.tensors[ql_input].quant = qp
            ag.tensors[ql_input].dtype = 3  # INT8

        # Find the DQL consumer(s) of this QL's output
        dql_indices = []
        for cons_idx, pos in uses.readers(ql_output):
            if ag.ops[cons_idx].op_type == "DequantizeLinear":
                dql_indices.append(cons_idx)

        for dql_idx in dql_indices:
            _bypass_dql(ag, fold, dql_idx, ql_input)

        # A graph output taken straight off the QuantizeLinear moves onto the
        # tensor it quantizes, which now carries the parameters. The declared
        # dtype stays recorded by position, so the boundary still converts.
        if ql_output in ag.model_outputs:
            ag.model_outputs = [
                ql_input if n == ql_output else n for n in ag.model_outputs
            ]

        # Remove QL intermediate tensor
        if ql_output in ag.tensors and ql_output != ql_input:
            del ag.tensors[ql_output]

        fold.removed.add(i)

        # Defer QL scale/zp cleanup
        for inp_name in op.inputs[1:]:
            if inp_name:
                fold.deferred_cleanup.add(inp_name)
    return requantized


def _fold_bn_into_quantized_conv(ag: AnalyzedGraph, conv_op: OpNode, gamma, beta,
                                 mean, var, eps: float, reads: Counter) -> bool:
    """Fold y = a * x + c, the BN after a quantized Conv, into the Conv.

    The Conv output is s_in * s_w[k] * (acc + bias[k]) per output channel k, so
    a moves into the weight scale and c into the int32 bias at the new
    accumulator scale; only the bias rounds. A negative a negates the channel's
    int8 weights, which -128 cannot follow. Returns False, changing nothing,
    when the Conv is not in that form.
    """
    x_info = ag.tensors.get(conv_op.inputs[0])
    w_name = conv_op.inputs[1]
    w_info = ag.tensors.get(w_name)
    weights = ag.weight_data[w_name]
    if (weights.dtype != np.int8 or x_info is None or x_info.quant is None
            or w_info is None or w_info.quant is None
            or x_info.quant.scale.size != 1):
        return False
    channels = weights.shape[0]
    a = (np.asarray(gamma, np.float64) / np.sqrt(np.asarray(var, np.float64) + eps)).reshape(-1)
    c = (np.asarray(beta, np.float64) - np.asarray(mean, np.float64) * a).reshape(-1)
    w_scale = np.broadcast_to(np.asarray(w_info.quant.scale, np.float64), (channels,))
    w_zero = np.broadcast_to(np.asarray(w_info.quant.zero_point), (channels,))
    if a.size != channels or np.any(w_zero != 0) or np.any(a == 0):
        return False
    negative = a < 0
    if np.any(weights[negative] == -128):
        return False

    s_in = float(np.asarray(x_info.quant.scale).reshape(-1)[0])
    has_bias = len(conv_op.inputs) >= 3 and conv_op.inputs[2] in ag.weight_data
    if has_bias:
        b_info = ag.tensors.get(conv_op.inputs[2])
        raw = ag.weight_data[conv_op.inputs[2]]
        if raw.dtype == np.int32:
            if b_info is None or b_info.quant is None:
                return False
            b_scale = np.broadcast_to(np.asarray(b_info.quant.scale, np.float64), (channels,))
            bias = raw.astype(np.float64).reshape(-1) * b_scale
        else:
            bias = raw.astype(np.float64).reshape(-1)
    else:
        bias = np.zeros(channels)

    new_scale = w_scale * np.abs(a)
    folded = weights.copy()
    folded[negative] = -folded[negative]
    accumulator_scale = s_in * new_scale
    new_bias = np.round((a * bias + c) / accumulator_scale)
    if np.any(np.abs(new_bias) > np.iinfo(np.int32).max):
        return False

    w_name = _own_constant(ag, conv_op, 1, reads)
    ag.weight_data[w_name] = folded
    ag.tensors[w_name].quant = QuantParam(scale=new_scale.astype(np.float32),
                                          zero_point=np.zeros(channels, dtype=np.int8), axis=0)
    bias_quant = QuantParam(scale=accumulator_scale.astype(np.float32),
                            zero_point=np.zeros(channels, dtype=np.int32), axis=0)
    if has_bias:
        b_name = _own_constant(ag, conv_op, 2, reads)
    else:
        b_name = _fresh_name(ag, w_name + "_bn_bias")
        reads[b_name] += 1
        conv_op.inputs = list(conv_op.inputs[:2]) + [b_name] + list(conv_op.inputs[3:])
    ag.weight_data[b_name] = new_bias.astype(np.int32)
    b_info = ag.tensors.get(b_name)
    if b_info is None:
        ag.tensors[b_name] = TensorInfo(name=b_name, shape=(channels,), dtype=6,
                                        is_constant=True, quant=bias_quant)
    else:
        b_info.dtype = 6  # INT32
        b_info.quant = bias_quant
    return True
