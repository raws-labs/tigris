"""Attributes and operands restated in the form the emitter and the runtime read."""

import numpy as np

from tigris.analysis.broadcast import DENSE, GENERAL, PERIODIC, stored_operands
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders.onnx.normalize.common import _fresh_name


def _explicit_auto_pad(ag: AnalyzedGraph) -> AnalyzedGraph:
    """State a convolution's or pool's auto_pad as the pads it stands for.

    VALID pads nothing. SAME_UPPER and SAME_LOWER pad so the output extent is
    the input's divided by the stride, rounded up, the odd element at the end
    or at the start, as ONNX defines them. ConvTranspose keeps its own
    definition and is left as it is.
    """
    for op in ag.ops:
        auto_pad = op.attrs.get("auto_pad", "NOTSET")
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode()
        if op.op_type not in ("Conv", "MaxPool", "AveragePool", "LpPool") or auto_pad in ("", "NOTSET"):
            continue
        # ONNX forbids pads next to auto_pad; such an operator stays refused.
        if any(int(v) for v in op.attrs.get("pads", [])):
            continue
        source = ag.tensors.get(op.inputs[0])
        if source is None or len(source.shape) < 3:
            continue
        spatial = [int(d) for d in source.shape[2:]]
        kernel = op.attrs.get("kernel_shape")
        if kernel is None and op.op_type == "Conv" and op.inputs[1] in ag.weight_data:
            kernel = ag.weight_data[op.inputs[1]].shape[2:]
        if kernel is None or len(kernel) != len(spatial):
            continue
        strides = [int(v) for v in op.attrs.get("strides", [1] * len(spatial))]
        dilations = [int(v) for v in op.attrs.get("dilations", [1] * len(spatial))]
        begins, ends = [], []
        for extent, k, stride, dilation in zip(spatial, kernel, strides, dilations):
            if auto_pad == "VALID":
                total = 0
            else:
                out = -(-extent // stride)
                total = max((out - 1) * stride + (int(k) - 1) * dilation + 1 - extent, 0)
            small = total // 2
            begins.append(total - small if auto_pad == "SAME_LOWER" else small)
            ends.append(small if auto_pad == "SAME_LOWER" else total - small)
        attrs = dict(op.attrs)
        attrs.pop("auto_pad")
        attrs["pads"] = begins + ends
        op.attrs = attrs
    return ag


def _ceil_mode_as_padding(ag: AnalyzedGraph) -> AnalyzedGraph:
    """State a pool's ceil_mode as end padding.

    Rounding the output extent up instead of down adds at most one window per
    axis, which starts inside the input and runs past its end. The same window
    comes from padding the end by the overhang: the pool kernels skip padded
    positions, so a max or an average excluding padding is unchanged. An
    overhang that would cover a whole window is left to the refusal.
    """
    for op in ag.ops:
        if op.op_type not in ("MaxPool", "AveragePool") or int(op.attrs.get("ceil_mode", 0)) != 1:
            continue
        source, result = ag.tensors.get(op.inputs[0]), ag.tensors.get(op.outputs[0])
        if source is None or result is None or len(source.shape) < 3 or len(result.shape) != len(source.shape):
            continue
        rank = len(source.shape) - 2
        kernel = [int(k) for k in op.attrs.get("kernel_shape", [])]
        strides = [int(v) for v in op.attrs.get("strides", [1] * rank)]
        dilations = [int(v) for v in op.attrs.get("dilations", [1] * rank)]
        pads = [int(v) for v in op.attrs.get("pads", [0] * (2 * rank))]
        if len(kernel) != rank:
            continue
        ends = list(pads[rank:])
        fits = True
        for axis in range(rank):
            span = (kernel[axis] - 1) * dilations[axis] + 1
            needed = (int(result.shape[2 + axis]) - 1) * strides[axis] + span
            extra = needed - (int(source.shape[2 + axis]) + pads[axis] + pads[rank + axis])
            if extra >= span:
                fits = False
            ends[axis] += max(extra, 0)
        if not fits:
            continue
        op.attrs = {**op.attrs, "ceil_mode": 0, "pads": pads[:rank] + ends}
    return ag


# Operand positions that carry shape or bound metadata rather than tensor data.
# Earlier passes lift these into op attributes and into the output shape, so by
# this point they describe the plan the compiler already emitted. Leaving them
# on the op makes the emitter bind an index vector as the operator's weight.
_METADATA_INPUTS: dict[str, int] = {
    "Clip": 1,
    "ReduceMean": 1,
    "ReduceMax": 1,
    "ReduceMin": 1,
    "ReduceSum": 1,
    "CumSum": 1,
    "ReduceAll": 1,
    "Reshape": 1,
    "Squeeze": 1,
    "Unsqueeze": 1,
}


def _lift_pad_operands(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Move a constant Pad's widths into an attribute and keep its fill as
    the operator's weight.

    The fill is stored in the output's encoding, quantized to it for int8. A
    fill of zero is dropped: the runtime pads with 0, or the output's zero
    point, by default.
    """
    for op in ag.ops:
        if op.op_type != "Pad" or len(op.inputs) < 2 or len(op.outputs) != 1:
            continue
        amounts = ag.weight_data.get(op.inputs[1])
        fill_name = op.inputs[2] if len(op.inputs) > 2 and op.inputs[2] else None
        fill = ag.weight_data.get(fill_name) if fill_name else None
        if amounts is None or (fill_name and fill is None):
            continue
        op.attrs["pad_amounts"] = [int(v) for v in amounts.reshape(-1)]
        op.inputs = [op.inputs[0]]
        value = float(fill.reshape(-1)[0]) if fill is not None and fill.size else 0.0
        if value == 0.0:
            continue
        output = ag.tensors.get(op.outputs[0])
        name = _fresh_name(ag, f"{op.name}_fill")
        if output is not None and output.quant is not None:
            scale = float(output.quant.scale[0])
            zero_point = int(output.quant.zero_point[0])
            data = np.array([np.clip(np.round(value / scale) + zero_point, -128, 127)], np.int8)
            dtype = 3
        else:
            data = np.array([value], np.float32)
            dtype = 1
        ag.weight_data[name] = data
        ag.tensors[name] = TensorInfo(name=name, shape=(1,), dtype=dtype, is_constant=True)
        op.inputs.append(name)
    return ag


def _strip_metadata_inputs(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Drop shape and bound operands the runtime never reads.

    A model can supply a Reshape target shape, Resize scales or Clip bounds
    either as a computed subgraph or as a plain initializer. Both forms mean
    the same thing to the compiler, so trim them to the data operands and let
    the resolved shapes and attributes carry the information.
    """
    for op in ag.ops:
        first = _METADATA_INPUTS.get(op.op_type)
        if first is None or len(op.inputs) <= first:
            continue
        for name in op.inputs[first:]:
            if name and name not in ag.weight_data:
                # A computed operand that no pass resolved is not metadata the
                # compiler can drop; leave the operator intact so validation
                # reports it.
                break
        else:
            del op.inputs[first:]
    return ag


def _order_broadcast_operands(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Put the full-shape operand first in an Add or Mul that broadcasts.

    A squeeze-and-excitation block multiplies a feature map by one value per
    channel, and exporters write either operand first. The kernels take the
    first operand's shape as the result's and repeat the second, so the
    commutative ops are reordered here rather than taught both orders.
    """
    for op in ag.ops:
        if op.op_type not in ("Add", "Mul") or len(op.inputs) != 2:
            continue
        if len(op.outputs) != 1 or op.outputs[0] not in ag.tensors:
            continue
        first, second = (ag.tensors.get(name) for name in op.inputs)
        if first is None or second is None:
            continue
        result = tuple(ag.tensors[op.outputs[0]].shape)
        if tuple(first.shape) != result and tuple(second.shape) == result:
            op.inputs = [op.inputs[1], op.inputs[0]]
    return ag


def _normalize_arg_outputs(ag: AnalyzedGraph) -> AnalyzedGraph:
    # An index cast to int32 is already stored as the runtime writes it.
    for cast in list(ag.ops):
        if (cast.op_type != "Cast" or int(cast.attrs.get("to", 0)) != 6 or len(cast.inputs) != 1
                or len(cast.outputs) != 1):
            continue
        index = cast.inputs[0]
        producer = next((op for op in ag.ops if index in op.outputs), None)
        if (producer is None or producer.op_type not in {"ArgMax", "ArgMin"} or index in ag.model_outputs
                or any(index in op.inputs for op in ag.ops if op is not cast)):
            continue
        producer.outputs = [cast.outputs[0]]
        ag.tensors[cast.outputs[0]].shape = ag.tensors[index].shape
        ag.tensors.pop(index, None)
        ag.ops.remove(cast)
    for op in ag.ops:
        if op.op_type not in {"ArgMax", "ArgMin"}:
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError(f"{op.op_type} requires one input and one output")
        name = op.outputs[0]
        source, output = ag.tensors[op.inputs[0]], ag.tensors[name]
        rank = len(source.shape)
        axis = int(op.attrs.get("axis", 0))
        if rank == 0 or axis not in range(-rank, rank):
            raise ValueError(f"{op.op_type} axis is outside the input rank")
        axis %= rank
        if not 0 < source.shape[axis] <= 2**31 - 1:
            raise ValueError(f"{op.op_type} axis extent cannot be indexed by int32")
        if int(op.attrs.get("select_last_index", 0)) != 0:
            raise ValueError(f"{op.op_type} select_last_index=1 is unsupported")
        if int(op.attrs.get("keepdims", 1)) not in {0, 1}:
            raise ValueError(f"{op.op_type} keepdims must be 0 or 1")
        if output.dtype not in {6, 7} or output.quant is not None:
            raise ValueError(f"{op.op_type} requires an unquantized integer output")
        output.dtype = 6
        op.attrs["axes"] = [axis]
    return ag


def _reduction_axes(ag: AnalyzedGraph, op: OpNode, prefer: str) -> list | None:
    """The axes an operator reduces: its attribute (opset < 18) and its constant
    second operand (opset >= 18). ``prefer`` is the one that wins when both are
    present, ``"attribute"`` or ``"operand"``."""
    attribute = op.attrs.get("axes")
    operand = None
    if len(op.inputs) >= 2 and op.inputs[1] in ag.weight_data:
        operand = ag.weight_data[op.inputs[1]].reshape(-1).tolist()
    first, second = (attribute, operand) if prefer == "attribute" else (operand, attribute)
    return second if first is None else first


def _normalize_reduction_axes(ag: AnalyzedGraph) -> AnalyzedGraph:
    for op in ag.ops:
        if op.op_type not in {"ReduceMax", "ReduceMin", "ReduceSum", "CumSum", "ReduceAll"}:
            continue
        source = ag.tensors.get(op.inputs[0]) if op.inputs else None
        rank = len(source.shape) if source is not None else 0
        axes = _reduction_axes(ag, op, prefer="operand")
        if axes is None or not axes or rank == 0 or any(int(a) < -rank or int(a) >= rank for a in axes):
            raise ValueError(f"{op.op_type} requires explicit constant axes within the input rank")
        op.attrs["axes"] = sorted({int(a) % rank for a in axes})
        if op.op_type == "ReduceMax" and ag.is_quantized and op.outputs:
            output = ag.tensors.get(op.outputs[0])
            if (source.quant is None or output is None or output.quant is None
                    or source.quant.scale.size != 1 or source.quant.zero_point.size != 1
                    or not source.quant.same_scale_and_zero_point(output.quant)):
                raise ValueError("ReduceMax requires identical input and output quantization")
    return ag


def _resolve_resize_conventions(ag: AnalyzedGraph) -> AnalyzedGraph:
    """State what a Resize samples, rather than leaving it to a default.

    Resize changed shape across opsets. Up to opset 10 it takes scales as its
    second input and always reads the source at ``out / scale``, which later
    versions call the asymmetric convention; there is no attribute saying so.
    From opset 11 the attributes exist and default to half-pixel coordinates
    and a round-half-down nearest rule. Reading the opset-11 default off an
    opset-10 graph moves every sample by half an output pixel, so the
    effective convention is written onto the operator here and every later
    pass reads it rather than guessing.
    """
    for op in ag.ops:
        if op.op_type != "Resize":
            continue
        if ag.opset and ag.opset < 11:
            op.attrs["coordinate_transformation_mode"] = "asymmetric"
            op.attrs.setdefault("nearest_mode", "floor")
        else:
            op.attrs.setdefault(
                "coordinate_transformation_mode", "half_pixel")
            op.attrs.setdefault("nearest_mode", "round_prefer_floor")
    return ag


def _extract_resize_scales(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Keep explicit Resize scales and encode the sampling convention."""
    for op in ag.ops:
        if op.op_type != "Resize":
            continue
        source = ag.tensors.get(op.inputs[0])
        result = ag.tensors.get(op.outputs[0])
        mode = op.attrs.get("mode", "nearest")
        coordinate = op.attrs.get("coordinate_transformation_mode", "half_pixel")
        scale_index = 1 if ag.opset and ag.opset < 11 else 2
        if len(op.inputs) > scale_index and op.inputs[scale_index]:
            name = op.inputs[scale_index]
            scales = ag.weight_data.get(name)
            if scales is None:
                op.attrs["resize_error"] = "Resize scales must be constant"
            elif scales.size:
                values = scales.flatten()
                if (values.size != 4 or not np.all(np.isfinite(values))
                        or np.any(values <= 0) or np.any(values[:2] != 1)):
                    op.attrs["resize_error"] = "Resize scales must preserve N/C and have positive finite H/W"
                else:
                    op.attrs["resize_scales"] = [float(v) for v in values[2:]]
        if len(op.inputs) > 3 and op.inputs[3] and op.inputs[3] not in ag.weight_data:
            op.attrs["resize_error"] = "Resize sizes must be constant"
        convention = {"asymmetric": 1, "align_corners": 2}.get(coordinate, 0)
        if mode == "linear":
            op.op_type = "ResizeLinear"
        elif coordinate in {"half_pixel", "tf_half_pixel_for_nn"}:
            convention = 3
        op.attrs["kernel_shape"] = [convention]
        strides = [0, 0]
        if source and result and len(source.shape) == len(result.shape) == 4:
            for axis in range(2):
                i, o = int(source.shape[axis + 2]), int(result.shape[axis + 2])
                if i > 0 and o % i == 0 and 1 <= o // i <= 255:
                    strides[axis] = o // i
            explicit = op.attrs.get("resize_scales")
            if explicit and all(s == e for s, e in zip(strides, explicit)):
                del op.attrs["resize_scales"]
        if convention >= 2 or "resize_scales" in op.attrs:
            strides = [0, 0]
        op.attrs["strides"] = strides
        constants = op.inputs[1:]
        op.inputs = [op.inputs[0]]
        for name in constants:
            if name and not any(name in other.inputs for other in ag.ops):
                ag.weight_data.pop(name, None)
                if name in ag.tensors and ag.tensors[name].is_constant:
                    del ag.tensors[name]
    return ag


_INT32_OPERATORS = {"Add", "Sub", "Mul", "Cast", "Equal", "Less", "LessOrEqual", "Greater",
                    "GreaterOrEqual"}


def _mark_untileable_broadcasts(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Mark a binary operator a tile cannot carry.

    A tile offsets a dense operand by its rows and reads a repeating constant
    whole. An operand read by output coordinate has neither, and a tensor
    operand that is not dense is banded only in one form: one value per
    channel as the second operand.
    """
    for op in ag.ops:
        # int32 arithmetic, comparisons and casts run untiled.
        if any(name in ag.tensors and ag.tensors[name].dtype == 6
               for name in (*op.inputs, *op.outputs)) and op.op_type in _INT32_OPERATORS:
            op.attrs["broadcast_untileable"] = True
            continue
        operands = stored_operands(ag, op)
        if operands is None:
            continue
        untileable = False
        for position, (name, constant, stored, how) in enumerate(operands):
            if how == GENERAL:
                untileable = True
            elif how == DENSE and constant and ag.weight_data[name].size > 1:
                # One constant value per element carries no row offset.
                untileable = True
            elif how == PERIODIC and not constant and not (
                    position == 1 and all(dim == 1 for dim in stored[:-1])):
                untileable = True
        op.attrs["broadcast_untileable"] = untileable
    return ag


def _validate_transposes(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Validate Transpose permutations retained in the deployment plan."""
    for op in ag.ops:
        if op.op_type != "Transpose":
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError(
                f"Transpose '{op.name}' must have exactly one input and one output"
            )
        input_info = ag.tensors.get(op.inputs[0])
        output_info = ag.tensors.get(op.outputs[0])
        if input_info is None or output_info is None:
            raise ValueError(f"Transpose '{op.name}' has unknown tensor metadata")
        rank = len(input_info.shape)
        perm = op.attrs.get("perm", list(reversed(range(rank))))
        perm = [int(axis) for axis in perm]
        if (rank != len(output_info.shape) or len(perm) != rank or
                sorted(perm) != list(range(rank))):
            raise ValueError(
                f"Transpose '{op.name}' has invalid perm={perm} for rank {rank}"
            )
        expected_shape = tuple(input_info.shape[axis] for axis in perm)
        if tuple(output_info.shape) != expected_shape:
            raise ValueError(
                f"Transpose '{op.name}' perm={perm} expects output shape "
                f"{expected_shape}, got {output_info.shape}"
            )
        op.attrs["perm"] = perm
    return ag
