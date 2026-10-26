"""Data movement operators: their indices and bounds as the metadata the runtime reads."""

import numpy as np

from tigris.graph.ir import AnalyzedGraph, Layout, OpNode, TensorInfo
from tigris.loaders.onnx.normalize.common import _fresh_name, _Uses


_MOVEMENT_OPS = frozenset({
    "Gather", "GatherND", "StridedSlice", "MirrorPad", "ReverseV2",
    "EmbeddingLookup", "DynamicUpdateSlice",
})
_PREFIXED_MOVEMENT_OPS = frozenset(f"tigris::{kind}" for kind in _MOVEMENT_OPS)
_GATHER_OPS = frozenset({"Gather", "GatherND", "EmbeddingLookup"})


def _normalize_movement(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Check each data movement operator and state its geometry in the
    ``movement`` attribute.

    ONNX Slice becomes StridedSlice, a reflect or symmetric Pad becomes
    MirrorPad, and a ``tigris::`` movement operator loses its prefix. Each
    family's function returns the output shape its operands imply and the
    metadata; an operator the runtime cannot execute as stated raises
    ValueError.
    """
    for op in ag.ops:
        if op.op_type in _PREFIXED_MOVEMENT_OPS:
            op.op_type = op.op_type.split("::", 1)[1]
        mode = op.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("ascii")
        if op.op_type == "Pad" and mode in {"reflect", "symmetric"}:
            op.op_type = "MirrorPad"
        onnx_slice = op.op_type == "Slice"
        if onnx_slice:
            op.op_type = "StridedSlice"
        if op.op_type not in _MOVEMENT_OPS:
            continue
        if not op.inputs or len(op.outputs) != 1:
            raise ValueError(f"{op.op_type} requires data and one output")
        source = ag.tensors[op.inputs[0]]
        shape = source.shape
        rank = len(shape)
        if (source.is_constant and op.op_type not in _GATHER_OPS) or not 1 <= rank <= 6 or any(d <= 0 for d in shape):
            raise ValueError(f"{op.op_type} requires a nonempty data tensor of rank 1 through 6")
        output = ag.tensors[op.outputs[0]]
        if source.dtype != output.dtype or source.dtype not in {1, 3}:
            raise ValueError(f"{op.op_type} requires matching float32 or int8 data")
        if source.dtype == 3 and (source.quant is None or output.quant is None
                or source.quant.scale.size != 1 or source.quant.zero_point.size != 1
                or not source.quant.same_scale_and_zero_point(output.quant)):
            raise ValueError(f"{op.op_type} requires identical per-tensor quantization")
        kind = op.op_type
        if source.is_constant:
            source.layout = Layout.LINEAR
            op.attrs["table_shape"] = list(shape)
        if kind in _GATHER_OPS:
            target, metadata = _gather_movement(ag, op, shape)
        elif kind == "StridedSlice":
            target, metadata = _strided_slice_movement(ag, op, shape, onnx_slice)
        elif kind == "MirrorPad":
            target, metadata = _mirror_pad_movement(ag, op, shape, mode)
        elif kind == "ReverseV2":
            target, metadata = _reverse_movement(ag, op, shape)
        else:
            target, metadata = _dynamic_update_slice_movement(ag, op, source)
        if tuple(target) != output.shape or len(target) > 6 or any(d <= 0 for d in target):
            raise ValueError(f"{kind} output shape does not match its indices or bounds")
        if any(not -(2**31) <= value < 2**31 for value in metadata):
            raise ValueError(f"{kind} metadata exceeds int32")
        op.attrs["movement"] = metadata
    return ag


def _movement_constant(ag: AnalyzedGraph, op: OpNode, position: int) -> np.ndarray:
    """The constant int32 or int64 operand at ``position``."""
    if len(op.inputs) <= position or op.inputs[position] not in ag.weight_data:
        raise ValueError(f"{op.op_type} requires constant indices or bounds")
    value = ag.weight_data[op.inputs[position]]
    if value.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError(f"{op.op_type} indices or bounds must be int32 or int64")
    return value


def _index_tensor(ag: AnalyzedGraph, op: OpNode, position: int) -> TensorInfo:
    """The index operand at ``position``; a computed one is held as int32 in the
    model's own order."""
    if len(op.inputs) <= position or op.inputs[position] not in ag.tensors:
        raise ValueError(f"{op.op_type} requires an index tensor")
    name = op.inputs[position]
    info = ag.tensors[name]
    if info.dtype not in {6, 7} or info.quant is not None:
        raise ValueError(f"{op.op_type} indices must be unquantized int32 or int64")
    if not info.is_constant:
        if info.dtype == 7 and name not in ag.model_inputs:
            raise ValueError(f"{op.op_type} computed indices must be int32")
        info.dtype = 6
        info.layout = Layout.LINEAR
    return info


def _gather_movement(ag: AnalyzedGraph, op: OpNode, shape: tuple) -> tuple[tuple, list]:
    """Gather, GatherND and EmbeddingLookup. Constant indices are checked
    against the data and stored as an int32 constant, negative ones wrapped."""
    kind = op.op_type
    rank = len(shape)
    if len(op.inputs) != 2:
        raise ValueError(f"{kind} requires data and indices")
    info = _index_tensor(ag, op, 1)
    indices = _movement_constant(ag, op, 1).copy() if info.is_constant else None
    index_shape = info.shape
    index_rank = len(index_shape)
    if index_rank > 6 or any(d <= 0 for d in index_shape):
        raise ValueError(f"{kind} requires nonempty indices of rank at most 6")
    if kind == "GatherND":
        if int(op.attrs.get("batch_dims", 0)) != 0 or index_rank == 0:
            raise ValueError("GatherND requires batch_dims=0 and nonscalar indices")
        width = index_shape[-1]
        if not 1 <= width <= min(rank, 5):
            raise ValueError("GatherND index depth must fit the input rank and be at most 5")
        if indices is not None:
            for a in range(width):
                component = indices[..., a]
                if np.any(component < -shape[a]) or np.any(component >= shape[a]):
                    raise ValueError("GatherND index is outside its dimension")
                indices[..., a] = np.where(component < 0, component + shape[a], component)
        target = (*index_shape[:-1], *shape[width:])
        metadata = [index_rank, *index_shape]
    else:
        axis = 0 if kind == "EmbeddingLookup" else int(op.attrs.get("axis", 0))
        batch = int(op.attrs.get("batch_dims", 0))
        if batch < 0:
            batch += index_rank
        if axis not in range(-rank, rank) or not 0 <= batch <= min(axis % rank, index_rank):
            raise ValueError(f"{kind} axis or batch_dims is invalid")
        axis %= rank
        if kind == "EmbeddingLookup" and (rank < 2 or index_rank != 1 or batch != 0):
            raise ValueError("EmbeddingLookup requires a rank-2-or-higher value and vector indices")
        if tuple(index_shape[:batch]) != shape[:batch]:
            raise ValueError("Gather batch dimensions must match")
        if indices is not None:
            if np.any(indices < -shape[axis]) or np.any(indices >= shape[axis]):
                raise ValueError(f"{kind} index is outside its dimension")
            indices = np.where(indices < 0, indices + shape[axis], indices)
        target = (*shape[:axis], *index_shape[batch:], *shape[axis + 1:])
        metadata = [axis, batch, index_rank, *index_shape]
        op.attrs["axis"] = axis
    if indices is not None:
        name = _fresh_name(ag, f"{op.name}_indices_i32")
        ag.weight_data[name] = np.asarray(indices, dtype=np.int32)
        ag.tensors[name] = TensorInfo(name=name, shape=index_shape, dtype=6, is_constant=True,
                                      layout=Layout.LINEAR)
        op.inputs = [op.inputs[0], name]
    return target, metadata


def _strided_slice_movement(
    ag: AnalyzedGraph, op: OpNode, shape: tuple, onnx_slice: bool,
) -> tuple[tuple, list]:
    """StridedSlice, and ONNX Slice with its optional axes and steps; the bounds
    become one (start, end, step) per axis, then the shrink mask if any."""
    rank = len(shape)
    if rank > 4:
        raise ValueError("StridedSlice supports rank at most 4")
    starts = _movement_constant(ag, op, 1).reshape(-1).tolist()
    ends = _movement_constant(ag, op, 2).reshape(-1).tolist()
    shrink = 0
    begin_mask = end_mask = 0
    if onnx_slice:
        axes = (_movement_constant(ag, op, 3).reshape(-1).tolist() if len(op.inputs) > 3 and op.inputs[3]
                else list(range(len(starts))))
        steps = (_movement_constant(ag, op, 4).reshape(-1).tolist() if len(op.inputs) > 4 and op.inputs[4]
                 else [1] * len(starts))
    else:
        axes = list(range(rank))
        steps = _movement_constant(ag, op, 3).reshape(-1).tolist()
        begin_mask = int(op.attrs.get("begin_mask", 0))
        end_mask = int(op.attrs.get("end_mask", 0))
        shrink = int(op.attrs.get("shrink_axis_mask", 0))
        if (any(mask < 0 or mask >= 1 << rank for mask in (begin_mask, end_mask, shrink))
                or int(op.attrs.get("ellipsis_mask", 0)) or int(op.attrs.get("new_axis_mask", 0))):
            raise ValueError("StridedSlice masks must fit the input rank; new axes and ellipses are unsupported")
    if not len(starts) == len(ends) == len(axes) == len(steps) or len(set(a % rank for a in axes)) != len(axes):
        raise ValueError("StridedSlice bounds must have equal lengths and unique axes")
    slices = [slice(None)] * rank
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if axis not in range(-rank, rank) or step == 0:
            raise ValueError("StridedSlice axis or step is invalid")
        a = axis % rank
        slices[a] = slice(None if begin_mask & (1 << a) else start,
                          None if end_mask & (1 << a) else end, step)
    normalized = [part.indices(dim) for part, dim in zip(slices, shape)]
    for a, (start, end, step) in enumerate(normalized):
        if shrink & (1 << a):
            if step <= 0 or not 0 <= start < shape[a]:
                raise ValueError("StridedSlice shrinking requires a valid start and positive stride")
            normalized[a] = (start, start + 1, 1)
    target = tuple(len(range(*part)) for a, part in enumerate(normalized) if not shrink & (1 << a))
    metadata = [value for part in normalized for value in part]
    if shrink:
        metadata.append(shrink)
    op.inputs = op.inputs[:1]
    return target, metadata


def _mirror_pad_movement(ag: AnalyzedGraph, op: OpNode, shape: tuple, mode: str) -> tuple[tuple, list]:
    """MirrorPad: whether it is symmetric, then the two widths of each axis."""
    rank = len(shape)
    pads = _movement_constant(ag, op, 1)
    if pads.shape == (rank, 2):
        pairs = pads.tolist()
    elif pads.shape == (2 * rank,):
        pairs = list(zip(pads[:rank].tolist(), pads[rank:].tolist()))
    else:
        raise ValueError("MirrorPad requires two padding widths per input dimension")
    symmetric = mode == "symmetric"
    if mode not in {"reflect", "symmetric"}:
        raise ValueError("MirrorPad mode must be reflect or symmetric")
    for dim, pair in zip(shape, pairs):
        if any(p < 0 or p > dim - (0 if symmetric else 1) for p in pair):
            raise ValueError("MirrorPad width exceeds the reflection domain")
    target = tuple(d + sum(pair) for d, pair in zip(shape, pairs))
    metadata = [int(symmetric), *(p for pair in pairs for p in pair)]
    op.inputs = op.inputs[:1]
    return target, metadata


def _reverse_movement(ag: AnalyzedGraph, op: OpNode, shape: tuple) -> tuple[tuple, list]:
    """ReverseV2: the reversed axes as one bit mask."""
    rank = len(shape)
    axes = _movement_constant(ag, op, 1).reshape(-1).tolist()
    if any(a not in range(-rank, rank) for a in axes) or len(set(a % rank for a in axes)) != len(axes):
        raise ValueError("ReverseV2 requires unique axes within the input rank")
    normalized_axes = sorted(a % rank for a in axes)
    if not normalized_axes or normalized_axes != list(range(normalized_axes[0], normalized_axes[-1] + 1)):
        raise ValueError("ReverseV2 requires a nonempty contiguous set of axes")
    metadata = [sum(1 << a for a in normalized_axes)]
    op.inputs = op.inputs[:1]
    return shape, metadata


def _dynamic_update_slice_movement(
    ag: AnalyzedGraph, op: OpNode, source: TensorInfo,
) -> tuple[tuple, list]:
    """DynamicUpdateSlice: the update's shape, after the clamped starts when
    they are constant."""
    shape = source.shape
    rank = len(shape)
    if len(op.inputs) != 3:
        raise ValueError("DynamicUpdateSlice requires data, update and starts")
    info = _index_tensor(ag, op, 2)
    starts = _movement_constant(ag, op, 2).reshape(-1).tolist() if info.is_constant else None
    update = ag.tensors[op.inputs[1]]
    if info.shape != (rank,) or len(update.shape) != rank:
        raise ValueError("DynamicUpdateSlice requires a data update and one start per input dimension")
    if update.dtype != source.dtype or any(u > d or u <= 0 for u, d in zip(update.shape, shape)):
        raise ValueError("DynamicUpdateSlice update must fit the input shape and dtype")
    if source.dtype == 3 and (update.quant is None
            or not source.quant.same_scale_and_zero_point(update.quant)):
        raise ValueError("DynamicUpdateSlice requires identical update quantization")
    metadata = list(update.shape)
    if starts is not None:
        metadata = [min(max(s, 0), d - u) for s, d, u in zip(starts, shape, update.shape)] + metadata
        op.inputs = op.inputs[:2]
    return shape, metadata


def _narrow_index_casts(ag: AnalyzedGraph) -> AnalyzedGraph:
    """An int64 model input cast to int32 that only feeds gather indices,
    through movement, enters as int32 and the Cast becomes an Identity."""
    uses = _Uses(ag)
    gathers = {"Gather", "GatherND", "EmbeddingLookup", "tigris::Gather", "tigris::GatherND"}
    for op in ag.ops:
        if (op.op_type != "Cast" or int(op.attrs.get("to", 0)) != 6
                or len(op.inputs) != 1 or op.inputs[0] not in ag.model_inputs
                or ag.tensors[op.inputs[0]].dtype != 7):
            continue
        pending, seen, valid = list(op.outputs), set(), True
        while pending and valid:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            readers = uses.readers(name)
            if name in ag.model_outputs or not readers:
                valid = False
                break
            for index, _ in readers:
                reader = ag.ops[index]
                if reader.op_type in gathers and reader.inputs[1] == name:
                    continue
                if reader.op_type not in {"Identity", "Reshape", "Transpose"} or reader.inputs[0] != name:
                    valid = False
                    break
                pending.extend(reader.outputs)
        if valid:
            ag.tensors[op.inputs[0]].dtype = 6
            op.op_type = "Identity"
            op.attrs = {}
    return ag
