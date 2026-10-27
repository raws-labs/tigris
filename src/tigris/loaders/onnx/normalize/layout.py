"""Tensor layouts, and the rewrites that depend on the order the runtime stores."""

from dataclasses import replace

import numpy as np

from tigris.graph.ir import (
    AnalyzedGraph,
    Layout,
    OpNode,
    TensorInfo,
    serialized_axis_map,
    serialized_shape,
)
from tigris.loaders.onnx.normalize.common import _fresh_name, _Uses
from tigris.loaders.onnx.normalize.movement import _MOVEMENT_OPS
from tigris.loaders.onnx.normalize.relabels import _transpose_product_weights


# Operators whose activation operands are spatial pictures: the runtime holds
# them channels-last and the kernels index them that way.
_SPATIAL_LAYOUT_OPS = frozenset({
    "Conv",
    "DepthwiseConv",
    "Conv1D",
    "ConvTranspose",
    "MaxPool",
    "AveragePool",
    "L2Pool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Resize",
})


# Operators whose operands already list their axes in storage order. A matrix
# product reduces over its last axis, so permuting it channels-last would move
# the reduction axis and compute something else.
_LINEAR_LAYOUT_OPS = frozenset({
    "MatMul",
    "Lstm",
})


def _trailing_axis_required_layout(
    ag: AnalyzedGraph, op: OpNode
) -> Layout | None:
    """The layout that puts Softmax's axis where the kernel reduces.

    The kernel normalizes along the final stored dimension, so which ONNX axis
    that is depends on how the tensor is held. Spatial storage puts the channel
    axis last, which is ONNX axis 1; the model's own order puts the last ONNX
    axis last. A Softmax over either one is expressible, and the layout is what
    says which. Anything else has no layout that helps and stays unsupported.
    """
    if not op.inputs:
        return None
    info = ag.tensors.get(op.inputs[0])
    if info is None or not info.shape:
        return None
    rank = len(info.shape)
    if rank < 3:
        return None
    axis = int(op.attrs.get("axis", -1))
    axis = axis + rank if axis < 0 else axis
    if axis == rank - 1:
        return Layout.LINEAR
    if axis == 1:
        return Layout.SPATIAL
    return None


def _flattened_into_a_product(ag: AnalyzedGraph, op: OpNode) -> bool:
    """Whether a matrix product downstream absorbs the difference in order.

    A flatten whose one reader is a product over a constant weight needs no
    conversion: the emitter permutes the weight's columns instead, which
    costs nothing at inference. Mirrors _build_fc_layout_permutations in the
    binary writer, which performs that permutation; a flatten this does not
    describe has to be converted rather than left to a compensation nobody
    applies.
    """
    if not op.outputs or not op.inputs:
        return False
    source = ag.tensors.get(op.inputs[0])
    vector = ag.tensors.get(op.outputs[0])
    if source is None or vector is None or len(vector.shape) != 2:
        return False
    # The emitter's column permutation assumes the batch axis stays the row
    # axis; a reshape that also regroups rows is not a flatten it can absorb.
    if len(source.shape) not in (3, 4) or vector.shape[0] != source.shape[0]:
        return False
    # A model output is read in the model's order, which no weight absorbs.
    if op.outputs[0] in ag.model_outputs:
        return False
    readers = [
        other for other in ag.ops
        if op.outputs[0] in other.inputs and other is not op
    ]
    if len(readers) != 1:
        return False
    reader = readers[0]
    if reader.op_type != "Gemm" or len(reader.inputs) < 2:
        return False
    if reader.inputs[0] != op.outputs[0]:
        return False
    weight = ag.weight_data.get(reader.inputs[1])
    return weight is not None and weight.ndim == 2


def _states_its_own_order(info) -> bool:
    """Whether this tensor lists its axes in the order the runtime stores.

    A tensor that states its own order, and one of rank 2 or less, always
    does. So does a feature map whose storage permutation only moves axes of
    extent one: a single-channel image is held row by row either way, and the
    permutation that formally moves its channel axis moves nothing. Reading
    that case as a reordering costs a copy for no reason, and refusing it
    costs an operator the runtime could have executed.
    """
    if info.layout is Layout.LINEAR or len(info.shape) <= 2:
        return True
    rank = len(info.shape)
    positions = serialized_axis_map(rank, info.layout)
    in_storage_order = sorted(range(rank), key=lambda axis: positions[axis])
    carried = [axis for axis in in_storage_order if info.shape[axis] > 1]
    return carried == sorted(carried)


def _reshape_keeps_its_order(ag: AnalyzedGraph, op: OpNode) -> bool:
    """Whether a regrouping is the same bytes in the order the runtime holds.

    A reshape moves no element: it renames axes over one sequence, and the
    kernel copies straight through. That states the model's regrouping when
    both sides list their axes in storage order, and when two feature maps
    regroup only the axes between the first and the last, since the runtime
    holds those contiguously. A regrouping that reaches across the channel
    axis of a tensor held channels-last does not, and copying it through
    would read the elements in another order. That is what a vision
    transformer's unfold does, and holding the operand in the model's own
    order first is what makes it expressible.

    Same question, same answer as pure_reinterpretations in the lifetime
    analysis, which decides when such a reshape can share its input's buffer.
    """
    if not op.inputs or not op.outputs:
        return True
    src = ag.tensors.get(op.inputs[0])
    dst = ag.tensors.get(op.outputs[0])
    if src is None or dst is None:
        return True
    src_own = _states_its_own_order(src)
    dst_own = _states_its_own_order(dst)
    if src_own and dst_own:
        return True
    if src_own or dst_own:
        return False
    first = serialized_shape(tuple(src.shape), src.layout)
    second = serialized_shape(tuple(dst.shape), dst.layout)
    return (bool(first) and bool(second)
            and first[0] == second[0] and first[-1] == second[-1])


def _transposes_to_storage_order(ag: AnalyzedGraph, op: OpNode) -> bool:
    """A rank-3 or rank-4 Transpose whose permutation is its spatial operand's
    channels-last storage order."""
    if op.op_type != "Transpose" or len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    source = ag.tensors.get(op.inputs[0])
    if source is None or source.is_constant or source.layout is not Layout.SPATIAL:
        return False
    rank = len(source.shape)
    if rank not in (3, 4):
        return False
    stored = serialized_axis_map(rank, Layout.SPATIAL)
    order = [0] * rank
    for raw_axis, stored_axis in enumerate(stored):
        order[stored_axis] = raw_axis
    return [int(axis) for axis in op.attrs.get("perm", [])] == order


def _required_layout(ag: AnalyzedGraph, op: OpNode) -> Layout | None:
    """The layout an operator needs, or None when it works in either."""
    if op.op_type in {"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Where"}:
        ranks = {len(ag.tensors[n].shape) for n in op.inputs + op.outputs
                 if n in ag.tensors and not ag.tensors[n].is_constant}
        if len(ranks) > 1:
            return Layout.LINEAR
    if op.op_type in _MOVEMENT_OPS:
        return Layout.LINEAR
    if op.op_type in ("If", "While"):
        # A subgraph holds its inputs and outputs as a graph boundary does,
        # and the operands are copied in byte for byte.
        return Layout.SPATIAL
    if op.op_type in _SPATIAL_LAYOUT_OPS:
        return Layout.SPATIAL
    if op.op_type in {"ReduceMax", "ReduceMin", "ReduceSum", "CumSum", "ArgMax", "ArgMin", "ReduceAll"}:
        return Layout.LINEAR
    if op.op_type in _LINEAR_LAYOUT_OPS:
        return Layout.LINEAR
    if op.op_type in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
        # A regrouping the runtime's own order does not state is expressible
        # once the operand is held the way the model states it, unless a
        # matrix product downstream already compensates for the difference.
        # Squeeze and Unsqueeze are the same question: they become a Reshape
        # further down the pipeline, and dropping the channel axis of a
        # feature map regroups across it exactly as a Reshape would.
        if _reshape_keeps_its_order(ag, op) or _flattened_into_a_product(ag, op):
            return None
        return Layout.LINEAR
    if op.op_type in ("Softmax", "LogSoftmax", "L2Normalization", "LayerNormalization"):
        # Both reduce along one axis and the kernel takes the final stored
        # one, so the layout is what says which ONNX axis that is.
        return _trailing_axis_required_layout(ag, op)
    return None


def _operand_layouts(ag: AnalyzedGraph, op: OpNode) -> list[Layout]:
    """The layouts of the operands whose layout can differ from each other.

    Constants carry no layout of their own and rank 2 and below cannot
    disagree, so neither constrains anything.
    """
    layouts = []
    for name in op.inputs:
        if not name:
            continue
        info = ag.tensors.get(name)
        if info is None or info.is_constant or len(info.shape) < 3:
            continue
        layouts.append(info.layout)
    return layouts


def _agreed_layout(ag: AnalyzedGraph, op: OpNode) -> Layout | None:
    """The layout an operator's operands have to share, when they must.

    An operator that works in either layout still needs its operands to agree
    with each other: an elementwise Add reads both at the same offset, so two
    operands stored in different axis orders add unrelated elements. The
    shapes reveal it only when they differ, so a square block passes the
    shape check and adds wrong data.

    Where they already agree there is nothing to do. Where they do not, the
    majority wins and a tie goes to the model's own order, because a
    disagreement arises only when a matrix product or a trailing-axis
    normalization produced one of the operands, and those are exactly the
    operators that will read the result.
    """
    layouts = _operand_layouts(ag, op)
    if len(set(layouts)) < 2:
        return None
    spatial = layouts.count(Layout.SPATIAL)
    linear = layouts.count(Layout.LINEAR)
    return Layout.SPATIAL if spatial > linear else Layout.LINEAR


def _align_state_layouts(ag: AnalyzedGraph) -> AnalyzedGraph:
    """A state leaves in the layout it entered in. Below rank 3 the two layouts
    store the same bytes, so the output takes the input's."""
    for port in ag.state_ports:
        if port.output is None:
            continue
        entered = ag.tensors[ag.model_inputs[port.input]]
        left = ag.tensors[ag.model_outputs[port.output]]
        if len(entered.shape) < 3:
            left.layout = entered.layout
    return ag


def _assign_tensor_layouts(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Give every tensor a layout and convert where producer and consumer differ.

    Layout only has consequences at rank 3 and rank 4, where ONNX axis order
    and storage order disagree. An operator either requires one of the two or
    works in whichever it is handed, and a tensor carries whatever its producer
    wrote. Where a consumer needs the other one, an explicit Transpose converts
    it: the emitter turns an identity permutation across differing layouts into
    the physical permutation, so the conversion costs no new machinery.

    A terminal Transpose stays linear: its output is a model-output boundary
    that keeps the observable ONNX shape and order.
    """
    converted: dict[tuple[str, Layout], str] = {}
    rewritten: list[OpNode] = []
    counter = 0
    uses = _Uses(ag)

    for op in ag.ops:
        required = _required_layout(ag, op)
        # An operator with no requirement of its own still needs its operands
        # to agree with each other. _agreed_layout is None when they already
        # do, which is every operator with one activation operand.
        enforced = required if required is not None else _agreed_layout(ag, op)
        if enforced is not None:
            for position, name in enumerate(op.inputs):
                if not name:
                    continue
                info = ag.tensors.get(name)
                if info is None or info.is_constant:
                    continue
                # Rank 2 and below cannot disagree, so never pay for a copy.
                if len(info.shape) < 3 or info.layout is enforced:
                    continue

                key = (name, enforced)
                target = converted.get(key)
                if target is None:
                    counter += 1
                    target = _fresh_name(ag, f"{name}_to_{enforced.value}_{counter}")
                    ag.tensors[target] = TensorInfo(
                        name=target,
                        shape=info.shape,
                        dtype=info.dtype,
                        quant=info.quant,
                        layout=enforced,
                    )
                    rewritten.append(OpNode(
                        name=f"layout_{counter}",
                        op_type="Transpose",
                        inputs=[name],
                        outputs=[target],
                        attrs={"perm": list(range(len(info.shape)))},
                    ))
                    converted[key] = target
                op.inputs[position] = target

        produced = enforced
        if produced is None:
            produced = Layout.SPATIAL
            for name in op.inputs:
                info = ag.tensors.get(name) if name else None
                if info is not None and not info.is_constant:
                    produced = info.layout
                    break
        if (produced is Layout.SPATIAL and _transposes_to_storage_order(ag, op)
                and op.outputs[0] not in ag.model_outputs
                and all(_required_layout(ag, ag.ops[reader]) is not Layout.SPATIAL
                        for reader, _ in uses.readers(op.outputs[0]))):
            # A Transpose to channels-last order writes the input's own stored
            # order, so held as stated it is a copy, and what reads it, such
            # as a flatten into a matrix product, sees the elements in the
            # order the model sums them.
            produced = Layout.LINEAR
        for name in op.outputs:
            info = ag.tensors.get(name)
            if info is not None:
                info.layout = produced

        rewritten.append(op)

    ag.ops = rewritten

    # Model inputs and outputs keep the convention callers already rely on, so
    # a boundary that ended up linear is converted back. Internal tensors are
    # free to be either; the interface is not.
    terminal_transpose = set() if ag.channels_last_source else {
        op.outputs[0]
        for op in ag.ops
        if op.op_type in {"Transpose", "ArgMax", "ArgMin"} and len(op.outputs) == 1
        and op.outputs[0] in ag.model_outputs
    }
    terminal_transpose.update(
        op.outputs[0] for op in ag.ops
        if op.op_type in {"ArgMax", "ArgMin"} and len(op.outputs) == 1
        and op.outputs[0] in ag.model_outputs)
    for name in list(ag.model_outputs):
        info = ag.tensors.get(name)
        if info is None or name in terminal_transpose:
            continue
        if info.layout is Layout.SPATIAL or len(info.shape) < 3:
            continue
        counter += 1
        produced = _fresh_name(ag, f"{name}_linear_{counter}")
        ag.tensors[produced] = TensorInfo(
            name=produced, shape=info.shape, dtype=info.dtype,
            quant=info.quant, layout=info.layout)
        for op in ag.ops:
            op.outputs = [produced if out == name else out for out in op.outputs]
            op.inputs = [produced if inp == name else inp for inp in op.inputs]
        info.layout = Layout.SPATIAL
        ag.ops.append(OpNode(
            name=f"layout_out_{counter}",
            op_type="Transpose",
            inputs=[produced],
            outputs=[name],
            attrs={"perm": list(range(len(info.shape)))},
        ))

    # A terminal Transpose is an explicit model-output boundary.
    for name in terminal_transpose:
        info = ag.tensors.get(name)
        if info is not None:
            info.layout = Layout.LINEAR

    return ag


def _lower_linear_matmul(ag: AnalyzedGraph, transposed: dict[str, str]) -> AnalyzedGraph:
    """Express a batched constant-weight matrix product with the FC kernel.

    ONNX MatMul reduces over the last axis and batches over everything before
    it. Once the operand is linear its rows are contiguous, so collapsing the
    leading axes is a pure reshape and the product becomes the rank-2 Gemm the
    fully-connected kernel already computes. Reshaping the result back restores
    the shape the model declares.

    Runs after layouts are assigned, because the collapse is only free on a
    linear tensor; on a spatial one the rows are interleaved. A product whose
    second operand is not a constant matrix is left alone: it needs a kernel
    that multiplies two activations, which the runtime does not have.
    ``transposed`` is the record ``_transpose_product_weights`` keeps.
    """
    products: dict[str, list[OpNode]] = {}
    for op in ag.ops:
        if (op.op_type != "MatMul" or len(op.inputs) not in (2, 3)
                or len(op.outputs) != 1):
            continue
        weight = ag.weight_data.get(op.inputs[1])
        data_info = ag.tensors.get(op.inputs[0])
        if (weight is None or weight.ndim != 2 or data_info is None
                or op.outputs[0] not in ag.tensors or len(data_info.shape) < 3
                or data_info.layout is not Layout.LINEAR
                or int(data_info.shape[-1]) != int(weight.shape[0])):
            continue
        products.setdefault(op.inputs[1], []).append(op)
    # The FC kernel indexes the weight as W[oc * IC + ic], which is Gemm
    # with transB=1; ONNX MatMul states it the other way round.
    _transpose_product_weights(ag, products, transposed)
    lowered = {id(op) for readers in products.values() for op in readers}

    rewritten: list[OpNode] = []
    counter = 0

    for op in ag.ops:
        if id(op) not in lowered:
            rewritten.append(op)
            continue

        data_info = ag.tensors[op.inputs[0]]
        out_info = ag.tensors[op.outputs[0]]
        reduction = int(data_info.shape[-1])
        rows = 1
        for extent in data_info.shape[:-1]:
            rows *= int(extent)
        columns = int(ag.weight_data[op.inputs[1]].shape[0])

        counter += 1
        flat_in = _fresh_name(ag, f"{op.inputs[0]}_rows_{counter}")
        ag.tensors[flat_in] = TensorInfo(
            name=flat_in, shape=(rows, reduction), dtype=data_info.dtype,
            quant=data_info.quant, layout=Layout.LINEAR)
        flat_out = _fresh_name(ag, f"{op.outputs[0]}_rows_{counter}")
        ag.tensors[flat_out] = TensorInfo(
            name=flat_out, shape=(rows, columns), dtype=out_info.dtype,
            quant=out_info.quant, layout=Layout.LINEAR)

        rewritten.append(OpNode(
            name=f"{op.name}_rows", op_type="Reshape",
            inputs=[op.inputs[0]], outputs=[flat_in]))
        # A bias folded onto the product before lowering travels with it:
        # the kernel reads one per output feature, which the collapse leaves
        # in place.
        rewritten.append(OpNode(
            name=op.name, op_type="Gemm",
            inputs=[flat_in, *op.inputs[1:]], outputs=[flat_out],
            attrs={"transB": 1}))
        rewritten.append(OpNode(
            name=f"{op.name}_shape", op_type="Reshape",
            inputs=[flat_out], outputs=[op.outputs[0]]))

    ag.ops = rewritten
    return ag


def _stored_extent(
    shape: tuple[int, ...], layout: Layout = Layout.SPATIAL
) -> tuple[int, ...]:
    """Non-unit axis sizes in the order the runtime serializes them.

    A tensor the runtime stores channels-last is described by a different axis
    order than the one the model states, and which order that is depends on
    the tensor's layout, not on its rank: a rank-4 matrix operand lists its
    axes in storage order already. Axes of size 1 contribute no stride, so
    dropping them leaves exactly the sequence the runtime walks. Two shapes
    with equal results hold their elements in the same order.
    """
    return tuple(
        extent for extent in serialized_shape(shape, layout) if extent != 1
    )


def _relabel_shape_ops_to_reshape(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel Squeeze and Unsqueeze to Reshape where element order survives.

    Both only add or drop axes of size 1, which is what Reshape does, and
    Reshape has a kernel while they do not. The relabel is only sound when the
    runtime layout walks the elements in the same order on both sides: a rank-4
    to rank-3 squeeze that removes the channel axis turns NHWC (h, w) order into
    NLC (w, h) order, which would transpose the data silently. Anything that
    fails the check keeps its original op type and is reported as unsupported.

    Runs after layouts are assigned, because which order the runtime walks is
    a property of the tensor rather than of its rank. Judging it by rank
    refuses every squeeze on a matrix tensor, which is what a batch of
    attention heads is.
    """
    for op in ag.ops:
        if op.op_type not in ("Squeeze", "Unsqueeze"):
            continue
        src = ag.tensors.get(op.inputs[0])
        dst = ag.tensors.get(op.outputs[0])
        if src is None or dst is None:
            continue
        if (_stored_extent(tuple(src.shape), src.layout)
                != _stored_extent(tuple(dst.shape), dst.layout)):
            continue
        op.op_type = "Reshape"
        # Squeeze and Unsqueeze carry their axes as a second input or an
        # attribute; the plan takes the output shape from the tensor table.
        op.inputs = op.inputs[:1]
        op.attrs.pop("axes", None)
    return ag


def _gather_one_index_to_split(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Rewrite a Gather of one constant index as a Split and drop the axis.

    Taking a single index out of an axis nothing precedes is a contiguous run
    of the stored bytes, which is exactly what a Split cuts. A classifier that
    reads the leading token of a sequence, as a vision transformer's head
    does, is the common case.

    Both sides have to state their own axis order, so the stored bytes follow
    the shape: a tensor the runtime stores channels-last would put the wanted
    index somewhere other than the front of a run.
    """
    replacements: list[tuple[int, list[OpNode]]] = []
    for index, op in enumerate(ag.ops):
        if op.op_type != "Gather" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        source = ag.tensors.get(op.inputs[0])
        result = ag.tensors.get(op.outputs[0])
        indices = ag.weight_data.get(op.inputs[1])
        if source is None or result is None or indices is None:
            continue
        if indices.size != 1 or not np.issubdtype(indices.dtype, np.integer):
            continue
        rank = len(source.shape)
        axis = int(op.attrs.get("axis", 0))
        if axis < 0:
            axis += rank
        if rank < 2 or axis < 0 or axis >= rank:
            continue
        if not _states_its_own_order(source) or not _states_its_own_order(result):
            continue
        if any(dim != 1 for dim in source.shape[:axis]):
            continue
        wanted = int(indices.reshape(-1)[0])
        if wanted < 0:
            wanted += int(source.shape[axis])
        if not 0 <= wanted < int(source.shape[axis]):
            continue
        expected = tuple(source.shape[:axis]) + tuple(source.shape[axis + 1:])
        if tuple(result.shape) != expected:
            continue

        rest = 1
        for dim in source.shape[axis + 1:]:
            rest *= int(dim)
        rows = int(source.shape[axis])
        flat_name = _fresh_name(ag, f"{op.inputs[0]}_rows")
        ag.tensors[flat_name] = replace(
            ag.tensors[op.inputs[0]], name=flat_name, shape=(rows, rest),
            layout=Layout.LINEAR)
        rewritten = [
            OpNode(name=f"{op.name}_rows", op_type="Reshape",
                   inputs=[op.inputs[0]], outputs=[flat_name], attrs={}),
        ]

        cuts = [(0, wanted), (wanted, wanted + 1), (wanted + 1, rows)]
        parts: list[str] = []
        for start, end in cuts:
            if end <= start:
                parts.append("")
                continue
            part = _fresh_name(ag, f"{op.outputs[0]}_part{start}")
            ag.tensors[part] = replace(
                ag.tensors[op.outputs[0]], name=part, shape=(end - start, rest),
                layout=Layout.LINEAR)
            parts.append(part)
        kept = parts[1]
        rewritten.append(
            OpNode(name=f"{op.name}_split", op_type="Split",
                   inputs=[flat_name], outputs=[p for p in parts if p],
                   attrs={"axis": 0}))
        rewritten.append(
            OpNode(name=op.name, op_type="Reshape",
                   inputs=[kept], outputs=[op.outputs[0]], attrs={}))
        replacements.append((index, rewritten))

    for index, rewritten in reversed(replacements):
        ag.ops[index:index + 1] = rewritten
    return ag


def _normalize_concat_axis(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Restate the Concat axis in the order the runtime stores the tensor.

    ONNX names the axis in the model's own order. The runtime reads it out of
    ``kernel_shape`` and applies it to the stored shape, so an image tensor's
    channel axis 1 has to come through as the last axis, and a tensor that
    already states its own order keeps the axis it was given. The layout map
    says which, so this runs once layouts are assigned.
    """
    for op in ag.ops:
        if op.op_type == "Split" and op.inputs:
            # A Split names its axis the same way, 0 being the outermost.
            source = ag.tensors.get(op.inputs[0])
            if source is not None and source.shape:
                rank = len(source.shape)
                axis = int(op.attrs.get("axis", 0)) % rank
                op.attrs["kernel_shape"] = [serialized_axis_map(rank, source.layout)[axis]]
            continue
        if op.op_type != "Concat":
            continue
        result = ag.tensors.get(op.outputs[0]) if op.outputs else None
        if result is None:
            continue
        rank = len(result.shape)
        axis = int(op.attrs.get("axis", 1))
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            continue
        stored = serialized_axis_map(rank, result.layout)[axis]
        op.attrs["kernel_shape"] = [stored]
        op.attrs["concat_last_axis"] = stored == rank - 1
    return ag
