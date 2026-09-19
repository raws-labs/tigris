"""ONNX-specific normalizations for deployment.

Passes applied in sequence (matches ``normalize()`` call order):

 0. **Constant folding**: Convert ONNX Constant ops to weight_data so
    downstream passes can read values directly.
 1. **QDQ fold**: For quantized (QDQ) models, fold QuantizeLinear /
    DequantizeLinear ops -- extract scale/zero-point, store as
    ``QuantParam`` on tensors, keep int8 weight data, remove Q/DQ ops.
 2. **BN fold**: For each Conv->BatchNormalization pattern, fold the BN
    parameters (gamma, beta, mean, var, epsilon) into the convolution
    weights and bias, then remove the BN op and rewire outputs.
 3. **SiLU decomposition**: Replace ``Silu`` ops with ``Sigmoid`` + ``Mul``
    sequence for runtime execution.
 4. **DepthwiseConv relabeling**: Convolutions where ``group == C_in``
    (i.e., depthwise) are relabeled from ``Conv`` to ``DepthwiseConv``
    so the runtime can dispatch a specialised kernel.
 5. **Conv1D relabeling**: Conv ops with ``len(kernel_shape) == 1`` are
    relabeled from ``Conv`` to ``Conv1D`` for dedicated 1D kernel dispatch.
 6. **Clip(0, 6) -> Relu6**: Replace ``Clip`` ops whose min/max are 0/6
    with the simpler ``Relu6`` op type.
 7. **ReduceMean -> GlobalAveragePool**: Replace ``ReduceMean(axes=[2,3])``
    with ``GlobalAveragePool`` (PyTorch >= 2.x uses ReduceMean for GAP).
 8. **Shape op folding**: Remove chains of shape-computation ops
    (Constant, Shape, Gather, Unsqueeze, Concat) that feed Reshape's
    shape input. All shapes are static at compile time.
 9. **Resize scale extraction**: Extract integer scale factors from
    ``Resize`` ops' constant inputs and store in ``strides`` attr.
10. **Concat axis normalization**: Translate Concat axis from NCHW to NHWC
    and store in ``kernel_shape`` attr for spatial packing.
11. **Transpose validation**: Validate explicit/default permutations before
    emission; the binary plan records them as per-operator attributes.
12. **Activation absorption**: Fuse Relu/Relu6 into preceding
    Conv/DepthwiseConv/Gemm/Conv1D ops as ``fused_activation`` attr.
    Runs last so all relabeling and rewiring is already done.
"""

import math
import numpy as np

from tigris.graph.ir import (
    AnalyzedGraph,
    Layout,
    OpNode,
    QuantParam,
    TensorInfo,
)


def normalize(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Apply all normalization passes in sequence."""
    ag = _drop_inference_identities(ag)
    ag = _fold_constant_ops(ag)
    ag = _fold_qdq(ag)
    ag = _fold_gemm_scalars(ag)
    ag = _relabel_matmul_to_gemm(ag)
    ag = _fold_bn(ag)
    ag = _neg_to_scalar_mul(ag)
    ag = _fold_sub_constant_to_add(ag)
    ag = _fold_div_constant_to_mul(ag)
    ag = _fold_constant_add_into_bias(ag)
    ag = _fold_channel_bias_add(ag)
    ag = _decompose_silu(ag)
    ag = _relabel_depthwise(ag)
    ag = _relabel_conv1d(ag)
    ag = _clip_to_relu6(ag)
    ag = _reduce_mean_to_gap(ag)
    ag = _fold_shape_ops(ag)
    ag = _relabel_shape_ops_to_reshape(ag)
    ag = _extract_resize_scales(ag)
    ag = _strip_metadata_inputs(ag)
    ag = _normalize_concat_axis(ag)
    ag = _validate_transposes(ag)
    ag = _absorb_activations(ag)
    ag = _assign_tensor_layouts(ag)
    ag = _lower_linear_matmul(ag)
    ag = _drop_unreferenced_weights(ag)
    return ag


# Operand positions that carry shape or bound metadata rather than tensor data.
# Earlier passes lift these into op attributes and into the output shape, so by
# this point they describe the plan the compiler already emitted. Leaving them
# on the op makes the emitter bind an index vector as the operator's weight.
_METADATA_INPUTS: dict[str, int] = {
    "Clip": 1,
    "Pad": 1,
    "ReduceMean": 1,
    "Reshape": 1,
    "Resize": 1,
    "Squeeze": 1,
    "Unsqueeze": 1,
}


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



# Operators whose activation operands are spatial pictures: the runtime holds
# them channels-last and the kernels index them that way.
_SPATIAL_LAYOUT_OPS = frozenset({
    "Conv",
    "DepthwiseConv",
    "Conv1D",
    "ConvTranspose",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "Resize",
})

# Operators whose operands already list their axes in storage order. A matrix
# product reduces over its last axis, so permuting it channels-last would move
# the reduction axis and compute something else.
_LINEAR_LAYOUT_OPS = frozenset({
    "MatMul",
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


def _required_layout(ag: AnalyzedGraph, op: OpNode) -> Layout | None:
    """The layout an operator needs, or None when it works in either."""
    if op.op_type in _SPATIAL_LAYOUT_OPS:
        return Layout.SPATIAL
    if op.op_type in _LINEAR_LAYOUT_OPS:
        return Layout.LINEAR
    if op.op_type in ("Softmax", "LayerNormalization"):
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
    shapes only reveal it when they differ, which is why a square block came
    back wrong rather than refused.

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


def _assign_tensor_layouts(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Give every tensor a layout and convert where producer and consumer differ.

    Layout only has consequences at rank 3 and rank 4, where ONNX axis order
    and storage order disagree. An operator either requires one of the two or
    works in whichever it is handed, and a tensor carries whatever its producer
    wrote. Where a consumer needs the other one, an explicit Transpose converts
    it: the emitter turns an identity permutation across differing layouts into
    the physical permutation, so the conversion costs no new machinery.

    A terminal Transpose stays linear for the reason it always did: its output
    is a model-output boundary that keeps the observable ONNX shape and order.
    """
    converted: dict[tuple[str, Layout], str] = {}
    rewritten: list[OpNode] = []
    counter = 0

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
                    target = f"{name}_to_{enforced.value}_{counter}"
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
        for name in op.outputs:
            info = ag.tensors.get(name)
            if info is not None:
                info.layout = produced

        rewritten.append(op)

    ag.ops = rewritten

    # Model inputs and outputs keep the convention callers already rely on, so
    # a boundary that ended up linear is converted back. Internal tensors are
    # free to be either; the interface is not.
    terminal_transpose = {
        op.outputs[0]
        for op in ag.ops
        if op.op_type == "Transpose" and len(op.outputs) == 1
        and op.outputs[0] in ag.model_outputs
    }
    for name in list(ag.model_outputs):
        info = ag.tensors.get(name)
        if info is None or name in terminal_transpose:
            continue
        if info.layout is Layout.SPATIAL or len(info.shape) < 3:
            continue
        counter += 1
        produced = f"{name}_linear_{counter}"
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

    for step, op in enumerate(ag.ops):
        op.step = step
    return ag



def _lower_linear_matmul(ag: AnalyzedGraph) -> AnalyzedGraph:
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
    """
    consumers: dict[str, int] = {}
    for op in ag.ops:
        for name in op.inputs:
            consumers[name] = consumers.get(name, 0) + 1

    rewritten: list[OpNode] = []
    counter = 0

    for op in ag.ops:
        if (op.op_type != "MatMul" or len(op.inputs) not in (2, 3)
                or len(op.outputs) != 1):
            rewritten.append(op)
            continue

        weight_name = op.inputs[1]
        weight = ag.weight_data.get(weight_name)
        data_info = ag.tensors.get(op.inputs[0])
        out_info = ag.tensors.get(op.outputs[0])
        if (weight is None or weight.ndim != 2 or data_info is None
                or out_info is None or len(data_info.shape) < 3
                or data_info.layout is not Layout.LINEAR
                or consumers.get(weight_name, 0) != 1):
            rewritten.append(op)
            continue

        reduction = int(data_info.shape[-1])
        if reduction != int(weight.shape[0]):
            rewritten.append(op)
            continue
        rows = 1
        for extent in data_info.shape[:-1]:
            rows *= int(extent)
        columns = int(weight.shape[1])

        counter += 1
        flat_in = f"{op.inputs[0]}_rows_{counter}"
        flat_out = f"{op.outputs[0]}_rows_{counter}"
        ag.tensors[flat_in] = TensorInfo(
            name=flat_in, shape=(rows, reduction), dtype=data_info.dtype,
            quant=data_info.quant, layout=Layout.LINEAR)
        ag.tensors[flat_out] = TensorInfo(
            name=flat_out, shape=(rows, columns), dtype=out_info.dtype,
            quant=out_info.quant, layout=Layout.LINEAR)

        # The FC kernel indexes the weight as W[oc * IC + ic], which is Gemm
        # with transB=1; ONNX MatMul states it the other way round.
        ag.weight_data[weight_name] = np.ascontiguousarray(weight.T)
        weight_info = ag.tensors.get(weight_name)
        if weight_info is not None:
            weight_info.shape = tuple(reversed(weight_info.shape))
            quant = weight_info.quant
            if quant is not None and quant.axis in (0, 1):
                weight_info.quant = QuantParam(
                    scale=quant.scale,
                    zero_point=quant.zero_point,
                    axis=1 - quant.axis,
                )

        rewritten.append(OpNode(
            name=f"{op.name}_rows", op_type="Reshape",
            inputs=[op.inputs[0]], outputs=[flat_in]))
        # A bias folded onto the product before lowering travels with it:
        # the kernel reads one per output feature, which the collapse leaves
        # in place.
        rewritten.append(OpNode(
            name=op.name, op_type="Gemm",
            inputs=[flat_in, weight_name, *op.inputs[2:]], outputs=[flat_out],
            attrs={"transB": 1}))
        rewritten.append(OpNode(
            name=f"{op.name}_shape", op_type="Reshape",
            inputs=[flat_out], outputs=[op.outputs[0]]))

    ag.ops = rewritten
    for step, op in enumerate(ag.ops):
        op.step = step
    return ag


def _drop_unreferenced_weights(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Forget constants no remaining operator consumes.

    Folded subgraphs and absorbed activations leave their operands behind, and
    the emitter writes every entry of ``weight_data`` into the plan blob.
    """
    referenced = {name for op in ag.ops for name in op.inputs if name}
    for name in [n for n in ag.weight_data if n not in referenced]:
        del ag.weight_data[name]
    return ag


# Inference-time identities and shape relabels


_INFERENCE_IDENTITY_OPS = frozenset({"Dropout", "Identity"})


def _drop_inference_identities(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Remove operators that are the identity at inference.

    Dropout scales nothing once training_mode is false, and Identity never did.
    Exporters leave both in place, and neither has a plan opcode, so a stock
    export is rejected for operators that would compute nothing. Dropout's
    optional second output is a mask that only training consumes; an op whose
    mask is actually read is left alone rather than silently dropped.
    """
    consumed = {name for op in ag.ops for name in op.inputs if name}
    outputs_kept = set(ag.model_outputs)

    surviving: list[OpNode] = []
    rename: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type not in _INFERENCE_IDENTITY_OPS:
            surviving.append(op)
            continue
        if len(op.outputs) > 1 and any(
            out in consumed or out in outputs_kept for out in op.outputs[1:]
        ):
            surviving.append(op)
            continue
        source = op.inputs[0]
        rename[op.outputs[0]] = rename.get(source, source)
        for stale in op.outputs[1:]:
            ag.tensors.pop(stale, None)

    if not rename:
        return ag

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in rename and name not in seen:
            seen.add(name)
            name = rename[name]
        return name

    for op in surviving:
        op.inputs = [resolve(n) if n else n for n in op.inputs]
    ag.model_outputs = [resolve(n) for n in ag.model_outputs]

    # A removed op's output tensor keeps the graph's declared name when it is a
    # model output, so only drop tensors nothing refers to any more.
    live = {n for op in surviving for n in op.inputs + op.outputs if n}
    live |= set(ag.model_inputs) | set(ag.model_outputs)
    for name in [n for n in rename if n not in live]:
        ag.tensors.pop(name, None)

    ag.ops = surviving
    return ag


def _stored_extent(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Non-unit axis sizes in the order the runtime serializes them.

    The IR keeps ONNX NCHW/NCL while the runtime stores NHWC/NLC, so the same
    element sequence is described by different axis orders at different ranks.
    Axes of size 1 contribute no stride, so dropping them leaves exactly the
    sequence the runtime walks. Two shapes with equal results hold their
    elements in the same order.
    """
    rank = len(shape)
    if rank == 4:
        order = (0, 2, 3, 1)   # N, H, W, C
    elif rank == 3:
        order = (0, 2, 1)      # N, L, C
    else:
        order = tuple(range(rank))
    return tuple(shape[axis] for axis in order if shape[axis] != 1)


def _relabel_shape_ops_to_reshape(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel Squeeze and Unsqueeze to Reshape where element order survives.

    Both only add or drop axes of size 1, which is what Reshape does, and
    Reshape has a kernel while they do not. The relabel is only sound when the
    runtime layout walks the elements in the same order on both sides: a rank-4
    to rank-3 squeeze that removes the channel axis turns NHWC (h, w) order into
    NLC (w, h) order, which would transpose the data silently. Anything that
    fails the check keeps its original op type and is reported as unsupported.
    """
    for op in ag.ops:
        if op.op_type not in ("Squeeze", "Unsqueeze"):
            continue
        src = ag.tensors.get(op.inputs[0])
        dst = ag.tensors.get(op.outputs[0])
        if src is None or dst is None:
            continue
        if _stored_extent(tuple(src.shape)) != _stored_extent(tuple(dst.shape)):
            continue
        op.op_type = "Reshape"
        # Squeeze and Unsqueeze carry their axes as a second input or an
        # attribute; the plan takes the output shape from the tensor table.
        op.inputs = op.inputs[:1]
        op.attrs.pop("axes", None)
    return ag


# Constant folding


def _fold_constant_ops(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold ONNX Constant ops into weight_data.

    ONNX Constant nodes produce a tensor from an attribute value rather
    than from a graph initializer.  This pass converts them into weight
    data entries so downstream passes (QDQ fold, Resize scale extraction)
    can find them in ``ag.weight_data``.
    """
    from onnx import numpy_helper

    removed: set[int] = set()
    for i, op in enumerate(ag.ops):
        if op.op_type != "Constant":
            continue
        val = op.attrs.get("value")
        if val is None:
            continue
        out_name = op.outputs[0]
        arr = numpy_helper.to_array(val)
        ag.weight_data[out_name] = arr
        if out_name in ag.tensors:
            ag.tensors[out_name].is_constant = True
        else:
            ag.tensors[out_name] = TensorInfo(
                name=out_name,
                shape=tuple(arr.shape),
                dtype=1,  # FLOAT
                is_constant=True,
            )
        removed.add(i)

    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag


# Quantization


def _fold_qdq(ag: AnalyzedGraph) -> AnalyzedGraph:
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
    4. Removes all Q/DQ ops and rewires connections.
    5. Sets ag.is_quantized = True.
    """
    qdq_types = {"QuantizeLinear", "DequantizeLinear"}
    has_qdq = any(op.op_type in qdq_types for op in ag.ops)
    if not has_qdq:
        return ag

    # Build output->op index map
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    # Build consumer map: tensor_name -> list of (op_idx, input_position)
    input_to_consumers: dict[str, list[tuple[int, int]]] = {}
    for i, op in enumerate(ag.ops):
        for pos, inp in enumerate(op.inputs):
            input_to_consumers.setdefault(inp, []).append((i, pos))

    removed: set[int] = set()
    # A scale or zero point may be shared by a weight pair and an activation
    # pair. Deleting it as soon as the weight pair is folded leaves the
    # activation pairs with no parameters to read, so every removal waits until
    # both passes have run.
    deferred_cleanup: set[str] = set()

    def _get_quant_param(op: OpNode) -> tuple[QuantParam | None, bool]:
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


    # Pass 1: Process DequantizeLinear on weight inputs.
    # Pattern: weight_init -> QuantizeLinear -> int8 -> DequantizeLinear -> fake_float
    # Or simply: int8_init -> DequantizeLinear -> fake_float
    for i, op in enumerate(ag.ops):
        if op.op_type != "DequantizeLinear":
            continue

        dql_input = op.inputs[0]
        dql_output = op.outputs[0]

        # Check if the input is a weight (constant/initializer)
        is_weight_dql = False

        # Case 1: int8 initializer -> DequantizeLinear
        if dql_input in ag.weight_data:
            is_weight_dql = True
            qp, unsigned = _get_quant_param(op)
            if qp is None:
                continue

            if unsigned:
                ag.weight_data[dql_input] = _to_signed(ag.weight_data[dql_input])

            # Store QuantParam on the weight tensor
            if dql_input in ag.tensors:
                ag.tensors[dql_input].quant = qp
                ag.tensors[dql_input].dtype = 3  # INT8

            # Rewire: all consumers of DQL output now read the weight directly
            for cons_idx, pos in input_to_consumers.get(dql_output, []):
                ag.ops[cons_idx].inputs[pos] = dql_input

            # Remove intermediate tensor
            if dql_output in ag.tensors and dql_output != dql_input:
                del ag.tensors[dql_output]

            removed.add(i)

        # Case 2: QuantizeLinear -> DequantizeLinear on a weight
        elif dql_input in output_to_op:
            ql_idx = output_to_op[dql_input]
            ql_op = ag.ops[ql_idx]
            if ql_op.op_type == "QuantizeLinear" and ql_op.inputs[0] in ag.weight_data:
                is_weight_dql = True
                original_weight = ql_op.inputs[0]

                # Get quant params from the DQL op
                qp, unsigned = _get_quant_param(op)
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

                # Rewire consumers of DQL output to the original weight
                for cons_idx, pos in input_to_consumers.get(dql_output, []):
                    ag.ops[cons_idx].inputs[pos] = original_weight

                # Remove intermediate tensors
                for tname in [dql_input, dql_output]:
                    if tname in ag.tensors and tname != original_weight:
                        del ag.tensors[tname]

                removed.add(i)
                removed.add(ql_idx)

        if is_weight_dql:
            for inp_name in op.inputs[1:]:
                if inp_name:
                    deferred_cleanup.add(inp_name)

    # Pass 2: Process activation Q/DQ pairs.
    # Pattern: activation -> QuantizeLinear -> int8 -> DequantizeLinear -> consumer
    for i, op in enumerate(ag.ops):
        if i in removed:
            continue
        if op.op_type != "QuantizeLinear":
            continue

        ql_output = op.outputs[0]
        ql_input = op.inputs[0]

        # Skip if input is a weight (already handled above)
        if ql_input in ag.weight_data:
            continue

        # Get quant params from this QL op
        qp, unsigned = _get_quant_param(op)
        if qp is None:
            continue

        # Store quant param on the activation tensor and set dtype to INT8
        if ql_input in ag.tensors:
            ag.tensors[ql_input].quant = qp
            ag.tensors[ql_input].dtype = 3  # INT8

        # Find the DQL consumer(s) of this QL's output
        dql_indices = []
        for cons_idx, pos in input_to_consumers.get(ql_output, []):
            if ag.ops[cons_idx].op_type == "DequantizeLinear":
                dql_indices.append(cons_idx)

        for dql_idx in dql_indices:
            dql_op = ag.ops[dql_idx]
            dql_output = dql_op.outputs[0]

            # Rewire: consumers of DQL output now read the original activation
            for cons_idx, pos in input_to_consumers.get(dql_output, []):
                ag.ops[cons_idx].inputs[pos] = ql_input

            # If this DQL output is a model output, update to point to ql_input
            if dql_output in ag.model_outputs:
                ag.model_outputs = [
                    ql_input if n == dql_output else n for n in ag.model_outputs
                ]

            # Remove DQL intermediate tensor
            if dql_output in ag.tensors and dql_output != ql_input:
                del ag.tensors[dql_output]

            removed.add(dql_idx)

            # Defer DQL scale/zp cleanup
            for inp_name in dql_op.inputs[1:]:
                if inp_name:
                    deferred_cleanup.add(inp_name)

        # Remove QL intermediate tensor
        if ql_output in ag.tensors and ql_output != ql_input:
            del ag.tensors[ql_output]

        removed.add(i)

        # Defer QL scale/zp cleanup
        for inp_name in op.inputs[1:]:
            if inp_name:
                deferred_cleanup.add(inp_name)

    # Now clean up all deferred scale/zp constants
    for name in deferred_cleanup:
        if name in ag.weight_data:
            del ag.weight_data[name]
        if name in ag.tensors:
            del ag.tensors[name]

    if removed:
        ag.ops = [op for idx, op in enumerate(ag.ops) if idx not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step
        ag.is_quantized = True

    return ag


# Batch normalization


def _fold_bn(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold Conv -> BatchNormalization into a single Conv with updated weights."""
    # Build output->op index map
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    # Build input consumer map: tensor_name -> list of op indices
    input_to_consumers: dict[str, list[int]] = {}
    for i, op in enumerate(ag.ops):
        for inp in op.inputs:
            input_to_consumers.setdefault(inp, []).append(i)

    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "BatchNormalization":
            continue

        # BN inputs: X, scale(gamma), B(beta), mean, var
        if len(op.inputs) < 5:
            continue

        bn_input = op.inputs[0]
        if bn_input not in output_to_op:
            continue

        conv_idx = output_to_op[bn_input]
        conv_op = ag.ops[conv_idx]

        if conv_op.op_type not in ("Conv",):
            continue

        # Conv must have exactly one consumer (the BN)
        consumers = input_to_consumers.get(bn_input, [])
        if len(consumers) != 1:
            continue

        # Get BN parameters from weight_data
        gamma_name = op.inputs[1]
        beta_name = op.inputs[2]
        mean_name = op.inputs[3]
        var_name = op.inputs[4]

        if not all(
            n in ag.weight_data for n in [gamma_name, beta_name, mean_name, var_name]
        ):
            continue

        gamma = ag.weight_data[gamma_name]
        beta = ag.weight_data[beta_name]
        mean = ag.weight_data[mean_name]
        var = ag.weight_data[var_name]
        eps = op.attrs.get("epsilon", 1e-5)

        # Conv weight: inputs[1]
        if len(conv_op.inputs) < 2 or conv_op.inputs[1] not in ag.weight_data:
            continue
        conv_w_name = conv_op.inputs[1]
        W = ag.weight_data[conv_w_name]

        # Quantized weights (int8/int32): just remove BN, don't touch weights
        if W.dtype in (np.int8, np.int32):
            # Rewire: conv's output becomes the BN's output
            bn_output = op.outputs[0]
            conv_op.outputs = [bn_output]

            # Remove the intermediate tensor
            if bn_input in ag.tensors and not ag.tensors[bn_input].is_constant:
                del ag.tensors[bn_input]

            # Clean up BN weight tensors
            for wname in [gamma_name, beta_name, mean_name, var_name]:
                if wname in ag.weight_data:
                    del ag.weight_data[wname]
                if wname in ag.tensors:
                    del ag.tensors[wname]

            removed.add(i)
            continue

        W = W.copy()

        # Conv bias: inputs[2] (optional)
        has_bias = len(conv_op.inputs) >= 3 and conv_op.inputs[2] in ag.weight_data
        if has_bias:
            conv_b_name = conv_op.inputs[2]
            B = ag.weight_data[conv_b_name].copy()
        else:
            B = np.zeros(W.shape[0], dtype=np.float32)

        # Fold: scale = gamma / sqrt(var + eps)
        inv_std = 1.0 / np.sqrt(var + eps)
        scale = gamma * inv_std

        # W_new[oc] = W[oc] * scale[oc]  (broadcast over spatial dims)
        # For NCHW layout, W shape is [OC, IC/G, KH, KW]
        scale_shape = [len(scale)] + [1] * (W.ndim - 1)
        W_new = W * scale.reshape(scale_shape)
        B_new = (B - mean) * inv_std * gamma + beta

        # Update weight_data
        ag.weight_data[conv_w_name] = W_new.astype(np.float32)

        if has_bias:
            ag.weight_data[conv_b_name] = B_new.astype(np.float32)
        else:
            # Create a new bias tensor
            bias_name = conv_w_name.replace("weight", "bias")
            if bias_name == conv_w_name:
                bias_name = conv_w_name + "_bias"
            ag.weight_data[bias_name] = B_new.astype(np.float32)
            ag.tensors[bias_name] = TensorInfo(
                name=bias_name,
                shape=B_new.shape,
                dtype=1,  # FLOAT
                is_constant=True,
            )
            conv_op.inputs = (
                list(conv_op.inputs[:2]) + [bias_name] + list(conv_op.inputs[3:])
            )

        # Rewire: conv's output becomes the BN's output
        bn_output = op.outputs[0]
        conv_op.outputs = [bn_output]

        # Remove the intermediate tensor
        if bn_input in ag.tensors and not ag.tensors[bn_input].is_constant:
            del ag.tensors[bn_input]

        # Clean up BN weight tensors from weight_data (no longer needed)
        for wname in [gamma_name, beta_name, mean_name, var_name]:
            if wname in ag.weight_data:
                del ag.weight_data[wname]
            if wname in ag.tensors:
                del ag.tensors[wname]

        removed.add(i)

    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
        # Re-assign step indices
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag


# Op relabeling


# Producers whose kernels read an optional bias operand after the weight.
# MatMul is absent because a relabelable one is already a Gemm by this point.
_BIAS_PRODUCERS = frozenset({"Conv", "DepthwiseConv", "Conv1D", "Gemm"})


def _takes_a_bias(ag: AnalyzedGraph, producer: OpNode) -> bool:
    """Whether this operator has a bias operand to fold a constant Add into.

    A batched matrix product over a constant weight is lowered to the same
    fully-connected kernel as a Gemm, which reads a bias, so it can take one
    here even though ONNX MatMul states no such operand. One over two
    activations cannot: there is no weight for a bias to sit beside.
    """
    if producer.op_type in _BIAS_PRODUCERS:
        return True
    if producer.op_type != "MatMul" or len(producer.inputs) != 2:
        return False
    weight = ag.weight_data.get(producer.inputs[1])
    return weight is not None and weight.ndim == 2



def _fold_gemm_scalars(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold Gemm's alpha and beta into the constants they scale.

    The plan has no field for either, and the fully-connected kernel computes
    Y = X * W^T + B, so a Gemm carrying them used to compile and return a
    silently wrong answer. Scaling a constant weight by alpha and a constant
    bias by beta is exact and leaves nothing for the plan to express. transA
    transposes an activation, which no constant can absorb; validation refuses
    it.
    """
    for op in ag.ops:
        if op.op_type != "Gemm" or len(op.inputs) < 2:
            continue

        alpha = float(op.attrs.get("alpha", 1.0))
        if alpha != 1.0:
            weight = ag.weight_data.get(op.inputs[1])
            if weight is not None and weight.dtype == np.float32:
                ag.weight_data[op.inputs[1]] = np.ascontiguousarray(
                    weight * np.float32(alpha))
                op.attrs["alpha"] = 1.0

        beta = float(op.attrs.get("beta", 1.0))
        if beta != 1.0:
            if len(op.inputs) < 3:
                # beta scales C; with no C there is no term for it to scale.
                op.attrs["beta"] = 1.0
            else:
                bias = ag.weight_data.get(op.inputs[2])
                if bias is not None and bias.dtype == np.float32:
                    ag.weight_data[op.inputs[2]] = np.ascontiguousarray(
                        bias * np.float32(beta))
                    op.attrs["beta"] = 1.0
    return ag


def _relabel_matmul_to_gemm(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Put a constant-weight matrix product in the layout the kernels read.

    The fully-connected kernels index the weight as ``W[oc * IC + ic]``, which
    is ONNX ``Gemm`` with ``transB=1``. An ONNX ``MatMul`` states the same
    product with the weight the other way round, and so does a ``Gemm`` that
    leaves ``transB`` at its default, so both need the constant transposed
    before they mean what the kernels compute.

    A product whose second operand is not a constant matrix is left alone: the
    runtime has no kernel for it and rejects the plan rather than guessing.
    """
    consumers: dict[str, int] = {}
    for op in ag.ops:
        for name in op.inputs:
            consumers[name] = consumers.get(name, 0) + 1

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

        weight_name = op.inputs[1]
        weight = ag.weight_data.get(weight_name)
        if weight is None or weight.ndim != 2:
            continue
        # Transposing in place would misstate the weight for any other reader.
        if consumers.get(weight_name, 0) != 1:
            continue
        data_info = ag.tensors.get(op.inputs[0])
        if data_info is None or len(data_info.shape) != 2:
            continue

        ag.weight_data[weight_name] = np.ascontiguousarray(weight.T)
        info = ag.tensors.get(weight_name)
        if info is not None:
            info.shape = tuple(reversed(info.shape))
            quant = info.quant
            if quant is not None and quant.axis in (0, 1):
                info.quant = QuantParam(
                    scale=quant.scale,
                    zero_point=quant.zero_point,
                    axis=1 - quant.axis,
                )
        op.op_type = "Gemm"
        op.attrs["transB"] = 1
    return ag




def _neg_to_scalar_mul(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Rewrite Neg as a multiplication by minus one.

    Neg has no opcode, so a graph holding one is rejected outright, yet it is
    exactly the scalar-constant Mul the kernels already carry. The scalar is a
    float, so this applies before quantization folding, where a quantized graph
    still states its operands in float.
    """
    for index, op in enumerate(ag.ops):
        if op.op_type != "Neg" or len(op.inputs) != 1:
            continue
        info = ag.tensors.get(op.inputs[0])
        if info is None or info.dtype != 1:
            continue

        scalar = f"{op.outputs[0]}_minus_one_{index}"
        ag.weight_data[scalar] = np.array([-1.0], dtype=np.float32)
        ag.tensors[scalar] = TensorInfo(
            name=scalar, shape=(1,), dtype=1, is_constant=True)
        op.op_type = "Mul"
        op.inputs = [op.inputs[0], scalar]
        op.attrs = {}
    return ag


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

    Div has no opcode, yet a constant divisor is exactly the Mul the kernels
    already carry: a GELU exports as ``Erf(x / sqrt(2))`` and an attention
    scale as a division by the head width. A divisor on the left has no such
    equivalent, and a zero in one would turn an infinity the model states into
    a finite number, so both are left for validation to reject.

    The reciprocal is taken in double precision and stored back as float32,
    which is the nearest representable inverse rather than the one a float32
    division would produce. The two differ by at most a half ulp of the
    reciprocal, well inside the tolerance the contract gate holds float to.
    """
    for op in ag.ops:
        if op.op_type != "Div" or len(op.inputs) != 2:
            continue
        divisor = op.inputs[1]
        if divisor not in ag.weight_data or op.inputs[0] in ag.weight_data:
            continue
        constant = ag.weight_data[divisor]
        if constant.dtype != np.float32 or constant.size == 0:
            continue
        if not np.all(np.isfinite(constant)) or np.any(constant == 0.0):
            continue
        reciprocal = (1.0 / constant.astype(np.float64)).astype(np.float32)
        if not np.all(np.isfinite(reciprocal)):
            continue
        ag.weight_data[divisor] = np.ascontiguousarray(reciprocal)
        op.op_type = "Mul"
    return ag


def _fold_constant_add_into_bias(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Turn a constant Add after a quantized producer into that producer's bias.

    A quantizer that leaves MatMul and its bias unfused writes the bias as a
    float Add reading the dequantized product, which is the one form the
    runtime cannot execute: a constant operand carries no scale or zero point
    in the plan, so a quantized Add fails closed on it. Requantizing the
    constant to the product's own int32 domain is how every quantized Gemm
    already carries its bias.

    Only the quantized case is folded. A float graph executes the Add as
    written, so there is nothing to recover there.
    """
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    consumers: dict[str, list[int]] = {}
    for i, op in enumerate(ag.ops):
        for inp in op.inputs:
            consumers.setdefault(inp, []).append(i)

    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "Add" or len(op.inputs) != 2:
            continue
        constants = [n for n in op.inputs if n in ag.weight_data]
        if len(constants) != 1:
            continue
        bias_name = constants[0]
        product = next(n for n in op.inputs if n != bias_name)

        producer_idx = output_to_op.get(product)
        if producer_idx is None or producer_idx in removed:
            continue
        producer = ag.ops[producer_idx]
        if producer.op_type not in _BIAS_PRODUCERS or len(producer.inputs) != 2:
            continue
        if consumers.get(product, []) != [i]:
            continue

        product_info = ag.tensors.get(product)
        weight_info = ag.tensors.get(producer.inputs[1])
        if product_info is None or weight_info is None:
            continue
        if product_info.quant is None or weight_info.quant is None:
            continue
        input_info = ag.tensors.get(producer.inputs[0])
        if input_info is None or input_info.quant is None:
            continue

        bias = ag.weight_data[bias_name]
        if bias.dtype != np.float32:
            continue
        channels = product_info.shape[-1] if product_info.shape else 0
        if bias.size != channels:
            continue

        # The int32 bias lives in the product's own accumulator domain, which
        # is the input scale times the weight scale, per channel where the
        # weights are.
        bias_scale = (
            input_info.quant.scale.reshape(-1) * weight_info.quant.scale.reshape(-1)
        )
        if bias_scale.size not in (1, bias.size):
            continue
        quantized = np.round(bias.reshape(-1) / bias_scale).astype(np.int64)
        if np.any(np.abs(quantized) > np.iinfo(np.int32).max):
            continue

        ag.weight_data[bias_name] = quantized.astype(np.int32)
        if bias_name in ag.tensors:
            ag.tensors[bias_name].dtype = 6  # INT32
        producer.inputs.append(bias_name)

        # The product's encoding becomes the model's: keep the Add's output
        # name so the plan still names what the model named, and give it the
        # product's dtype and quantization.
        result = op.outputs[0]
        result_info = ag.tensors.get(result)
        if result_info is not None:
            result_info.dtype = product_info.dtype
            result_info.quant = product_info.quant
        producer.outputs = [result]
        del ag.tensors[product]
        removed.add(i)

    if removed:
        ag.ops = [op for idx, op in enumerate(ag.ops) if idx not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag


def _axis_broadcast_size(
    constant_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    axis: int,
) -> int | None:
    """Extent when a constant broadcasts only along *axis* of the reference.

    ONNX right-aligns operands, so ``(C,)``, ``(C, 1, 1)`` and ``(1, C, 1, 1)``
    all address the channel axis of an NCHW activation while ``(1, 1, 1, W)``
    addresses width. Size alone cannot tell those apart when C equals W, so the
    aligned position is what decides.
    """
    if len(reference_shape) < 2 or len(constant_shape) > len(reference_shape):
        return None
    extent_on_axis = reference_shape[axis]
    if extent_on_axis <= 0:
        return None
    offset = len(reference_shape) - len(constant_shape)
    for position, extent in enumerate(constant_shape):
        aligned = position + offset
        if aligned == axis:
            if extent != extent_on_axis:
                return None
        elif extent != 1:
            return None
    # A constant shorter than the reference must still reach the axis.
    if offset > axis:
        return None
    return extent_on_axis


def _channel_broadcast_size(
    constant_shape: tuple[int, ...], reference_shape: tuple[int, ...]
) -> int | None:
    """The channel count a constant addresses on an NCHW or NCL activation."""
    return _axis_broadcast_size(constant_shape, reference_shape, 1)


def _bias_broadcast_size(
    producer_type: str,
    constant_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
) -> int | None:
    """The bias length a constant states for *producer_type*'s output.

    Which axis a bias addresses is a property of the operator that produced
    the tensor, not of the tensor's rank. A convolution biases its output
    channel, which the model states second. A matrix product biases its output
    feature, which is the last axis on both sides and is what the
    fully-connected kernel indexes its bias by.
    """
    matrix = producer_type in ("Gemm", "MatMul")
    axis = len(reference_shape) - 1 if matrix else 1
    return _axis_broadcast_size(constant_shape, reference_shape, axis)


def _fold_channel_bias_add(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold a float per-channel constant Add into its producer's bias.

    Exporters emit a per-channel bias as a standalone Add whenever they do not
    fuse it, and a decomposed BatchNorm leaves the same shape behind. The Add
    kernel takes two operands of one shape, so such a graph is rejected for
    broadcasting it cannot do, even though the producer already has a bias slot
    holding exactly this quantity. Adding into that slot is exact for float.

    The quantized form is handled by _fold_constant_add_into_bias, which has to
    requantize into the accumulator domain. Here the bias is already float.
    """
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    consumers: dict[str, list[int]] = {}
    for i, op in enumerate(ag.ops):
        for inp in op.inputs:
            consumers.setdefault(inp, []).append(i)

    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "Add" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        constants = [n for n in op.inputs if n in ag.weight_data]
        if len(constants) != 1:
            continue
        const_name = constants[0]
        product = next(n for n in op.inputs if n != const_name)

        producer_idx = output_to_op.get(product)
        if producer_idx is None or producer_idx in removed:
            continue
        producer = ag.ops[producer_idx]
        if not _takes_a_bias(ag, producer):
            continue
        if len(producer.inputs) not in (2, 3):
            continue
        # Folding past another consumer would change what that consumer reads.
        if consumers.get(product, []) != [i]:
            continue

        product_info = ag.tensors.get(product)
        if product_info is None or product_info.quant is not None:
            continue

        constant = ag.weight_data[const_name]
        if constant.dtype != np.float32:
            continue
        channels = _bias_broadcast_size(
            producer.op_type, tuple(constant.shape), tuple(product_info.shape)
        )
        if channels is None or constant.size != channels:
            continue

        addend = constant.reshape(-1).astype(np.float32)
        if len(producer.inputs) == 3:
            existing = ag.weight_data.get(producer.inputs[2])
            if existing is None or existing.dtype != np.float32:
                continue
            if existing.reshape(-1).size != channels:
                continue
            ag.weight_data[producer.inputs[2]] = (
                existing.reshape(-1).astype(np.float32) + addend
            )
        else:
            ag.weight_data[const_name] = addend
            if const_name in ag.tensors:
                ag.tensors[const_name].shape = (channels,)
            producer.inputs.append(const_name)

        # Keep the Add's output name so the plan still names what the model did.
        result = op.outputs[0]
        result_info = ag.tensors.get(result)
        if result_info is not None:
            result_info.dtype = product_info.dtype
        producer.outputs = [result]
        del ag.tensors[product]
        removed.add(i)

    if removed:
        ag.ops = [op for idx, op in enumerate(ag.ops) if idx not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag


def _decompose_silu(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace Silu ops with Sigmoid + Mul sequence.

    SiLU(x) = x * sigmoid(x). Some exporters (e.g. Ultralytics opset-11)
    already emit Sigmoid + Mul - this pass handles the case where a single
    ``Silu`` op is present.
    """
    new_ops: list[OpNode] = []
    changed = False

    for op in ag.ops:
        if op.op_type != "Silu":
            new_ops.append(op)
            continue

        changed = True
        x_name = op.inputs[0]
        y_name = op.outputs[0]

        # Create intermediate tensor for sigmoid output
        sig_name = f"{x_name}_sigmoid"
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

    if changed:
        ag.ops = new_ops
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag


def _relabel_depthwise(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel Conv ops with group == C_in as DepthwiseConv."""
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
            # Depthwise: group == OC and IC/G == 1
            if W.ndim >= 2 and W.shape[1] == 1 and group == W.shape[0]:
                op.op_type = "DepthwiseConv"
        elif group > 1:
            # No weight data but group > 1 - check tensor info
            if w_name and w_name in ag.tensors:
                w_info = ag.tensors[w_name]
                if (
                    len(w_info.shape) >= 2
                    and w_info.shape[1] == 1
                    and group == w_info.shape[0]
                ):
                    op.op_type = "DepthwiseConv"

    return ag


def _relabel_conv1d(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Relabel Conv ops with 1D kernel_shape as Conv1D.

    Must run after BN fold (which only matches Conv->BN, not Conv1D->BN)
    and after depthwise relabeling (which already changed group==C_in convs).
    """
    for op in ag.ops:
        if op.op_type != "Conv":
            continue
        ks = op.attrs.get("kernel_shape", [])
        if len(ks) == 1:
            op.op_type = "Conv1D"

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

        # Check inputs: Clip(input, min, max) - opset >= 11
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

    Newer ONNX exporters (PyTorch >= 2.x) emit ReduceMean over spatial
    dimensions instead of GlobalAveragePool.  They are semantically
    identical for 4-D NHWC/NCHW tensors, but the runtime only has a GAP
    kernel.

    Handles both attribute-based axes (opset < 18) and input-based axes
    (opset >= 18).
    """
    reductions = {"ReduceMean": "GlobalAveragePool", "ReduceMax": "GlobalMaxPool"}

    for op in ag.ops:
        if op.op_type not in reductions:
            continue

        # Try axes from attribute first (opset < 18)
        axes = op.attrs.get("axes")

        # Try axes from second input (opset >= 18)
        if axes is None and len(op.inputs) >= 2:
            axes_name = op.inputs[1]
            if axes_name in ag.weight_data:
                axes = ag.weight_data[axes_name].flatten().tolist()

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

        op.op_type = reductions[op.op_type]
        op.attrs = {}

        # Remove axes input and clean up axes tensor
        if len(op.inputs) >= 2:
            axes_name = op.inputs[1]
            op.inputs = [op.inputs[0]]
            if axes_name in ag.weight_data:
                del ag.weight_data[axes_name]
            if axes_name in ag.tensors:
                del ag.tensors[axes_name]

    return ag


# Shape & layout normalization

_SHAPE_OP_TYPES = frozenset({"Shape", "Gather", "Constant", "Unsqueeze", "Concat"})


def _fold_shape_ops(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Remove shape-computation op chains that feed Reshape's shape input.

    In models like MobileNetV2, a chain of Constant->Shape->Gather->Unsqueeze->
    Concat computes the target shape for Reshape at runtime.  Since all shapes
    are static at compile time, these ops can be removed.  The intermediate
    tensors are marked ``is_constant=True`` so the binary emitter skips them.
    """
    # Build output->op index map
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "Reshape":
            continue
        if len(op.inputs) < 2:
            continue

        # Reshape's second input is the target shape tensor
        shape_tensor = op.inputs[1]

        # BFS backward to collect all producer ops
        visited_ops: set[int] = set()
        queue = [shape_tensor]
        seen_tensors: set[str] = set()
        all_shape_ops = True

        while queue:
            tname = queue.pop()
            if tname in seen_tensors:
                continue
            seen_tensors.add(tname)

            if tname not in output_to_op:
                continue  # model input or initializer - not an op output

            prod_idx = output_to_op[tname]
            if prod_idx in visited_ops:
                continue
            visited_ops.add(prod_idx)

            prod_op = ag.ops[prod_idx]
            if prod_op.op_type not in _SHAPE_OP_TYPES:
                all_shape_ops = False
                break

            # Shape and Constant ops don't need to follow inputs:
            # Shape reads shape metadata (not data), Constant has none.
            if prod_op.op_type in ("Shape", "Constant"):
                continue

            # Continue BFS through this op's inputs
            for inp in prod_op.inputs:
                queue.append(inp)

        if all_shape_ops and visited_ops:
            removed |= visited_ops
            # Mark all intermediate tensors as constant
            for tname in seen_tensors:
                if tname in ag.tensors:
                    ag.tensors[tname].is_constant = True

    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step

    # Fix Reshape output shapes that ONNX shape inference couldn't resolve
    _fix_reshape_shapes(ag)

    return ag


# NOTE: Mutates ag in place (unlike other passes which return ag).
def _fix_reshape_shapes(ag: AnalyzedGraph) -> None:
    """Compute correct output shapes for Reshape ops with unknown outputs.

    ONNX shape inference often can't resolve Reshape output shapes when the
    target shape comes from a dynamic computation chain (Shape->Gather->Concat).
    After folding those chains, we can compute the shape statically from the
    data input shape.
    """
    for op in ag.ops:
        if op.op_type != "Reshape":
            continue
        if len(op.inputs) < 2 or len(op.outputs) < 1:
            continue

        out_name = op.outputs[0]
        out_info = ag.tensors.get(out_name)
        if out_info is None or out_info.shape:
            continue  # shape already known

        data_name = op.inputs[0]
        data_info = ag.tensors.get(data_name)
        if data_info is None or not data_info.shape:
            continue

        # Try to get target shape from weight_data (constant initializer)
        shape_name = op.inputs[1]
        if shape_name in ag.weight_data:
            target = ag.weight_data[shape_name].flatten().tolist()
        else:
            # Infer from data input: flatten to [batch, -1]
            target = [data_info.shape[0], -1]

        # Resolve -1 dimension
        total = 1
        for d in data_info.shape:
            total *= d
        neg_idx = None
        known_product = 1
        resolved = []
        for i, d in enumerate(target):
            d = int(d)
            if d == -1:
                neg_idx = i
                resolved.append(-1)
            elif d == 0:
                # 0 means "copy from input"
                resolved.append(data_info.shape[i] if i < len(data_info.shape) else 1)
                known_product *= resolved[-1]
            else:
                resolved.append(d)
                known_product *= d

        if neg_idx is not None:
            resolved[neg_idx] = total // known_product

        out_info.shape = tuple(resolved)


def _extract_resize_scales(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Extract integer scale factors from Resize ops.

    ONNX Resize has inputs: X, roi, scales, [sizes]. This pass reads the
    constant ``scales`` or ``sizes`` input, computes integer H/W scale
    factors, and stores them in ``op.attrs["strides"]`` so the binary
    writer packs them into spatial.stride_h/w.
    """
    for op in ag.ops:
        if op.op_type != "Resize":
            continue

        x_name = op.inputs[0]
        x_info = ag.tensors.get(x_name)
        y_name = op.outputs[0]
        y_info = ag.tensors.get(y_name)

        scale_h, scale_w = 1, 1

        # Try scales input (index 2)
        if len(op.inputs) >= 3 and op.inputs[2] and op.inputs[2] in ag.weight_data:
            scales = ag.weight_data[op.inputs[2]].flatten()
            if len(scales) == 4:
                # NCHW: [N=1, C=1, H_scale, W_scale]
                scale_h = max(1, int(round(float(scales[2]))))
                scale_w = max(1, int(round(float(scales[3]))))

        # Try sizes input (index 3) if scales didn't work
        if (
            scale_h == 1
            and scale_w == 1
            and len(op.inputs) >= 4
            and op.inputs[3]
            and op.inputs[3] in ag.weight_data
        ):
            sizes = ag.weight_data[op.inputs[3]].flatten()
            if len(sizes) == 4 and x_info and len(x_info.shape) == 4:
                # NCHW input shape
                scale_h = max(1, int(round(float(sizes[2]) / float(x_info.shape[2]))))
                scale_w = max(1, int(round(float(sizes[3]) / float(x_info.shape[3]))))

        # Infer from input/output shapes as fallback
        if scale_h == 1 and scale_w == 1 and x_info and y_info:
            if len(x_info.shape) == 4 and len(y_info.shape) == 4:
                in_h, in_w = x_info.shape[2], x_info.shape[3]  # NCHW
                out_h, out_w = y_info.shape[2], y_info.shape[3]
                if in_h > 0 and in_w > 0:
                    scale_h = max(1, out_h // in_h)
                    scale_w = max(1, out_w // in_w)

        op.attrs["strides"] = [scale_h, scale_w]

        # Strip constant inputs (roi, scales, sizes) - keep only X
        for inp_name in op.inputs[1:]:
            if inp_name and inp_name in ag.weight_data:
                del ag.weight_data[inp_name]
            if inp_name and inp_name in ag.tensors and ag.tensors[inp_name].is_constant:
                del ag.tensors[inp_name]
        op.inputs = [op.inputs[0]]

    return ag


def _normalize_concat_axis(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Translate Concat axis from NCHW to NHWC convention.

    ONNX Concat has an ``axis`` attribute. For 4D tensors, NCHW axis=1
    (channel concat) maps to NHWC axis=3. The axis is stored in
    ``kernel_shape`` for packing into spatial.kernel_h.
    """
    for op in ag.ops:
        if op.op_type != "Concat":
            continue

        axis = op.attrs.get("axis", 1)

        # Normalize negative axes for 4D
        if axis < 0:
            axis = 4 + axis

        # Map NCHW axis to NHWC
        if axis == 1:
            nhwc_axis = 3  # channel axis
        elif axis == 0:
            nhwc_axis = 0  # batch (rare)
        elif axis == 2:
            nhwc_axis = 1  # H
        elif axis == 3:
            nhwc_axis = 2  # W
        else:
            nhwc_axis = axis

        # Store axis in kernel_shape so _pack_spatial_attrs writes kernel_h
        op.attrs["kernel_shape"] = [nhwc_axis]
        # Clear pads/strides/dilations to avoid spurious spatial packing
        op.attrs.pop("pads", None)
        op.attrs.pop("strides", None)
        op.attrs.pop("dilations", None)

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


# Activation fusion

_FUSABLE_PRODUCERS = frozenset(
    {"Conv", "DepthwiseConv", "Gemm", "Conv1D", "Add"}
)
_FUSABLE_ACTIVATIONS = frozenset({"Relu", "Relu6"})


def _absorb_activations(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fuse Relu/Relu6 into preceding Conv/DepthwiseConv/Gemm/Conv1D.

    Sets ``attrs["fused_activation"]`` on the producer op, rewires outputs,
    removes the activation op and its intermediate tensor.
    """
    # Build output->op index map
    output_to_op: dict[str, int] = {}
    for i, op in enumerate(ag.ops):
        for out in op.outputs:
            output_to_op[out] = i

    # Build input consumer map: tensor_name -> list of op indices
    input_to_consumers: dict[str, list[int]] = {}
    for i, op in enumerate(ag.ops):
        for inp in op.inputs:
            input_to_consumers.setdefault(inp, []).append(i)

    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type not in _FUSABLE_ACTIVATIONS:
            continue

        act_input = op.inputs[0]
        act_output = op.outputs[0]

        # Check that the input comes from a fusable producer
        if act_input not in output_to_op:
            continue
        prod_idx = output_to_op[act_input]
        producer = ag.ops[prod_idx]

        if producer.op_type not in _FUSABLE_PRODUCERS:
            continue

        # Producer must have exactly one consumer (the activation op)
        consumers = input_to_consumers.get(act_input, [])
        if len(consumers) != 1:
            continue

        # An intermediate carrying its own scale is a quantization step of its
        # own, and folding the activation past it would drop that rounding.
        # Fusing is exact only where the producer output is an unquantized edge
        # inside the region, which is how a QDQ exporter writes an activation it
        # expects the consumer to absorb.
        intermediate = ag.tensors.get(act_input)
        if intermediate is not None and intermediate.quant is not None:
            continue

        # Fuse: set attr on producer, rewire output
        producer.attrs["fused_activation"] = op.op_type
        producer.outputs = [act_output]

        # Remove intermediate tensor
        if act_input in ag.tensors and not ag.tensors[act_input].is_constant:
            del ag.tensors[act_input]

        removed.add(i)

    if removed:
        ag.ops = [op for idx, op in enumerate(ag.ops) if idx not in removed]
        for step, op in enumerate(ag.ops):
            op.step = step

    return ag
