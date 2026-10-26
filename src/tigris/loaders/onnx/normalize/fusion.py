"""Operator chains folded into one operator or into a constant weight."""

from dataclasses import replace

import numpy as np

from tigris.graph.ir import AnalyzedGraph, OpNode, QuantParam, TensorInfo
from tigris.loaders.onnx.normalize.common import _forget_unread, _fresh_name, _own_constant, _Uses
from tigris.loaders.onnx.normalize.quantization import _fold_bn_into_quantized_conv


def _split_axis_before_reshape(
    source: tuple[int, ...], result: tuple[int, ...], axis: int
) -> tuple[int, int] | None:
    """Where *axis* of a reshape's result sits in its source, and its stride.

    A split is a cut into contiguous blocks, so pushing one back through a
    reshape is only sound when the axis being cut is the outermost factor of a
    single source axis: everything before it has to line up, and everything
    after it has to come from that same axis. Then cutting the result at that
    axis is cutting the source axis into blocks of the trailing product.
    """
    prefix = 1
    for extent in result[:axis]:
        prefix *= int(extent)
    trailing = 1
    for extent in result[axis + 1:]:
        trailing *= int(extent)
    running = 1
    for position, extent in enumerate(source):
        if running == prefix:
            if int(extent) != int(result[axis]) * trailing:
                return None
            return position, trailing
        running *= int(extent)
    return None


def _fold_split_into_its_weight(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Cut the weight rather than the product it produced.

    A fused projection computes query, key and value in one matrix product and
    the model cuts them apart afterwards, through whatever reshapes and
    transposes put the stacked axis where the cut wants it. Cutting a product
    along its output-feature axis is the same as three products against three
    blocks of the weight, so the cut can be done once at compile time. The
    stacked tensor then never exists: neither the product that holds all three
    parts, nor the copy that separates them.

    The split is pushed back through shape-only operators one at a time and
    folded only if it reaches a product's feature axis. Anything else keeps
    its split, which the runtime executes.
    """
    while True:
        if not _fold_one_split(ag):
            return ag


def _fold_one_split(ag: AnalyzedGraph) -> bool:
    uses = _Uses(ag)
    for split_index, split in enumerate(ag.ops):
        if split.op_type != "Split" or len(split.inputs) < 1:
            continue
        parts = [ag.tensors.get(name) for name in split.outputs]
        if len(parts) < 2 or any(part is None for part in parts):
            continue

        axis = int(split.attrs.get("axis", 0))
        name = split.inputs[0]
        # Each entry: the op, and the axis its output stacks the parts on, with that axis' extent.
        chain: list[tuple[int, int, int]] = []
        while True:
            index = uses.producers.get(name)
            if index is None or uses.sole_reader(name) is None:
                break
            op = ag.ops[index]
            current = ag.tensors.get(name)
            source = ag.tensors.get(op.inputs[0]) if op.inputs else None
            if current is None or source is None:
                break
            stacked = axis
            if op.op_type == "Transpose":
                perm = [int(a) for a in op.attrs.get("perm", [])]
                if len(perm) != len(current.shape):
                    break
                axis = perm[axis]
            elif op.op_type == "Reshape":
                mapped = _split_axis_before_reshape(
                    tuple(source.shape), tuple(current.shape), axis)
                if mapped is None:
                    break
                axis, _ = mapped
            elif op.op_type in ("MatMul", "Gemm"):
                if axis != len(current.shape) - 1:
                    break
                if _fold_split_onto(ag, split_index, chain, index, parts):
                    return True
                break
            else:
                break
            chain.append((index, stacked, int(current.shape[stacked])))
            name = op.inputs[0]
    return False


def _fold_split_onto(
    ag: AnalyzedGraph,
    split_index: int,
    chain: list[tuple[int, int, int]],
    product_index: int,
    parts: list,
) -> bool:
    """Rebuild the chain once per part, over a block of the weight.

    Every part is built before the graph is changed, so a fold that does not
    go through leaves the graph as it was.
    """
    product = ag.ops[product_index]
    if len(product.inputs) not in (2, 3):
        return False
    weight = ag.weight_data.get(product.inputs[1])
    if weight is None or weight.ndim != 2:
        return False
    if int(product.attrs.get("transB", 0)):
        return False
    features = int(weight.shape[1])
    result = ag.tensors.get(product.outputs[0])
    if result is None or int(result.shape[-1]) != features:
        return False

    split = ag.ops[split_index]
    axis = int(split.attrs.get("axis", 0))
    widths = [int(part.shape[axis]) for part in parts]
    total = sum(widths)
    stride, remainder = divmod(features, total)
    if remainder or stride <= 0:
        return False

    bias = None
    if len(product.inputs) == 3:
        bias = ag.weight_data.get(product.inputs[2])
        if bias is None or bias.reshape(-1).size != features:
            return False
        bias = bias.reshape(-1)

    starts = np.cumsum([0, *widths[:-1]]) * stride
    blocks = [(int(start), width * stride) for start, width in zip(starts, widths)]
    weight_quants = _quant_of_blocks(ag.tensors[product.inputs[1]].quant, 2, 1, features, blocks)
    bias_quants = (_quant_of_blocks(ag.tensors[product.inputs[2]].quant, 1, 0, features, blocks)
                   if bias is not None else [None] * len(blocks))
    if weight_quants is None or bias_quants is None:
        return False
    # Every part's activations take the stacked tensors' quantization, which holds only per tensor.
    stacked_quants = [result.quant, *(ag.tensors[ag.ops[index].outputs[0]].quant for index, _, _ in chain)]
    if any(quant is not None and np.asarray(quant.scale).size > 1 for quant in stacked_quants):
        return False

    taken: set[str] = set()

    def fresh(base: str) -> str:
        name = _fresh_name(ag, base, taken)
        taken.add(name)
        return name

    weights: dict[str, np.ndarray] = {}
    tensors: dict[str, TensorInfo] = {}
    rebuilt: list[OpNode] = []
    for position, (offset, block) in enumerate(blocks):
        tag = f"{product.name}_part{position}"
        weight_name = fresh(f"{product.inputs[1]}_part{position}")
        weights[weight_name] = np.ascontiguousarray(weight[:, offset:offset + block])
        tensors[weight_name] = TensorInfo(
            name=weight_name, shape=(int(weight.shape[0]), block),
            dtype=ag.tensors[product.inputs[1]].dtype, is_constant=True,
            quant=weight_quants[position])
        inputs = [product.inputs[0], weight_name]
        if bias is not None:
            bias_name = fresh(f"{product.inputs[2]}_part{position}")
            weights[bias_name] = np.ascontiguousarray(bias[offset:offset + block])
            tensors[bias_name] = TensorInfo(
                name=bias_name, shape=(block,),
                dtype=ag.tensors[product.inputs[2]].dtype, is_constant=True,
                quant=bias_quants[position])
            inputs.append(bias_name)

        carried = fresh(f"{tag}_product") if chain else split.outputs[position]
        if chain:
            tensors[carried] = TensorInfo(
                name=carried,
                shape=(*tuple(result.shape[:-1]), block),
                dtype=result.dtype, quant=result.quant, layout=result.layout)
        rebuilt.append(OpNode(
            name=tag, op_type=product.op_type, inputs=inputs,
            outputs=[carried], attrs=dict(product.attrs)))

        # The chain is walked from the split back to the product, so replaying
        # it forwards means reversing it. Every shape it states carries this
        # part's share of the stacked axis instead of all of it.
        for depth, (index, stacked, extent) in enumerate(reversed(chain)):
            step = ag.ops[index]
            last = depth == len(chain) - 1
            output = split.outputs[position] if last else fresh(f"{tag}_step{depth}")
            if not last:
                reference = ag.tensors[step.outputs[0]]
                shape = [int(dim) for dim in reference.shape]
                shape[stacked] = widths[position] * extent // total
                tensors[output] = TensorInfo(
                    name=output, shape=tuple(shape), dtype=reference.dtype,
                    quant=reference.quant, layout=reference.layout)
            rebuilt.append(OpNode(
                name=f"{tag}_{step.name.split('/')[-1]}",
                op_type=step.op_type,
                inputs=[carried],
                outputs=[output],
                attrs=dict(step.attrs)))
            carried = output

    ag.weight_data.update(weights)
    ag.tensors.update(tensors)
    replaced = {split_index, product_index, *(index for index, _, _ in chain)}
    ops: list[OpNode] = []
    for index, op in enumerate(ag.ops):
        if index == product_index:
            ops.extend(rebuilt)
        elif index in replaced:
            continue
        else:
            ops.append(op)
    ag.ops = ops
    return True


def _quant_of_blocks(
    quant: QuantParam | None, rank: int, axis: int, extent: int,
    blocks: list[tuple[int, int]],
) -> list[QuantParam | None] | None:
    """Each block's quantization when a tensor is cut along ``axis`` into ``(start, size)`` blocks.

    A per-tensor quantization holds for every block. A per-channel one is cut
    with the blocks when it runs along ``axis`` with one entry per index;
    otherwise the blocks have none of their own and None is returned.
    """
    if quant is None or np.asarray(quant.scale).size == 1:
        return [quant] * len(blocks)
    scale = np.asarray(quant.scale).reshape(-1)
    zero = np.asarray(quant.zero_point).reshape(-1)
    if quant.axis % rank != axis or scale.size != extent or zero.size != extent:
        return None
    return [QuantParam(scale=scale[start:start + size].copy(),
                       zero_point=zero[start:start + size].copy(), axis=axis)
            for start, size in blocks]


def _fold_elementwise_patterns(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Recognize elementwise compositions without crossing quantization edges."""
    uses = _Uses(ag)
    removed: set[str] = set()

    def match(name: str, kind: str, arity: int) -> OpNode | None:
        op = uses.producer(name)
        return op if op is not None and op.op_type == kind and len(op.inputs) == arity else None

    def constant(name: str) -> bool:
        source = uses.producer(name)
        if source is not None and source.op_type == "DequantizeLinear" and source.inputs:
            name = source.inputs[0]
        info = ag.tensors.get(name)
        return name in ag.weight_data or (info is not None and info.is_constant)

    def zero(name: str) -> bool:
        value = ag.weight_data.get(name)
        return value is not None and value.size == 1 and float(value.reshape(-1)[0]) == 0.0

    def replace(root: OpNode, kind: str, inputs: list[str], parts: list[OpNode]) -> None:
        # One operand may be a constant, dequantized or not; the binary
        # checks judge its extent.
        operands = [ag.tensors.get(name) for name in inputs]
        dynamic = [t for name, t in zip(inputs, operands)
                   if t is not None and not constant(name)]
        if (any(t is None for t in operands) or not dynamic
                or any(t.shape != dynamic[0].shape for t in dynamic[1:])):
            return
        names = {name for part in parts for name in part.outputs}
        group = {id(part) for part in parts} | {id(root)}
        if names.intersection(ag.model_outputs) or any(
            id(op) not in group and names.intersection(op.inputs) for op in ag.ops
        ):
            return
        root.op_type, root.inputs, root.attrs = kind, inputs, {}
        removed.update(names)

    for op in ag.ops:
        if op.op_type == "LpNormalization" and int(op.attrs.get("p", 2)) == 2:
            op.op_type = "L2Normalization"
            op.attrs = {"axis": int(op.attrs.get("axis", -1)), "epsilon": 0.0}
        if op.op_type == "Sqrt" and len(op.inputs) == 1:
            pool = match(op.inputs[0], "AveragePool", 1)
            square = match(pool.inputs[0], "Mul", 2) if pool is not None else None
            if square is not None and square.inputs[0] == square.inputs[1]:
                replace(op, "L2Pool", square.inputs[:1], [square, pool])
                if op.op_type == "L2Pool":
                    op.attrs = pool.attrs.copy()
        elif op.op_type == "Div" and len(op.inputs) == 2:
            # x / max(sqrt(sum(x * x)), epsilon) over one axis: TFLite's L2
            # normalization with its norm floor.
            x = op.inputs[0]
            floor = match(op.inputs[1], "Max", 2)
            norm = match(floor.inputs[0], "Sqrt", 1) if floor is not None else None
            total = match(norm.inputs[0], "ReduceSum", 2) if norm is not None else None
            square = match(total.inputs[0], "Mul", 2) if total is not None else None
            epsilon = ag.weight_data.get(floor.inputs[1]) if floor is not None else None
            axes = ag.weight_data.get(total.inputs[1]) if total is not None else None
            if (square is None or square.inputs != [x, x] or epsilon is None or epsilon.size != 1
                    or axes is None or axes.size != 1
                    or int(total.attrs.get("keepdims", 1)) != 1):
                continue
            replace(op, "L2Normalization", [x], [square, total, norm, floor])
            if op.op_type == "L2Normalization":
                op.attrs = {"axis": int(axes.reshape(-1)[0]),
                            "epsilon": float(epsilon.reshape(-1)[0])}
        elif op.op_type == "Cast" and len(op.inputs) == 1 and int(op.attrs.get("to", 0)) == 9:
            # All of a bool axis: the minimum of its values as uint8, which
            # ONNX can state where it has no bool reduction.
            reduce = match(op.inputs[0], "ReduceMin", 1)
            widen = match(reduce.inputs[0], "Cast", 1) if reduce is not None else None
            source = ag.tensors.get(widen.inputs[0]) if widen is not None else None
            if source is None or source.dtype != 9 or int(widen.attrs.get("to", 0)) != 2:
                continue
            attrs = {"axes": list(reduce.attrs.get("axes", [])),
                     "keepdims": int(reduce.attrs.get("keepdims", 1))}
            replace(op, "ReduceAll", widen.inputs.copy(), [widen, reduce])
            if op.op_type == "ReduceAll":
                op.attrs = attrs
        elif op.op_type == "Reciprocal" and len(op.inputs) == 1:
            inner = match(op.inputs[0], "Sqrt", 1)
            if inner is not None:
                replace(op, "Rsqrt", inner.inputs.copy(), [inner])
        elif op.op_type == "Mul" and len(op.inputs) == 2 and op.inputs[0] == op.inputs[1]:
            inner = match(op.inputs[0], "Sub", 2)
            if inner is not None:
                replace(op, "SquaredDifference", inner.inputs.copy(), [inner])
        elif op.op_type == "Floor" and len(op.inputs) == 1:
            inner = match(op.inputs[0], "Div", 2)
            if inner is not None:
                replace(op, "FloorDiv", inner.inputs.copy(), [inner])
        elif op.op_type == "Where" and len(op.inputs) == 3:
            remainder = match(op.inputs[2], "Mod", 2)
            added = match(op.inputs[1], "Add", 2)
            condition = match(op.inputs[0], "And", 2)
            if remainder is None or added is None or condition is None:
                continue
            a, b = remainder.inputs
            r = op.inputs[2]
            if remainder.attrs.get("fmod", 0) != 1 or added.inputs != [r, b]:
                continue
            nonzero = match(condition.inputs[0], "Not", 1)
            different = match(condition.inputs[1], "Xor", 2)
            if nonzero is None or different is None:
                continue
            equal = match(nonzero.inputs[0], "Equal", 2)
            negative_b = match(different.inputs[0], "Less", 2)
            negative_r = match(different.inputs[1], "Less", 2)
            if equal is None or negative_b is None or negative_r is None:
                continue
            if (equal.inputs[0] == r and zero(equal.inputs[1])
                    and negative_b.inputs[0] == b and zero(negative_b.inputs[1])
                    and negative_r.inputs[0] == r and zero(negative_r.inputs[1])):
                replace(op, "FloorMod", [a, b], [remainder, added, condition, nonzero,
                                               different, equal, negative_b, negative_r])
    ag.ops = [op for op in ag.ops if not removed.intersection(op.outputs)]
    for name in removed:
        ag.tensors.pop(name, None)
    return ag


def _fold_bn(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fold Conv -> BatchNormalization into a single Conv with updated weights."""
    uses = _Uses(ag)
    reads = uses.reads
    removed: set[int] = set()
    folded_parameters: list[str] = []

    for i, op in enumerate(ag.ops):
        if op.op_type != "BatchNormalization":
            continue

        # BN inputs: X, scale(gamma), B(beta), mean, var
        if len(op.inputs) < 5:
            continue

        bn_input = op.inputs[0]
        if bn_input not in uses.producers:
            continue

        conv_idx = uses.producers[bn_input]
        conv_op = ag.ops[conv_idx]

        if conv_op.op_type not in ("Conv",):
            continue

        # Exactly one consumer (the BN), and not a model output.
        if uses.sole_reader(bn_input) is None:
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

        # The BN goes with the fold, so its reads no longer count.
        reads.subtract(op.inputs)
        # Quantized weights: fold the BN into the per-channel weight scale and
        # the int32 bias. A BN the fold cannot express stays in the graph, and
        # validation rejects it as unsupported.
        if W.dtype in (np.int8, np.int32):
            if not _fold_bn_into_quantized_conv(
                    ag, conv_op, gamma, beta, mean, var, eps, reads):
                reads.update(op.inputs)
                continue
        else:
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
            conv_w_name = _own_constant(ag, conv_op, 1, reads)
            ag.weight_data[conv_w_name] = W_new.astype(np.float32)

            if has_bias:
                conv_b_name = _own_constant(ag, conv_op, 2, reads)
                ag.weight_data[conv_b_name] = B_new.astype(np.float32)
            else:
                # Create a new bias tensor
                bias_name = conv_w_name.replace("weight", "bias")
                if bias_name == conv_w_name:
                    bias_name = conv_w_name + "_bias"
                bias_name = _fresh_name(ag, bias_name)
                reads[bias_name] += 1
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
        conv_op.outputs = [op.outputs[0]]

        # Remove the intermediate tensor
        if bn_input in ag.tensors and not ag.tensors[bn_input].is_constant:
            del ag.tensors[bn_input]

        folded_parameters += [gamma_name, beta_name, mean_name, var_name]
        removed.add(i)

    if removed:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
        _forget_unread(ag, folded_parameters)

    return ag


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

    A Dequantize between the product and Add prevents folding across rounding.
    """
    uses = _Uses(ag)
    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type != "Add" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        constants = [n for n in op.inputs if n in ag.weight_data]
        if len(constants) != 1:
            continue
        const_name = constants[0]
        product = next(n for n in op.inputs if n != const_name)

        producer_idx = uses.producers.get(product)
        if producer_idx is None or producer_idx in removed:
            continue
        producer = ag.ops[producer_idx]
        if not _takes_a_bias(ag, producer):
            continue
        if len(producer.inputs) not in (2, 3):
            continue
        # Another consumer or a model output still reads the product.
        if uses.sole_reader(product) != i:
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

    return ag


def _fold_gelu(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Recognize the erf and tanh GELU expressions without crossing a shared edge."""
    uses = _Uses(ag)

    def scalar(name, expected):
        value = ag.weight_data.get(name)
        return (value is not None and value.size == 1 and value.dtype.kind == "f"
                and np.float32(value.reshape(-1)[0]) == np.float32(expected))

    def node(name, kind):
        index = uses.producers.get(name)
        if index is None or ag.ops[index].op_type != kind:
            return None
        return index, ag.ops[index]

    def scaled(name, factor):
        held = node(name, "Mul")
        if held is None:
            return None
        i, op = held
        if len(op.inputs) != 2:
            return None
        for a, b in (op.inputs, op.inputs[::-1]):
            if scalar(b, factor):
                return a, {i}
        return None

    def product(name, x, expected, factor):
        used = set()
        leaves = []
        coefficient = 1.0
        def visit(value):
            nonlocal coefficient
            if value == x:
                leaves.append(value)
                return
            constant = ag.weight_data.get(value)
            if constant is not None and constant.size == 1 and constant.dtype.kind == "f":
                coefficient *= float(constant.reshape(-1)[0])
                return
            held = node(value, "Mul")
            if held is not None and len(held[1].inputs) == 2:
                used.add(held[0])
                for operand in held[1].inputs:
                    visit(operand)
            else:
                leaves.append(value)
        visit(name)
        if sorted(leaves) == sorted(expected) and np.float32(coefficient) == np.float32(factor):
            return used
        return None

    def cube(name, x):
        held = node(name, "Pow")
        if held is not None and held[1].inputs[0] == x and scalar(held[1].inputs[1], 3):
            return {held[0]}
        return product(name, x, [x, x, x], 1.0)

    removed = set()
    for activation_index, activation in enumerate(ag.ops):
        if activation.op_type not in {"Erf", "Tanh"} or len(activation.inputs) != 1:
            continue
        used = {activation_index}
        x = None
        if activation.op_type == "Erf":
            held = node(activation.inputs[0], "Div")
            if held is not None and scalar(held[1].inputs[1], np.sqrt(2.0)):
                x = held[1].inputs[0]
                used.add(held[0])
            else:
                held = scaled(activation.inputs[0], 1.0 / np.sqrt(2.0))
                if held is not None:
                    x, indices = held
                    used.update(indices)
        else:
            held = scaled(activation.inputs[0], np.sqrt(2.0 / np.pi))
            addition = node(held[0], "Add") if held else None
            if addition is not None:
                for candidate, term in (addition[1].inputs, addition[1].inputs[::-1]):
                    cubic = scaled(term, 0.044715)
                    powers = cube(cubic[0], candidate) if cubic else None
                    if powers is None:
                        powers = product(term, candidate, [candidate] * 3, 0.044715)
                    if powers is not None:
                        x = candidate
                        used.update(held[1] | {addition[0]} | powers)
                        if cubic is not None:
                            used.update(cubic[1])
                        break
        if x is None:
            continue
        for add_index in uses.reading_ops(activation.outputs[0]):
            gate = ag.ops[add_index]
            if gate.op_type != "Add" or len(gate.inputs) != 2 or not any(
                    scalar(value, 1.0) for value in gate.inputs):
                continue
            gate_name = gate.outputs[0]
            # 0.5 * x * gate takes at most two Muls, so the last one sits
            # within two consumer hops of the gate.
            first = {i for i in uses.reading_ops(gate_name) if ag.ops[i].op_type == "Mul"}
            second = {j for i in first for j in uses.reading_ops(ag.ops[i].outputs[0])
                      if ag.ops[j].op_type == "Mul"}
            for index in sorted(first | second):
                final = ag.ops[index]
                if len(final.outputs) != 1:
                    continue
                multiplications = product(final.outputs[0], x, [x, gate_name], 0.5)
                if multiplications is None:
                    continue
                group = used | {add_index, index} | multiplications
                if group & removed:
                    continue
                intermediate = [name for i in group if i != index for name in ag.ops[i].outputs]
                if any(name in ag.model_outputs or not uses.reading_ops(name) <= group
                       for name in intermediate):
                    continue
                if any(ag.tensors[name].dtype != 1 or ag.tensors[name].quant is not None
                       for name in [x, final.outputs[0], *intermediate]):
                    continue
                ag.ops[index] = replace(final, op_type="Gelu", inputs=[x],
                                        attrs={"approximate": "none" if activation.op_type == "Erf" else "tanh"})
                removed.update(group - {index})
                for name in intermediate:
                    ag.tensors.pop(name, None)
                break
    ag.ops = [op for i, op in enumerate(ag.ops) if i not in removed]
    return ag


def _fold_pad_into_conv(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Give a convolution the zero padding a Pad ahead of it was adding.

    A converter that lowers "same" padding writes it as an explicit Pad and
    leaves the convolution unpadded, which materializes a whole padded copy of
    the feature map for nothing. The convolution already pads with zero, and
    in the quantized domain with the input's zero point, which is the same
    value a constant Pad of zero writes, so the two are exchangeable.

    Only the spatial axes may be padded, the fill has to be zero, and the
    padded tensor may have no other reader: anything else and the Pad stays.
    """
    uses = _Uses(ag)
    folded: set[int] = set()
    for index, op in enumerate(ag.ops):
        if op.op_type != "Pad" or not op.outputs or len(op.inputs) < 2:
            continue
        if op.attrs.get("mode", "constant") != "constant":
            continue
        reader = uses.sole_reader(op.outputs[0])
        if reader is None:
            continue
        target = ag.ops[reader]
        if target.op_type not in ("Conv", "DepthwiseConv"):
            continue
        if target.inputs[0] != op.outputs[0]:
            continue
        amounts = ag.weight_data.get(op.inputs[1])
        if amounts is None:
            continue
        fill = ag.weight_data.get(op.inputs[2]) if len(op.inputs) > 2 else None
        if fill is not None and fill.size and float(fill.reshape(-1)[0]) != 0.0:
            continue
        source = ag.tensors.get(op.inputs[0])
        if source is None or len(source.shape) != 4:
            continue
        values = [int(v) for v in amounts.reshape(-1)]
        if len(values) != 8:
            continue
        begins, ends = values[:4], values[4:]
        if begins[0] or begins[1] or ends[0] or ends[1]:
            continue
        if any(v < 0 for v in values):
            continue
        existing = list(target.attrs.get("pads", [0, 0, 0, 0]))
        if len(existing) != 4:
            continue
        target.attrs["pads"] = [
            existing[0] + begins[2], existing[1] + begins[3],
            existing[2] + ends[2], existing[3] + ends[3],
        ]
        target.inputs[0] = op.inputs[0]
        folded.add(index)

    if folded:
        ag.ops = [op for i, op in enumerate(ag.ops) if i not in folded]
    return ag


_FUSABLE_PRODUCERS = frozenset(
    {"Conv", "DepthwiseConv", "Gemm", "Conv1D", "Add"}
)


_FUSABLE_ACTIVATIONS = frozenset({"Relu", "Relu6"})


def _absorb_activations(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Fuse Relu/Relu6 into preceding Conv/DepthwiseConv/Gemm/Conv1D.

    Sets ``attrs["fused_activation"]`` on the producer op, rewires outputs,
    removes the activation op and its intermediate tensor.
    """
    uses = _Uses(ag)
    removed: set[int] = set()

    for i, op in enumerate(ag.ops):
        if op.op_type not in _FUSABLE_ACTIVATIONS:
            continue

        act_input = op.inputs[0]
        act_output = op.outputs[0]

        # Check that the input comes from a fusable producer
        if act_input not in uses.producers:
            continue
        prod_idx = uses.producers[act_input]
        producer = ag.ops[prod_idx]

        if producer.op_type not in _FUSABLE_PRODUCERS:
            continue

        # Exactly one consumer (the activation), and not a model output.
        if uses.sole_reader(act_input) is None:
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

    return ag
