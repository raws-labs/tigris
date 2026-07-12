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
11. **Output transpose validation**: Reject trailing ``Transpose`` ops until
    their permutation can be represented in the deployment contract.
12. **Activation absorption**: Fuse Relu/Relu6 into preceding
    Conv/DepthwiseConv/Gemm/Conv1D ops as ``fused_activation`` attr.
    Runs last so all relabeling and rewiring is already done.
"""

import numpy as np

from tigris.graph.ir import AnalyzedGraph, OpNode, QuantParam, TensorInfo


def normalize(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Apply all normalization passes in sequence."""
    ag = _fold_constant_ops(ag)
    ag = _fold_qdq(ag)
    ag = _fold_bn(ag)
    ag = _decompose_silu(ag)
    ag = _relabel_depthwise(ag)
    ag = _relabel_conv1d(ag)
    ag = _clip_to_relu6(ag)
    ag = _reduce_mean_to_gap(ag)
    ag = _fold_shape_ops(ag)
    ag = _extract_resize_scales(ag)
    ag = _normalize_concat_axis(ag)
    ag = _validate_output_transposes(ag)
    ag = _absorb_activations(ag)
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

    def _get_quant_param(op: OpNode) -> QuantParam | None:
        """Extract QuantParam from a QuantizeLinear or DequantizeLinear op.

        Inputs: x, y_scale, y_zero_point (optional).
        For DQL: x, x_scale, x_zero_point (optional).
        """
        if len(op.inputs) < 2:
            return None
        scale_name = op.inputs[1]
        if scale_name not in ag.weight_data:
            return None
        scale = ag.weight_data[scale_name].astype(np.float32).flatten()

        if len(op.inputs) >= 3 and op.inputs[2] and op.inputs[2] in ag.weight_data:
            zp = ag.weight_data[op.inputs[2]].flatten()
        else:
            zp = np.zeros_like(scale, dtype=np.int8)

        axis = op.attrs.get("axis", 1)
        if scale.size == 1:
            axis = -1  # per-tensor

        return QuantParam(scale=scale, zero_point=zp, axis=axis)

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
            qp = _get_quant_param(op)
            if qp is None:
                continue

            # Store QuantParam on the weight tensor
            if dql_input in ag.tensors:
                ag.tensors[dql_input].quant = qp

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
                qp = _get_quant_param(op)
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
            # Clean up scale/zp initializers from weight_data
            for inp_name in op.inputs[1:]:
                if inp_name and inp_name in ag.weight_data:
                    del ag.weight_data[inp_name]
                if inp_name and inp_name in ag.tensors:
                    del ag.tensors[inp_name]

    # Pass 2: Process activation Q/DQ pairs.
    # Pattern: activation -> QuantizeLinear -> int8 -> DequantizeLinear -> consumer
    # Defer scale/zp cleanup to avoid breaking shared constants (e.g. MaxPool
    # and Resize Q/DQ pairs share scale/zp with their input activation's Q/DQ).
    deferred_cleanup: set[str] = set()

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
        qp = _get_quant_param(op)
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
    """Replace Clip(min=0, max=6) with Relu6."""
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

        if min_val is not None and max_val is not None:
            if abs(min_val) < 1e-6 and abs(max_val - 6.0) < 1e-6:
                op.op_type = "Relu6"
                # Keep only the data input, drop min/max constant inputs
                op.inputs = [op.inputs[0]]
                op.attrs = {}

    return ag


def _reduce_mean_to_gap(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Replace ReduceMean(axes=[2,3]) with GlobalAveragePool.

    Newer ONNX exporters (PyTorch >= 2.x) emit ReduceMean over spatial
    dimensions instead of GlobalAveragePool.  They are semantically
    identical for 4-D NHWC/NCHW tensors, but the runtime only has a GAP
    kernel.

    Handles both attribute-based axes (opset < 18) and input-based axes
    (opset >= 18).
    """
    for op in ag.ops:
        if op.op_type != "ReduceMean":
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

        # Normalize negative axes for 4D tensors (ndim=4)
        axes_norm = set(int(a) % 4 for a in axes)

        # Accept axes {2,3} (NCHW spatial) or {1,2} (NHWC spatial)
        if axes_norm not in ({2, 3}, {1, 2}):
            continue

        op.op_type = "GlobalAveragePool"
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


def _validate_output_transposes(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Reject output Transpose ops whose observable permutation would be lost.

    The plan schema has no output-layout transform metadata and the runtime has
    no general Transpose kernel.  Removing a trailing Transpose therefore
    changes the model output name, shape, and element order.  Keep the graph
    fail-closed until such a permutation is explicit in the deployment
    contract rather than silently delegating it to unspecified host code.
    """
    output_to_op: dict[str, OpNode] = {}
    for op in ag.ops:
        for out in op.outputs:
            output_to_op[out] = op

    for out_name in ag.model_outputs:
        op = output_to_op.get(out_name)
        if op is None or op.op_type != "Transpose":
            continue

        input_name = op.inputs[0] if op.inputs else ""
        input_info = ag.tensors.get(input_name)
        output_info = ag.tensors.get(out_name)
        rank = len(input_info.shape) if input_info is not None else 0
        perm = op.attrs.get("perm")
        if perm is None and rank:
            perm = list(reversed(range(rank)))

        input_shape = input_info.shape if input_info is not None else "unknown"
        output_shape = output_info.shape if output_info is not None else "unknown"
        raise ValueError(
            f"Cannot preserve model output '{out_name}': trailing Transpose "
            f"'{op.name}' uses perm={perm}, changing shape {input_shape} to "
            f"{output_shape}. The deployment contract does not encode output "
            "permutations, so TiGrIS refuses to remove this operator."
        )

    return ag


# Activation fusion

_FUSABLE_PRODUCERS = frozenset({"Conv", "DepthwiseConv", "Gemm", "Conv1D"})
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
