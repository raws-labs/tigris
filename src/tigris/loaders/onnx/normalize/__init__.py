"""ONNX-specific normalizations; the pass order is the body of ``normalize()``."""

from tigris.graph.ir import AnalyzedGraph
from tigris.loaders.onnx.normalize.adoption import _adopt_compiler_ops, detection_scratch_bytes
from tigris.loaders.onnx.normalize.attributes import (
    _ceil_mode_as_padding, _explicit_auto_pad, _extract_resize_scales, _lift_pad_operands,
    _mark_untileable_broadcasts, _normalize_arg_outputs, _normalize_reduction_axes,
    _order_broadcast_operands, _resolve_resize_conventions, _strip_metadata_inputs,
    _validate_transposes
)
from tigris.loaders.onnx.normalize.fusion import (
    _absorb_activations, _fold_bn, _fold_channel_bias_add, _fold_elementwise_patterns, _fold_gelu,
    _fold_pad_into_conv, _fold_split_into_its_weight
)
from tigris.loaders.onnx.normalize.layout import (
    _align_state_layouts, _assign_tensor_layouts, _gather_one_index_to_split, _lower_linear_matmul,
    _normalize_concat_axis, _relabel_shape_ops_to_reshape
)
from tigris.loaders.onnx.normalize.movement import _narrow_index_casts, _normalize_movement
from tigris.loaders.onnx.normalize.outputs import _restore_output_names
from tigris.loaders.onnx.normalize.quantization import (
    _fold_qdq, _quantize_float_biases, _quantize_inputs_first, _quantize_uniform_constants,
    _transpose_quantized_weights
)
from tigris.loaders.onnx.normalize.relabels import (
    _clip_to_relu6, _fold_div_constant_to_mul, _fold_float_square,
    _fold_gemm_scalars, _fold_sub_constant_to_add, _lower_legacy_softmax, _reduce_mean_to_gap,
    _relabel_conv1d, _relabel_depthwise, _relabel_matmul_to_gemm, _split_grouped_convs,
    _standalone_bn_as_affine, _whole_map_average_pool_to_gap
)
from tigris.loaders.onnx.normalize.static import (
    _drop_inference_identities, _drop_unreferenced_weights, _fold_static_subgraphs
)

__all__ = ["detection_scratch_bytes", "normalize"]


def normalize(ag: AnalyzedGraph, subgraph: bool = False) -> AnalyzedGraph:
    """Apply all normalization passes in sequence; ``subgraph`` for a graph an
    If or While runs."""
    declared_outputs = list(ag.model_outputs)
    transposed: dict[str, str] = {}
    ag = _adopt_compiler_ops(ag)
    ag = _normalize_arg_outputs(ag)
    # Names tigris::Gather and tigris::GatherND: _normalize_movement strips the prefix.
    ag = _narrow_index_casts(ag)
    ag = _drop_inference_identities(ag)
    ag = _lower_legacy_softmax(ag)
    ag = _fold_static_subgraphs(ag)
    ag = _fold_gelu(ag)
    ag = _fold_elementwise_patterns(ag)
    ag = _quantize_inputs_first(ag)
    ag = _transpose_quantized_weights(ag)
    ag = _fold_qdq(ag, subgraph)
    ag = _quantize_float_biases(ag)
    ag = _fold_float_square(ag)
    ag = _fold_gemm_scalars(ag)
    ag = _relabel_matmul_to_gemm(ag, transposed)
    ag = _fold_bn(ag)
    ag = _standalone_bn_as_affine(ag)
    ag = _fold_sub_constant_to_add(ag)
    ag = _fold_div_constant_to_mul(ag)
    ag = _quantize_uniform_constants(ag)
    ag = _fold_channel_bias_add(ag)
    ag = _explicit_auto_pad(ag)
    ag = _ceil_mode_as_padding(ag)
    ag = _fold_pad_into_conv(ag)
    ag = _relabel_depthwise(ag)
    ag = _split_grouped_convs(ag)
    ag = _relabel_conv1d(ag)
    ag = _clip_to_relu6(ag)
    ag = _normalize_reduction_axes(ag)
    ag = _reduce_mean_to_gap(ag)
    ag = _whole_map_average_pool_to_gap(ag)
    ag = _resolve_resize_conventions(ag)
    ag = _extract_resize_scales(ag)
    ag = _normalize_movement(ag)
    ag = _lift_pad_operands(ag)
    ag = _strip_metadata_inputs(ag)
    ag = _validate_transposes(ag)
    ag = _order_broadcast_operands(ag)
    ag = _absorb_activations(ag)
    # Reaches MatMul and the transB=0 Gemms that _relabel_matmul_to_gemm skipped.
    ag = _fold_split_into_its_weight(ag)
    ag = _assign_tensor_layouts(ag)
    ag = _align_state_layouts(ag)
    ag = _normalize_concat_axis(ag)
    ag = _mark_untileable_broadcasts(ag)
    ag = _gather_one_index_to_split(ag)
    ag = _relabel_shape_ops_to_reshape(ag)
    ag = _lower_linear_matmul(ag, transposed)
    ag = _drop_unreferenced_weights(ag)
    ag = _restore_output_names(ag, declared_outputs)
    for step, op in enumerate(ag.ops):
        op.step = step
    return ag
