#!/usr/bin/env python3
"""Generate one-operator TFLite models and their reference outputs.

Each case is converted to an int8 TFLite model and run on seeded inputs through
TFLite Micro and through TFLite's reference kernels. The recorded outputs are
TFLite Micro's; where TFLite Micro disagrees with the reference kernels, the
reference kernels' are recorded and the case is named as a deviation. Models and
outputs go to tests/fixtures/tflite/ops/ so the frontend tests need neither
package. Needs tensorflow and tflite-micro, which are not project dependencies:

    python scripts/gen_tflite_fixtures.py [case ...]
    python scripts/gen_tflite_fixtures.py --models
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import flatbuffers
import numpy as np
from flatbuffers import flexbuffers
import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime as micro
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tflite" / "ops"
SAMPLES = 4


def _weights(seed: int, *shape: int) -> tf.Tensor:
    return tf.constant(np.random.default_rng(seed).normal(0.0, 0.5, shape).astype(np.float32))


def _unary(fn, shape):
    return [shape], lambda x: fn(x)


def _binary(fn, a, b):
    return [a, b], lambda x, y: fn(x, y)


CASES = {
    "max_pool_valid": _unary(lambda x: tf.nn.max_pool2d(x, 2, 2, "VALID"), (1, 8, 8, 4)),
    "max_pool_same": _unary(lambda x: tf.nn.max_pool2d(x, 3, 2, "SAME"), (1, 9, 9, 4)),
    "avg_pool_valid": _unary(lambda x: tf.nn.avg_pool2d(x, 2, 2, "VALID"), (1, 8, 8, 4)),
    "avg_pool_same": _unary(lambda x: tf.nn.avg_pool2d(x, 3, 1, "SAME"), (1, 7, 7, 4)),
    "concat_channels": _binary(lambda x, y: tf.concat([x, y], -1), (1, 6, 6, 3), (1, 6, 6, 5)),
    "concat_rows": _binary(lambda x, y: tf.concat([x, y], 1), (1, 4, 6, 3), (1, 2, 6, 3)),
    "add_broadcast": _binary(tf.add, (1, 5, 5, 4), (1, 1, 1, 4)),
    "mul": _binary(tf.multiply, (1, 5, 5, 4), (1, 5, 5, 4)),
    "mul_broadcast": _binary(tf.multiply, (1, 5, 5, 4), (1, 1, 1, 4)),
    "sub": _binary(tf.subtract, (1, 5, 5, 4), (1, 5, 5, 4)),
    "logistic": _unary(tf.sigmoid, (1, 6, 6, 4)),
    "tanh": _unary(tf.tanh, (1, 6, 6, 4)),
    "hard_swish": _unary(lambda x: x * tf.nn.relu6(x + 3.0) / 6.0, (1, 6, 6, 4)),
    "relu": _unary(tf.nn.relu, (1, 6, 6, 4)),
    "relu6": _unary(tf.nn.relu6, (1, 6, 6, 4)),
    "mean_spatial": _unary(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True), (1, 6, 6, 4)),
    "mean_spatial_flat": _unary(lambda x: tf.reduce_mean(x, [1, 2]), (1, 6, 6, 4)),
    "mean_channels": _unary(lambda x: tf.reduce_mean(x, [3], keepdims=True), (1, 6, 6, 4)),
    "mean_rows": _unary(lambda x: tf.reduce_mean(x, [1]), (1, 6, 6, 4)),
    "mean_all": _unary(lambda x: tf.reduce_mean(x, [1, 2, 3]), (1, 6, 6, 4)),
    "resize_nearest": _unary(lambda x: tf.image.resize(x, (8, 8), "nearest"), (1, 4, 4, 3)),
    "resize_bilinear": _unary(lambda x: tf.image.resize(x, (8, 8), "bilinear"), (1, 4, 4, 3)),
    "resize_bilinear_asymmetric": _unary(
        lambda x: tf.compat.v1.image.resize_bilinear(x, (8, 8)), (1, 4, 4, 3)),
    "resize_nearest_asymmetric": _unary(
        lambda x: tf.compat.v1.image.resize_nearest_neighbor(x, (8, 12)), (1, 4, 4, 3)),
    "transpose": _unary(lambda x: tf.transpose(x, [0, 2, 1, 3]), (1, 4, 6, 3)),
    "split": _unary(lambda x: tf.split(x, 2, axis=-1), (1, 4, 4, 6)),
    "split_v": _unary(lambda x: tf.split(x, [2, 4], axis=-1), (1, 4, 4, 6)),
    "reshape": _unary(lambda x: tf.reshape(x, (1, 8, 12)), (1, 4, 4, 6)),
    "squeeze": _unary(lambda x: tf.squeeze(x, [1]), (1, 1, 8, 6)),
    "expand_dims": _unary(lambda x: tf.expand_dims(x, 1), (1, 8, 6)),
    "conv_dilated": _unary(
        lambda x: tf.nn.conv2d(x, _weights(1, 3, 3, 4, 6), 1, "SAME", dilations=2), (1, 9, 9, 4)),
    "depthwise_multiplier": _unary(
        lambda x: tf.nn.depthwise_conv2d(x, _weights(2, 3, 3, 4, 2), [1, 1, 1, 1], "SAME"),
        (1, 6, 6, 4)),
    "transpose_conv": _unary(
        lambda x: tf.nn.conv2d_transpose(x, _weights(3, 3, 3, 5, 4), (1, 8, 8, 5), 2, "SAME"),
        (1, 4, 4, 4)),
    "batch_matmul": _binary(tf.matmul, (1, 6, 8), (1, 8, 5)),
    "fully_connected_rank3": _unary(
        lambda x: tf.tensordot(x, _weights(4, 8, 5), 1), (1, 6, 8)),
    "pad_conv": _unary(
        lambda x: tf.nn.conv2d(tf.pad(x, [[0, 0], [1, 1], [2, 2], [0, 0]]),
                               _weights(5, 3, 3, 4, 4), 1, "VALID"), (1, 6, 6, 4)),
}


FLOAT_BOUNDARIES = {"float_boundaries": _unary(lambda x: tf.nn.relu(x) * x, (1, 6, 6, 4))}
for _kind, _resize in (("nearest", tf.raw_ops.ResizeNearestNeighbor), ("bilinear", tf.raw_ops.ResizeBilinear)):
    for _coordinate in ("asymmetric", "half_pixel", "align_corners"):
        CASES[f"resize_{_kind}_{_coordinate}_down"] = _unary(
            lambda x, fn=_resize, mode=_coordinate: fn(
                images=x, size=(7, 5), align_corners=mode == "align_corners", half_pixel_centers=mode == "half_pixel"),
            (1, 17, 9, 4))
CASES.update(FLOAT_BOUNDARIES)

_MAP = (1, 6, 6, 4)
CASES.update({
    "abs": _unary(tf.abs, _MAP),
    "rsqrt": _unary(tf.math.rsqrt, _MAP),
    "squared_difference": _binary(tf.math.squared_difference, _MAP, _MAP),
    "maximum": _binary(tf.maximum, _MAP, _MAP),
    "minimum": _binary(tf.minimum, _MAP, _MAP),
    "div": _binary(tf.divide, _MAP, _MAP),
})
_CHANNELS = np.linspace(-1.5, 2.0, 4).astype(np.float32)
_DIVISORS = np.linspace(0.6, 2.4, 4).astype(np.float32)
_FULL = np.random.default_rng(7).uniform(-2.0, 2.0, (1, 6, 6, 4)).astype(np.float32)
CASES.update({
    "add_constant_channels": _unary(lambda x: x + _CHANNELS, _MAP),
    "add_constant_full": _unary(lambda x: x + _FULL, _MAP),
    "mul_constant_scalar": _unary(lambda x: x * 0.37, _MAP),
    "sub_constant_second": _unary(lambda x: x - _CHANNELS, _MAP),
    "sub_constant_first": _unary(lambda x: 1.25 - x, _MAP),
    "div_constant": _unary(lambda x: x / _DIVISORS, _MAP),
    "squared_difference_constant": _unary(
        lambda x: tf.math.squared_difference(x, _CHANNELS), _MAP),
})
_ROWS = np.linspace(-1.0, 1.5, 6).reshape(1, 6, 1, 1).astype(np.float32)
CASES.update({
    "add_broadcast_rows": _binary(tf.add, _MAP, (1, 6, 1, 4)),
    "add_rank": _binary(tf.add, _MAP, (6, 4)),
    "mul_rank_vector": _binary(tf.multiply, _MAP, (4,)),
    "maximum_rank": _binary(tf.maximum, (6, 1, 4), _MAP),
    "less_rank": _binary(tf.less, _MAP, (6, 1, 4)),
    "select_v2_rank": _binary(lambda x, y: tf.where(x > 0.5, x, y), _MAP, (4,)),
    "mul_broadcast_both": _binary(tf.multiply, (1, 6, 1, 4), (1, 1, 6, 4)),
    "sub_broadcast_first": _binary(tf.subtract, (1, 1, 6, 4), _MAP),
    "add_constant_rows": _unary(lambda x: x + _ROWS, _MAP),
})
CASES.update({
    "pad": _unary(lambda x: tf.pad(x, [[0, 0], [1, 2], [0, 1], [0, 0]]), _MAP),
    "padv2": _unary(lambda x: tf.pad(x, [[0, 0], [1, 1], [2, 0], [0, 0]], constant_values=0.75),
                    _MAP),
})
# Converted without quantization: TFLite Micro runs these in float only.
FLOAT_MODELS = {
    "float_abs": _unary(tf.abs, _MAP),
    "float_neg": _unary(tf.negative, _MAP),
    "float_exp": _unary(tf.exp, _MAP),
    "float_log": _unary(tf.math.log, _MAP),
    "float_sqrt": _unary(tf.sqrt, _MAP),
    "float_rsqrt": _unary(tf.math.rsqrt, _MAP),
    "float_square": _unary(tf.square, _MAP),
    "float_squared_difference": _binary(tf.math.squared_difference, _MAP, _MAP),
    "float_div": _binary(tf.divide, _MAP, _MAP),
    "float_maximum": _binary(tf.maximum, _MAP, _MAP),
    "float_minimum": _binary(tf.minimum, _MAP, _MAP),
    "float_floor": _unary(tf.floor, _MAP),
    "float_ceil": _unary(tf.math.ceil, _MAP),
    "float_round": _unary(tf.round, _MAP),
    "float_sin": _unary(tf.sin, _MAP),
    "float_cos": _unary(tf.cos, _MAP),
    "float_floor_div": _binary(tf.math.floordiv, _MAP, _MAP),
    "float_floor_mod": _binary(tf.math.floormod, _MAP, _MAP),
    "float_div_constant_first": _unary(lambda x: 2.5 / x, _MAP),
    "float_floor_mod_constant": _unary(lambda x: tf.math.floormod(x, _DIVISORS), _MAP),
    "float_maximum_constant": _unary(lambda x: tf.maximum(x, _CHANNELS), _MAP),
    "float_div_broadcast": _binary(tf.divide, _MAP, (1, 1, 6, 4)),
    "float_maximum_broadcast": _binary(tf.maximum, (1, 6, 1, 4), (1, 1, 6, 4)),
}
CASES.update(FLOAT_MODELS)
# Input ranges per operand where the default [-3, 3] leaves the domain.
_POSITIVE, _DIVISOR = (0.05, 4.0), (0.5, 3.0)
RANGES = {
    "rsqrt": [_POSITIVE], "float_rsqrt": [_POSITIVE], "float_log": [_POSITIVE],
    "float_sqrt": [(0.0, 4.0)],
    **{name: [(-1.0, 1.0), (0.0, 1.0)] for name in ("float_detection_fast", "float_detection_regular",
                                                    "detection_fast", "detection_regular")},
    "div": [(-3.0, 3.0), _DIVISOR], "float_div": [(-3.0, 3.0), _DIVISOR], "float_floor_div": [(-3.0, 3.0), _DIVISOR],
    "float_floor_mod": [(-3.0, 3.0), _DIVISOR], "float_div_constant_first": [_DIVISOR],
    "float_div_broadcast": [(-3.0, 3.0), _DIVISOR],
    # Non-negative inputs put the int8 input zero point near -128.
    "cumsum_offset": [(0.0, 3.0)], "cumsum_offset_exclusive_reverse": [(0.0, 3.0)],
    "float_while": [(0.5, 3.0)], "float_while_two_variables": [(0.5, 3.0), (0.5, 3.0)],
    "float_while_feature_map": [(0.5, 3.0)],
    "float_if_if": [(-2.8, 2.5)],
    # Past both of the float logistic's cutoffs, -9 and about 16.6.
    "float_logistic_tails": [(-24.0, 24.0)],
}


def _as_operator(kind: str):
    """Rewrites the converter's one RESHAPE into SQUEEZE or EXPAND_DIMS, which
    the converter never emits itself, keeping its tensors."""
    def rewrite(model: bytes) -> bytes:
        tree = schema.ModelT.InitFromPackedBuf(model, 0)
        graph = tree.subgraphs[0]
        op = graph.operators[0]
        code = tree.operatorCodes[op.opcodeIndex]
        code.builtinCode = getattr(schema.BuiltinOperator, kind)
        code.deprecatedBuiltinCode = min(code.builtinCode, 127)
        code.version = 1
        source = graph.tensors[op.inputs[0]].shape
        target = graph.tensors[op.outputs[0]].shape
        if kind == "SQUEEZE":
            op.inputs = op.inputs[:1]
            op.builtinOptionsType = schema.BuiltinOptions.SqueezeOptions
            op.builtinOptions = schema.SqueezeOptionsT()
            op.builtinOptions.squeezeDims = [i for i, d in enumerate(source) if d == 1][:1]
        else:
            axis = next(i for i, (a, b) in enumerate(zip(list(source) + [None], target)) if a != b)
            buffer = schema.BufferT()
            buffer.data = np.array([axis], np.int32).tobytes()
            tree.buffers.append(buffer)
            tensor = schema.TensorT()
            tensor.shape = np.array([1], np.int32)
            tensor.type = schema.TensorType.INT32
            tensor.buffer = len(tree.buffers) - 1
            tensor.name = b"axis"
            graph.tensors.append(tensor)
            op.inputs = np.array([op.inputs[0], len(graph.tensors) - 1], np.int32)
            op.builtinOptionsType = schema.BuiltinOptions.ExpandDimsOptions
            op.builtinOptions = schema.ExpandDimsOptionsT()
        builder = flatbuffers.Builder(4096)
        builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
        return bytes(builder.Output())
    return rewrite


REWRITES = {"squeeze_op": _as_operator("SQUEEZE"), "expand_dims_op": _as_operator("EXPAND_DIMS")}
CASES.update({
    "squeeze_op": _unary(lambda x: tf.squeeze(x, [1]), (1, 1, 8, 6)),
    "expand_dims_op": _unary(lambda x: tf.expand_dims(x, 1), (1, 8, 6)),
})

CASES["softmax"] = _unary(tf.nn.softmax, (1, 10))
CASES.update({
    "pack": _binary(lambda x, y: tf.stack([x, y], axis=1), (1, 4, 6), (1, 4, 6)),
    "pack_outer": _binary(lambda x, y: tf.stack([x, y], axis=0), (1, 4), (1, 4)),
    "concat_outer": _binary(lambda x, y: tf.concat([x, y], 0), (2, 3, 4), (1, 3, 4)),
    "concat_rank2": _binary(lambda x, y: tf.concat([x, y], 1), (3, 4), (3, 2)),
    "unpack": _unary(lambda x: tf.unstack(x, axis=1), (1, 3, 4, 6)),
    "slice": _unary(lambda x: tf.slice(x, [0, 1, 2, 0], [1, 3, 3, 4]), _MAP),
    "strided_slice": _unary(lambda x: x[:, 1:5, :, 1:3], _MAP),
    "strided_slice_shrink": _unary(lambda x: x[:, 2, :, :], _MAP),
    "gather": _unary(lambda x: tf.gather(x, [2, 3, 4], axis=2), _MAP),
    "gather_scalar": _unary(lambda x: tf.gather(x, 1, axis=3), _MAP),
    "space_to_depth": _unary(lambda x: tf.nn.space_to_depth(x, 2), (1, 4, 6, 3)),
    "depth_to_space": _unary(lambda x: tf.nn.depth_to_space(x, 2), (1, 3, 3, 8)),
    "space_to_batch": _unary(
        lambda x: tf.space_to_batch_nd(x, [2, 2], [[1, 1], [0, 2]]), (1, 4, 4, 3)),
    "batch_to_space": _unary(
        lambda x: tf.batch_to_space(x, [2, 2], [[0, 1], [1, 0]]), (4, 3, 3, 3)),
    "broadcast_to": _unary(lambda x: tf.broadcast_to(x, (1, 6, 6, 4)), (1, 1, 6, 4)),
})
_PRELU = tf.keras.layers.PReLU(alpha_initializer=tf.constant_initializer([0.1, -0.3, 0.6, 1.4]),
                               shared_axes=[1, 2])
_PRELU.build((None, 6, 6, 4))
ACTIVATIONS = {
    "leaky_relu": _unary(lambda x: tf.nn.leaky_relu(x, 0.2), _MAP),
    "prelu": _unary(_PRELU, _MAP),
    "elu": _unary(tf.nn.elu, _MAP),
    "log_softmax": _unary(tf.nn.log_softmax, (1, 10)),
    "log_softmax_channels": _unary(tf.nn.log_softmax, _MAP),
    "l2_normalization": _unary(lambda x: tf.math.l2_normalize(x, -1), _MAP),
    "reduce_max_channels": _unary(lambda x: tf.reduce_max(x, -1, keepdims=True), _MAP),
    "reduce_max_spatial": _unary(lambda x: tf.reduce_max(x, [1, 2]), _MAP),
    "reduce_min_rows": _unary(lambda x: tf.reduce_min(x, 1), _MAP),
    "sum_channels": _unary(lambda x: tf.reduce_sum(x, -1), _MAP),
    "sum_spatial": _unary(lambda x: tf.reduce_sum(x, [1, 2], keepdims=True), _MAP),
    "cumsum": _unary(lambda x: tf.math.cumsum(x, 2), _MAP),
    "cumsum_exclusive_reverse": _unary(
        lambda x: tf.math.cumsum(x, -1, exclusive=True, reverse=True), _MAP),
    "cumsum_offset": _unary(lambda x: tf.math.cumsum(x, 1), _MAP),
    "cumsum_offset_exclusive_reverse": _unary(
        lambda x: tf.math.cumsum(x, 2, exclusive=True, reverse=True), _MAP),
}
CASES.update(ACTIVATIONS)
# TFLite Micro runs L2_POOL_2D in float only; the converter emits it for no
# TensorFlow operation, so an AVERAGE_POOL_2D is recoded with its options kept.
CASES["float_l2_pool"] = CASES["avg_pool_same"]


# TFLite's reference int8 SUM does not requantize its result in this release
# (a spatial sum comes out -1 everywhere); TFLite Micro matches the exact sum.
BROKEN_REFERENCE = {"sum_channels", "sum_spatial"}


def _recode(kind: str):
    def rewrite(model: bytes) -> bytes:
        tree = schema.ModelT.InitFromPackedBuf(model, 0)
        code = tree.operatorCodes[tree.subgraphs[0].operators[0].opcodeIndex]
        code.builtinCode = getattr(schema.BuiltinOperator, kind)
        code.deprecatedBuiltinCode = min(code.builtinCode, 127)
        code.version = 1
        builder = flatbuffers.Builder(4096)
        builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
        return bytes(builder.Output())
    return rewrite


REWRITES["float_l2_pool"] = _recode("L2_POOL_2D")


def _int8_island(centered: bool = False, shared: bool = False):
    """The converter keeps ELU, CUMSUM, DYNAMIC_UPDATE_SLICE, NOT_EQUAL and
    ADD_N in float after DEQUANTIZEs, and before a QUANTIZE where the result
    is not bool; this runs the operator itself on the int8 tensors, as
    TFLite Micro registers it. `centered` moves the input
    zero point to 0; `shared` gives every data tensor the first input's
    quantization, as a kernel that copies raw bytes requires."""
    def rewrite(model: bytes) -> bytes:
        tree = schema.ModelT.InitFromPackedBuf(model, 0)
        graph = tree.subgraphs[0]
        operators = list(graph.operators)
        quantize = operators.pop() if tree.operatorCodes[operators[-1].opcodeIndex].builtinCode \
            == schema.BuiltinOperator.QUANTIZE else None
        *dequantizes, op = operators
        sources = {d.outputs[0]: d.inputs[0] for d in dequantizes}
        op.inputs = np.array([sources.get(i, i) for i in op.inputs], np.int32)
        if quantize is not None:
            op.outputs = np.array(quantize.outputs, np.int32)
        graph.operators = [op]
        first = graph.tensors[op.inputs[0]].quantization
        if centered:
            first.zeroPoint = np.zeros(1, np.int64)
        if shared:
            for index in [*sources.values(), *op.outputs]:
                graph.tensors[index].quantization.scale = np.array(first.scale)
                graph.tensors[index].quantization.zeroPoint = np.array(first.zeroPoint)
        builder = flatbuffers.Builder(4096)
        builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
        return bytes(builder.Output())
    return rewrite


try:
    from tensorflow.compiler.tf2xla.python import xla as _xla
except ImportError:
    _xla = None
INDEXING = {
    "arg_max": _unary(lambda x: tf.argmax(x, -1, output_type=tf.int32), (1, 10)),
    "arg_min_rows": _unary(lambda x: tf.argmin(x, 1, output_type=tf.int32), (1, 6, 4)),
    "arg_max_channels": _unary(lambda x: tf.argmax(x, -1, output_type=tf.int32), _MAP),
    "gather_indices": _unary(lambda x: tf.gather(x, [4, 0, 4, 2], axis=2), _MAP),
    "gather_matrix": _unary(lambda x: tf.gather(x, [[0, 2], [5, 5]], axis=1), _MAP),
    "gather_nd": _unary(lambda x: tf.gather_nd(x, [[0, 1], [0, 4], [0, 1]]), _MAP),
    "gather_batch": _unary(lambda x: tf.gather(x, [[4, 0, 4], [1, 2, 3]], axis=1, batch_dims=1),
                           (2, 5, 4)),
    "gather_batch_matrix": _unary(
        lambda x: tf.gather(x, [[[1, 0], [2, 2]], [[0, 3], [3, 1]]], axis=2, batch_dims=1),
        (2, 3, 4, 2)),
    "strided_slice_steps": _unary(lambda x: x[:, ::2, 5:0:-2, :], _MAP),
    "mirror_pad_reflect": _unary(
        lambda x: tf.pad(x, [[0, 0], [1, 2], [2, 1], [0, 0]], "REFLECT"), _MAP),
    "mirror_pad_symmetric": _unary(
        lambda x: tf.pad(x, [[0, 0], [2, 0], [1, 3], [0, 0]], "SYMMETRIC"), _MAP),
    "reverse_spatial": _unary(lambda x: tf.reverse(x, [1, 2]), _MAP),
    "reverse_channels": _unary(lambda x: tf.reverse(x, [3]), _MAP),
    "embedding_lookup": _unary(lambda x: tf.gather(x, [3, 0, 3, 5]), (6, 4)),
}
if _xla is not None:
    INDEXING["dynamic_update_slice"] = _binary(
        lambda x, u: _xla.dynamic_update_slice(x, u, tf.constant([0, 3, 1, 0])), _MAP, (1, 2, 3, 4))
# Indices supplied at run time: int32 inputs drawn from [low, high), keyed by
# case and input position. Dynamic update starts reach past both ends, where
# TFLite clamps them.
INDEX_INPUTS = {"gather_runtime": {1: (0, 6)}, "gather_nd_runtime": {1: (0, 6)},
                "embedding_lookup_runtime": {1: (0, 6)},
                "dynamic_update_slice_runtime": {2: (-2, 7)}, "if_int32": {1: (-3, 6)}}
INDEXING.update({
    "gather_runtime": ([_MAP, (3,)], lambda x, i: tf.gather(x, i, axis=2)),
    "gather_nd_runtime": ([(6, 6, 4), (3, 2)], lambda x, i: tf.gather_nd(x, i)),
    "embedding_lookup_runtime": ([(6, 4), (4,)], lambda x, i: tf.gather(x, i)),
    "arg_max_gather": _unary(lambda x: tf.gather(x, tf.argmax(x, 0, output_type=tf.int32)),
                             (6, 4)),
})
if _xla is not None:
    INDEXING["dynamic_update_slice_runtime"] = (
        [_MAP, (1, 2, 3, 4), (4,)], lambda x, u, s: _xla.dynamic_update_slice(x, u, s))
# A Keras LSTM unrolled over its time steps, which the converter writes as
# plain operators; seeded weights keep the case reproducible.
_LSTM = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True,
                             kernel_initializer=tf.keras.initializers.RandomNormal(seed=1),
                             recurrent_initializer=tf.keras.initializers.RandomNormal(seed=2),
                             bias_initializer="ones")
_LSTM.build((1, 3, 2))
INDEXING["lstm_unrolled"] = _unary(_LSTM, (1, 3, 2))
# int32 arithmetic beside int8 data: indices offset at run time, and an int32
# comparison choosing between two int8 tensors.
INDEXING.update({
    "gather_index_arithmetic": ([(6, 4), (3,)], lambda x, i: tf.gather(x, i * 2 - 1)),
    "select_int32_condition": ([_MAP, (1, 1, 1, 1)], lambda x, k: tf.where(
        k > 2, x, tf.reverse(x, [2]))),
})
INDEX_INPUTS.update({"gather_index_arithmetic": {1: (1, 4)}, "select_int32_condition": {1: (0, 6)}})
# Weighted operators with a non-zero bias, which TFLite adds after the sum.
INDEXING.update({
    "conv_bias": _unary(lambda x: tf.nn.conv2d(x, _weights(5, 3, 3, 4, 6), 1, "SAME")
                        + _weights(6, 6), (1, 7, 7, 4)),
    "depthwise_bias": _unary(lambda x: tf.nn.depthwise_conv2d(
        x, _weights(7, 3, 3, 4, 1), [1, 1, 1, 1], "SAME") + _weights(8, 4), (1, 7, 7, 4)),
    "fully_connected_bias": _unary(lambda x: tf.matmul(x, _weights(9, 8, 5)) + _weights(10, 5),
                                   (3, 8)),
    # A per-channel convolution feeding a byte copy that needs equal encodings.
    "conv_split": _unary(lambda x: tf.split(tf.nn.conv2d(x, _weights(11, 3, 3, 4, 6), 1, "SAME"),
                                            [2, 4], -1), (1, 6, 6, 4)),
})


class _Window(tf.Module):
    """A sliding window held in a variable across invocations, as an SVDF
    keeps its activation state; each call shifts in one projected column."""

    def __init__(self):
        super().__init__()
        self.feature = tf.constant(np.linspace(-1.0, 1.0, 8).reshape(2, 4).astype(np.float32))
        self.time = tf.constant(np.linspace(0.5, -0.5, 12).reshape(4, 3).astype(np.float32))
        self.state = tf.Variable(tf.zeros((1, 4, 3)), trainable=False)

    def __call__(self, x):
        window = tf.concat([self.state[:, :, 1:], tf.expand_dims(tf.matmul(x, self.feature), 2)], 2)
        self.state.assign(window)
        return tf.reduce_sum(window * self.time, 2)


_WINDOW = _Window()
# The int8 conversion keeps the variable in float, so the case is float only.
# Cases whose model holds variables convert against the object that owns them.
TRACKABLES = {"variable_window": _WINDOW}
CASES.update(INDEXING)

# Comparisons end in bool outputs; the logical, select and cast operators take
# bool operands, so a comparison produces them inside the model.
_TRIPLE = ([_MAP, _MAP, _MAP], lambda x, y, z: tf.add_n([x, y, z]))
BOOLEAN = {
    "equal": _binary(tf.equal, _MAP, _MAP),
    "not_equal": _binary(tf.not_equal, _MAP, _MAP),
    "less": _binary(tf.less, _MAP, (1, 1, 1, 4)),
    "less_equal": _binary(tf.less_equal, _MAP, _MAP),
    "greater_constant": _unary(lambda x: tf.greater(x, _CHANNELS), _MAP),
    "greater_equal": _binary(tf.greater_equal, _MAP, _MAP),
    "logical_and": _binary(lambda x, y: tf.logical_and(x > 0.5, y > -0.5), _MAP, _MAP),
    "logical_or": _binary(lambda x, y: tf.logical_or(x > 0.5, y < -0.5), _MAP, _MAP),
    "logical_not": _binary(lambda x, y: tf.logical_not(tf.logical_or(x > 0.5, y < -0.5)),
                           _MAP, _MAP),
    "select_v2": _binary(lambda x, y: tf.where(x > y, x, y), _MAP, _MAP),
    "select_v2_broadcast": _binary(lambda x, y: tf.where(x > 0.5, x, y), _MAP, (1, 1, 1, 4)),
    "cast_bool": _binary(lambda x, y: tf.cast(x > y, tf.float32), _MAP, _MAP),
    "add_n": _TRIPLE,
    "reduce_all_channels": _unary(lambda x: tf.reduce_all(x > -2.0, -1), _MAP),
    "reduce_all_spatial": _unary(lambda x: tf.reduce_all(x > -2.9, [1, 2], keepdims=True), _MAP),
}
CASES.update(BOOLEAN)


def _select_v2(model: bytes) -> bytes:
    """The converter's SELECT for operands of one shape, recoded as the
    SELECT_V2 TFLite Micro registers; the two agree without broadcasting."""
    tree = schema.ModelT.InitFromPackedBuf(model, 0)
    for code in tree.operatorCodes:
        if code.builtinCode == schema.BuiltinOperator.SELECT:
            code.builtinCode = schema.BuiltinOperator.SELECT_V2
            code.deprecatedBuiltinCode = min(code.builtinCode, 127)
            code.version = 1
    builder = flatbuffers.Builder(4096)
    builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


REWRITES["select_v2"] = _select_v2


def _embedding_lookup(model: bytes) -> bytes:
    """GATHER on axis 0 recoded as EMBEDDING_LOOKUP, which takes the ids first
    and the table second; the converter emits it for no TensorFlow operation."""
    tree = schema.ModelT.InitFromPackedBuf(model, 0)
    graph = tree.subgraphs[0]
    op = graph.operators[0]
    code = tree.operatorCodes[op.opcodeIndex]
    code.builtinCode = schema.BuiltinOperator.EMBEDDING_LOOKUP
    code.deprecatedBuiltinCode = code.builtinCode
    code.version = 1
    op.inputs = np.array([op.inputs[1], op.inputs[0]], np.int32)
    op.builtinOptionsType = schema.BuiltinOptions.NONE
    op.builtinOptions = None
    builder = flatbuffers.Builder(4096)
    builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


REWRITES["embedding_lookup"] = _embedding_lookup
REWRITES["embedding_lookup_runtime"] = _embedding_lookup
def _one_operator(code, options_type, options, tensors, op_inputs, op_outputs, inputs, outputs):
    """A model of one builtin operator. `tensors` are (shape, type, data, scale,
    zero point, variable) records; data None marks an activation."""
    return _operators([(code, options_type, options, op_inputs, op_outputs)], tensors, inputs, outputs)


def _operators(operators, tensors, inputs, outputs):
    """A model of operators in order, each (code, options type, options,
    inputs, outputs); `tensors` as for _one_operator. A custom operator's code
    is its name and its options are the FlexBuffer bytes."""
    tree = schema.ModelT()
    tree.version = 3
    tree.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.tensors = []
    for i, (shape, kind, data, scale, zero_point, variable) in enumerate(tensors):
        tensor = schema.TensorT()
        tensor.name = f"t{i}".encode()
        tensor.shape = np.asarray(shape, np.int32)
        tensor.type = kind
        tensor.isVariable = variable
        tensor.buffer = len(tree.buffers)
        buffer = schema.BufferT()
        if data is not None:
            buffer.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), np.uint8)
        tree.buffers.append(buffer)
        if scale is not None:
            quant = schema.QuantizationParametersT()
            quant.scale = np.asarray([scale], np.float32)
            quant.zeroPoint = np.asarray([zero_point], np.int64)
            tensor.quantization = quant
        graph.tensors.append(tensor)
    codes = []
    graph.operators = []
    for code, options_type, options, op_inputs, op_outputs in operators:
        if code not in codes:
            codes.append(code)
        op = schema.OperatorT()
        op.opcodeIndex = codes.index(code)
        op.inputs = np.asarray(op_inputs, np.int32)
        op.outputs = np.asarray(op_outputs, np.int32)
        if isinstance(code, str):
            op.customOptions = np.frombuffer(options, np.uint8)
        else:
            op.builtinOptionsType = options_type
            op.builtinOptions = options
        graph.operators.append(op)
    graph.inputs = np.asarray(inputs, np.int32)
    graph.outputs = np.asarray(outputs, np.int32)
    tree.subgraphs = [graph]
    tree.operatorCodes = []
    for code in codes:
        opcode = schema.OperatorCodeT()
        if isinstance(code, str):
            opcode.customCode = code.encode()
            code = schema.BuiltinOperator.CUSTOM
        opcode.builtinCode = code
        opcode.deprecatedBuiltinCode = min(code, 127)
        opcode.version = 1
        tree.operatorCodes.append(opcode)
    builder = flatbuffers.Builder(4096)
    builder.Finish(tree.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def _quantize(values, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(np.round(values / scale), info.min, info.max).astype(dtype)


def _svdf(batch, features, units, rank, memory, activation, quantized=False):
    """SVDF on a [batch, features] input, its state a variable tensor; seeded
    weights. The int8 form keeps its state and time weights in int16. TFLite
    Micro's Prepare dereferences the bias, so every case has one."""
    rng = np.random.default_rng(features * 100 + units * 10 + rank)
    filters = units * rank
    feature = rng.normal(0.0, 0.4, (filters, features)).astype(np.float32)
    time = rng.normal(0.0, 0.4, (filters, memory)).astype(np.float32)
    offsets = rng.normal(0.0, 0.3, (units,)).astype(np.float32)
    options = schema.SVDFOptionsT()
    options.rank = rank
    options.fusedActivationFunction = activation
    T = schema.TensorType
    if not quantized:
        tensors = [((batch, features), T.FLOAT32, None, None, 0, False),
                   ((filters, features), T.FLOAT32, feature, None, 0, False),
                   ((filters, memory), T.FLOAT32, time, None, 0, False),
                   ((units,), T.FLOAT32, offsets, None, 0, False),
                   ((batch, filters * memory), T.FLOAT32, None, None, 0, True),
                   ((batch, units), T.FLOAT32, None, None, 0, False)]
    else:
        x_scale, state_scale, out_scale = 6.0 / 255, 8.0 / 32767, 0.06
        f_scale = float(np.abs(feature).max()) / 127
        t_scale = float(np.abs(time).max()) / 32767
        b_scale = float(np.float32(state_scale) * np.float32(t_scale))
        tensors = [((batch, features), T.INT8, None, x_scale, 2, False),
                   ((filters, features), T.INT8, _quantize(feature, f_scale, np.int8), f_scale, 0, False),
                   ((filters, memory), T.INT16, _quantize(time, t_scale, np.int16), t_scale, 0, False),
                   ((units,), T.INT32, _quantize(offsets, b_scale, np.int32), b_scale, 0, False),
                   ((batch, filters * memory), T.INT16, None, state_scale, 0, True),
                   ((batch, units), T.INT8, None, out_scale, -3, False)]
    op_inputs = [0, 1, 2, 3, 4]
    return _one_operator(schema.BuiltinOperator.SVDF, schema.BuiltinOptions.SVDFOptions,
                         options, tensors, op_inputs, [5], [0], [5])


def _lstm(time_major, batch, steps, features, units, cell_clip=0.0, quantized=False):
    """UNIDIRECTIONAL_SEQUENCE_LSTM without peepholes, projection or layer
    normalization, which TFLite Micro does not run; its hidden and cell states
    are variable tensors. Seeded weights, gates in TFLite's order i, f, c, o.
    The int8 form keeps its cell in int16 at a power-of-two scale."""
    rng = np.random.default_rng(steps * 100 + features * 10 + units)
    T = schema.TensorType
    shape = (steps, batch, features) if time_major else (batch, steps, features)
    out = (steps, batch, units) if time_major else (batch, steps, units)
    x_scale, hidden_scale, hidden_zero = 6.0 / 255, 1.0 / 128, 3
    if quantized:
        tensors = [(shape, T.INT8, None, x_scale, 2, False)]
    else:
        tensors = [(shape, T.FLOAT32, None, None, 0, False)]
    input_scales = []
    for columns in (features, units):
        for _ in range(4):
            w = rng.normal(0.0, 0.5, (units, columns)).astype(np.float32)
            if quantized:
                scale = float(np.abs(w).max()) / 127
                input_scales.append(scale)
                tensors.append(((units, columns), T.INT8, _quantize(w, scale, np.int8), scale, 0, False))
            else:
                tensors.append(((units, columns), T.FLOAT32, w, None, 0, False))
    for gate in range(4):
        b = rng.normal(0.0, 0.3, (units,)).astype(np.float32)
        if quantized:
            scale = float(np.float32(x_scale) * np.float32(input_scales[gate]))
            tensors.append(((units,), T.INT32, _quantize(b, scale, np.int32), scale, 0, False))
        else:
            tensors.append(((units,), T.FLOAT32, b, None, 0, False))
    if quantized:
        tensors += [((batch, units), T.INT8, None, hidden_scale, hidden_zero, True),
                    ((batch, units), T.INT16, None, 2.0 ** -11, 0, True),
                    (out, T.INT8, None, hidden_scale, hidden_zero, False)]
    else:
        tensors += [((batch, units), T.FLOAT32, None, None, 0, True),
                    ((batch, units), T.FLOAT32, None, None, 0, True),
                    (out, T.FLOAT32, None, None, 0, False)]
    options = schema.UnidirectionalSequenceLSTMOptionsT()
    options.fusedActivationFunction = schema.ActivationFunctionType.TANH
    options.cellClip = cell_clip
    options.timeMajor = time_major
    op_inputs = [0, *range(1, 9), -1, -1, -1, *range(9, 13), -1, -1, 13, 14, -1, -1, -1, -1]
    return _one_operator(schema.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                         schema.BuiltinOptions.UnidirectionalSequenceLSTMOptions, options,
                         tensors, op_inputs, [15], [0], [15])


def _shape_ops(case):
    """Shape and fill operators the converter folds in static models, ahead of
    an ADD or MUL with the input x [1, 4]."""
    B, T = schema.BuiltinOperator, schema.TensorType

    def act(shape, kind=T.FLOAT32):
        return shape, kind, None, None, 0, False

    def const(array, kind):
        return array.shape, kind, array, None, 0, False

    add = (B.ADD, schema.BuiltinOptions.NONE, None)
    if case == "zeros_like":
        tensors = [act((1, 4)), act((1, 4)), act((1, 4))]
        ops = [(B.ZEROS_LIKE, schema.BuiltinOptions.NONE, None, [0], [1]), (*add, [0, 1], [2])]
    elif case == "fill":
        tensors = [act((1, 4)), const(np.asarray([1, 4], np.int32), T.INT32),
                   const(np.asarray(0.5, np.float32), T.FLOAT32), act((1, 4)), act((1, 4))]
        ops = [(B.FILL, schema.BuiltinOptions.NONE, None, [1, 2], [3]),
               (B.MUL, schema.BuiltinOptions.NONE, None, [0, 3], [4])]
    elif case == "shape_reshape":
        # TFLite Micro's FILL takes constant dims only; RESHAPE reads its shape.
        options = schema.ShapeOptionsT()
        options.outType = T.INT32
        tensors = [act((1, 4)), act((2,), T.INT32), act((1, 4)), act((1, 4))]
        ops = [(B.SHAPE, schema.BuiltinOptions.ShapeOptions, options, [0], [1]),
               (B.RESHAPE, schema.BuiltinOptions.NONE, None, [0, 1], [2]), (*add, [0, 2], [3])]
    else:  # broadcast_args
        tensors = [act((4,)), const(np.asarray([1, 4], np.int32), T.INT32),
                   const(np.asarray([4], np.int32), T.INT32), act((2,), T.INT32), act((1, 4))]
        # TFLite Micro's BROADCAST_TO takes a constant shape; RESHAPE reads it.
        ops = [(B.BROADCAST_ARGS, schema.BuiltinOptions.NONE, None, [1, 2], [3]),
               (B.RESHAPE, schema.BuiltinOptions.NONE, None, [0, 3], [4])]
    return _operators(ops, tensors, [0], [len(tensors) - 1])


def _detection(regular, quantized=False, boxes=16, classes=3, max_detections=8):
    """TFLite_Detection_PostProcess over an SSD head of `boxes` anchors and
    `classes` classes after a background column; seeded anchors on an
    overlapping grid. TFLite Micro reads float operands only, so the int8 form
    dequantizes the box encodings and the scores in front of it."""
    T = schema.TensorType
    rng = np.random.default_rng(boxes * 31 + classes + 2 * regular)
    centers = rng.uniform(0.35, 0.65, (boxes, 2))
    sizes = rng.uniform(0.2, 0.5, (boxes, 2))
    anchors = np.concatenate([centers, sizes], 1).astype(np.float32)
    options = flexbuffers.Dumps({
        "max_detections": max_detections, "max_classes_per_detection": 1,
        "detections_per_class": 3, "use_regular_nms": bool(regular),
        "nms_score_threshold": 0.7, "nms_iou_threshold": 0.45, "num_classes": classes,
        "y_scale": 10.0, "x_scale": 10.0, "h_scale": 5.0, "w_scale": 5.0})

    def act(shape, kind=T.FLOAT32, scale=None, zero=0):
        return shape, kind, None, scale, zero, False

    tensors = [act((1, boxes, 4)), act((1, boxes, classes + 1)),
               ((boxes, 4), T.FLOAT32, anchors, None, 0, False),
               act((1, max_detections, 4)), act((1, max_detections)), act((1, max_detections)),
               act((1,))]
    ops = [("TFLite_Detection_PostProcess", None, options, [0, 1, 2], [3, 4, 5, 6])]
    inputs = [0, 1]
    if quantized:
        tensors += [act((1, boxes, 4), T.INT8, 1.0 / 64, 0),
                    act((1, boxes, classes + 1), T.INT8, 1.0 / 255, -128)]
        dequantize = schema.BuiltinOperator.DEQUANTIZE
        ops = [(dequantize, schema.BuiltinOptions.NONE, None, [7], [0]),
               (dequantize, schema.BuiltinOptions.NONE, None, [8], [1]), *ops]
        inputs = [7, 8]
    return _operators(ops, tensors, inputs, [3, 4, 5, 6])


# Models no converter writes, built operator by operator.
_RELU = schema.ActivationFunctionType.RELU
_NONE = schema.ActivationFunctionType.NONE
HANDMADE = {
    "float_svdf": lambda: _svdf(1, 8, 4, 1, 5, _NONE),
    "float_svdf_rank2_relu": lambda: _svdf(2, 6, 3, 2, 4, _RELU),
    "svdf": lambda: _svdf(1, 8, 4, 2, 5, _NONE, quantized=True),
    "svdf_batch": lambda: _svdf(2, 6, 3, 1, 4, _NONE, quantized=True),
    "float_lstm": lambda: _lstm(False, 1, 3, 4, 5),
    "float_lstm_time_major_clip": lambda: _lstm(True, 2, 3, 3, 4, cell_clip=0.8),
    "lstm": lambda: _lstm(False, 1, 3, 4, 5, quantized=True),
    "float_zeros_like": lambda: _shape_ops("zeros_like"),
    "float_fill": lambda: _shape_ops("fill"),
    "float_detection_fast": lambda: _detection(False),
    "float_detection_regular": lambda: _detection(True),
    "detection_fast": lambda: _detection(False, quantized=True),
    "detection_regular": lambda: _detection(True, quantized=True),
    "float_shape_reshape": lambda: _shape_ops("shape_reshape"),
    "float_broadcast_args": lambda: _shape_ops("broadcast_args"),
    "lstm_time_major_clip": lambda: _lstm(True, 2, 3, 3, 4, cell_clip=0.8, quantized=True),
}
# TFLite's float SVDF and LSTM compute in another order than TFLite Micro, so
# the two differ in the last bits; TFLite Micro's outputs are recorded for these.
SUMMATION_ORDER = {"float_svdf", "float_svdf_rank2_relu", "float_lstm",
                   "float_lstm_time_major_clip"}
# TFLite's detection postprocess is another implementation than TFLite
# Micro's; TFLite Micro's outputs are recorded. Its fast form leaves the rows
# past the detection count unwritten, so they are recorded as zero.
DETECTION = {"float_detection_fast", "float_detection_regular", "detection_fast",
             "detection_regular"}
# The tier-1 cases again, converted without quantization.
_FLOAT_TIER1 = (
    "max_pool_valid", "max_pool_same", "avg_pool_valid", "avg_pool_same", "concat_channels",
    "concat_rows", "add_broadcast", "mul", "sub", "logistic", "tanh", "hard_swish", "relu",
    "relu6", "mean_spatial", "mean_spatial_flat", "mean_channels", "mean_rows", "mean_all",
    "resize_nearest", "resize_bilinear",
    "resize_bilinear_asymmetric", "transpose", "split", "split_v", "reshape", "conv_dilated",
    "depthwise_multiplier", "transpose_conv", "batch_matmul", "fully_connected_rank3",
    "pad_conv", "pad", "padv2", "squeeze_op", "expand_dims_op", "softmax", "pack", "unpack",
    "slice", "strided_slice", "strided_slice_shrink", "gather", "gather_scalar",
    "space_to_depth", "depth_to_space", "space_to_batch", "batch_to_space", "broadcast_to",
    "add_rank", "mul_rank_vector", "maximum_rank", "less_rank", "select_v2_rank",
    "pack_outer", "concat_outer", "concat_rank2",
    *ACTIVATIONS, *INDEXING, *BOOLEAN,
)
FLOAT_MODELS["float_l2_pool"] = CASES["float_l2_pool"]
FLOAT_MODELS["float_logistic_tails"] = _unary(tf.sigmoid, (1, 64))
# A branch chosen at run time; TFLite writes IF over two branch subgraphs.
FLOAT_MODELS["float_if"] = _unary(
    lambda x: tf.cond(tf.reduce_sum(x) > 0.0, lambda: x * 2.0 + 1.0, lambda: tf.nn.relu(x) - 3.0),
    (1, 4))
# A loop run until a condition fails; TFLite writes WHILE over a condition and
# a body subgraph. Positive inputs that only grow keep every loop finite.
FLOAT_MODELS["float_while"] = _unary(
    lambda x: tf.while_loop(lambda v: tf.reduce_sum(v) < 40.0, lambda v: [v * 1.5 + 0.25], [x])[0],
    (1, 4))
FLOAT_MODELS["float_while_two_variables"] = _binary(
    lambda x, y: tf.while_loop(lambda a, b: tf.reduce_sum(a) < 30.0,
                               lambda a, b: [a + b, b * 1.1], [x, y])[0],
    (1, 4), (1, 4))
_BRANCH_FILTER = _weights(91, 3, 3, 4, 4)
FLOAT_MODELS["float_if_feature_map"] = _unary(
    lambda x: tf.cond(tf.reduce_sum(x) > 0.0,
                      lambda: tf.nn.conv2d(x, _BRANCH_FILTER, 1, "SAME"),
                      lambda: tf.nn.relu(x) * 0.5),
    _MAP)
FLOAT_MODELS["float_while_feature_map"] = _unary(
    lambda x: tf.while_loop(lambda v: tf.reduce_sum(v) < 200.0, lambda v: [v * 1.5 + 0.25], [x])[0],
    (1, 4, 4, 2))
# A branch inside a branch.
FLOAT_MODELS["float_if_if"] = _unary(
    lambda x: tf.cond(tf.reduce_sum(x) > 0.0,
                      lambda: tf.cond(tf.reduce_max(x) > 2.0, lambda: x * 0.5, lambda: x + 1.0),
                      lambda: tf.nn.relu(x) - 3.0),
    (1, 4))
FLOAT_MODELS["float_if_two_operands"] = _binary(
    lambda x, y: tf.cond(tf.reduce_max(x) > tf.reduce_max(y), lambda: x + y, lambda: x * y),
    (1, 6), (1, 6))
# Loops counting in int32, the counter also read as a float.
FLOAT_MODELS["float_while_counter"] = _unary(
    lambda x: tf.while_loop(lambda i, v: i < 5,
                            lambda i, v: [i + 1, v * 1.5 + tf.cast(i, tf.float32)], [0, x])[1],
    (1, 4))
FLOAT_MODELS["float_while_countdown"] = _unary(
    lambda x: tf.while_loop(lambda i, v: i > 0,
                            lambda i, v: [i - 2, v + tf.cast(i * i, tf.float32)], [7, x])[1],
    (1, 4))
FLOAT_MODELS["float_if_int32"] = ([(1, 4), (1,)], lambda x, k: tf.cond(
    tf.reduce_sum(x) > 0.0, lambda: x * tf.cast(k + 1, tf.float32),
    lambda: x - tf.cast(k, tf.float32)))
# int8 control flow: the converter keeps each subgraph boundary in float32, with
# QUANTIZE and DEQUANTIZE on either side; loops count in int32 so they end.
_CF_FILTER = _weights(93, 3, 3, 4, 4)
CASES["if_conv"] = _unary(
    lambda x: tf.cond(tf.reduce_sum(x) > 0.0, lambda: tf.nn.relu(tf.nn.conv2d(x, _CF_FILTER, 1, "SAME")),
                      lambda: x * 0.5),
    _MAP)
CASES["while_conv"] = _unary(
    lambda x: tf.while_loop(lambda i, v: i < 3,
                            lambda i, v: [i + 1, tf.nn.relu(tf.nn.conv2d(v, _CF_FILTER, 1, "SAME"))],
                            [0, x])[1],
    _MAP)
FLOAT_MODELS["float_variable_window"] = _unary(lambda x: _WINDOW(x), (1, 2))
for _name in _FLOAT_TIER1:
    FLOAT_MODELS[f"float_{_name}"] = CASES[_name]
    if _name in REWRITES:
        REWRITES[f"float_{_name}"] = REWRITES[_name]
CASES.update(FLOAT_MODELS)
# The float islands exist only in the int8 conversions.
REWRITES.update({"elu": _int8_island(), "cumsum": _int8_island(True),
                 "cumsum_exclusive_reverse": _int8_island(True),
                 "dynamic_update_slice": _int8_island(shared=True),
                 "dynamic_update_slice_runtime": _int8_island(shared=True),
                 "not_equal": _int8_island(), "add_n": _int8_island(),
                 "cumsum_offset": _int8_island(),
                 "cumsum_offset_exclusive_reverse": _int8_island()})


def _convert(fn, shapes, ranges, rng, float_io=False, quantize=True, trackable=None,
             index_inputs=None):
    index_inputs = index_inputs or {}
    specs = [tf.TensorSpec(shape, tf.int32 if i in index_inputs else tf.float32)
             for i, shape in enumerate(shapes)]
    concrete = tf.function(fn).get_concrete_function(*specs)

    def representative():
        for _ in range(64):
            yield [rng.integers(*index_inputs[i], shape, dtype=np.int32) if i in index_inputs
                   else rng.uniform(lo, hi, shape).astype(np.float32)
                   for i, (shape, (lo, hi)) in enumerate(zip(shapes, ranges))]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete], trackable if trackable is not None else tf.function(fn))
    if not quantize:
        return converter.convert()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32 if float_io else tf.int8
    converter.inference_output_type = tf.float32 if float_io else tf.int8
    return converter.convert()


def _inputs(details, rng, value_range=None, index_range=None):
    shape = (SAMPLES, *details["shape"])
    if index_range is not None:
        return rng.integers(*index_range, shape, dtype=np.int32)
    if details["dtype"] == np.float32:
        return rng.uniform(*(value_range or (-3.0, 3.0)), shape).astype(np.float32)
    if value_range is None:
        return rng.integers(-128, 128, shape, dtype=np.int8)
    scale, zero_point = details["quantization"]
    values = np.round(rng.uniform(*value_range, shape) / scale) + zero_point
    return np.clip(values, -128, 127).astype(np.int8)


def generate(name: str) -> bool:
    """Writes the case; True when TFLite Micro deviates from the reference kernels."""
    rng = np.random.default_rng(sum(name.encode()))
    ranges = RANGES.get(name)
    if name in HANDMADE:
        model = HANDMADE[name]()
    else:
        shapes, fn = CASES[name]
        model = _convert(fn, shapes, ranges or [(-3.0, 3.0)] * len(shapes), rng,
                         float_io=name in FLOAT_BOUNDARIES, quantize=name not in FLOAT_MODELS,
                         trackable=TRACKABLES.get(name.removeprefix("float_")),
                         index_inputs=INDEX_INPUTS.get(name.removeprefix("float_")))
    if name in REWRITES:
        model = REWRITES[name](model)
    micro_interpreter = micro.Interpreter.from_bytes(model, arena_size=1024 * 1024)
    graph = schema.Model.GetRootAs(model, 0).Subgraphs(0)
    details = []
    for index in range(graph.InputsLength()):
        found = micro_interpreter.get_input_details(index)
        quantization = found["quantization_parameters"]
        details.append({"shape": found["shape"], "dtype": found["dtype"],
                        "quantization": (float(quantization["scales"][0]),
                                         int(quantization["zero_points"][0]))
                        if len(quantization["scales"]) else (0.0, 0)})
    try:
        reference = tf.lite.Interpreter(
            model_content=model,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        reference.allocate_tensors()
    except (ValueError, RuntimeError):
        # TFLite's reference resolver lacks some operators (CEIL, ELU, int8
        # CUMSUM); TFLite Micro's outputs are recorded unchecked for those.
        reference = None
    index_inputs = INDEX_INPUTS.get(name.removeprefix("float_"), {})
    inputs = [_inputs(found, rng, ranges[i] if ranges else None, index_inputs.get(i))
              for i, found in enumerate(details)]
    if name == "div":
        # TFLite refuses a divisor whose raw byte is 0; keep its value nonzero too.
        divisor, zero_point = inputs[1], details[1]["quantization"][1]
        divisor[(divisor == 0) | (divisor == zero_point)] = zero_point + 1 or 1
        # A numerator of 0 or -1 after its zero point makes TFLite's arithmetic
        # shift a 32-bit value by 32 or more, which is undefined there.
        numerator, zero_point = inputs[0], details[0]["quantization"][1]
        numerator[(numerator == zero_point) | (numerator == zero_point - 1)] = zero_point + 1
    if name.startswith("float_detection"):
        # Scores on a coarse grid tie, which the selection order has to settle.
        inputs[1] = np.round(inputs[1] * 8.0) / np.float32(8.0)
    outputs_count = graph.OutputsLength()
    micro_outputs, reference_outputs = [], []
    for sample in range(SAMPLES):
        for index, values in enumerate(inputs):
            micro_interpreter.set_input(values[sample], index)
        micro_interpreter.invoke()
        micro_outputs.append([micro_interpreter.get_output(i).copy() for i in range(outputs_count)])
        if name in DETECTION:
            count = int(micro_outputs[-1][3][0])
            for written in micro_outputs[-1][:3]:
                written[:, count:] = 0
        if reference is not None:
            for index, values in enumerate(inputs):
                reference.set_tensor(reference.get_input_details()[index]["index"], values[sample])
            try:
                reference.invoke()
            except RuntimeError:
                # Allocated but not run there (int8 ADD_N); recorded unchecked.
                reference = None
                continue
            reference_outputs.append([reference.get_tensor(d["index"]).copy()
                                      for d in reference.get_output_details()])
    has_reference = (reference is not None and name not in BROKEN_REFERENCE
                     and name not in SUMMATION_ORDER and name not in DETECTION)
    deviates = has_reference and any(not np.array_equal(m, r) for ms, rs in
                                     zip(micro_outputs, reference_outputs) for m, r in zip(ms, rs))
    outputs = reference_outputs if deviates else micro_outputs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tflite").write_bytes(model)
    arrays = {f"input_{i}": values for i, values in enumerate(inputs)}
    for index in range(outputs_count):
        arrays[f"output_{index}"] = np.stack([sample[index] for sample in outputs])
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    print(f"{name}: {len(model)} bytes" + (", TFLite Micro deviates" if deviates else ""))
    return deviates


# Whole reference models next to the one-operator cases, recorded the same way.
MODELS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tflite"
REFERENCE_MODELS = ("vww_96_int8", "pretrainedResnet_quant")


def generate_model(name: str) -> None:
    """Records TFLite Micro's outputs for seeded int8 inputs to a whole model."""
    model = (MODELS / f"{name}.tflite").read_bytes()
    interpreter = micro.Interpreter.from_bytes(model, arena_size=4 * 1024 * 1024)
    details = interpreter.get_input_details(0)
    rng = np.random.default_rng(sum(name.encode()))
    inputs = rng.integers(-128, 128, (2, *details["shape"]), dtype=np.int8)
    outputs = []
    for sample in inputs:
        interpreter.set_input(sample, 0)
        interpreter.invoke()
        outputs.append(interpreter.get_output(0).copy())
    np.savez_compressed(MODELS / f"{name}_tflm.npz", input_0=inputs, output_0=np.stack(outputs))
    print(f"{name}: {len(model)} bytes")


def main(names: list[str]) -> None:
    if names == ["--models"]:
        for name in REFERENCE_MODELS:
            generate_model(name)
        return
    deviations = [name for name in names or sorted({*CASES, *HANDMADE}) if generate(name)]
    if deviations:
        print("recorded from the reference kernels: " + ", ".join(deviations))
    print(f"tensorflow {tf.__version__}, tflite-micro {version('tflite-micro')}")


if __name__ == "__main__":
    main(sys.argv[1:])
