#!/usr/bin/env python3
"""Generate one-operator TFLite models and their reference outputs.

Each case is converted to an int8 TFLite model and run on seeded inputs through
TFLite Micro and through TFLite's reference kernels. The recorded outputs are
TFLite Micro's; where TFLite Micro disagrees with the reference kernels, the
reference kernels' are recorded and the case is named as a deviation. Models and
outputs go to tests/fixtures/tflite/ops/ so the frontend tests need neither
package. Needs tensorflow and tflite-micro, which are not project dependencies:

    python scripts/gen_tflite_fixtures.py [case ...]
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import flatbuffers
import numpy as np
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
    "div": [(-3.0, 3.0), _DIVISOR], "float_div": [(-3.0, 3.0), _DIVISOR], "float_floor_div": [(-3.0, 3.0), _DIVISOR],
    "float_floor_mod": [(-3.0, 3.0), _DIVISOR], "float_div_constant_first": [_DIVISOR],
    "float_div_broadcast": [(-3.0, 3.0), _DIVISOR],
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
})
# The tier-1 cases again, converted without quantization.
_FLOAT_TIER1 = (
    "max_pool_valid", "max_pool_same", "avg_pool_valid", "avg_pool_same", "concat_channels",
    "concat_rows", "add_broadcast", "mul", "sub", "logistic", "tanh", "hard_swish", "relu",
    "relu6", "mean_spatial", "mean_spatial_flat", "resize_nearest", "resize_bilinear",
    "resize_bilinear_asymmetric", "transpose", "split", "split_v", "reshape", "conv_dilated",
    "depthwise_multiplier", "transpose_conv", "batch_matmul", "fully_connected_rank3",
    "pad_conv", "pad", "padv2", "squeeze_op", "expand_dims_op", "softmax", "pack", "unpack",
    "slice", "strided_slice", "strided_slice_shrink", "gather", "gather_scalar",
    "space_to_depth", "depth_to_space", "space_to_batch", "batch_to_space",
)
for _name in _FLOAT_TIER1:
    FLOAT_MODELS[f"float_{_name}"] = CASES[_name]
    if _name in REWRITES:
        REWRITES[f"float_{_name}"] = REWRITES[_name]
CASES.update(FLOAT_MODELS)


def _convert(fn, shapes, ranges, rng, float_io=False, quantize=True):
    specs = [tf.TensorSpec(shape, tf.float32) for shape in shapes]
    concrete = tf.function(fn).get_concrete_function(*specs)

    def representative():
        for _ in range(64):
            yield [rng.uniform(lo, hi, shape).astype(np.float32)
                   for shape, (lo, hi) in zip(shapes, ranges)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], tf.function(fn))
    if not quantize:
        return converter.convert()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32 if float_io else tf.int8
    converter.inference_output_type = tf.float32 if float_io else tf.int8
    return converter.convert()


def _inputs(details, rng, value_range=None):
    shape = (SAMPLES, *details["shape"])
    if details["dtype"] == np.float32:
        return rng.uniform(*(value_range or (-3.0, 3.0)), shape).astype(np.float32)
    if value_range is None:
        return rng.integers(-128, 128, shape, dtype=np.int8)
    scale, zero_point = details["quantization"]
    values = np.round(rng.uniform(*value_range, shape) / scale) + zero_point
    return np.clip(values, -128, 127).astype(np.int8)


def generate(name: str) -> bool:
    """Writes the case; True when TFLite Micro deviates from the reference kernels."""
    shapes, fn = CASES[name]
    rng = np.random.default_rng(sum(name.encode()))
    ranges = RANGES.get(name)
    model = _convert(fn, shapes, ranges or [(-3.0, 3.0)] * len(shapes), rng,
                     float_io=name in FLOAT_BOUNDARIES, quantize=name not in FLOAT_MODELS)
    if name in REWRITES:
        model = REWRITES[name](model)
    try:
        reference = tf.lite.Interpreter(
            model_content=model,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    except ValueError:
        # The reference resolver lacks some float operators (CEIL).
        reference = tf.lite.Interpreter(model_content=model)
    reference.allocate_tensors()
    inputs = [_inputs(details, rng, ranges[i] if ranges else None)
              for i, details in enumerate(reference.get_input_details())]
    if name == "div":
        # TFLite refuses a divisor whose raw byte is 0; keep its value nonzero too.
        divisor, zero_point = inputs[1], reference.get_input_details()[1]["quantization"][1]
        divisor[(divisor == 0) | (divisor == zero_point)] = zero_point + 1 or 1
        # A numerator of 0 or -1 after its zero point makes TFLite's arithmetic
        # shift a 32-bit value by 32 or more, which is undefined there.
        numerator, zero_point = inputs[0], reference.get_input_details()[0]["quantization"][1]
        numerator[(numerator == zero_point) | (numerator == zero_point - 1)] = zero_point + 1
    output_details = reference.get_output_details()
    micro_interpreter = micro.Interpreter.from_bytes(model, arena_size=1024 * 1024)
    micro_outputs, reference_outputs = [], []
    for sample in range(SAMPLES):
        for index, values in enumerate(inputs):
            micro_interpreter.set_input(values[sample], index)
            reference.set_tensor(reference.get_input_details()[index]["index"], values[sample])
        micro_interpreter.invoke()
        reference.invoke()
        micro_outputs.append([micro_interpreter.get_output(i).copy() for i in range(len(output_details))])
        reference_outputs.append([reference.get_tensor(d["index"]).copy() for d in output_details])
    deviates = any(not np.array_equal(m, r) for ms, rs in zip(micro_outputs, reference_outputs)
                   for m, r in zip(ms, rs))
    outputs = reference_outputs if deviates else micro_outputs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tflite").write_bytes(model)
    arrays = {f"input_{i}": values for i, values in enumerate(inputs)}
    for index in range(len(output_details)):
        arrays[f"output_{index}"] = np.stack([sample[index] for sample in outputs])
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    print(f"{name}: {len(model)} bytes" + (", TFLite Micro deviates" if deviates else ""))
    return deviates


def main(names: list[str]) -> None:
    deviations = [name for name in names or sorted(CASES) if generate(name)]
    if deviations:
        print("recorded from the reference kernels: " + ", ".join(deviations))
    print(f"tensorflow {tf.__version__}, tflite-micro {version('tflite-micro')}")


if __name__ == "__main__":
    main(sys.argv[1:])
