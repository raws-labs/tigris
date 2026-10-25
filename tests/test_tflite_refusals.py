"""Reasons the TFLite frontend gives for refusing an operator and the order of
the checks that give them, each from a model the writer builds, and the reader
paths no fixture reaches."""

import random
import re
import struct
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from flatbuffers import flexbuffers
from onnx import helper

from tigris.analysis.plan_graph import slow_arena_required
from tigris.cli import cli
from tigris.emitters.binary.reader import read_binary_plan
from tigris.frontends import tflite
from tigris.frontends import tflite_writer as tw
from tigris.frontends.flatbuffer import Table

OPS = Path(__file__).parent / "fixtures" / "tflite" / "ops"
_FLOAT32, _INT32, _INT64, _BOOL, _INT16, _INT8 = 0, 2, 4, 6, 7, 9


def _edited(name: str, *edits) -> bytes:
    """The fixture `name` read by the writer, changed by each `edit(graph, model)`
    on its main graph, and written again."""
    model = tw.read((OPS / f"{name}.tflite").read_bytes())
    for edit in edits:
        edit(model.subgraphs[0], model)
    return tw.write(model)


def _tensor(index: int, **fields):
    def edit(graph, model):
        for key, value in fields.items():
            setattr(graph.tensors[index], key, value)
    return edit


def _quantization(index: int, **fields):
    def edit(graph, model):
        for key, value in fields.items():
            setattr(graph.tensors[index].quantization, key, value)
    return edit


def _option(op: int, slot: int, fmt: str, value):
    def edit(graph, model):
        graph.operators[op].options.fields[slot] = (fmt, value)
    return edit


def _operator(op: int, **fields):
    def edit(graph, model):
        for key, value in fields.items():
            setattr(graph.operators[op], key, value)
    return edit


def _data(index: int, values, dtype):
    """Tensor `index` holds `values` in its own buffer."""
    def edit(graph, model):
        model.buffers[graph.tensors[index].buffer] = np.asarray(values, dtype).tobytes()
    return edit


def _constant(graph, model, values, type_: int, quantization=None) -> int:
    model.buffers.append(np.ascontiguousarray(values).tobytes())
    graph.tensors.append(tw.Tensor(list(np.shape(values)), type_, len(model.buffers) - 1,
                                   b"c%d" % len(graph.tensors), quantization))
    return len(graph.tensors) - 1


def _code(model, kind: str) -> int:
    """The opcode of builtin `kind`, added to the model when it has none."""
    code = tflite._BUILTIN_OPERATORS.index(kind)
    codes = [max(c.builtin_code, c.deprecated_builtin_code) for c in model.operator_codes]
    if code not in codes:
        model.operator_codes.append(tw.OperatorCode(code, 1, None, min(code, 127)))
        codes.append(code)
    return codes.index(code)


def _constant_operands(op: int, *values):
    """Operator `op` reads only new float32 constants."""
    def edit(graph, model):
        graph.operators[op].inputs = [_constant(graph, model, np.asarray(v, np.float32), _FLOAT32)
                                      for v in values]
    return edit


def _detection(**changes):
    """The detection operator's custom options with `changes`; None removes a key."""
    def edit(graph, model):
        op = graph.operators[0]
        options = flexbuffers.Loads(op.custom_options)
        for key, value in changes.items():
            if value is None:
                del options[key]
            else:
                options[key] = value
        op.custom_options = bytes(flexbuffers.Dumps(options))
    return edit


def _variable_read_as_int8(graph, model):
    """The main graph only reads its variable, into an int8 model output."""
    graph.operators = graph.operators[:3]
    graph.outputs = [graph.operators[2].outputs[0]]
    tensor = graph.tensors[graph.outputs[0]]
    tensor.type, tensor.quantization = _INT8, tw.Quantization([0.05], [0])


def _if_on_its_input(graph, model):
    """The IF branches on the float32 model input itself."""
    graph.operators = [graph.operators[2]]
    graph.operators[0].inputs = [graph.inputs[0], graph.inputs[0]]


def _int32_overflow(graph, model):
    """The index arithmetic's MUL multiplies two constants past the int32 range."""
    graph.operators[0].inputs = [_constant(graph, model, np.full(3, 1 << 30, np.int32), _INT32),
                                 graph.operators[0].inputs[1]]


SCALES_2 = [0.05, 0.06]

REASONS = [
    # _operator_reason
    pytest.param("float_variable_window", [_variable_read_as_int8],
                 ["operator 2 READ_VARIABLE: variables other than float32"], id="variables"),
    pytest.param("mean_spatial", [_tensor(1, buffer=0)],
                 ["operator 0 MEAN: input 1 must be a constant"], id="constant-operand"),
    pytest.param("float_if", [_if_on_its_input],
                 ["operator 0 IF: a condition other than one bool"], id="if-condition"),
    pytest.param("float_if", [_tensor(5, type=_INT8, quantization=tw.Quantization([0.05], [0]))],
                 ["operator 2 IF: operands other than float32 or int32"], id="if-operands"),
    pytest.param("float_conv_bias", [_tensor(2, type=_INT32)],
                 ["operator 0 CONV_2D: bias of a float32 operator must be float32"],
                 id="float-bias"),
    pytest.param("fully_connected_bias", [_quantization(2, zero_point=[1] * 5)],
                 ["operator 0 FULLY_CONNECTED: weights must be constant symmetric int8"],
                 id="int8-weights"),
    pytest.param("conv_bias", [_tensor(1, type=_FLOAT32)],
                 ["operator 0 CONV_2D: bias must be int32"], id="int8-bias"),
    pytest.param("equal", [_tensor(1, type=_BOOL, quantization=None)],
                 ["operator 0 EQUAL: operands of the wrong dtype"], id="bool-operands"),
    pytest.param("equal", [_tensor(2, type=_INT8, quantization=tw.Quantization([0.05], [0]))],
                 ["operator 0 EQUAL: an output of the wrong dtype"], id="bool-output"),
    pytest.param("cast_bool", [_tensor(3, type=_INT8, quantization=tw.Quantization([0.5], [0]))],
                 ["operator 1 CAST: int8 output encoding other than scale 1 and zero point 0"],
                 id="cast-int8"),
    pytest.param("arg_max", [_tensor(2, type=_INT64)],
                 ["operator 0 ARG_MAX: INT64 indices; TFLite Micro writes int32"],
                 id="index-dtype"),
    pytest.param("uint8_boundaries", [_tensor(5, type=_FLOAT32)],
                 ["operator 2 QUANTIZE: only float or int8 to int8 converts"], id="quantize"),
    pytest.param("float_boundaries",
                 [_tensor(4, type=_INT8, quantization=tw.Quantization([0.05], [0]))],
                 ["operator 3 DEQUANTIZE: only int8 to float converts"], id="dequantize"),
    pytest.param("add_rank", [_quantization(1, scale=SCALES_2, zero_point=[-1, -1])],
                 ["operator 0 ADD: activations must be quantized per tensor"], id="per-tensor"),
    pytest.param("float_mul", [_option(0, 0, "b", 2)],
                 ["operator 0 MUL: fused relu_n1_to_1"], id="fused"),
    pytest.param("float_div", [_constant_operands(0, np.ones((1, 6, 6, 4)), np.full((1, 6, 6, 4), 2))],
                 ["operator 0 DIV: only constant operands"], id="constant-operands"),
    pytest.param("concat_channels", [_quantization(1, scale=[0.05])],
                 ["operator 0 CONCATENATION: inputs quantized differently from the output"],
                 id="concat-quantization"),
    pytest.param("float_l2_normalization", [_option(0, 0, "b", 1)],
                 ["operator 0 L2_NORMALIZATION: fused relu"], id="l2-fused"),
    pytest.param("float_softmax", [_option(0, 0, "f", 0.5)],
                 ["operator 0 SOFTMAX: beta other than 1"], id="softmax-beta"),
    pytest.param("float_fully_connected_bias", [_option(0, 1, "b", 1)],
                 ["operator 0 FULLY_CONNECTED: shuffled weights format"], id="fc-shuffled"),
    pytest.param("float_batch_matmul", [_tensor(1, shape=[8, 5])],
                 ["operator 0 BATCH_MATMUL: operands of different rank"], id="matmul-rank"),
    pytest.param("float_resize_bilinear", [_option(0, 2, "?", True), _option(0, 3, "?", True)],
                 ["operator 0 RESIZE_BILINEAR: align_corners and half_pixel_centers together"],
                 id="resize-corners"),
    # _int32_reason
    pytest.param("gather_index_arithmetic", [_option(0, 0, "b", 1)],
                 ["operator 0 MUL: a fused activation on int32"], id="int32-fused"),
    pytest.param("gather_index_arithmetic", [_int32_overflow],
                 ["operator 0 MUL: only constant operands"], id="int32-constant-operands"),
    pytest.param("float_less", [_tensor(1, type=_INT32)],
                 ["operator 0 LESS: int32 compared with another dtype"], id="int32-compare"),
    # _svdf_reason
    pytest.param("float_svdf", [_tensor(0, shape=[1, 1, 8])],
                 ["operator 0 SVDF: input of rank other than 2"], id="svdf-rank"),
    pytest.param("float_svdf", [_option(0, 0, "i", 0)],
                 ["operator 0 SVDF: rank 0 for 4 filters"], id="svdf-rank-option"),
    pytest.param("float_svdf", [_option(0, 0, "i", 3)],
                 ["operator 0 SVDF: rank 3 for 4 filters"], id="svdf-rank-filters"),
    pytest.param("float_svdf", [_option(0, 0, "i", 2)],
                 ["operator 0 SVDF: a 1x4 output where rank 2 gives 1x2"], id="svdf-output"),
    pytest.param("float_svdf", [_tensor(3, type=_INT32)],
                 ["operator 0 SVDF: float32 input with operands of another dtype"],
                 id="svdf-float-operands"),
    pytest.param("float_svdf", [_option(0, 1, "b", 2)],
                 ["operator 0 SVDF: fused relu_n1_to_1"], id="svdf-fused"),
    pytest.param("svdf", [_tensor(5, type=_FLOAT32)],
                 ["operator 0 SVDF: activations must be all int8 or all float32"],
                 id="svdf-int8-operands"),
    pytest.param("svdf", [_quantization(0, scale=SCALES_2, zero_point=[2, 2])],
                 ["operator 0 SVDF: operands must be quantized per tensor"], id="svdf-per-tensor"),
    # _lstm_reason
    pytest.param("float_lstm", [_tensor(0, shape=[3, 4])],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: input of rank other than 3"],
                 id="lstm-rank"),
    pytest.param("float_lstm", [_tensor(1, buffer=0)],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: weights and biases must be constant"],
                 id="lstm-constant"),
    pytest.param("float_lstm", [_option(0, 3, "?", True)],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: a 1x5 hidden state where time_major "
                  "true gives 3x5"], id="lstm-time-major"),
    pytest.param("float_lstm", [_tensor(14, shape=[1, 4])],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: a 1x4 cell state where time_major "
                  "false gives 1x5"], id="lstm-cell-state"),
    pytest.param("lstm", [_tensor(1, type=_INT16)],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: activations must be all int8 or all "
                  "float32"], id="lstm-int8-operands"),
    pytest.param("lstm", [_quantization(0, scale=SCALES_2, zero_point=[2, 2])],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: operands must be quantized per tensor"],
                 id="lstm-per-tensor"),
    # _detection_reason
    pytest.param("float_detection_fast", [_operator(0, outputs=[3, 4, 5])],
                 ["operator 0 CUSTOM:TFLite_Detection_PostProcess: operands other than box "
                  "encodings, scores and anchors, and four outputs"], id="detection-operands"),
    pytest.param("float_detection_fast", [_tensor(2, type=_INT8)],
                 ["operator 0 CUSTOM:TFLite_Detection_PostProcess: operands or outputs other "
                  "than float32"], id="detection-dtype"),
    pytest.param("float_detection_fast", [_tensor(2, shape=[4, 16])],
                 ["operator 0 CUSTOM:TFLite_Detection_PostProcess: shapes other than box "
                  "encodings [1, boxes, 4], scores [1, boxes, classes] and anchors [boxes, 4]"],
                 id="detection-shapes"),
    pytest.param("float_detection_regular", [_detection(detections_per_class=0)],
                 ["operator 0 CUSTOM:TFLite_Detection_PostProcess: no detections per class"],
                 id="detection-per-class"),
    pytest.param("float_detection_fast", [_tensor(4, shape=[1, 7])],
                 ["operator 0 CUSTOM:TFLite_Detection_PostProcess: outputs other than "
                  "[1, detections, 4], [1, detections], [1, detections] and [1]"],
                 id="detection-outputs"),
    # _window_reason
    pytest.param("max_pool_same", [_option(0, 1, "i", 0)],
                 ["operator 0 MAX_POOL_2D: a 2x0 stride; strides must be at least 1"],
                 id="window-stride"),
    pytest.param("max_pool_same", [_option(0, 3, "i", 0)],
                 ["operator 0 MAX_POOL_2D: a 3x0 filter; filter sizes must be at least 1"],
                 id="window-filter"),
    # _data_movement_reason
    pytest.param("pack", [_quantization(1, scale=[0.05])],
                 ["operator 0 PACK: inputs and outputs quantized differently"],
                 id="movement-quantization"),
    pytest.param("strided_slice", [_option(0, 2, "i", 1)],
                 ["operator 0 STRIDED_SLICE: ellipsis, new axes or offset slices"],
                 id="strided-ellipsis"),
    pytest.param("strided_slice", [_tensor(0, shape=[1, 1, 6, 6, 4])],
                 ["operator 0 STRIDED_SLICE: slices of rank above 4"], id="strided-rank"),
    pytest.param("strided_slice", [_data(3, [0, 1, 0], np.int32), _tensor(3, shape=[3])],
                 ["operator 0 STRIDED_SLICE: begin, end or strides with fewer entries than the "
                  "input has axes"], id="strided-short"),
    pytest.param("float_slice", [_data(1, [0, 1, 2], np.int32), _tensor(1, shape=[3])],
                 ["operator 0 SLICE: begin or size with fewer entries than the input has axes"],
                 id="slice-short"),
    pytest.param("strided_slice", [_data(1, [-1, 1, 1, 1], np.int32), _option(0, 4, "i", 1)],
                 ["operator 0 STRIDED_SLICE: a dropped axis with a negative stride"],
                 id="strided-shrink-negative"),
    pytest.param("strided_slice_shrink",
                 [_data(3, [0, 6, 0, 0], np.int32), _data(2, [0, 7, 0, 0], np.int32)],
                 ["operator 0 STRIDED_SLICE: a dropped axis at an index outside the axis"],
                 id="strided-shrink-past-end"),
    pytest.param("strided_slice_shrink",
                 [_data(3, [0, -7, 0, 0], np.int32), _data(2, [0, -6, 0, 0], np.int32)],
                 ["operator 0 STRIDED_SLICE: a dropped axis at an index outside the axis"],
                 id="strided-shrink-before-start"),
    pytest.param("reverse_spatial", [_data(1, [1, 3], np.int32)],
                 ["operator 0 REVERSE_V2: axes that are not adjacent"], id="reverse-axes"),
    pytest.param("space_to_batch", [_tensor(0, shape=[1, 4, 12])],
                 ["operator 0 SPACE_TO_BATCH_ND: only rank-4 inputs convert"], id="space-rank"),
    # _option_shape_reason
    pytest.param("float_pack", [_option(0, 1, "i", 2)],
                 ["operator 0 PACK: a 1x2x4x6 output where axis 2 gives 1x4x2x6"], id="pack-axis"),
    pytest.param("float_pack", [_option(0, 1, "i", 4)],
                 ["operator 0 PACK: axis 4 for inputs 1x4x6, 1x4x6"], id="pack-axis-range"),
    pytest.param("float_unpack", [_tensor(1, shape=[1, 6, 4])],
                 ["operator 0 UNPACK: a 1x6x4 output where axis 1 gives 1x4x6"], id="unpack-output"),
    pytest.param("float_unpack", [_option(0, 1, "i", 2)],
                 ["operator 0 UNPACK: axis 2 for a 1x3x4x6 input and 3 outputs"], id="unpack-axis"),
    pytest.param("float_concat_channels", [_tensor(2, shape=[1, 6, 6, 9])],
                 ["operator 0 CONCATENATION: a 1x6x6x9 output where axis -1 gives 1x6x6x8"],
                 id="concat-output"),
    pytest.param("float_concat_channels", [_option(0, 0, "i", 1)],
                 ["operator 0 CONCATENATION: axis 1 for inputs 1x6x6x3, 1x6x6x5"], id="concat-axis"),
    pytest.param("float_gather_indices", [_option(0, 0, "i", 0)],
                 ["operator 0 GATHER: a 1x6x4x4 output where axis 0, batch_dims 0 gives 4x6x6x4"],
                 id="gather-axis"),
    pytest.param("float_gather_batch", [_option(0, 1, "i", 0)],
                 ["operator 0 GATHER: a 2x3x4 output where axis 1, batch_dims 0 gives 2x2x3x4"],
                 id="gather-batch-dims"),
    pytest.param("float_gather_indices", [_option(0, 0, "i", 4)],
                 ["operator 0 GATHER: axis 4, batch_dims 0 for inputs 1x6x6x4, 4"],
                 id="gather-axis-range"),
    pytest.param("float_batch_matmul", [_option(0, 1, "?", True), _tensor(0, shape=[1, 6, 5])],
                 ["operator 0 BATCH_MATMUL: a 1x6x5 output where adj_x false, adj_y true gives "
                  "1x6x8"], id="matmul-adjoint"),
    pytest.param("float_batch_matmul", [_option(0, 0, "?", True)],
                 ["operator 0 BATCH_MATMUL: adj_x true, adj_y false for inputs 1x6x8, 1x8x5"],
                 id="matmul-adjoint-inputs"),
    pytest.param("strided_slice", [_tensor(4, shape=[1, 4, 2, 6])],
                 ["operator 0 STRIDED_SLICE: a 1x4x2x6 output where the slice gives 1x4x6x2"],
                 id="strided-output"),
]


@pytest.mark.parametrize(("name", "edits", "reasons"), REASONS)
def test_each_refusal_reason(name, edits, reasons):
    assert tflite.unsupported(_edited(name, *edits)) == reasons


# Models failing two checks that each give their own reason alone; the reason for
# both shows which check runs first. The second check is one every operator goes
# through, except for svdf, which returns before those and orders its own.
PER_TENSOR = [_quantization(0, scale=SCALES_2, zero_point=[0, 0])]
INT8_OUTPUT = tw.Quantization([0.05], [0])
INT64_INDICES = tw.Quantization(max=[float(1 << 40)])
ORDER = [
    # Checks of the operator's kind that run before the dtype and quantization checks.
    pytest.param("fully_connected_bias", [_option(0, 5, "[B", [1])], PER_TENSOR,
                 ["operator 0 FULLY_CONNECTED: a non-standard quantization contract in quant_spec"],
                 id="fc-quant-spec"),
    pytest.param("pack", [_option(0, 0, "i", 1)],
                 [_quantization(1, scale=SCALES_2, zero_point=[-1, -1])],
                 ["operator 0 PACK: values_count 1 for 2 inputs"], id="pack-count"),
    pytest.param("mean_spatial", [_tensor(1, buffer=0)], PER_TENSOR,
                 ["operator 0 MEAN: input 1 must be a constant"], id="constant-operand"),
    pytest.param("gather_runtime", [_tensor(1, type=_INT64, quantization=INT64_INDICES)], PER_TENSOR,
                 ["model boundary 'i' is INT64",
                  "indices 'i' are computed by an operator other than ARG_MAX, ARG_MIN or int32 "
                  "arithmetic",
                  "operator 0 GATHER: INT64 run-time indices; the runtime takes int32"],
                 id="run-time-indices"),
    pytest.param("svdf", [_tensor(5, type=_FLOAT32)],
                 [_quantization(0, scale=SCALES_2, zero_point=[2, 2])],
                 ["operator 0 SVDF: activations must be all int8 or all float32"], id="svdf"),
    pytest.param("svdf", [_option(0, 0, "i", 3)], [_tensor(5, type=_FLOAT32)],
                 ["operator 0 SVDF: rank 3 for 8 filters"], id="svdf-rank-option"),
    pytest.param("lstm", [_option(0, 3, "?", True)], [_tensor(1, type=_INT16)],
                 ["operator 0 UNIDIRECTIONAL_SEQUENCE_LSTM: a 1x5 hidden state where time_major "
                  "true gives 3x5"], id="lstm-time-major"),
    pytest.param("float_conv_bias", [_tensor(2, type=_INT32)],
                 [_tensor(3, type=_INT8, quantization=INT8_OUTPUT)],
                 ["operator 0 CONV_2D: bias of a float32 operator must be float32"],
                 id="float-bias"),
    pytest.param("fully_connected_bias", [_quantization(2, zero_point=[1] * 5)], PER_TENSOR,
                 ["operator 0 FULLY_CONNECTED: weights must be constant symmetric int8"],
                 id="int8-weights"),
    pytest.param("conv_bias", [_tensor(1, type=_FLOAT32)], PER_TENSOR,
                 ["operator 0 CONV_2D: bias must be int32"], id="int8-bias"),
    pytest.param("equal", [_tensor(1, type=_BOOL, quantization=None)], PER_TENSOR,
                 ["operator 0 EQUAL: operands of the wrong dtype"], id="bool-operands"),
    pytest.param("equal", [_tensor(2, type=_INT8, quantization=INT8_OUTPUT)], PER_TENSOR,
                 ["operator 0 EQUAL: an output of the wrong dtype"], id="bool-output"),
    pytest.param("arg_max", [_tensor(2, type=_INT64)], PER_TENSOR,
                 ["operator 0 ARG_MAX: INT64 indices; TFLite Micro writes int32"],
                 id="index-dtype"),
    # Checks of the operator's kind that run after them.
    pytest.param("add_rank", [_option(0, 0, "b", 2)], PER_TENSOR,
                 ["operator 0 ADD: activations must be quantized per tensor"], id="fused"),
    pytest.param("float_div",
                 [_constant_operands(0, np.ones((1, 6, 6, 4)), np.full((1, 6, 6, 4), 2))],
                 [_tensor(2, type=_INT8, quantization=INT8_OUTPUT)],
                 ["operator 0 DIV: activations must be all int8 or all float32"],
                 id="constant-operands"),
    pytest.param("concat_channels", [_quantization(1, scale=[0.05])], PER_TENSOR,
                 ["operator 0 CONCATENATION: activations must be quantized per tensor"],
                 id="concat-quantization"),
    pytest.param("pack", [_quantization(1, scale=[0.05])], PER_TENSOR,
                 ["operator 0 PACK: activations must be quantized per tensor"],
                 id="movement-quantization"),
    pytest.param("strided_slice", [_option(0, 2, "i", 1)], PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-ellipsis"),
    pytest.param("strided_slice", [_tensor(0, shape=[1, 1, 6, 6, 4])], PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-rank"),
    pytest.param("strided_slice", [_data(3, [0, 1, 0], np.int32), _tensor(3, shape=[3])],
                 PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-short"),
    pytest.param("slice", [_data(2, [0, 1, 2], np.int32), _tensor(2, shape=[3])], PER_TENSOR,
                 ["operator 0 SLICE: activations must be quantized per tensor"], id="slice-short"),
    pytest.param("strided_slice", [_data(1, [1, 0, 1, 1], np.int32)], PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-zero"),
    pytest.param("strided_slice", [_data(1, [-1, 1, 1, 1], np.int32), _option(0, 4, "i", 1)],
                 PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-shrink-negative"),
    pytest.param("strided_slice_shrink",
                 [_data(3, [0, 6, 0, 0], np.int32), _data(2, [0, 7, 0, 0], np.int32)], PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-shrink-outside"),
    pytest.param("reverse_spatial", [_data(1, [1, 3], np.int32)], PER_TENSOR,
                 ["operator 0 REVERSE_V2: activations must be quantized per tensor"],
                 id="reverse-axes"),
    pytest.param("space_to_batch", [_tensor(0, shape=[1, 4, 12])], PER_TENSOR,
                 ["operator 0 SPACE_TO_BATCH_ND: activations must be quantized per tensor"],
                 id="space-rank"),
    pytest.param("depth_to_space", [_option(0, 0, "i", 3)], PER_TENSOR,
                 ["operator 0 DEPTH_TO_SPACE: activations must be quantized per tensor"],
                 id="block-size"),
    pytest.param("depth_to_space", [_option(0, 0, "i", 1)], PER_TENSOR,
                 ["operator 0 DEPTH_TO_SPACE: activations must be quantized per tensor"],
                 id="block-output"),
    pytest.param("l2_normalization", [_option(0, 0, "b", 1)], PER_TENSOR,
                 ["operator 0 L2_NORMALIZATION: activations must be quantized per tensor"],
                 id="l2-fused"),
    pytest.param("sum_spatial", [_data(1, [1, 3], np.int32)], PER_TENSOR,
                 ["operator 0 SUM: activations must be quantized per tensor"], id="reduction-axes"),
    pytest.param("softmax", [_option(0, 0, "f", 0.5)], PER_TENSOR,
                 ["operator 0 SOFTMAX: activations must be quantized per tensor"], id="softmax-beta"),
    pytest.param("fully_connected_bias", [_option(0, 1, "b", 1)], PER_TENSOR,
                 ["operator 0 FULLY_CONNECTED: activations must be quantized per tensor"],
                 id="fc-shuffled"),
    pytest.param("max_pool_same", [_option(0, 0, "b", 2)], [_option(0, 5, "b", 2)],
                 ["operator 0 MAX_POOL_2D: fused relu_n1_to_1"], id="window"),
    pytest.param("max_pool_same", [_option(0, 0, "b", 2)], PER_TENSOR,
                 ["operator 0 MAX_POOL_2D: activations must be quantized per tensor"],
                 id="window-padding"),
    pytest.param("max_pool_same", [_option(0, 1, "i", 0)], PER_TENSOR,
                 ["operator 0 MAX_POOL_2D: activations must be quantized per tensor"],
                 id="window-stride"),
    pytest.param("max_pool_same", [_option(0, 3, "i", 0)], PER_TENSOR,
                 ["operator 0 MAX_POOL_2D: activations must be quantized per tensor"],
                 id="window-filter"),
    pytest.param("depthwise_multiplier", [_option(0, 3, "i", 1)], PER_TENSOR,
                 ["operator 0 DEPTHWISE_CONV_2D: activations must be quantized per tensor"],
                 id="depth-multiplier"),
    pytest.param("transpose_conv", [_option(0, 0, "b", 1)], PER_TENSOR,
                 ["operator 0 TRANSPOSE_CONV: activations must be quantized per tensor"],
                 id="window-input"),
    pytest.param("pad_conv", [_option(1, 0, "b", 0)],
                 [_quantization(5, scale=SCALES_2, zero_point=[1, 1])],
                 ["operator 1 CONV_2D: activations must be quantized per tensor"],
                 id="window-output"),
    pytest.param("float_batch_matmul", [_tensor(1, shape=[8, 5])], [_tensor(1, type=_INT8)],
                 ["operator 0 BATCH_MATMUL: activations must be all int8 or all float32"],
                 id="matmul-rank"),
    pytest.param("float_batch_matmul", [_tensor(1, shape=[8, 5])], [_option(0, 0, "?", True)],
                 ["operator 0 BATCH_MATMUL: operands of different rank"], id="matmul-rank-adjoint"),
    pytest.param("pack", [_option(0, 1, "i", 2)], PER_TENSOR,
                 ["operator 0 PACK: activations must be quantized per tensor"], id="pack-axis"),
    pytest.param("strided_slice", [_tensor(4, shape=[1, 4, 2, 6])], PER_TENSOR,
                 ["operator 0 STRIDED_SLICE: activations must be quantized per tensor"],
                 id="strided-output"),
    pytest.param("resize_bilinear", [_option(0, 2, "?", True), _option(0, 3, "?", True)],
                 PER_TENSOR,
                 ["operator 0 RESIZE_BILINEAR: activations must be quantized per tensor"],
                 id="resize-corners"),
]


@pytest.mark.parametrize(("name", "own", "shared", "reasons"), ORDER)
def test_the_first_failing_check_gives_the_reason(name, own, shared, reasons):
    alone = [tflite.unsupported(_edited(name, *edits)) for edits in (own, shared)]
    assert alone[0] != alone[1] and reasons in alone
    assert tflite.unsupported(_edited(name, *own, *shared)) == reasons


def test_a_per_channel_cast_output_fails_the_cast_encoding_check_first():
    """The int8 output of the CAST fails its encoding check and the per-tensor
    check every operator goes through; the QUANTIZE reading it fails the second."""
    per_channel = tw.Quantization([1.0, 1.0], [0, 0])
    assert tflite.unsupported(_edited("cast_bool", _tensor(3, type=_INT8, quantization=per_channel))) == [
        "operator 1 CAST: int8 output encoding other than scale 1 and zero point 0",
        "operator 2 QUANTIZE: activations must be quantized per tensor"]


def test_detection_options_left_out_take_tflite_micros_defaults():
    data = _edited("float_detection_fast", _detection(detections_per_class=None, use_regular_nms=None))
    stated = flexbuffers.Loads(tw.read(data).subgraphs[0].operators[0].custom_options)
    assert not {"detections_per_class", "use_regular_nms"} & set(stated)
    assert tflite.unsupported(data) == []
    node = next(n for n in tflite.to_onnx(data, "detection").graph.node if n.domain == "tigris")
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    assert (attributes["detections_per_class"], attributes["use_regular_nms"]) == (100, 0)


def _left_out(*slots):
    """Operator 0's options without the fields at `slots`."""
    def edit(graph, model):
        for slot in slots:
            graph.operators[0].options.fields[slot] = ("i", None)
    return edit


# Builtin options a file leaves out read as the schema's defaults, which are 0
# for beta, strides and pool filters, as TFLite Micro reads them.
LEFT_OUT = [
    pytest.param("float_softmax", (0,), "SOFTMAX: beta other than 1", id="softmax-beta"),
    pytest.param("float_conv_bias", (1, 2), "CONV_2D: a 0x0 stride; strides must be at least 1",
                 id="conv-stride"),
    pytest.param("float_depthwise_bias", (2,),
                 "DEPTHWISE_CONV_2D: a 0x1 stride; strides must be at least 1", id="depthwise-stride"),
    pytest.param("float_transpose_conv", (1,),
                 "TRANSPOSE_CONV: a 2x0 stride; strides must be at least 1", id="transpose-conv-stride"),
    pytest.param("float_max_pool_same", (1, 2), "MAX_POOL_2D: a 0x0 stride; strides must be at least 1",
                 id="pool-stride"),
    pytest.param("float_max_pool_same", (3, 4),
                 "MAX_POOL_2D: a 0x0 filter; filter sizes must be at least 1", id="pool-filter"),
]


@pytest.mark.parametrize(("name", "slots", "reason"), LEFT_OUT)
def test_an_option_left_out_takes_the_schema_default(name, slots, reason):
    data = _edited(name, _left_out(*slots))
    assert not set(slots) & set(tw.read(data).subgraphs[0].operators[0].options.source.present())
    assert tflite.unsupported(data) == [f"operator 0 {reason}"]


@pytest.mark.parametrize(("name", "reason"), [
    pytest.param("float_conv_bias", "CONV_2D: a 1x0 stride", id="conv"),
    pytest.param("float_depthwise_bias", "DEPTHWISE_CONV_2D: a 1x0 stride", id="depthwise"),
    pytest.param("float_transpose_conv", "TRANSPOSE_CONV: a 2x0 stride", id="transpose-conv"),
    pytest.param("float_max_pool_same", "MAX_POOL_2D: a 2x0 stride", id="max-pool"),
    pytest.param("float_avg_pool_same", "AVERAGE_POOL_2D: a 1x0 stride", id="average-pool"),
    pytest.param("float_l2_pool", "L2_POOL_2D: a 1x0 stride", id="l2-pool"),
])
def test_a_zero_stride_is_refused_by_every_entry_point(name, reason, tmp_path):
    data = _edited(name, _option(0, 1, "i", 0))
    reason = f"operator 0 {reason}; strides must be at least 1"
    assert tflite.unsupported(data) == [reason]
    with pytest.raises(ValueError, match=re.escape(reason)):
        tflite.to_onnx(data, "model")
    model = tmp_path / "model.tflite"
    model.write_bytes(data)
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(tmp_path / "model.tgrs")])
    assert result.exit_code != 0 and isinstance(result.exception, SystemExit), result.output
    assert reason in result.output


def _past_the_last_axis(begin: int, end: int, stride: int, shrink: bool):
    """Operator 0's begin, end and strides each get a fifth entry, and with
    `shrink` its shrink_axis_mask bit 4."""
    def edit(graph, model):
        op = graph.operators[0]
        for slot, value in zip((1, 2, 3), (begin, end, stride)):
            tensor = graph.tensors[op.inputs[slot]]
            values = [*np.frombuffer(model.buffers[tensor.buffer], np.int32), value]
            model.buffers[tensor.buffer] = np.asarray(values, np.int32).tobytes()
            tensor.shape = [5]
        if shrink:
            op.options.fields[4] = ("i", 16)
    return edit


@pytest.mark.parametrize(("begin", "end", "stride", "shrink"), [
    pytest.param(0, 1, 0, False, id="zero-stride"),
    pytest.param(0, 1, -1, True, id="negative-drop"),
    pytest.param(7, -9, 0, True, id="zero-stride-drop"),
])
def test_strided_slice_entries_past_the_last_axis_are_not_read(begin, end, stride, shrink):
    """TFLite Micro reads begin, end and strides at each input axis only, so a
    fifth entry on a rank-4 input changes neither its result nor the checks."""
    assert tflite.unsupported(_edited("strided_slice", _past_the_last_axis(begin, end, stride, shrink))) == []


def _with_external_buffer(model: tw.Model, index: int, overstated: int = 0) -> bytes:
    """`model`, one subgraph, with buffer `index` stored after the FlatBuffer
    by offset and size, as the converter writes models above 2 GB."""
    data = model.buffers[index]

    def build(offset: int) -> bytes:
        w = tw._Writer(1024)
        buffers = [w.table([(1, "Q", offset), (2, "Q", len(data) + overstated)]) if i == index
                   else w.table([(0, "o", w.data(b))]) for i, b in enumerate(model.buffers)]
        graph = model.subgraphs[0]
        tensors = [tw._tensor_table(w, t) for t in graph.tensors]
        operators = [tw._operator_table(w, op) for op in graph.operators]
        subgraph = w.table([(0, "o", w.tables(tensors)), (1, "o", w.numbers(graph.inputs, "i")),
                            (2, "o", w.numbers(graph.outputs, "i")), (3, "o", w.tables(operators))])
        codes = [w.table([(0, "b", c.deprecated_builtin_code or None), (3, "i", c.builtin_code or None)])
                 for c in model.operator_codes]
        root = w.table([(0, "I", model.version), (1, "o", w.tables(codes)),
                        (2, "o", w.tables([subgraph])), (4, "o", w.tables(buffers))])
        w.b.Finish(root, file_identifier=tflite.MAGIC)
        return bytes(w.b.Output())

    # The offset is a fixed-width field, so its value does not change the length.
    offset = -(-len(build(2)) // 16) * 16
    return build(offset).ljust(offset, b"\0") + data


def test_a_buffer_stored_after_the_flatbuffer_converts_as_one_inside_it():
    model = tw.read((OPS / "float_conv_bias.tflite").read_bytes())
    weights = model.subgraphs[0].tensors[1]
    external = _with_external_buffer(model, weights.buffer)
    _, graphs = tflite._read(external)
    assert graphs[0][1][1].data == model.buffers[weights.buffer]
    assert tflite.unsupported(external) == []
    internal = tflite.to_onnx(tw.write(model), "conv")
    assert tflite.to_onnx(external, "conv").graph.initializer == internal.graph.initializer
    with pytest.raises(ValueError, match="buffers stored outside the FlatBuffer are not supported"):
        tw.read(external)


def test_a_buffer_running_past_the_file_is_an_error():
    model = tw.read((OPS / "float_conv_bias.tflite").read_bytes())
    data = _with_external_buffer(model, model.subgraphs[0].tensors[1].buffer, overstated=1)
    with pytest.raises(ValueError, match="buffer 2 runs past the file"):
        tflite.unsupported(data)


@pytest.mark.parametrize("size", [2, 5])
def test_a_malformed_vtable_is_an_error(size):
    data = bytearray((OPS / "float_conv_bias.tflite").read_bytes())
    root = struct.unpack_from("<I", data, 0)[0]
    vtable = root - struct.unpack_from("<i", data, root)[0]
    struct.pack_into("<H", data, vtable, size)
    with pytest.raises(ValueError, match=f"malformed vtable at byte {vtable}"):
        tflite.unsupported(bytes(data))


def test_a_buffer_shorter_than_a_flatbuffer_is_an_error():
    with pytest.raises(ValueError, match="buffer too short for a FlatBuffer"):
        Table.root(b"\x08\0\0\0TFL")


def _in_subgraph(position: int, op: int, **fields):
    def edit(graph, model):
        for key, value in fields.items():
            setattr(model.subgraphs[position].operators[op], key, value)
    return edit


OUT_OF_RANGE = [
    pytest.param("float_conv_bias", [_operator(0, inputs=[0, 1, -2])],
                 "operator 0 CONV_2D: tensor index -2 is out of range for 4 tensors", id="input"),
    pytest.param("float_conv_bias", [_operator(0, outputs=[4])],
                 "operator 0 CONV_2D: tensor index 4 is out of range for 4 tensors", id="output"),
    pytest.param("float_conv_bias", [_operator(0, outputs=[-1])],
                 "operator 0 CONV_2D: tensor index -1 is out of range for 4 tensors",
                 id="output-left-out"),
    pytest.param("float_conv_bias", [lambda graph, model: setattr(graph, "inputs", [7])],
                 "model inputs: tensor index 7 is out of range for 4 tensors", id="model-input"),
    pytest.param("if_conv", [_in_subgraph(2, 3, outputs=[9])],
                 "subgraph 2 operator 3 CONV_2D: tensor index 9 is out of range for 9 tensors",
                 id="branch-operator"),
    pytest.param("if_conv", [lambda graph, model: setattr(model.subgraphs[1], "outputs", [6])],
                 "subgraph 1 outputs: tensor index 6 is out of range for 6 tensors",
                 id="branch-output"),
    pytest.param("if_conv", [_option(3, 1, "i", 3)],
                 "operator 3 IF: subgraph index 3 is out of range for 3 subgraphs", id="if"),
    pytest.param("float_while", [_option(0, 0, "i", -1)],
                 "operator 0 WHILE: subgraph index -1 is out of range for 3 subgraphs",
                 id="while"),
    pytest.param("float_variable_window", [_option(0, 0, "i", 99)],
                 "operator 0 CALL_ONCE: subgraph index 99 is out of range for 2 subgraphs",
                 id="call-once"),
]


def _operand_left_out(op: int, position: int):
    def edit(graph, model):
        graph.operators[op].inputs[position] = -1
    return edit


OPERAND_LEFT_OUT = [
    pytest.param("float_conv_bias", [_operand_left_out(0, 0)],
                 "operator 0 CONV_2D: input 0 is missing", id="conv-data"),
    pytest.param("conv_bias", [_operand_left_out(0, 1)],
                 "operator 0 CONV_2D: input 1 is missing", id="conv-weights"),
    pytest.param("float_transpose_conv", [_operand_left_out(0, 2)],
                 "operator 0 TRANSPOSE_CONV: input 2 is missing", id="transpose-conv-data"),
    pytest.param("float_mean_channels", [_operand_left_out(0, 1)],
                 "operator 0 MEAN: input 1 is missing", id="mean-axes"),
    pytest.param("strided_slice", [_operand_left_out(0, 3)],
                 "operator 0 STRIDED_SLICE: input 3 is missing", id="strided-slice-strides"),
    pytest.param("float_strided_slice", [_operand_left_out(0, 0)],
                 "operator 0 STRIDED_SLICE: input 0 is missing", id="strided-slice-data"),
    pytest.param("float_relu", [_operand_left_out(0, 0)],
                 "operator 0 RELU: input 0 is missing", id="relu"),
    pytest.param("float_gather", [_operand_left_out(0, 0)],
                 "operator 0 GATHER: input 0 is missing", id="gather"),
    pytest.param("float_svdf", [_operand_left_out(0, 4)],
                 "operator 0 SVDF: input 4 is missing", id="svdf-state"),
    pytest.param("float_concat_channels", [_operand_left_out(0, 1)],
                 "operator 0 CONCATENATION: input 1 is missing", id="concatenation"),
    pytest.param("float_mean_channels", [_operator(0, inputs=[0])],
                 "operator 0 MEAN: input 1 is missing", id="past-the-end"),
    pytest.param("float_conv_bias", [_operator(0, inputs=[])],
                 "operator 0 CONV_2D: input 0 is missing", id="no-inputs"),
    pytest.param("float_max_pool_valid", [_operator(0, inputs=[])],
                 "operator 0 MAX_POOL_2D: input 0 is missing", id="pool-no-inputs"),
    pytest.param("float_variable_window", [_operator(7, inputs=[8])],
                 "operator 7 ASSIGN_VARIABLE: input 1 is missing", id="assign-variable"),
    pytest.param("float_conv_bias", [_operator(0, outputs=[])],
                 "operator 0 CONV_2D: output 0 is missing", id="no-outputs"),
    pytest.param("float_if", [_operator(2, outputs=[])],
                 "operator 2 IF: output 0 is missing", id="if-no-outputs"),
    pytest.param("if_conv", [_in_subgraph(2, 3, outputs=[])],
                 "subgraph 2 operator 3 CONV_2D: output 0 is missing", id="branch"),
]


@pytest.mark.parametrize(("name", "edits", "message"), OUT_OF_RANGE + OPERAND_LEFT_OUT)
def test_an_index_outside_its_table_or_an_operand_left_out_is_an_error(name, edits, message,
                                                                       tmp_path):
    data = _edited(name, *edits)
    with pytest.raises(ValueError, match=re.escape(message)):
        tflite.unsupported(data)
    model = tmp_path / "model.tflite"
    model.write_bytes(data)
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(tmp_path / "model.tgrs")])
    assert result.exit_code != 0 and isinstance(result.exception, SystemExit), result.output
    assert f"Error: {message}" in result.output


def test_every_kind_the_importer_converts_or_folds_has_its_operand_counts():
    assert set(tflite._SUPPORTED) - set(tflite._OPERANDS) == {tflite._DETECTION}


def _first_of_each_kind():
    """(fixture, subgraph, operator) of the first operator of each kind in the ops fixtures."""
    seen, found = set(), []
    for path in sorted(OPS.glob("*.tflite")):
        model = tflite._Model(path.read_bytes())
        for position, subgraph in enumerate(model.subgraphs):
            for index, table in enumerate(subgraph.tables(tflite._SG_OPERATORS)):
                kind = tflite._Operator(model, table, index).kind
                if kind not in seen:
                    seen.add(kind)
                    found.append((path.stem, position, index))
    return found


def test_any_operand_left_out_raises_nothing_but_value_errors():
    """Each input of the first operator of each kind given as -1 or cut from the
    list, and its outputs emptied: the importer refuses with ValueError, returns
    reasons or converts."""
    for name, position, index in _first_of_each_kind():
        op = tw.read((OPS / f"{name}.tflite").read_bytes()).subgraphs[position].operators[index]
        inputs = list(op.inputs)
        variants = [{"inputs": [*inputs[:k], -1, *inputs[k + 1:]]} for k in range(len(inputs))]
        variants += [{"inputs": inputs[:k]} for k in range(len(inputs))]
        variants += [{"outputs": []}] if op.outputs else []
        for fields in variants:
            if fields.get("inputs") == inputs:
                continue
            data = _edited(name, _in_subgraph(position, index, **fields))
            try:
                if not tflite.unsupported(data):
                    tflite.to_onnx(data, "model")
            except ValueError:
                pass


@pytest.mark.parametrize(("name", "position"), [
    ("float_reshape", 1), ("float_split_v", 1), ("float_transpose_conv", 0)])
def test_an_operand_the_conversion_takes_from_the_stated_shapes_may_be_left_out(name, position):
    stated = tflite.to_onnx(_edited(name), "model").graph
    left_out = tflite.to_onnx(_edited(name, _operand_left_out(0, position)), "model").graph
    assert left_out.node == stated.node and left_out.output == stated.output


@pytest.mark.parametrize("name", ["padv2", "float_padv2"])
def test_a_padv2_value_given_as_minus_one_pads_as_one_left_out(name):
    left_out = tflite.to_onnx(_edited(name, _operand_left_out(0, 2)), "model").graph
    def first_two(graph, model):
        graph.operators[0].inputs = graph.operators[0].inputs[:2]
    two_inputs = tflite.to_onnx(_edited(name, first_two), "model").graph
    assert left_out.node == two_inputs.node and left_out.initializer == two_inputs.initializer


def test_one_corrupted_word_raises_nothing_but_value_errors():
    """One 4-byte word in the tables at the front of the file overwritten, 3000
    times; the reader and the importer refuse with ValueError or return reasons."""
    data = (OPS / "conv_bias.tflite").read_bytes()
    rng = random.Random(1)
    limit = min(len(data), 4096)
    for _ in range(3000):
        corrupted = bytearray(data)
        position = rng.randrange(8, limit - 4)
        struct.pack_into("<I", corrupted, position, rng.choice(
            [0, 1, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF, rng.getrandbits(32)]))
        for entry in (tw.read, tflite.unsupported):
            try:
                entry(bytes(corrupted))
            except ValueError:
                pass


def _run(data: bytes, feed: dict, tmp_path: Path) -> list[np.ndarray]:
    """The outputs of the compiled plan of `data` on the host runtime."""
    from tigris.runtime import Session

    model, plan = tmp_path / "model.tflite", tmp_path / "model.tgrs"
    model.write_bytes(data)
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    slow = slow_arena_required(read_binary_plan(plan.read_bytes()))
    with Session(plan, slow_capacity=slow) as session:
        produced = session.run(feed)
        return [produced[output["name"]] for output in session.outputs]


# The constant that reaches each shape operator, the operator's own operands
# after it, and its options.
SHAPE_OPERATORS = {
    "RESHAPE": ((25, 4), [np.array([1, 5, 5, 4], np.int32)], None),
    "SQUEEZE": ((1, 5, 5, 1, 4), [], tw.Options(30, None, {0: ("[i", [3])})),
    "EXPAND_DIMS": ((5, 5, 4), [np.array([0], np.int32)], None),
}


@pytest.mark.parametrize("kind", SHAPE_OPERATORS)
def test_a_shape_operator_on_a_constant_is_folded(kind, tmp_path, require_host_library):
    """float_mul with its second operand computed from a constant by `kind`."""
    shape, operands, options = SHAPE_OPERATORS[kind]
    values = (np.arange(100, dtype=np.float32) / 7 - 5).reshape(shape)

    def through(graph, model):
        y = graph.inputs[1]
        inputs = [_constant(graph, model, values, _FLOAT32),
                  *(_constant(graph, model, v, _INT32) for v in operands)]
        graph.operators.insert(0, tw.Operator(_code(model, kind), inputs, [y], options))
        graph.inputs = graph.inputs[:1]
    data = _edited("float_mul", through)
    _, graphs = tflite._read(data)
    _, tensors, _, _, operators = graphs[0]
    assert [op.kind for op in operators] == ["MUL"]
    np.testing.assert_array_equal(tensors[operators[0].inputs[1]].array(), values.reshape(1, 5, 5, 4))
    assert tflite.unsupported(data) == []
    x = np.random.default_rng(0).standard_normal((1, 5, 5, 4)).astype(np.float32)
    [y] = _run(data, {"x": x}, tmp_path)
    np.testing.assert_array_equal(y, x * values.reshape(1, 5, 5, 4))


_VALUES = (np.arange(100, dtype=np.float32) / 7 - 5).reshape(1, 5, 5, 4)
_OPTIONS_TYPE = {"PACK": 59, "CONCATENATION": 10, "STRIDED_SLICE": 32}  # SLICE has none
# An operator of constants whose options give another shape than the stated one,
# or no result in TFLite Micro: its operands, options, stated shape and reason.
KEPT = [
    pytest.param("PACK", [_VALUES[0, ..., k] for k in range(4)], {0: ("i", 4)}, [1, 5, 5, 4],
                 "PACK: a 1x5x5x4 output where axis 0 gives 4x5x5", id="pack-axis"),
    pytest.param("CONCATENATION", [_VALUES[..., :1], _VALUES[..., 1:]], {0: ("i", 2)},
                 [1, 5, 5, 4], "CONCATENATION: axis 2 for inputs 1x5x5x1, 1x5x5x3",
                 id="concatenation-axis"),
    pytest.param("STRIDED_SLICE", [_VALUES.reshape(1, 5, 4, 5), [0, 0, 0, 0], [1, 5, 4, 5],
                                   [1, 1, 1, 1]], {}, [1, 5, 5, 4],
                 "STRIDED_SLICE: a 1x5x5x4 output where the slice gives 1x5x4x5",
                 id="strided-slice-output"),
    pytest.param("STRIDED_SLICE", [_VALUES, [0], [1], [1]], {}, [1, 5, 5, 4],
                 "STRIDED_SLICE: begin, end or strides with fewer entries than the input has axes",
                 id="strided-slice-short"),
    pytest.param("STRIDED_SLICE", [_VALUES, [0, 0, 0, 0], [1, 5, 5, 4], [-1, 1, 1, 1]],
                 {4: ("i", 1)}, [5, 5, 4], "STRIDED_SLICE: a dropped axis with a negative stride",
                 id="strided-slice-negative-drop"),
    pytest.param("SLICE", [_VALUES, [0], [1]], {}, [1, 5, 5, 4],
                 "SLICE: begin or size with fewer entries than the input has axes", id="slice-short"),
]


@pytest.mark.parametrize(("kind", "operands", "fields", "shape", "reason"), KEPT)
def test_a_constant_operator_without_the_stated_result_is_kept_and_refused(kind, operands, fields,
                                                                            shape, reason):
    """float_mul with its second operand computed by `kind` from constants; the
    fold leaves the operator, so the operator check refuses it."""
    def through(graph, model):
        y = graph.inputs[1]
        graph.tensors[y].shape = shape
        inputs = [_constant(graph, model, np.asarray(v, np.float32), _FLOAT32)
                  if np.asarray(v).dtype.kind == "f"
                  else _constant(graph, model, np.asarray(v, np.int32), _INT32) for v in operands]
        options = (tw.Options(_OPTIONS_TYPE[kind], None, {0: ("i", len(operands)), **fields}
                              if kind == "PACK" else fields) if kind in _OPTIONS_TYPE else None)
        graph.operators.insert(0, tw.Operator(_code(model, kind), inputs, [y], options))
        graph.inputs = graph.inputs[:1]
    data = _edited("float_mul", through)
    _, graphs = tflite._read(data)
    assert [op.kind for op in graphs[0][4]] == [kind, "MUL"]
    assert f"operator 0 {reason}" in tflite.unsupported(data)


@pytest.mark.parametrize(("edits", "reasons"), [
    pytest.param([_option(2, 1, "i", 1)], ["operator 2 PACK: axis 1 for inputs scalar, scalar"],
                 id="pack-axis"),
    pytest.param([_tensor(7, shape=[1, 2])], ["operator 2 PACK: a 1x2 output where axis 0 gives 2"],
                 id="pack-output"),
    pytest.param([_tensor(5, shape=[1]), _tensor(7, shape=[2, 1])],
                 ["operator 1 STRIDED_SLICE: a 1 output where the slice gives scalar",
                  "operator 2 PACK: activations must be all int8 or all float32"],
                 id="strided-slice-output"),
])
def test_int32_shape_arithmetic_of_constants_is_kept_with_its_shape_reason(edits, reasons):
    """float_shape_pack_reshape computes the RESHAPE shape with SHAPE, STRIDED_SLICE
    and PACK in int32; an operator of that chain the fold keeps is refused for
    its shape, and the operators after it no longer read constants."""
    assert tflite.unsupported(_edited("float_shape_pack_reshape", *edits)) == [
        *reasons, "operator 3 RESHAPE: input 1 must be a constant"]


def test_custom_quantization_details_are_written_back():
    model = tw.read((OPS / "conv_bias.tflite").read_bytes())
    stated = model.subgraphs[0].tensors[0].quantization
    stated.custom = b"\x01\x02\x03"
    data = tw.write(model)
    for _ in range(2):
        data = tw.write(tw.read(data))
        assert tw.read(data).subgraphs[0].tensors[0].quantization == stated


def test_quantization_details_of_another_type_are_refused():
    model = tw.read((OPS / "conv_bias.tflite").read_bytes())
    model.subgraphs[0].tensors[0].quantization.custom = b"\x01"
    data = bytearray(tw.write(model))
    quantization = Table.root(bytes(data)).tables(2)[0].tables(0)[0].table(4)
    data[quantization._field(4)] = 2
    with pytest.raises(ValueError, match="quantization details of type 2 are not supported"):
        tw.read(bytes(data))


@pytest.mark.parametrize("begin", [5, -2])
def test_a_dropped_axis_of_a_slice_with_steps(begin, tmp_path, require_host_library):
    """float_strided_slice_steps with stride 1 on its third axis, which is
    dropped at `begin`; the second axis keeps stride 2."""
    data = _edited("float_strided_slice_steps", _data(1, [0, 0, begin, 0], np.int32),
                   _data(3, [1, 2, 1, 1], np.int32), _option(0, 4, "i", 4),
                   _tensor(4, shape=[1, 3, 4]))
    assert tflite.unsupported(data) == []
    slices = [n for n in tflite.to_onnx(data, "slice").graph.node if n.op_type in ("Slice", "Reshape")]
    assert [n.op_type for n in slices] == ["Slice", "Reshape"]
    x = np.random.default_rng(1).standard_normal((1, 6, 6, 4)).astype(np.float32)
    [y] = _run(data, {"x": x}, tmp_path)
    np.testing.assert_array_equal(y, x[:, ::2, begin, :])


@pytest.mark.parametrize("begin", [2, 6])
@pytest.mark.parametrize("stride", [1, 2], ids=["stride-1", "any-stride"])
def test_a_dropped_axis_under_its_begin_mask_bit_takes_index_zero(begin, stride, tmp_path,
                                                                  require_host_library):
    """float_strided_slice_shrink with every begin_mask bit set: TFLite Micro takes
    row 0 of the dropped axis whatever `begin` says, with or without a stride
    other than 1 on another axis."""
    data = _edited("float_strided_slice_shrink", _data(1, [0, begin, 0, 0], np.int32),
                   _data(3, [1, 1, stride, 1], np.int32), _option(0, 0, "i", 15),
                   _tensor(4, shape=[1, 6 // stride, 4]))
    assert tflite.unsupported(data) == []
    x = np.random.default_rng(2).standard_normal((1, 6, 6, 4)).astype(np.float32)
    [y] = _run(data, {"x": x}, tmp_path)
    np.testing.assert_array_equal(y, x[:, 0, ::stride, :])


def test_a_folded_dropped_axis_under_its_begin_mask_bit_takes_index_zero(tmp_path,
                                                                         require_host_library):
    """float_mul with its second operand a constant STRIDED_SLICE that drops
    axis 1 at begin 2 with every begin_mask bit set."""
    values = (np.arange(300, dtype=np.float32) / 7 - 20).reshape(1, 3, 5, 5, 4)

    def through(graph, model):
        y = graph.inputs[1]
        inputs = [_constant(graph, model, values, _FLOAT32),
                  *(_constant(graph, model, np.asarray(v, np.int32), _INT32)
                    for v in ([0, 2, 0, 0, 0], [1, 3, 5, 5, 4], [1, 1, 1, 1, 1]))]
        options = tw.Options(32, None, {0: ("i", 31), 1: ("i", 31), 4: ("i", 2)})
        graph.operators.insert(0, tw.Operator(_code(model, "STRIDED_SLICE"), inputs, [y], options))
        graph.inputs = graph.inputs[:1]
    data = _edited("float_mul", through)
    _, graphs = tflite._read(data)
    _, tensors, _, _, operators = graphs[0]
    assert [op.kind for op in operators] == ["MUL"]
    np.testing.assert_array_equal(tensors[operators[0].inputs[1]].array(), values[:, 0])
    x = np.random.default_rng(0).standard_normal((1, 5, 5, 4)).astype(np.float32)
    [y] = _run(data, {"x": x}, tmp_path)
    np.testing.assert_array_equal(y, x * values[:, 0])
