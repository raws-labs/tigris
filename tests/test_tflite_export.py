"""TFLite export: row geometry, the FlatBuffer writer, and exported models run
on TFLite Micro bit for bit in the arena the planner model predicts."""

import functools
import itertools
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.cli.export import export_report
from tigris.frontends import tflite
from tigris.frontends import tflite_writer as tw
from tigris.frontends.flatbuffer import Table
from tigris.frontends.tflite import Window, window
from tigris.frontends.tflite_export import (
    Export,
    _conv_geometry,
    _host_exact,
    _kind,
    _option,
    arena,
    export,
    host_arena,
    input_rows,
    persistent,
    tflm_tensor_arena,
)
from tigris.frontends.tflite_schema import (
    BUILTIN_OPERATORS,
    MODEL_BUFFERS,
    SG_OPERATORS,
    SG_TENSORS,
    WINDOWED,
)
from tigris.frontends.tflm import TFLM_TARGETS

from conftest import save_onnx

FIXTURES = Path(__file__).parent / "fixtures" / "tflite"
_FLOAT32, _INT32, _INT16, _INT8 = 0, 2, 7, 9
_TARGET = TFLM_TARGETS["arm-none-eabi"]


def _wheel_tail(data: bytes) -> int:
    """The tail the tflite-micro package measures: a 64-bit build with
    compression whose interpreter records its allocations."""
    return persistent(tw.read(data), TFLM_TARGETS["lp64"], compression=True, recording=True)


def _require_tflite_micro():
    """Skip without the tflite-micro package; fail under CI, which installs it."""
    try:
        import tflite_micro  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("tflite-micro is not installed")
        pytest.skip("tflite-micro is not installed")


@pytest.fixture
def tflite_micro():
    _require_tflite_micro()


# Runs a model in a child process: TFLite Micro prints its arena from C.
_CHILD = r"""
import sys, numpy as np
from tflite_micro.python.tflite_micro import runtime
interpreter = runtime.Interpreter.from_bytes(open(sys.argv[1], 'rb').read(), arena_size=16 << 20)
interpreter.print_allocations()
sys.stdout.flush()
inputs = np.load(sys.argv[2])
outputs = {}
for run in range(len(inputs.files)):
    interpreter.set_input(inputs[f"x{run}"], 0)
    interpreter.invoke()
    k = 0
    while True:
        try:
            outputs[f"{run}_{k}"] = interpreter.get_output(k).copy()
        except Exception:
            break
        k += 1
np.savez(sys.argv[3], **outputs)
"""


# Allocates every model given as an argument in one process, a marker line before each.
_ARENA_CHILD = r"""
import os, sys
from tflite_micro.python.tflite_micro import runtime
for index, path in enumerate(sys.argv[1:]):
    os.write(2, f"\n@@model {index}\n".encode())
    interpreter = runtime.Interpreter.from_bytes(open(path, 'rb').read(), arena_size=16 << 20)
    interpreter.print_allocations()
"""


def _arena(printed: str) -> tuple[int, int]:
    return tuple(int(re.search(rf"Arena allocation {kind} (\d+) bytes", printed).group(1))
                 for kind in ("head", "tail"))


def _tflm(data: bytes, inputs: list, tmp_path: Path, name: str):
    """TFLite Micro's arena head and tail and the outputs of each input."""
    model, given, taken = (tmp_path / f"{name}{suffix}" for suffix in (".tflite", "_in.npz", "_out.npz"))
    model.write_bytes(data)
    np.savez(given, **{f"x{k}": x for k, x in enumerate(inputs)})
    result = subprocess.run([sys.executable, "-c", _CHILD, str(model), str(given), str(taken)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    printed = result.stdout + result.stderr
    head, tail = _arena(printed)
    stored = np.load(taken)
    return head, tail, {key: stored[key] for key in stored.files}


class _Builder:
    """Small int8 models with random weights, written by the writer."""

    def __init__(self, seed: int):
        self.tensors, self.buffers, self.codes, self.operators = [], [None], [], []
        self.rng = np.random.default_rng(seed)

    def tensor(self, shape, type_, quantization=None, variable=False):
        self.tensors.append(tw.Tensor(list(shape), type_, 0, f"a{len(self.tensors)}".encode(),
                                      quantization, variable))
        return len(self.tensors) - 1

    def activation(self, shape, scale=0.05, zero_point=-3):
        return self.tensor(shape, _INT8, tw.Quantization([scale], [zero_point]))

    def constant(self, values, type_, quantization=None):
        values = np.asarray(values)
        self.buffers.append(values.tobytes())
        self.tensors.append(tw.Tensor(list(values.shape), type_, len(self.buffers) - 1,
                                      f"c{len(self.tensors)}".encode(), quantization))
        return len(self.tensors) - 1

    def operator(self, kind, inputs, output, options=None, fields=None):
        code = BUILTIN_OPERATORS.index(kind)
        if code not in [c.builtin_code for c in self.codes]:
            self.codes.append(tw.OperatorCode(code, 1, None, min(code, 127)))
        opcode = [c.builtin_code for c in self.codes].index(code)
        self.operators.append(tw.Operator(opcode, list(inputs), [output], None if options is None
                                          else tw.Options(options, None, fields or {})))
        return output

    def conv(self, x, channels, kernel=3, stride=1, padding=0, activation=0, dilation=1,
             depthwise=False, scale=0.05, zero_point=2, groups=1):
        """CONV_2D, or DEPTHWISE_CONV_2D, with padding 0 SAME or 1 VALID."""
        source = self.tensors[x]
        h, w, depth = source.shape[1:]
        span = (kernel - 1) * dilation + 1
        out = ((-(-h // stride), -(-w // stride)) if padding == 0 else
               ((h - span) // stride + 1, (w - span) // stride + 1))
        channels = depth if depthwise else channels
        shape = (1, kernel, kernel, channels) if depthwise else (channels, kernel, kernel,
                                                                 depth // groups)
        scales = [float(v) for v in self.rng.uniform(0.002, 0.01, channels).astype(np.float32)]
        weights = self.constant(self.rng.integers(-127, 128, shape, dtype=np.int8), _INT8,
                                tw.Quantization(scales, [0] * channels, 3 if depthwise else 0))
        bias = self.constant(self.rng.integers(-2000, 2000, channels, dtype=np.int32), _INT32,
                             tw.Quantization([source.quantization.scale[0] * s for s in scales],
                                             [0] * channels))
        y = self.activation([1, *out, channels], scale, zero_point)
        if depthwise:
            fields = {0: ("b", padding), 1: ("i", stride), 2: ("i", stride), 3: ("i", 1),
                      4: ("b", activation), 5: ("i", dilation), 6: ("i", dilation)}
            return self.operator("DEPTHWISE_CONV_2D", [x, weights, bias], y, 2, fields)
        fields = {0: ("b", padding), 1: ("i", stride), 2: ("i", stride), 3: ("b", activation),
                  4: ("i", dilation), 5: ("i", dilation)}
        return self.operator("CONV_2D", [x, weights, bias], y, 1, fields)

    def max_pool(self, x, kernel=3, stride=2, padding=0):
        source = self.tensors[x]
        h, w, depth = source.shape[1:]
        out = ((-(-h // stride), -(-w // stride)) if padding == 0 else
               ((h - kernel) // stride + 1, (w - kernel) // stride + 1))
        q = source.quantization
        y = self.activation([1, *out, depth], q.scale[0], q.zero_point[0])
        return self.operator("MAX_POOL_2D", [x], y, 5, {
            0: ("b", padding), 1: ("i", stride), 2: ("i", stride), 3: ("i", kernel),
            4: ("i", kernel), 5: ("b", 0)})

    def pad(self, x, top, bottom, left, right):
        source = self.tensors[x]
        h, w, depth = source.shape[1:]
        q = source.quantization
        y = self.activation([1, h + top + bottom, w + left + right, depth], q.scale[0],
                            q.zero_point[0])
        paddings = self.constant(np.array([[0, 0], [top, bottom], [left, right], [0, 0]],
                                          np.int32), _INT32)
        return self.operator("PAD", [x, paddings], y, 22)

    def add(self, a, b, scale=0.08, zero_point=4):
        y = self.activation(self.tensors[a].shape, scale, zero_point)
        return self.operator("ADD", [a, b], y, 11, {0: ("b", 0)})

    def mean(self, x, scale=0.02, zero_point=-1):
        """MEAN over rows and columns, keeping the dimensions."""
        axes = self.constant(np.array([1, 2], np.int32), _INT32)
        y = self.activation([1, 1, 1, self.tensors[x].shape[3]], scale, zero_point)
        return self.operator("MEAN", [x, axes], y, 27, {0: ("?", True)})

    def model(self, x, y) -> bytes:
        graph = tw.Subgraph(self.tensors, [x], [y], self.operators, b"main")
        return tw.write(tw.Model(3, self.codes, [graph], self.buffers, b"test"))


def _inverted_residual():
    """1x1 expand, explicit PAD, depthwise 3x3 stride 2, 1x1 project, as
    converters emit an inverted residual block, then a MEAN."""
    b = _Builder(1)
    x = b.activation([1, 32, 32, 8], 0.04, -5)
    y = b.conv(x, 24, 1, activation=3, scale=0.03, zero_point=-128)
    y = b.pad(y, 0, 1, 0, 1)
    y = b.conv(y, 0, 3, 2, padding=1, activation=3, depthwise=True, scale=0.03, zero_point=-128)
    y = b.conv(y, 8, 1, scale=0.05, zero_point=3)
    return b.model(x, b.mean(b.conv(y, 64, 3, scale=0.06, zero_point=-2)))


def _stem():
    """PAD, 7x7 stride 2 convolution, zero-point PAD, 3x3 stride 2 max pool."""
    b = _Builder(2)
    x = b.activation([1, 48, 40, 3], 0.02, 10)
    y = b.conv(b.pad(x, 2, 3, 2, 3), 32, 7, 2, padding=1, activation=1, zero_point=-128)
    y = b.max_pool(b.pad(y, 0, 1, 0, 1), 3, 2, padding=1)
    return b.model(x, b.conv(y, 8, 1, zero_point=0))


def _same_padding():
    """A dilated SAME convolution and a SAME max pool, whose bands need PAD
    and PADV2, and a residual ADD."""
    b = _Builder(3)
    x = b.activation([1, 32, 24, 4], 0.04, -5)
    y = b.max_pool(b.conv(x, 32, 3, dilation=2, zero_point=-3), 3, 2)
    z = b.add(b.conv(y, 32, 3, activation=1, zero_point=-128), y)
    return b.model(x, b.conv(z, 4, 1, scale=0.1, zero_point=1))


def _residual(groups=1):
    """A stride 2 convolution whose output both feeds the next convolution
    and joins it again in an ADD."""
    b = _Builder(4)
    x = b.activation([1, 32, 32, 4], 0.04, 7)
    y = b.conv(x, 32, 3, 2, zero_point=-1, groups=groups)
    z = b.add(b.conv(y, 32, 3, activation=3, zero_point=-128), y)
    return b.model(x, b.conv(z, 4, 1, scale=0.1, zero_point=1))


def _grouped():
    """The residual block with a grouped stride 2 convolution."""
    return _residual(groups=2)


def _row_constant(kind="ADD", shape=(1, 64, 32)):
    """A stride 2 convolution, then an elementwise operator with an int8
    constant of `shape`, broadcast against the rows, columns and channels."""
    b = _Builder(10)
    x = b.activation([1, 128, 128, 8], 0.04, 7)
    y = b.conv(x, 32, 3, 2, zero_point=-1)
    c = b.constant(b.rng.integers(-100, 100, shape, dtype=np.int8), _INT8,
                   tw.Quantization([0.05], [0]))
    y = b.operator(kind, [y, c], b.activation([1, 64, 64, 32], 0.08, 4),
                   {"ADD": 11, "MUL": 21}[kind], {0: ("b", 0)})
    y = b.conv(y, 32, 3, zero_point=-1)
    return b.model(x, b.conv(y, 4, 1, scale=0.1, zero_point=1))


def _with_output(data: bytes, name: bytes) -> bytes:
    """The model with the tensor `name` added to its outputs."""
    model = tw.read(data)
    graph = model.subgraphs[0]
    graph.outputs = [*graph.outputs, [t.name for t in graph.tensors].index(name)]
    return tw.write(model)


@pytest.mark.parametrize("build", [_inverted_residual, _stem, _same_padding, _residual,
                                   _grouped, _row_constant])
def test_exported_model_runs_bit_exact_in_the_predicted_arena(build, tflite_micro, tmp_path):
    data = build()
    result = export(data)
    assert result.data is not None, result.reason
    assert result.region is not None and 2 <= result.bands <= 10
    assert result.arena_after < result.arena_before == tflm_tensor_arena(data)
    rng = np.random.default_rng(7)
    shape = tw.read(data).subgraphs[0].tensors[0].shape
    inputs = [rng.integers(-128, 128, shape, dtype=np.int8) for _ in range(3)]
    head_before, tail_before, _ = _tflm(data, inputs, tmp_path, "before")
    head_after, tail_after, _ = _tflm(result.data, inputs, tmp_path, "after")
    assert head_before == result.arena_before
    assert head_after == result.arena_after
    assert (tail_before, tail_after) == (_wheel_tail(data), _wheel_tail(result.data))
    assert result.persistent_growth == (persistent(tw.read(result.data), _TARGET)
                                        - persistent(tw.read(data), _TARGET))
    # The banded region's own output, exposed, matches as well as the model's.
    graph = tw.read(data).subgraphs[0]
    name = graph.tensors[graph.operators[result.region[1]].outputs[0]].name
    _, _, expected = _tflm(_with_output(data, name), inputs, tmp_path, "expected")
    _, _, actual = _tflm(_with_output(result.data, name), inputs, tmp_path, "actual")
    assert sorted(expected) == sorted(actual) and len(expected) == 2 * len(inputs)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert len(np.unique(np.concatenate([v.ravel() for v in expected.values()]))) > 64


@pytest.mark.parametrize("build", [_inverted_residual, _stem, _same_padding, _residual,
                                   _grouped, _row_constant])
def test_the_added_macs_are_what_the_written_model_adds(build):
    result = export(build())
    assert result.extra_macs > 0
    assert export(result.data).macs - result.macs == result.extra_macs


def _resized():
    """An inverted residual block, then a nearest-neighbor resize no region holds,
    whose output nearly matches the block's peak: the rewrite wins by 432 bytes."""
    b = _Builder(16)
    x = b.activation([1, 32, 32, 8], 0.04, -5)
    y = b.conv(x, 24, 1, activation=3, scale=0.03, zero_point=-128)
    y = b.conv(y, 0, 3, 2, activation=3, depthwise=True, scale=0.03, zero_point=-128)
    y = b.conv(y, 8, 1, scale=0.05, zero_point=3)
    size = b.constant(np.array([59, 59], np.int32), _INT32)
    return b.model(x, b.operator("RESIZE_NEAREST_NEIGHBOR", [y, size],
                                 b.activation([1, 59, 59, 8], 0.05, 3), 74,
                                 {0: ("?", False), 1: ("?", False)}))


@pytest.mark.parametrize("source", ["vww_96_int8.tflite", "pretrainedResnet_quant.tflite",
                                    _inverted_residual, _stem, _same_padding, _residual,
                                    _row_constant, _resized])
def test_the_search_bounds_skip_only_candidates_that_lose(source, monkeypatch):
    data = source() if callable(source) else (FIXTURES / source).read_bytes()
    bounded = export(data)
    assert bounded.data is not None
    monkeypatch.setattr("tigris.frontends.tflite_export._worse", lambda *args: False)
    assert export(data) == bounded


@pytest.mark.parametrize(("kind", "shape"), [("ADD", (64, 64, 32)), ("MUL", (64, 64, 1)),
                                             ("ADD", (1, 64, 64, 32))])
def test_a_constant_that_varies_over_rows_keeps_its_operator_out_of_a_region(kind, shape):
    result = export(_row_constant(kind, shape))
    assert result.region is None or not result.region[0] <= 1 <= result.region[1]


def test_metadata_that_may_hold_tensor_indices_is_not_exported():
    model = tw.read(_inverted_residual())
    model.buffers += [b"1.14.0\0", b"\0" * 16]
    model.metadata = [(b"min_runtime_version", len(model.buffers) - 2),
                      (b"CONVERSION_METADATA", len(model.buffers) - 1)]
    assert export(tw.write(model)).data is not None
    model.buffers.append(np.array([1, 0, 1, -1], np.int32).tobytes())
    model.metadata.append((b"OfflineMemoryAllocation", len(model.buffers) - 1))
    result = export(tw.write(model))
    assert result.data is None
    assert result.reason == "metadata OfflineMemoryAllocation is not supported"


def _signature(model: tw.Model, output: int, subgraph: int = 0) -> tw.Signature:
    sg = model.subgraphs[0]
    return tw.Signature([(b"input", sg.inputs[0])], [(b"output", output)], b"serving_default",
                        subgraph_index=subgraph)


def _op(model: tw.Model, index: int) -> tw.Operator:
    return model.subgraphs[0].operators[index]


DANGLING = [
    pytest.param(lambda m: setattr(_op(m, 0), "opcode", 4),
                 "operator 0: operator code index 4 is out of range for 4 operator codes",
                 id="opcode"),
    pytest.param(lambda m: setattr(_op(m, 1), "inputs", [-2, 5]),
                 "operator 1: tensor index -2 is out of range for 17 tensors", id="input"),
    pytest.param(lambda m: setattr(_op(m, 1), "outputs", [17]),
                 "operator 1: tensor index 17 is out of range for 17 tensors", id="output"),
    pytest.param(lambda m: setattr(_op(m, 1), "intermediates", [20]),
                 "operator 1: tensor index 20 is out of range for 17 tensors", id="intermediate"),
    pytest.param(lambda m: setattr(m.subgraphs[0], "outputs", [-1]),
                 "model outputs: tensor index -1 is out of range for 17 tensors", id="boundary"),
    pytest.param(lambda m: setattr(m.subgraphs[0].tensors[3], "buffer", 11),
                 "tensor 3: buffer index 11 is out of range for 11 buffers", id="tensor-buffer"),
    pytest.param(lambda m: setattr(m, "signatures", [_signature(m, 22)]),
                 "signature 0: tensor index 22 is out of range for 17 tensors", id="signature"),
    pytest.param(lambda m: setattr(m, "signatures", [_signature(m, 16, subgraph=1)]),
                 "signature 0: subgraph index 1 is out of range for 1 subgraph",
                 id="signature-subgraph"),
    pytest.param(lambda m: setattr(m, "metadata", [(b"min_runtime_version", 12)]),
                 "metadata min_runtime_version: buffer index 12 is out of range for 11 buffers",
                 id="metadata"),
    pytest.param(lambda m: setattr(m, "metadata_buffer", [11]),
                 "metadata_buffer: buffer index 11 is out of range for 11 buffers",
                 id="metadata-buffer"),
]


@pytest.mark.parametrize(("edit", "reason"), DANGLING)
def test_an_index_outside_its_table_is_not_exported(edit, reason):
    model = tw.read(_inverted_residual())
    edit(model)
    result = export(tw.write(model))
    assert (result.data, result.reason) == (None, reason)


def _signature_keeps(model: tw.Model, index: int) -> None:
    output = model.subgraphs[0].outputs[0]
    model.signatures = [replace(_signature(model, output),
                                outputs=[(b"output", output), (b"unread", index)])]


def _export_with_unread(keep) -> Export:
    """The inverted residual export with a tensor no operator reads, kept by `keep`."""
    model = tw.read(_inverted_residual())
    sg = model.subgraphs[0]
    sg.tensors.append(tw.Tensor([1, 40, 40, 24], _INT8, 0, b"unread",
                                tw.Quantization([0.05], [0])))
    keep(model, len(sg.tensors) - 1)
    return export(tw.write(model))


@pytest.mark.parametrize("keep", [
    pytest.param(_signature_keeps, id="signature"),
    pytest.param(lambda m, t: setattr(_op(m, 5), "intermediates", [t]), id="intermediate"),
])
def test_the_planned_arena_counts_every_tensor_the_written_model_keeps(keep):
    result = _export_with_unread(keep)
    assert result.data is not None
    assert result.arena_after == arena(tw.read(result.data)) == 40 * 40 * 24


def test_tflite_micro_places_a_tensor_only_a_signature_names(tflite_micro, tmp_path):
    result = _export_with_unread(_signature_keeps)
    head, _, _ = _tflm(result.data, [np.zeros((1, 32, 32, 8), np.int8)], tmp_path, "unread")
    assert head == result.arena_after == 40 * 40 * 24


def test_a_dangling_signature_keeps_the_reason_found_first():
    b = _Builder(9)
    x = b.activation([1, 8, 8, 4])
    paddings = b.constant(np.array([[0, 0], [1, 1], [1, 1], [0, 0]], np.int32), _INT32)
    y = b.operator("MIRROR_PAD", [x, paddings], b.activation([1, 10, 10, 4]), 77, {0: ("b", 0)})
    model = tw.read(b.model(x, b.max_pool(y, 2, 2)))
    model.signatures = [_signature(model, len(model.subgraphs[0].tensors) + 5)]
    result = export(tw.write(model))
    assert result.reason == "TFLite Micro's scratch buffers for MIRROR_PAD are not modeled"


def test_analyze_reports_a_dangling_signature_as_the_export_reason(tmp_path):
    model = tw.read((FIXTURES / "vww_96_int8.tflite").read_bytes())
    model.signatures = [_signature(model, 94)]
    path = tmp_path / "dangling.tflite"
    path.write_bytes(tw.write(model))
    result = CliRunner().invoke(cli, ["analyze", str(path), "-m", "256K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "TFLite Micro export" in line)
    assert row.rstrip().endswith("none   signature 0: tensor index 94 is out of range for 89 "
                                 "tensors")


def _external_buffers(data: bytes) -> bytes:
    """`data` with the bytes of every buffer stored after the FlatBuffer, each
    buffer table naming their offset and size."""
    model = tw.read(data)
    contents, model.buffers = model.buffers, [None] * len(model.buffers)
    out = bytearray(tw.write(model))
    vector, _ = Table.root(bytes(out))._vector(MODEL_BUFFERS)
    for index, content in enumerate(contents):
        if not content:
            continue
        out += bytes(-len(out) % 16)
        offset = len(out)
        out += content
        out += bytes(-len(out) % 8)
        # A vtable with slots 1 and 2 (offset, size), then the table 12 bytes on.
        out += struct.pack("<5H2x", 10, 20, 0, 4, 12)
        table = len(out)
        out += struct.pack("<iQQ", 12, offset, len(content))
        entry = vector + 4 * index
        struct.pack_into("<I", out, entry, table - entry)
    return bytes(out)


def test_analyze_and_compile_report_no_tflite_micro_rows_for_external_buffers(tmp_path):
    source = FIXTURES / "ops" / "conv_bias.tflite"
    path = tmp_path / "conv_bias.tflite"
    path.write_bytes(_external_buffers(source.read_bytes()))
    reports, plans = [], []
    for model in (source, path):
        result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "256K"])
        assert result.exit_code == 0, result.output
        rows = [line.split()[2] for line in result.output.splitlines() if "TFLite Micro" in line]
        assert rows == (["arena", "export"] if model == source else [])
        result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "256K", "--json"])
        assert result.exit_code == 0, result.output
        reports.append(json.loads(result.output))
        out = tmp_path / f"{len(plans)}.tgrs"
        result = CliRunner().invoke(cli, ["compile", str(model), "-m", "256K", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert ("TFLite Micro" in result.output) == (model == source)
        plans.append(out.read_bytes())
    with pytest.raises(ValueError, match="buffers stored outside the FlatBuffer"):
        tw.read(path.read_bytes())
    assert tflm_tensor_arena(path.read_bytes()) is None
    assert reports[0].pop("tflite_micro")["tensor_arena_bytes"] == 512
    assert reports[1].pop("tflite_micro") is None
    assert reports[0] == reports[1] and plans[0] == plans[1]


def test_the_arena_counts_the_scratch_buffers_of_a_reduction(tflite_micro, tmp_path):
    b = _Builder(9)
    x = b.activation([1, 8, 8, 64])
    data = b.model(x, b.mean(x))
    head, _, _ = _tflm(data, [np.zeros((1, 8, 8, 64), np.int8)], tmp_path, "mean")
    # Input, output, then MEAN's int32 accumulator per output, its index and axis arrays.
    assert head == arena(tw.read(data)) == tflm_tensor_arena(data) == 4096 + 64 + 256 + 16 + 16


def _softmax_int16():
    """An int16 SOFTMAX, whose kernel builds two lookup tables."""
    b = _Builder(11)
    x = b.tensor([1, 10], _INT16, tw.Quantization([0.001], [0]))
    y = b.tensor([1, 10], _INT16, tw.Quantization([1 / 32768], [0]))
    return b.model(x, b.operator("SOFTMAX", [x], y, 9, {0: ("f", 1.0)}))


def _batch_matmul(adj_x, adj_y):
    """A float BATCH_MATMUL whose adjoint flags decide which operands its kernel transposes."""
    b = _Builder(12)
    x = b.tensor([1, 6, 4] if adj_x else [1, 4, 6], _FLOAT32)
    w = b.constant(b.rng.standard_normal([1, 5, 6] if adj_y else [1, 6, 5]).astype(np.float32),
                   _FLOAT32)
    return b.model(x, b.operator("BATCH_MATMUL", [x, w], b.tensor([1, 4, 5], _FLOAT32), 101,
                                 {0: ("?", adj_x), 1: ("?", adj_y)}))


def _variable_add():
    """An ADD that reads a variable tensor, whose buffer TFLite Micro keeps in the tail."""
    b = _Builder(13)
    x = b.activation([1, 8])
    return b.model(x, b.add(x, b.tensor([1, 8], _INT8, tw.Quantization([0.05], [-3]), True)))


def _reshape_stretch():
    """A RESHAPE whose output shape holds -1, which its kernel resolves in writable dims."""
    b = _Builder(14)
    x = b.activation([1, 4, 4, 2])
    shape = b.constant(np.array([1, -1], np.int32), _INT32)
    return b.model(x, b.operator("RESHAPE", [x, shape], b.activation([1, -1]), 17,
                                 {0: ("[i", [1, -1])}))


def _gather_nd():
    """GATHER_ND, whose kernel writes its output dims into the tail."""
    b = _Builder(15)
    x = b.activation([4, 3])
    indices = b.constant(np.array([[2], [0]], np.int32), _INT32)
    return b.model(x, b.operator("GATHER_ND", [x, indices], b.activation([2, 3])))


def _repeated(name, copies=16):
    """A one-operator fixture whose operator runs `copies` times on the same
    inputs, each copy writing its own outputs, so that a builtin data record of
    a few bytes moves the tail past the next 16-byte boundary."""
    model = tw.read((FIXTURES / "ops" / name).read_bytes())
    graph = model.subgraphs[0]
    (op,) = graph.operators
    for copy in range(1, copies):
        outputs = []
        for index in op.outputs:
            graph.tensors.append(replace(graph.tensors[index], name=f"copy{copy}_{index}".encode()))
            outputs.append(len(graph.tensors) - 1)
        graph.operators.append(replace(op, outputs=outputs))
        graph.outputs = [*graph.outputs, *outputs]
    return tw.write(model)


# Models for the persistent allocations no fixture shows.
_BRANCHES = {
    "softmax_int16": _softmax_int16, "batch_matmul_adj_x": lambda: _batch_matmul(True, False),
    "batch_matmul_adj_y": lambda: _batch_matmul(False, True),
    "batch_matmul_adj_xy": lambda: _batch_matmul(True, True), "variable_add": _variable_add,
    "reshape_stretch": _reshape_stretch, "gather_nd": _gather_nd,
    **{f"{name} x16": lambda name=name: _repeated(f"{name}.tflite") for name in (
        "arg_max", "arg_min_rows", "cumsum", "div", "float_reduce_max_channels",
        "float_reduce_min_rows", "float_resize_bilinear", "float_resize_nearest", "split_v",
        "float_sum_channels")}}


# TFLite Micro's tail measured on builds for each target, then on the tflite-micro package.
_MEASURED_TAILS = {
    "kws_ref_model.tflite": (6772, 6796, 8016, 8272),
    "ops/abs.tflite": (340, 364, 624, 864),
    "ops/batch_matmul.tflite": (532, 556, 896, 1136),
    "ops/cast_bool.tflite": (548, 572, 960, 1200),
    "ops/conv_bias.tflite": (500, 508, 800, 1056),
    "ops/cumsum.tflite": (404, 412, 704, 944),
    "ops/depth_to_space.tflite": (356, 364, 608, 848),
    "ops/depthwise_bias.tflite": (468, 476, 768, 1024),
    "ops/elu.tflite": (612, 620, 896, 1136),
    "ops/embedding_lookup.tflite": (372, 380, 640, 880),
    "ops/equal.tflite": (400, 424, 704, 944),
    "ops/float_batch_matmul.tflite": (560, 584, 880, 1120),
    "ops/float_boundaries.tflite": (576, 584, 944, 1184),
    "ops/float_cumsum.tflite": (288, 296, 544, 784),
    "ops/float_greater_equal.tflite": (368, 392, 640, 880),
    "ops/float_hard_swish.tflite": (304, 312, 544, 784),
    "ops/float_l2_normalization.tflite": (304, 312, 528, 768),
    "ops/float_l2_pool.tflite": (352, 360, 592, 832),
    "ops/float_leaky_relu.tflite": (320, 328, 560, 800),
    "ops/float_less.tflite": (368, 392, 640, 880),
    "ops/float_less_equal.tflite": (368, 392, 640, 880),
    "ops/float_log_softmax.tflite": (272, 280, 512, 752),
    "ops/float_lstm_unrolled.tflite": (3824, 3928, 6304, 6624),
    "ops/float_max_pool_same.tflite": (368, 376, 592, 832),
    "ops/float_not_equal.tflite": (368, 392, 640, 880),
    "ops/float_pad.tflite": (352, 360, 608, 848),
    "ops/float_padv2.tflite": (368, 376, 624, 864),
    "ops/float_prelu.tflite": (336, 344, 576, 816),
    "ops/float_reduce_all_channels.tflite": (448, 456, 768, 1008),
    "ops/float_relu6.tflite": (304, 312, 528, 768),
    "ops/float_rsqrt.tflite": (304, 328, 560, 800),
    "ops/float_select_v2.tflite": (432, 456, 752, 992),
    "ops/float_shape_arithmetic.tflite": (1344, 1384, 2208, 2480),
    "ops/float_space_to_batch.tflite": (320, 328, 576, 816),
    "ops/float_space_to_depth.tflite": (320, 328, 544, 784),
    "ops/float_squeeze_op.tflite": (320, 328, 576, 816),
    "ops/fully_connected_bias.tflite": (484, 492, 800, 1040),
    "ops/gather_matrix.tflite": (372, 380, 640, 880),
    "ops/log_softmax.tflite": (356, 364, 624, 864),
    "ops/mean_channels.tflite": (388, 396, 672, 912),
    "ops/softmax.tflite": (388, 396, 672, 912),
    "ops/squared_difference.tflite": (500, 524, 816, 1056),
    "ops/strided_slice_steps.tflite": (468, 476, 768, 1008),
    "ops/uint8_classifier.tflite": (1060, 1084, 1680, 1936),
    "pretrainedResnet_quant.tflite": (5188, 5212, 6576, 6832),
    "vww_96_int8.tflite": (26932, 26956, 29680, 29952),
    "arg_max x16": (1520, 1696, 3024, 3264),
    "arg_min_rows x16": (1520, 1696, 3024, 3264),
    "batch_matmul_adj_x": (624, 632, 928, 1168),
    "batch_matmul_adj_xy": (464, 472, 752, 992),
    "batch_matmul_adj_y": (336, 344, 608, 848),
    "cumsum x16": (2656, 2784, 4320, 4560),
    "div x16": (2416, 2608, 4160, 4400),
    "float_reduce_max_channels x16": (2400, 2544, 3984, 4224),
    "float_reduce_min_rows x16": (2400, 2544, 3984, 4224),
    "float_resize_bilinear x16": (1504, 1648, 2960, 3200),
    "float_resize_nearest x16": (1504, 1648, 2960, 3200),
    "float_sum_channels x16": (2400, 2544, 3984, 4224),
    "gather_nd": (356, 364, 624, 864),
    "reshape_stretch": (388, 396, 656, 896),
    "softmax_int16": (2468, 2476, 2752, 2992),
    "split_v x16": (2992, 3248, 5616, 5856),
    "variable_add": (420, 428, 704, 944),
}


def _measured(name: str) -> bytes:
    return _BRANCHES[name]() if name in _BRANCHES else (FIXTURES / name).read_bytes()


@pytest.mark.parametrize("name", sorted(_MEASURED_TAILS))
def test_the_persistent_model_reproduces_tflite_micros_tail(name):
    model = tw.read(_measured(name))
    predicted = tuple(persistent(model, TFLM_TARGETS[target])
                      for target in ("arm-none-eabi", "ilp32", "lp64"))
    wheel = persistent(model, TFLM_TARGETS["lp64"], compression=True, recording=True)
    assert (*predicted, wheel) == _MEASURED_TAILS[name]


@pytest.fixture(scope="module")
def installed_arenas(tmp_path_factory):
    """Head and tail of every measured model, allocated by one TFLite Micro process."""
    _require_tflite_micro()
    names = sorted(_MEASURED_TAILS)
    directory = tmp_path_factory.mktemp("arenas")
    paths = []
    for index, name in enumerate(names):
        paths.append(directory / f"{index}.tflite")
        paths[-1].write_bytes(_measured(name))
    result = subprocess.run([sys.executable, "-c", _ARENA_CHILD, *map(str, paths)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    sections = re.split(r"^@@model (\d+)$", result.stderr, flags=re.MULTILINE)[1:]
    printed = {names[int(index)]: text for index, text in zip(sections[::2], sections[1::2])}
    assert sorted(printed) == names, result.stderr
    return {name: _arena(text) for name, text in printed.items()}


@pytest.mark.parametrize("name", sorted(_MEASURED_TAILS))
def test_the_installed_package_allocates_head_and_tail_as_modeled(name, installed_arenas):
    data = _measured(name)
    head, tail = installed_arenas[name]
    assert (head, tail) == (arena(tw.read(data)), _wheel_tail(data))


def test_the_target_sets_the_persistent_growth(tmp_path):
    data = (FIXTURES / "vww_96_int8.tflite").read_bytes()
    growth = {target: export(data, target).persistent_growth for target in TFLM_TARGETS}
    assert growth == {"arm-none-eabi": 2368, "ilp32": 2368, "lp64": 3248}
    out = tmp_path / "vww.tflite"
    result = CliRunner().invoke(cli, ["export", str(FIXTURES / "vww_96_int8.tflite"), "-o",
                                      str(out), "--target", "lp64"])
    assert result.exit_code == 0, result.output
    assert "TFLite Micro arena, reference kernels on lp64" in result.output.splitlines()
    assert "  persistent      +3.17 KiB" in result.output.splitlines()


def test_a_kind_without_a_persistent_model_is_not_exported():
    result = export((FIXTURES / "ops" / "gelu.tflite").read_bytes())
    assert result.data is None and result.arena_before is None
    assert result.reason == "TFLite Micro's persistent allocations for GELU are not modeled"


def test_a_model_with_scratch_buffers_the_arena_model_does_not_size_is_not_exported():
    b = _Builder(9)
    x = b.activation([1, 8, 8, 4])
    paddings = b.constant(np.array([[0, 0], [1, 1], [1, 1], [0, 0]], np.int32), _INT32)
    y = b.operator("MIRROR_PAD", [x, paddings], b.activation([1, 10, 10, 4]), 77, {0: ("b", 0)})
    result = export(b.model(x, b.max_pool(y, 2, 2)))
    assert result.data is None
    assert result.reason == "TFLite Micro's scratch buffers for MIRROR_PAD are not modeled"
    # A head without those scratch buffers would understate it.
    assert result.arena_before is None


def test_a_pad_equal_to_same_padding_is_folded(tflite_micro, tmp_path):
    b = _Builder(5)
    x = b.activation([1, 16, 16, 8], 0.04, -5)
    data = b.model(x, b.conv(b.pad(x, 1, 1, 1, 1), 0, 3, padding=1, depthwise=True))
    result = export(data)
    assert (result.folded, result.region, result.operators_after) == (1, None, 1)
    # A fold alone: no region, no bands, no added MACs.
    assert export_report(result) == {
        "arena_head_before_bytes": 4640, "arena_head_after_bytes": 4096,
        "persistent_growth_bytes": -112, "change_bytes": -656, "region": None,
        "bands": None, "pads_folded": 1, "macs": 18432, "extra_macs": 0,
        "target": "arm-none-eabi", "reason": None}
    inputs = [np.random.default_rng(8).integers(-128, 128, (1, 16, 16, 8), dtype=np.int8)]
    _, tail_before, expected = _tflm(data, inputs, tmp_path, "before")
    head, tail_after, actual = _tflm(result.data, inputs, tmp_path, "after")
    assert head == result.arena_after < result.arena_before
    assert (tail_before, tail_after) == (_wheel_tail(data), _wheel_tail(result.data))
    np.testing.assert_array_equal(actual["0_0"], expected["0_0"])


def test_analyze_reports_a_pad_fold_without_mac_change(tmp_path):
    b = _Builder(5)
    x = b.activation([1, 16, 16, 8], 0.04, -5)
    model = tmp_path / "fold.tflite"
    model.write_bytes(b.model(x, b.conv(b.pad(x, 1, 1, 1, 1), 0, 3, padding=1, depthwise=True)))
    result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "64K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "TFLite Micro export" in line)
    assert row.rstrip().endswith("-656 B   head 4.53 KiB -> 4.00 KiB, persistent -112 B, "
                                 "reference kernels on arm-none-eabi")


def test_a_pad_other_than_same_padding_is_kept():
    b = _Builder(5)
    x = b.activation([1, 16, 16, 8], 0.04, -5)
    result = export(b.model(x, b.conv(b.pad(x, 2, 0, 1, 1), 0, 3, padding=1, depthwise=True)))
    assert result.data is None and result.folded == 0


@pytest.mark.parametrize(("kind", "fields", "weights", "expected"), [
    pytest.param("CONV_2D", {0: 1, 1: 2, 2: 3, 3: 1, 4: 4, 5: 5}, [8, 3, 2, 4],
                 Window(1, (3, 2), (3, 2), (5, 4)), id="conv"),
    pytest.param("DEPTHWISE_CONV_2D", {1: 2, 2: 3, 3: 1, 4: 1, 5: 4, 6: 5}, [1, 3, 2, 4],
                 Window(0, (3, 2), (3, 2), (5, 4)), id="depthwise"),
    pytest.param("TRANSPOSE_CONV", {0: 1, 1: 2, 2: 3, 4: 4, 5: 5}, [8, 3, 2, 4],
                 Window(1, (3, 2), (3, 2), (1, 1)), id="transpose-conv"),
    pytest.param("MAX_POOL_2D", {1: 2, 2: 3, 3: 4, 4: 5}, None, Window(0, (5, 4), (3, 2), (1, 1)),
                 id="max-pool"),
    pytest.param("AVERAGE_POOL_2D", {0: 1, 3: 4, 4: 5}, None, Window(1, (5, 4), (0, 0), (1, 1)),
                 id="average-pool"),
    pytest.param("L2_POOL_2D", {}, None, Window(0, (0, 0), (0, 0), (1, 1)), id="l2-pool"),
    pytest.param("CONV_2D", {}, [8, 3, 2, 4], Window(0, (3, 2), (0, 0), (1, 1)), id="conv-defaults"),
])
def test_a_window_reads_its_kinds_slots_with_the_schema_defaults(kind, fields, weights, expected):
    assert window(kind, lambda slot, fmt, default: fields.get(slot, default), weights) == expected


@pytest.mark.parametrize("padding", [0, 1])
def test_a_windows_pads_give_its_output_size_under_onnx_arithmetic(padding):
    for extent, kernel, stride, dilation in itertools.product(range(1, 10), range(1, 5),
                                                              range(1, 4), range(1, 3)):
        win = Window(padding, (kernel, 1), (stride, 1), (dilation, 1))
        span = win.spans[0]
        if padding == 1 and extent < span:
            continue
        out = win.outputs((extent, 1))[0]
        begin, end = win.pads((extent, 1), (out, 1))[0]
        assert out == (-(-extent // stride) if padding == 0 else (extent - span) // stride + 1)
        assert (extent + begin + end - span) // stride + 1 == out
        assert 0 <= end - begin <= 1 and (padding == 0 or begin == end == 0)


def test_the_importer_and_export_read_the_same_window_from_a_file():
    seen = 0
    for path in sorted(FIXTURES.glob("**/*.tflite")):
        data = path.read_bytes()
        model, written = tflite._Model(data), tw.read(data)
        for table, graph in zip(model.subgraphs, written.subgraphs):
            tensors = [tflite._Tensor(model, t) for t in table.tables(SG_TENSORS)]
            for i, op_table in enumerate(table.tables(SG_OPERATORS)):
                op, kind = tflite._Operator(model, op_table, i), _kind(written, graph.operators[i])
                if op.kind not in WINDOWED:
                    continue
                assert kind == op.kind
                weights = tensors[op.inputs[1]].shape if kind in tflite._WEIGHTED else None
                assert window(kind, op.option, weights) == window(
                    kind, functools.partial(_option, graph.operators[i]), weights), (path.name, i)
                seen += 1
    assert seen > 50


def test_export_keeps_no_geometry_for_an_input_smaller_than_a_valid_filter():
    for kernel, rows in [(3, 0), (2, 1)]:
        b = _Builder(9)
        x = b.activation([1, 2, 2, 4])
        model = tw.read(b.model(x, b.max_pool(x, kernel, 2, padding=1)))
        graph = model.subgraphs[0]
        assert graph.tensors[graph.operators[0].outputs[0]].shape[1] == rows
        # A 2x2 input under a 3x3 VALID filter: the output formula gives the stated 0 rows.
        geometry = _conv_geometry(model, graph, graph.operators[0], "MAX_POOL_2D")
        assert (geometry is None) == (kernel == 3)


def test_export_declines_when_no_rewrite_lowers_the_arena():
    b = _Builder(6)
    x = b.activation([1, 4, 4, 4])
    result = export(b.model(x, b.max_pool(x, 2, 2, padding=1)))
    assert result.data is None and result.reason == "no smaller arena found"
    # No strided or pooling operator: no region to band.
    assert export((FIXTURES / "ops" / "conv_bias.tflite").read_bytes()).data is None


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 33])
@pytest.mark.parametrize(("kernel", "stride", "dilation"),
                         [(1, 1, 1), (3, 1, 1), (3, 2, 1), (5, 2, 1), (3, 1, 2), (2, 2, 1), (7, 3, 1)])
@pytest.mark.parametrize("same", [True, False])
def test_input_rows_are_the_rows_every_tap_of_the_band_reads(rows, kernel, stride, dilation, same):
    span = (kernel - 1) * dilation + 1
    if not same and rows < span:
        return
    out = -(-rows // stride) if same else (rows - span) // stride + 1
    top = max((out - 1) * stride + span - rows, 0) // 2 if same else 0
    for lo in range(out):
        for hi in range(lo + 1, out + 1):
            taps = [o * stride - top + j * dilation for o in range(lo, hi) for j in range(kernel)]
            first, last = min(taps), max(taps) + 1
            assert input_rows(lo, hi, stride, span, top, rows) == (
                max(first, 0), min(last, rows), max(-first, 0), max(last - rows, 0))


def _onnx_conv_chain(path):
    rng = np.random.default_rng(0)
    nodes, weights, x = [], [], "x"
    for k, (cin, cout, kernel, stride, pads) in enumerate(
            [(4, 8, 3, 2, [1, 1, 1, 1]), (8, 8, 5, 1, [2, 2, 2, 2]), (8, 4, 3, 2, [0, 0, 1, 1])]):
        weights += [numpy_helper.from_array(rng.standard_normal((cout, cin, kernel, kernel))
                                            .astype(np.float32), f"w{k}"),
                    numpy_helper.from_array(rng.standard_normal(cout).astype(np.float32), f"b{k}")]
        nodes.append(helper.make_node("Conv", [x, f"w{k}", f"b{k}"], [f"y{k}"],
                                      kernel_shape=[kernel, kernel], strides=[stride, stride],
                                      pads=pads))
        x = f"y{k}"
    save_onnx(path.parent, "chain", nodes,
              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 33, 33])],
              [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 4, 8, 8])], weights,
              ir_version=8, filename=path.name)
    # (stride, kernel, pad_top, input rows, output rows) of each convolution.
    return [(2, 3, 1, 33, 17), (1, 5, 2, 17, 17), (2, 3, 0, 17, 8)]


@pytest.mark.parametrize("budget", ["3K", "6K", "12K"])
def test_input_rows_reproduce_the_runtime_tiles(budget, require_host_library, tmp_path):
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.runtime import Session

    geometry = _onnx_conv_chain(tmp_path / "chain.onnx")
    plan = tmp_path / "chain.tgrs"
    written = CliRunner().invoke(cli, ["compile", str(tmp_path / "chain.onnx"), "-m", budget,
                                       "-m", "1M", "-o", str(plan)])
    assert written.exit_code == 0, written.output
    stages = read_binary_plan(plan.read_bytes())["stages"]
    with Session(plan, slow_capacity=1 << 20) as session:
        _, events, _ = session.trace({"x": np.zeros((1, 33, 33, 4), np.float32)})
    checked = 0
    for k, event in enumerate(events):
        if event["kind"] != "stage_begin" or event["path"] not in ("tiled", "chain"):
            continue
        stage = stages[event["stage"]]
        members = stages[event["stage"]:event["stage"] + stage["chain_len"]] \
            if event["path"] == "chain" else [stage]
        ops = [geometry[op] for member in members for op in member["ops"]]
        tiles = [(e["row0"], e["row1"]) for e in events[k + 1:] if e["kind"] == "tile_begin"
                 and e["stage"] == event["stage"]]
        out_rows = ops[-1][4]

        def rows(lo, hi):
            for stride, kernel, top, size, _ in reversed(ops):
                lo, hi, _, _ = input_rows(lo, hi, stride, kernel, top, size)
            return lo, hi

        assert any(all(rows(t * h, min((t + 1) * h, out_rows)) == tile
                       for t, tile in enumerate(tiles))
                   for h in range(1, out_rows + 1)), (event, tiles)
        checked += 1
    assert checked


def test_the_writer_reproduces_every_fixture(tflite_micro):
    from tflite_micro.tensorflow.lite.micro.python import schema_py_generated as schema

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if hasattr(value, "__dict__"):
            return {k: plain(v) for k, v in vars(value).items()}
        return value

    paths = sorted(FIXTURES.rglob("*.tflite"))
    assert len(paths) > 300
    for path in paths:
        data = path.read_bytes()
        written = tw.write(tw.read(data))
        assert plain(schema.ModelT.InitFromPackedBuf(written, 0)) == plain(
            schema.ModelT.InitFromPackedBuf(data, 0)), path.name


def test_the_writer_knows_which_option_tables_hold_vectors(tflite_micro):
    from tflite_micro.tensorflow.lite.micro.python import schema_py_generated as schema

    parts = re.split(r"^class (\w+)\(object\):", Path(schema.__file__).read_text(), flags=re.M)
    classes = dict(zip(parts[1::2], parts[2::2]))
    for union, counts, vectors in (("BuiltinOptions", tw._OPTION_FIELDS, set(tw._VECTOR_OPTIONS)),
                                   ("BuiltinOptions2", tw._OPTION2_FIELDS, tw._VECTOR_OPTIONS2)):
        members = {v: k for k, v in vars(getattr(schema, union)).items()
                   if not k.startswith("_") and isinstance(v, int) and v}
        assert len(counts) == max(members) + 1
        found = set()
        for code, name in members.items():
            source = classes[name]
            slots = [int(v) // 2 - 2 for v in re.findall(r"self\._tab\.Offset\((\d+)\)", source)]
            assert counts[code] == max(slots, default=-1) + 1, name
            packer = classes[name + "T"]
            if "StartVector" in packer or "CreateString" in packer or "CreateNumpyVector" in packer:
                found.add(code)
        assert found == set(vectors), union


def test_the_writer_keeps_a_fully_connected_quant_spec(tflite_micro):
    from tflite_micro.tensorflow.lite.micro.python import schema_py_generated as schema

    model = tw.read((FIXTURES / "ops" / "fully_connected_bias.tflite").read_bytes())
    operator = model.subgraphs[0].operators[0]
    operator.options.fields[5] = ("[B", [3, 1, 4])
    written = tw.write(model)
    options = schema.ModelT.InitFromPackedBuf(written, 0).subgraphs[0].operators[0].builtinOptions
    assert options.quantSpec.tolist() == [3, 1, 4]
    assert tw.read(written).subgraphs[0].operators[0].options.fields[5][1] == [3, 1, 4]


def test_the_writer_refuses_a_field_it_does_not_know(monkeypatch):
    monkeypatch.setitem(tw._KNOWN, "tensor", tw._KNOWN["tensor"] - {3})
    with pytest.raises(ValueError, match="the tensor table sets field 3"):
        tw.read((FIXTURES / "ops" / "conv_bias.tflite").read_bytes())


def test_export_command_writes_the_model_and_reports_the_arena(tmp_path):
    out = tmp_path / "vww.tflite"
    result = CliRunner().invoke(cli, ["export", str(FIXTURES / "vww_96_int8.tflite"), "-o", str(out)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "vww_96_int8.tflite   31 operators"
    assert lines[1].startswith("operators 2-3 computed in 4 row bands")
    assert "  head            45.00 KiB   72.00 KiB before" in lines
    assert tflm_tensor_arena(out.read_bytes()) == 45 * 1024
    assert lines[-1].startswith(f"wrote {out}   ")


def test_export_command_names_a_single_banded_operator(monkeypatch, tmp_path):
    found = Export(1024, 512, 64, b"model", region=(1, 1), bands=3, operators_before=2,
                   operators_after=7)
    monkeypatch.setattr("tigris.frontends.tflite_export.export", lambda data, target: found)
    result = CliRunner().invoke(cli, ["export", str(FIXTURES / "vww_96_int8.tflite"),
                                      "-o", str(tmp_path / "out.tflite")])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1] == "operator 1 computed in 3 row bands; 7 operators after export"


def test_export_command_copies_a_model_it_cannot_lower(tmp_path):
    source = FIXTURES / "kws_ref_model.tflite"
    out = tmp_path / "kws.tflite"
    result = CliRunner().invoke(cli, ["export", str(source), "-o", str(out),
                                      "--json", str(tmp_path / "kws.json")])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1] == f"no smaller arena found; {out} is an unchanged copy"
    assert out.read_bytes() == source.read_bytes()
    report = json.loads((tmp_path / "kws.json").read_text())
    assert report == {"runtime": "tflite-micro", "source": "model",
                      **export_report(export(source.read_bytes()))}
    assert report["reason"] == "no smaller arena found"
    assert report["change_bytes"] is None
    gain = CliRunner().invoke(cli, ["export", str(source), "-o", str(out), "--require-gain"])
    assert gain.exit_code == 3
    assert f"no smaller arena found; {out} is an unchanged copy" in gain.output


def test_export_command_reports_a_written_model_as_json(tmp_path):
    source = FIXTURES / "vww_96_int8.tflite"
    result = CliRunner().invoke(cli, ["export", str(source), "-o", str(tmp_path / "vww.tflite"),
                                      "--json", str(tmp_path / "vww.json"), "--require-gain"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "vww.json").read_text())
    assert report["runtime"] == "tflite-micro" and report["source"] == "model"
    assert report["arena_head_after_bytes"] == 45 * 1024 and report["reason"] is None


@pytest.mark.parametrize("option", [["--cpu", "M7"], ["--allow-input-overwrite"],
                                    ["--arena", "64K"], ["--method", "32K"],
                                    ["--max-extra-macs", "5%"], ["--header", "x.h"]])
def test_executorch_options_are_a_usage_error_for_tflite(option, tmp_path):
    out = tmp_path / "vww.tflite"
    result = CliRunner().invoke(cli, ["export", str(FIXTURES / "vww_96_int8.tflite"), "-o",
                                      str(out), *option])
    assert result.exit_code == 2
    assert f"Error: {option[0]} applies to ExecuTorch programs" in result.output
    assert not out.exists()


def test_export_refuses_a_file_of_another_format(tmp_path):
    other = tmp_path / "model.bin"
    other.write_bytes(b"\0" * 64)
    result = CliRunner().invoke(cli, ["export", str(other)])
    assert result.exit_code == 1
    assert "Error: model.bin is neither a TFLite model nor an ExecuTorch program" in result.output


def test_analyze_reports_the_tflite_micro_arena_after_export():
    result = CliRunner().invoke(cli, ["analyze", str(FIXTURES / "vww_96_int8.tflite"), "-m", "64K"])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "TFLite Micro export" in line)
    assert row.rstrip().endswith("head 72.00 KiB -> 45.00 KiB, persistent +2.31 KiB, +0.25% MACs, "
                                 "reference kernels on arm-none-eabi")
    declined = CliRunner().invoke(cli, ["analyze", str(FIXTURES / "kws_ref_model.tflite"), "-m", "64K"])
    assert ("TFLite Micro export        none   no smaller arena found, reference kernels on "
            "arm-none-eabi") in declined.output
    # Declined before any figure was planned: the reason alone.
    unmodeled = CliRunner().invoke(cli, ["analyze", str(FIXTURES / "ops" / "gelu.tflite"), "-m", "64K"])
    row = next(line for line in unmodeled.output.splitlines() if "TFLite Micro export" in line)
    assert row.rstrip().endswith("none   TFLite Micro's persistent allocations for GELU are not "
                                 "modeled")
    reduction = CliRunner().invoke(cli, ["analyze", str(FIXTURES / "ops" / "mean_channels.tflite"),
                                         "-m", "64K"])
    assert ("TFLite Micro arena    368 B   its planner and reduction scratch; persistent "
            "allocations and other kernel scratch excluded") in reduction.output


def _variable_in_one_subgraph(path: Path) -> Path:
    """float_variable_window without its CALL_ONCE initializer subgraph: one
    subgraph that keeps a RESOURCE tensor."""
    model = tw.read((FIXTURES / "ops" / "float_variable_window.tflite").read_bytes())
    assert model.operator_codes[model.subgraphs[0].operators[0].opcode].builtin_code == \
        BUILTIN_OPERATORS.index("CALL_ONCE")
    model.subgraphs = model.subgraphs[:1]
    model.subgraphs[0].operators = model.subgraphs[0].operators[1:]
    path.write_bytes(tw.write(model))
    return path


def test_analyze_leaves_out_both_tflite_micro_rows_where_the_arena_is_not_modeled(tmp_path):
    for model, reason in [
            (FIXTURES / "ops" / "float_while_countdown.tflite",
             "models with several subgraphs are not exported"),
            (_variable_in_one_subgraph(tmp_path / "variable.tflite"),
             "a tensor type TFLite Micro's arena model does not know")]:
        result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "64K"])
        assert result.exit_code == 0, result.output
        assert "TFLite Micro" not in result.output
        report = json.loads(CliRunner().invoke(cli, ["analyze", str(model), "-m", "64K",
                                                     "--json"]).output)
        assert report["tflite_micro"] is None
        declined = CliRunner().invoke(cli, ["export", str(model), "-o", str(tmp_path / "out"),
                                            "--json", str(tmp_path / "out.json")])
        assert declined.exit_code == 1
        assert f"no model written: {reason}" in declined.output
        assert not (tmp_path / "out").exists() and not (tmp_path / "out.json").exists()


def test_top_level_help_shows_every_command_summary_whole():
    # A real terminal, unlike CliRunner, formats help 78 columns wide from 80 columns up.
    result = subprocess.run([sys.executable, "-c", "from tigris.cli import main; main()", "--help"],
                            capture_output=True, text=True, check=True,
                            env={**os.environ, "COLUMNS": "80", "NO_COLOR": "1"})
    rows = result.stdout.split("Commands:\n", 1)[1].splitlines()
    assert ("  export   Rewrite a TFLite model or an ExecuTorch program to run in less RAM."
            in rows)
    assert not [row for row in rows if row.endswith("...")]


def _tflite_micro_json(model) -> dict:
    result = CliRunner().invoke(cli, ["analyze", str(model), "-m", "64K", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert (report["report"], report["version"]) == ("tigris-analysis", 2)
    return report["tflite_micro"]


def _export_json(model) -> dict:
    return _tflite_micro_json(model)["export"]


def test_analyze_json_reports_the_written_export():
    assert _export_json(FIXTURES / "vww_96_int8.tflite") == {
        "arena_head_before_bytes": 73728, "arena_head_after_bytes": 46080,
        "persistent_growth_bytes": 2368, "change_bytes": -25280,
        "region": [2, 3], "bands": 4, "pads_folded": 0,
        "macs": 7489664, "extra_macs": 18432, "target": "arm-none-eabi", "reason": None}


def test_analyze_json_reports_a_declined_export(tmp_path):
    declined = {"arena_head_after_bytes": None, "persistent_growth_bytes": None,
                "change_bytes": None, "region": None, "bands": None, "pads_folded": None,
                "extra_macs": None, "target": "arm-none-eabi", "reason": "no smaller arena found"}
    assert _export_json(FIXTURES / "kws_ref_model.tflite") == {
        **declined, "arena_head_before_bytes": 16000, "macs": 2656768}
    # Planned without convolutions: zero MACs, not null.
    assert _export_json(FIXTURES / "ops" / "abs.tflite") == {
        **declined, "arena_head_before_bytes": 288, "macs": 0}
    # MEAN's scratch buffers count in both figures, so they agree.
    reported = _tflite_micro_json(FIXTURES / "ops" / "mean_channels.tflite")
    assert reported["tensor_arena_bytes"] == 368
    assert reported["export"] == {**declined, "arena_head_before_bytes": 368, "macs": 0}
    # Declined before planning: no head is reported.
    model = tw.read(_inverted_residual())
    model.buffers.append(np.array([1, 0, 1, -1], np.int32).tobytes())
    model.metadata = [(b"OfflineMemoryAllocation", len(model.buffers) - 1)]
    path = tmp_path / "offline.tflite"
    path.write_bytes(tw.write(model))
    assert _export_json(path) == {
        **declined, "arena_head_before_bytes": None, "macs": None,
        "reason": "metadata OfflineMemoryAllocation is not supported"}


# tigris run on TFLite Micro

def _run_tflite(model: Path, given: Path, out: Path, *options):
    return CliRunner().invoke(cli, ["run", str(model), "--input", str(given), "--output", str(out),
                                    *options])


def test_run_gives_tflite_micros_recorded_bytes(tflite_micro, tmp_path):
    """The recorded inputs of the keyword-spotting model, as .bin and as
    .npy, give the outputs TFLite Micro recorded, byte for byte."""
    recorded = np.load(FIXTURES / "kws_ref_model_tflm.npz")
    for k, (row, expected) in enumerate(zip(recorded["inputs"], recorded["outputs"])):
        suffix = ".npy" if k % 2 else ".bin"
        given = tmp_path / f"x{k}{suffix}"
        if k % 2:
            np.save(given, row)
        else:
            row.tofile(given)
        out = tmp_path / f"y{k}.bin"
        result = _run_tflite(FIXTURES / "kws_ref_model.tflite", given, out)
        assert result.exit_code == 0, result.output
        assert out.read_bytes() == expected.tobytes()


@pytest.mark.usefixtures("require_host_library")
def test_run_gives_the_same_bytes_for_a_model_and_its_plan(tflite_micro, tmp_path):
    model = FIXTURES / "kws_ref_model.tflite"
    plan = tmp_path / "kws.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(model), "-m", "16K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    recorded = np.load(FIXTURES / "kws_ref_model_tflm.npz")
    recorded["inputs"][0].tofile(tmp_path / "x.bin")
    for name in (model, plan):
        out = tmp_path / f"{name.suffix[1:]}.bin"
        result = _run_tflite(name, tmp_path / "x.bin", out)
        assert result.exit_code == 0, result.output
        assert out.read_bytes() == recorded["outputs"][0].tobytes()


@pytest.mark.parametrize("name", ["vww_96_int8", "pretrainedResnet_quant"])
def test_run_gives_the_same_bytes_for_a_model_and_its_export(tflite_micro, tmp_path, name):
    model = FIXTURES / f"{name}.tflite"
    written = tmp_path / f"{name}_export.tflite"
    result = CliRunner().invoke(cli, ["export", str(model), "-o", str(written)])
    assert result.exit_code == 0, result.output
    assert written.read_bytes() != model.read_bytes()
    recorded = np.load(FIXTURES / f"{name}_tflm.npz")
    for k, (row, expected) in enumerate(zip(recorded["input_0"], recorded["output_0"])):
        row.tofile(tmp_path / f"x{k}.bin")
        for program in (model, written):
            out = tmp_path / f"{program.stem}_{k}.bin"
            result = _run_tflite(program, tmp_path / f"x{k}.bin", out)
            assert result.exit_code == 0, result.output
            assert out.read_bytes() == expected.tobytes()


def test_run_report_and_json(tflite_micro, tmp_path):
    from importlib.metadata import version

    model = FIXTURES / "kws_ref_model.tflite"
    np.load(FIXTURES / "kws_ref_model_tflm.npz")["inputs"][0].tofile(tmp_path / "x.bin")
    out = tmp_path / "y.npz"
    result = _run_tflite(model, tmp_path / "x.bin", out)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"kws_ref_model.tflite   TFLite Micro {version('tflite-micro')}, reference kernels",
        "",
        f"wrote {out}   1x12 int8",
    ]
    with np.load(out) as values:
        assert values.files == ["Identity"]
    result = _run_tflite(model, tmp_path / "x.bin", tmp_path / "y.bin", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "runtime": "tflite-micro", "runtime_version": version("tflite-micro"),
        "backend": "reference", "output": str(tmp_path / "y.bin"),
        "tensors": [{"name": "Identity", "dtype": "int8", "shape": [1, 12]}],
    }


def test_the_host_arena_is_head_tail_and_the_planning_bound():
    # 35 tensors, no scratch buffers: (32 + 40) x 35 + 16 x 12 + 64 bytes of planning.
    data = (FIXTURES / "kws_ref_model.tflite").read_bytes()
    assert host_arena(data) == 16000 + _wheel_tail(data) + 2776 == 27048


# Allocates each model given as an argument in the arena host_arena sizes for it;
# one that fails there is tried in 64 MB to tell a short arena from a kernel
# TFLite Micro lacks.
_HOST_ARENA_CHILD = r"""
import sys
from tflite_micro.python.tflite_micro import runtime
from tigris.frontends.tflite_export import host_arena
for path in sys.argv[1:]:
    data = open(path, 'rb').read()
    try:
        runtime.Interpreter.from_bytes(data, arena_size=host_arena(data))
    except RuntimeError:
        try:
            runtime.Interpreter.from_bytes(data, arena_size=64 << 20)
        except RuntimeError:
            print("cannot load", path)
            continue
        print("too small", path)
"""

_GELU = {"float_gelu", "float_gelu_approximate", "float_gelu_approximate_wide", "float_gelu_wide",
         "gelu", "gelu_approximate", "gelu_approximate_offset", "gelu_wide"}


def _fixtures() -> list[Path]:
    return sorted([*FIXTURES.glob("*.tflite"), *FIXTURES.glob("ops/*.tflite")])


def test_the_host_arena_fits_every_model_tflite_micro_loads(tflite_micro):
    """TFLite Micro allocates every fixture it can load in the arena
    host_arena gives it, the models the arena model covers and those the
    upper bound sizes; the only ones it cannot load use GELU, which the
    package does not register."""
    paths = [str(path) for path in _fixtures()]
    assert len(paths) > 340
    result = subprocess.run([sys.executable, "-c", _HOST_ARENA_CHILD, *paths],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    assert [line for line in lines if line.startswith("too small")] == []
    assert {Path(line.split(" ", 2)[2]).stem for line in lines} == _GELU


def test_the_bound_covers_what_tflite_micro_allocates(tflite_micro, tmp_path):
    """For each fixture the arena model does not cover and TFLite Micro
    loads, the upper bound is at least the head and tail TFLite Micro
    records: several subgraphs, LSTM, SVDF, TRANSPOSE_CONV, MIRROR_PAD,
    ADD_N and the detection postprocess."""
    paths = [path for path in _fixtures() if path.stem not in _GELU
             and _host_exact(tw.read(path.read_bytes())) is None]
    assert len(paths) == 37
    result = subprocess.run([sys.executable, "-c", _ARENA_CHILD, *map(str, paths)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    sections = re.split(r"^@@model (\d+)$", result.stderr, flags=re.MULTILINE)[1:]
    assert len(sections) == 2 * len(paths)
    for index, text in zip(sections[::2], sections[1::2]):
        path = paths[int(index)]
        assert host_arena(path.read_bytes()) >= sum(_arena(text)), path.name


def test_run_reports_tflite_micros_message_for_a_kind_it_lacks(tflite_micro, tmp_path):
    np.zeros((1, 6, 6, 4), np.float32).tofile(tmp_path / "x.bin")
    result = _run_tflite(FIXTURES / "ops" / "float_gelu_approximate.tflite", tmp_path / "x.bin",
                         tmp_path / "y.bin")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: TFLite Micro cannot run float_gelu_approximate.tflite: Didn't find op for "
        "builtin opcode 'GELU'"]


@pytest.mark.parametrize("name", ["lstm", "float_lstm", "float_if", "float_while", "if_conv",
                                  "while_conv", "detection_fast", "float_detection_regular"])
def test_run_executes_a_model_the_bound_sizes(tflite_micro, tmp_path, name):
    """LSTM, control flow and the detection postprocess run in the arena the
    upper bound gives and produce what TFLite Micro produces in 16 MB; the
    detection outputs past the detections it reports are left unwritten."""
    from tflite_micro.python.tflite_micro import runtime

    data = (FIXTURES / "ops" / f"{name}.tflite").read_bytes()
    graph = tflite.describe(data)["graph"]
    rng = np.random.default_rng(0)
    given, values = [], []
    for k, item in enumerate(graph["inputs"]):
        dtype = np.dtype(item["dtype"])
        value = (rng.integers(-100, 100, item["shape"]).astype(dtype) if dtype.kind == "i"
                 else rng.uniform(0, 1, item["shape"]).astype(dtype))
        np.save(tmp_path / f"x{k}.npy", value)
        given += ["--input", f"{item['name']}={tmp_path / f'x{k}.npy'}"]
        values.append(value)
    out = tmp_path / "y.npz"
    result = CliRunner().invoke(cli, ["run", str(FIXTURES / "ops" / f"{name}.tflite"), *given,
                                      "--output", str(out)])
    assert result.exit_code == 0, result.output
    interpreter = runtime.Interpreter.from_bytes(data, arena_size=16 << 20)
    for k, value in enumerate(values):
        interpreter.set_input(value, k)
    interpreter.invoke()
    expected = [interpreter.get_output(k) for k in range(len(graph["outputs"]))]
    with np.load(out) as written:
        got = [written[item["name"]] for item in graph["outputs"]]
    if "detection" in name:
        count = int(expected[3][0])
        assert got[3] == expected[3] and count > 0
        got, expected = ([a[:, :count] for a in arrays[:3]] for arrays in (got, expected))
    assert [a.tobytes() for a in got] == [a.tobytes() for a in expected]


def test_run_folds_what_tflite_micro_prints_into_the_error(tflite_micro, tmp_path, capfd,
                                                           monkeypatch):
    """An arena TFLite Micro finds too small, and a kernel it lacks: the line
    it prints from C becomes the reason, and nothing else reaches stderr."""
    import tigris.frontends.tflite_export as te

    monkeypatch.setattr(te, "host_arena", lambda data: 24000)
    np.load(FIXTURES / "kws_ref_model_tflm.npz")["inputs"][0].tofile(tmp_path / "x.bin")
    result = _run_tflite(FIXTURES / "kws_ref_model.tflite", tmp_path / "x.bin",
                         tmp_path / "y.bin")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: TFLite Micro reports the arena too small for kws_ref_model.tflite: Failed to "
        "resize buffer. Requested: 16000, available 15952, missing: 48; tigris run sized it at "
        "24000 bytes from its model of TFLite Micro's arena"]
    # The reference build has no GELU kernel.
    monkeypatch.setattr(te, "host_arena", lambda data: 1 << 16)
    np.zeros((1, 6, 6, 4), np.float32).tofile(tmp_path / "x.bin")
    result = _run_tflite(FIXTURES / "ops" / "float_gelu_approximate.tflite", tmp_path / "x.bin",
                         tmp_path / "y.bin")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: TFLite Micro cannot run float_gelu_approximate.tflite: Didn't find op for "
        "builtin opcode 'GELU'"]
    assert capfd.readouterr().err == ""
    assert not (tmp_path / "y.bin").exists()


def test_run_without_tflite_micro_names_the_package(tmp_path, monkeypatch):
    for name in ("tflite_micro", "tflite_micro.python", "tflite_micro.python.tflite_micro"):
        monkeypatch.setitem(sys.modules, name, None)
    np.zeros((1, 49, 10, 1), np.int8).tofile(tmp_path / "x.bin")
    result = _run_tflite(FIXTURES / "kws_ref_model.tflite", tmp_path / "x.bin", tmp_path / "y.bin")
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "Error: running a TFLite model needs tflite-micro, which has wheels for Linux x86_64: "
        "pip install tflite-micro"]


def test_run_refuses_executorch_for_a_tflite_model(tmp_path):
    result = _run_tflite(FIXTURES / "kws_ref_model.tflite", tmp_path / "x.bin", tmp_path / "y.bin",
                         "--executorch", str(tmp_path))
    assert result.exit_code == 2
    assert "Error: --executorch applies to ExecuTorch programs" in result.output


def test_custom_options_read_as_tflite_micro_reads_them():
    """By position, a slot of the map's width cut to 32 bits; by key,
    flexbuffers' AsInt32: a float truncated, a bool as 1, a string parsed
    whole as a decimal integer or else 0, a vector as its length, a null or
    a missing key as 0."""
    import flatbuffers.flexbuffers as flex

    from tigris.frontends.flatbuffer import flexbuffer_int, flexbuffer_slots

    assert flexbuffer_slots(bytes(flex.Dumps({"delay_length": 7, "z": -3})), 2) == [7, -3]
    data = bytes(flex.Dumps({"a": 7, "b": " -12", "c": [1, "x"], "d": 2.9, "e": True, "f": "1x",
                             "h": None}))
    assert [flexbuffer_int(data, key) for key in "abcdefhz"] == [7, -12, 2, 2, 1, 0, 0, 0]


# Loads the model in the first argument in each arena size that follows and
# prints ok or fail for each, then its allocations in 1 MB.
_SIZES_CHILD = r"""
import sys
from tflite_micro.python.tflite_micro import runtime
data = open(sys.argv[1], 'rb').read()
for size in map(int, sys.argv[2:]):
    try:
        runtime.Interpreter.from_bytes(data, arena_size=size)
        print("ok", size)
    except RuntimeError:
        print("fail", size)
sys.stdout.flush()
runtime.Interpreter.from_bytes(data, arena_size=1 << 20).print_allocations()
"""


def _loads(data: bytes, sizes, tmp_path: Path) -> tuple[list[bool], str]:
    """Whether TFLite Micro loads the model in each arena size, and what it
    prints of its allocations."""
    path = tmp_path / "model.tflite"
    path.write_bytes(data)
    result = subprocess.run([sys.executable, "-c", _SIZES_CHILD, str(path), *map(str, sizes)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return [line.startswith("ok") for line in result.stdout.splitlines()], result.stderr


def _per_channel_fully_connected(channels: int) -> bytes:
    """An int8 FULLY_CONNECTED of one input element with per-channel weights and bias."""
    b = _Builder(16)
    scales = [float(v) for v in b.rng.uniform(0.002, 0.01, channels).astype(np.float32)]
    x = b.activation([1, 1])
    weights = b.constant(b.rng.integers(-127, 128, (channels, 1), dtype=np.int8), _INT8,
                         tw.Quantization(scales, [0] * channels))
    bias = b.constant(b.rng.integers(-100, 100, channels, dtype=np.int32), _INT32,
                      tw.Quantization([0.05 * s for s in scales], [0] * channels))
    return b.model(x, b.operator("FULLY_CONNECTED", [x, weights, bias],
                                 b.activation([1, channels], 0.1, 1), 8))


def test_the_exact_arena_holds_the_temporaries_a_kernel_prepares_with(tflite_micro, tmp_path):
    """A kernel's temporary tensors sit past the tail while it prepares, before
    the head is planned: per-channel quantization of 1000 channels takes more
    there than head and planning do, and the arena holds it."""
    from tigris.frontends.tflite_export import HOST_TARGET
    from tigris.frontends.tflm import tflm_planning

    data = _per_channel_fully_connected(1000)
    model = tw.read(data)
    planned = (arena(model) + _wheel_tail(data)
               + tflm_planning(len(model.subgraphs[0].tensors), 0, HOST_TARGET))
    assert host_arena(data) > planned
    assert _loads(data, [host_arena(data), planned], tmp_path)[0] == [True, False]


def _custom_model(custom: str, options, inputs: list, outputs: list) -> bytes:
    """One custom operator. Each operand is (shape, type) or (array, type) for
    a constant, optionally followed by its quantization."""
    import flatbuffers.flexbuffers as flex

    tensors, buffers = [], [None]
    for value, type_, *quantization in [*inputs, *outputs]:
        buffer = 0
        if isinstance(value, np.ndarray):
            buffers.append(value.tobytes())
            value, buffer = value.shape, len(buffers) - 1
        tensors.append(tw.Tensor(list(value), type_, buffer, f"t{len(tensors)}".encode(),
                                 *quantization))
    ins, outs = list(range(len(inputs))), list(range(len(inputs), len(tensors)))
    code = tw.OperatorCode(BUILTIN_OPERATORS.index("CUSTOM"), 1, custom.encode(), 32)
    op = tw.Operator(0, ins, outs, None, None if options is None else bytes(flex.Dumps(options)))
    graph = tw.Subgraph(tensors, [k for k in ins if not tensors[k].buffer], outs, [op], b"main")
    return tw.write(tw.Model(3, [code], [graph], buffers, b"test"))


_UINT8, _BOOL, _UINT64, _UINT32 = 3, 6, 12, 15


def _int16(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.int16)


# A loadable one-operator model of each kind the package registers that no
# fixture carries: (custom code, custom options, inputs, outputs).
_BOUND_KINDS = {
    "BasicClassifier": (None, [([4], _INT32), (np.array([0, 1], np.int32), _INT32),
                               (np.array([5, 5], np.int32), _INT32)], [([2], _BOOL), ([2], _INT32)]),
    "CIRCULAR_BUFFER": ({"cycles_max": 1}, [([1, 1, 1, 8], _INT8)], [([1, 3, 1, 8], _INT8)]),
    "SignalDelay": ({"delay_length": 7}, [([3, 16], _INT16)], [([3, 16], _INT16)]),
    "SignalEnergy": ({"end_index": 8, "start_index": 1}, [([18], _INT16)], [([9], _UINT32)]),
    "SignalFftAutoScale": (None, [([64], _INT16)], [([64], _INT16), ([], _INT32)]),
    "SignalFilterBank": ({"num_channels": 6}, [([33], _UINT32), *[(_int16(n), _INT16)
                                                                  for n in (20, 20, 7, 7, 7)]],
                         [([7], _UINT64)]),
    "SignalFilterBankLog": ({"input_correction_bits": 3, "output_scale": 1600}, [([6], _UINT32)],
                            [([6], _INT16)]),
    "SignalFilterBankSpectralSubtraction": (
        {"alternate_one_minus_smoothing": 15729, "alternate_smoothing": 655, "clamping": False,
         "min_signal_remaining": 819, "num_channels": 6, "one_minus_smoothing": 15729,
         "smoothing": 655, "smoothing_bits": 10, "spectral_subtraction_bits": 14},
        [([6], _UINT32)], [([6], _UINT32), ([6], _UINT32)]),
    "SignalFilterBankSquareRoot": (None, [([6], _UINT64), (np.array(3, np.int32), _INT32)],
                                   [([6], _UINT32)]),
    "SignalFramer": ({"frame_size": 40, "frame_step": 16, "prefill": False}, [([2, 16], _INT16)],
                     [([2, 1, 40], _INT16), ([], _BOOL)]),
    "SignalIrfft": ({"T": 7, "fft_length": 64}, [([66], _INT16)], [([64], _INT16)]),
    "SignalOverlapAdd": ({"T": 0, "frame_step": 8}, [([2, 3, 16], _FLOAT32)],
                         [([2, 24], _FLOAT32)]),
    "SignalPCAN": ({"snr_shift": 6}, [([6], _UINT32), ([6], _UINT32), (_int16(10), _INT16)],
                   [([6], _UINT32)]),
    "SignalRfft": ({"T": 2, "fft_length": 64}, [([64], _INT32)], [([66], _INT32)]),
    "SignalStacker": ({"num_channels": 8, "stacker_left_context": 2, "stacker_right_context": 1,
                       "stacker_step": 1}, [([8], _INT16)], [([32], _INT16), ([], _BOOL)]),
    "SignalWindow": ({"shift": 12}, [([16], _INT16), (_int16(16), _INT16)], [([16], _INT16)]),
    # A pruned int8 tensor with per-channel zero points: the largest decode state and its copy
    # of the zero points. The 16-byte header holds the type, 2, and the version, 1, twice.
    "TFLM_DECODE": (None, [(np.zeros(8, np.uint8), _UINT8),
                           (np.array([2, 1, 0, 0, 1] + [0] * 15, np.uint8), _UINT8)],
                    [([4, 2], _INT8, tw.Quantization([0.1, 0.2], [1, -1]))]),
}

# Allocates each model given as an argument in the arena host_arena sizes for it,
# a marker line before each.
_BOUND_CHILD = r"""
import os, sys
from tflite_micro.python.tflite_micro import runtime
from tigris.frontends.tflite_export import host_arena
for index, path in enumerate(sys.argv[1:]):
    os.write(2, f"\n@@model {index}\n".encode())
    data = open(path, 'rb').read()
    runtime.Interpreter.from_bytes(data, arena_size=host_arena(data)).print_allocations()
"""


def _kernel_allocations(data: bytes) -> tuple[int, int, int]:
    """For a one-operator model, the bytes and count of the persistent buffers
    the bound gives the kernel and the graph boundary's pointer arrays, and
    the head: every placed tensor and scratch buffer, 16-byte aligned."""
    from tigris.frontends.tflite_export import (HOST_TARGET, _host_kind, _host_options,
                                                _host_tensors)
    from tigris.frontends.tflm import (TFLM_BOUND_KERNELS, _tflm_bound_scratch, _tflm_bytes,
                                       _tflm_layout)

    model = tw.read(data)
    graph = model.subgraphs[0]
    (op,) = graph.operators
    kind = _host_kind(model, op.opcode)
    options = _host_options(op, kind)
    tensors = [(*t, t[1]) for t in _host_tensors(model, graph)]
    _, init, prepare, _ = TFLM_BOUND_KERNELS[kind]
    sizes = [_tflm_layout(init, HOST_TARGET)[0]] if init else []
    sizes += prepare(tensors, op.inputs, op.outputs, options, HOST_TARGET) if prepare else []
    sizes += [HOST_TARGET.pointer * len(graph.inputs), HOST_TARGET.pointer * len(graph.outputs)]
    scratch = _tflm_bound_scratch(kind, tensors, op.inputs, op.outputs, options, HOST_TARGET,
                                  False)
    head = sum((size + 15) // 16 * 16 for size in [
        *(_tflm_bytes(t) for t in tensors if not t[2] and not t[5]), *scratch])
    return sum(sizes), len(sizes), head


def test_the_bound_sizes_each_kernel_as_tflite_micro_allocates_it(tflite_micro, tmp_path):
    """For one operator of each kind no fixture carries, TFLite Micro loads in
    the arena the bound gives, and the kernel's persistent buffers, counted
    with the graph boundary's, and the head it records are those the bound
    takes."""
    paths = []
    for kind, (options, inputs, outputs) in _BOUND_KINDS.items():
        paths.append(tmp_path / f"{kind}.tflite")
        paths[-1].write_bytes(_custom_model(kind, options, inputs, outputs))
    result = subprocess.run([sys.executable, "-c", _BOUND_CHILD, *map(str, paths)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    sections = re.split(r"^@@model (\d+)$", result.stderr, flags=re.MULTILINE)[1:]
    assert len(sections) == 2 * len(paths)
    for index, text in zip(sections[::2], sections[1::2]):
        path = paths[int(index)]
        buffers = re.search(r"'Persistent buffer data' used \d+ bytes with alignment overhead "
                            r"\(requested (\d+) bytes for (\d+) allocations\)", text)
        head = _arena(text)[0]
        assert (int(buffers.group(1)), int(buffers.group(2)), head) == _kernel_allocations(
            path.read_bytes()), path.stem


def _compression_metadata(tensor: int, value_buffer: int, bit_width: int) -> bytes:
    """TFLite Micro's compression Metadata table with one LutTensor in one subgraph."""
    import flatbuffers

    b = flatbuffers.Builder(0)
    b.StartObject(3)
    b.PrependInt32Slot(0, tensor, 0)
    b.PrependUint32Slot(1, value_buffer, 0)
    b.PrependUint8Slot(2, bit_width, 0)
    tables = b.EndObject()
    for fields in (1, 2):
        b.StartVector(4, 1, 4)
        b.PrependUOffsetTRelative(tables)
        vector = b.EndVector()
        b.StartObject(fields)
        if fields == 2:
            b.PrependUint32Slot(0, 1, 1)
        b.PrependUOffsetTRelativeSlot(fields - 1, vector, 0)
        tables = b.EndObject()
    b.Finish(tables)
    return bytes(b.Output())


def _compressed_fully_connected() -> bytes:
    """A float FULLY_CONNECTED whose 64x32 weights are 2-bit indices into a
    table of four values, which its kernel decompresses into a scratch buffer."""
    rng = np.random.default_rng(17)
    indices = np.packbits(np.unpackbits(rng.integers(0, 4, 64 * 32).astype(np.uint8)[:, None],
                                        axis=1)[:, -2:]).tobytes()
    values = np.array([-1.0, -0.5, 0.5, 1.0], np.float32).tobytes()
    tensors = [tw.Tensor([1, 32], _FLOAT32, 0, b"x"), tw.Tensor([64, 32], _FLOAT32, 1, b"w"),
               tw.Tensor([1, 64], _FLOAT32, 0, b"y")]
    graph = tw.Subgraph(tensors, [0], [2], [tw.Operator(0, [0, 1, -1], [2],
                                                         tw.Options(8, None, {}))], b"main")
    code = BUILTIN_OPERATORS.index("FULLY_CONNECTED")
    return tw.write(tw.Model(3, [tw.OperatorCode(code, 1, None, code)], [graph],
                             [None, indices, values, _compression_metadata(1, 2, 2)], b"test",
                             None, [(b"COMPRESSION_METADATA", 3)]))


def _offline_planned_abs() -> bytes:
    """ABS whose input the OfflineMemoryAllocation metadata places at 64 KB."""
    tensors = [tw.Tensor([1000], _FLOAT32, 0, b"x"), tw.Tensor([1000], _FLOAT32, 0, b"y")]
    code = BUILTIN_OPERATORS.index("ABS")
    graph = tw.Subgraph(tensors, [0], [1], [tw.Operator(0, [0], [1])], b"main")
    offsets = np.array([1, 0, 2, 65536, -1], np.int32).tobytes()
    return tw.write(tw.Model(3, [tw.OperatorCode(code, 1, None, code)], [graph], [None, offsets],
                             b"test", None, [(b"OfflineMemoryAllocation", 1)]))


def test_the_bound_places_offline_planned_and_decompressed_buffers(tflite_micro, tmp_path):
    """TFLite Micro loads in the arena the bound gives a model with an
    offline planned buffer at 64 KB, and one whose weights it decompresses
    into an 8 KB scratch buffer; the compression records it allocates are
    those the bound counts for each tensor."""
    from tigris.frontends.tflite_export import HOST_TARGET
    from tigris.frontends.tflm import _tflm_layout

    data = _offline_planned_abs()
    loaded, printed = _loads(data, [host_arena(data)], tmp_path)
    assert loaded == [True] and _arena(printed)[0] == 65536 + 4000
    data = _compressed_fully_connected()
    loaded, printed = _loads(data, [host_arena(data)], tmp_path)
    assert loaded == [True] and _arena(printed)[0] == 128 + 256 + 64 * 32 * 4
    records = sum(_tflm_layout(name, HOST_TARGET)[0]
                  for name in ("CompressionTensorData", "LookupTableData"))
    assert re.search(r"'Persistent compression data' used \d+ bytes with alignment overhead "
                     r"\(requested (\d+) bytes for 3 allocations\)", printed).group(1) == str(
                         records + 3 * HOST_TARGET.pointer)


def test_the_bound_gives_a_resource_variable_the_bytes_its_value_declares(tflite_micro, tmp_path):
    """ASSIGN_VARIABLE's resource buffer takes the bytes of its value's
    declared shape, not of the shape GATHER writes into it."""
    tensors = [tw.Tensor([1], _INT32, 0, b"coords"), tw.Tensor([10], _FLOAT32, 1, b"params"),
               tw.Tensor([100000], _FLOAT32, 0, b"gathered"), tw.Tensor([], 13, 0, b"resource")]
    codes = [tw.OperatorCode(BUILTIN_OPERATORS.index(kind))
             for kind in ("VAR_HANDLE", "GATHER", "ASSIGN_VARIABLE")]
    ops = [tw.Operator(0, [], [3], tw.Options(111, None, {0: ("s", b""), 1: ("s", b"v")})),
           tw.Operator(1, [1, 0], [2], tw.Options(23, None, {0: ("i", 0), 1: ("i", 0)})),
           tw.Operator(2, [3, 2], [], tw.Options(113, None, {}))]
    graph = tw.Subgraph(tensors, [0], [2], ops, b"main")
    data = tw.write(tw.Model(3, codes, [graph], [None, np.arange(10, dtype=np.float32).tobytes()],
                             b"test"))
    assert host_arena(data) > 400000
    assert _loads(data, [host_arena(data)], tmp_path)[0] == [True]


def test_run_refuses_a_model_tflite_micro_cannot_plan(tflite_micro, tmp_path):
    """No arena is enough where TFLite Micro plans scratch buffers from
    memory it never sets, a MEAN in a subgraph nothing invokes, or for a
    model without subgraphs: run refuses both instead of crashing."""
    mean = tw.Subgraph([tw.Tensor([2, 8], _FLOAT32, 0, b"a"), tw.Tensor([1], _INT32, 1, b"axis"),
                        tw.Tensor([2], _FLOAT32, 0, b"m")], [0], [2],
                       [tw.Operator(1, [0, 1], [2], tw.Options(27, None, {0: ("b", 0)}))],
                       b"other")
    main = tw.Subgraph([tw.Tensor([4], _FLOAT32, 0, b"x"), tw.Tensor([4], _FLOAT32, 0, b"y")],
                       [0], [1], [tw.Operator(0, [0], [1])], b"main")
    codes = [tw.OperatorCode(BUILTIN_OPERATORS.index(kind)) for kind in ("ABS", "MEAN")]
    unreached = tw.Model(3, codes, [main, mean], [None, np.array([1], np.int32).tobytes()],
                         b"test")
    np.zeros(4, np.float32).tofile(tmp_path / "x.bin")
    for name, model, reason in [
            ("unreached", unreached, "no IF, WHILE or CALL_ONCE reaches subgraph 1, so TFLite "
             "Micro plans the scratch buffers of its MEAN from uninitialized memory"),
            ("empty", replace(unreached, subgraphs=[]), "the model has no subgraphs")]:
        (tmp_path / f"{name}.tflite").write_bytes(tw.write(model))
        result = _run_tflite(tmp_path / f"{name}.tflite", tmp_path / "x.bin", tmp_path / "y.bin")
        assert result.exit_code == 1
        assert result.output.splitlines() == [
            f"Error: tigris run cannot size TFLite Micro's arena for {name}.tflite: {reason}"]
