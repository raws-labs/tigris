"""TFLite models rewritten so that TFLite Micro runs them in a smaller arena.

A run of int8 operators that ends at or after a strided or pooling operator is
computed in row bands. Per band, SLICE takes the input rows the band reads,
halo included; PAD (or PADV2 with -128 in front of a max pool) adds the
padding rows the original operator applied at that band's edge and its column
padding; copies of the operators run with VALID padding; CONCATENATION joins
the band outputs into the original output tensor. An explicit PAD that equals
the SAME padding of the convolution reading it is folded into that
convolution. Every candidate is scored by the arena head TFLite Micro plans
for it, tensors and kernel scratch buffers, plus the bytes its persistent
allocations grow by on the target; the best one is written only when it lowers
that total.
"""

import itertools
from dataclasses import dataclass, replace

import numpy as np

from tigris.frontends import tflite_writer as tw
from tigris.frontends.qdq import same_padding
from tigris.frontends.tflite import (
    _BUILTIN_OPERATORS,
    _TENSOR_TYPES,
    _TFLM_KERNELS,
    TFLM_TARGETS,
    TflmTarget,
    tflm_arena,
    tflm_persistent,
    tflm_tensor_bytes,
)

# CONCATENATION takes at most this many inputs in TFLite Micro.
MAX_BANDS = 10
# Longest run of operators tried as one region.
MAX_REGION = 12
# The TFLM_TARGETS entry export plans for unless told otherwise.
DEFAULT_TARGET = "arm-none-eabi"

_INT8, _INT32 = _TENSOR_TYPES.index("INT8"), _TENSOR_TYPES.index("INT32")
_SAME, _VALID = 0, 1
_CODES = {name: code for code, name in enumerate(_BUILTIN_OPERATORS)}
# BuiltinOptions union types of the operators the rewrite adds.
_OPTION_TYPES = {"CONCATENATION": 10, "PAD": 22, "PADV2": 43, "SLICE": 48}
_CONVS = ("CONV_2D", "DEPTHWISE_CONV_2D")
_POOLS = ("MAX_POOL_2D", "AVERAGE_POOL_2D")
_PADS = ("PAD", "PADV2")
_UNARY = ("RELU", "RELU6", "LOGISTIC", "TANH", "HARD_SWISH", "LEAKY_RELU", "QUANTIZE", "ABS")
_BINARY = ("ADD", "SUB", "MUL", "MAXIMUM", "MINIMUM")
# Kernels whose scratch tflm_arena does not place; int4 weights fail earlier as an unknown type.
_UNMODELED_SCRATCH = ("CUSTOM", "MIRROR_PAD", "UNIDIRECTIONAL_SEQUENCE_LSTM", "SVDF",
                       "TRANSPOSE_CONV", "ADD_N")
# Metadata that holds no tensor index, so it stays valid when tensors are renumbered.
_METADATA = (b"min_runtime_version", b"CONVERSION_METADATA")

def input_rows(out_start: int, out_end: int, stride: int, kernel: int, pad_top: int,
               in_rows: int) -> tuple[int, int, int, int]:
    """The input rows [start, end) a spatial operator reads to produce output
    rows [out_start, out_end), and the padding rows it adds above and below
    them: `kernel` is the effective kernel height, dilation included, and
    `pad_top` the operator's padding above the whole input. This is the
    runtime's tile mapping, clamped to the input."""
    start = out_start * stride - pad_top
    end = (out_end - 1) * stride - pad_top + kernel
    return max(start, 0), min(end, in_rows), max(-start, 0), max(end - in_rows, 0)


@dataclass
class Export:
    """What export found for a model. `data` is the rewritten model, None when
    no rewrite lowers TFLite Micro's total and `reason` says why.
    `arena_before` is None when the model was declined before its head was planned."""
    arena_before: int | None
    arena_after: int | None = None
    persistent_growth: int = 0
    data: bytes | None = None
    reason: str | None = None
    region: tuple[int, int] | None = None  # first and last operator banded, as in the input
    bands: int = 0
    folded: int = 0
    operators_before: int = 0
    operators_after: int = 0
    macs: int = 0
    extra_macs: int = 0
    target: str = DEFAULT_TARGET

    @property
    def change(self) -> int:
        """Change of TFLite Micro's total arena in bytes."""
        if self.data is None:
            return 0
        return self.arena_after - self.arena_before + self.persistent_growth


@dataclass
class _Geometry:
    stride: int
    kernel: int
    top: int
    bottom: int
    left: int
    right: int


@dataclass
class _Node:
    """An operator a region may hold: its activation inputs by position, its
    output, the row geometry of a spatial operator and its MACs."""
    kind: str
    inputs: list
    output: int
    geometry: _Geometry | None = None
    macs: int = 0
    downsamples: bool = False


def _kind(model: tw.Model, op: tw.Operator) -> str:
    return _code_kind(model, op.opcode)


def _code_kind(model: tw.Model, index: int) -> str:
    code = model.operator_codes[index]
    if code.custom_code is not None:
        return "CUSTOM"
    builtin = max(code.builtin_code, code.deprecated_builtin_code)
    return _BUILTIN_OPERATORS[builtin] if 0 <= builtin < len(_BUILTIN_OPERATORS) else "UNKNOWN"


def _option(op: tw.Operator, slot: int, fmt: str, default):
    options = op.options
    if options is None:
        return default
    if slot in options.fields:
        return options.fields[slot][1]
    return options.source.scalar(slot, fmt, default) if options.source is not None else default


def _data(model: tw.Model, tensor: tw.Tensor) -> bytes | None:
    return model.buffers[tensor.buffer] if tensor.buffer < len(model.buffers) else None


def _is_constant(model: tw.Model, tensor: tw.Tensor) -> bool:
    return bool(_data(model, tensor))


def _values(model: tw.Model, tensor: tw.Tensor) -> np.ndarray | None:
    dtype = {_INT32: "<i4", _TENSOR_TYPES.index("INT64"): "<i8"}.get(tensor.type)
    data = _data(model, tensor)
    if dtype is None or not data:
        return None
    return np.frombuffer(data, dtype).reshape(tensor.shape)


def _same_quantization(a: tw.Tensor, b: tw.Tensor) -> bool:
    qa, qb = a.quantization, b.quantization
    return (qa is not None and qb is not None and qa.scale == qb.scale
            and qa.zero_point == qb.zero_point)


def _band_tensor(tensor: tw.Tensor) -> bool:
    shape = tensor.shape or []
    return (len(shape) == 4 and shape[0] == 1 and tensor.type == _INT8
            and not tensor.is_variable and tensor.quantization is not None
            and len(tensor.quantization.scale or ()) == 1)


def _conv_geometry(model: tw.Model, sg: tw.Subgraph, op: tw.Operator, kind: str):
    x, y = sg.tensors[op.inputs[0]], sg.tensors[op.outputs[0]]
    if kind in _CONVS:
        w = sg.tensors[op.inputs[1]]
        if not _is_constant(model, w) or len(w.shape or []) != 4:
            return None
        kh, kw = w.shape[1], w.shape[2]
        slots = (4, 5) if kind == "CONV_2D" else (5, 6)
        dw, dh = _option(op, slots[0], "i", 1), _option(op, slots[1], "i", 1)
    else:
        kh, kw = _option(op, 4, "i", 0), _option(op, 3, "i", 0)
        dw = dh = 1
    padding = _option(op, 0, "b", _SAME)
    sw, sh = _option(op, 1, "i", 0), _option(op, 2, "i", 0)
    if min(kh, kw, sw, sh, dw, dh) < 1:
        return None
    eh, ew = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    (h, w_), (oh, ow) = x.shape[1:3], y.shape[1:3]
    if padding == _SAME:
        if (oh, ow) != (-(-h // sh), -(-w_ // sw)):
            return None
        top, bottom = same_padding(h, oh, eh, sh)
        left, right = same_padding(w_, ow, ew, sw)
    elif padding == _VALID:
        if h < eh or w_ < ew or (oh, ow) != ((h - eh) // sh + 1, (w_ - ew) // sw + 1):
            return None
        top = bottom = left = right = 0
    else:
        return None
    return _Geometry(sh, eh, top, bottom, left, right)


def _node(model: tw.Model, sg: tw.Subgraph, op: tw.Operator) -> _Node | None:
    """The operator as a region member, or None when a region cannot hold it."""
    kind = _kind(model, op)
    if len(op.outputs) != 1 or op.options2 is not None or op.intermediates:
        return None
    y = sg.tensors[op.outputs[0]]
    if not _band_tensor(y) or y.shape[1] < 2:
        return None
    if kind in _CONVS or kind in _POOLS:
        if not _band_tensor(sg.tensors[op.inputs[0]]):
            return None
        if kind in _CONVS and len(op.inputs) > 2 and op.inputs[2] >= 0 and not _is_constant(
                model, sg.tensors[op.inputs[2]]):
            return None
        geometry = _conv_geometry(model, sg, op, kind)
        if geometry is None:
            return None
        # A padded average pool divides by the taps inside the image, which no PAD reproduces.
        if kind == "AVERAGE_POOL_2D" and any((geometry.top, geometry.bottom, geometry.left,
                                              geometry.right)):
            return None
        return _Node(kind, [(0, op.inputs[0])], op.outputs[0], geometry,
                     _op_macs(sg, op, kind), geometry.stride > 1 or kind in _POOLS)
    if kind in _PADS:
        x = sg.tensors[op.inputs[0]]
        pads = _values(model, sg.tensors[op.inputs[1]])
        if not _band_tensor(x) or pads is None or pads.shape != (4, 2) or pads[0].any() \
                or pads[3].any():
            return None
        if kind == "PADV2" and (len(op.inputs) < 3 or not _is_constant(
                model, sg.tensors[op.inputs[2]])):
            return None
        (top, bottom), (left, right) = (int(v) for v in pads[1]), (int(v) for v in pads[2])
        if min(top, bottom, left, right) < 0:
            return None
        return _Node(kind, [(0, op.inputs[0])], op.outputs[0],
                     _Geometry(1, 1, top, bottom, left, right))
    if kind in _UNARY:
        x = sg.tensors[op.inputs[0]]
        if not _band_tensor(x) or x.shape != y.shape:
            return None
        return _Node(kind, [(0, op.inputs[0])], op.outputs[0])
    if kind in _BINARY:
        inputs = []
        for position, index in enumerate(op.inputs):
            t = sg.tensors[index]
            if _is_constant(model, t):
                # Broadcasting aligns trailing axes, so a constant's row axis is its third last.
                if len(t.shape or []) >= 3 and t.shape[-3] > 1:
                    return None
                continue
            if not _band_tensor(t) or t.shape[1] not in (1, y.shape[1]):
                return None
            inputs.append((position, index))
        return _Node(kind, inputs, op.outputs[0])
    return None


def _op_macs(sg: tw.Subgraph, op: tw.Operator, kind: str) -> int:
    """Multiply-accumulates of a convolution or fully connected operator, else 0."""
    if kind not in (*_CONVS, "FULLY_CONNECTED") or len(op.inputs) < 2:
        return 0
    w, y = sg.tensors[op.inputs[1]], sg.tensors[op.outputs[0]]
    if not w.shape or not y.shape:
        return 0
    per_output = int(np.prod(w.shape[1:])) if kind != "DEPTHWISE_CONV_2D" else \
        w.shape[1] * w.shape[2]
    return int(np.prod(y.shape)) * per_output


def _macs(model: tw.Model, sg: tw.Subgraph) -> int:
    """Multiply-accumulates of the model's convolutions and fully connected operators."""
    return sum(_op_macs(sg, op, _kind(model, op)) for op in sg.operators)


class _Edit:
    """A subgraph being rewritten: the original records are shared and new
    tensors, buffers and operator codes appended."""

    def __init__(self, model: tw.Model, sg: tw.Subgraph):
        self.model, self.sg = model, sg
        self.tensors = list(sg.tensors)
        self.buffers = list(model.buffers)
        self.codes = list(model.operator_codes)
        self.names = {t.name for t in sg.tensors}
        self.constants = {}
        self.added_tensors = 0
        self.empty = next((i for i, data in enumerate(self.buffers) if not data), None)
        if self.empty is None:
            self.buffers.append(None)
            self.empty = len(self.buffers) - 1

    def _name(self, prefix: str) -> bytes:
        k = self.added_tensors
        while (name := f"{prefix}{k}".encode()) in self.names:
            k += 1
        self.names.add(name)
        return name

    def activation(self, like: int, rows: int, columns: int | None = None) -> int:
        source = self.tensors[like]
        shape = [1, rows, source.shape[2] if columns is None else columns, source.shape[3]]
        self.tensors.append(tw.Tensor(shape, source.type, self.empty, self._name("t"),
                                      source.quantization, has_rank=source.has_rank))
        self.added_tensors += 1
        return len(self.tensors) - 1

    def constant(self, values, type_: int, quantization=None) -> int:
        dtype = "<i4" if type_ == _INT32 else "i1"
        array = np.asarray(values, dtype)
        key = (type_, array.shape, array.tobytes(),
               None if quantization is None else (tuple(quantization.scale),
                                                  tuple(quantization.zero_point)))
        if key not in self.constants:
            self.buffers.append(array.tobytes())
            self.tensors.append(tw.Tensor(list(array.shape), type_, len(self.buffers) - 1,
                                          self._name("c"), quantization))
            self.added_tensors += 1
            self.constants[key] = len(self.tensors) - 1
        return self.constants[key]

    def opcode(self, kind: str) -> int:
        code = _CODES[kind]
        for index, entry in enumerate(self.codes):
            if (entry.custom_code is None and entry.version == 2
                    and max(entry.builtin_code, entry.deprecated_builtin_code) == code):
                return index
        # Version 2 of each added operator is its int8 form.
        self.codes.append(tw.OperatorCode(code, 2, None, min(code, 127)))
        return len(self.codes) - 1

    def operator(self, kind: str, inputs: list, output: int, fields=None) -> tw.Operator:
        return tw.Operator(self.opcode(kind), inputs, [output],
                           tw.Options(_OPTION_TYPES[kind], None, fields or {}))

    def model_with(self, operators: list) -> tw.Model:
        sg = replace(self.sg, tensors=self.tensors, operators=operators)
        return replace(self.model, subgraphs=[sg], buffers=self.buffers,
                       operator_codes=self.codes)


def _band_ranges(rows: int, bands: int) -> list[tuple[int, int]]:
    edges = [round(i * rows / bands) for i in range(bands + 1)]
    return [(edges[i], edges[i + 1]) for i in range(bands)]


@dataclass
class _Rewrite:
    model: tw.Model
    macs: int


def _band_region(model: tw.Model, nodes: list, a: int, b: int, bands: int) -> _Rewrite:
    """Operators a..b computed in `bands` row bands."""
    sg = model.subgraphs[0]
    edit = _Edit(model, sg)
    region = nodes[a:b + 1]
    internal = {node.output for node in region}
    last = region[-1].output
    operators = list(sg.operators[:a])
    joined, macs = [], 0
    for band, (lo, hi) in enumerate(_band_ranges(sg.tensors[last].shape[1], bands)):
        need = {last: (lo, hi)}
        uses = {}
        for k in range(len(region) - 1, -1, -1):
            node = region[k]
            out = need[node.output]
            for position, index in node.inputs:
                if node.geometry is not None:
                    g = node.geometry
                    start, end, above, below = input_rows(
                        out[0], out[1], g.stride, g.kernel, g.top, sg.tensors[index].shape[1])
                elif sg.tensors[index].shape[1] == 1:
                    continue
                else:
                    (start, end), above, below = out, 0, 0
                uses[k, position] = (start, end, above, below)
                have = need.get(index)
                need[index] = (start, end) if have is None else (min(have[0], start),
                                                                 max(have[1], end))
        produced, slices = {}, {}
        for k, node in enumerate(region):
            op = sg.operators[a + k]
            inputs = list(op.inputs)
            for position, index in node.inputs:
                if (k, position) not in uses:
                    continue
                start, end, above, below = uses[k, position]
                source = sg.tensors[index]
                if index in internal:
                    current, (have_start, have_end) = produced[index]
                    if (have_start, have_end) != (start, end):
                        sliced = edit.activation(index, end - start)
                        operators.append(edit.operator("SLICE", [
                            current,
                            edit.constant([0, start - have_start, 0, 0], _INT32),
                            edit.constant([1, end - start, *source.shape[2:]], _INT32)], sliced))
                        current = sliced
                else:
                    current = slices.get((index, start, end))
                    if current is None:
                        current = edit.activation(index, end - start)
                        operators.append(edit.operator("SLICE", [
                            index, edit.constant([0, start, 0, 0], _INT32),
                            edit.constant([1, end - start, *source.shape[2:]], _INT32)], current))
                        slices[index, start, end] = current
                g = node.geometry
                if g is not None and node.kind not in _PADS and (above or below or g.left
                                                                   or g.right):
                    padded = edit.activation(index, end - start + above + below,
                                             source.shape[2] + g.left + g.right)
                    pads = edit.constant([[0, 0], [above, below], [g.left, g.right], [0, 0]],
                                         _INT32)
                    if node.kind == "MAX_POOL_2D":
                        # The lowest int8 value never wins a max over a window holding real values.
                        fill = edit.constant(np.int8(-128).reshape([]), _INT8, source.quantization)
                        operators.append(edit.operator("PADV2", [current, pads, fill], padded))
                    else:
                        # Int8 PAD fills with the zero point, which a padded tap of the operator reads.
                        operators.append(edit.operator("PAD", [current, pads], padded))
                    current = padded
                inputs[position] = current
            rows = need[node.output]
            output = edit.activation(node.output, rows[1] - rows[0])
            copy = replace(op, inputs=inputs, outputs=[output])
            if node.kind in _PADS:
                _, _, above, below = uses[k, 0]
                g = node.geometry
                copy.inputs[1] = edit.constant([[0, 0], [above, below], [g.left, g.right], [0, 0]],
                                                _INT32)
            elif node.geometry is not None:
                copy.options = replace(op.options, fields={**op.options.fields, 0: ("b", _VALID)})
            operators.append(copy)
            macs += node.macs // sg.tensors[node.output].shape[1] * (rows[1] - rows[0])
            produced[node.output] = (output, rows)
        joined.append(produced[last][0])
    operators.append(edit.operator("CONCATENATION", joined, last,
                                   {0: ("i", 1), 1: ("b", 0)}))
    operators += sg.operators[b + 1:]
    macs -= sum(node.macs for node in region)
    return _Rewrite(edit.model_with(operators), macs)


def _fold_pads(model: tw.Model) -> tuple[tw.Model, list[int]]:
    """Each PAD read only by a VALID convolution, with the zero-point fill and
    exactly that convolution's SAME padding, folded into the convolution. Also
    the original index of each remaining operator."""
    sg = model.subgraphs[0]
    readers = {}
    for k, op in enumerate(sg.operators):
        for index in op.inputs:
            readers.setdefault(index, []).append(k)
    folded, changed = set(), {}
    for k, op in enumerate(sg.operators):
        if _kind(model, op) != "PAD" or len(op.inputs) != 2 or op.outputs[0] in sg.outputs:
            continue
        users = readers.get(op.outputs[0], [])
        if len(users) != 1:
            continue
        conv = sg.operators[users[0]]
        kind = _kind(model, conv)
        x, padded = sg.tensors[op.inputs[0]], sg.tensors[op.outputs[0]]
        if (kind not in _CONVS or conv.inputs[0] != op.outputs[0] or conv.inputs.count(op.outputs[0]) != 1
                or not _band_tensor(x) or not _same_quantization(x, padded)
                or _option(conv, 0, "b", _SAME) != _VALID):
            continue
        pads = _values(model, sg.tensors[op.inputs[1]])
        g = _conv_geometry(model, sg, conv, kind)
        if pads is None or pads.shape != (4, 2) or pads[0].any() or pads[3].any() or g is None:
            continue
        y = sg.tensors[conv.outputs[0]]
        h, w = x.shape[1:3]
        oh, ow = y.shape[1:3]
        weights = sg.tensors[conv.inputs[1]].shape
        slots = (4, 5) if kind == "CONV_2D" else (5, 6)
        ew = (weights[2] - 1) * _option(conv, slots[0], "i", 1) + 1
        sw = _option(conv, 1, "i", 0)
        if (oh, ow) != (-(-h // g.stride), -(-w // sw)):
            continue
        if ([tuple(int(v) for v in pads[1]), tuple(int(v) for v in pads[2])]
                != [same_padding(h, oh, g.kernel, g.stride), same_padding(w, ow, ew, sw)]):
            continue
        folded.add(k)
        changed[users[0]] = replace(conv, inputs=[op.inputs[0], *conv.inputs[1:]],
                                    options=replace(conv.options, fields={
                                        **conv.options.fields, 0: ("b", _SAME)}))
    kept = [k for k in range(len(sg.operators)) if k not in folded]
    operators = [changed.get(k, sg.operators[k]) for k in kept]
    return replace(model, subgraphs=[replace(sg, operators=operators)]), kept


def _kept(model: tw.Model) -> list[int]:
    """The tensors an operator, the graph boundary or a signature references."""
    sg = model.subgraphs[0]
    used = set(sg.inputs) | set(sg.outputs)
    for op in sg.operators:
        used.update(i for i in (*op.inputs, *op.outputs, *(op.intermediates or ())) if i >= 0)
    for signature in model.signatures or ():
        used.update(index for _, index in (*signature.inputs, *signature.outputs))
    return sorted(used)


def _prune(model: tw.Model) -> tw.Model:
    """The model without the tensors no operator, graph boundary or signature
    references and without the buffers nothing references, renumbered."""
    sg = model.subgraphs[0]
    keep = _kept(model)
    number = {old: new for new, old in enumerate(keep)}
    buffers_used = {0, *(sg.tensors[i].buffer for i in keep),
                    *(buffer for _, buffer in model.metadata or ()), *(model.metadata_buffer or ())}
    buffer_keep = [i for i in range(len(model.buffers)) if i in buffers_used]
    buffer_number = {old: new for new, old in enumerate(buffer_keep)}

    def renumber(indices):
        return None if indices is None else [number[i] if i >= 0 else i for i in indices]

    tensors = [replace(sg.tensors[i], buffer=buffer_number[sg.tensors[i].buffer]) for i in keep]
    operators = [replace(op, inputs=renumber(op.inputs), outputs=renumber(op.outputs),
                         intermediates=renumber(op.intermediates)) for op in sg.operators]
    signatures = None if model.signatures is None else [
        replace(s, inputs=[(n, number[i]) for n, i in s.inputs],
                outputs=[(n, number[i]) for n, i in s.outputs]) for s in model.signatures]
    return replace(
        model, buffers=[model.buffers[i] for i in buffer_keep],
        subgraphs=[replace(sg, tensors=tensors, operators=operators, inputs=renumber(sg.inputs),
                           outputs=renumber(sg.outputs))],
        metadata=None if model.metadata is None else [
            (name, buffer_number[buffer]) for name, buffer in model.metadata],
        metadata_buffer=None if model.metadata_buffer is None else [
            buffer_number[buffer] for buffer in model.metadata_buffer],
        signatures=signatures)


def _tensors(model: tw.Model, referenced: bool = False) -> list:
    """Each tensor as tflm_arena takes it: (storage type, shape, placed). With
    `referenced`, a tensor nothing references is not placed."""
    sg = model.subgraphs[0]
    used = None
    if referenced:
        used = {i for op in sg.operators for i in (*op.inputs, *op.outputs)}
        used |= set(sg.inputs) | set(sg.outputs)
    return [(_TENSOR_TYPES[t.type] if 0 <= t.type < len(_TENSOR_TYPES) else None, t.shape or [],
             not (_is_constant(model, t) or t.is_variable) and (used is None or i in used))
            for i, t in enumerate(sg.tensors)]


def arena(model: tw.Model, referenced: bool = False) -> int | None:
    """TFLite Micro's arena head for a single-subgraph model, tflm_arena's
    figure for the model as written. With `referenced`, as planned once the
    tensors nothing references are pruned. None when a tensor has a storage
    type the planner model does not know."""
    sg = model.subgraphs[0]
    return tflm_arena(_tensors(model, referenced),
                      [(_kind(model, op), op.inputs, op.outputs) for op in sg.operators],
                      sg.inputs, sg.outputs)


def persistent(model: tw.Model, target: TflmTarget, compression: bool = False,
               recording: bool = False, referenced: bool = False) -> int | None:
    """TFLite Micro's persistent bytes for a single-subgraph model, as
    tflm_persistent models them; None when it does not model the model. With
    `referenced`, as the model is once the tensors nothing references are pruned."""
    sg = model.subgraphs[0]
    keep = _kept(model) if referenced else range(len(sg.tensors))
    number = {old: new for new, old in enumerate(keep)}
    kinds = [_code_kind(model, index) for index in range(len(model.operator_codes))]

    def renumber(indices):
        return [number[i] if i >= 0 else i for i in indices]

    tensors = []
    for i in keep:
        t = sg.tensors[i]
        q = t.quantization
        tensors.append((_TENSOR_TYPES[t.type] if 0 <= t.type < len(_TENSOR_TYPES) else None,
                        t.shape or [], t.is_variable,
                        len(q.scale or ()) if q else 0, len(q.zero_point or ()) if q else 0))
    operators = []
    for op in sg.operators:
        kind = kinds[op.opcode]
        options = {"adj_x": bool(_option(op, 0, "b", 0)),
                   "adj_y": bool(_option(op, 1, "b", 0))} if kind == "BATCH_MATMUL" else {}
        operators.append((kind, renumber(op.inputs), renumber(op.outputs), options))
    return tflm_persistent(tensors, operators, renumber(sg.inputs), renumber(sg.outputs), target,
                           compression, recording)


def _live(model: tw.Model) -> list[int]:
    """Live tensor bytes at each operator scope, 16-byte aligned as planned:
    a lower bound of the arena head at that scope."""
    sg = model.subgraphs[0]
    sizes = tflm_tensor_bytes(_tensors(model), [(_kind(model, op), op.inputs, op.outputs)
                                                for op in sg.operators])
    scopes = len(sg.operators) + 1
    first, last = {}, {}
    for index in sg.inputs:
        first.setdefault(index, 0)
        last[index] = 0
    for scope, op in enumerate(sg.operators, start=1):
        for index in op.outputs:
            first.setdefault(index, scope)
        for index in (*op.inputs, *op.outputs):
            if index >= 0:
                last[index] = scope
    for index in sg.outputs:
        first.setdefault(index, scopes - 1)
        last[index] = scopes - 1
    change = [0] * (scopes + 1)
    for index, start in first.items():
        size = (sizes[index] + 15) // 16 * 16
        change[start] += size
        change[last[index] + 1] -= size
    return list(itertools.accumulate(change[:-1]))


def _live_bounds(live: list[int]) -> tuple[list[int], list[int]]:
    """Largest live bytes at or before each scope, and at or after it."""
    return list(itertools.accumulate(live, max)), list(itertools.accumulate(live[::-1], max))[::-1]


def _regions(model: tw.Model, nodes: list):
    """Operator runs [a, b] a region can cover: every operator bandable, every
    intermediate read only inside the run, and a strided or pooling operator
    in it."""
    sg = model.subgraphs[0]
    last_reader = {}
    for k, op in enumerate(sg.operators):
        for index in op.inputs:
            last_reader[index] = k
    outputs = set(sg.outputs)
    for a in range(len(nodes)):
        if nodes[a] is None:
            continue
        downsamples = False
        for b in range(a, min(len(nodes), a + MAX_REGION)):
            if nodes[b] is None:
                break
            downsamples = downsamples or nodes[b].downsamples
            inner = (nodes[k].output for k in range(a, b))
            if downsamples and all(index not in outputs and a < last_reader.get(index, -1) <= b
                                   for index in inner):
                yield a, b


def export(data: bytes, target: str = DEFAULT_TARGET) -> Export:
    """The model rewritten for a smaller TFLite Micro arena on `target`, a key
    of TFLM_TARGETS, or the reason there is none."""
    abi = TFLM_TARGETS[target]
    try:
        model = tw.read(data)
        if len(model.subgraphs) != 1:
            raise ValueError("models with several subgraphs are not exported")
        for name, _ in model.metadata or ():
            if name not in _METADATA:
                raise ValueError(f"metadata {name.decode(errors='replace')} is not supported")
        before = arena(model)
        if before is None:
            raise ValueError("a tensor type TFLite Micro's arena model does not know")
        kinds = [_kind(model, op) for op in model.subgraphs[0].operators]
        for kind in kinds:
            if kind in _UNMODELED_SCRATCH:
                raise ValueError(f"TFLite Micro's scratch buffers for {kind} are not modeled")
        for kind in kinds:
            if kind not in _TFLM_KERNELS:
                raise ValueError(f"TFLite Micro's persistent allocations for {kind} are not "
                                 "modelled")
    except ValueError as exc:
        return Export(None, reason=str(exc), target=target)
    tail = persistent(model, abi)
    operators = len(model.subgraphs[0].operators)
    total_macs = _macs(model, model.subgraphs[0])
    folded_model, origin = _fold_pads(model)
    bases = [(model, list(range(operators)), 0)]
    if len(origin) < operators:
        bases.append((folded_model, origin, operators - len(origin)))
    best = None
    for base, origin, folded in bases:
        sg = base.subgraphs[0]
        base_growth = persistent(base, abi, referenced=True) - tail
        if folded:
            best = _better(best, (arena(base, referenced=True), base_growth, base, None, 0,
                                  folded, 0))
        nodes = [_node(base, sg, op) for op in sg.operators]
        before_scope, after_scope = _live_bounds(_live(base))
        # Live bytes outside a region stay as they are, so they bound its arena from below.
        regions = sorted(_regions(base, nodes), key=lambda r: max(
            before_scope[r[0]], after_scope[r[1] + 2] if r[1] + 2 < len(after_scope) else 0))
        for a, b in regions:
            floor = max(before_scope[a], after_scope[b + 2] if b + 2 < len(after_scope) else 0)
            # A rewrite only adds persistent allocations to those of its base.
            if _worse(floor + base_growth, best, before):
                continue
            rows = sg.tensors[nodes[b].output].shape[1]
            for bands in range(2, min(MAX_BANDS, rows) + 1):
                rewrite = _band_region(base, nodes, a, b, bands)
                growth = persistent(rewrite.model, abi, referenced=True) - tail
                if _worse(max(_live(rewrite.model)) + growth, best, before):
                    continue
                best = _better(best, (arena(rewrite.model, referenced=True), growth, rewrite.model,
                                      (origin[a], origin[b]), bands, folded, rewrite.macs))
    result = Export(before, operators_before=operators, macs=total_macs, target=target)
    if best is None or best[0] + best[1] >= before:
        result.reason = "no rewrite lowers the arena"
        return result
    head, growth, candidate, region, bands, folded, macs = best
    candidate = _prune(candidate)
    written = tw.write(candidate)
    check = tw.read(written)
    if arena(check) != head or persistent(check, abi) - tail != growth:
        raise RuntimeError("the written model's arena differs from the planned one")
    result.arena_after, result.persistent_growth, result.data = head, growth, written
    result.region, result.bands, result.folded, result.extra_macs = region, bands, folded, macs
    result.operators_after = len(candidate.subgraphs[0].operators)
    return result


def _worse(bound: int, best, before: int) -> bool:
    """Whether a candidate whose total is at least `bound` loses to the best
    so far or fails to lower the arena."""
    return bound >= before or (best is not None and bound > best[0] + best[1])


def _better(best, candidate):
    """The candidate with the lower total, then fewer operators."""
    def key(c):
        return (c[0] + c[1], len(c[2].subgraphs[0].operators))
    return candidate if best is None or key(candidate) < key(best) else best
