"""ExecuTorch Cortex-M programs rewritten so that they run in less RAM.

A region is a run of cortex_m convolutions, depthwise convolutions and
additions whose intermediates are read only inside the run. Each region is
computed in row bands with the program's own operators and no copies. A
band's input is a new tensor value placed inside the bytes of the tensor it
reads, at the first row it needs; the band's output is placed at its final
rows of the region's output, or at row 0 of a buffer every band reuses for an
intermediate. A convolution band's padding argument is the padding rows above
the band and the original left padding; the rows below follow from the band's
output height. An addition band reads equal-size row views of both operands.
A band keeps the original scratch tensor when CMSIS-NN asks the same bytes for
it, else gets one of the exact size; a band count that would change the
CMSIS-NN kernel a layer runs is not used. A region's output may overlap its
largest input (in place) when no band writes bytes that it or a later band
still reads; over the method input only when allowed or when the original
program already writes there.

The memory plan is redone for the whole program: roots, and groups of roots
overlapping in place, are placed greedily at the lowest free offset under
several orders, largest first among them; each order is placed again with
the items that end at the top moved to the front, and the smallest arena is
kept.
`pte_check.check` checks every written plan against the original program's dataflow,
from the bytes each instruction reads and writes. An instruction that reads
or writes part of a tensor's bytes, as in a program export wrote, is no
region member, and tensors that share bytes through the dataflow keep their
offsets in the group they form. A candidate's total
is its planned buffers, the method allocator ExecuTorch 1.6.0's Method::init
takes on a 32-bit target, and the smallest temp allocator. Regions are chosen
by a dynamic program over the peak of live bytes each region reaches when
banded alone, one choice per peak; every choice is planned in full, and the
lowest total that meets the budgets is written.
"""

import bisect
from dataclasses import dataclass, field, replace

from tigris.frontends import pte
from tigris.frontends import pte_check as pc
from tigris.frontends import pte_cortex_m as cm
from tigris.frontends import pte_method as pm

# Band counts tried per region, up to the rows of its output.
BAND_COUNTS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56)
# Longest run of instructions tried as one region.
MAX_REGION = 12
# Placement passes per start order.
ROUNDS = 40

_CONV = cm.NAMESPACE + "quantized_conv2d"
_DEPTHWISE = cm.NAMESPACE + "quantized_depthwise_conv2d"
_ADD = cm.NAMESPACE + "quantized_add"
_LINEAR = cm.NAMESPACE + "quantized_linear"
_CHAR = 1  # ScalarType of int8
_CHANNELS_LAST = [0, 2, 3, 1]
_ALIGN = pc.ALIGNMENT
# In-place modes: none, over an input other than the method input, over any input.
_NONE, _NOT_INPUT, _ANY = 0, 1, 2


def _r16(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


@dataclass
class Region:
    """Instructions first..last of the original program computed in `bands`
    row bands, the output in place over an input or not."""
    first: int
    last: int
    bands: int
    in_place: bool


@dataclass
class PteExport:
    """What export found for a program. `data` is the rewritten program when
    `written`, else the input unchanged: `refused` for a program export does
    not take, with a `reason` that reads after the file name; `unmet` lists
    (limit, smallest bytes found, total of that program) for a limit no
    program found meets, the limit "arena", "method", or "both" when each is
    met alone; otherwise no program within the limits lowers the total.
    Figures are None when the program was refused before they were taken. Planned buffer sizes
    are indexed by memory id; ExecuTorch reserves id 0. `origin` gives each
    instruction of `data` its original instruction and band (None outside a
    region); the offsets are those of the method inputs and outputs in `data`."""
    data: bytes
    written: bool = False
    reason: str | None = None
    refused: bool = False
    unmet: list = field(default_factory=list)
    classes: tuple = ()
    cmsis_class: str | None = None
    instructions_before: int | None = None
    instructions_after: int | None = None
    origin: list = field(default_factory=list)
    input_offsets: list = field(default_factory=list)
    output_offsets: list = field(default_factory=list)
    planned_before: list | None = None
    planned_after: list | None = None
    method_before: int | None = None
    method_after: int | None = None
    temp_before: int | None = None
    temp_after: int | None = None
    macs: int | None = None
    extra_macs: int = 0
    pte_bytes_before: int = 0
    pte_bytes_after: int | None = None
    regions: list = field(default_factory=list)
    input_overwritten_before: bool | None = None
    input_overwritten_after: bool | None = None
    warnings: list = field(default_factory=list)

    @property
    def total_before(self) -> int | None:
        if self.method_before is None:
            return None
        return sum(self.planned_before) + self.method_before + self.temp_before

    @property
    def total_after(self) -> int | None:
        if not self.written:
            return None
        return sum(self.planned_after) + self.method_after + self.temp_after


class _Infeasible(Exception):
    """A band count that changes the CMSIS-NN kernel of a layer, or whose
    scratch size differs between the classes the program fits."""


@dataclass
class _Node:
    """An instruction a region may hold: the positions and values of its
    activation inputs, its output value and the row geometry of a
    convolution."""
    kind: str
    call: cm.Call
    inputs: list
    output: int
    rows: int
    in_rows: int = 0
    stride: int = 1
    kernel: int = 1
    pad: int = 0


def _activation(value) -> bool:
    return (isinstance(value, pte.Tensor) and value.allocation_info is not None
            and value.scalar_type == _CHAR and len(value.sizes or []) == 4
            and value.sizes[0] == 1 and list(value.dim_order or []) == _CHANNELS_LAST)


def _row_bytes(tensor: pte.Tensor) -> int:
    n, c, _, w = tensor.sizes
    return n * c * w * pte.SCALAR_TYPES[tensor.scalar_type][1]


def _dims(tensor: pte.Tensor, rows: int | None = None) -> cm.Dims:
    n, c, h, w = tensor.sizes
    return cm.Dims(n, h if rows is None else rows, w, c)


def _filter(call: cm.Call) -> cm.Dims:
    weight = call.value("weight").sizes
    if call.signature.name == _DEPTHWISE:
        return cm.Dims(weight[3], weight[1], weight[2], 1)
    return cm.Dims(*weight)


def _instruction_macs(plan: pte.ExecutionPlan, op: str, args: list) -> int:
    """Multiply-accumulates of a convolution, depthwise convolution or linear
    layer, else 0."""
    if op not in (_CONV, _DEPTHWISE, _LINEAR):
        return 0
    values = plan.values
    count = 1
    for s in values[args[-2]].sizes:
        count *= s
    weight = values[args[1]].sizes
    if op == _CONV:
        return count * weight[1] * weight[2] * weight[3]
    if op == _DEPTHWISE:
        return count * weight[1] * weight[2]
    return count * weight[-1]


def program_macs(plan: pte.ExecutionPlan) -> int:
    """Multiply-accumulates of the plan's convolutions, depthwise convolutions
    and linear layers."""
    return sum(_instruction_macs(plan, plan.operators[i.op_index].name, i.args)
               for c in plan.chains or () for i in c.instructions or ())


def _aliases(plan: pte.ExecutionPlan, calls: list) -> tuple[dict, list]:
    """Planned values that share bytes through the dataflow (a read that
    holds bytes of another value's write), each mapped to the root of its
    group and its byte offset in it, and the records of roots that are no
    value of the plan. A group's root spans all of it: a method input, else
    the earliest written member, else a value appended for it. A view
    ExecuTorch's emitter makes for a reshape shares its source's root at
    offset 0; a row view or a band's output sits inside its root."""
    values = plan.values
    first = {v: -1 for v in plan.inputs or ()}
    for call in calls:
        for position in (*call.signature.scratch, len(call.args) - 2):
            first.setdefault(call.args[position], call.index)
    parent = {}

    def find(v):
        while parent.get(v, v) != v:
            v = parent[v]
        return v

    for reader, writers in pc.sources(plan).items():
        for w in writers:
            a, b = find(reader), find(w)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = {}
    for v in parent:
        groups.setdefault(find(v), {find(v)}).add(v)
    alias, extra = {}, []
    for members in groups.values():
        low = min(values[v].allocation_info.offset for v in members)
        top = max(values[v].allocation_info.offset + values[v].nbytes for v in members)
        spanning = [v for v in members if values[v].allocation_info.offset == low
                    and values[v].nbytes == top - low]
        if spanning:
            root = min(spanning, key=lambda v: (first.get(v, len(calls)), v))
        else:
            root = len(values) + len(extra)
            extra.append(pte.Tensor(_CHAR, sizes=[top - low], dim_order=[0]))
        for v in members:
            if v != root:
                alias[v] = (root, values[v].allocation_info.offset - low)
    return alias, extra


class _Base:
    """The original program, its kernel calls, region members and pools of
    Int and IntList values."""

    def __init__(self, plan: pte.ExecutionPlan, classes: tuple):
        self.plan = plan
        self.values = plan.values
        self.classes = classes
        self.calls = cm.kernel_calls(plan)
        self.n = len(self.calls)
        self.alias, self.extra = _aliases(plan, self.calls)
        self.ints, self.lists = {}, {}
        for i, v in enumerate(self.values):
            if isinstance(v, pte.Int):
                self.ints.setdefault(v.int_val, i)
        for i, v in enumerate(self.values):
            if isinstance(v, pte.IntList) and all(
                    0 <= k < len(self.values) and isinstance(self.values[k], pte.Int)
                    for k in v.items or ()):
                self.lists.setdefault(tuple(self.values[k].int_val for k in v.items or ()), i)
        self.readers = {}
        for call in self.calls:
            for position in call.signature.inputs:
                self.readers.setdefault(self.root(call.args[position]), set()).add(call.index)
        self.output_roots = {self.root(v) for v in plan.outputs or ()}
        self.input_roots = {self.root(v) for v in plan.inputs or ()}
        self.nodes = [self._node(c) for c in self.calls]
        self.positions = [(c.signature.inputs, c.signature.scratch, len(c.args) - 2)
                          for c in self.calls]
        self.cache = {}

    def root(self, value: int) -> int:
        return self.alias.get(value, (value, 0))[0]

    def whole(self, value: int) -> bool:
        """Whether `value` holds all bytes of its root."""
        root, offset = self.alias.get(value, (value, 0))
        return not offset and (root == value or (
            root < len(self.values) and self.values[root].nbytes == self.values[value].nbytes))

    def _node(self, call: cm.Call) -> _Node | None:
        op = call.signature.name
        out = call.value("out")
        if not _activation(out) or not all(
                self.whole(call.args[p]) for p in (*call.signature.inputs, len(call.args) - 2)):
            return None
        if op == _ADD:
            a, b = call.value("self"), call.value("other")
            if not (_activation(a) and _activation(b) and a.sizes == out.sizes == b.sizes):
                return None
            return _Node("add", call, [(call.signature.index("self"), call.args[0]),
                                       (call.signature.index("other"), call.args[4])],
                         call.args[-2], out.sizes[2])
        if op not in (_CONV, _DEPTHWISE):
            return None
        x = call.value("input")
        weight = call.value("weight").sizes or []
        if not _activation(x) or len(weight) != 4:
            return None
        if op == _CONV and weight[3] != x.sizes[1]:
            return None
        stride, padding, dilation = (call.ints(n, self.plan)
                                     for n in ("stride", "padding", "dilation"))
        rows, in_rows, extent = out.sizes[2], x.sizes[2], (weight[1] - 1) * dilation[0] + 1
        if not (padding[0] < extent and (rows - 1) * stride[0] - padding[0] < in_rows):
            raise pte.Refusal(
                f"has instruction {call.index} ({op}) computing {rows} output rows from "
                f"{in_rows} input rows with stride {stride[0]}, padding {padding[0]} and a "
                f"kernel {extent} rows high, so some output rows read no input row")
        return _Node("conv", call, [(0, call.args[0])], call.args[-2], rows, in_rows,
                     stride[0], extent, padding[0])

    def kernel(self, j: int, rows: tuple | None = None, pad_top: int | None = None) -> tuple:
        """The CMSIS-NN kernel and scratch bytes of convolution instruction `j`
        per class, for the whole layer or for (input rows, output rows) and
        the padding above."""
        key = (j, rows, pad_top)
        if key not in self.cache:
            self.cache[key] = tuple(self._kernel(self.calls[j], cls, rows, pad_top)
                                    for cls in self.classes)
        return self.cache[key]

    def _kernel(self, call: cm.Call, cls: str, rows: tuple | None,
                pad_top: int | None) -> tuple[str, int]:
        x, out = call.value("input"), call.value("out")
        inp = _dims(x, rows and rows[0])
        dst = _dims(out, rows and rows[1])
        filt = _filter(call)
        padding, stride, dilation = (tuple(call.ints(n, self.plan))
                                     for n in ("padding", "stride", "dilation"))
        if pad_top is not None:
            padding = (pad_top, padding[1])
        if call.signature.name == _CONV:
            return (cm.conv_kernel(cls, inp, filt, padding, stride, dilation),
                    cm.conv_scratch(cls, inp, filt, dst, padding, stride, dilation))
        return (cm.depthwise_kernel(cls, inp, filt, dst, padding, stride, dilation),
                cm.depthwise_scratch(cls, inp, filt, dst, padding, stride, dilation))

    def regions(self):
        """Instruction runs [a, b] of region members whose intermediates are no
        method output and are read only inside the run, with an output of at
        least two rows."""
        for a in range(self.n):
            if self.nodes[a] is None:
                continue
            for b in range(a, min(self.n, a + MAX_REGION)):
                if self.nodes[b] is None:
                    break
                inner = (self.root(self.nodes[k].output) for k in range(a, b))
                if self.nodes[b].rows >= 2 and all(
                        r not in self.output_roots
                        and all(a <= k <= b for k in self.readers.get(r, ())) for r in inner):
                    yield a, b

    def band_rows(self, a: int, b: int, count: int) -> list:
        """Per band: its output rows and, per instruction it runs, the rows it
        computes, the reads (position, value, first row, end row) and the
        padding rows above a convolution's input; then the rows of each value
        the band needs, by root."""
        nodes = self.nodes
        rows = nodes[b].rows
        bands = []
        for k in range(count):
            lo, hi = rows * k // count, rows * (k + 1) // count
            need = {self.root(nodes[b].output): (lo, hi)}
            runs = {}
            for j in range(b, a - 1, -1):
                node = nodes[j]
                if self.root(node.output) not in need:
                    continue
                o0, o1 = need[self.root(node.output)]
                if node.kind == "add":
                    reads, pad = [(p, v, o0, o1) for p, v in node.inputs], None
                else:
                    first = o0 * node.stride - node.pad
                    end = (o1 - 1) * node.stride - node.pad + node.kernel
                    r0, r1 = max(first, 0), min(end, node.in_rows)
                    reads, pad = [(p, v, r0, r1) for p, v in node.inputs], r0 - first
                runs[j] = (o0, o1, reads, pad)
                for _, v, r0, r1 in reads:
                    v = self.root(v)
                    have = need.get(v)
                    need[v] = (r0, r1) if have is None else (min(have[0], r0), max(have[1], r1))
            bands.append(((lo, hi), runs, need))
        return bands


@dataclass
class _Build:
    """A program being rewritten: the original values shared and new ones
    appended, instructions as (op index, args), and per new instruction its
    original instruction, region and band."""
    base: _Base
    values: list
    ints: dict
    lists: dict
    view_of: dict  # value -> (root value, byte offset)
    instructions: list = field(default_factory=list)
    origin: list = field(default_factory=list)
    storages: set = field(default_factory=set)
    groups: list = field(default_factory=list)  # (X root, Y root, byte offset of Y in X)
    regions: list = field(default_factory=list)
    reads: list = field(default_factory=list)  # per region: band rows, external reads
    cache: dict = field(default_factory=dict)

    @classmethod
    def start(cls, base: _Base) -> "_Build":
        return cls(base, list(base.values) + base.extra, dict(base.ints), dict(base.lists),
                   dict(base.alias))

    def _append(self, record) -> int:
        self.values.append(record)
        return len(self.values) - 1

    def int_value(self, x: int) -> int:
        if x not in self.ints:
            self.ints[x] = self._append(pte.Int(x))
        return self.ints[x]

    def int_list(self, xs) -> int:
        key = tuple(xs)
        if key not in self.lists:
            self.lists[key] = self._append(pte.IntList([self.int_value(x) for x in key]))
        return self.lists[key]

    def tensor(self, like: int, sizes: list) -> int:
        return self._append(replace(self.values[like], sizes=list(sizes)))

    def view(self, parent: int, r0: int, rows: int) -> int:
        """A tensor of rows r0..r0+rows of `parent`, placed inside it."""
        t = self.values[parent]
        if r0 == 0 and rows == t.sizes[2]:
            return parent
        root, offset = self.view_of.get(parent, (parent, 0))
        offset += r0 * _row_bytes(t)
        sizes = [t.sizes[0], t.sizes[1], rows, t.sizes[3]]
        key = ("view", root, offset, tuple(sizes))
        if key not in self.cache:
            self.cache[key] = index = self.tensor(parent, sizes)
            self.view_of[index] = (root, offset)
        return self.cache[key]

    def scratch(self, original: int, nbytes: int) -> int:
        if self.values[original].nbytes == nbytes:
            return original
        key = ("scratch", original, nbytes)
        if key not in self.cache:
            self.cache[key] = self.tensor(original, [nbytes])
        return self.cache[key]

    def copy(self, k: int) -> None:
        call = self.base.calls[k]
        self.instructions.append((self.base.plan.chains[0].instructions[k].op_index, call.args))
        self.origin.append((k, None, None))

    def band(self, a: int, b: int, count: int) -> None:
        """Instructions a..b in `count` row bands; raises _Infeasible."""
        base = self.base
        nodes = base.nodes
        region = len(self.regions)
        bands = base.band_rows(a, b, count)
        y = nodes[b].output
        internal = {base.root(nodes[j].output) for j in range(a, b)}
        store = {}
        for v in sorted(internal):
            heights = [need[v][1] - need[v][0] for _, _, need in bands if v in need]
            if heights:
                t = self.values[v]
                store[v] = self.tensor(v, [t.sizes[0], t.sizes[1], max(heights), t.sizes[3]])
                self.storages.add(store[v])
        external = {}  # value -> [(band, instruction, first row, end row)]
        originals = base.plan.chains[0].instructions
        for k, (_, runs, need) in enumerate(bands):
            for j in range(a, b + 1):
                if j not in runs:
                    continue
                node, (o0, o1, reads, pad) = nodes[j], runs[j]
                call = node.call
                args = list(call.args)
                for position, v, r0, r1 in reads:
                    root = base.root(v)
                    if root in internal:
                        args[position] = self.view(store[root], r0 - need[root][0], r1 - r0)
                    else:
                        args[position] = self.view(v, r0, r1 - r0)
                        external.setdefault(v, []).append((k, j, r0, r1))
                out = self.view(y, o0, o1 - o0) if j == b else self.view(
                    store[base.root(node.output)], 0, o1 - o0)
                if node.kind == "conv":
                    padding = call.ints("padding", base.plan)
                    args[call.signature.index("padding")] = self.int_list([pad, padding[1]])
                    kernels = base.kernel(j, (reads[0][3] - reads[0][2], o1 - o0), pad)
                    if [name for name, _ in kernels] != [name for name, _ in base.kernel(j)]:
                        raise _Infeasible
                    scratch = {nbytes for _, nbytes in kernels}
                    if len(scratch) != 1:
                        raise _Infeasible
                    position = call.signature.index("scratch")
                    args[position] = self.scratch(call.args[position], scratch.pop())
                args[-2] = args[-1] = out
                self.instructions.append((originals[j].op_index, args))
                self.origin.append((j, region, k))
        self.regions.append(Region(a, b, len(bands), False))
        self.reads.append((bands, external))

    def in_place(self, region: int, mode: int, may_overwrite: bool) -> tuple | None:
        """The (input, output, offset of the output in the input) that places
        region `region`'s output over its largest input, or None when no input
        qualifies or it saves no bytes. The offset keeps every band's writes
        clear of input bytes that the same band's last instruction or a later
        band reads."""
        base = self.base
        b = self.regions[region].last
        bands, external = self.reads[region]
        y = base.nodes[b].output
        candidates = []
        for x in sorted(external):
            if (x in self.view_of or x in base.output_roots
                    or max(base.readers.get(x, ())) > b):
                continue
            if x in base.input_roots and (mode != _ANY or not may_overwrite):
                continue
            candidates.append((self.values[x].nbytes, x))
        if not candidates or y in self.view_of:
            return None
        _, x = max(candidates)
        in_row, out_row = _row_bytes(self.values[x]), _row_bytes(self.values[y])
        bad = []
        for k, ((lo, hi), _, _) in enumerate(bands):
            for kr, j, r0, r1 in external[x]:
                if kr > k or (kr == k and j == b):
                    bad.append((r0 * in_row - hi * out_row, r1 * in_row - lo * out_row))
        merged = []
        for lo, hi in sorted(bad):
            if merged and lo < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        in_bytes, out_bytes = self.values[x].nbytes, self.values[y].nbytes
        options = {0, in_bytes - out_bytes}
        for lo, hi in merged:
            options |= {lo, hi}
        starts = [lo for lo, _ in merged]
        best = None
        for d in sorted(options):
            for delta in (d // _ALIGN * _ALIGN, -(-d // _ALIGN) * _ALIGN):
                k = bisect.bisect_left(starts, delta) - 1
                if k < 0 or merged[k][1] <= delta:
                    span = max(in_bytes, delta + out_bytes) - min(0, delta)
                    if best is None or (span, abs(delta), delta) < best:
                        best = (span, abs(delta), delta)
        if best is None or best[0] >= _r16(in_bytes) + _r16(out_bytes):
            return None
        return x, y, best[2]


@dataclass
class _Candidate:
    """A rewritten method before placement: its plan record, every planned
    value's root and offset in it, each root's size, lifetimes and in-place
    groups, all by root key, and the figures that do not depend on
    placement."""
    plan: pte.ExecutionPlan
    origin: list
    regions: list
    roots: dict  # planned value -> (root key, byte offset)
    sizes: dict  # root key -> bytes
    intervals: dict  # root key -> [(first, last)]
    groups: list
    method: int
    macs: int
    number: dict  # build value -> plan value
    arena: int | None = None
    places: dict | None = None  # root key -> offset

    @property
    def planned(self) -> int:
        return self.arena

    @property
    def total(self) -> int:
        return self.arena + self.method + pm.temp_allocator_min_bytes(self.plan)


def _assemble(build: _Build, keep_input: bool) -> _Candidate:
    """The build as a plan record with the values nothing references dropped
    and the rest renumbered in order."""
    base = build.base
    plan = base.plan
    values = build.values
    used = set(plan.inputs or ()) | set(plan.outputs or ())
    for _, args in build.instructions:
        used.update(args)
    for v in list(used):
        record = values[v]
        if isinstance(record, (pte.IntList, pte.TensorList, pte.OptionalTensorList)):
            used.update(i for i in record.items or () if i >= 0)
    keep = sorted(used)
    number = {old: new for new, old in enumerate(keep)}
    records = []
    for old in keep:
        record = values[old]
        if isinstance(record, (pte.IntList, pte.TensorList, pte.OptionalTensorList)):
            items = [number[i] if i >= 0 else i for i in record.items or ()]
            if items != record.items:
                record = replace(record, items=items)
        records.append(record)
    instructions = [pte.KernelCall(op, [number[i] for i in args])
                    for op, args in build.instructions]
    chain = plan.chains[0]
    stacktrace = None if chain.stacktrace is None else [
        chain.stacktrace[k] for k, _, _ in build.origin]
    new_plan = replace(plan, values=records, inputs=[number[i] for i in plan.inputs or ()],
                       outputs=[number[i] for i in plan.outputs or ()],
                       chains=[replace(chain, instructions=instructions, stacktrace=stacktrace)])

    def key(old):
        return number[old] if old in number else f"s{old}"

    roots, sizes = {}, {}
    for old in keep:
        record = values[old]
        if isinstance(record, pte.Tensor) and record.allocation_info is not None:
            root, offset = build.view_of.get(old, (old, 0))
            roots[number[old]] = (key(root), offset)
            sizes[key(root)] = _r16(values[root].nbytes)
    events = {}
    end = len(instructions)
    view_of, storages = build.view_of, build.storages

    def use(v, t, generation):
        root = view_of.get(v, (v, 0))[0]
        if generation is None and root in storages:
            generation = band
        events.setdefault(key(root), {}).setdefault(generation, []).append(t)

    for t, ((_, args), (j, _, band)) in enumerate(zip(build.instructions, build.origin)):
        inputs, scratch, out = base.positions[j]
        for position in (*inputs, *scratch, out):
            if number[args[position]] in roots:
                use(args[position], t, t if position in scratch else None)
    for v in plan.inputs or ():
        root = roots[number[v]][0]
        events.setdefault(root, {}).setdefault(None, []).extend([-1, end] if keep_input else [-1])
    for v in plan.outputs or ():
        events.setdefault(roots[number[v]][0], {}).setdefault(None, []).append(end)
    intervals = {root: sorted((min(ts), max(ts)) for ts in gens.values())
                 for root, gens in events.items()}
    groups = [(key(x), key(y), d) for x, y, d in build.groups]
    return _Candidate(new_plan, build.origin, build.regions, roots, sizes, intervals, groups,
                      pm.method_allocator_bytes(new_plan), program_macs(new_plan), number)


def _items(c: _Candidate) -> list:
    """Roots placed together: each in-place group as (root, offset) members
    from offset 0, every other root alone."""
    member = {r: [(r, 0)] for r in c.intervals}
    owner = {r: r for r in c.intervals}
    for x, y, delta in c.groups:
        ox = owner[x]
        rel = dict(member[ox])[x]
        member[ox].append((y, rel + delta))
        del member[owner[y]]
        owner[y] = ox
    items = []
    for mem in member.values():
        low = min(rel for _, rel in mem)
        items.append([(r, rel - low) for r, rel in mem])
    return items


def _live(c: _Candidate, items: list) -> list:
    """Live bytes at each instruction: per item, the span of its members whose
    lifetime holds the instruction."""
    n = len(c.plan.chains[0].instructions)
    change = [0] * (n + 2)
    live = [0] * n
    for mem in items:
        if len(mem) == 1:
            r = mem[0][0]
            for first, last in c.intervals[r]:
                first, last = max(first, 0), min(last, n - 1)
                if first <= last:
                    change[first] += c.sizes[r]
                    change[last + 1] -= c.sizes[r]
            continue
        lo = max(min(f for r, _ in mem for f, _ in c.intervals[r]), 0)
        hi = min(max(e for r, _ in mem for _, e in c.intervals[r]), n - 1)
        for t in range(lo, hi + 1):
            on = [(r, rel) for r, rel in mem if any(f <= t <= e for f, e in c.intervals[r])]
            if on:
                live[t] += max(rel + c.sizes[r] for r, rel in on) - min(rel for _, rel in on)
    total = 0
    for t in range(n):
        total += change[t]
        live[t] += total
    return live


def _mask(intervals: list) -> int:
    """The instructions of `intervals` as bits, bit 0 for the method's start."""
    mask = 0
    for first, last in intervals:
        mask |= ((1 << (last - first + 1)) - 1) << (first + 1)
    return mask


def _place(c: _Candidate) -> None:
    """Places every item at the lowest offset where none of its members meets
    a placed root it shares an instruction with. Each of several orders,
    largest first among them, is placed again with the items that end at the
    top of the arena moved to the front, until an order repeats or ROUNDS
    passes are done; the smallest arena is kept."""
    items = _items(c)
    sizes, intervals = c.sizes, c.intervals
    masks = {r: _mask(spans) for r, spans in intervals.items()}
    extent = [max(rel + sizes[r] for r, rel in m) for m in items]
    first = [min(f for r, _ in m for f, _ in intervals[r]) for m in items]
    last = [max(e for r, _ in m for _, e in intervals[r]) for m in items]
    duration = [sum(e - f + 1 for r, _ in m for f, e in intervals[r]) for m in items]
    every = range(len(items))

    def place(order):
        placed, offsets = [], [0] * len(items)
        for i in order:
            offset, moved = 0, True
            while moved:
                moved = False
                for r, rel in items[i]:
                    if not sizes[r]:
                        continue
                    for start, size, mask in placed:
                        if (offset + rel < start + size and start < offset + rel + sizes[r]
                                and masks[r] & mask):
                            offset = _r16(start + size - rel)
                            moved = True
            offsets[i] = offset
            placed.extend((offset + rel, sizes[r], masks[r]) for r, rel in items[i] if sizes[r])
        return offsets, _r16(max([offsets[i] + extent[i] for i in every] + [0]))

    orders = [sorted(every, key=lambda i: -extent[i]),
              sorted(every, key=lambda i: (first[i] - last[i], -extent[i])),
              sorted(every, key=lambda i: -extent[i] * duration[i]),
              sorted(every, key=lambda i: (-last[i], -extent[i])),
              sorted(every, key=lambda i: (first[i], -extent[i])),
              sorted(every, key=lambda i: (-len(items[i]), -extent[i]))]
    best = None
    for order in orders:
        seen = set()
        for _ in range(ROUNDS):
            offsets, arena = place(order)
            if best is None or arena < best[0]:
                best = (arena, offsets)
            seen.add(tuple(order))
            top = [i for i in order if offsets[i] + extent[i] >= arena]
            order = top + [i for i in order if offsets[i] + extent[i] < arena]
            if tuple(order) in seen:
                break
    c.arena = best[0]
    c.places = {r: best[1][i] + rel for i in every for r, rel in items[i]}


def _finish(c: _Candidate) -> pte.ExecutionPlan:
    """The plan record with every planned tensor at its place."""
    values = list(c.plan.values)
    for v, (root, offset) in c.roots.items():
        at = c.places[root] + offset
        values[v] = replace(values[v], allocation_info=pte.AllocationDetails(
            1, at & 0xFFFFFFFF, at >> 32))
    sizes = list(c.plan.non_const_buffer_sizes)
    sizes[1] = c.arena
    return replace(c.plan, values=values, non_const_buffer_sizes=sizes)


def _build(base: _Base, regions: list, may_overwrite: bool) -> _Build:
    """The program with each (first, last, bands, in-place mode) region
    banded; raises _Infeasible."""
    build = _Build.start(base)
    at = {r[0]: r for r in regions}
    k = 0
    while k < base.n:
        if k in at:
            a, b, count, mode = at[k]
            build.band(a, b, count)
            group = None if mode == _NONE else build.in_place(len(build.regions) - 1, mode,
                                                               may_overwrite)
            if group is not None:
                build.groups.append(group)
                build.regions[-1].in_place = True
            k = b + 1
        else:
            build.copy(k)
            k += 1
    return build


@dataclass
class _Score:
    peak: int
    method: int
    macs: int


def _dp(best: dict, live0: list, limit: int):
    """The disjoint regions, from each (first, last) run's best scored choice
    within `limit`, that keep every other instruction's live bytes at or
    below `limit` at the least summed key, and the peak they reach; None when
    no choice keeps the limit."""
    n = len(live0)
    inf = (float("inf"), float("inf"))
    cost = [inf] * (n + 1)
    cost[0] = (0, 0)
    back = [None] * (n + 1)
    ends = {}
    for (a, b), choice in best.items():
        ends.setdefault(b, []).append((a, choice))
    for j in range(1, n + 1):
        if live0[j - 1] <= limit and cost[j - 1] != inf:
            cost[j], back[j] = cost[j - 1], None
        for a, (key, job, s) in ends.get(j - 1, ()):
            if cost[a] == inf:
                continue
            option = (cost[a][0] + key[0], cost[a][1] + key[1])
            if option < cost[j]:
                cost[j], back[j] = option, (a, job, s)
    if cost[n] == inf:
        return None
    chosen, peak, j = [], 0, n
    while j > 0:
        if back[j] is None:
            peak = max(peak, live0[j - 1])
            j -= 1
        else:
            a, job, s = back[j]
            chosen.append(job)
            peak = max(peak, s.peak)
            j = a
    return tuple(sorted(chosen)), peak


def _choices(scores: dict, live0: list, modes: set, by_macs: bool) -> list:
    """Region choices ranked by estimated total: for each live-bytes limit
    at which a run's best choice or an instruction outside the regions
    changes, the dynamic program's choice under that limit."""
    runs = {}
    for job, s in scores.items():
        if job[3] in modes:
            key = (s.macs, s.method) if by_macs else (s.method, s.macs)
            runs.setdefault(job[:2], []).append((s.peak, key, job, s))
    steps = []
    for run, options in runs.items():
        least = None
        for peak, key, job, s in sorted(options):
            if least is None or key < least:
                least = key
                steps.append((peak, run, (key, job, s)))
    steps.sort(key=lambda x: (x[0], x[1]))
    limits = sorted({p for p, _, _ in steps} | set(live0))
    best, estimates, k = {}, {}, 0
    for limit in limits:
        while k < len(steps) and steps[k][0] <= limit:
            best[steps[k][1]] = steps[k][2]
            k += 1
        found = _dp(best, live0, limit)
        if found is None or not found[0]:
            continue
        chosen, peak = found
        s = [scores[j] for j in chosen]
        estimate = (peak + sum(x.method for x in s), sum(x.macs for x in s))
        if chosen not in estimates or estimate < estimates[chosen]:
            estimates[chosen] = estimate
    return sorted(estimates, key=lambda c: (estimates[c], c))


def _search(base: _Base, may_overwrite: bool, keep_input: bool,
            max_macs: int | None = None) -> list:
    """Candidates planned in full, lowest total first. The model memory and
    method allocator limits only select among them, so the candidates of one
    program are the same under every such limit."""
    identity = _assemble(_build(base, [], may_overwrite), keep_input)
    live0 = _live(identity, _items(identity))
    scores = {}
    for a, b in base.regions():
        reads_input = any(base.root(v) in base.input_roots
                          for k in range(a, b + 1) for _, v in base.nodes[k].inputs)
        modes = (_NONE, _NOT_INPUT) + ((_ANY,) if may_overwrite and reads_input else ())
        for count in BAND_COUNTS:
            if count > base.nodes[b].rows:
                break
            try:
                build = _build(base, [(a, b, count, _NONE)], may_overwrite)
            except _Infeasible:
                continue
            c = _assemble(build, keep_input)
            extra = c.macs - identity.macs
            if max_macs is not None and extra > max_macs:
                continue
            times = [t for t, o in enumerate(c.origin) if o[1] is not None]
            for mode in modes:
                group = None if mode == _NONE else build.in_place(0, mode, may_overwrite)
                if mode != _NONE and group is None:
                    continue
                c.groups = [] if group is None else [
                    tuple(c.number.get(v, f"s{v}") for v in group[:2]) + group[2:]]
                live = _live(c, _items(c))
                scores[a, b, count, mode] = _Score(max(live[t] for t in times),
                                                   c.method - identity.method, extra)
    mode_sets = [{_NONE}, {_NONE, _NOT_INPUT}]
    if may_overwrite:
        mode_sets.append({_NONE, _NOT_INPUT, _ANY})
    chosen = {}
    for modes in mode_sets:
        for by_macs in (False, True) if max_macs is not None else (False,):
            chosen.update(dict.fromkeys(_choices(scores, live0, modes, by_macs)))
    candidates = []
    for regions in chosen:
        c = _assemble(_build(base, list(regions), may_overwrite), keep_input)
        _place(c)
        candidates.append(c)
    candidates.sort(key=lambda c: (c.total, c.macs, len(c.origin)))
    return candidates


def _method(program: pte.Program) -> int:
    """The index of the one method with instructions; raises pte.Refusal for
    what the rewrite does not take."""
    plans = program.execution_plan or []
    running = [i for i, p in enumerate(plans)
               if any(c.instructions for c in p.chains or ()) or p.delegates]
    if len(running) != 1:
        raise pte.Refusal(f"has {len(running)} methods with instructions; tigris export "
                          f"rewrites programs with one")
    plan = plans[running[0]]
    for delegate in plan.delegates or ():
        raise pte.Refusal(f"delegates to {delegate.id}; tigris export rewrites programs that "
                          f"run on cortex_m operators")
    if program.mutable_data_segments:
        raise pte.Refusal("has mutable data segments; tigris export rewrites programs whose "
                          "data is constant")
    if program.named_data:
        raise pte.Refusal("has named data; tigris export rewrites programs that keep their "
                          "constants in the constant buffer or constant segment")
    for chain in plan.chains or ():
        if chain.inputs or chain.outputs:
            raise pte.Refusal("lists inputs or outputs on its chain; tigris export rewrites "
                              "programs whose chains take the method's inputs and outputs")
        count = len(chain.instructions or ())
        if chain.stacktrace is not None and len(chain.stacktrace) != count:
            raise pte.Refusal(f"is malformed: its chain has {len(chain.stacktrace)} stack "
                              f"traces for {count} instructions")
    sizes = plan.non_const_buffer_sizes or []
    if len(sizes) != 2:
        raise pte.Refusal(f"plans {max(len(sizes) - 1, 0)} buffers; tigris export rewrites "
                          f"programs with one planned buffer")
    for role, indices in (("input", plan.inputs), ("output", plan.outputs)):
        for i in indices or ():
            value = plan.values[i] if 0 <= i < len(plan.values or []) else None
            if not isinstance(value, pte.Tensor) or value.allocation_info is None:
                raise pte.Refusal(f"has method {role} {i} outside the memory plan; tigris "
                                  f"export rewrites programs whose inputs and outputs are "
                                  f"planned tensors")
    return running[0]


def _warnings(base: _Base) -> list:
    if base.classes != (cm.MVE,):
        return []
    text = "lowered for " + cm.describe(base.classes)
    if any(c.signature.name == _LINEAR for c in base.calls):
        return [text + "; firmware without MVE runs its linear layers without their bias"]
    return [text + "; it runs on firmware built with MVE"]


def _offsets(plan: pte.ExecutionPlan, indices) -> list:
    return [plan.values[i].allocation_info.offset for i in indices or ()]


def _unmet(pool: list, arena: int | None, method: int | None) -> list:
    """For each limit no (planned, method, total) of `pool` meets: the limit,
    the smallest figure found and the lowest total among the programs with it."""
    unmet = []
    for limit, k, name in ((arena, 0, "arena"), (method, 1, "method")):
        if limit is None:
            continue
        best = min(pool, key=lambda p: (p[k], p[2]))
        if best[k] > limit:
            unmet.append((name, best[k], best[2]))
    return unmet or [("both", None, None)]


def export(data: bytes, cpu: str | None = None, allow_input_overwrite: bool = False,
           arena: int | None = None, method: int | None = None,
           max_extra_macs: float | None = None) -> PteExport:
    """The program in `data` rewritten for less RAM, or the reason there is
    none. `cpu` names the Cortex-M core and must match the CMSIS-NN class the
    program was lowered for. `arena` bounds the planned buffers, `method` the
    method allocator in bytes, and `max_extra_macs` the extra MACs as a share
    of the program's own."""
    data = bytes(data)
    result = PteExport(data, pte_bytes_before=len(data))
    try:
        if not pte.is_pte(data):
            raise pte.Refusal("is not an ExecuTorch program (no ET12 identifier)")
        try:
            program = pte.read(data)
        except pte.Refusal:
            raise
        except ValueError as error:
            raise pte.Refusal(f"fails to parse: {error}") from None
        index = _method(program)
        plan = program.execution_plan[index]
        try:
            calls = cm.kernel_calls(plan)
            classes = cm.infer_class(plan)
            classes = classes if cpu is None else (cm.check_cpu(classes, cpu),)
            method_before = pm.method_allocator_bytes(plan)
        except pte.Refusal:
            raise
        except ValueError as error:
            raise pte.Refusal(f"is malformed: {error}") from None
        result.classes = classes
        result.cmsis_class = " or ".join(classes)
        result.instructions_before = len(calls)
        result.origin = [(k, None) for k in range(len(calls))]
        result.input_offsets = _offsets(plan, plan.inputs)
        result.output_offsets = _offsets(plan, plan.outputs)
        result.planned_before = list(plan.non_const_buffer_sizes)
        result.method_before = method_before
        result.temp_before = pm.temp_allocator_min_bytes(plan)
        result.macs = program_macs(plan)
        try:
            overwritten = pc.input_overwritten(plan)
            result.input_overwritten_before = overwritten
            pc.check(plan, not overwritten, classes)
        except ValueError as error:
            raise pte.Refusal(f"has a memory plan that fails the check: {error}") from None
        try:
            base = _Base(plan, classes)
        except pte.Refusal:
            raise
        except ValueError as error:
            raise pte.Refusal(f"is malformed: {error}") from None
        result.warnings = _warnings(base)
        may_overwrite = allow_input_overwrite or overwritten
        max_macs = None if max_extra_macs is None else int(max_extra_macs * result.macs)
        candidates = _search(base, may_overwrite, not may_overwrite, max_macs)
    except pte.Refusal as error:
        result.reason, result.refused = str(error), True
        return result
    before = result.total_before

    def meets(planned: int, method_bytes: int) -> bool:
        return (arena is None or planned <= arena) and (method is None or method_bytes <= method)

    found = [c for c in candidates if max_macs is None or c.macs - result.macs <= max_macs]
    fits = [c for c in found if meets(c.planned, c.method)]
    original_fits = meets(sum(result.planned_before), result.method_before)
    if not fits and not original_fits:
        pool = [(c.planned, c.method, c.total) for c in found]
        pool.append((sum(result.planned_before), result.method_before, before))
        result.unmet = _unmet(pool, arena, method)
        return result
    if not fits or (original_fits and fits[0].total >= before):
        return result
    best = fits[0]
    new_plan = _finish(best)
    plans = list(program.execution_plan)
    plans[index] = new_plan
    written = pte.write(replace(program, execution_plan=plans))
    check = pte.read(written).execution_plan[index]
    try:
        pc.check(check, not may_overwrite, classes, original=plan)
    except ValueError as error:
        raise RuntimeError(f"the chosen plan fails the memory check: {error}") from None
    if (pm.method_allocator_bytes(check) != best.method
            or check.non_const_buffer_sizes[1] != best.arena):
        raise RuntimeError("the written program's figures differ from the planned ones")
    result.data, result.written = written, True
    result.planned_after = list(check.non_const_buffer_sizes)
    result.method_after = best.method
    result.temp_after = pm.temp_allocator_min_bytes(check)
    result.extra_macs = best.macs - result.macs
    result.pte_bytes_after = len(written)
    result.regions = list(best.regions)
    result.input_overwritten_after = pc.input_overwritten(check)
    result.instructions_after = len(check.chains[0].instructions)
    result.origin = [(j, band) for j, _, band in best.origin]
    result.input_offsets = _offsets(check, check.inputs)
    result.output_offsets = _offsets(check, check.outputs)
    return result
