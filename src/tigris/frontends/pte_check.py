"""A check of an ExecuTorch program's memory plan, byte by byte.

ExecuTorch places each planned tensor at the offset its allocation_info gives
and checks only that it fits its buffer, so a plan that overwrites bytes a
later instruction still reads loads and runs with wrong results. `check`
refuses such a plan. It takes a planned tensor's bytes from its
allocation_info and size, and which tensors an instruction reads (its inputs)
and writes (its out and scratch tensors) from the cortex_m operator table.

The check replays the method over the bytes of each planned buffer and tags
each byte with the write that last set it: a method input, a scratch tensor,
or an instruction's output together with the byte's position in that output.
Method inputs are written at the start and method outputs read at the end;
instructions are numbered from 0, the start is -1 and the end the instruction
count. An instruction writes no byte it reads, and its writes share no byte.

Every byte an instruction reads must hold an instruction's output or a method
input, never scratch, and a tensor that an instruction writes must still hold
its own last write when it is read. Any other tensor (a view, such as
ExecuTorch makes for a reshape, or a row view inside a larger tensor) may
hold bytes of several writes. What each read holds, byte by byte as producer
and position in its output, is its content.

A rewrite is checked against the contents of its original. Each instruction
must call the operator of an original instruction with the same arguments,
apart from the row counts and places of its planned tensors, its scratch
tensors and a convolution's padding rows, and each tensor it reads must hold
consecutive bytes of what that original instruction reads, with every output
byte of the rewrite standing for the byte of the original instruction it
computes. A convolution reads whole rows and computes, from those rows and
its padding rows, rows the original computes, padding only where the original
pads; an addition reads both operands at one position; any other instruction
reads what the original reads, whole. Original instructions with the same
operator, arguments and contents count as one. At the end each method output
holds what the original's holds, and a kept method input itself. Each byte
then holds what the original computes at its tag, so the rewrite returns the
original's outputs.
"""

from dataclasses import dataclass

import numpy as np

from tigris.frontends import pte
from tigris.frontends import pte_cortex_m as cm

# Offset alignment of every scratch tensor: CMSIS-NN uses its scratch as 16-
# and 32-bit words.
ALIGNMENT = 16
_CHANNELS_LAST = [0, 2, 3, 1]
_CONVOLUTIONS = (cm.NAMESPACE + "quantized_conv2d", cm.NAMESPACE + "quantized_depthwise_conv2d")
_ADD = cm.NAMESPACE + "quantized_add"
# Producer tags of bytes that no write set and of scratch bytes; method input
# k is -1 - k, an instruction its index.
_NONE, _SCRATCH = -(1 << 30), -(1 << 30) + 1


@dataclass(frozen=True)
class _Place:
    memory_id: int
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def overlaps(self, other: "_Place") -> bool:
        return (self.memory_id == other.memory_id and self.offset < other.end
                and other.offset < self.end)


def _planned(plan: pte.ExecutionPlan) -> dict[int, _Place]:
    sizes = plan.non_const_buffer_sizes or []
    places = {}
    for index, value in enumerate(plan.values or []):
        if not isinstance(value, pte.Tensor) or value.allocation_info is None:
            continue
        info = value.allocation_info
        if value.data_buffer_idx:
            raise ValueError(f"value {index} is both planned and constant")
        if value.storage_offset:
            raise ValueError(f"value {index} has storage offset {value.storage_offset}")
        if not 0 < info.memory_id < len(sizes):
            raise ValueError(f"value {index} is planned in buffer {info.memory_id}, which "
                             f"does not exist")
        place = _Place(info.memory_id, info.offset, value.nbytes)
        if place.end > sizes[info.memory_id]:
            raise ValueError(f"value {index} ({place.nbytes} bytes at {place.offset}) runs "
                             f"past buffer {info.memory_id} of {sizes[info.memory_id]} bytes")
        places[index] = place
    return places


def _scratch(plan: pte.ExecutionPlan, calls: list, classes: tuple) -> None:
    """Raise ValueError for a scratch tensor whose size is not the one
    CMSIS-NN asks for that call in each of `classes`."""
    for call in calls:
        need = cm.required_scratch(call, plan)
        if need is None:
            continue
        (position,) = call.signature.scratch
        stored = call.values[position].nbytes
        for cls in classes:
            if need[cls] != stored:
                raise ValueError(f"instruction {call.index} ({call.signature.name}) has a "
                                 f"scratch tensor of {stored} bytes where {cls} kernels need "
                                 f"{need[cls]}")


def _steps(plan: pte.ExecutionPlan, calls: list, places: dict, keep_input: bool) -> list:
    """Per step from the start to the end: (time, reads, writes), each a list
    of planned values, a write as (value, is scratch)."""
    end = len(calls)
    steps = [(-1, [], [(v, False) for v in plan.inputs or [] if v in places])]
    for call in calls:
        args = call.args
        reads = [args[p] for p in call.signature.inputs if args[p] in places]
        writes = []
        for position in (*call.signature.scratch, len(args) - 2):
            if args[position] not in places:
                raise ValueError(f"instruction {call.index} writes value {args[position]}, "
                                 f"which is not planned")
            writes.append((args[position], position != len(args) - 2))
        steps.append((call.index, reads, writes))
    reads = [v for v in plan.outputs or [] if v in places]
    if keep_input:
        reads += [v for v in plan.inputs or [] if v in places]
    steps.append((end, reads, []))
    return steps


def _when(time: int) -> str:
    return "the start" if time < 0 else f"instruction {time}"


def _channels_last(tensor) -> bool:
    return (isinstance(tensor, pte.Tensor) and tensor.allocation_info is not None
            and len(tensor.sizes or []) == 4 and tensor.sizes[0] == 1
            and list(tensor.dim_order or []) == _CHANNELS_LAST)


def _rows(call: cm.Call) -> bool:
    """Whether `call` may compute part of its output's rows: a convolution or
    depthwise convolution of channels-last tensors, or an addition of three
    channels-last tensors of one size."""
    name = call.signature.name
    if name in _CONVOLUTIONS:
        return _channels_last(call.value("input")) and _channels_last(call.value("out"))
    if name == _ADD:
        tensors = [call.value(n) for n in ("self", "other", "out")]
        return (all(_channels_last(t) for t in tensors)
                and all(t.sizes[1::2] == tensors[2].sizes[1::2] for t in tensors))
    return False


def _key(plan: pte.ExecutionPlan, call: cm.Call, rows: bool) -> tuple:
    """The operator and arguments of `call` apart from its scratch tensors
    and planned tensors' places; with `rows`, also apart from a convolution's
    padding rows and the row count of each planned tensor."""
    sig = call.signature
    parts = [sig.name]
    for position, (name, record) in enumerate(zip(sig.names[:-1], call.values[:-1])):
        if position in sig.scratch:
            parts.append("scratch")
        elif isinstance(record, pte.Tensor) and record.allocation_info is not None:
            sizes = list(record.sizes or [])
            if rows:
                sizes[2] = None
            parts.append(("planned", record.scalar_type, tuple(record.dim_order or ()),
                          tuple(sizes)))
        elif isinstance(record, pte.IntList):
            ints = call.ints(name, plan)
            parts.append(("ints", tuple(ints[1:] if rows and name == "padding" else ints)))
        else:
            parts.append(repr(record))
    return tuple(parts)


def _reads(call: cm.Call, places: dict) -> list:
    """Positions of the planned tensors `call` reads."""
    return [p for p in call.signature.inputs if call.args[p] in places]


class _Replay:
    """The bytes of each planned buffer, each holding the number of the write
    that last set it (0 for none) and its position in that write's output;
    per write its value, time, producer tag and position of its first byte."""

    def __init__(self, plan: pte.ExecutionPlan, places: dict):
        self.places = places
        ends = dict.fromkeys(range(1, len(plan.non_const_buffer_sizes or [])), 0)
        for place in places.values():
            ends[place.memory_id] = max(ends[place.memory_id], place.end)
        # Bytes past the last planned tensor are never read or written.
        self.who = {m: np.zeros(end, np.int64) for m, end in ends.items()}
        self.pos = {m: np.zeros_like(who) for m, who in self.who.items()}
        self.writes = [(None, None, _NONE)]  # write number -> (value, time, tag)
        self.tags = np.array([_NONE], np.int64)
        self.last = {}  # value -> number of its last write

    def bytes(self, value: int) -> tuple:
        place = self.places[value]
        return (self.who[place.memory_id][place.offset:place.end],
                self.pos[place.memory_id][place.offset:place.end])

    def write(self, value: int, time: int, tag: int, start: int = 0) -> None:
        self.writes.append((value, time, tag))
        self.tags = np.append(self.tags, tag)
        number = len(self.writes) - 1
        self.last[value] = number
        place = self.places[value]
        self.who[place.memory_id][place.offset:place.end] = number
        self.pos[place.memory_id][place.offset:place.end] = start + np.arange(place.nbytes)

    def wrong(self, reader: int, time: int, k: int, state: bool = False) -> ValueError:
        """The error for byte `k` of `reader`, which does not hold what it
        reads at `time`."""
        who, _ = self.bytes(reader)
        number = int(who[k])
        end = k
        while end < len(who) and who[end] == number:
            end += 1
        place = self.places[reader]
        lo, hi = place.offset + k, place.offset + end
        if not number:
            if state:
                return _state(reader, time)
            return ValueError(f"value {reader} is read at {_when(time)} before anything writes "
                              f"bytes {lo}..{hi} of buffer {place.memory_id}")
        writer, at, tag = self.writes[number]
        if tag == _SCRATCH:
            return ValueError(f"value {reader} is read at {_when(time)} from bytes {lo}..{hi} "
                              f"of buffer {place.memory_id}, which {_when(at)} last wrote as "
                              f"its scratch tensor")
        return ValueError(f"value {writer} is written at {_when(at)} over bytes {lo}..{hi} of "
                          f"buffer {place.memory_id} that value {reader} still needs at "
                          f"{_when(time)}")

    def read(self, value: int, time: int, written: bool) -> tuple:
        """The content of planned `value`, read at `time`: every byte must
        hold an output or a method input, and the value its own last write
        when `written`."""
        if not self.places[value].nbytes:
            return ()
        who, pos = self.bytes(value)
        if not who.all():
            raise self.wrong(value, time, int(np.argmin(who)), True)
        if written:
            own = self.last.get(value, 0)
            if not own:
                raise _state(value, time)
            bad = np.flatnonzero(who != own)
            if bad.size:
                raise self.wrong(value, time, int(bad[0]))
        tags = self.tags[who]
        bad = np.flatnonzero(tags == _SCRATCH)
        if bad.size:
            raise self.wrong(value, time, int(bad[0]))
        return _segments(tags, pos)

    def writers(self, value: int) -> set:
        """The values whose writes the bytes of planned `value` hold."""
        who, _ = self.bytes(value)
        return {self.writes[int(n)][0] for n in np.unique(who) if n}

    def holds(self, value: int, time: int, content: tuple) -> int:
        """The position in `content` of the first byte of planned `value`,
        whose bytes must hold consecutive bytes of `content`."""
        n = self.places[value].nbytes
        if not n:
            return 0
        who, pos = self.bytes(value)
        tags = self.tags[who]
        mine = _segments(tags, pos)
        tag, first, _ = mine[0]
        starts, offset = [], 0
        for other, at, count in content:
            if other == tag and at <= first < at + count:
                start = offset + first - at
                if _slice(content, start, n) == mine:
                    return start
                starts.append(start)
            offset += count
        want_tags, want_pos = _expand(content, starts[0] if starts else 0, n)
        bad = np.flatnonzero((tags != want_tags) | (pos != want_pos))
        raise self.wrong(value, time, int(bad[0]) if bad.size else 0)


def _segments(tags: np.ndarray, pos: np.ndarray) -> tuple:
    """Bytes as runs (tag, position of the first byte, bytes) of consecutive
    positions of one tag, each run as long as it goes."""
    if not len(tags):
        return ()
    cut = np.flatnonzero((tags[1:] != tags[:-1]) | (pos[1:] != pos[:-1] + 1)) + 1
    firsts, ends = np.concatenate(([0], cut)), np.concatenate((cut, [len(tags)]))
    return tuple((int(tags[a]), int(pos[a]), int(b - a)) for a, b in zip(firsts, ends))


def _slice(content: tuple, start: int, n: int) -> tuple | None:
    """Bytes start..start+n of `content` as runs, None past its end."""
    runs, offset = [], 0
    for tag, at, count in content:
        lo, hi = max(start, offset), min(start + n, offset + count)
        if lo < hi:
            runs.append((tag, at + lo - offset, hi - lo))
        offset += count
    return tuple(runs) if sum(r[2] for r in runs) == n else None


def _expand(content: tuple, start: int, n: int) -> tuple:
    """Tag and position of each of bytes start..start+n of `content`, with
    the tag of no write past its end."""
    tags, pos = np.full(n, _NONE, np.int64), np.full(n, -1, np.int64)
    offset = 0
    for tag, at, count in content:
        lo, hi = max(start, offset), min(start + n, offset + count)
        if lo < hi:
            tags[lo - start:hi - start] = tag
            pos[lo - start:hi - start] = at + lo - offset + np.arange(hi - lo)
        offset += count
    return tags, pos


def _state(value: int, time: int) -> ValueError:
    return ValueError(f"value {value} is read at {_when(time)} before anything writes it, so "
                      f"it holds state; tigris export rewrites stateless methods")


def _overlaps(time: int, places: dict, reads: list, writes: list) -> None:
    """Raise ValueError when a write of one instruction shares a byte with
    one of its reads or an earlier of its writes."""
    for k, value in enumerate(writes):
        for other in [*reads, *writes[:k]]:
            a, b = places[value], places[other]
            if a.nbytes and b.nbytes and a.overlaps(b):
                raise ValueError(
                    f"value {value} is written at {_when(time)} over bytes "
                    f"{max(a.offset, b.offset)}..{min(a.end, b.end)} of buffer "
                    f"{a.memory_id} that value {other} still needs at {_when(time)}")


class _Original:
    """The content of every read of a program, with instructions of the same
    operator, arguments and contents merged into the first of them; per read
    value, the values whose writes it holds."""

    def __init__(self, plan: pte.ExecutionPlan, calls: list, places: dict, keep_input: bool):
        self.plan, self.calls, self.places = plan, calls, places
        steps = _steps(plan, calls, places, keep_input)
        written = {v for _, _, writes in steps for v, _ in writes}
        replay = _Replay(plan, places)
        content, self.sources = {}, {}

        def read(value: int, time: int) -> tuple:
            self.sources.setdefault(value, set()).update(replay.writers(value))
            return replay.read(value, time, value in written)

        for k, v in enumerate(plan.inputs or []):
            if v in places:
                replay.write(v, -1, -1 - k)
        for call in calls:
            reads = _reads(call, places)
            for p in reads:
                content[call.index, p] = read(call.args[p], call.index)
            writes = [call.args[p] for p in (*call.signature.scratch, len(call.args) - 2)]
            _overlaps(call.index, places, [call.args[p] for p in reads], writes)
            for p in call.signature.scratch:
                replay.write(call.args[p], call.index, _SCRATCH)
            replay.write(call.args[-2], call.index, call.index)
        end = len(calls)
        outputs = [read(v, end) for v in plan.outputs or [] if v in places]
        if keep_input:
            for k, v in enumerate(plan.inputs or []):
                if v in places and read(v, end) not in ((), _whole(-1 - k, places[v].nbytes)):
                    raise replay.wrong(v, end, 0)
        self.same = {}
        self.content = {}
        self.exact = {}
        self.candidates = {}
        merged = {}
        for call in calls:
            j = call.index
            sources = tuple((p, self.tag(content[j, p])) for p in _reads(call, places))
            self.exact[j] = _key(plan, call, False)
            first = merged.setdefault((self.exact[j], sources), j)
            self.same[j] = first
            if first == j:
                self.content.update({(j, p): runs for p, runs in sources})
                group = self.candidates.setdefault(_key(plan, call, _rows(call)), [])
                group.append(j)
        self.outputs = [self.tag(runs) for runs in outputs]

    def tag(self, content: tuple) -> tuple:
        """`content` with each instruction named by the first it merges into."""
        runs = []
        for tag, at, count in content:
            tag = self.same.get(tag, tag)
            if runs and runs[-1][0] == tag and runs[-1][1] + runs[-1][2] == at:
                runs[-1] = (tag, runs[-1][1], runs[-1][2] + count)
            else:
                runs.append((tag, at, count))
        return tuple(runs)


def _whole(tag: int, nbytes: int) -> tuple:
    return ((tag, 0, nbytes),) if nbytes else ()


def _band(call: cm.Call, plan: pte.ExecutionPlan, original: cm.Call,
          oplan: pte.ExecutionPlan, starts: dict) -> int:
    """The position in its original's output of the first byte `call` writes,
    from the positions in what the original reads of the first bytes it
    reads; raises ValueError when these are not rows its original computes
    from the same bytes."""
    name, t = call.signature.name, call.index
    if not _rows(original):
        for p, start in starts.items():
            if start:
                raise ValueError(f"instruction {t} ({name}) reads value {call.args[p]} from "
                                 f"byte {start} of what the original instruction "
                                 f"{original.index} reads, which reads it whole")
        return 0
    out = call.value("out")
    out_row = out.nbytes // out.sizes[2] if out.sizes[2] else 0
    if name == _ADD:
        a, b = starts[call.signature.index("self")], starts[call.signature.index("other")]
        if a != b or call.value("self").sizes != out.sizes or call.value("other").sizes != (
                out.sizes):
            raise ValueError(f"instruction {t} ({name}) reads its operands at bytes {a} and "
                             f"{b} of what the original instruction {original.index} reads, "
                             f"which reads both at one position")
        return a
    x = call.value("input")
    in_row = x.nbytes // x.sizes[2] if x.sizes[2] else 0
    start = starts[call.signature.index("input")]
    height, rows = original.value("input").sizes[2], original.value("out").sizes[2]
    pad = call.ints("padding", plan)[0]
    first = original.ints("padding", oplan)[0]
    stride = original.ints("stride", oplan)[0]
    dilation = original.ints("dilation", oplan)[0]
    kernel = original.value("weight").sizes[1]
    r0, h_in, h_out = start // max(in_row, 1), x.sizes[2], out.sizes[2]
    o0, skew = divmod(r0 - pad + first, stride)
    taps = (np.arange(h_out)[:, None] * stride - pad
            + np.arange(kernel)[None, :] * dilation).ravel()
    above, below = taps[taps < 0], taps[taps >= h_in]
    if (not in_row or start % in_row or pad < 0 or skew or o0 < 0 or o0 + h_out > rows
            or r0 + h_in > height or (above.size and r0 + above.max() >= 0)
            or (below.size and r0 + below.min() < height)):
        raise ValueError(f"instruction {t} ({name}) computes {h_out} rows from bytes "
                         f"{start}..{start + x.nbytes} of what the original reads with "
                         f"{pad} padding rows above, which are no rows the original "
                         f"instruction {original.index} computes")
    return o0 * out_row


def _io(plan: pte.ExecutionPlan, v: int) -> tuple:
    t = plan.values[v]
    if not isinstance(t, pte.Tensor) or t.allocation_info is None:
        return (repr(t),)
    return t.scalar_type, t.sizes, t.dim_order


def _rewrite(plan: pte.ExecutionPlan, calls: list, places: dict, keep_input: bool,
             original: _Original) -> None:
    """Raise ValueError when `plan` does not compute what `original` does."""
    oplan, ocalls = original.plan, original.calls
    for role, mine, theirs in (("input", plan.inputs, oplan.inputs),
                               ("output", plan.outputs, oplan.outputs)):
        if [_io(plan, v) for v in mine or []] != [_io(oplan, v) for v in theirs or []]:
            raise ValueError(f"the method {role}s differ from the original's")
    replay = _Replay(plan, places)
    for k, v in enumerate(plan.inputs or []):
        if v in places:
            replay.write(v, -1, -1 - k)
    for call in calls:
        reads = _reads(call, places)
        group = original.candidates.get(_key(plan, call, _rows(call)), [])
        if not group:
            raise ValueError(f"instruction {call.index} ({call.signature.name}) has the "
                             f"operator and arguments of no original instruction")
        exact = _key(plan, call, False)
        errors = []
        for j in sorted(group, key=lambda j: original.exact[j] != exact):
            try:
                starts = {p: replay.holds(call.args[p], call.index, original.content[j, p])
                          for p in reads}
                start = _band(call, plan, ocalls[j], oplan, starts)
                break
            except ValueError as error:
                errors.append(error)
        else:
            raise errors[0]
        writes = [call.args[p] for p in (*call.signature.scratch, len(call.args) - 2)]
        _overlaps(call.index, places, [call.args[p] for p in reads], writes)
        for p in call.signature.scratch:
            replay.write(call.args[p], call.index, _SCRATCH)
        replay.write(call.args[-2], call.index, j, start)
    end = len(calls)
    ends = list(zip([v for v in plan.outputs or [] if v in places], original.outputs))
    if keep_input:
        ends += [(v, _whole(-1 - k, places[v].nbytes))
                 for k, v in enumerate(plan.inputs or []) if v in places]
    for v, content in ends:
        if replay.holds(v, end, content):
            raise replay.wrong(v, end, 0)


def check(plan: pte.ExecutionPlan, keep_input: bool = True, classes: tuple = (),
          original: pte.ExecutionPlan | None = None) -> None:
    """Raise ValueError when the plan of `plan` lets a write overwrite a byte
    that is still read later, a planned tensor leaves its buffer, a scratch
    tensor is not aligned or its size is not what CMSIS-NN asks for in each
    of `classes`, or a planned tensor is read before it is written. With
    `original`, `plan` is a rewrite of it and must compute its outputs. With
    `keep_input`, the method inputs stay live to the end."""
    calls = cm.kernel_calls(plan)
    _scratch(plan, calls, classes)
    places = _planned(plan)
    for _, _, writes in _steps(plan, calls, places, keep_input):
        for value, scratch in writes:
            if scratch and places[value].offset % ALIGNMENT:
                raise ValueError(f"scratch value {value} is at offset {places[value].offset}, "
                                 f"which is not a multiple of {ALIGNMENT}")
    if original is None:
        _Original(plan, calls, places, keep_input)
        return
    ocalls = cm.kernel_calls(original)
    _rewrite(plan, calls, places, keep_input,
             _Original(original, ocalls, _planned(original), False))


def sources(plan: pte.ExecutionPlan) -> dict:
    """For a plan `check` accepts: per planned value that an instruction or
    the end reads, the values whose writes its bytes hold."""
    return _Original(plan, cm.kernel_calls(plan), _planned(plan), False).sources


def input_overwritten(plan: pte.ExecutionPlan) -> bool:
    """Whether an instruction writes a byte of a method input."""
    places = _planned(plan)
    inputs = [places[i] for i in plan.inputs or [] if i in places and places[i].nbytes]
    for time, _, writes in _steps(plan, cm.kernel_calls(plan), places, False):
        if time >= 0 and any(places[v].overlaps(i) for v, _ in writes for i in inputs):
            return True
    return False
