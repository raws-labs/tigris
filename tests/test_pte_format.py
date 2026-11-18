"""The ExecuTorch program reader and writer: byte-identical round trips,
the fields read, and refusal of files it cannot reproduce."""

import importlib
import os
import random
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tigris.frontends import pte
from tigris.frontends.flatbuffer import Table
from tigris.frontends.pte_schema import HEADER, HEADER_OFFSET, SCALAR_TYPES

FIXTURES = Path(__file__).parent / "fixtures" / "pte"
PTES = sorted(FIXTURES.glob("*.pte"))
_CHAR, _INT = 1, 3


def _require(module: str):
    """Skip without `module`; fail under TIGRIS_REQUIRE_PYTORCH."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if os.environ.get("TIGRIS_REQUIRE_PYTORCH"):
            pytest.fail(f"{module} is not installed")
        pytest.skip(f"{module} is not installed")


def _data(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _kernels(plan: pte.ExecutionPlan) -> list[tuple[str, list]]:
    return [(plan.operators[i.op_index].name, [plan.values[a] for a in i.args])
            for i in plan.chains[0].instructions]


def test_fixtures_cover_each_class():
    assert {p.stem.rsplit("_", 1)[1] for p in PTES} == {"m0", "m7", "m55"}
    assert all(p.stat().st_size < 60_000 for p in PTES)


@pytest.mark.parametrize("path", PTES, ids=lambda p: p.stem)
def test_round_trip_is_byte_identical(path):
    data = path.read_bytes()
    program = pte.read(data)
    assert pte.write(program) == data
    assert pte.read(pte.write(program)) == program


@pytest.mark.parametrize("path", PTES, ids=lambda p: p.stem)
def test_vtable_rule_is_detected(path):
    """Both ways of sharing vtables write a valid program; the reader finds
    the one each file was written with."""
    data = path.read_bytes()
    program = pte.read(data)
    other = pte.write(replace(program, vtables_by_size=not program.vtables_by_size))
    assert other != data
    again = pte.read(other)
    assert again.vtables_by_size != program.vtables_by_size
    assert pte.write(again) == other
    assert replace(again, vtables_by_size=program.vtables_by_size, header=None) == replace(
        program, header=None)


def test_builder_shares_vtables_like_installed_flatbuffers():
    """Two tables with the same field offsets but different inline sizes: one
    rule shares their vtable, the other does not, and the installed FlatBuffers
    builder follows the rule of its version."""
    import flatbuffers

    ref = flatbuffers.Builder(0)
    ref.StartObject(1)
    ref.PrependInt32Slot(0, 7, 0)
    first = ref.EndObject()
    ref.StartObject(1)
    ref.PrependInt64Slot(0, 9, 0)
    second = ref.EndObject()
    ref.StartVector(4, 2, 4)
    ref.PrependUOffsetTRelative(second)
    ref.PrependUOffsetTRelative(first)
    vector = ref.EndVector()
    ref.StartObject(1)
    ref.PrependUOffsetTRelativeSlot(0, vector, 0)
    ref.Finish(ref.EndObject(), file_identifier=b"ET12")
    expected = bytes(ref.Output())

    built = {}
    for by_size in (False, True):
        b = pte._Builder(by_size)
        b.start(1)
        b.add_scalar(0, "i", 7, 0)
        first = b.end()
        b.start(1)
        b.add_scalar(0, "q", 9, 0)
        second = b.end()
        vector = b.offsets([first, second])
        b.start(1)
        b.add_offset(0, vector)
        built[by_size] = b.finish(b.end(), b"ET12")
    assert built[False] != built[True]
    version = tuple(int(part) for part in flatbuffers.__version__.split(".")[:2])
    assert built[version >= (25, 12)] == expected


def test_header_padded_for_a_larger_alignment():
    """ExecuTorch pads the extended header to the largest force_align it
    serialized with, here 64 bytes for constant tensors."""
    data = _data("resblock_align64_m7.pte")
    program = pte.read(data)
    assert program.header_size == 64
    assert not any(data[HEADER_OFFSET + HEADER.size:HEADER_OFFSET + 64])
    plain = pte.read(_data("resblock_m7.pte"))
    assert program.header.program_size - 64 == plain.header.program_size - 32
    assert struct.unpack_from("<I", data, 0)[0] == struct.unpack_from(
        "<I", _data("resblock_m7.pte"), 0)[0] + 32
    gap = bytearray(data)
    gap[HEADER_OFFSET + HEADER.size + 5] = 1
    with pytest.raises(ValueError, match="is laid out differently from the programs ExecuTorch 1.6 writes"):
        pte.read(bytes(gap))
    with pytest.raises(ValueError, match="does not keep the program aligned"):
        pte.write(replace(program, header_size=48))


def test_fields_of_a_residual_program():
    program = pte.read(_data("resblock_m7.pte"))
    plan = program.plan()
    assert plan.name == "forward"
    assert sorted({op.name for op in plan.operators}) == [
        "cortex_m::quantized_add", "cortex_m::quantized_avg_pool2d",
        "cortex_m::quantized_conv2d", "cortex_m::quantized_linear"]
    assert all(op.overload == "out" for op in plan.operators)
    assert [name.split("::")[1] for name, _ in _kernels(plan)] == [
        "quantized_conv2d", "quantized_conv2d", "quantized_conv2d", "quantized_add",
        "quantized_avg_pool2d", "quantized_linear"]
    assert plan.non_const_buffer_sizes[0] == 0
    assert len(plan.chains) == 1
    (x,), (y,) = [plan.values[i] for i in plan.inputs], [plan.values[i] for i in plan.outputs]
    assert x.scalar_type == y.scalar_type == _CHAR
    assert x.sizes == [1, 3, 16, 16] and x.dim_order == [0, 2, 3, 1]
    assert y.sizes == [1, 10] and y.nbytes == 10
    for tensor in (x, y):
        assert tensor.allocation_info.memory_id == 1
        assert tensor.allocation_info.offset + tensor.nbytes <= plan.non_const_buffer_sizes[1]
    weights = [v for v in plan.values if isinstance(v, pte.Tensor) and v.data_buffer_idx]
    assert weights and all(v.allocation_info is None for v in weights)
    assert program.constant_segment.segment_index == 0
    assert len(program.constant_segment.offsets) == 1 + max(v.data_buffer_idx for v in weights)
    assert program.header.segment_base_offset % 128 == 0
    assert program.header.segment_data_size == sum(s.size for s in program.segments)
    assert [len(d) for d in program.segment_data] == [s.size for s in program.segments]
    assert program.named_data is None and program.mutable_data_segments is None
    assert all(isinstance(v, (pte.Tensor, pte.Int, pte.IntList, pte.Null, pte.Bool,
                              pte.Double, pte.String, pte.TensorList,
                              pte.OptionalTensorList))
               for v in plan.values)


def test_depthwise_scratch_differs_by_class():
    """The 5x5 depthwise convolution's scratch tensor is the one argument the
    three lowerings of the same model plan differently."""
    scratch = {}
    for cpu in ("m0", "m7", "m55"):
        plan = pte.read(_data(f"chain_{cpu}.pte")).plan()
        (args,) = [a for name, a in _kernels(plan) if name.endswith("depthwise_conv2d")]
        scratch[cpu] = args[13].nbytes
    assert scratch == {"m0": 0, "m7": 800, "m55": 12400}


def test_inputs_and_outputs_match_the_program():
    for path in PTES:
        record = np.load(path.with_suffix(".npz"))
        plan = pte.read(path.read_bytes()).plan()
        (x,), (y,) = [plan.values[i] for i in plan.inputs], [plan.values[i] for i in plan.outputs]
        inputs = record["inputs"]
        assert inputs.dtype == np.int8 and inputs[0].nbytes == x.nbytes
        if "output_bytes" in record:
            assert record["output_bytes"].shape == (len(inputs), y.nbytes)
        else:
            assert path.stem.endswith("_m55")


def test_edit_moves_the_segments():
    """A program that grows keeps its segments, behind a new extended header."""
    data = _data("chain_m7.pte")
    program = pte.read(data)
    plan = program.plan()
    plan.values.append(pte.Int(int_val=7))
    plan.values.append(pte.String(string_val="x" * 300))
    out = pte.write(program)
    again = pte.read(out)
    assert again.plan().values[-2:] == [pte.Int(7), pte.String("x" * 300)]
    assert again.segment_data == program.segment_data
    assert again.header.program_size > program.header.program_size
    assert again.header.segment_base_offset % 128 == 0
    assert again.header.segment_base_offset + again.header.segment_data_size == len(out)


def test_scalar_types_have_sizes():
    assert SCALAR_TYPES[_CHAR] == ("CHAR", 1) and SCALAR_TYPES[_INT] == ("INT", 4)
    with pytest.raises(ValueError, match="unknown scalar type"):
        pte.Tensor(scalar_type=99, sizes=[1]).nbytes


@pytest.mark.parametrize("path", PTES[:2], ids=lambda p: p.stem)
def test_truncated_file_is_refused(path):
    data = path.read_bytes()
    for n in range(len(data)):
        with pytest.raises(ValueError):
            pte.read(data[:n])


def test_truncated_flatbuffer_is_refused():
    """Without segments a .pte is a bare FlatBuffer, which ends in the first
    string written: a cut anywhere loses part of the program."""
    program = pte.read(_data("resblock_m7.pte"))
    bare = replace(program, segments=[], segment_data=[], header=None)
    data = pte.write(bare)
    assert data[HEADER_OFFSET:HEADER_OFFSET + 4] != b"eh00"
    assert pte.read(data) == bare
    for n in range(len(data)):
        with pytest.raises(ValueError):
            pte.read(data[:n])


def test_corrupt_bytes_are_refused_or_written_back():
    """A corrupted program either raises ValueError or reads as records that
    write back to the same bytes."""
    data = _data("resblock_m0.pte")
    size = pte.read(data).header.program_size
    rng = random.Random(0)
    accepted = 0
    for _ in range(300):
        corrupt = bytearray(data)
        for _ in range(3):
            corrupt[rng.randrange(HEADER_OFFSET + 32, size)] = rng.randrange(256)
        try:
            program = pte.read(bytes(corrupt))
        except ValueError:
            continue
        accepted += 1
        assert pte.write(program) == corrupt
    assert accepted


def test_unknown_field_is_refused(monkeypatch):
    """An operator table written with a third field, which the schema does not have."""
    program = pte.read(_data("chain_m7.pte"))
    kind, fields = pte._SPEC[pte.Operator]
    monkeypatch.setitem(pte._SPEC, pte.Operator, (kind, fields + (("name", 2, "s"),)))
    monkeypatch.setitem(pte.FIELDS, kind, 3)
    data = pte.write(program)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="the operator table sets field 2"):
        pte.read(data)


def test_unknown_union_type_is_refused(monkeypatch):
    """An Int value written under union type 12, past the schema's eleven members."""
    program = pte.read(_data("chain_m7.pte"))
    kind, members = pte._UNIONS["values"]
    moved = members[:2] + (type("Unused", (), {}),) + members[3:] + (pte.Int,)
    monkeypatch.setitem(pte._UNIONS, "values", (kind, moved))
    data = pte.write(program)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="union type 12"):
        pte.read(data)


def _evalue_tables(data: bytes) -> list[Table]:
    program = pte.read(data)
    root = Table.root(data[:program.header.program_size])
    (plan,) = [t for t in root.tables(1) if t.text(0) == "forward"]
    return plan.tables(2)


def test_union_type_without_value_is_refused():
    """The value slot of the first value's vtable cleared: its union type
    names a member that is absent."""
    data = bytearray(_data("chain_m7.pte"))
    first = _evalue_tables(bytes(data))[0]
    vtable = first.position - struct.unpack_from("<i", data, first.position)[0]
    struct.pack_into("<H", data, vtable + 4 + 2 * 1, 0)
    with pytest.raises(ValueError, match="union type and value disagree"):
        pte.read(bytes(data))


def test_bool_that_is_not_0_or_1_is_refused(monkeypatch):
    program = pte.read(_data("chain_m7.pte"))
    program.plan().values.append(pte.Bool(bool_val=2))
    monkeypatch.setitem(pte._SPEC, pte.Bool, ("bool", (("bool_val", 0, "B"),)))
    data = pte.write(program)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="neither 0 nor 1"):
        pte.read(data)


def test_malformed_string_is_refused():
    data = _data("chain_m7.pte")
    name = b"cortex_m::quantized_conv2d"
    at = data.index(name + b"\0")
    unterminated = bytearray(data)
    unterminated[at + len(name)] = ord("x")
    with pytest.raises(ValueError, match="not terminated"):
        pte.read(bytes(unterminated))
    invalid = bytearray(data)
    invalid[at] = 0xFF
    with pytest.raises(ValueError, match="not UTF-8"):
        pte.read(bytes(invalid))


def test_flatbuffer_it_cannot_rebuild_is_refused():
    """Valid FlatBuffers that ExecuTorch does not write: trailing bytes, a
    non-zero padding byte, and a vtable with an extra zero entry."""
    message = "is laid out differently from the programs ExecuTorch 1.6 writes"
    bare = pte.write(replace(pte.read(_data("resblock_m7.pte")),
                             segments=[], segment_data=[], header=None))
    for tail in (bytes(8), b"GARBAGE!"):
        with pytest.raises(ValueError, match=message):
            pte.read(bare + tail)

    # Program(version=1): root offset, identifier, 2 bytes of padding, the
    # vtable (6, 8, 4), the table's vtable offset and the version.
    small = pte.write(pte.Program(version=1))
    assert small[8:20] == bytes(2) + struct.pack("<3Hi", 6, 8, 4, 6)
    assert pte.read(small).version == 1
    padded = bytearray(small)
    padded[9] = 1
    with pytest.raises(ValueError, match=message):
        pte.read(bytes(padded))
    longer = small[:8] + struct.pack("<4Hi", 8, 8, 4, 0, 8) + small[20:]
    assert Table.root(longer).scalar(0, "I") == 1
    with pytest.raises(ValueError, match=message):
        pte.read(longer)


def test_segment_alignment_is_kept():
    """A program serialized with a larger segment alignment keeps it when it
    grows. Every segment but a named-data one is aligned alike, so a second
    segment's offset bounds the alignment."""
    program = pte.read(_data("chain_m7.pte"))
    aligned = pte.read(pte.write(replace(program, segment_alignment=4096)))
    assert aligned.segment_alignment >= 4096
    aligned.plan().values.append(pte.String(string_val="x" * 3000))
    grown = pte.read(pte.write(aligned))
    assert grown.header.segment_base_offset % 4096 == 0
    assert grown.header.segment_base_offset > aligned.header.segment_base_offset

    def second(offset: int) -> pte.Program:
        return replace(program, segment_alignment=4096,
                       segments=program.segments + [pte.DataSegment(offset=offset, size=4)],
                       segment_data=program.segment_data + [b"abcd"])

    assert pte.read(pte.write(second(4096))).segment_alignment == 4096
    with pytest.raises(ValueError, match="first aligned offset"):
        pte.read(pte.write(second(2432)))
    named = replace(second(2432), named_data=[pte.NamedData(key="w", segment_index=1)])
    assert pte.read(pte.write(named)).segment_alignment >= 4096


def test_layout_it_cannot_reproduce_is_refused():
    data = _data("chain_m0.pte")
    program = pte.read(data)
    header = program.header
    with pytest.raises(ValueError, match="does not match the file"):
        pte.read(data + b"\0")
    gap = bytearray(data)
    gap[header.program_size] = 1
    with pytest.raises(ValueError, match="not zero"):
        pte.read(bytes(gap))
    longer = bytearray(data)
    struct.pack_into("<I", longer, HEADER_OFFSET + 4, HEADER.size + 8)
    with pytest.raises(ValueError, match="extended header of 40 bytes"):
        pte.read(bytes(longer))
    # The segment data moved 256 bytes later, past the first aligned offset.
    base = header.segment_base_offset + 256
    late = bytearray(data[:header.program_size] + bytes(base - header.program_size)
                     + data[header.segment_base_offset:])
    struct.pack_into("<Q", late, HEADER_OFFSET + 16, base)
    with pytest.raises(ValueError, match="first aligned offset"):
        pte.read(bytes(late))
    with pytest.raises(ValueError, match="not an ExecuTorch program"):
        pte.read(b"\0" * 64)


def test_write_checks_segments():
    program = pte.read(_data("chain_m0.pte"))
    with pytest.raises(ValueError, match="records"):
        pte.write(replace(program, segment_data=[program.segment_data[0][:-1]]))
    with pytest.raises(ValueError, match="segment payloads"):
        pte.write(replace(program, segment_data=[]))


def test_reader_needs_neither_torch_nor_executorch():
    code = ("import sys; from tigris.frontends import pte; "
            f"pte.write(pte.read(open({str(PTES[0])!r}, 'rb').read())); "
            "print(sorted(m for m in ('torch', 'executorch', 'flatbuffers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_records_agree_with_executorch():
    """ExecuTorch's own deserializer reads the same program."""
    _require("executorch")
    from executorch.exir._serialize._program import deserialize_pte_binary

    for path in PTES:
        data = path.read_bytes()
        ours = pte.read(data).plan()
        theirs = deserialize_pte_binary(data).program.execution_plan[0]
        assert [(o.name, o.overload) for o in ours.operators] == [
            (o.name, o.overload) for o in theirs.operators]
        assert ours.non_const_buffer_sizes == list(theirs.non_const_buffer_sizes)
        assert ours.inputs == list(theirs.inputs) and ours.outputs == list(theirs.outputs)
        assert len(ours.values) == len(theirs.values)
        for mine, value in zip(ours.values, theirs.values):
            value = value.val
            assert type(mine).__name__ == type(value).__name__
            if isinstance(mine, pte.Tensor):
                assert mine.sizes == list(value.sizes)
                assert mine.dim_order == list(value.dim_order)
                assert mine.scalar_type == int(value.scalar_type)
                assert mine.data_buffer_idx == value.data_buffer_idx
                info = value.allocation_info
                assert (mine.allocation_info is None) == (info is None)
                if info is not None:
                    assert mine.allocation_info.offset == info.memory_offset_low + (
                        info.memory_offset_high << 32)
            elif isinstance(mine, pte.Int):
                assert mine.int_val == value.int_val
            elif isinstance(mine, (pte.IntList, pte.TensorList, pte.OptionalTensorList)):
                assert mine.items == list(value.items)
        for mine, instruction in zip(ours.chains[0].instructions, theirs.chains[0].instructions):
            args = instruction.instr_args
            assert (mine.op_index, mine.args) == (args.op_index, list(args.args))
