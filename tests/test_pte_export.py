"""ExecuTorch program export: written programs re-read and pass the memory
check, the check refuses broken plans, and refusals name their reason."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tigris.frontends import pte
from tigris.frontends import pte_check as pc
from tigris.frontends import pte_cortex_m as cm
from tigris.frontends import pte_export as pe
from tigris.frontends import pte_method as pm
from tigris.frontends.pte_cortex_m import DSP, MVE, SCALAR
from scripts.pte_host_check import MOVED, input_past_plan, summary, variants

FIXTURES = Path(__file__).parent / "fixtures" / "pte"
_CLASS = {"m0": SCALAR, "m7": DSP, "m55": MVE}
# Method allocator bytes of the written bands programs, measured on an
# emulated Cortex-M7 (QEMU mps2-an500) running the ExecuTorch 1.6.0 runtime
# with a 64 KiB temp allocator.
_WRITTEN_METHOD = {"bands_m0": 7780, "bands_m7": 7780, "bands_m55": 7788}


def _data(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _plan(name: str) -> pte.ExecutionPlan:
    return pte.read(_data(name)).execution_plan[0]


def _written(program: pte.Program, plan: pte.ExecutionPlan) -> bytes:
    return pte.write(replace(program, execution_plan=[plan, *program.execution_plan[1:]]))


def _int_list(plan: pte.ExecutionPlan, ints) -> int:
    for value in ints:
        plan.values.append(pte.Int(int_val=value))
    plan.values.append(pte.IntList(items=list(range(len(plan.values) - len(ints),
                                                     len(plan.values)))))
    return len(plan.values) - 1


@pytest.mark.parametrize("name", ["bands_m0", "bands_m7", "bands_m55", "chain_m55"])
def test_written_programs_pass_the_check(name):
    data = _data(f"{name}.pte")
    result = pe.export(data)
    assert result.written and result.reason is None
    assert result.cmsis_class == _CLASS[name.rsplit("_", 1)[1]]
    assert result.total_after < result.total_before
    original, program = pte.read(data), pte.read(result.data)
    plan = program.execution_plan[0]
    pc.check(plan, not result.input_overwritten_before, (result.cmsis_class,),
             original=original.execution_plan[0])
    assert result.planned_after == plan.non_const_buffer_sizes
    assert result.method_after == pm.method_allocator_bytes(plan)
    assert result.temp_after == result.temp_before == pm.temp_allocator_min_bytes(plan)
    assert result.pte_bytes_after == len(result.data)
    assert result.macs + result.extra_macs == pe.program_macs(plan)
    assert result.input_overwritten_after == pc.input_overwritten(plan)
    # Only the method's values, instructions and plan change.
    assert plan.operators == original.execution_plan[0].operators
    assert program.execution_plan[1:] == original.execution_plan[1:]
    assert program.segment_data == original.segment_data
    assert program.constant_buffer == original.constant_buffer
    assert {plan.operators[i.op_index].name for i in plan.chains[0].instructions} <= {
        o.name for o in original.execution_plan[0].operators}
    assert result.regions and all(r.bands >= 2 for r in result.regions)
    if name in _WRITTEN_METHOD:
        assert result.method_after == _WRITTEN_METHOD[name]


def test_bands_fixture_figures():
    result = pe.export(_data("bands_m7.pte"))
    assert result.planned_before == [0, 28224] and result.method_before == 4212
    assert result.planned_after == [0, 13632] and result.method_after == 7780
    assert result.total_before == 32884 and result.total_after == 21860
    assert [(r.first, r.last, r.bands, r.in_place) for r in result.regions] == [
        (2, 7, 6, True)]
    assert (result.input_overwritten_before, result.input_overwritten_after) == (True, True)
    assert result.warnings == []


def test_export_is_deterministic():
    for name in ("bands_m7.pte", "bands_m55.pte"):
        first, second = pe.export(_data(name)), pe.export(_data(name))
        assert first.data == second.data


def test_written_programs_run_bit_for_bit(tmp_path):
    """Needs TIGRIS_PTE_RUNNER, a host runner of the ExecuTorch 1.6.0 runtime
    with the cortex_m operators on CMSIS-NN's portable C kernels (usage:
    `runner model.pte input.bin output.bin`); the scalar and DSP programs'
    outputs must equal those the generator recorded."""
    runner = os.environ.get("TIGRIS_PTE_RUNNER")
    if not runner:
        pytest.skip("TIGRIS_PTE_RUNNER is not set")
    for name in ("bands_m0", "bands_m7"):
        record = np.load(FIXTURES / f"{name}.npz")
        (tmp_path / "model.pte").write_bytes(pe.export(_data(f"{name}.pte")).data)
        record["inputs"].tofile(tmp_path / "input.bin")
        subprocess.run([runner, str(tmp_path / "model.pte"), str(tmp_path / "input.bin"),
                        str(tmp_path / "output.bin")], check=True, capture_output=True)
        out = np.fromfile(tmp_path / "output.bin", np.int8).reshape(record["output_bytes"].shape)
        assert np.array_equal(out, record["output_bytes"]), name


_RECORD = json.loads((FIXTURES / "written.json").read_text())


def test_every_fixture_is_recorded():
    assert sorted(_RECORD) == sorted(p.name for p in FIXTURES.glob("*.pte"))


@pytest.mark.parametrize("name", sorted(_RECORD))
def test_written_programs_match_the_checked_ones(name):
    """Every variant scripts/pte_host_check.py exports for a fixture is the
    program it ran on the ExecuTorch runtime and recorded in written.json, or
    is again not written; a change to what export writes needs that check
    rerun with --record, for the M55 fixtures with `--classes MVE` and the
    runner built for an emulated Cortex-M55 as
    src/tigris/frontends/pte_host_runner.cpp describes."""
    data, expected = _data(name), _RECORD[name]
    moved = input_past_plan(data)
    names = variants(cm.infer_class(pte.read(data).execution_plan[0]))
    assert sorted(expected) == sorted(n for n, _ in names)
    for variant, kwargs in names:
        result = pe.export(moved if variant.startswith(MOVED) else data, **kwargs)
        assert (summary(result) if result.written else None) == expected[variant], variant


def test_original_programs_pass_the_check():
    for path in sorted(FIXTURES.glob("*.pte")):
        plan = pte.read(path.read_bytes()).execution_plan[0]
        pc.check(plan, not pc.input_overwritten(plan), cm.infer_class(plan))


def _band(name: str, mode: int):
    """The bands fixture's written region rebuilt in `mode`, and its base."""
    plan = _plan(name)
    base = pe._Base(plan, cm.infer_class(plan))
    region = pe.export(_data(name)).regions[0]
    return base, pe._build(base, [(region.first, region.last, region.bands, mode)], True)


def _checked(build: pe._Build) -> pte.ExecutionPlan:
    c = pe._assemble(build, False)
    pe._place(c)
    return pe._finish(c)


def test_check_refuses_an_output_one_row_past_the_in_place_bound():
    base, build = _band("bands_m7.pte", pe._ANY)
    x, y, delta = build.groups[0]
    pc.check(_checked(build), False, (DSP,), original=base.plan)
    # Bytes of the input a band still reads when a band writes, as offsets of the output in it.
    bands, external = build.reads[0]
    last = build.regions[0].last
    in_row, out_row = pe._row_bytes(build.values[x]), pe._row_bytes(build.values[y])
    bad = [(r0 * in_row - hi * out_row, r1 * in_row - lo * out_row)
           for k, ((lo, hi), _, _) in enumerate(bands) for kr, j, r0, r1 in external[x]
           if kr > k or (kr == k and j == last)]
    assert any(lo == delta for lo, _ in bad)
    build.groups[0] = (x, y, delta + out_row)
    with pytest.raises(ValueError, match="still needs"):
        pc.check(_checked(build), False, (DSP,), original=base.plan)


def test_check_refuses_a_collision():
    plan = _plan("bands_m7.pte")
    call = plan.chains[0].instructions[1]
    source, out = plan.values[call.args[0]], plan.values[call.args[-2]]
    plan.values[call.args[-2]] = replace(out, allocation_info=source.allocation_info)
    with pytest.raises(ValueError, match=r"written at instruction 1 over bytes .* still needs "
                       r"at instruction 1"):
        pc.check(plan, False, (DSP,))


def test_check_refuses_a_wrong_scratch():
    plan = _plan("bands_m7.pte")
    call = plan.chains[0].instructions[0]
    scratch = plan.values[call.args[12]]
    plan.values[call.args[12]] = replace(scratch, sizes=[scratch.sizes[0] - 16])
    with pytest.raises(ValueError, match=r"instruction 0 \(cortex_m::quantized_conv2d\) has a "
                       r"scratch tensor of 96 bytes where DSP kernels need 112"):
        pc.check(plan, False, (DSP,))


def _with_state(plan: pte.ExecutionPlan) -> int:
    """A planned tensor no instruction writes, read by instruction 1, at the
    end of the buffer."""
    call = plan.chains[0].instructions[1]
    source = plan.values[call.args[0]]
    size = plan.non_const_buffer_sizes[1]
    plan.values.append(replace(source, allocation_info=pte.AllocationDetails(1, size, 0)))
    plan.non_const_buffer_sizes = [0, size + source.nbytes]
    call.args[0] = len(plan.values) - 1
    return call.args[0]


def test_check_refuses_a_state_tensor():
    plan = _plan("bands_m7.pte")
    state = _with_state(plan)
    with pytest.raises(ValueError, match=f"value {state} is read at instruction 1 before "
                       "anything writes it"):
        pc.check(plan, False, (DSP,))
    program = pte.read(_data("bands_m7.pte"))
    _with_state(program.execution_plan[0])
    result = pe.export(_written(program, program.execution_plan[0]))
    assert result.refused and not result.written
    assert result.reason.startswith("has a memory plan that fails the check: value ")
    assert result.reason.endswith("so it holds state; tigris export rewrites stateless methods")


def test_check_refuses_misplaced_tensors():
    plan = _plan("bands_m7.pte")
    call = plan.chains[0].instructions[0]
    scratch = plan.values[call.args[12]]
    plan.values[call.args[12]] = replace(scratch, allocation_info=pte.AllocationDetails(1, 8, 0))
    with pytest.raises(ValueError, match=f"scratch value {call.args[12]} is at offset 8, which "
                       "is not a multiple of 16"):
        pc.check(plan, False, (DSP,))
    plan = _plan("bands_m7.pte")
    out = plan.values[call.args[-2]]
    plan.values[call.args[-2]] = replace(out, allocation_info=pte.AllocationDetails(1, 28224, 0))
    with pytest.raises(ValueError, match="runs past buffer 1"):
        pc.check(plan, False, (DSP,))


def test_finish_places_views_by_their_bytes():
    """Every view of the planner's map moved onto another root, at an offset
    that keeps its bytes: the same plan, which passes."""
    base, build = _band("bands_m7.pte", pe._ANY)
    c = pe._assemble(build, False)
    pe._place(c)
    right = pe._finish(c)
    views = {v: c.places[r] + o for v, (r, o) in c.roots.items() if r != v}
    assert views
    for v, at in views.items():
        other = next(r for r in c.places if r != c.roots[v][0])
        c.roots[v] = (other, at - c.places[other])
    wrong = pe._finish(c)
    assert wrong == right
    pc.check(wrong, False, (DSP,), original=base.plan)


def test_check_refuses_a_view_map_whose_bytes_collide():
    """The planner's map claims band 0's output as a view of the region input,
    at the rows past those band 0 reads: band 1 reads them after band 0
    wrote there."""
    base, build = _band("bands_m7.pte", pe._NONE)
    bands, external = build.reads[0]
    (x,) = external
    first, last = build.regions[0].first, build.regions[0].last
    reads = {k: (r0, r1) for k, j, r0, r1 in external[x] if j == first}
    (y0,) = [args[-2] for (_, args), (j, _, k) in zip(build.instructions, build.origin)
             if j == last and k == 0]
    offset = reads[0][1] * pe._row_bytes(build.values[x])
    assert reads[1][0] * pe._row_bytes(build.values[x]) <= offset
    assert offset + build.values[y0].nbytes <= build.values[x].nbytes
    build.view_of[y0] = (x, offset)
    with pytest.raises(ValueError, match=r"is written at instruction 7 over bytes \d+\.\.\d+ of "
                       r"buffer 1 that value \d+ still needs at instruction 8"):
        pc.check(_checked(build), False, (DSP,), original=base.plan)


def _exported(name: str, moves: dict) -> tuple:
    """The original plan and export's plan of fixture `name`, with each value
    of `moves` moved by its bytes."""
    data = _data(name)
    plan = pte.read(pe.export(data).data).execution_plan[0]
    for v, delta in moves.items():
        info = plan.values[v].allocation_info
        plan.values[v] = replace(plan.values[v], allocation_info=pte.AllocationDetails(
            1, info.offset + delta, 0))
    return pte.read(data).execution_plan[0], plan


@pytest.mark.parametrize("moves, match", [
    # Band 0's intermediate storage moved with the two views inside it: the
    # addition reads its operands at different rows.
    ({96: 384, 82: 384, 86: 384}, r"instruction 6 \(cortex_m::quantized_add\) reads its "
     r"operands at bytes 0 and 384"),
    # Band 0's last convolution output placed over the view the addition
    # reads next, whose rows fit it.
    ({81: 4992 - 11136}, r"value 81 is written at instruction 5 over bytes 4992\.\.6528 of "
     r"buffer 1 that value 82 still needs at instruction 6"),
])
def test_check_refuses_bytes_of_another_producer(moves, match):
    """Moves that keep every view's row size and alignment; the host runner
    returns wrong outputs for both."""
    original, plan = _exported("bands_m7.pte", {})
    pc.check(plan, False, (DSP,), original=original)
    places = pc._planned(plan)
    assert [places[v].nbytes for v in (96, 82, 86, 81)] == [2688, 1536, 1920, 1536]
    assert places[96].offset == places[82].offset < places[86].offset < places[96].end
    original, plan = _exported("bands_m7.pte", moves)
    with pytest.raises(ValueError, match=match):
        pc.check(plan, False, (DSP,), original=original)


def test_check_refuses_band_padding_the_original_does_not_have():
    original, plan = _exported("bands_m7.pte", {})
    call = plan.chains[0].instructions[10]
    call.args[cm.SIGNATURES[cm.NAMESPACE + "quantized_conv2d"].index("padding")] = _int_list(
        plan, (1, 1))
    with pytest.raises(ValueError, match=r"instruction 10 \(cortex_m::quantized_conv2d\) "
                       r"computes 7 rows from bytes \d+\.\.\d+ of what the original reads with 1 "
                       r"padding rows above, which are no rows the original instruction 4 "
                       r"computes"):
        pc.check(plan, False, (DSP,), original=original)


def _moved(plan: pte.ExecutionPlan, value: int, offset: int) -> pte.ExecutionPlan:
    moved = replace(plan, values=list(plan.values))
    moved.values[value] = replace(plan.values[value],
                                  allocation_info=pte.AllocationDetails(1, offset, 0))
    return moved


def test_check_without_the_original_reads_bytes():
    """Export's plan reads row views inside larger tensors and writes band
    outputs into one tensor. Without the original, every self-consistent
    placement passes: a row view moved one row down still reads bytes an
    instruction wrote, though not the rows the original reads. Bytes last
    written as scratch or never written fail, and so does a tensor an
    instruction wrote whose bytes another write overwrites before it is read."""
    original, plan = _exported("bands_m7.pte", {})
    pc.check(plan, False, (DSP,))
    places = pc._planned(plan)
    call = cm.kernel_calls(plan)[8]
    view = call.args[call.signature.index("input")]
    rows = plan.values[view].sizes[2]
    nbytes = places[view].nbytes
    assert (rows, nbytes, places[view].offset) == (9, 1728, 576)
    shifted = _moved(plan, view, 576 + nbytes // rows)
    pc.check(shifted, False, (DSP,))
    with pytest.raises(ValueError, match=r"instruction 13 \(cortex_m::quantized_conv2d\) "
                       r"computes 2 rows from bytes 1536\.\.3456 of what the original reads "
                       r"with 0 padding rows above, which are no rows the original "
                       r"instruction 7 computes"):
        pc.check(shifted, False, (DSP,), original=original)
    scratch = cm.kernel_calls(plan)[7]
    scratch = places[scratch.args[scratch.signature.scratch[0]]]
    with pytest.raises(ValueError, match=rf"value {view} is read at instruction 8 from bytes "
                       rf"{scratch.offset}\.\.{scratch.offset + 64} of buffer 1, which "
                       rf"instruction 7 last wrote as its scratch tensor"):
        pc.check(_moved(plan, view, scratch.offset + 64 - nbytes), False, (DSP,))
    end = plan.non_const_buffer_sizes[1]
    with pytest.raises(ValueError, match=rf"value {view} is read at instruction 8 before "
                       rf"anything writes it, so it holds state"):
        pc.check(_moved(plan, view, end - nbytes), False, (DSP,))
    # In the original, instruction 6 adds instruction 3's output to instruction 5's.
    calls = cm.kernel_calls(original)
    places = pc._planned(original)
    kept, over = calls[3].args[-2], calls[5].args[-2]
    with pytest.raises(ValueError, match=rf"value {over} is written at instruction 5 over bytes "
                       rf"18432\.\.27648 of buffer 1 that value {kept} still needs at "
                       rf"instruction 6"):
        pc.check(_moved(original, over, places[kept].offset), False, (DSP,))


@pytest.mark.parametrize("name", ["bands_m0.pte", "bands_m7.pte", "bands_m55.pte",
                                  "chain_m0.pte", "chain_m55.pte"])
@pytest.mark.parametrize("allow", [False, True])
def test_a_written_program_exports_again(name, allow):
    """A program export wrote is read back like any other: the check passes,
    and export finds no smaller total, so it keeps the program as it is."""
    first = pe.export(_data(name), allow_input_overwrite=allow)
    assert first.written
    again = pe.export(first.data, allow_input_overwrite=allow)
    assert not again.refused and not again.written and not again.unmet
    assert again.data == first.data and again.reason is None
    assert again.total_before == first.total_after
    assert again.instructions_before == first.instructions_after
    assert again.input_overwritten_before == first.input_overwritten_after


def test_kept_input_is_never_overwritten():
    """With the method input moved past the original plan, the original no
    longer writes it, so export keeps it unless allowed otherwise."""
    data = input_past_plan(_data("bands_m7.pte"))
    kept = pe.export(data)
    assert kept.written and kept.input_overwritten_before is False
    assert kept.input_overwritten_after is False
    original = pte.read(data).execution_plan[0]
    pc.check(pte.read(kept.data).execution_plan[0], True, (DSP,), original=original)
    allowed = pe.export(data, allow_input_overwrite=True)
    assert allowed.written and allowed.total_after <= kept.total_after
    pc.check(pte.read(allowed.data).execution_plan[0], False, (DSP,), original=original)


def test_budgets():
    data = _data("bands_m7.pte")
    best = pe.export(data)
    result = pe.export(data, arena=1000)
    assert not result.written and result.data == data
    # The smallest planned buffer comes at a higher total than the best one.
    assert not result.refused and result.unmet == [("arena", 11328, 23436)]
    assert best.total_after == 21860
    assert result.reason is None
    result = pe.export(data, method=4000)
    assert not result.written and result.unmet == [("method", 4212, best.total_before)]
    assert result.reason is None
    result = pe.export(data, arena=20000, method=4212)
    assert not result.written and result.unmet == [("both", None, None)]
    assert result.reason is None
    result = pe.export(data, method=4212)
    assert not result.written and not result.unmet and not result.refused
    assert result.reason is None
    fits = pe.export(data, arena=20000, method=10000)
    assert fits.written and fits.planned_after[1] <= 20000 and fits.method_after <= 10000
    capped = pe.export(data, max_extra_macs=0.2)
    assert capped.written and capped.extra_macs <= 0.2 * capped.macs
    assert capped.total_after >= best.total_after
    assert pe.export(data, arena=28224).written


def test_unmet_figures_are_programs_export_writes():
    """The candidates do not depend on the size limits: the smallest figure an
    unmet limit reports is written under that figure as the limit, and a
    limit each candidate meets alone is reported together with the other."""
    data = _data("bands_m7.pte")
    for arena in (1000, 5000, 11327):
        (limit, smallest, total), = pe.export(data, arena=arena).unmet
        assert limit == "arena"
        written = pe.export(data, arena=smallest)
        assert written.written and written.planned_after[1] == smallest
    assert not pe.export(data, arena=smallest - 1).written
    alone = pe.export(data, arena=20000)
    assert alone.written and alone.method_after > 4212
    assert pe.export(data, arena=20000, method=4212).unmet == [("both", None, None)]


def test_nothing_to_gain_is_not_written():
    data = _data("resblock_m7.pte")
    result = pe.export(data)
    assert not result.written and result.data == data
    assert result.reason is None and not result.refused and not result.unmet
    assert result.total_before == 3648 + 2504 + 420


def test_mve_programs_are_written_with_a_warning():
    result = pe.export(_data("bands_m55.pte"))
    assert result.written and result.cmsis_class == MVE
    assert result.warnings == [
        "lowered for MVE kernels (Cortex-M55, M85); firmware without MVE runs its linear layers "
        "without their bias"]
    assert pe.export(_data("chain_m55.pte")).warnings == [
        "lowered for MVE kernels (Cortex-M55, M85); it runs on firmware built with MVE"]


def test_cpu_is_checked():
    data = _data("bands_m7.pte")
    assert pe.export(data, cpu="cortex-m7").written
    result = pe.export(data, cpu="M55")
    assert not result.written and result.data == data
    assert result.refused and result.reason == (
        "was lowered for DSP kernels (Cortex-M4, M7, M33, M35P); --cpu M55 runs MVE kernels")
    assert not pe.export(_data("bands_m55.pte"), cpu="M7").written


def test_mve_bands_keep_the_kernel():
    """A one-row band of a 1x3 convolution would run CMSIS-NN's 1xN kernel
    on MVE, so MVE refuses that band count and DSP takes it."""
    for name, refused in (("bands_m55.pte", True), ("bands_m7.pte", False)):
        plan = _plan(name)
        base = pe._Base(plan, cm.infer_class(plan))
        assert base.calls[3].value("weight").sizes[1:3] == [1, 3]
        if refused:
            with pytest.raises(pe._Infeasible):
                pe._build(base, [(3, 3, 24, pe._NONE)], False)
        else:
            pe._build(base, [(3, 3, 24, pe._NONE)], False)
        pe._build(base, [(3, 3, 12, pe._NONE)], False)


def test_depthwise_bands_keep_the_kernel():
    """A 3x3 depthwise convolution padded by two rows runs CMSIS-NN's generic
    kernel on DSP; a band padded by one row or less would run the 3x3 one."""
    plan = _plan("bands_m7.pte")
    base = pe._Base(plan, (DSP,))
    pe._build(base, [(1, 1, 2, pe._NONE)], False)
    call = plan.chains[0].instructions[1]
    call.args[4] = _int_list(plan, (2, 1))
    base = pe._Base(plan, (DSP,))
    assert [k for k, _ in base.kernel(1)] == ["opt"]
    with pytest.raises(pe._Infeasible):
        pe._build(base, [(1, 1, 2, pe._NONE)], False)


@pytest.mark.parametrize("edit, match", [
    (lambda p, plan: setattr(plan.operators[plan.chains[0].instructions[0].op_index], "name",
                             "cortex_m::quantized_conv2d_nhwc"),
     "uses the explicit NHWC layout (cortex_m::quantized_conv2d_nhwc), which tigris export "
     "does not rewrite"),
    (lambda p, plan: setattr(plan.operators[plan.chains[0].instructions[0].op_index], "name",
                             "cortex_m::minimum"),
     "calls operator cortex_m::minimum; tigris export rewrites programs of the cortex_m "
     "operators quantized_conv2d, "),
    (lambda p, plan: setattr(plan, "delegates", [pte.BackendDelegate(id="EthosUBackend")]),
     "delegates to EthosUBackend; tigris export rewrites programs that run on cortex_m "
     "operators"),
    (lambda p, plan: setattr(p, "mutable_data_segments", [pte.SubsegmentOffsets(0, [0])]),
     "mutable data segments"),
    (lambda p, plan: setattr(p, "named_data", [pte.NamedData("x", 0)]), "named data"),
    (lambda p, plan: setattr(plan, "non_const_buffer_sizes", [0, 28224, 16]),
     "plans 2 buffers; tigris export rewrites programs with one planned buffer"),
    (lambda p, plan: p.execution_plan.append(plan), "2 methods with instructions"),
])
def test_refusals(edit, match):
    program = pte.read(_data("bands_m7.pte"))
    edit(program, program.execution_plan[0])
    data = pte.write(program)
    result = pe.export(data)
    assert result.refused and not result.written and result.data == data
    assert match in result.reason


def _set(index: int, **fields):
    def edit(program, plan):
        plan.values[index] = replace(plan.values[index], **fields)
    return edit


@pytest.mark.parametrize("name, edit, reason", [
    ("chain_m7.pte", lambda p, plan: plan.non_const_buffer_sizes.__setitem__(1, 100),
     "has a memory plan that fails the check: value 16 (768 bytes at 2048) runs past buffer 1 "
     "of 100 bytes"),
    ("chain_m7.pte", _set(18, storage_offset=4),
     "has a memory plan that fails the check: value 18 has storage offset 4"),
    ("chain_m7.pte", _set(18, scalar_type=99),
     "has a memory plan that fails the check: unknown scalar type 99"),
    ("resblock_m7.pte", _set(12, sizes=[]),
     "is malformed: the quantized_linear weight has sizes [], expected two dimensions"),
    ("resblock_m7.pte", _set(45, sizes=None),
     "is malformed: the quantized_linear output has sizes [], expected at least one dimension"),
    ("bands_m7.pte", lambda p, plan: setattr(plan.chains[0], "stacktrace", []),
     "is malformed: its chain has 0 stack traces for 10 instructions"),
], ids=["past-buffer", "storage-offset", "scalar-type", "linear-weights", "linear-out",
        "stacktrace"])
def test_malformed_programs_are_refused(name, edit, reason):
    program = pte.read(_data(name))
    edit(program, program.execution_plan[0])
    data = pte.write(program)
    result = pe.export(data)
    assert result.refused and not result.written and result.data == data
    assert result.reason == reason


def test_a_declared_buffer_larger_than_its_tensors_is_checked():
    """The check replays the bytes the planned tensors span, not the bytes
    the program declares."""
    program = pte.read(_data("chain_m7.pte"))
    plan = program.execution_plan[0]
    plan.non_const_buffer_sizes[1] = 1 << 36
    pc.check(plan, not pc.input_overwritten(plan), (DSP,))
    result = pe.export(pte.write(program))
    assert not result.refused and result.planned_before == [0, 1 << 36]


@pytest.mark.parametrize("j, name, ints", [
    *((j, "stride", (3, 3)) for j in range(6)),
    (3, "padding", (2, 1)),
])
def test_geometry_that_leaves_rows_without_input_is_refused(j, name, ints):
    program = pte.read(_data("bands_m7.pte"))
    plan = program.execution_plan[0]
    call = cm.kernel_calls(plan)[j]
    plan.chains[0].instructions[j].args[call.signature.index(name)] = _int_list(plan, ints)
    data = pte.write(program)
    result = pe.export(data)
    assert not result.written and result.data == data
    assert result.refused and result.reason.startswith(f"has instruction {j} (")
    assert "so some output rows read no input row" in result.reason


@pytest.mark.parametrize("j", range(1, 8))
def test_reads_through_an_alias_inside_a_region(j):
    """An instruction that reads its input through a second planned tensor at
    the same place, as ExecuTorch's emitter makes for a reshape."""
    program = pte.read(_data("bands_m7.pte"))
    plan = program.execution_plan[0]
    args = plan.chains[0].instructions[j].args
    source = args[0]
    plan.values.append(replace(plan.values[source]))
    args[0] = len(plan.values) - 1
    alias, extra = pe._aliases(plan, cm.kernel_calls(plan))
    assert alias[args[0]] == (source, 0) and extra == []
    data = pte.write(program)
    result = pe.export(data)
    assert result.written, result.reason
    pc.check(pte.read(result.data).execution_plan[0],
             not result.input_overwritten_before, (DSP,), original=plan)


def test_chain_inputs_and_outputs_are_refused():
    program = pte.read(_data("bands_m7.pte"))
    program.execution_plan[0].chains[0].inputs = [program.execution_plan[0].inputs[0]]
    result = pe.export(pte.write(program))
    assert not result.written
    assert result.reason == ("lists inputs or outputs on its chain; tigris export rewrites "
                             "programs whose chains take the method's inputs and outputs")


def test_not_a_program():
    result = pe.export(b"not a program")
    assert not result.written and result.data == b"not a program"
    assert result.refused and result.reason == "is not an ExecuTorch program (no ET12 identifier)"


def test_module_needs_neither_torch_nor_executorch():
    code = ("import sys; from tigris.frontends import pte_export; "
            f"pte_export.export(open({str(FIXTURES / 'bands_m7.pte')!r}, 'rb').read()); "
            "print(sorted(m for m in ('torch', 'executorch', 'flatbuffers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"


# Planned buffers plus method allocator after export, at most: the figures
# a search over the same regions reached on four ExecuTorch 1.6.0 programs
# lowered for the Cortex-M7 with int8 inputs (DS-CNN, ResNet-8,
# MobileNetV2-0.35 at 96x96, MobileNetV2 at 224x224), and the MACs they cost.
_PROGRAMS = {
    "dscnn": (20052, None),
    "r8": (34056, None),
    "mbv2": (68448, None),
    "m224": (271860, 0.0628),
}
# The same with the method input moved past the plan, so export keeps it.
_KEPT = {"dscnn": 20356, "r8": 37064, "mbv2": 82260, "m224": 354196}
# The search alone, with no region output in place over the method input
# but its bytes free after their last read.
_NOT_OVER_INPUT = {"mbv2": 77560, "m224": 325068}


def _program(name: str) -> bytes:
    root = os.environ.get("TIGRIS_PTE_PROGRAMS")
    if not root:
        pytest.skip("TIGRIS_PTE_PROGRAMS is not set")
    return (Path(root) / name / "model.pte").read_bytes()


@pytest.mark.parametrize("name", _PROGRAMS)
def test_larger_programs(name):
    """Needs TIGRIS_PTE_PROGRAMS, a directory with <name>/model.pte for each."""
    data = _program(name)
    total, max_extra_macs = _PROGRAMS[name]
    for kwargs in ({}, {"allow_input_overwrite": True}, {"max_extra_macs": max_extra_macs}):
        if kwargs.get("max_extra_macs", 0) is None:
            continue
        result = pe.export(data, **kwargs)
        assert result.written, result.reason
        assert result.planned_after[1] + result.method_after <= total, kwargs
        pc.check(pte.read(result.data).execution_plan[0],
                 not result.input_overwritten_before, (DSP,),
                 original=pte.read(data).execution_plan[0])
        if max_extra_macs is not None and "max_extra_macs" in kwargs:
            assert result.extra_macs <= max_extra_macs * result.macs
        # The written program exports again: a further gain, or the same program.
        again = pe.export(result.data, **kwargs)
        assert not again.refused and not again.unmet, again.reason
        assert again.total_before == result.total_after
        if again.written:
            assert again.total_after < result.total_after
            pc.check(pte.read(again.data).execution_plan[0],
                     not again.input_overwritten_before, (DSP,),
                     original=pte.read(result.data).execution_plan[0])
        else:
            assert again.data == result.data


@pytest.mark.parametrize("name", _KEPT)
def test_larger_programs_keep_their_input(name):
    """Needs TIGRIS_PTE_PROGRAMS."""
    data = input_past_plan(_program(name))
    result = pe.export(data)
    assert result.written, result.reason
    assert result.input_overwritten_before is False and result.input_overwritten_after is False
    assert result.planned_after[1] + result.method_after <= _KEPT[name]
    pc.check(pte.read(result.data).execution_plan[0], True, (DSP,),
             original=pte.read(data).execution_plan[0])


def test_limits_on_a_larger_program():
    """Needs TIGRIS_PTE_PROGRAMS. A model memory limit met alone and a method
    allocator limit met alone, but by no program together."""
    data = _program("mbv2")
    arena = pe.export(data, arena=32 * 1024)
    assert arena.written and arena.method_after > 40 * 1024
    assert pe.export(data, method=40 * 1024).method_before <= 40 * 1024
    both = pe.export(data, arena=32 * 1024, method=40 * 1024)
    assert not both.written and both.unmet == [("both", None, None)]
    (limit, smallest, _), = pe.export(data, arena=31000).unmet
    assert limit == "arena" and smallest == arena.planned_after[1]


@pytest.mark.parametrize("name", _NOT_OVER_INPUT)
def test_larger_programs_search_without_in_place_over_the_input(name):
    """Needs TIGRIS_PTE_PROGRAMS."""
    plan = pte.read(_program(name)).execution_plan[0]
    best = pe._search(pe._Base(plan, (DSP,)), False, False)[0]
    assert best.arena + best.method <= _NOT_OVER_INPUT[name]
