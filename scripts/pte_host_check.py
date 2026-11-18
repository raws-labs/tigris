#!/usr/bin/env python3
"""Check that ExecuTorch programs written by `pte_export` compute the same
bytes as the originals on the ExecuTorch runtime.

Each program is exported in several variants: as given, with
`allow_input_overwrite`, with each CPU class the program fits named through
`cpu` (with and without `allow_input_overwrite`), and the same for the
program with its method input moved past the end of its plan, which export
keeps unless allowed to overwrite it. The original, the moved original and
every distinct written program run through the runner over all inputs, once
per poison byte in the planned buffers. The check fails with exit status 1 on
any output byte that differs from the original's or between poison bytes, on
a variant that must keep its input (without `allow_input_overwrite`, of the
moved program or of any program that leaves its input intact) and changes a
byte of it, on a runner failure or an export error, on a
program that cannot be read or whose class cannot be inferred, on a named
program the runner cannot run, and when no written program ran at all.

The runner is a command `runner model.pte input.bin output.bin input_after.bin
poison` that fills the planned buffers with the byte `poison`, runs method
"forward" once per input-sized slice of input.bin and writes the raw bytes of
its outputs to output.bin and of its input after each run to input_after.bin.
`src/tigris/frontends/pte_host_runner.cpp` is one: the ExecuTorch 1.6.0
runtime with the cortex_m operators on CMSIS-NN's portable C kernels, built
for the host by `tigris run` or as its header describes. Those kernels compute the SCALAR and DSP classes
bit for bit; programs lowered for MVE kernels run when `--classes` names MVE
for a runner that executes them, such as the same source built for an
emulated Cortex-M55 as its header describes, and fixtures of a class the
runner does not run are skipped.

A program's inputs are `<stem>.npz` next to it (int8 `inputs`, one row per
input, and optionally the `output_bytes` the original must produce) or
`input.bin` in its directory. Without programs, every fixture in
tests/fixtures/pte is checked. `--record` writes, for the fixtures checked,
each variant's digest and figures, or null when export writes nothing, to
tests/fixtures/pte/written.json, which tests compare export against without a
runner:

    python scripts/pte_host_check.py --runner PATH [--classes C,...] [--record] [program.pte ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from tigris.cli.export import pte_reason
from tigris.frontends import pte
from tigris.frontends import pte_cortex_m as cm
from tigris.frontends import pte_export as pe

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pte"
RECORD = FIXTURES / "written.json"
# The CPU each class is named by in the cpu variants.
CLASS_CPU = {cm.SCALAR: "M0", cm.DSP: "M7", cm.MVE: "M55"}
MOVED = "input moved"
# Planned buffers are filled with each before every inference.
POISONS = (0xA5, 0x5A)


def input_past_plan(data: bytes) -> bytes | None:
    """The program with its method input moved past the end of its plan, or
    None when the program has no single planned input."""
    program = pte.read(data)
    plan = program.execution_plan[0]
    sizes = plan.non_const_buffer_sizes or []
    if len(plan.inputs or ()) != 1 or len(sizes) != 2:
        return None
    index = plan.inputs[0]
    value = plan.values[index]
    if not isinstance(value, pte.Tensor) or value.allocation_info is None:
        return None
    plan.values[index] = replace(value, allocation_info=pte.AllocationDetails(1, sizes[1], 0))
    plan.non_const_buffer_sizes = [0, sizes[1] + value.nbytes]
    return pte.write(replace(program, execution_plan=[plan, *program.execution_plan[1:]]))


def variants(classes: tuple) -> list[tuple[str, dict]]:
    """Export arguments per variant name, for a program that fits `classes`."""
    out = [("as given", {}), ("allow_input_overwrite", {"allow_input_overwrite": True})]
    for cls in classes:
        cpu = CLASS_CPU[cls]
        out.append((f"cpu={cpu}", {"cpu": cpu}))
        out.append((f"cpu={cpu} allow_input_overwrite",
                    {"cpu": cpu, "allow_input_overwrite": True}))
    return out + [(f"{MOVED} {name}", kwargs) for name, kwargs in out]


def summary(result: pe.PteExport) -> dict:
    """What `--record` keeps of a written variant."""
    return {
        "sha256": hashlib.sha256(result.data).hexdigest(),
        "planned": result.planned_after,
        "method": result.method_after,
        "temp": result.temp_after,
        "regions": [[r.first, r.last, r.bands, r.in_place] for r in result.regions],
        "input_overwritten": result.input_overwritten_after,
    }


def _inputs(path: Path) -> tuple[bytes, np.ndarray | None]:
    record = path.with_suffix(".npz")
    if record.is_file():
        data = np.load(record)
        expected = data["output_bytes"] if "output_bytes" in data.files else None
        return np.ascontiguousarray(data["inputs"], np.int8).tobytes(), expected
    raw = path.parent / "input.bin"
    if raw.is_file():
        return raw.read_bytes(), None
    raise SystemExit(f"{path}: neither {record.name} nor input.bin found")


def _run(runner: list, program: Path, inputs: Path, poison: int,
         timeout: float) -> tuple[bytes, bytes]:
    """The program's outputs and its input as left after each inference."""
    output = program.with_suffix(f".{poison:02x}.out")
    after = program.with_suffix(f".{poison:02x}.in")
    done = subprocess.run([*runner, str(program), str(inputs), str(output), str(after),
                           f"0x{poison:02x}"], capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        tail = " | ".join(done.stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"runner exit {done.returncode} with poison 0x{poison:02x}: {tail}")
    return output.read_bytes(), after.read_bytes()


def _run_all(runner: list, program: Path, inputs: Path,
             timeout: float) -> tuple[bytes, list[bytes]]:
    """The program's outputs, equal under every poison byte, and its input
    after each run."""
    runs = [_run(runner, program, inputs, poison, timeout) for poison in POISONS]
    for poison, (out, _) in zip(POISONS[1:], runs[1:]):
        diff = _differences(out, runs[0][0])
        if diff:
            raise RuntimeError(f"{diff} output bytes change between poison 0x{POISONS[0]:02x} "
                               f"and 0x{poison:02x}")
    return runs[0][0], [after for _, after in runs]


def _differences(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return int(np.count_nonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8)))


def _figures(result: pe.PteExport) -> str:
    regions = ", ".join(f"{r.first}-{r.last}x{r.bands}{' in place' if r.in_place else ''}"
                        for r in result.regions)
    return (f"planned {result.planned_after[1]} method {result.method_after} "
            f"(was {result.planned_before[1]} + {result.method_before}), regions {regions}")


def check(path: Path, runner: list, run_classes: set, work: Path, timeout: float,
          records: dict | None, named: bool) -> tuple[int, int]:
    """Check one program, keeping its files in the new directory `work`;
    returns the number of failures and of written programs run. A program
    `named` on the command line fails when the runner cannot run it."""
    data = path.read_bytes()
    inputs, expected = _inputs(path)
    try:
        classes = cm.infer_class(pte.read(data).execution_plan[0])
    except ValueError as error:
        print(f"{path}: FAILED: {error}")
        return 1, 0
    print(f"{path}: CMSIS-NN class {' or '.join(classes)}, {len(inputs)} input bytes")
    if not run_classes.intersection(classes):
        if named:
            print(f"  FAILED: the runner runs {', '.join(sorted(run_classes))} kernels only")
            return 1, 0
        print("  not run: the runner does not run these kernels")
        return 0, 0
    allowed = tuple(c for c in classes if c in run_classes)
    work.mkdir()
    feed = work / "input.bin"
    feed.write_bytes(inputs)
    failures = 0
    programs = {"original": data}
    moved = input_past_plan(data)
    if moved is not None:
        programs[MOVED] = moved
    outputs = {}
    for name, program in programs.items():
        file = work / f"{name.replace(' ', '_')}.pte"
        file.write_bytes(program)
        try:
            outputs[name] = _run_all(runner, file, feed, timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"  {name}: FAILED: {error}")
            return failures + 1, 0
    reference = outputs["original"][0]
    print(f"  original: {len(reference)} output bytes")
    if expected is not None:
        diff = _differences(reference, np.ascontiguousarray(expected, np.int8).tobytes())
        if diff:
            print(f"  original: FAILED: {diff} bytes differ from the recorded outputs")
            failures += 1
        else:
            print("  original: equals the recorded outputs")
    if MOVED in outputs:
        out, afters = outputs[MOVED]
        diff = _differences(out, reference)
        state = f"FAILED: {diff} bytes differ" if diff else "identical"
        if not diff and any(after != inputs for after in afters):
            state, diff = "FAILED: the runner reports its kept input as overwritten", 1
        print(f"  {MOVED}: {state}")
        failures += bool(diff)
    seen = {}
    kept = {}
    recorded = {}
    ran = 0
    for name, kwargs in variants(allowed):
        if name.startswith(MOVED) and MOVED not in programs:
            continue
        source = programs[MOVED] if name.startswith(MOVED) else data
        try:
            result = pe.export(source, **kwargs)
        except Exception as error:  # an internal check that fired is a failure
            print(f"  {name}: FAILED: export raised {type(error).__name__}: {error}")
            failures += 1
            continue
        keep = not kwargs.get("allow_input_overwrite") and (
            name.startswith(MOVED) or result.input_overwritten_before is False)
        if keep and result.input_overwritten_before is not False:
            print(f"  {name}: FAILED: export reads the moved input as overwritten")
            failures += 1
            continue
        if not result.written:
            print(f"  {name}: not written: {pte_reason(result, None, None, None)}")
            recorded[name] = None
            continue
        recorded[name] = summary(result)
        if keep and result.input_overwritten_after is not False:
            print(f"  {name}: FAILED: export overwrites the input it must keep")
            failures += 1
            continue
        digest = recorded[name]["sha256"]
        if digest not in seen:
            seen[digest] = name
            file = work / f"written_{len(seen)}.pte"
            file.write_bytes(result.data)
            try:
                out, afters = _run_all(runner, file, feed, timeout)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                print(f"  {name}: FAILED: {error}")
                failures += 1
                continue
            ran += 1
            kept[digest] = all(after == inputs for after in afters)
            diff = _differences(out, reference)
            state = f"FAILED: {diff} of {len(reference)} bytes differ" if diff else "identical"
            print(f"  {name}: {_figures(result)}: {state}")
            failures += bool(diff)
        elif digest in kept:
            print(f"  {name}: same program as {seen[digest]}")
        else:  # its first run failed and was counted there
            continue
        if keep and not kept[digest]:
            print(f"  {name}: FAILED: the runner reports the kept input as overwritten")
            failures += 1
    if records is not None and not failures:
        records[path.name] = recorded
    return failures, ran


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", required=True, help="runner command (split on spaces)")
    parser.add_argument("--classes", default=f"{cm.SCALAR},{cm.DSP}",
                        help="CMSIS-NN classes the runner computes bit for bit")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds per runner call")
    parser.add_argument("--record", action="store_true",
                        help="record the checked fixtures' written programs")
    parser.add_argument("--artifacts", type=Path,
                        help="keep programs and outputs in this empty directory")
    parser.add_argument("programs", nargs="*", type=Path)
    args = parser.parse_args()
    runner = args.runner.split()
    run_classes = {c.strip().upper() for c in args.classes.split(",") if c.strip()}
    if not run_classes <= set(cm.CLASSES):
        raise SystemExit(f"--classes takes {', '.join(cm.CLASSES)}")
    named = bool(args.programs)
    paths = [p.resolve() for p in args.programs] or sorted(FIXTURES.glob("*.pte"))
    if args.record and any(p.parent != FIXTURES for p in paths):
        raise SystemExit(f"--record takes programs from {FIXTURES} only")
    records = {} if args.record else None

    def run(work: Path) -> tuple[int, list[int]]:
        totals = [check(p, runner, run_classes, work / f"{i}_{p.parent.name}_{p.stem}",
                        args.timeout, records, named) for i, p in enumerate(paths)]
        return sum(f for f, _ in totals), [r for _, r in totals if r]

    if args.artifacts:
        work = args.artifacts.resolve()
        work.mkdir(parents=True, exist_ok=True)
        if any(work.iterdir()):
            raise SystemExit(f"artifact directory must be empty: {work}")
        failures, ran = run(work)
    else:
        with tempfile.TemporaryDirectory(prefix="tigris-pte-") as temp:
            failures, ran = run(Path(temp))
    if failures:
        print(f"PTE host check failed: {failures} failures.")
        sys.exit(1)
    if not ran:
        print("PTE host check failed: no written program ran.")
        sys.exit(1)
    if records is not None:
        old = json.loads(RECORD.read_text()) if RECORD.is_file() else {}
        old.update(records)
        RECORD.write_text(json.dumps(dict(sorted(old.items())), indent=1) + "\n")
        print(f"Recorded {len(records)} programs in {RECORD.name}.")
    print(f"PTE host check passed: {sum(ran)} written programs over {len(ran)} programs "
          "identical.")


if __name__ == "__main__":
    main()
